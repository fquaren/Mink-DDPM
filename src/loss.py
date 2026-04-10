import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MinkowskiLoss(nn.Module):
    def __init__(self, quantile_levels):
        super().__init__()
        # Ensure float32 casting for integration stability
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, pred_log, target_log, a=1.0, b=1.0, c=1.0):
        """
        Calculates the 1-Wasserstein (L1 Minkowski) distance between the log-transformed
        quantile curves.

        Args:
            pred_log: Tensor [B, 3, N_Quantiles]
            target_log: Tensor [B, 3, N_Quantiles]
        Returns:
            total_dist: Tensor [B] (Sum of distances)
            dist_area: Tensor [B]
            dist_perimeter: Tensor [B]
            dist_betti0: Tensor [B]
        """
        # Ensure identical dtype
        pred_log = pred_log.float()
        target_log = target_log.float()

        abs_diff = torch.abs(pred_log - target_log)  # [B, 3, Q]

        # Integrate across the Quantile/Threshold dimension (dim=2)
        # result shape: [B, 3]
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)

        # Apply coefficients and sum across the 3 functionals
        # result shape: [B]
        total_dist = a * dist[:, 0] + b * dist[:, 1] + c * dist[:, 2]

        return total_dist, dist[:, 0], dist[:, 1], dist[:, 2]


# ------------------------------------------------------------------------------
# Analytical Loss Module
# ------------------------------------------------------------------------------
class AnalyticalMinkowskiLoss(nn.Module):
    def __init__(
        self,
        thresholds,
        pixel_size_km=2.0,
        init_factor=0.1,
        min_temp=1e-3,
        persistence_thresh=1.8699999839067458,
    ):
        super().__init__()
        self.pixel_size_km = pixel_size_km
        self.pixel_area = pixel_size_km**2
        self.persistence_thresh = persistence_thresh

        # Reshape to [1, Q, 1, 1] for direct broadcasting against [B, Q, H, W]
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1),
        )

        base_temps = np.maximum(np.array(thresholds) * init_factor, min_temp)
        self.register_buffer(
            "base_temps",
            torch.tensor(base_temps, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(self, pred_phys, target_gamma_log, anneal_factor=1.0):
        """
        pred_phys: Tensor [B, 1, H, W] in physical space.
        target_gamma_log: Tensor [B, 4, Q] containing log1p-transformed targets.
        """
        current_temps = self.base_temps * anneal_factor

        # --- TOPOLOGY PRE-PROCESSING ---
        # Operations applied on [B, 1, H, W] prior to threshold broadcasting

        # 1. Morphological Closing (Fills small false holes / Betti-1)
        dilated = F.max_pool2d(pred_phys, kernel_size=3, stride=1, padding=1)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

        # 2. Morphological Opening (Removes small false spikes)
        eroded = -F.max_pool2d(-closed, kernel_size=3, stride=1, padding=1)
        field_phys_topo = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

        # 3. Local Maximum for Persistence
        local_max = F.max_pool2d(field_phys_topo, kernel_size=15, stride=1, padding=7)

        # --- PATH 1: EXACT GEOMETRY (Area & Perimeter) ---
        # Broadcasts to [B, Q, H, W]
        p_raw = torch.sigmoid((pred_phys - self.thresholds) / current_temps)

        area = torch.sum(p_raw, dim=(2, 3)) * self.pixel_area

        # Use symmetric padding for central differences
        p_pad = F.pad(p_raw, (1, 1, 1, 1), mode="replicate")

        # Symmetric central differences (x[i+1] - x[i-1]) / 2
        dx = (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) / 2.0
        dy = (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]) / 2.0

        perimeter = (
            torch.sum(
                torch.sqrt(dx**2 + dy**2 + 1e-8),
                dim=(2, 3),
            )
            * self.pixel_size_km
        )

        # --- PATH 2: AMPLITUDE-AWARE TOPOLOGY (Euler) ---
        # Broadcasts to [B, Q, H, W]
        p_base = torch.sigmoid((field_phys_topo - self.thresholds) / current_temps)

        # Binary mask: approaches 1 if neighborhood peak > thresh + persistence, else 0
        persistence_mask = torch.sigmoid(
            (local_max - (self.thresholds + self.persistence_thresh)) / current_temps
        )

        # Apply mask using Gödel T-norm (min) to prevent fractional distortion
        p_topo = torch.min(p_base, persistence_mask)

        # Gödel T-norm Expected Euler Characteristic
        V = torch.sum(p_topo, dim=(2, 3))
        E_x = torch.sum(
            torch.min(p_topo[:, :, :, :-1], p_topo[:, :, :, 1:]), dim=(2, 3)
        )
        E_y = torch.sum(
            torch.min(p_topo[:, :, :-1, :], p_topo[:, :, 1:, :]), dim=(2, 3)
        )
        F_faces = torch.sum(
            torch.min(
                torch.min(p_topo[:, :, :-1, :-1], p_topo[:, :, :-1, 1:]),
                torch.min(p_topo[:, :, 1:, :-1], p_topo[:, :, 1:, 1:]),
            ),
            dim=(2, 3),
        )
        euler = V - E_x - E_y + F_faces

        # --- LOSS COMPUTATION ---

        # Stack into [B, 3, Q]
        pred_gamma_phys = torch.stack([area, perimeter, euler], dim=1)

        pred_gamma_log = torch.sign(pred_gamma_phys) * torch.log1p(
            torch.abs(pred_gamma_phys)
        )

        # Process Targets: Exact inverse transform to linear space
        target_raw = torch.sign(target_gamma_log) * torch.expm1(
            torch.abs(target_gamma_log)
        )

        target_area = target_raw[:, 0, :]
        target_perim = target_raw[:, 1, :]
        target_euler = target_raw[:, 2, :] - target_raw[:, 3, :]

        target_gamma_processed = torch.stack(
            [target_area, target_perim, target_euler], dim=1
        )

        # Re-apply log transform to the combined tensor
        target_gamma_log_processed = torch.sign(target_gamma_processed) * torch.log1p(
            torch.abs(target_gamma_processed)
        )

        # Compute Wasserstein distance approximation via trapezoidal integration
        abs_diff = torch.abs(pred_gamma_log - target_gamma_log_processed.float())
        dist = torch.trapezoid(abs_diff, self.thresholds.view(-1), dim=2)

        return dist.sum(dim=1).mean()
