import torch
import torch.nn as nn


class MinkowskiLoss(nn.Module):
    def __init__(self, quantile_levels):
        super().__init__()
        # Ensure float32 casting for integration stability
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, pred_log, target_log):
        """
        Calculates the 1-Wasserstein (L1 Minkowski) distance between the log-transformed
        geometric quantile curves.

        Args:
            pred_log: Tensor [B, 3, N_Quantiles]
            target_log: Tensor [B, 3, N_Quantiles]
        Returns:
            total_dist: Tensor [B] (Sum of distances across Area, Perimeter, Euler)
        """
        # Ensure identical dtype
        pred_log = pred_log.float()
        target_log = target_log.float()

        abs_diff = torch.abs(pred_log - target_log)

        # Integrate w.r.t dq (quantiles)
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)

        # Sum the three topological components (Area, Perimeter, Euler) per sample
        total_dist = dist[:, 0] + dist[:, 1] + dist[:, 2]

        return total_dist
