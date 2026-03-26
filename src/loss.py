import torch
import torch.nn as nn


class MinkowskiLoss(nn.Module):
    def __init__(self, quantile_levels):
        super().__init__()
        # Ensure float32 casting for integration stability
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, pred_log, target_log, a=1.0, b=1.0, c=1.0, d=1.0):
        """
        Calculates the 1-Wasserstein (L1 Minkowski) distance between the log-transformed
        quantile curves.

        Args:
            pred_log: Tensor [B, 4, N_Quantiles]
            target_log: Tensor [B, 4, N_Quantiles]
        Returns:
            total_dist: Tensor [B] (Sum of distances)
            dist_area: Tensor [B]
            dist_perimeter: Tensor [B]
            dist_betti0: Tensor [B]
            dist_betti1: Tensor [B]
        """
        # Ensure identical dtype
        pred_log = pred_log.float()
        target_log = target_log.float()

        abs_diff = torch.abs(pred_log - target_log)

        # Integrate w.r.t dq (quantiles)
        dist = torch.trapezoid(abs_diff, self.quantiles, dim=2)

        # Sum the four topological components per sample
        total_dist = a * dist[:, 0] + b * dist[:, 1] + c * dist[:, 2] + d * dist[:, 3]

        return total_dist, dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
