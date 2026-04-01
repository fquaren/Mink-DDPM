import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFT2DKernelCRPSLoss(nn.Module):
    """FFT2D + sample-based kernel-CRPS / energy-score style loss on image tensors."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def _to_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x[:, None, ...]
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        mag = torch.abs(x_fft)
        return mag.flatten(start_dim=-3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self._to_features(pred)
        tgt_f = self._to_features(target)

        if pred_f.shape[0] != tgt_f.shape[0]:
            raise ValueError("Batch size mismatch between pred and target")
        if pred_f.shape[-1] != tgt_f.shape[-1]:
            raise ValueError("Feature size mismatch between pred and target")

        if tgt_f.shape[1] == 1 and pred_f.shape[1] > 1:
            tgt_f = tgt_f.expand(pred_f.shape[0], pred_f.shape[1], tgt_f.shape[2])

        diff = torch.abs(pred_f - tgt_f).clamp_min(self.eps)
        d1 = diff.pow(self.alpha).mean(dim=-1).pow(1.0 / self.alpha)
        term1 = d1.mean(dim=1)

        ensemble_size = pred_f.shape[1]
        if ensemble_size <= 1:
            return term1.mean()

        p1 = pred_f[:, :, None, :]
        p2 = pred_f[:, None, :, :]
        pdiff = torch.abs(p1 - p2).clamp_min(self.eps)
        d2 = pdiff.pow(self.alpha).mean(dim=-1).pow(1.0 / self.alpha)
        term2 = 0.5 * d2.mean(dim=(1, 2))
        return (term1 - term2).mean()


class ImageSSIMLoss(nn.Module):
    """Structural similarity loss on standard image tensors [B, C, H, W]."""

    def __init__(
        self,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = int(window_size)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have identical shapes")
        kernel = pred.new_full(
            (pred.shape[1], 1, self.window_size, self.window_size),
            1.0 / (self.window_size**2),
        )
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
        mu_target = F.conv2d(target, kernel, padding=pad, groups=target.shape[1])

        sigma_pred = F.conv2d(pred * pred, kernel, padding=pad, groups=pred.shape[1]) - mu_pred.square()
        sigma_target = F.conv2d(target * target, kernel, padding=pad, groups=target.shape[1]) - mu_target.square()
        sigma_cross = F.conv2d(pred * target, kernel, padding=pad, groups=pred.shape[1]) - mu_pred * mu_target

        sigma_pred = torch.clamp(sigma_pred, min=0.0)
        sigma_target = torch.clamp(sigma_target, min=0.0)

        dynamic_max = torch.maximum(pred, target).amax(dim=(-2, -1), keepdim=True)
        dynamic_min = torch.minimum(pred, target).amin(dim=(-2, -1), keepdim=True)
        data_range = (dynamic_max - dynamic_min).clamp_min(self.eps)

        c1 = (self.k1 * data_range).square()
        c2 = (self.k2 * data_range).square()

        numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)
        denominator = (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2)
        ssim_map = numerator / (denominator + self.eps)
        loss_map = torch.clamp((1.0 - ssim_map) * 0.5, min=0.0)
        return loss_map.mean()


class WeightedSoftWetAreaLoss(nn.Module):
    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 0.1,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)

    def _soft_wet_mask(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.threshold) / self.temperature)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_wet = self._soft_wet_mask(pred)
        target_wet = self._soft_wet_mask(target)
        diff = pred_wet - target_wet
        fp = torch.relu(diff)
        fn = torch.relu(-diff)
        loss = self.false_positive_weight * fp.square() + self.false_negative_weight * fn.square()
        return loss.mean()


def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d.div_(kernel_1d.sum())
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(x.shape[1], 1, -1, -1)
    padded = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=x.shape[1])


def _shift_with_border(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_left = max(-dx, 0)
    pad_right = max(dx, 0)
    pad_top = max(-dy, 0)
    pad_bottom = max(dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    x0 = pad_right
    y0 = pad_bottom
    return padded[..., y0 : y0 + height, x0 : x0 + width]


class OpticalFlowConsistencyLoss(nn.Module):
    """Optical-flow-style auxiliary loss using the conditioning precipitation as baseline.

    A coarse flow field is estimated between the conditioning precipitation field
    and the target, then the conditioning field is advected and used as a teacher
    for the prediction.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        patch_size: int = 21,
        search_radius: int = 6,
        downsample: int = 4,
        preprocess_sigma: float = 1.0,
        delta: float = 0.2,
        mask_threshold: float | None = 0.05,
        loss_type: str = "huber",
        rain_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.patch_size = int(patch_size)
        self.search_radius = int(search_radius)
        self.downsample = int(downsample)
        self.preprocess_sigma = float(preprocess_sigma)
        self.delta = float(delta)
        self.mask_threshold = mask_threshold if mask_threshold is None else float(mask_threshold)
        self.loss_type = str(loss_type)
        self.rain_threshold = float(rain_threshold)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.y_dim),
            torch.linspace(-1.0, 1.0, self.x_dim),
            indexing="ij",
        )
        self.register_buffer("_base_grid", torch.stack((xx, yy), dim=-1).unsqueeze(0), persistent=False)

    def _working_size(self) -> tuple[int, int]:
        return max(1, self.y_dim // self.downsample), max(1, self.x_dim // self.downsample)

    def _prepare_working_frame(self, frame: torch.Tensor) -> torch.Tensor:
        work = frame.to(dtype=torch.float32).clone()
        work.clamp_min_(0.0)
        work.log1p_()
        if self.downsample > 1:
            work = F.interpolate(work, size=self._working_size(), mode="bilinear", align_corners=False)
        if self.preprocess_sigma > 0:
            work = _gaussian_blur2d(work, self.preprocess_sigma)
        return work

    def _estimate_dense_flow(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prev_work = self._prepare_working_frame(prev_frame)
        curr_work = self._prepare_working_frame(curr_frame)
        work_height, work_width = prev_work.shape[-2:]
        curr_rain = curr_frame.to(dtype=torch.float32)
        if curr_rain.shape[-2:] != (work_height, work_width):
            curr_rain = F.interpolate(curr_rain, size=(work_height, work_width), mode="bilinear", align_corners=False)
        rain_mask = curr_rain.ge(self.rain_threshold)

        costs = []
        dx_values = []
        dy_values = []
        radius = self.patch_size // 2
        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                shifted = _shift_with_border(curr_work, dx=dx, dy=dy)
                cost = prev_work - shifted
                cost.square_()
                if radius > 0:
                    cost = F.avg_pool2d(cost, kernel_size=self.patch_size, stride=1, padding=radius)
                costs.append(cost)
                dx_values.append(dx)
                dy_values.append(dy)

        cost_volume = torch.cat(costs, dim=1)
        best_indices = cost_volume.argmin(dim=1)
        dx_lookup = torch.tensor(dx_values, device=cost_volume.device, dtype=prev_work.dtype)
        dy_lookup = torch.tensor(dy_values, device=cost_volume.device, dtype=prev_work.dtype)
        flow_x_small = dx_lookup[best_indices].unsqueeze(1)
        flow_y_small = dy_lookup[best_indices].unsqueeze(1)

        if rain_mask.shape != flow_x_small.shape:
            rain_mask = rain_mask.expand_as(flow_x_small)
        flow_x_small.mul_(rain_mask)
        flow_y_small.mul_(rain_mask)

        if self.preprocess_sigma > 0:
            flow_x_small = _gaussian_blur2d(flow_x_small, self.preprocess_sigma)
            flow_y_small = _gaussian_blur2d(flow_y_small, self.preprocess_sigma)
            flow_x_small.mul_(rain_mask)
            flow_y_small.mul_(rain_mask)

        flow_x = F.interpolate(flow_x_small, size=(self.y_dim, self.x_dim), mode="bilinear", align_corners=False)
        flow_y = F.interpolate(flow_y_small, size=(self.y_dim, self.x_dim), mode="bilinear", align_corners=False)
        flow_x.mul_(self.x_dim / work_width)
        flow_y.mul_(self.y_dim / work_height)
        return flow_x, flow_y

    def _advect(self, field: torch.Tensor, flow_x: torch.Tensor, flow_y: torch.Tensor) -> torch.Tensor:
        batch_size = field.shape[0]
        base_grid = self._base_grid.to(device=field.device, dtype=field.dtype)
        grid = torch.empty((batch_size, self.y_dim, self.x_dim, 2), device=field.device, dtype=field.dtype)
        x_scale = 0.0 if self.x_dim <= 1 else 2.0 / (self.x_dim - 1)
        y_scale = 0.0 if self.y_dim <= 1 else 2.0 / (self.y_dim - 1)
        grid[..., 0] = base_grid[..., 0] - flow_x[:, 0].mul(x_scale)
        grid[..., 1] = base_grid[..., 1] - flow_y[:, 0].mul(y_scale)
        advected = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return advected.clamp_min_(0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        if reference is None:
            reference = target
        prev_frame = reference.float()
        curr_frame = target.float()
        try:
            flow_x, flow_y = self._estimate_dense_flow(prev_frame, curr_frame)
            teacher = self._advect(prev_frame, flow_x, flow_y).to(dtype=pred.dtype)
        except Exception:
            teacher = curr_frame.to(dtype=pred.dtype)

        if self.loss_type == "l1":
            out = torch.abs(pred - teacher)
        elif self.loss_type == "huber":
            out = F.huber_loss(pred, teacher, reduction="none", delta=self.delta)
        else:
            raise ValueError(f"Unsupported optical-flow loss_type: {self.loss_type}")

        if self.mask_threshold is not None:
            mask = teacher.gt(self.mask_threshold)
            mask.logical_or_(target > self.mask_threshold)
            out = out * mask.to(dtype=out.dtype)
        return out.mean()


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
