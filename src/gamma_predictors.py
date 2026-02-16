import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# Shared Mathematical Library
# ==========================================


class InputNormalization(nn.Module):
    """
    Project input onto a stable numerical range.
    Essential for preventing immediate gradient explosion at initialization.
    """

    def __init__(self, max_precip_value=150.0):
        super().__init__()
        self.register_buffer(
            "scale_factor", torch.tensor(max_precip_value, dtype=torch.float32)
        )

    def forward(self, x):
        return x / (self.scale_factor + 1e-8)


class RobustBlock(nn.Module):
    """
    Standard ResBlock with GroupNorm.
    GroupNorm is preferred over BatchNorm for physical emulators as it is
    independent of batch statistics, which can be unstable with sparse data.
    """

    def __init__(self, channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(x))
        out = self.conv1(out)
        out = F.gelu(self.gn2(out))
        out = self.conv2(out)
        return residual + out


# ==========================================
# MODEL 1: Simple CNN
# ==========================================


class BaselineCNN(nn.Module):
    """
    Standard 'Black Box' CNN.
    - No Spectral Norm (Prone to exploding/vanishing gradients)
    - MaxPool (Aliasing/Shift-Variance)
    - ReLU (Dead neurons)
    - GlobalAvgPool (Destroys count/mass information)
    """

    def __init__(self, n_quantiles=30, input_shape=(1, 128, 128)):
        super().__init__()
        self.norm = InputNormalization()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten_dim = 256
        self.head_A = nn.Linear(self.flatten_dim, n_quantiles)
        self.head_P = nn.Linear(self.flatten_dim, n_quantiles)
        self.head_CC = nn.Linear(self.flatten_dim, n_quantiles)

    def forward(self, x):
        x = self.norm(x)
        feat = self.features(x).flatten(1)

        # Unconstrained ReLU regression
        pred_A = F.relu(self.head_A(feat))
        pred_P = F.relu(self.head_P(feat))
        pred_CC = F.relu(self.head_CC(feat))

        return torch.stack([pred_A, pred_P, pred_CC], dim=1)


# ==========================================
# MODEL 2: Lipschitz CNN
# ==========================================


class LipschitzCNN(nn.Module):
    """
    Model 2: The Theorist ("Lipschitz CNN") - Robust Version

    Architecture matches ConstrainedLipschitzCNN (Standard Conv + MaxPool + SumPool)
    to ensure fair comparison, but outputs are statistically independent
    (no monotonicity or isoperimetric constraints enforced).
    """

    def __init__(self, n_quantiles=30, input_shape=(1, 128, 128), max_input_val=5.5):
        super().__init__()

        self.register_buffer(
            "input_scale", torch.tensor(max_input_val, dtype=torch.float32)
        )

        # --- Layer 1: High-Res Feature Extraction ---
        self.entry = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)

        # --- Downsampling Path ---
        # Match Constrained Model: MaxPool to preserve sparse signals
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64
        self.res1 = RobustBlock(32)

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32
        self.res2 = RobustBlock(64)

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.pool3 = nn.MaxPool2d(2)  # 32 -> 16
        self.res3 = RobustBlock(128)

        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1))
        self.pool4 = nn.MaxPool2d(2)  # 16 -> 8
        self.res4 = RobustBlock(256)

        self.res5 = RobustBlock(256)  # Deep processing

        # --- Global Integration ---
        self.fc = nn.Linear(256, 256)

        # --- Heads (Unconstrained) ---
        # Standard Softplus regression.
        # Unlike the Constrained model, these predict the values directly
        # and independently, without enforcing PDF integration or Geometry.
        self.head_A = nn.Linear(256, n_quantiles)
        self.head_P = nn.Linear(256, n_quantiles)
        self.head_CC = nn.Linear(256, n_quantiles)

    def forward(self, x):
        x = x / (self.input_scale + 1e-8)

        # L1: Entry (Full Res)
        x = F.gelu(self.entry(x))

        # L2: 64x64
        x = self.pool1(x)
        x = self.res1(x)

        # L3: 32x32
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        x = self.res2(x)

        # L4: 16x16
        x = F.gelu(self.conv3(x))
        x = self.pool3(x)
        x = self.res3(x)

        # L5: 8x8
        x = F.gelu(self.conv4(x))
        x = self.pool4(x)
        x = self.res4(self.res5(x))

        # --- Sum Pooling ---
        # Match Constrained Model: Sum without scaling to preserve gradient magnitude
        feat = x.sum(dim=(2, 3))

        latent = F.gelu(self.fc(feat))

        # Independent Soft Regressions (Theorist Mode)
        pred_A = F.softplus(self.head_A(latent))
        pred_P = F.softplus(self.head_P(latent))
        pred_CC = F.softplus(self.head_CC(latent))

        return torch.stack([pred_A, pred_P, pred_CC], dim=1)


# ==========================================
# MODEL 3: Constrained CNN
# ==========================================
class ConstrainedLipschitzCNN(nn.Module):
    def __init__(
        self,
        n_quantiles=30,
        input_shape=(1, 128, 128),
        quantile_levels=None,
        pixel_area_km2=4.0,
        max_input_val=5.5,
    ):
        super().__init__()
        if quantile_levels is None:
            raise ValueError("Quantile levels required.")

        # Normalize input to [0, 1]
        # We assume input is Log-Transformed. We divide by max expected log value.
        self.register_buffer(
            "input_scale", torch.tensor(max_input_val, dtype=torch.float32)
        )

        # Store max physical area for bounding
        # shape is (C, H, W) -> H * W * pixel_area
        max_physical_area = input_shape[1] * input_shape[2] * pixel_area_km2
        self.register_buffer(
            "max_total_area", torch.tensor(max_physical_area, dtype=torch.float32)
        )

        self.n_quantiles = n_quantiles
        self.pixel_area_km2 = pixel_area_km2

        # --- Layer 1 ---
        self.entry = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1)

        self.pool1 = nn.MaxPool2d(2)
        self.res1 = RobustBlock(32)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.pool2 = nn.MaxPool2d(2)
        self.res2 = RobustBlock(64)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.pool3 = nn.MaxPool2d(2)
        self.res3 = RobustBlock(128)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=1))
        self.pool4 = nn.MaxPool2d(2)
        self.res4 = RobustBlock(256)
        self.res5 = RobustBlock(256)
        self.fc = nn.Linear(256, 256)

        # --- Heads ---
        self.head_A_total = nn.Linear(256, 1)
        self.head_A_logits = nn.Linear(256, n_quantiles)
        self.head_P_roughness = nn.Linear(256, n_quantiles)
        self.head_CC = nn.Linear(256, n_quantiles)

        self._init_weights()

    def _init_weights(self):
        """
        Scientific Initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize regression head biases
        nn.init.constant_(self.head_A_total.weight, 0)
        # Initialize total area bias to produce small initial output (e.g. sigmoid(0) * max)
        nn.init.constant_(self.head_A_total.bias, -2.0)

        # Initialize Logits to Zero for Max Entropy (Uniform Distribution start)
        nn.init.constant_(self.head_A_logits.weight, 0)
        nn.init.constant_(self.head_A_logits.bias, 0)

    def forward(self, x_phys):
        # Apply Normalization
        x = x_phys / (self.input_scale + 1e-8)

        epsilon = 1e-6

        # Forward Pass Layers (Identical to LipschitzCNN for fair comparison)
        x = F.gelu(self.entry(x))
        x = self.pool1(x)
        x = self.res1(x)
        x = F.gelu(self.conv2(x))
        x = self.pool2(x)
        x = self.res2(x)
        x = F.gelu(self.conv3(x))
        x = self.pool3(x)
        x = self.res3(x)
        x = F.gelu(self.conv4(x))
        x = self.pool4(x)
        x = self.res4(self.res5(x))
        feat = x.sum(dim=(2, 3))
        latent = F.gelu(self.fc(feat))

        # --- Head 1: Area ---
        # Bounded Total Area (Sigmoid * MaxArea)
        # Using Sigmoid allows 0 -> Max, physically correct
        pred_total_area = torch.sigmoid(self.head_A_total(latent)) * self.max_total_area

        # PDF distribution
        probs_A = F.softmax(self.head_A_logits(latent), dim=1)

        # Distribute Total Area according to PDF
        scaled_A_pdf = probs_A * (pred_total_area + epsilon)

        # Integrate (CumSum) to get Excursion Sets
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_A_pdf, dims=[1]), dim=1), dims=[1]
        )

        # --- Head 2: Perimeter ---
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R = 1.0 + F.softplus(self.head_P_roughness(latent))
        pred_P = P_min * R

        # --- Head 3: CC ---
        pred_CC = F.softplus(self.head_CC(latent))

        return torch.stack([pred_A, pred_P, pred_CC], dim=1)
