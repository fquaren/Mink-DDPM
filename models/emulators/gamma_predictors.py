import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Shared Mathematical Library
# ==========================================


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
        # Direct pass to feature extractor
        feat = self.features(x).flatten(1)

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
    """

    def __init__(self, n_quantiles=30, input_shape=(1, 128, 128)):
        super().__init__()

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

        self.head_A = nn.Linear(256, n_quantiles)
        self.head_P = nn.Linear(256, n_quantiles)
        self.head_CC = nn.Linear(256, n_quantiles)

    def forward(self, x):
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
    ):
        super().__init__()
        if quantile_levels is None:
            raise ValueError("Quantile levels required.")

        max_physical_area = input_shape[1] * input_shape[2] * pixel_area_km2
        self.register_buffer(
            "max_total_area", torch.tensor(max_physical_area, dtype=torch.float32)
        )

        self.n_quantiles = n_quantiles
        self.pixel_area_km2 = pixel_area_km2

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

        self.head_A_total = nn.Linear(256, 1)
        self.head_A_logits = nn.Linear(256, n_quantiles)
        self.head_P_roughness = nn.Linear(256, n_quantiles)
        self.head_CC = nn.Linear(256, n_quantiles)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.head_A_total.weight, 0)
        nn.init.constant_(self.head_A_total.bias, -2.0)

        nn.init.constant_(self.head_A_logits.weight, 0)
        nn.init.constant_(self.head_A_logits.bias, 0)

    def forward(self, x):
        # Expected input is [0, 1]
        epsilon = 1e-6

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
        pred_total_area = torch.sigmoid(self.head_A_total(latent)) * self.max_total_area

        probs_A = F.softmax(self.head_A_logits(latent), dim=1)
        scaled_A_pdf = probs_A * (pred_total_area + epsilon)

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
