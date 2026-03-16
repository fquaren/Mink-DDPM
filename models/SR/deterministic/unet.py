import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LogSpaceResidualUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            # Bilinear upsampling to avoid checkerboard artifacts
            self.ups.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
            self.ups.append(DoubleConv(feature * 2 + feature, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, 2, H, W] -> channel 0 is interp_precip, channel 1 is dem
        interp_log = x[:, 0:1, :, :]

        skip_connections = []
        out = x

        for down in self.downs:
            out = down(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            out = self.ups[i](out)
            skip_connection = skip_connections[i // 2]

            # Handle rounding errors in pooling
            if out.shape != skip_connection.shape:
                out = F.interpolate(
                    out,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            out = torch.cat((skip_connection, out), dim=1)
            out = self.ups[i + 1](out)

        residual_log = self.final_conv(out)

        # Additive in log space. No ReLU needed here if the residual
        # is allowed to suppress the interpolated field (R < 0).
        predicted_log = interp_log + residual_log

        # Physical bounding and residual addition
        return torch.clamp(predicted_log, min=0.0)
