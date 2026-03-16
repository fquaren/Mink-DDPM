import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Classes (DoubleConv, Down, Up, SelfAttention) remain unchanged ---
# (I have kept them condensed here to save space, paste the previous versions back if needed)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to (B, H*W, C) for MultiheadAttention
        x_reshaped = x.view(B, C, H * W).swapaxes(1, 2)
        x_norm = self.ln(x_reshaped)

        q = k = v = x_norm
        attention_val = F.scaled_dot_product_attention(
            q, k, v
        )  # Use PyTorch's built-in function for efficiency

        # Residual connection and reshape back to (B, C, H, W)
        attention_val = attention_val.swapaxes(1, 2).view(B, C, H, W)
        return x + attention_val


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ContextUnet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        c_in_condition=1,
        n_classes=1,
        time_dim=256,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.total_in_channels = in_channels + c_in_condition

        # --- Encoder ---
        self.inc = DoubleConv(self.total_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.sa1 = SelfAttention(256)  # Attention at 32x32

        self.down3 = Down(256, 512)
        self.sa2 = SelfAttention(512)  # Attention at 16x16

        # --- Bottleneck ---
        self.bot1 = DoubleConv(512, 1024)
        self.sa3 = SelfAttention(1024)  # Attention at 16x16
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 512)

        # --- Decoder ---
        self.up1 = Up(768, 256)
        self.sa4 = SelfAttention(256)  # Attention on upsample

        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, condition):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x = torch.cat([x, condition], dim=1)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t)

        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.sa3(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder
        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)

        output = self.outc(x)
        return output
