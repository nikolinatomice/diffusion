import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Time embedding
# ---------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        # t: [B]
        return self.mlp(t.float().unsqueeze(-1))


# ---------------------------------------------------------
# Basic diffusion block
# ---------------------------------------------------------

class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add timestep conditioning
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


# ---------------------------------------------------------
# Tiny Diffusion U-Net
# ---------------------------------------------------------

class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        time_dim=128,
    ):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        # Encoder
        self.enc1 = DiffusionBlock(
            in_channels,
            base_channels,
            time_dim,
        )

        self.enc2 = DiffusionBlock(
            base_channels,
            base_channels * 2,
            time_dim,
        )

        # Bottleneck
        self.mid = DiffusionBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim,
        )

        # Decoder
        self.dec2 = DiffusionBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.dec1 = DiffusionBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        self.final = nn.Conv2d(
            base_channels,
            out_channels,
            1,
        )

    def forward(self, x, t):

        # ---------------------------------------------
        # Time embedding
        # ---------------------------------------------

        t_emb = self.time_embedding(t)

        # ---------------------------------------------
        # Encoder
        # ---------------------------------------------

        e1 = self.enc1(x, t_emb)

        e2 = self.enc2(
            F.avg_pool2d(e1, 2),
            t_emb,
        )

        # ---------------------------------------------
        # Bottleneck
        # ---------------------------------------------

        mid = self.mid(
            F.avg_pool2d(e2, 2),
            t_emb,
        )

        # ---------------------------------------------
        # Decoder
        # ---------------------------------------------

        d2 = F.interpolate(
            mid,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2, t_emb)

        d1 = F.interpolate(
            d2,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1, t_emb)

        # ---------------------------------------------
        # Output
        # ---------------------------------------------

        return self.final(d1)