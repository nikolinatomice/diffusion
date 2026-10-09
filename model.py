"""Tiny diffusion U-Net (this is your old main.py, renamed + improved)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Time embedding
# ---------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Maps integer timesteps -> fixed sin/cos features (like Transformer positions).

    Feeding the raw integer t (0..999) into a Linear layer, as before, makes it very
    hard for the network to tell nearby timesteps apart. Sinusoidal features fix that.
    """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "time_dim must be even"
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        # t: [B] integer timesteps
        return self.mlp(t)


# ---------------------------------------------------------
# Basic diffusion block
# ---------------------------------------------------------

class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

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

        # Timestep conditioning
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


# ---------------------------------------------------------
# Self-attention (global context)
# ---------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.permute(1, 0, 2, 4, 3)  # each [B, heads, HW, d]
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class Stage(nn.Module):
    """Residual block, optionally followed by self-attention."""

    def __init__(self, in_ch, out_ch, time_dim, attn):
        super().__init__()
        self.res = DiffusionBlock(in_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------
# Diffusion U-Net
# ---------------------------------------------------------

class DiffusionUNet(nn.Module):
    """DDPM-style U-Net.

    For 32x32 input and the defaults, resolutions are 32 -> 16 -> 8 -> 4 with
    channels 64 / 128 / 256 / 256, `blocks_per_level` residual blocks per level,
    and self-attention at the 8x8 and 4x4 levels (attn_levels=(2, 3)) + bottleneck.
    Image size must be divisible by 2 ** (len(channel_mults) - 1).
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_dim=256,
        channel_mults=(1, 2, 4, 4),
        attn_levels=(2, 3),
        blocks_per_level=2,
    ):
        super().__init__()
        self.downscale = 2 ** (len(channel_mults) - 1)
        self.time_embedding = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        skip_chs = [ch]
        for lvl, mult in enumerate(channel_mults):
            out = base_channels * mult
            level = nn.ModuleList()
            for _ in range(blocks_per_level):
                level.append(Stage(ch, out, time_dim, attn=lvl in attn_levels))
                ch = out
                skip_chs.append(ch)
            self.down.append(level)
            if lvl != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                skip_chs.append(ch)

        # ---- bottleneck ----
        self.mid1 = Stage(ch, ch, time_dim, attn=True)
        self.mid2 = Stage(ch, ch, time_dim, attn=False)

        # ---- decoder ----
        self.up = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl in reversed(range(len(channel_mults))):
            out = base_channels * channel_mults[lvl]
            level = nn.ModuleList()
            for _ in range(blocks_per_level + 1):
                level.append(Stage(ch + skip_chs.pop(), out, time_dim, attn=lvl in attn_levels))
                ch = out
            self.up.append(level)
            if lvl != 0:
                self.upsamples.append(Upsample(ch))

        self.final_norm = nn.GroupNorm(8, ch)
        self.final = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        assert x.shape[-1] % self.downscale == 0 and x.shape[-2] % self.downscale == 0, \
            f"image height/width must be divisible by {self.downscale}"

        t_emb = self.time_embedding(t)

        # encoder
        h = self.in_conv(x)
        skips = [h]
        for lvl, level in enumerate(self.down):
            for stage in level:
                h = stage(h, t_emb)
                skips.append(h)
            if lvl < len(self.downsamples):
                h = self.downsamples[lvl](h)
                skips.append(h)

        # bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # decoder
        for i, level in enumerate(self.up):
            for stage in level:
                h = stage(torch.cat([h, skips.pop()], dim=1), t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.final(F.silu(self.final_norm(h)))
