from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=t.device, dtype=t.dtype
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        arguments = t * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ConditioningEmbedding(nn.Module):
    def __init__(self, num_classes: int, time_embed_dim: int = 128, cond_dim: int = 512):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_embed = nn.Embedding(num_classes, cond_dim)

    def forward(self, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_embed(t)) + self.class_embed(labels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.cond_proj(F.silu(cond)).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + residual


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        num_heads = max(1, channels // 64)
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, channels, height, width = x.shape
        tokens = self.norm(x).reshape(batch_size, channels, height * width).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = self.out_proj(attended)
        attended = attended.transpose(1, 2).reshape(batch_size, channels, height, width)
        return residual + attended


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ClassConditionedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: int = 50,
        base_channels: int = 128,
        num_res_blocks: int = 3,
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        cond_dim: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        self.level_channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.conditioning = ConditioningEmbedding(num_classes=num_classes, cond_dim=cond_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_levels = nn.ModuleList()
        current_channels = base_channels
        resolution = 64
        for level_index, level_channels in enumerate(self.level_channels):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(current_channels, level_channels, cond_dim))
                attentions.append(SelfAttentionBlock(level_channels) if resolution == 32 else nn.Identity())
                current_channels = level_channels
            downsample = Downsample(current_channels) if level_index < len(self.level_channels) - 1 else nn.Identity()
            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "attentions": attentions,
                        "downsample": downsample,
                    }
                )
            )
            if level_index < len(self.level_channels) - 1:
                resolution //= 2

        self.mid_block1 = ResBlock(current_channels, current_channels, cond_dim)
        self.mid_block2 = ResBlock(current_channels, current_channels, cond_dim)

        self.up_levels = nn.ModuleList()
        resolution = 64 // (2 ** (len(self.level_channels) - 1))
        for reversed_index, level_channels in enumerate(reversed(self.level_channels)):
            level_index = len(self.level_channels) - 1 - reversed_index
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(current_channels + level_channels, level_channels, cond_dim))
                attentions.append(SelfAttentionBlock(level_channels) if resolution == 32 else nn.Identity())
                current_channels = level_channels
            upsample = Upsample(current_channels) if level_index > 0 else nn.Identity()
            self.up_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "attentions": attentions,
                        "upsample": upsample,
                    }
                )
            )
            if level_index > 0:
                resolution *= 2

        self.out_norm = nn.GroupNorm(group_count(current_channels), current_channels)
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.conditioning(t, labels)
        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for level in self.down_levels:
            for block, attention in zip(level["blocks"], level["attentions"]):
                h = block(h, cond)
                h = attention(h)
                skips.append(h)
            h = level["downsample"](h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        for level in self.up_levels:
            for block, attention in zip(level["blocks"], level["attentions"]):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond)
                h = attention(h)
            h = level["upsample"](h)

        return self.out_conv(F.silu(self.out_norm(h)))
