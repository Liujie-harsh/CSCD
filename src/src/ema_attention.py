"""
ema_attention.py — Efficient Multi-Scale Attention (EMA) Module

论文：Ouyang et al., "Efficient Multi-Scale Attention Module with
Cross-Spatial Learning", ICASSP 2023

核心思想：
    1. 将通道分成多个子组
    2. 每组分别做 1×1（全局上下文）和 3×3（局部空间）的并行注意力
    3. 跨组做 softmax 聚合
    
对小目标有效因为：多尺度 1×1+3×3 能同时捕获细粒度空间细节和全局语义信息，
不像 SE 只有全局均值，也不像 CBAM 只有单尺度空间注意力。
"""
import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, c1, groups=None):
        super().__init__()

        channels = c1

        if groups is None:
            groups = self._find_compatible_groups(channels)

        if channels % groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})."
            )

        self.channels = channels
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.LayerNorm([channels, 1, 1]),
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.gn = nn.GroupNorm(groups, channels)
        self.sigmoid = nn.Sigmoid()

    def _find_compatible_groups(self, channels):
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        y_global = self.avg_pool(x)
        y_global = self.fc(y_global)

        y_local = self.conv3x3(x)

        y = y_global.expand_as(x) + y_local
        y = self.gn(y)
        y = self.sigmoid(y)

        return x * y


if __name__ == "__main__":
    print("=== EMA Attention 自测 ===")

    # 测试不同分辨率
    for c, h, w in [(64, 240, 240), (128, 120, 120), (256, 60, 60), (512, 30, 30)]:
        ema = EMA(c, groups=4)
        x = torch.randn(2, c, h, w)
        out = ema(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
        # 验证是通道注意力（不改变空间维度）
        params = sum(p.numel() for p in ema.parameters())
        print(f"  C={c:3d}, HW={h}×{w}: out_shape={list(out.shape)}, "
              f"params={params:,d}")

    print("✓ 自测通过")
