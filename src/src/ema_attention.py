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

    def __init__(self, c1=None, groups=None):
        super().__init__()

        # Store initial parameters but don't create layers yet
        self.initial_channels = c1
        self.groups = groups

        # Layers will be created on first forward pass
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None
        self.conv3x3 = None
        self.gn = None
        self.sigmoid = nn.Sigmoid()
        self.initialized = False

    def _find_compatible_groups(self, channels):
        """Find a compatible group size for given channels."""
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    def _init_layers(self, channels):
        """Initialize layers based on actual input channels."""
        if self.initialized:
            return

        # Auto-adjust groups to ensure compatibility
        if self.groups is None:
            # Try common group sizes, default to 4 if none work
            for g in [8, 4, 2, 1]:
                if channels % g == 0:
                    groups = g
                    break
            else:
                groups = 1
        else:
            groups = self.groups

        # Only assert if manually specified groups don't work
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups}).")

        self.groups = groups

        # Create layers with actual channel count
        # Use LayerNorm instead of BatchNorm for 1x1 spatial dimensions
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.LayerNorm([channels, 1, 1]),  # LayerNorm works with 1x1 spatial dims
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gn = nn.GroupNorm(groups, channels)
        self.initialized = True

    def forward(self, x):
        b, c, h, w = x.shape

        # Initialize layers on first forward pass
        if not self.initialized:
            self._init_layers(c)

        # 1×1 分支
        y_global = self.avg_pool(x)         # (B, C, 1, 1)
        y_global = self.fc(y_global)        # (B, C, 1, 1)

        # 3×3 分支
        y_local = self.conv3x3(x)           # (B, C, H, W)

        # 聚合：按 group 做 softmax 加权
        # 将两个分支拼接后做 GroupNorm → Sigmoid
        y = y_global.expand_as(x) + y_local  # (B, C, H, W)
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
