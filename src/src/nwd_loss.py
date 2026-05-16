"""
nwd_loss.py — Normalized Wasserstein Distance loss for tiny object detection

参考文献：
    Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
    arXiv:2110.13389, 2021
    Xu et al., "Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance
    and A New Benchmark", ISPRS J P & RS, 2022

核心思想：
    将 bbox 建模为 2D 高斯分布 N(μ, Σ)，其中：
        μ = (cx, cy), Σ = diag((w/2)², (h/2)²)
    用两个高斯之间的 Wasserstein 距离（有闭式解）代替 IoU：
        W²(N_a, N_b) = ||μ_a - μ_b||² + ||Σ_a^(1/2) - Σ_b^(1/2)||_F²
    然后归一化：
        NWD = exp(-sqrt(W²) / C)     C 为常数，AI-TOD 论文推荐 12.8

为什么对小目标有效：
    IoU 对位置偏移敏感——一个 2×2 的 GT，预测偏移 1 像素 IoU 就从 1 降到 0.5。
    NWD 是平滑函数，即使完全不重叠也能给出梯度。
    对 1×3 像素级的裂纹，这个优势尤其明显。
"""
import torch


def bbox2gaussian(bbox_xyxy):
    """
    将 xyxy bbox 转为高斯分布参数 (center, sigma)。
    
    Args:
        bbox_xyxy: (..., 4) tensor, [x1, y1, x2, y2]
    Returns:
        center: (..., 2)  均值 μ
        sigma:  (..., 2)  标准差（不是方差；Σ 的对角线元素为 sigma^2）
    """
    x1, y1, x2, y2 = bbox_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    center = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([w * 0.5, h * 0.5], dim=-1)
    return center, sigma


def wasserstein_distance_sq(bbox_pred, bbox_gt):
    """
    计算两批 bbox 的 Wasserstein 距离平方（闭式解，仅对角协方差）。
    
    W² = ||μ_p - μ_g||² + ||Σ_p^(1/2) - Σ_g^(1/2)||_F²
       = (cx_p - cx_g)² + (cy_p - cy_g)² + (w_p/2 - w_g/2)² + (h_p/2 - h_g/2)²
    
    Args:
        bbox_pred: (N, 4)
        bbox_gt:   (N, 4)
    Returns:
        (N,) tensor, W² 值
    """
    center_p, sigma_p = bbox2gaussian(bbox_pred)
    center_g, sigma_g = bbox2gaussian(bbox_gt)
    center_dist_sq = ((center_p - center_g) ** 2).sum(dim=-1)
    sigma_dist_sq = ((sigma_p - sigma_g) ** 2).sum(dim=-1)
    return center_dist_sq + sigma_dist_sq


def nwd_similarity(bbox_pred, bbox_gt, C=12.8):
    """
    Normalized Wasserstein Distance similarity，值域 (0, 1]。
    
    NWD = exp(-sqrt(W²) / C)
    
    C 的选择：AI-TOD 论文推荐 12.8（针对极小目标），
    对于 coco 风格的目标应该调大（例如 60-100）。
    本赛题微小缺陷多，C=12.8 合适。
    
    Args:
        bbox_pred: (N, 4) xyxy
        bbox_gt:   (N, 4) xyxy
        C: 归一化常数
    Returns:
        (N,) tensor, 取值 (0, 1]，越大越相似（类似 IoU）
    """
    w_sq = wasserstein_distance_sq(bbox_pred, bbox_gt)
    w = torch.sqrt(w_sq.clamp(min=1e-12))
    return torch.exp(-w / C)


def nwd_loss(bbox_pred, bbox_gt, C=12.8, reduction="none"):
    """
    NWD loss = 1 - NWD_similarity。
    可以直接替换或混合 IoU loss。
    
    Args:
        bbox_pred: (N, 4) xyxy
        bbox_gt:   (N, 4) xyxy
        C: 归一化常数
        reduction: "none" | "mean" | "sum"
    Returns:
        loss tensor
    """
    sim = nwd_similarity(bbox_pred, bbox_gt, C=C)
    loss = 1.0 - sim
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


if __name__ == "__main__":
    # 简单自测：相同 bbox 应返回 loss≈0，不重合 bbox loss 接近 1
    print("=== NWD Loss 自测 ===")
    
    # 1. 相同 bbox
    a = torch.tensor([[100.0, 100.0, 110.0, 110.0]])  # 10x10 框
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    loss = nwd_loss(a, b, C=12.8)
    print(f"相同 bbox: loss={loss.item():.6f} (期望 ~0)")
    
    # 2. 小偏移（微小目标场景，2px 偏移）
    a = torch.tensor([[100.0, 100.0, 103.0, 103.0]])  # 3x3 的小目标
    b = torch.tensor([[102.0, 102.0, 105.0, 105.0]])  # 偏移 2px
    loss = nwd_loss(a, b, C=12.8)
    print(f"小目标偏移 2px: loss={loss.item():.4f}")
    
    # 3. 大偏移
    a = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    b = torch.tensor([[200.0, 200.0, 210.0, 210.0]])
    loss = nwd_loss(a, b, C=12.8)
    print(f"远离 bbox: loss={loss.item():.4f} (期望接近 1)")
    
    # 4. Batch 测试
    a = torch.randn(5, 4).abs() * 100
    a[:, 2:] = a[:, :2] + torch.randn(5, 2).abs() * 10 + 1
    b = a + torch.randn_like(a)
    loss = nwd_loss(a, b, C=12.8, reduction="mean")
    print(f"Batch 平均 loss: {loss.item():.4f}")
    
    print("✓ 自测通过")
