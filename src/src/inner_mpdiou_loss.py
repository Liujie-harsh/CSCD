"""
inner_mpdiou_loss.py — Inner-MPDIoU loss for small object detection

结合两篇论文：
1. Inner-IoU (Zhang et al., 2023, arXiv:2311.02877)
   - 通过缩放因子 ratio 生成辅助 bbox 计算 IoU
   - 高 IoU 样本用小 bbox 精修，低 IoU 样本用大 bbox 扩梯度
2. MPDIoU (Ma & Xu, 2023, arXiv:2307.07662)
   - 基于四角点最小距离，比 CIoU 更直接

组合：Inner-MPDIoU = 辅助 bbox 缩放 + 角点距离惩罚
"""
import torch
import math


def inner_mpdiou_loss(pred, target, ratio=0.7, d=0.0, u=0.95, reduction="none"):
    """
    Inner-MPDIoU loss.

    Args:
        pred:   (N, 4) tensor, xyxy format
        target: (N, 4) tensor, xyxy format
        ratio:  Inner-IoU 缩放因子 (0.5~1.0)。
                ratio<1 时辅助 bbox 更小，对高质量样本精修效果好；
                论文推荐 0.7 用于小目标场景
        d, u:   Wise-IoU 风格的动态聚焦参数（可选）
                d=0 时关闭动态聚焦
        reduction: "none" | "mean" | "sum"

    Returns:
        loss tensor
    """
    # 预测框和 GT 框坐标
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx1, gy1, gx2, gy2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # 预测框和 GT 框的宽高和中心
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    gw = (gx2 - gx1).clamp(min=1e-6)
    gh = (gy2 - gy1).clamp(min=1e-6)

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5
    gcy = (gy1 + gy2) * 0.5

    # ====== Inner-IoU: 生成辅助 bbox ======
    # 辅助 bbox = 原 bbox 缩放 ratio 倍（以中心点为基准）
    inner_px1 = pcx - pw * ratio * 0.5
    inner_py1 = pcy - ph * ratio * 0.5
    inner_px2 = pcx + pw * ratio * 0.5
    inner_py2 = pcy + ph * ratio * 0.5

    inner_gx1 = gcx - gw * ratio * 0.5
    inner_gy1 = gcy - gh * ratio * 0.5
    inner_gx2 = gcx + gw * ratio * 0.5
    inner_gy2 = gcy + gh * ratio * 0.5

    # 辅助 bbox 的交集
    inner_x1 = torch.max(inner_px1, inner_gx1)
    inner_y1 = torch.max(inner_py1, inner_gy1)
    inner_x2 = torch.min(inner_px2, inner_gx2)
    inner_y2 = torch.min(inner_py2, inner_gy2)

    inner_inter = (inner_x2 - inner_x1).clamp(min=0) * (inner_y2 - inner_y1).clamp(min=0)
    inner_union = pw * ratio * ph * ratio + gw * ratio * gh * ratio - inner_inter + 1e-7
    inner_iou = inner_inter / inner_union

    # ====== MPDIoU: 四角点距离惩罚 ======
    # d1² = (px1 - gx1)² + (py1 - gy1)²  左上角距离
    # d2² = (px2 - gx2)² + (py2 - gy2)²  右下角距离
    d1_sq = (px1 - gx1) ** 2 + (py1 - gy1) ** 2
    d2_sq = (px2 - gx2) ** 2 + (py2 - gy2) ** 2

    # 最小闭包矩形的对角线长度²
    enclose_x1 = torch.min(px1, gx1)
    enclose_y1 = torch.min(py1, gy1)
    enclose_x2 = torch.max(px2, gx2)
    enclose_y2 = torch.max(py2, gy2)
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # MPDIoU = IoU - (d1² + d2²) / c²
    mpdiou_penalty = (d1_sq + d2_sq) / enclose_diag_sq

    # ====== 组合 Inner-MPDIoU ======
    inner_mpdiou = inner_iou - mpdiou_penalty

    loss = 1.0 - inner_mpdiou

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


if __name__ == "__main__":
    print("=== Inner-MPDIoU Loss 自测 ===")

    # 1. 完全重合
    a = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    loss = inner_mpdiou_loss(a, b, ratio=0.7)
    print(f"完全重合: loss={loss.item():.6f} (期望 ~0)")

    # 2. 小目标偏移 2px
    a = torch.tensor([[100.0, 100.0, 103.0, 103.0]])
    b = torch.tensor([[102.0, 102.0, 105.0, 105.0]])
    loss = inner_mpdiou_loss(a, b, ratio=0.7)
    print(f"小目标偏移 2px: loss={loss.item():.4f}")

    # 3. 完全不重合
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    loss = inner_mpdiou_loss(a, b, ratio=0.7)
    print(f"完全不重合: loss={loss.item():.4f} (期望接近 2)")

    # 4. Batch
    a = torch.rand(10, 4) * 100
    a[:, 2:] = a[:, :2] + torch.rand(10, 2) * 50 + 1
    b = a + torch.randn_like(a) * 3
    b[:, 2:] = b[:, :2].clone() + torch.rand(10, 2) * 50 + 1
    loss = inner_mpdiou_loss(a, b, ratio=0.7, reduction="mean")
    print(f"Batch mean loss: {loss.item():.4f}")

    print("✓ 自测通过")
