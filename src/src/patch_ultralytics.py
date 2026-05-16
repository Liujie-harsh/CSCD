"""
patch_ultralytics.py v2 — 注入三个改进到 ultralytics

1. Inner-MPDIoU + NWD 混合 loss（替换 BboxLoss.forward）
2. EMA 注意力模块（注册到 ultralytics 模块系统，在 yaml 中使用）

用法：
    from patch_ultralytics import apply_all_patches
    apply_all_patches()  # 在训练前调用一次
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import inspect

from nwd_loss import nwd_loss
from inner_mpdiou_loss import inner_mpdiou_loss
from ema_attention import EMA

# ======== 可调参数 ========
_INNER_MPDIOU_RATIO = 0.5  # Inner-MPDIoU loss 权重
_NWD_RATIO = 0.5           # NWD loss 权重
_NWD_C = 12.8              # NWD 归一化常数
_INNER_RATIO = 0.7         # Inner-IoU 缩放因子

_PATCH_APPLIED = False


# ======== Loss Patch ========

def _patched_forward_new(
    self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
    target_scores, target_scores_sum, fg_mask, imgsz, stride,
):
    """替换版 BboxLoss.forward：Inner-MPDIoU + NWD 混合 loss。"""
    from ultralytics.utils.tal import bbox2dist

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

    # Inner-MPDIoU loss
    loss_impdiou = inner_mpdiou_loss(
        pred_bboxes[fg_mask],
        target_bboxes[fg_mask],
        ratio=_INNER_RATIO,
        reduction="none",
    ).unsqueeze(-1)
    loss_impdiou = (loss_impdiou * weight).sum() / target_scores_sum

    # NWD loss
    nwd_val = nwd_loss(
        pred_bboxes[fg_mask],
        target_bboxes[fg_mask],
        C=_NWD_C,
        reduction="none",
    ).unsqueeze(-1)
    loss_nwd = (nwd_val * weight).sum() / target_scores_sum

    # 混合
    loss_iou = _INNER_MPDIOU_RATIO * loss_impdiou + _NWD_RATIO * loss_nwd

    # DFL loss（保持原样）
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
            target_ltrb[fg_mask],
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
            .mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def _patched_forward_legacy(
    self, pred_dist, pred_bboxes, anchor_points,
    target_bboxes, target_scores, target_scores_sum, fg_mask,
):
    """旧版 ultralytics 签名（无 imgsz/stride 参数）。"""
    from ultralytics.utils.tal import bbox2dist

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

    loss_impdiou = inner_mpdiou_loss(
        pred_bboxes[fg_mask], target_bboxes[fg_mask],
        ratio=_INNER_RATIO, reduction="none",
    ).unsqueeze(-1)
    loss_impdiou = (loss_impdiou * weight).sum() / target_scores_sum

    nwd_val = nwd_loss(
        pred_bboxes[fg_mask], target_bboxes[fg_mask],
        C=_NWD_C, reduction="none",
    ).unsqueeze(-1)
    loss_nwd = (nwd_val * weight).sum() / target_scores_sum

    loss_iou = _INNER_MPDIOU_RATIO * loss_impdiou + _NWD_RATIO * loss_nwd

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
            target_ltrb[fg_mask],
        ) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.tensor(0.0).to(pred_dist.device)

    return loss_iou, loss_dfl


# ======== EMA Patch ========

def _register_ema_module():
    """将 EMA 注册到 ultralytics 的模块系统中。"""
    try:
        import ultralytics.nn.modules as modules

        # Register EMA in multiple places for compatibility
        modules.EMA = EMA

        # Register to __all__ list if it exists
        if hasattr(modules, '__all__'):
            if 'EMA' not in modules.__all__:
                # Handle both list and tuple types
                if isinstance(modules.__all__, list):
                    modules.__all__.append('EMA')
                elif isinstance(modules.__all__, tuple):
                    modules.__all__ = list(modules.__all__) + ['EMA']

        # Also register to conv submodule if it exists
        if hasattr(modules, 'conv'):
            modules.conv.EMA = EMA

        # Register to tasks module for YAML parsing
        try:
            import ultralytics.nn.tasks as tasks
            tasks.EMA = EMA
        except:
            pass

        # Register to models.common for older versions
        try:
            import ultralytics.models.common as common
            common.EMA = EMA
        except:
            pass

        print("[EMA Patch] ✓ EMA 模块已注册到 ultralytics")
    except Exception as e:
        print(f"[EMA Patch] ⚠ 注册失败: {e}")
        print("[EMA Patch] 将在训练脚本中手动注入")


# ======== 统一入口 ========

def apply_all_patches(
    inner_mpdiou_ratio=0.5,
    nwd_ratio=0.5,
    nwd_c=12.8,
    inner_ratio=0.7,
):
    """
    一次性应用所有 patch。

    Args:
        inner_mpdiou_ratio: Inner-MPDIoU loss 权重
        nwd_ratio:          NWD loss 权重
        nwd_c:              NWD 归一化常数
        inner_ratio:        Inner-IoU 辅助 bbox 缩放因子
    """
    global _PATCH_APPLIED, _INNER_MPDIOU_RATIO, _NWD_RATIO, _NWD_C, _INNER_RATIO

    _INNER_MPDIOU_RATIO = inner_mpdiou_ratio
    _NWD_RATIO = nwd_ratio
    _NWD_C = nwd_c
    _INNER_RATIO = inner_ratio

    if not _PATCH_APPLIED:
        # 1. Loss patch
        try:
            from ultralytics.utils.loss import BboxLoss
            sig = inspect.signature(BboxLoss.forward)
            params = list(sig.parameters.keys())

            if "imgsz" in params and "stride" in params:
                BboxLoss.forward = _patched_forward_new
            else:
                BboxLoss.forward = _patched_forward_legacy

            print(f"[Loss Patch] ✓ Inner-MPDIoU({inner_mpdiou_ratio}) + "
                  f"NWD({nwd_ratio}, C={nwd_c}) 注入成功")
        except Exception as e:
            print(f"[Loss Patch] ❌ 失败: {e}")

        # 2. EMA patch
        _register_ema_module()

        _PATCH_APPLIED = True
    else:
        print(f"[Patch] 参数已更新: Inner-MPDIoU={inner_mpdiou_ratio}, "
              f"NWD={nwd_ratio}, inner_ratio={inner_ratio}")


# 保持向后兼容
def apply_nwd_patch(iou_ratio=0.5, nwd_ratio=0.5, C=12.8):
    """向后兼容接口：自动升级为 Inner-MPDIoU + NWD。"""
    apply_all_patches(inner_mpdiou_ratio=iou_ratio, nwd_ratio=nwd_ratio, nwd_c=C)


if __name__ == "__main__":
    apply_all_patches()
    print("✓ 所有 patch 已生效")
