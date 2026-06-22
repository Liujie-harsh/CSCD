"""
eval_single.py — 单模型评测（不走 WBF）

复用 09_ensemble_wbf.py 的指标逻辑（mAP50 / small-Recall / large-mIoU），
但直接用单个模型的原始输出（infer_one_image），不做 WBF 融合 / 不改写置信度。
目的：确认单模型能否独立越过三条线，决定是否丢掉双模型集成。

用法：
    python src/src/eval_single.py \
        --weights runs/RT-DERT/rtdetr_l_p2-6/weights/best.pt \
        --trainval_json dataset/trainval/trainval.json \
        --split_info RT-DETR/yolo_dataset/split_info.json \
        --img_dir RT-DETR/yolo_dataset/images \
        --conf 0.001
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import load_image_gray, infer_one_image, iou_xyxy
from importlib import import_module

# 复用集成脚本里的指标函数，保证口径完全一致
_wbf = import_module("09_ensemble_wbf") if Path(__file__).with_name("09_ensemble_wbf.py").exists() \
    else None
if _wbf is not None:
    match_preds_to_gts = _wbf.match_preds_to_gts
    compute_ap50 = _wbf.compute_ap50
else:
    # 兜底：内联同款实现
    def match_preds_to_gts(preds, gts, iou_thr=0.5):
        preds_sorted = sorted(preds, key=lambda p: -p["score"])
        gt_matched = [False] * len(gts)
        tp_flags = []
        for p in preds_sorted:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if gt_matched[j]:
                    continue
                v = iou_xyxy(p["xyxy"], g["xyxy"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                gt_matched[best_j] = True
                tp_flags.append((True, p["score"]))
            else:
                tp_flags.append((False, p["score"]))
        return tp_flags, gt_matched

    def compute_ap50(all_tp_scores, total_gts):
        if not all_tp_scores or total_gts == 0:
            return 0.0
        scores = np.array([s for _, s in all_tp_scores])
        tp = np.array([1 if t else 0 for t, _ in all_tp_scores])
        order = np.argsort(-scores)
        tp = tp[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / (total_gts + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recall >= t
            ap += (precision[mask].max() if mask.any() else 0) / 11
        return ap


def single_predict(model, img_gray, conf=0.001, iou=0.6):
    """单模型原始输出，返回 (N,6)=[x1,y1,x2,y2,score,cls]。"""
    return infer_one_image(model, img_gray, conf=conf, iou=iou)


def evaluate_single(model, trainval_json, split_info, img_dir, conf=0.001):
    split = json.loads(Path(split_info).read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])

    ds = json.loads(Path(trainval_json).read_text(encoding="utf-8"))["Dataset"]
    val_samples = [s for s in ds if s["ID"] in val_ids]
    img_dir = Path(img_dir)
    print(f"验证集: {len(val_samples)} 张 | conf={conf} | img_dir={img_dir}")

    all_records = []
    missing = 0
    for s in tqdm(val_samples, desc="单模型推理"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists():
            missing += 1
            continue
        gray, _ = load_image_gray(img_path)
        if gray is None:
            missing += 1
            continue

        dets = single_predict(model, gray, conf=conf)
        preds = [{"xyxy": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                  "score": float(d[4])} for d in dets]
        gts = []
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            gts.append({"xyxy": [x, y, x + bw, y + bh], "area": bw * bh})
        all_records.append({"preds": preds, "gts": gts})

    if missing:
        print(f"[warn] {missing} 张图片找不到，已跳过（检查 --img_dir）")

    # mAP50
    all_tp, total_gts = [], 0
    for r in all_records:
        total_gts += len(r["gts"])
        tp_flags, _ = match_preds_to_gts(r["preds"], r["gts"], 0.5)
        all_tp.extend(tp_flags)
    ap = compute_ap50(all_tp, total_gts)

    # small-Recall (面积 ≤ 50 px²)
    tp_small, total_small = 0, 0
    for r in all_records:
        gts_small = [g for g in r["gts"] if g["area"] <= 50]
        total_small += len(gts_small)
        _, gt_matched = match_preds_to_gts(r["preds"], gts_small, 0.5)
        tp_small += sum(gt_matched)
    small_recall = tp_small / (total_small + 1e-6) if total_small else 0.0

    # large-mIoU (面积 ≥ 300x300 px²)
    large_ious = []
    for r in all_records:
        gts_large = [g for g in r["gts"] if g["area"] >= 300 * 300]
        if not gts_large:
            continue
        preds_sorted = sorted(r["preds"], key=lambda p: -p["score"])
        used = [False] * len(preds_sorted)
        for g in gts_large:
            best_iou, best_i = 0.0, -1
            for i, p in enumerate(preds_sorted):
                if used[i]:
                    continue
                v = iou_xyxy(p["xyxy"], g["xyxy"])
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0:
                used[best_i] = True
            large_ious.append(best_iou)
    large_miou = float(np.mean(large_ious)) if large_ious else 0.0

    sim = (min(small_recall / 0.85, 1.0) * 10 +
           min(large_miou / 0.80, 1.0) * 10 +
           min(ap / 0.65, 1.0) * 10)

    print("\n" + "=" * 50)
    print("单模型评测结果（无 WBF）")
    print("=" * 50)
    print(f"mAP50          : {ap:.4f}   {'✓过线' if ap >= 0.65 else '✗未过线'} (阈值0.65)")
    print(f"small-Recall   : {small_recall:.4f}   [{tp_small}/{total_small}]   "
          f"{'✓过线' if small_recall >= 0.85 else '✗未过线'} (阈值0.85)")
    print(f"large-mIoU     : {large_miou:.4f}   [n={len(large_ious)}]   "
          f"{'✓过线' if large_miou >= 0.80 else '✗未过线'} (阈值0.80)")
    print(f"模拟得分       : {sim:.2f} / 30")
    return {"mAP50": ap, "small_recall": small_recall,
            "large_miou": large_miou, "sim": sim}


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", required=True, help="单个模型权重路径")
    ap.add_argument("--trainval_json", default="dataset/trainval/trainval.json",
                    help="trainval.json 路径")
    ap.add_argument("--split_info", default="RT-DETR/yolo_dataset/split_info.json")
    ap.add_argument("--img_dir", default="RT-DETR/yolo_dataset/images",
                    help="存放图片的目录")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    evaluate_single(model, args.trainval_json, args.split_info,
                    args.img_dir, conf=args.conf)


if __name__ == "__main__":
    main()
