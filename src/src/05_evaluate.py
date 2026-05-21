"""
05_evaluate.py — 本地验证集评估（对齐赛题三项指标）

用法：
    python src/05_evaluate.py \
        --weights runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt \
        --data_root ./dataset \
        --split_info yolo_dataset/split_info.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import infer_one_image, load_image_gray, iou_xyxy


def match_preds_to_gts(preds, gts, iou_thr=0.5):
    preds_sorted = sorted(preds, key=lambda p: -p["score"])
    gt_matched = [False] * len(gts)
    tp_flags = []
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if gt_matched[j]: continue
            iou = iou_xyxy(p["xyxy"], g["xyxy"])
            if iou > best_iou:
                best_iou, best_j = iou, j
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
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gts + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0) / 11
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--split_info", type=str, required=True)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou_match", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.fuse
    if args.device == "cpu":
        model.to("cpu")

    split = json.loads(Path(args.split_info).read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])
    data_root = Path(args.data_root)

    with open(data_root / "trainval" / "trainval.json", "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]
    val_samples = [s for s in ds if s["ID"] in val_ids]
    img_dir = data_root / "trainval" / "images"
    print(f"验证集: {len(val_samples)} 张")

    all_records = []
    times = []

    for s in tqdm(val_samples, desc="Evaluating"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists(): continue
        gray, _ = load_image_gray(img_path)
        if gray is None: continue

        t0 = time.perf_counter()
        dets = infer_one_image(model, gray, args.conf, 0.6)
        times.append((time.perf_counter() - t0) * 1000)

        preds = [
            {"xyxy": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
             "score": float(d[4])}
            for d in dets
        ]
        gts = []
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            gts.append({"xyxy": [x, y, x + bw, y + bh], "area": bw * bh})
        all_records.append({"preds": preds, "gts": gts})

    # mAP50
    all_tp = []
    total_gts = 0
    for r in all_records:
        total_gts += len(r["gts"])
        tp_flags, _ = match_preds_to_gts(r["preds"], r["gts"], args.iou_match)
        all_tp.extend(tp_flags)
    ap = compute_ap50(all_tp, total_gts)

    # small-Recall
    tp_small, total_small = 0, 0
    for r in all_records:
        gts_small = [g for g in r["gts"] if g["area"] <= 50]
        total_small += len(gts_small)
        _, gt_matched = match_preds_to_gts(r["preds"], gts_small, args.iou_match)
        tp_small += sum(gt_matched)
    small_recall = tp_small / (total_small + 1e-6) if total_small else 0.0

    # large-mIoU
    large_ious = []
    for r in all_records:
        gts_large = [g for g in r["gts"] if g["area"] >= 300 * 300]
        if not gts_large: continue
        preds_sorted = sorted(r["preds"], key=lambda p: -p["score"])
        used = [False] * len(preds_sorted)
        for g in gts_large:
            best_iou, best_i = 0.0, -1
            for i, p in enumerate(preds_sorted):
                if used[i]: continue
                iou_v = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou_v > best_iou:
                    best_iou, best_i = iou_v, i
            if best_i >= 0: used[best_i] = True
            large_ious.append(best_iou)
    large_miou = float(np.mean(large_ious)) if large_ious else 0.0

    print("\n" + "="*50)
    print("赛题评分维度本地评估")
    print("="*50)
    print(f"mAP50          : {ap:.4f}")
    print(f"small-Recall   : {small_recall:.4f}   "
          f"[命中 {tp_small}/{total_small}]   (面积≤50 px²)")
    print(f"large-mIoU     : {large_miou:.4f}   "
          f"[n={len(large_ious)}]   (面积≥300×300)")
    if times:
        print(f"推理时延       : mean={np.mean(times):.1f}ms, "
              f"p50={np.percentile(times,50):.1f}ms, "
              f"p95={np.percentile(times,95):.1f}ms, max={max(times):.1f}ms")

    sim = (
        min(small_recall / 0.85, 1.0) * 10 +
        min(large_miou   / 0.80, 1.0) * 10 +
        min(ap           / 0.65, 1.0) * 10
    )
    print(f"\n模拟性能得分（检测性能 30/40，不含时延）: {sim:.2f}")


if __name__ == "__main__":
    main()
