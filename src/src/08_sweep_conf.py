"""
08_sweep_conf.py — conf 阈值扫描（P2+NWD 版本）

扫描不同 conf 值，找出三项指标综合得分最高的。
不需要重新训练，10 分钟内出结果。

用法：
    python src/08_sweep_conf.py \
        --weights "runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt" \
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
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / (total_gts + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0) / 11
    return ap


def evaluate_with_conf(all_raw_records, conf_thr):
    """对已有的全量预测做 conf 过滤后再评估，避免重复推理。"""
    all_tp = []
    total_gts = 0
    tp_small, total_small = 0, 0
    large_ious = []

    for r in all_raw_records:
        # conf 过滤
        preds = [p for p in r["preds"] if p["score"] >= conf_thr]
        gts = r["gts"]

        # mAP50
        total_gts += len(gts)
        tp_flags, _ = match_preds_to_gts(preds, gts, 0.5)
        all_tp.extend(tp_flags)

        # small-Recall
        gts_small = [g for g in gts if g["area"] <= 50]
        total_small += len(gts_small)
        _, gt_matched = match_preds_to_gts(preds, gts_small, 0.5)
        tp_small += sum(gt_matched)

        # large-mIoU
        gts_large = [g for g in gts if g["area"] >= 300 * 300]
        if gts_large:
            preds_sorted = sorted(preds, key=lambda p: -p["score"])
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

    ap = compute_ap50(all_tp, total_gts)
    sr = tp_small / (total_small + 1e-6) if total_small else 0.0
    miou = float(np.mean(large_ious)) if large_ious else 0.0
    return ap, sr, tp_small, total_small, miou, len(large_ious)


def score(ap, sr, miou):
    return (min(sr / 0.85, 1.0) * 10 +
            min(miou / 0.80, 1.0) * 10 +
            min(ap / 0.65, 1.0) * 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--split_info", type=str, required=True)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
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

    # ====== 第 1 步：用极低 conf 跑一次推理，保存所有预测 ======
    print("\n[Step 1] 用 conf=0.001 跑一次全量推理（保存所有候选框）...")
    all_raw_records = []

    for s in tqdm(val_samples, desc="推理"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists(): continue
        gray, _ = load_image_gray(img_path)
        if gray is None: continue

        # 极低 conf 获取所有候选
        dets = infer_one_image(model, gray, conf=0.001, iou=0.6)

        preds = [
            {"xyxy": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
             "score": float(d[4])}
            for d in dets
        ]
        gts = []
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            gts.append({"xyxy": [x, y, x + bw, y + bh], "area": bw * bh})
        all_raw_records.append({"preds": preds, "gts": gts})

    print(f"✓ 推理完成，共 {len(all_raw_records)} 张图，"
          f"总候选框 {sum(len(r['preds']) for r in all_raw_records)} 个")

    # ====== 第 2 步：扫描 conf ======
    conf_grid = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05,
                 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    print(f"\n[Step 2] 扫描 {len(conf_grid)} 个 conf 值...")
    results = []
    for conf in conf_grid:
        ap_v, sr, tp_s, tot_s, miou, n_large = evaluate_with_conf(all_raw_records, conf)
        s = score(ap_v, sr, miou)
        results.append({
            "conf": conf, "mAP50": ap_v, "small_R": sr,
            "tp_small": tp_s, "total_small": tot_s,
            "large_mIoU": miou, "n_large": n_large, "score": s
        })

    # ====== 排序输出 ======
    print("\n" + "="*90)
    print(f"{'':>2}{'conf':>7} {'mAP50':>8} {'small_R':>9} {'(命中)':>8} "
          f"{'lgIoU':>7} {'score':>7}")
    print("-"*90)
    for r in sorted(results, key=lambda x: -x["score"]):
        best = r == max(results, key=lambda x: x["score"])
        marker = "★" if best else " "
        print(f"{marker} {r['conf']:>6.3f} {r['mAP50']:>8.4f} "
              f"{r['small_R']:>9.4f} ({r['tp_small']}/{r['total_small']}) "
              f"{r['large_mIoU']:>7.4f} {r['score']:>7.2f}")

    best = max(results, key=lambda x: x["score"])
    print("\n" + "="*90)
    print(f"[推荐] conf = {best['conf']}")
    print(f"  mAP50      = {best['mAP50']:.4f}")
    print(f"  small-R    = {best['small_R']:.4f} ({best['tp_small']}/{best['total_small']})")
    print(f"  large-mIoU = {best['large_mIoU']:.4f}")
    print(f"  模拟得分   = {best['score']:.2f} / 30")
    print(f"\n生成最终提交：")
    print(f"  python src/04_inference.py --weights {args.weights} "
          f"--data_root ./dataset --out_path results_final.json --conf {best['conf']}")

    # 保存
    out_dir = Path("outputs/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conf_sweep_p2_nwd.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✓ 完整结果: outputs/sweep/conf_sweep_p2_nwd.json")


if __name__ == "__main__":
    main()