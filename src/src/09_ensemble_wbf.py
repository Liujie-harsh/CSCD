"""
09_ensemble_wbf.py — WBF (Weighted Boxes Fusion) 多模型集成

将多个模型的预测框加权融合，利用各模型优势互补提升 mAP。

原理（Solovyev et al., 2021, Image and Vision Computing）：
    与 NMS 丢弃低置信度框不同，WBF 保留所有框并用置信度加权求平均。
    这样能：
    1. 减少漏检（多个模型中至少一个检测到的目标不会被丢弃）
    2. 提高定位精度（多个框求平均比单个框更接近 GT）

安装依赖：
    pip install ensemble-boxes

用法（评估）：
    python src/09_ensemble_wbf.py \
        --weights "runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt" \
                  "runs/detect/runs/crack/yolov8s_p2_nwd/weights/best.pt" \
                  "runs/detect/runs/crack/yolov8s_p2_ema_impdiou_nwd/weights/best.pt" \
        --data_root ./dataset \
        --split_info yolo_dataset/split_info.json \
        --mode eval

用法（生成提交）：
    python src/09_ensemble_wbf.py \
        --weights "runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt" \
                  "runs/detect/runs/crack/yolov8s_p2_nwd/weights/best.pt" \
                  "runs/detect/runs/crack/yolov8s_p2_ema_impdiou_nwd/weights/best.pt" \
        --data_root ./dataset \
        --mode submit \
        --out_path results_ensemble.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import load_image_gray, infer_one_image, bbox_to_rle, iou_xyxy


def wbf_fuse(boxes_list, scores_list, iou_thr=0.55, skip_box_thr=0.0001):
    """
    WBF 融合。使用 ensemble_boxes 库。

    Args:
        boxes_list: list of (N_i, 4) arrays, 坐标已归一化到 [0,1]
        scores_list: list of (N_i,) arrays
        iou_thr: WBF 的 IoU 阈值（越大越倾向于融合）
        skip_box_thr: 跳过置信度低于此值的框
    Returns:
        fused_boxes: (M, 4)
        fused_scores: (M,)
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("[ERROR] 请安装 ensemble-boxes: pip install ensemble-boxes")
        raise

    labels_list = [[0] * len(s) for s in scores_list]

    # WBF 要求坐标在 [0,1] 范围内
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=None,  # 等权重
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return boxes, scores


def ensemble_predict(models, img_gray, conf=0.001, iou=0.6):
    """
    用多个模型分别推理，然后 WBF 融合。

    Returns:
        (N, 6) ndarray = [x1, y1, x2, y2, score, cls]
    """
    h, w = img_gray.shape[:2]
    boxes_list = []
    scores_list = []

    for model in models:
        dets = infer_one_image(model, img_gray, conf=conf, iou=iou)
        if len(dets) == 0:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.zeros(0))
            continue

        # 归一化到 [0,1]
        boxes_norm = dets[:, :4].copy()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        boxes_norm = np.clip(boxes_norm, 0, 1)

        boxes_list.append(boxes_norm.tolist())
        scores_list.append(dets[:, 4].tolist())

    # WBF 融合
    fused_boxes, fused_scores = wbf_fuse(
        boxes_list, scores_list,
        iou_thr=0.55,
        skip_box_thr=0.0001,
    )

    if len(fused_boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 反归一化
    result = np.zeros((len(fused_boxes), 6), dtype=np.float32)
    result[:, 0] = fused_boxes[:, 0] * w
    result[:, 1] = fused_boxes[:, 1] * h
    result[:, 2] = fused_boxes[:, 2] * w
    result[:, 3] = fused_boxes[:, 3] * h
    result[:, 4] = fused_scores
    result[:, 5] = 0  # class

    return result


def match_preds_to_gts(preds, gts, iou_thr=0.5):
    preds_sorted = sorted(preds, key=lambda p: -p["score"])
    gt_matched = [False] * len(gts)
    tp_flags = []
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if gt_matched[j]: continue
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


def evaluate_ensemble(models, data_root, split_info, conf=0.001):
    """对验证集跑集成评估。"""
    split = json.loads(Path(split_info).read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])

    with open(data_root / "trainval" / "trainval.json", "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]
    val_samples = [s for s in ds if s["ID"] in val_ids]
    img_dir = data_root / "trainval" / "images"
    print(f"验证集: {len(val_samples)} 张, 模型数: {len(models)}")

    all_records = []
    for s in tqdm(val_samples, desc="Ensemble 推理"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists(): continue
        gray, _ = load_image_gray(img_path)
        if gray is None: continue

        dets = ensemble_predict(models, gray, conf=conf)

        preds = [{"xyxy": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                  "score": float(d[4])} for d in dets]
        gts = []
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            gts.append({"xyxy": [x, y, x + bw, y + bh], "area": bw * bh})
        all_records.append({"preds": preds, "gts": gts})

    # mAP50
    all_tp, total_gts = [], 0
    for r in all_records:
        total_gts += len(r["gts"])
        tp_flags, _ = match_preds_to_gts(r["preds"], r["gts"], 0.5)
        all_tp.extend(tp_flags)
    ap = compute_ap50(all_tp, total_gts)

    # small-Recall
    tp_small, total_small = 0, 0
    for r in all_records:
        gts_small = [g for g in r["gts"] if g["area"] <= 50]
        total_small += len(gts_small)
        _, gt_matched = match_preds_to_gts(r["preds"], gts_small, 0.5)
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
                v = iou_xyxy(p["xyxy"], g["xyxy"])
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0: used[best_i] = True
            large_ious.append(best_iou)
    large_miou = float(np.mean(large_ious)) if large_ious else 0.0

    sim = (min(small_recall / 0.85, 1.0) * 10 +
           min(large_miou / 0.80, 1.0) * 10 +
           min(ap / 0.65, 1.0) * 10)

    print("\n" + "=" * 50)
    print("WBF 集成评估结果")
    print("=" * 50)
    print(f"mAP50          : {ap:.4f}")
    print(f"small-Recall   : {small_recall:.4f}   [{tp_small}/{total_small}]")
    print(f"large-mIoU     : {large_miou:.4f}   [n={len(large_ious)}]")
    print(f"模拟得分       : {sim:.2f} / 30")


def submit_ensemble(models, data_root, out_path, conf=0.001):
    """生成测试集提交文件。"""
    test_json = data_root / "test" / "test.json"
    test_img_dir = data_root / "test" / "images"

    with open(test_json, "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]

    results = []
    times = []
    for sample in tqdm(ds, desc="Ensemble 提交"):
        img_path = test_img_dir / Path(sample["Image"]).name
        if not img_path.exists():
            img_path = data_root / "test" / sample["Image"]

        gray, _ = load_image_gray(img_path)
        if gray is None:
            results.append({"ID": sample["ID"], "image path": sample["Image"],
                            "inference_time_ms": 0.0,
                            "groundtruth_bboxes": [], "predict_bboxes": []})
            continue

        h, w = gray.shape[:2]
        t0 = time.perf_counter()
        dets = ensemble_predict(models, gray, conf=conf)
        inf_ms = (time.perf_counter() - t0) * 1000
        times.append(inf_ms)

        predict_bboxes = []
        for d in dets:
            x1, y1, x2, y2, score = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(w), x2); y2 = min(float(h), y2)
            if x2 <= x1 or y2 <= y1: continue
            predict_bboxes.append({
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "score": score, "label": "crack",
                "mask": bbox_to_rle((x1, y1, x2, y2), h, w),
            })

        results.append({"ID": sample["ID"], "image path": sample["Image"],
                        "inference_time_ms": round(inf_ms, 2),
                        "groundtruth_bboxes": [], "predict_bboxes": predict_bboxes})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\n✓ 已写出 {out_path}，共 {len(results)} 条记录")
    if times:
        print(f"[时延] mean={np.mean(times):.1f}ms, max={max(times):.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="多个模型权重路径（空格分隔）")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--split_info", type=str, default="yolo_dataset/split_info.json")
    ap.add_argument("--mode", choices=["eval", "submit", "both"], default="both")
    ap.add_argument("--out_path", type=str, default="results_ensemble.json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"加载 {len(args.weights)} 个模型...")
    models = []
    for w in args.weights:
        print(f"  → {w}")
        m = YOLO(w)
        if args.device != "cpu":
            pass  # ultralytics 自动使用 GPU
        models.append(m)

    data_root = Path(args.data_root)

    if args.mode in ("eval", "both"):
        evaluate_ensemble(models, data_root, args.split_info, conf=args.conf)

    if args.mode in ("submit", "both"):
        submit_ensemble(models, data_root, args.out_path, conf=args.conf)


if __name__ == "__main__":
    main()