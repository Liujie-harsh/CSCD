"""
04_inference.py — 测试集推理 + 生成 results.json

用法：
    python src/04_inference.py \
        --weights runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt \
        --data_root ./dataset \
        --out_path results_final.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import infer_one_image, load_image_gray, bbox_to_rle, NORMAL_IMGSZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--out_path", type=str, default="results_final.json")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="P2+NWD 模型对小目标置信度偏低，建议 0.15")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    if args.device == "cpu":
        model.to("cpu")

    # 预热
    dummy = np.zeros((NORMAL_IMGSZ, NORMAL_IMGSZ, 3), dtype=np.uint8)
    _ = model.predict(dummy, imgsz=NORMAL_IMGSZ, verbose=False, device=model.device)

    data_root = Path(args.data_root)
    test_json = data_root / "test" / "test.json"
    test_img_dir = data_root / "test" / "images"
    assert test_json.exists(), f"Not found: {test_json}"

    with open(test_json, "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]

    results = []
    times = []
    for sample in tqdm(ds, desc="Inference"):
        img_path = test_img_dir / Path(sample["Image"]).name
        if not img_path.exists():
            img_path = data_root / "test" / sample["Image"]

        gray, _ = load_image_gray(img_path)
        if gray is None:
            print(f"[warn] 无法读取 {img_path}")
            results.append({
                "ID": sample["ID"],
                "image path": sample["Image"],
                "inference_time_ms": 0.0,
                "groundtruth_bboxes": [],
                "predict_bboxes": [],
            })
            continue

        h, w = gray.shape[:2]
        t0 = time.perf_counter()
        dets = infer_one_image(model, gray, args.conf, args.iou)
        inf_ms = (time.perf_counter() - t0) * 1000
        times.append(inf_ms)

        predict_bboxes = []
        for d in dets:
            x1, y1, x2, y2, score, _cls = d
            x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
            x2 = min(float(w), float(x2)); y2 = min(float(h), float(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            predict_bboxes.append({
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "score": float(score),
                "label": "crack",
                "mask": bbox_to_rle((x1, y1, x2, y2), h, w),
            })

        results.append({
            "ID": sample["ID"],
            "image path": sample["Image"],
            "inference_time_ms": round(inf_ms, 2),
            "groundtruth_bboxes": [],
            "predict_bboxes": predict_bboxes,
        })

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\n✓ 已写出 {args.out_path}，共 {len(results)} 条记录")

    if times:
        print(f"[时延] mean={np.mean(times):.1f}ms, "
              f"p50={np.percentile(times,50):.1f}ms, "
              f"p95={np.percentile(times,95):.1f}ms, "
              f"max={max(times):.1f}ms")
        n_fast = sum(1 for t in times if t < 100)
        print(f"[合格率] <100ms: {n_fast}/{len(times)} ({100*n_fast/len(times):.1f}%)")


if __name__ == "__main__":
    main()
