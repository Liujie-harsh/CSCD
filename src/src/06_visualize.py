"""
06_visualize.py — 可视化 GT / 预测

用法：
    # 可视化 GT
    python src/06_visualize.py --mode gt --data_root ./dataset --n 20 --out_dir outputs/vis_gt

    # 可视化预测
    python src/06_visualize.py --mode pred --weights PATH --data_root ./dataset \
        --split_info yolo_dataset/split_info.json --n 20 --out_dir outputs/vis_pred
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import infer_one_image, load_image_gray


def draw_bbox(img, xyxy, text="", color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        font_scale = max(0.3, min(1.2, max(img.shape[:2]) / 1000))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, text, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def resize_for_display(img, max_side=1600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side: return img, 1.0
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s))), s


def visualize_gt(data_root, out_dir, n, seed=42):
    with open(data_root / "trainval" / "trainval.json", "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]
    img_dir = data_root / "trainval" / "images"
    rng = random.Random(seed); rng.shuffle(ds)
    sampled = ds[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in tqdm(sampled, desc="GT visualization"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists(): continue
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            area = bw * bh
            tag = "tiny" if area <= 50 else ("LARGE" if area >= 300*300 else "")
            color = (0, 255, 255) if tag == "tiny" else \
                    (0, 0, 255) if tag == "LARGE" else (0, 255, 0)
            draw_bbox(img, (x, y, x + bw, y + bh), tag, color)
        disp, _ = resize_for_display(img)
        cv2.imwrite(str(out_dir / f"ID{s['ID']:04d}_{Path(s['Image']).stem}.jpg"), disp)
    print(f"✓ GT 可视化 → {out_dir}")


def visualize_pred(data_root, weights, split_info, out_dir, n, device, conf, seed=42):
    from ultralytics import YOLO
    model = YOLO(weights)
    if device == "cpu": model.to("cpu")
    split = json.loads(Path(split_info).read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])
    with open(data_root / "trainval" / "trainval.json", "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]
    val_samples = [s for s in ds if s["ID"] in val_ids]
    rng = random.Random(seed); rng.shuffle(val_samples)
    sampled = val_samples[:n]
    img_dir = data_root / "trainval" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in tqdm(sampled, desc="Pred visualization"):
        img_path = img_dir / Path(s["Image"]).name
        if not img_path.exists(): continue
        gray, bgr = load_image_gray(img_path)
        if gray is None: continue
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if bgr.ndim == 2 else bgr[:, :, :3].copy()
        for a in s["Annotations"]:
            x, y, bw, bh = a["bbox"]
            draw_bbox(canvas, (x, y, x+bw, y+bh), "GT", (0, 255, 0), 2)
        dets = infer_one_image(model, gray, conf, 0.5)
        for d in dets:
            x1, y1, x2, y2, score, _ = d
            draw_bbox(canvas, (x1, y1, x2, y2), f"{score:.2f}", (0, 0, 255), 2)
        disp, _ = resize_for_display(canvas)
        cv2.imwrite(str(out_dir / f"ID{s['ID']:04d}_{Path(s['Image']).stem}.jpg"), disp)
    print(f"✓ 预测可视化 → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "pred"], required=True)
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split_info", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()
    if args.mode == "gt":
        visualize_gt(Path(args.data_root), Path(args.out_dir), args.n, args.seed)
    else:
        assert args.weights and args.split_info
        visualize_pred(Path(args.data_root), args.weights, args.split_info,
                       Path(args.out_dir), args.n, args.device, args.conf, args.seed)


if __name__ == "__main__":
    main()
