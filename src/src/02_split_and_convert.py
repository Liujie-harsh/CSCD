"""
02_split_and_convert.py — 分层划分 train/val + 转 YOLO 格式

关键：bbox 按 COCO xywh 解析（文档写错了，实测 xywh）。
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def image_bucket(max_dim: int) -> str:
    if max_dim <= 300: return "small"
    elif max_dim <= 1280: return "medium"
    else: return "large"


def stratified_split(items, val_ratio, seed):
    by_bucket = defaultdict(list)
    for b, p in items:
        by_bucket[b].append(p)
    rng = random.Random(seed)
    train, val = [], []
    for bucket, lst in by_bucket.items():
        lst = lst.copy()
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_ratio))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
        print(f"  [{bucket}] 总数 {len(lst)}, 训练 {len(lst)-n_val}, 验证 {n_val}")
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def convert_sample_to_yolo(sample, src_img_dir, dst_img_dir, dst_lbl_dir):
    src_img = src_img_dir / Path(sample["Image"]).name
    if not src_img.exists():
        src_img = src_img_dir.parent / sample["Image"]
    if not src_img.exists():
        return False
    img = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    H, W = img.shape[:2]
    dst_img = dst_img_dir / f"{sample['ID']}{src_img.suffix.lower()}"
    shutil.copy2(src_img, dst_img)
    dst_lbl = dst_lbl_dir / f"{sample['ID']}.txt"
    lines = []
    for a in sample["Annotations"]:
        x, y, w, h = a["bbox"]
        x = max(0.0, x); y = max(0.0, y)
        w = max(0.0, min(w, W - x)); h = max(0.0, min(h, H - y))
        if w < 1 or h < 1: continue
        cx = (x + w/2) / W; cy = (y + h/2) / H
        nw = w / W; nh = h / H
        cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 1e-6), 1.0); nh = min(max(nh, 1e-6), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    dst_lbl.write_text("\n".join(lines), encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root); out_dir = Path(args.out_dir)
    src_json = data_root / "trainval" / "trainval.json"
    src_img_dir = data_root / "trainval" / "images"
    with open(src_json, "r", encoding="utf-8") as f:
        ds = json.load(f)["Dataset"]

    items = []
    for s in ds:
        if not s["Annotations"]:
            img_path = src_img_dir / Path(s["Image"]).name
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                H, W = img.shape[:2]
            else: continue
        else:
            H, W = s["Annotations"][0]["segmentation"]["size"]
        items.append((image_bucket(max(H, W)), s))

    print(f"总样本: {len(items)}，开始分层划分...")
    train, val = stratified_split(items, args.val_ratio, args.seed)
    print(f"训练集: {len(train)}，验证集: {len(val)}")

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("转换训练集...")
    for s in tqdm(train):
        convert_sample_to_yolo(s, src_img_dir, out_dir/"images"/"train", out_dir/"labels"/"train")
    print("转换验证集...")
    for s in tqdm(val):
        convert_sample_to_yolo(s, src_img_dir, out_dir/"images"/"val", out_dir/"labels"/"val")

    yaml_content = f"""# auto-generated
path: {out_dir.resolve()}
train: images/train
val: images/val
names:
  0: crack
"""
    (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n✓ YOLO 数据集已生成: {out_dir}")

    (out_dir / "split_info.json").write_text(
        json.dumps({
            "train_ids": [s["ID"] for s in train],
            "val_ids": [s["ID"] for s in val],
            "seed": args.seed,
            "val_ratio": args.val_ratio,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
