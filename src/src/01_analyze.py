"""
01_analyze.py — 数据分布探查

用法：
    python src/01_analyze.py --data_root ./dataset
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(data_root: Path, out_dir: Path):
    trainval_json = data_root / "trainval" / "trainval.json"
    test_json = data_root / "test" / "test.json"
    assert trainval_json.exists(), f"Not found: {trainval_json}"

    with open(trainval_json, "r", encoding="utf-8") as f:
        tv = json.load(f)
    tv_ds = tv["Dataset"]
    print(f"[trainval] 总图像数: {len(tv_ds)}")

    sizes = []
    exts = Counter()
    bbox_ws, bbox_hs, bbox_areas = [], [], []
    anns_per_img = []

    for s in tv_ds:
        ext = Path(s["Image"]).suffix.lower().lstrip(".")
        exts[ext] += 1
        anns_per_img.append(len(s["Annotations"]))
        if s["Annotations"]:
            H, W = s["Annotations"][0]["segmentation"]["size"]
            sizes.append((H, W))
            for a in s["Annotations"]:
                x, y, w, h = a["bbox"]
                bbox_ws.append(w)
                bbox_hs.append(h)
                bbox_areas.append(w * h)

    max_dims = [max(h, w) for h, w in sizes]
    buckets = [
        ("≤150", lambda d: d <= 150),
        ("150-300", lambda d: 150 < d <= 300),
        ("300-640", lambda d: 300 < d <= 640),
        ("640-1280", lambda d: 640 < d <= 1280),
        ("1280-3000", lambda d: 1280 < d <= 3000),
        ("3000-5000", lambda d: 3000 < d <= 5000),
        (">5000", lambda d: d > 5000),
    ]
    print("\n[图像最大边长分档]")
    bucket_counts = []
    for name, pred in buckets:
        c = sum(1 for d in max_dims if pred(d))
        bucket_counts.append(c)
        print(f"  {name:>10}: {c:5d} ({100*c/len(max_dims):5.1f}%)")

    print(f"\n[缺陷统计] 总缺陷数: {len(bbox_areas)}")
    tiny_area = sum(1 for a in bbox_areas if a <= 50)
    large_area = sum(1 for a in bbox_areas if a >= 300*300)
    print(f"  面积 ≤ 50 px² (赛题微小): {tiny_area} ({100*tiny_area/len(bbox_areas):.1f}%)")
    print(f"  面积 ≥ 300×300 (赛题极大): {large_area} ({100*large_area/len(bbox_areas):.1f}%)")
    print(f"  缺陷宽度 (min/p50/p95/max): "
          f"{min(bbox_ws):.1f} / {np.percentile(bbox_ws,50):.1f} / "
          f"{np.percentile(bbox_ws,95):.1f} / {max(bbox_ws):.1f}")
    print(f"  缺陷高度 (min/p50/p95/max): "
          f"{min(bbox_hs):.1f} / {np.percentile(bbox_hs,50):.1f} / "
          f"{np.percentile(bbox_hs,95):.1f} / {max(bbox_hs):.1f}")

    print(f"\n[扩展名] {dict(exts)}")
    print(f"\n[每图缺陷数] mean={np.mean(anns_per_img):.2f}, "
          f"p50={int(np.percentile(anns_per_img,50))}, "
          f"p95={int(np.percentile(anns_per_img,95))}, max={max(anns_per_img)}")

    if test_json.exists():
        with open(test_json, "r", encoding="utf-8") as f:
            te = json.load(f)
        print(f"\n[test] 图像数: {len(te['Dataset'])}")
        te_exts = Counter(Path(s["Image"]).suffix.lower().lstrip(".") for s in te["Dataset"])
        print(f"[test 扩展名] {dict(te_exts)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar([b[0] for b in buckets], bucket_counts, color="steelblue")
    axes[0].set_title("Image Max-Dim Bucket")
    axes[0].tick_params(axis="x", rotation=30)

    log_areas = np.log10(np.array(bbox_areas) + 1)
    axes[1].hist(log_areas, bins=40, color="orange")
    axes[1].axvline(np.log10(50), color="r", linestyle="--", label="tiny (50px²)")
    axes[1].axvline(np.log10(300*300), color="g", linestyle="--", label="large (300²)")
    axes[1].set_xlabel("log10(bbox area+1)")
    axes[1].set_title("Defect Area Distribution")
    axes[1].legend()

    axes[2].hist(anns_per_img, bins=range(0, max(anns_per_img)+2), color="green")
    axes[2].set_xlabel("Annotations per image")
    axes[2].set_title("Annotations per Image")
    plt.tight_layout()
    out_path = out_dir / "analysis.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n✓ 图表保存至 {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/analysis")
    args = ap.parse_args()
    analyze(Path(args.data_root), Path(args.out_dir))
