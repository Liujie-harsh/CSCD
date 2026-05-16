"""
07_export_onnx.py — 导出 ONNX（可选，CPU 部署用）

用法：
    python src/07_export_onnx.py --weights PATH
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--dynamic", action="store_true", default=True)
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"==> 导出 ONNX ...")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        half=args.half,
    )
    print(f"✓ ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
