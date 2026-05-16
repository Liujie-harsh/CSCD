"""
03_train.py — YOLOv8s-P2-EMA + Inner-MPDIoU + NWD

三重改进：
    1. P2 检测头（stride=4）
    2. EMA 注意力（Neck 每个 C2f 后）
    3. Inner-MPDIoU + NWD 混合 loss

用法：
    python src/03_train.py --data yolo_dataset/data.yaml --device 0

消融实验：
    # 仅 P2 + EMA（无 NWD/Inner-MPDIoU）
    python src/03_train.py --data yolo_dataset/data.yaml --no-nwd --name ablation_no_loss

    # P2 + NWD（无 EMA，用旧 yaml）
    python src/03_train.py --data yolo_dataset/data.yaml --cfg configs/yolov8s-p2.yaml --name ablation_no_ema
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--cfg", type=str, default="configs/yolov8s-p2-ema.yaml")
    ap.add_argument("--pretrained", type=str, default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/crack")
    ap.add_argument("--name", type=str, default="yolov8s_p2_ema_impdiou_nwd")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-nwd", action="store_true")
    ap.add_argument("--impdiou-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-c", type=float, default=12.8)
    ap.add_argument("--inner-ratio", type=float, default=0.7)
    args = ap.parse_args()

    # ========== 第 1 步：注入所有 patch ==========
    if not args.no_nwd:
        from patch_ultralytics import apply_all_patches
        apply_all_patches(
            inner_mpdiou_ratio=args.impdiou_ratio,
            nwd_ratio=args.nwd_ratio,
            nwd_c=args.nwd_c,
            inner_ratio=args.inner_ratio,
        )
    else:
        # 仍然注册 EMA 模块（yaml 里用了）
        from patch_ultralytics import _register_ema_module
        _register_ema_module()
        print("[INFO] --no-nwd：使用默认 CIoU loss（消融对比）")

    # ========== 第 2 步：注册 EMA 到 ultralytics 解析系统 ==========
    # 确保 yaml 里的 EMA 能被识别
    try:
        from ema_attention import EMA
        import ultralytics.nn.modules as modules

        # Register EMA in multiple ways for maximum compatibility
        modules.EMA = EMA

        if hasattr(modules, '__all__') and 'EMA' not in modules.__all__:
            modules.__all__ = list(modules.__all__) + ['EMA']

        # Register to tasks module
        try:
            import ultralytics.nn.tasks as tasks
            tasks.EMA = EMA
        except Exception as e:
            print(f"[WARN] Could not register EMA to tasks: {e}")

        # Register to models.common
        try:
            import ultralytics.models.common as common
            common.EMA = EMA
        except Exception as e:
            print(f"[WARN] Could not register EMA to models.common: {e}")

        print("[INFO] EMA module registered successfully")
    except Exception as e:
        print(f"[ERROR] Failed to register EMA module: {e}")
        raise

    # ========== 第 3 步：验证配置并加载模型 ==========
    from ultralytics import YOLO

    print(f"\n==> 验证配置: {args.cfg}")
    try:
        # First try to validate the config can be parsed
        import yaml
        with open(args.cfg, 'r') as f:
            cfg_data = yaml.safe_load(f)
        print(f"[INFO] Configuration loaded successfully")
        print(f"[INFO] Number of classes: {cfg_data.get('nc', 'not specified')}")
        print(f"[INFO] Scales: {cfg_data.get('scales', 'not specified')}")

        print(f"\n==> 构建架构: {args.cfg}")
        model = YOLO(args.cfg)
        print("[INFO] Model architecture built successfully")
    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        print("[ERROR] This might be due to EMA module not being properly registered")
        raise

    try:
        print(f"==> 迁移权重: {args.pretrained}")
        model.load(args.pretrained)
        print("✓ 权重迁移完成（EMA 等新层从头初始化）")
    except Exception as e:
        print(f"[warn] 权重迁移失败 ({e})，从头训练")

    # ========== 第 4 步：训练 ==========
    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        optimizer="SGD",
        lr0=0.01, lrf=0.01,
        momentum=0.937, weight_decay=5e-4,
        warmup_epochs=3.0,

        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=10.0, translate=0.1,
        scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.15,
        copy_paste=0.5,
        close_mosaic=20,

        box=7.5, cls=0.5, dfl=1.5,

        val=True, save=True,
        save_period=25, patience=50,
        plots=True,

        conf=0.001, iou=0.6, max_det=300,
        workers=8, amp=True,
        cache=False, rect=False,
        overlap_mask=True, single_cls=True,
    )

    print("\n==== 训练配置 ====")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")
    if not args.no_nwd:
        print(f"  [Loss] Inner-MPDIoU({args.impdiou_ratio}) + NWD({args.nwd_ratio})")
        print(f"  [Loss] inner_ratio={args.inner_ratio}, C={args.nwd_c}")
    print(f"  [Arch] EMA attention in Neck")
    print("==================\n")

    model.train(**train_args)
    print("\n✓ 训练完成")

    metrics = model.val()
    print(f"\n[最终验证] mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
