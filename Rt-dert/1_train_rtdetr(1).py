"""
10_train_rtdetr.py — RT-DETR (CVPR 2024) 训练

RT-DETR 是百度提出的实时检测 Transformer，ultralytics 已内置支持。
与 YOLO 相比的优势：
    1. Transformer 编码器对全局上下文建模更强，跨尺度目标天然友好
    2. NMS-free 端到端检测，减少后处理对小目标的误删
    3. COCO 上 mAP 比 YOLOv8 高 0.5-1.0 个点

本脚本训练 RT-DETR-l（large 版本），然后可以加入 WBF 集成。

用法：
    python src/10_train_rtdetr.py --data yolo_dataset/data.yaml --device 0

    # 如果显存不够（rtdetr-l 比 yolov8s 大），用 batch=2
    python src/10_train_rtdetr.py --data yolo_dataset/data.yaml --device 0 --batch 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="rtdetr-l.pt",
                    help="RT-DETR 预训练权重。可选: rtdetr-l.pt, rtdetr-x.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4,
                    help="RT-DETR-l 显存占用较大，8GB 显存建议 batch=2~4")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/crack")
    ap.add_argument("--name", type=str, default="rtdetr_l")
    ap.add_argument("--resume", action="store_true")
    # 是否注入 NWD loss（可选）
    ap.add_argument("--use-nwd", action="store_true", default=False,
                    help="是否注入 NWD loss（RT-DETR 的 loss 机制与 YOLO 不同，谨慎使用）")
    args = ap.parse_args()

    # NWD patch（可选，RT-DETR 的 loss 和 YOLO 不同，不一定适用）
    if args.use_nwd:
        try:
            from patch_ultralytics import apply_all_patches
            apply_all_patches()
            print("[INFO] NWD patch 已注入（注意：RT-DETR 可能不使用 BboxLoss）")
        except Exception as e:
            print(f"[WARN] NWD patch 失败: {e}，使用默认 loss")

    from ultralytics import RTDETR

    print(f"\n==> 加载 RT-DETR 模型: {args.model}")
    model = RTDETR(args.model)

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # RT-DETR 推荐用 AdamW 优化器
        optimizer="AdamW",
        lr0=0.0001,        # RT-DETR 推荐较小学习率
        lrf=0.01,
        weight_decay=0.0001,
        warmup_epochs=3.0,

        # 数据增强（与 YOLO 实验保持一致）
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=10.0, translate=0.1,
        scale=0.5,
        flipud=0.5, fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,     # RT-DETR 对 copy_paste 敏感，不要太高
        close_mosaic=20,

        # 验证 & 保存
        val=True,
        save=True,
        save_period=25,
        patience=50,
        plots=True,

        # 推理相关
        conf=0.001,
        iou=0.6,
        max_det=300,

        # 性能
        workers=8,
        amp=True,
        cache=False,
        single_cls=True,
    )

    print("\n==== RT-DETR 训练配置 ====")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")
    print("==========================\n")

    model.train(**train_args)
    print("\n✓ RT-DETR 训练完成")

    metrics = model.val()
    print(f"\n[最终验证] mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

    print("\n==== 下一步 ====")
    print("1. 评估:")
    print(f'   python src/05_evaluate.py --weights "runs/detect/{args.project}/{args.name}/weights/best.pt" --data_root ./dataset --split_info yolo_dataset/split_info.json')
    print("2. Conf 扫描:")
    print(f'   python src/08_sweep_conf.py --weights "runs/detect/{args.project}/{args.name}/weights/best.pt" --data_root ./dataset --split_info yolo_dataset/split_info.json')
    print("3. 加入 WBF 集成（4 模型）:")
    print(f'   python src/09_ensemble_wbf.py --weights \\')
    print(f'       "runs/detect/runs/crack/yolov8n_p2_nwd/weights/best.pt" \\')
    print(f'       "runs/detect/runs/crack/yolov8s_p2_nwd/weights/best.pt" \\')
    print(f'       "runs/detect/runs/crack/yolov8s_p2_ema_impdiou_nwd/weights/best.pt" \\')
    print(f'       "runs/detect/{args.project}/{args.name}/weights/best.pt" \\')
    print(f'       --data_root ./dataset --split_info yolo_dataset/split_info.json --mode both')


if __name__ == "__main__":
    main()