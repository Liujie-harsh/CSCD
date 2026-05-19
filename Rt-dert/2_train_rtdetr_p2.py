"""
2_train_rtdetr_p2.py — RT-DETR-l + P2 小目标检测头训练

三大改进：
    1. P2 检测头：stride=4，覆盖 4×4 像素级微小裂纹（原版最小 stride=8）
    2. 4 尺度 RTDETRDecoder：P2/P3/P4/P5 四尺度 Transformer 解码
    3. Transformer 全局注意力 + CNN 局部特征的混合架构

与 YOLO-P2 的区别：
    - YOLO 用 NMS 后处理 → RT-DETR NMS-free 端到端
    - YOLO 只有局部感受野 → RT-DETR 的 AIFI 模块有全局自注意力
    - YOLO large-mIoU=0.807 → RT-DETR large-mIoU=0.919（大目标定位极准）

数据集路径说明：
    本脚本在 RT-DETR 目录下运行，数据集在原项目目录中。
    使用绝对路径或相对路径 ..\\【4】...\\yolo_dataset\\data.yaml

用法：
    cd C:\\Users\\LYC\\Desktop\\4 跨尺度芯片图像的裂纹缺陷智能检测算法设计\\RT-DETR

    # 标准训练（推荐）
    python 2_train_rtdetr_p2.py --device 0

    # 显存不够时
    python 2_train_rtdetr_p2.py --device 0 --batch 2

    # 自定义数据路径
    python 2_train_rtdetr_p2.py --data "C:\\path\\to\\data.yaml" --device 0
"""
import argparse
import sys
import os
from pathlib import Path


def find_data_yaml():
    """自动查找 data.yaml 路径。"""
    candidates = [
        # 相对路径（从 RT-DETR 目录出发）
        Path("..") / "【4】跨尺度芯片图像的裂纹缺陷智能检测算法设计" / "yolo_dataset" / "data.yaml",
        # 绝对路径
        Path(r"C:\Users\LYC\Desktop\4 跨尺度芯片图像的裂纹缺陷智能检测算法设计") 
            / "【4】跨尺度芯片图像的裂纹缺陷智能检测算法设计" / "yolo_dataset" / "data.yaml",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="data.yaml 路径（不指定则自动查找）")
    ap.add_argument("--cfg", type=str, default="rtdetr-l-p2.yaml",
                    help="模型配置文件")
    ap.add_argument("--pretrained", type=str, default="rtdetr-l.pt",
                    help="预训练权重（迁移 backbone）")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/crack")
    ap.add_argument("--name", type=str, default="rtdetr_l_p2")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # 自动查找数据路径
    if args.data is None:
        args.data = find_data_yaml()
        if args.data is None:
            print("[ERROR] 无法自动找到 data.yaml，请手动指定 --data 参数")
            print("示例: python 2_train_rtdetr_p2.py --data \"..\\【4】...\\yolo_dataset\\data.yaml\"")
            sys.exit(1)
    print(f"[INFO] 数据集: {args.data}")

    from ultralytics import RTDETR

    # 构建 P2 版 RT-DETR
    print(f"\n==> 构建 RT-DETR-P2 架构: {args.cfg}")
    model = RTDETR(args.cfg)

    # 迁移预训练权重
    try:
        print(f"==> 迁移 backbone 权重: {args.pretrained}")
        model.load(args.pretrained)
        print("✓ 权重迁移完成（P2 新增层从头初始化）")
    except Exception as e:
        print(f"[warn] 权重迁移失败 ({e})，从头训练")

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # RT-DETR 推荐 AdamW + 小学习率
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_epochs=3.0,

        # 数据增强
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=10.0, translate=0.1,
        scale=0.5,
        flipud=0.5, fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        close_mosaic=20,

        # 验证 & 保存
        val=True, save=True,
        save_period=25, patience=50,
        plots=True,

        conf=0.001, iou=0.6, max_det=300,
        workers=8, amp=True,
        cache=False,
        single_cls=True,
    )

    print("\n==== RT-DETR-P2 训练配置 ====")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")
    print("  [Arch] RT-DETR-l + P2 head (4-scale decoder)")
    print("  [Arch] AIFI Transformer encoder (global attention)")
    print("=============================\n")

    model.train(**train_args)
    print("\n✓ RT-DETR-P2 训练完成")

    metrics = model.val()
    print(f"\n[最终验证] mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

    # 下一步指令
    weight_path = f"runs/detect/{args.project}/{args.name}/weights/best.pt"
    print(f"\n==== 下一步 ====")
    print(f"1. Conf 扫描（在原项目目录运行）:")
    print(f'   cd "..\\【4】跨尺度芯片图像的裂纹缺陷智能检测算法设计"')
    print(f'   python src/08_sweep_conf.py --weights "..\\RT-DETR\\{weight_path}" --data_root ./dataset --split_info yolo_dataset/split_info.json')
    print(f"\n2. 生成提交文件:")
    print(f'   python src/04_inference.py --weights "..\\RT-DETR\\{weight_path}" --data_root ./dataset --out_path results_rtdetr_p2.json --conf 0.001')


if __name__ == "__main__":
    main()
