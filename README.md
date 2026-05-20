# 跨尺度芯片裂纹检测 · 实验框架

## 项目结构

```
cscd/
config/  
    └── yolov8n-p2.yaml
    └── yolov8s-p2-ema.yaml
    └── yolov8s-p2.yaml
dataset/                              ← 已有数据
    ├── test/
    │   ├── images/                       (jpg/png/bmp 混合)
    │   └── test.json
    ├── trainval/
    │   ├── images/
    │   └── trainval.json
    ├── 仅供格式参考results.json
    └── 数据说明与提交指南.docx
Rt-dert/
    └── 1_train_rtdetr.py
    └── 2_train_rtdetr_p2.py
    └── conf_sweep_p2_nwd.json
    └── rtdetr-l-p2.yaml
src/
    src/
        └── 01_analyze.py
        └── 02_split_and_convert.py
        └── 03_train.py
        └── 04_inference.py
        └── 05_evaluate.py
        └── 06_visualize.py
        └── 07_export_onnx.py
        └── 08_sweep_conf.py
        └── 09_ensemble_wbf.py
        └── common.py
        └── ema_attention.py
        └── inner_mpdiou_loss.py
        └── nwd_loss.py
        └── patch_ultralytics.py
    weights/
    └── yolov8n_p2_nwd_best.onnx
    └── yolov8n_p2_nwd_best.pt
    └── yolov8s_p2_ema_best.onnx
    └── yolov8s_p2_ema_best.pt
    └── yolov8s_p2_nwd_best.onnx
    └── yolov8s_p2_nwd_best.pt
├── requirements.txt
├── README.md
└── .gitignore
```

运行产物将自动生成在项目根目录：
- `yolo_dataset/`  (02 脚本生成的 YOLO 格式数据)
- `runs/`          (03 脚本的训练日志与权重)
- `outputs/`       (01/06 脚本的分析图与可视化)
- `results.json`   (04 脚本的最终提交文件)

---



赛题本身要求处理 **59×46 到 7465×9263** 的灰度芯片图像，以及 **1×3 到 3947×3703** 的裂纹缺陷，重点考核极小缺陷召回、极大缺陷定位、mAP50 和推理速度。

| 脚本                        | 作用                                                         | 在项目中的位置                                               |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **01_analyze.py**           | 数据分布探查：统计图像尺寸、bbox 面积、极小/极大缺陷比例、每图缺陷数，并输出 `analysis.png`。 | 项目第一步，用来证明“跨尺度问题”有多严重，也为报告和策略设计提供依据。 |
| **02_split_and_convert.py** | 按图像尺度分层划分 train/val，并把原始标注转成 YOLO 格式。   | 训练前的数据准备脚本，生成 `images/train`、`images/val`、`labels/train`、`labels/val`、`data.yaml`、`split_info.json`。 |
| **03_train.py**             | 训练主脚本：YOLOv8s-P2-EMA + Inner-MPDIoU + NWD。            | 项目核心训练入口。P2 头针对小裂纹，EMA 增强特征，NWD/Inner-MPDIoU 改善小目标和定位。 |
| **04_inference.py**         | 测试集推理并生成 `results_final.json`。                      | 最终提交文件生成脚本。会读取 test.json，对测试图像推理，输出 bbox、score、label 和 mask/RLE。 |
| **05_evaluate.py**          | 本地验证集评估。                                             | 对齐赛题指标：mAP50、small-Recall、large-mIoU、推理时延，并给出模拟得分。 |
| **06_visualize.py**         | 可视化 GT 或预测结果。                                       | 用于检查标注质量、模型漏检/误检，输出带框图片，适合写报告和做 bad case 分析。 |
| **07_export_onnx.py**       | 导出 ONNX。                                                  | 可选部署脚本，用于 CPU/ONNXRuntime/TensorRT 等推理部署准备。 |
| **08_sweep_conf.py**        | 扫描不同 conf 阈值。                                         | 不重新训练，只在验证集上找最优置信度阈值，平衡小目标召回、mAP50 和大目标 IoU。 |
| **09_ensemble_wbf.py**      | 多模型 WBF 加权框融合。                                      | 用多个模型集成提升 mAP 和定位精度，可用于验证集评估或生成最终提交。 |



**RT-DETR**

### 1. `1_train_rtdetr(1).py`

这是**普通 RT-DETR-l 训练脚本**。

作用是直接加载 `rtdetr-l.pt` 预训练模型，在你的裂纹数据集上训练。它没有改模型结构，主要作为 RT-DETR baseline。它支持 AdamW、小学习率、mosaic、mixup、copy_paste 等训练配置，也预留了 `--use-nwd`，但文件里也提醒 RT-DETR 的 loss 机制和 YOLO 不同，NWD patch 不一定真正适用。

适合作为：
**YOLO baseline vs 原版 RT-DETR baseline** 的对比实验。

------

### 2. `rtdetr-l-p2.yaml`

这是**改造后的 RT-DETR-P2 模型结构配置文件**。

核心改动：

- 加入 **P2 特征层**，stride=4；
- 原本 RT-DETR 主要用 P3/P4/P5，现在改成 **P2/P3/P4/P5 四尺度 Decoder**；
- P2 用来提升极小裂纹检测能力；
- P5 + AIFI Transformer 用来保留大目标和全局上下文建模能力。

简单说：
这个文件是你“换 RT-DETR 并针对小目标改造”的核心创新点。

------

### 3. `2_train_rtdetr_p2.py`

这是**RT-DETR-P2 训练脚本**，专门配合 `rtdetr-l-p2.yaml` 使用。

它做的事情是：

- 自动查找 `data.yaml`；
- 用 `RTDETR(args.cfg)` 构建 P2 版 RT-DETR；
- 加载 `rtdetr-l.pt` 做预训练迁移；
- 用 AdamW、imgsz=960、batch=4、epochs=200 训练；
- 训练结束后跑一次 `model.val()`；
- 最后提示下一步做 conf 扫描和生成提交文件。

这个脚本是你最终想跑的主要版本。

## 环境

```bash
pip install -r requirements.txt
```

---

## ⚠️ 两个必读的数据坑

### 1. bbox 格式与文档不符
`数据说明与提交指南.docx` 声称 bbox 为 `[x1, y1, x2, y2]`，但实际 `trainval.json` 的 bbox 是 COCO `[x, y, w, h]` 格式（已通过解码 RLE mask 反推验证，一致率 100%）。

**本框架已在 `02_split_and_convert.py` 中正确处理为 xywh 解析。** 如果你自己改脚本，务必注意这一点——否则所有训练标签都会错位。

### 2. 输出格式用的是 x1y1x2y2
虽然输入是 xywh，但 `results.json` 要求的字段就是 `x1/y1/x2/y2`。`04_inference.py` 会自动处理这一转换。

---

## 运行顺序

```bash
# 在项目根目录执行，均使用 ./dataset 作为默认数据路径

# 1. 分布探查（可选，生成 outputs/analysis/analysis.png）
python src/01_analyze.py --data_root ./dataset

# 2. 数据转换（生成 yolo_dataset/ 及 split_info.json）
python src/02_split_and_convert.py --data_root ./dataset --out_dir yolo_dataset --val_ratio 0.2

# 2.5 [建议] 先可视化 GT 确认标注转换正确
python src/06_visualize.py --mode gt --data_root ./dataset --n 20 --out_dir outputs/vis_gt

# 3. 训练（GPU，约 4-8 小时）
python src/03_train.py --data yolo_dataset/data.yaml --epochs 200 --imgsz 640 --batch 16 --device 0

# 4. 在测试集上生成 results.json
python src/04_inference.py --weights runs/crack/yolov8n_baseline/weights/best.pt --data_root ./dataset --out_path results.json

# 5. 本地验证集评估（对齐赛题评分）
python src/05_evaluate.py --weights runs/crack/yolov8n_baseline/weights/best.pt --data_root ./dataset --split_info yolo_dataset/split_info.json

# 6. [可选] 可视化预测结果（用于调参 / 答辩演示）
python src/06_visualize.py --mode pred --weights runs/crack/yolov8n_baseline/weights/best.pt --data_root ./dataset --split_info yolo_dataset/split_info.json --n 20 --out_dir outputs/vis_pred

# 7. [可选] 导出 ONNX 供 CPU 部署
python src/07_export_onnx.py --weights runs/crack/yolov8n_baseline/weights/best.pt
```

---

## AMSIP 推理流水线（`common.py` 中实现）

| 档位 | 触发条件 (max edge) | 处理策略 | 覆盖数据占比 |
|------|---------------------|----------|-----|
| 小图 | ≤ 320 px | 双线性放大至 320×320 后推理 | ~50% |
| 中图 | 320~1280 px | letterbox 到 640×640 直通 | ~45% |
| 大图 | > 1280 px | SAHI 640×640 切片（20% 重叠）+ 空白早退 + 全图 NMS | ~5% |

空白切片早退：灰度标准差 <3.0 的切片视为纯背景，跳过推理。工业图像中大图多为均匀基板，通常可过滤掉 60-80% 切片，使超大图推理时间从 ~5s 降至 ~1.5s。

---

## 评分对齐

| 赛题指标 | 权重 | 本框架监控变量 | 目标 |
|---|---|---|---|
| 极小缺陷召回 | 10% | `small-Recall` (面积 ≤50) | ≥ 0.85 |
| 极大缺陷 mIoU | 10% | `large-mIoU` (≥300×300) | ≥ 0.80 |
| 综合 mAP50 | 10% | `mAP50` | ≥ 0.65 |
| 推理时延 | 10% | `mean / p95 / max` ms | <100ms 常规图 |

`05_evaluate.py` 在验证集上直接计算这四项并输出"模拟性能得分"。

---

## 可能的进一步优化

1. **小目标增强**：如果首版模型 small-Recall < 0.7，尝试：
   - 用 `yolov8n-p2.yaml` 配置（加 P2 检测头，stride=4）
   - 提高 `copy_paste=0.5`
   - 对面积 ≤50 的 GT 所在图像过采样 2-3×

2. **大目标定位**：如果 large-mIoU < 0.7，尝试：
   - 切片重叠率提升至 30%（`SLICE_OVERLAP=0.3` in common.py）
   - 用 SIoU 替换默认 CIoU loss（需改 ultralytics 源码或自定义 trainer）
   - 加入长裂纹连通域级合并（WBF）

3. **CPU 时延**：用 `07_export_onnx.py --to_openvino` 导出 IR，再用 NNCF 做 INT8 PTQ，通常可再提速 2-3×。
