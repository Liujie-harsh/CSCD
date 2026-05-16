"""
common.py — 共享推理逻辑 (v3.0)

AMSIP 自适应推理流水线 + 训推分辨率统一。

更新日志：
    v1.0: 初版，大图使用 SAHI 切片
    v1.1: _to_bgr 增加异常通道布局兜底
    v1.2: 大图从 SAHI 切片改为 letterbox 直接推理
    v2.0: 中图推理分辨率 640 → 960 匹配 P2 训练
    v3.0: 小图档位也改为 960。原因：训练时所有图被 letterbox 到 960 训练，
          推理时小图 letterbox 到 320 会造成训推尺度不一致。
          现在三档统一使用 imgsz=960，只是路由用于决定是否需要特殊处理
          （如超大图降分辨率避免显存爆炸）。

三档策略简化说明：
    - 正常图 (max_dim ≤ 1280): letterbox 到 NORMAL_IMGSZ (960)
    - 大图   (max_dim >  1280): letterbox 到 LARGE_IMGSZ (1280)，略大保留细节
"""
import numpy as np
import cv2

# ======== 推理分辨率 ========
NORMAL_IMGSZ = 960   # 与训练 imgsz 完全一致
LARGE_IMGSZ = 1280   # 大图略大，但不超过太多避免速度下降
MEDIUM_MAX_DIM = 1280  # 分档阈值：≤1280 走正常档，>1280 走大图档


def nms_numpy(boxes, scores, iou_thr=0.5):
    """NumPy 版 NMS。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def _to_bgr(img):
    """把任意布局的 numpy 图像转为 (H, W, 3) uint8 BGR。"""
    if img is None:
        raise ValueError("_to_bgr: input is None")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        h_or_c, w_or_h, last = img.shape
        if last == 1:
            return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
        if last == 3:
            return img
        if last == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if h_or_c in (1, 3, 4):
            return _to_bgr(img.transpose(1, 2, 0))
    raise ValueError(f"_to_bgr: unsupported image shape {img.shape}")


def _yolo_predict(model, img, imgsz, conf, iou):
    """统一的 YOLO 预测包装。返回 (N, 6) = [x1,y1,x2,y2,score,cls]."""
    results = model.predict(img, imgsz=imgsz, conf=conf, iou=iou,
                            verbose=False, device=model.device)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy().reshape(-1, 1)
    labels = r.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, scores, labels], axis=1)


def infer_normal(model, img, conf=0.25, iou=0.5):
    """正常图（max_dim ≤ 1280）：letterbox 到 NORMAL_IMGSZ。"""
    img = _to_bgr(img)
    return _yolo_predict(model, img, NORMAL_IMGSZ, conf, iou)


def infer_large(model, img, conf=0.2, iou=0.5):
    """大图（max_dim > 1280）：letterbox 到 LARGE_IMGSZ。"""
    img = _to_bgr(img)
    return _yolo_predict(model, img, LARGE_IMGSZ, conf, iou)


def infer_one_image(model, img_gray, conf=0.25, iou=0.5):
    """
    AMSIP 主入口：按图像尺寸路由。

    Args:
        model: ultralytics YOLO 实例
        img_gray: 灰度图 (H, W) 或 BGR (H, W, 3)
        conf, iou: 阈值

    Returns:
        (N, 6) ndarray = [x1, y1, x2, y2, score, class_id]
    """
    h, w = img_gray.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MEDIUM_MAX_DIM:
        return infer_normal(model, img_gray, conf, iou)
    else:
        return infer_large(model, img_gray, conf, iou)


def load_image_gray(path):
    """
    兼容读取 jpg/png/bmp，返回 (gray, original)。
    gray 保证是 2D (H, W) uint8 数组。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            gray = img.squeeze(-1)
        elif c == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        return None, None

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray, img


def bbox_to_rle(bbox_xyxy, img_h, img_w):
    """bbox 转 COCO RLE mask。"""
    from pycocotools import mask as maskUtils
    x1, y1, x2, y2 = bbox_xyxy
    m = np.zeros((img_h, img_w), dtype=np.uint8, order="F")
    x1i = max(0, int(round(x1)))
    y1i = max(0, int(round(y1)))
    x2i = min(img_w, int(round(x2)))
    y2i = min(img_h, int(round(y2)))
    if x2i > x1i and y2i > y1i:
        m[y1i:y2i, x1i:x2i] = 1
    rle = maskUtils.encode(np.asfortranarray(m))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def iou_xyxy(a, b):
    """单对 bbox IoU。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)
