"""标注贴合度检查工具 — 经典CV前景提取 + 贴合度指标

混合分割方案：
- 小目标（max(w,h)<30）：ROI 超分 + 多候选阈值前景提取 + 形态学 + 连通域筛选
- 指标：fg_area/area_ratio/bbox_iou/mask_iou/centroid_offset/overflow/deficit/boundary_gradient
纯数据计算，不依赖界面状态。
"""

import cv2
import numpy as np
from dataclasses import dataclass

# 混合分割阈值：max(w,h) 小于该值走经典 CV 管道，否则走交互式分割模型
SMALL_BOX_THRESHOLD = 30


@dataclass
class ExtractResult:
    detected: bool = False
    mask: np.ndarray = None           # patch(超分) 坐标系二值掩码
    bbox_orig: tuple = (0, 0, 0, 0)   # 未放大前 ROI 坐标系 (x, y, w, h)，由调用方加 ROI 偏移得原图坐标
    centroid_orig: tuple = (0.0, 0.0) # 未放大前 ROI 坐标系
    area_orig: float = 0.0            # 未放大前像素面积
    area_ratio: float = 0.0           # 前景面积 / 标注框面积
    method: str = ""
    polarity: str = ""


def choose_scale(box_w, box_h, target_min=64, max_scale=8, min_scale=1):
    """根据标注框尺寸选择放大倍数：放大后最小边接近 target_min 像素。"""
    min_side = min(float(box_w), float(box_h))
    if min_side <= 0:
        return min_scale
    scale = int(np.ceil(target_min / min_side))
    return max(min_scale, min(scale, max_scale))


def upscale_patch(patch, scale, interpolation=cv2.INTER_LANCZOS4, sharpen=True, amount=0.6):
    """经典插值超分：Lanczos 放大 + 反锐化掩模增强边缘。"""
    if scale <= 1:
        return patch.copy()
    up = cv2.resize(patch, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    if not sharpen:
        return up
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=max(1.0, scale))
    sharp = cv2.addWeighted(up, 1.0 + amount, blur, -amount, 0)
    return sharp


def crop_roi(img, box, margin_ratio=1.0):
    """按标注框外扩裁剪 ROI。返回 (roi, 左上角原图坐标)；越界返回 (None, None)。"""
    H, W = img.shape[:2]
    x, y, w, h = box
    mw = int(round(w * margin_ratio))
    mh = int(round(h * margin_ratio))
    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(W, x + w + mw)
    y2 = min(H, y + h + mh)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img[y1:y2, x1:x2], (x1, y1)


class ForegroundExtractor:
    """经典 CV 前景提取管道。

    extract(patch, inner_box, scale):
        patch     - BGR uint8 图像（可为超分放大后的 ROI）
        inner_box - 标注框在 patch 坐标系的 (x, y, w, h)
        scale     - patch 相对原图的放大倍数，统计坐标映射回原图用
    """

    def __init__(self, min_area_ratio=0.005, max_area_ratio=0.95):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def extract(self, patch, inner_box, scale=1.0):
        if patch is None or patch.size == 0:
            return ExtractResult()
        x, y, w, h = [int(v) for v in inner_box]
        if w <= 0 or h <= 0:
            return ExtractResult()

        enhanced = self._preprocess(patch)
        polarity = self._decide_polarity(enhanced, (x, y, w, h))

        candidates = []
        otsu = self._threshold_otsu(enhanced, polarity)
        candidates.append(("otsu", otsu))
        adaptive = self._threshold_adaptive(enhanced, polarity)
        candidates.append(("adaptive", adaptive))
        if min(w, h) / scale >= 30:
            gc = self._grabcut(patch, (x, y, w, h))
            if gc is not None:
                candidates.append(("grabcut", gc))

        best = None
        for name, m in candidates:
            cleaned = self._postprocess(m, patch.shape[:2], (x, y, w, h))
            if cleaned is None:
                continue
            score = self._score(cleaned, (x, y, w, h))
            if best is None or score > best[0]:
                best = (score, name, cleaned)

        if best is None:
            inv_polarity = "dark" if polarity == "bright" else "bright"
            otsu_inv = self._threshold_otsu(enhanced, inv_polarity)
            cleaned = self._postprocess(otsu_inv, patch.shape[:2], (x, y, w, h))
            if cleaned is not None:
                best = (self._score(cleaned, (x, y, w, h)), "otsu(inv)", cleaned)

        if best is None:
            return ExtractResult(detected=False, polarity=polarity)

        _, method, mask = best
        bbox, centroid, area = self._measure(mask, scale)
        mask_area = float(np.count_nonzero(mask))
        box_area = float(w * h)
        return ExtractResult(
            detected=True,
            mask=mask,
            bbox_orig=bbox,
            centroid_orig=centroid,
            area_orig=area,
            area_ratio=mask_area / box_area if box_area > 0 else 0.0,
            method=method,
            polarity=polarity,
        )

    def _preprocess(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def _decide_polarity(self, enhanced, box):
        x, y, w, h = box
        H, W = enhanced.shape[:2]
        inner = enhanced[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        if inner.size == 0:
            return "bright"
        m = int(min(H, W) * 0.5)
        ax1, ay1 = max(0, x - m), max(0, y - m)
        ax2, ay2 = min(W, x + w + m), min(H, y + h + m)
        ring = enhanced[ay1:ay2, ax1:ax2].copy()
        ring[max(y, 0) - ay1:min(y + h, H) - ay1,
             max(x, 0) - ax1:min(x + w, W) - ax1] = 0
        nz = ring[ring > 0]
        if nz.size == 0:
            return "bright"
        return "bright" if inner.mean() > nz.mean() else "dark"

    def _threshold_otsu(self, enhanced, polarity):
        flag = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
        _, m = cv2.threshold(enhanced, 0, 255, flag + cv2.THRESH_OTSU)
        return m

    def _threshold_adaptive(self, enhanced, polarity):
        H, W = enhanced.shape[:2]
        block = max(3, int(min(H, W) // 2))
        block = block if block % 2 == 1 else block + 1
        flag = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
        return cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     flag, block, 5)

    def _grabcut(self, patch, box):
        x, y, w, h = box
        H, W = patch.shape[:2]
        if w < 3 or h < 3:
            return None
        band = int(0.5 * max(w, h))
        bx0, by0 = max(0, x - band), max(0, y - band)
        bx1, by1 = min(W, x + w + band), min(H, y + h + band)
        gc_mask = np.full((H, W), cv2.GC_BGD, np.uint8)
        gc_mask[by0:by1, bx0:bx1] = cv2.GC_PR_BGD
        gc_mask[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = cv2.GC_PR_FGD
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(patch, gc_mask, None, bgd, fgd, 10, cv2.GC_INIT_WITH_MASK)
        keep = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        return np.where(keep, 255, 0).astype(np.uint8)

    def _postprocess(self, mask, size, box):
        x, y, w, h = box
        H, W = size
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        n, labels, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        total = H * W
        bx0, by0 = max(0, x), max(0, y)
        bx1, by1 = min(W, x + w), min(H, y + h)
        if bx1 <= bx0 or by1 <= by0:
            return None
        box_region = np.zeros((H, W), np.uint8)
        box_region[by0:by1, bx0:bx1] = 1
        best = None
        best_area = 0
        for i in range(1, n):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if a < self.min_area_ratio * total or a > self.max_area_ratio * total:
                continue
            sub = labels == i
            inter = int(np.count_nonzero(sub & (box_region > 0)))
            if inter <= 0 or inter / a < 0.5:
                continue
            if a > best_area:
                best_area = a
                best = i
        if best is None:
            return None
        comp = labels == best
        result = (comp & (box_region > 0)).astype(np.uint8) * 255
        if np.count_nonzero(result) == 0:
            return None
        return result

    def _score(self, mask, box):
        x, y, w, h = box
        H, W = mask.shape[:2]
        mask_area = float(np.count_nonzero(mask))
        if mask_area <= 0:
            return -1.0
        box_area = float(max(w, 1) * max(h, 1))
        inner = mask[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        inter = float(np.count_nonzero(inner))
        in_ratio = inter / box_area if box_area > 0 else 0.0
        out_ratio = (mask_area - inter) / box_area if box_area > 0 else 0.0
        return in_ratio - 0.5 * out_ratio

    def _measure(self, mask, scale):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0), (0.0, 0.0), 0.0
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max()), float(ys.max())
        area = float(len(xs)) / (scale * scale)
        bbox = (
            int(round(x1 / scale)),
            int(round(y1 / scale)),
            int(round((x2 - x1 + 1) / scale)),
            int(round((y2 - y1 + 1) / scale)),
        )
        centroid = (float((x1 + x2) / 2.0 / scale), float((y1 + y2) / 2.0 / scale))
        return bbox, centroid, area


_DEFAULT_EXTRACTOR = ForegroundExtractor()


def traditional_mask(image, box):
    """小目标经典 CV 管道：ROI 超分 + 前景提取，返回与原图等大的 0/255 掩码。"""
    x, y, w, h = box
    roi, roi_tl = crop_roi(image, box, 1.0)
    if roi is None:
        return None
    scale = choose_scale(w, h)
    up = upscale_patch(roi, scale)
    inner_box = (
        int(round((x - roi_tl[0]) * scale)),
        int(round((y - roi_tl[1]) * scale)),
        int(round(w * scale)),
        int(round(h * scale)),
    )
    res = _DEFAULT_EXTRACTOR.extract(up, inner_box, scale=scale)
    if not res.detected:
        return None
    H, W = image.shape[:2]
    rh, rw = roi.shape[:2]
    m = (res.mask > 0).astype(np.uint8)
    m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((H, W), np.uint8)
    y0, x0 = roi_tl[1], roi_tl[0]
    y1, x1 = min(H, y0 + rh), min(W, x0 + rw)
    mask[y0:y1, x0:x1] = m[0:y1 - y0, 0:x1 - x0]
    return mask


def fit_metrics(mask, box, ann_mask=None):
    """基于检测掩码与标注计算贴合度指标。box: [x, y, w, h]（图像坐标）。
    ann_mask 可选：标注多边形渲染的掩码；传入时 mask_iou 为检测掩码与标注掩码的交并比，
    否则 mask_iou 为检测掩码与标注框的交并比（框内填充度，适用于矩形框标注）。"""
    x, y, w, h = box
    x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
    box_area = max(w * h, 1)
    H, W = mask.shape[:2]
    mask_bin = mask > 0.5
    fg_area = float(mask_bin.sum())
    if fg_area <= 0:
        return None
    ann_bin = None
    if ann_mask is not None and ann_mask.shape[:2] == (H, W):
        cand = ann_mask > 0.5
        if float(cand.sum()) > 0:
            ann_bin = cand
    if ann_bin is not None:
        inter_mask = float((mask_bin & ann_bin).sum())
        mask_iou = inter_mask / (float(ann_bin.sum()) + fg_area - inter_mask + 1e-6)
    else:
        xi0, xi1 = max(0, int(round(x1))), min(W, int(round(x2)))
        yi0, yi1 = max(0, int(round(y1))), min(H, int(round(y2)))
        inter_mask = 0.0
        if xi1 > xi0 and yi1 > yi0:
            inter_mask = float(mask_bin[yi0:yi1, xi0:xi1].sum())
        mask_iou = inter_mask / (box_area + fg_area - inter_mask + 1e-6)
    ys, xs = np.nonzero(mask_bin)
    fx1, fy1, fx2, fy2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    inter_w = min(fx2, x2) - max(fx1, x1)
    inter_h = min(fy2, y2) - max(fy1, y1)
    inter_bbox = max(inter_w, 0) * max(inter_h, 0)
    bbox_area = max((fx2 - fx1) * (fy2 - fy1), 1)
    bbox_iou = inter_bbox / (box_area + bbox_area - inter_bbox + 1e-6)
    area_ratio = fg_area / box_area
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diag = float(np.hypot(w, h))
    centroid_offset = float(np.hypot(cx - bx, cy - by)) / max(diag, 1e-6)
    return {
        "fg_area": round(fg_area, 1),
        "box_area": round(box_area, 1),
        "area_ratio": round(area_ratio, 3),
        "bbox_iou": round(bbox_iou, 3),
        "mask_iou": round(mask_iou, 3),
        "centroid_offset": round(centroid_offset, 3),
        "fg_bbox": [fx1, fy1, fx2, fy2],
        "overflow": [max(x1 - fx1, 0), max(y1 - fy1, 0), max(fx2 - x2, 0), max(fy2 - y2, 0)],
        "deficit": [max(fx1 - x1, 0), max(fy1 - y1, 0), max(x2 - fx2, 0), max(y2 - fy2, 0)],
    }


def boundary_gradient(img, box, edge_width=1):
    """标注框边界处的梯度强度均值：数值越高说明目标边缘紧贴标注框。"""
    x, y, w, h = box
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = []
    for dx in range(edge_width):
        yy = y + dx
        if 0 <= yy < H:
            vals.append(mag[yy, max(0, x):min(W, x + w)])
        yy = y + h - 1 - dx
        if 0 <= yy < H:
            vals.append(mag[yy, max(0, x):min(W, x + w)])
        xx = x + dx
        if 0 <= xx < W:
            vals.append(mag[max(0, y):min(H, y + h), xx])
        xx = x + w - 1 - dx
        if 0 <= xx < W:
            vals.append(mag[max(0, y):min(H, y + h), xx])
    if not vals:
        return 0.0
    return float(np.concatenate(vals).mean())


def mask_to_polygon(mask, epsilon_ratio=0.001):
    """将二值掩码转换为多边形点集 [[x, y], ...]（图像坐标），空掩码返回 None。"""
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) <= 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    epsilon = max(epsilon_ratio * peri, 2.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        approx = contour
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def mask_to_bbox(mask):
    """二值掩码的外接矩形 [x, y, w, h]（图像坐标），空掩码返回 None。"""
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) <= 0:
        return None
    x, y, w, h = cv2.boundingRect(m)
    return [int(x), int(y), int(w), int(h)]
