"""
SAHI 风格的后处理策略模块
提供 NMS、NMM、GREEDYNMM 三种后处理算法
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class PostprocessType(Enum):
    """后处理类型"""
    NMS = "NMS"           # 非极大抑制
    NMM = "NMM"           # 非极大合并
    GREEDYNMM = "GREEDYNMM"  # 贪婪非极大合并（默认）


class MatchMetric(Enum):
    """匹配度量标准"""
    IOU = "IOU"  # 交并比
    IOS = "IOS"  # 交小比（对小目标更敏感）


@dataclass
class Detection:
    """检测结果数据类，用于后处理"""
    bbox: List[float]  # [x, y, w, h]
    score: float       # 置信度
    label: str         # 类别标签
    polygons: Optional[List[List[List[float]]]] = None  # 可选的多边形
    mask: Optional[np.ndarray] = None  # 可选的掩码


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的 IOU (Intersection over Union)

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]

    Returns:
        IOU 值 (0.0 - 1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_ios(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的 IOS (Intersection over Smaller)
    对小目标更敏感

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]

    Returns:
        IOS 值 (0.0 - 1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    smaller_area = min(area1, area2)

    return intersection / smaller_area if smaller_area > 0 else 0.0


def compute_overlap(box1: List[float], box2: List[float],
                   metric: MatchMetric = MatchMetric.IOU) -> float:
    """
    计算两个边界框的重叠度

    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]
        metric: 匹配度量标准

    Returns:
        重叠度 (0.0 - 1.0)
    """
    if metric == MatchMetric.IOU:
        return compute_iou(box1, box2)
    else:
        return compute_ios(box1, box2)


def annotation_to_detection(ann: Dict[str, Any]) -> Detection:
    """将标注字典转换为 Detection 对象"""
    return Detection(
        bbox=ann.get('bbox', [0, 0, 0, 0]),
        score=ann.get('confidence', 0.0),
        label=ann.get('label', 'unknown'),
        polygons=ann.get('polygons'),
        mask=ann.get('mask')
    )


def detection_to_annotation(det: Detection) -> Dict[str, Any]:
    """将 Detection 对象转换为标注字典"""
    ann = {
        'bbox': det.bbox,
        'confidence': det.score,
        'label': det.label
    }
    if det.polygons:
        ann['polygons'] = det.polygons
    if det.mask is not None:
        ann['mask'] = det.mask
    return ann


class NMSPostprocess:
    """
    非极大抑制 (Non-Maximum Suppression)

    算法流程：
    1. 按置信度降序排序所有检测框
    2. 选择置信度最高的框加入结果列表
    3. 计算该框与其余所有框的 IOU/IOS
    4. 删除 IOU/IOS > 阈值的所有框（认为是同一目标）
    5. 重复步骤 2-4 直到所有框处理完毕
    """

    def __init__(self,
                 match_threshold: float = 0.5,
                 match_metric: MatchMetric = MatchMetric.IOU,
                 class_agnostic: bool = False):
        """
        Args:
            match_threshold: 匹配阈值，超过此值认为是同一目标
            match_metric: 匹配度量标准 (IOU 或 IOS)
            class_agnostic: 是否忽略类别，True 时只根据位置判断
        """
        self.match_threshold = match_threshold
        self.match_metric = match_metric
        self.class_agnostic = class_agnostic

    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行 NMS 后处理"""
        if not annotations:
            return []

        # 转换为 Detection 对象
        detections = [annotation_to_detection(ann) for ann in annotations]

        # 按类别分组（如果不是 class_agnostic）
        if self.class_agnostic:
            result = self._nms_single_group(detections)
        else:
            # 按类别分组处理
            label_groups = {}
            for det in detections:
                if det.label not in label_groups:
                    label_groups[det.label] = []
                label_groups[det.label].append(det)

            # 对每个类别分别执行 NMS
            result = []
            for label, group in label_groups.items():
                result.extend(self._nms_single_group(group))

        return [detection_to_annotation(det) for det in result]

    def _nms_single_group(self, detections: List[Detection]) -> List[Detection]:
        """对单组检测框执行 NMS"""
        if not detections:
            return []

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda x: x.score, reverse=True)

        keep = []  # 保留的检测框索引
        suppressed = [False] * len(sorted_dets)  # 标记是否被抑制

        for i in range(len(sorted_dets)):
            if suppressed[i]:
                continue

            # 保留当前框
            keep.append(sorted_dets[i])

            # 抑制所有与当前框重叠度高的框
            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue

                overlap = compute_overlap(
                    sorted_dets[i].bbox,
                    sorted_dets[j].bbox,
                    self.match_metric
                )

                if overlap > self.match_threshold:
                    suppressed[j] = True

        return keep


class NMMPostprocess:
    """
    非极大合并 (Non-Maximum Merging)

    算法流程：
    1. 按置信度降序排序所有检测框
    2. 构建重叠组：从最高置信度开始，将所有重叠度 > 阈值的框加入同一组
    3. 对每个组内的框进行合并（加权平均）
    4. 返回合并后的框

    与 NMS 的区别：NMS 直接删除重叠框，NMM 合并重叠框的信息
    """

    def __init__(self,
                 match_threshold: float = 0.5,
                 match_metric: MatchMetric = MatchMetric.IOU,
                 class_agnostic: bool = False):
        """
        Args:
            match_threshold: 匹配阈值
            match_metric: 匹配度量标准
            class_agnostic: 是否忽略类别
        """
        self.match_threshold = match_threshold
        self.match_metric = match_metric
        self.class_agnostic = class_agnostic

    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行 NMM 后处理"""
        if not annotations:
            return []

        detections = [annotation_to_detection(ann) for ann in annotations]

        if self.class_agnostic:
            merged = self._nmm_single_group(detections)
        else:
            label_groups = {}
            for det in detections:
                if det.label not in label_groups:
                    label_groups[det.label] = []
                label_groups[det.label].append(det)

            merged = []
            for label, group in label_groups.items():
                merged.extend(self._nmm_single_group(group))

        return [detection_to_annotation(det) for det in merged]

    def _nmm_single_group(self, detections: List[Detection]) -> List[Detection]:
        """对单组检测框执行 NMM"""
        if not detections:
            return []

        if len(detections) == 1:
            return detections

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda x: x.score, reverse=True)

        # 构建重叠组
        groups = []  # 每个组是一个 Detection 列表
        used = [False] * len(sorted_dets)

        for i in range(len(sorted_dets)):
            if used[i]:
                continue

            # 创建新组
            group = [sorted_dets[i]]
            used[i] = True

            # 找到所有与该组成员重叠的框
            for j in range(i + 1, len(sorted_dets)):
                if used[j]:
                    continue

                # 检查是否与组内任何成员重叠
                for member in group:
                    overlap = compute_overlap(
                        member.bbox,
                        sorted_dets[j].bbox,
                        self.match_metric
                    )
                    if overlap > self.match_threshold:
                        group.append(sorted_dets[j])
                        used[j] = True
                        break

            groups.append(group)

        # 合并每个组
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_detections(group))

        return merged

    def _merge_detections(self, detections: List[Detection]) -> Detection:
        """合并多个检测框"""
        if not detections:
            return None
        if len(detections) == 1:
            return detections[0]

        # 计算加权平均的 bbox
        total_score = sum(d.score for d in detections)
        weights = [d.score / total_score for d in detections]

        # 加权平均中心点和尺寸
        centers_x = [d.bbox[0] + d.bbox[2] / 2 for d in detections]
        centers_y = [d.bbox[1] + d.bbox[3] / 2 for d in detections]
        widths = [d.bbox[2] for d in detections]
        heights = [d.bbox[3] for d in detections]

        merged_cx = sum(cx * w for cx, w in zip(centers_x, weights))
        merged_cy = sum(cy * w for cy, w in zip(centers_y, weights))
        merged_w = sum(w * wt for w, wt in zip(widths, weights))
        merged_h = sum(h * wt for h, wt in zip(heights, weights))

        merged_bbox = [
            merged_cx - merged_w / 2,
            merged_cy - merged_h / 2,
            merged_w,
            merged_h
        ]

        # 使用最高置信度和类别
        best_det = max(detections, key=lambda x: x.score)

        return Detection(
            bbox=merged_bbox,
            score=best_det.score,
            label=best_det.label,
            polygons=best_det.polygons,
            mask=best_det.mask
        )


class GreedyNMMPostprocess:
    """
    贪婪非极大合并 (Greedy Non-Maximum Merging)

    SAHI 默认的后处理算法，平衡了 NMS 和 NMM 的优点。

    算法流程：
    1. 按置信度降序排序
    2. 从最高置信度的框开始
    3. 贪婪地找到第一个重叠度 > 阈值的框，与之合并
    4. 用合并后的框继续寻找下一个重叠框
    5. 重复直到没有重叠框，将结果加入输出
    6. 处理下一个未处理的框

    与 NMM 的区别：NMM 一次性构建所有重叠组，GreedyNMM 是渐进式合并
    """

    def __init__(self,
                 match_threshold: float = 0.5,
                 match_metric: MatchMetric = MatchMetric.IOS,  # 默认使用 IOS，对小目标更敏感
                 class_agnostic: bool = False):
        """
        Args:
            match_threshold: 匹配阈值
            match_metric: 匹配度量标准，默认 IOS 对小目标更友好
            class_agnostic: 是否忽略类别
        """
        self.match_threshold = match_threshold
        self.match_metric = match_metric
        self.class_agnostic = class_agnostic

    def __call__(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行 Greedy NMM 后处理"""
        if not annotations:
            return []

        detections = [annotation_to_detection(ann) for ann in annotations]

        if self.class_agnostic:
            merged = self._greedy_nmm_single_group(detections)
        else:
            label_groups = {}
            for det in detections:
                if det.label not in label_groups:
                    label_groups[det.label] = []
                label_groups[det.label].append(det)

            merged = []
            for label, group in label_groups.items():
                merged.extend(self._greedy_nmm_single_group(group))

        return [detection_to_annotation(det) for det in merged]

    def _greedy_nmm_single_group(self, detections: List[Detection]) -> List[Detection]:
        """对单组检测框执行 Greedy NMM"""
        if not detections:
            return []

        if len(detections) == 1:
            return detections

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda x: x.score, reverse=True)

        merged = []  # 最终结果
        used = [False] * len(sorted_dets)

        for i in range(len(sorted_dets)):
            if used[i]:
                continue

            # 以当前框为基准开始合并
            current = sorted_dets[i]
            used[i] = True

            # 贪婪地合并重叠框
            changed = True
            while changed:
                changed = False
                for j in range(len(sorted_dets)):
                    if used[j]:
                        continue

                    overlap = compute_overlap(
                        current.bbox,
                        sorted_dets[j].bbox,
                        self.match_metric
                    )

                    if overlap > self.match_threshold:
                        # 合并两个框
                        current = self._merge_two_detections(current, sorted_dets[j])
                        used[j] = True
                        changed = True
                        break  # 重新开始搜索

            merged.append(current)

        return merged

    def _merge_two_detections(self, det1: Detection, det2: Detection) -> Detection:
        """合并两个检测框"""
        # 计算加权平均
        total_score = det1.score + det2.score
        w1 = det1.score / total_score
        w2 = det2.score / total_score

        # 加权平均中心点
        cx1 = det1.bbox[0] + det1.bbox[2] / 2
        cy1 = det1.bbox[1] + det1.bbox[3] / 2
        cx2 = det2.bbox[0] + det2.bbox[2] / 2
        cy2 = det2.bbox[1] + det2.bbox[3] / 2

        merged_cx = cx1 * w1 + cx2 * w2
        merged_cy = cy1 * w1 + cy2 * w2
        merged_w = det1.bbox[2] * w1 + det2.bbox[2] * w2
        merged_h = det1.bbox[3] * w1 + det2.bbox[3] * w2

        merged_bbox = [
            merged_cx - merged_w / 2,
            merged_cy - merged_h / 2,
            merged_w,
            merged_h
        ]

        # 保留更高置信度和对应的多边形/掩码
        if det1.score >= det2.score:
            return Detection(
                bbox=merged_bbox,
                score=max(det1.score, det2.score),
                label=det1.label,
                polygons=det1.polygons,
                mask=det1.mask
            )
        else:
            return Detection(
                bbox=merged_bbox,
                score=max(det1.score, det2.score),
                label=det2.label,
                polygons=det2.polygons,
                mask=det2.mask
            )


def create_postprocess(postprocess_type: PostprocessType = PostprocessType.GREEDYNMM,
                      match_threshold: float = 0.5,
                      match_metric: MatchMetric = MatchMetric.IOS,
                      class_agnostic: bool = False):
    """
    创建后处理器

    Args:
        postprocess_type: 后处理类型
        match_threshold: 匹配阈值
        match_metric: 匹配度量标准
        class_agnostic: 是否忽略类别

    Returns:
        后处理器实例
    """
    if postprocess_type == PostprocessType.NMS:
        return NMSPostprocess(match_threshold, match_metric, class_agnostic)
    elif postprocess_type == PostprocessType.NMM:
        return NMMPostprocess(match_threshold, match_metric, class_agnostic)
    else:  # GREEDYNMM
        return GreedyNMMPostprocess(match_threshold, match_metric, class_agnostic)
