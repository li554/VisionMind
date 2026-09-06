"""
推理策略模块
提供ROI推理和切片推理的策略接口及实现
使用策略模式，统一不同推理模式的接口
"""

from abc import ABC, abstractmethod
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from .utils import polygons_to_mask, mask_to_polygons
from .postprocess import create_postprocess, PostprocessType, MatchMetric

@dataclass
class ImageSlice:
    """图像切片数据类"""
    image: np.ndarray
    bbox: List[int]  # [x, y, w, h] 在原图中的位置
    row: int
    col: int


@dataclass
class InferenceContext:
    """推理上下文，保存推理过程中的状态"""
    original_image: np.ndarray
    slices: List[ImageSlice]
    results: List[Dict[str, Any]]
    h: int
    w: int
    
    def __init__(self, original_image: np.ndarray):
        self.original_image = original_image
        self.slices = []
        self.results = []
        self.h, self.w = original_image.shape[:2]


class InferenceStrategy(ABC):
    """
    推理策略接口（抽象基类）
    所有推理模式（标准、ROI、切片）都需要实现此接口
    """
    
    @abstractmethod
    def prepare(self, context: InferenceContext, **kwargs) -> bool:
        """
        准备阶段：生成ROI图或切片
        
        Args:
            context: 推理上下文
            **kwargs: 额外参数
            
        Returns:
            是否准备成功
        """
        pass
    
    @abstractmethod
    def infer(self, context: InferenceContext, 
              inference_fn: Callable[[np.ndarray], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        推理阶段：对每个处理单元执行推理
        
        Args:
            context: 推理上下文
            inference_fn: 推理函数
            
        Returns:
            所有处理单元的推理结果列表
        """
        pass
    
    @abstractmethod
    def postprocess(self, context: InferenceContext, 
                   annotations: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        后处理阶段：将结果映射回原图或合并结果
        
        Args:
            context: 推理上下文
            annotations: 所有处理单元的标注结果
            **kwargs: 额外参数
            
        Returns:
            处理后的标注结果
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass


class StandardInferenceStrategy(InferenceStrategy):
    """标准推理策略（全图推理）"""
    
    @property
    def name(self) -> str:
        return "standard"
    
    def prepare(self, context: InferenceContext, **kwargs) -> bool:
        """标准模式：将整个图像作为单个处理单元"""
        context.slices = [ImageSlice(
            image=context.original_image,
            bbox=[0, 0, context.w, context.h],
            row=0,
            col=0
        )]
        return True
    
    def infer(self, context: InferenceContext, 
              inference_fn: Callable[[np.ndarray], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准模式：直接对整个图像推理"""
        result = inference_fn(context.original_image)
        return [result] if result else []
    
    def postprocess(self, context: InferenceContext, 
                   annotations: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """标准模式：无需坐标映射，直接返回"""
        if annotations and len(annotations) > 0:
            result = annotations[0]
            if result.get('status') == 'success':
                return result.get('annotations', [])
        return []


class ROIInferenceStrategy(InferenceStrategy):
    """ROI推理策略"""
    
    def __init__(self, roi: Tuple[int, int, int, int]):
        """
        Args:
            roi: (x, y, w, h) ROI边界框
        """
        self.roi = roi
        self.offset_x = 0
        self.offset_y = 0
    
    @property
    def name(self) -> str:
        return "roi"
    
    def prepare(self, context: InferenceContext, **kwargs) -> bool:
        """ROI模式：裁剪ROI区域"""
        x, y, w, h = self.roi
        
        # 确保ROI在图像范围内
        x1 = max(0, min(x, context.w - 1))
        y1 = max(0, min(y, context.h - 1))
        x2 = max(x1 + 1, min(x + w, context.w))
        y2 = max(y1 + 1, min(y + h, context.h))
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        self.offset_x = x1
        self.offset_y = y1
        
        roi_image = context.original_image[y1:y2, x1:x2]
        context.slices = [ImageSlice(
            image=roi_image,
            bbox=[x1, y1, x2 - x1, y2 - y1],
            row=0,
            col=0
        )]
        return True
    
    def infer(self, context: InferenceContext, 
              inference_fn: Callable[[np.ndarray], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ROI模式：对ROI区域推理"""
        if not context.slices:
            return []
        
        roi_slice = context.slices[0]
        result = inference_fn(roi_slice.image)
        return [result] if result else []
    
    def postprocess(self, context: InferenceContext, 
                   annotations: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """ROI模式：将坐标映射回原图"""
        if not annotations or len(annotations) == 0:
            return []
        
        result = annotations[0]
        if result.get('status') != 'success':
            return []
        
        # 将ROI内的坐标映射回原图坐标
        for ann in result.get('annotations', []):
            if 'bbox' in ann:
                ann['bbox'] = [
                    ann['bbox'][0] + self.offset_x,
                    ann['bbox'][1] + self.offset_y,
                    ann['bbox'][2],
                    ann['bbox'][3]
                ]
            if 'polygons' in ann:
                ann['polygons'] = [
                    [[p[0] + self.offset_x, p[1] + self.offset_y] for p in poly]
                    for poly in ann['polygons']
                ]
        
        return result.get('annotations', [])


class SliceInferenceStrategy(InferenceStrategy):
    """切片推理策略"""

    def __init__(self, rows: int = 2, cols: int = 2, overlap_ratio: float = 0.2,
                 postprocess_type: PostprocessType = PostprocessType.GREEDYNMM,
                 match_threshold: float = 0.5,
                 match_metric: MatchMetric = MatchMetric.IOS):
        """
        Args:
            rows: 切片行数
            cols: 切片列数
            overlap_ratio: 相邻切片之间的重叠比例 (0.0-1.0)，默认 20%
            postprocess_type: 后处理类型 (NMS, NMM, GREEDYNMM)
            match_threshold: 匹配阈值
            match_metric: 匹配度量标准 (IOU 或 IOS)
        """
        self.rows = rows
        self.cols = cols
        self.overlap_ratio = overlap_ratio
        self.postprocess_type = postprocess_type
        self.match_threshold = match_threshold
        self.match_metric = match_metric

    @property
    def name(self) -> str:
        return "slice"

    def prepare(self, context: InferenceContext, **kwargs) -> bool:
        """切片模式：将图像切分成多个切片"""
        context.slices = self._create_slices(
            context.original_image,
            self.rows,
            self.cols,
            self.overlap_ratio
        )
        return len(context.slices) > 0
    
    def infer(self, context: InferenceContext,
              inference_fn: Callable[[np.ndarray], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """切片模式：对原图和每个切片分别推理，然后合并结果"""
        results = []
        total_detections = 0

        # 1. 首先对原图进行推理
        print(f"[SliceInferenceStrategy] ====== 开始切片推理 ======")
        print(f"[SliceInferenceStrategy] 步骤1: 对原图进行推理...")
        original_result = inference_fn(context.original_image)
        if original_result:
            original_count = len(original_result.get('annotations', []))
            total_detections += original_count
            print(f"[SliceInferenceStrategy]   原图检测到 {original_count} 个目标")
            results.append({
                'result': original_result,
                'slice': None  # None 表示原图结果
            })
        else:
            print(f"[SliceInferenceStrategy]   原图未检测到目标")

        # 2. 对每个切片进行推理
        print(f"[SliceInferenceStrategy] 步骤2: 对 {len(context.slices)} 个切片进行推理...")
        slice_detection_counts = []
        for i, slice_info in enumerate(context.slices):
            result = inference_fn(slice_info.image)
            if result:
                count = len(result.get('annotations', []))
                total_detections += count
                slice_detection_counts.append((slice_info.row, slice_info.col, count))
                print(f"[SliceInferenceStrategy]   切片 ({slice_info.row},{slice_info.col}): 检测到 {count} 个目标")
                results.append({
                    'result': result,
                    'slice': slice_info
                })
            else:
                slice_detection_counts.append((slice_info.row, slice_info.col, 0))
                print(f"[SliceInferenceStrategy]   切片 ({slice_info.row},{slice_info.col}): 未检测到目标")

        print(f"[SliceInferenceStrategy] 步骤3: 推理完成统计")
        print(f"[SliceInferenceStrategy]   - 切片总数: {len(context.slices)}")
        print(f"[SliceInferenceStrategy]   - 原图检测: {len(original_result.get('annotations', [])) if original_result else 0} 个目标")
        print(f"[SliceInferenceStrategy]   - 切片检测: {total_detections - (len(original_result.get('annotations', [])) if original_result else 0)} 个目标")
        print(f"[SliceInferenceStrategy]   - 检测总数: {total_detections} 个目标")

        return results
    
    def postprocess(self, context: InferenceContext,
                   annotations: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """切片模式：合并原图和所有切片的结果"""
        all_annotations = []
        original_count = 0
        slice_count = 0

        print(f"[SliceInferenceStrategy] 步骤4: 收集所有检测结果...")

        for item in annotations:
            result = item.get('result', {})
            slice_info = item.get('slice')

            if result.get('status') == 'success':
                anns = result.get('annotations', [])
                if slice_info is None:
                    # 原图结果，直接使用，置信度设为1
                    original_count = len(anns)
                    for ann in anns:
                        ann['source'] = 'original'
                        ann['confidence'] = 1.0
                        all_annotations.append(ann)
                else:
                    # 切片结果，将坐标转换为原图坐标
                    slice_count += len(anns)
                    for ann in anns:
                        if 'bbox' in ann:
                            ann['bbox'] = [
                                ann['bbox'][0] + slice_info.bbox[0],
                                ann['bbox'][1] + slice_info.bbox[1],
                                ann['bbox'][2],
                                ann['bbox'][3]
                            ]
                        if 'polygons' in ann:
                            ann['polygons'] = [
                                [[p[0] + slice_info.bbox[0], p[1] + slice_info.bbox[1]] for p in poly]
                                for poly in ann['polygons']
                            ]
                        ann['source'] = f"slice({slice_info.row},{slice_info.col})"
                        all_annotations.append(ann)

        print(f"[SliceInferenceStrategy]   - 原图标注: {original_count} 个")
        print(f"[SliceInferenceStrategy]   - 切片标注: {slice_count} 个")
        print(f"[SliceInferenceStrategy]   - 合并前总数: {len(all_annotations)} 个")

        # 合并重叠区域的检测结果
        print(f"[SliceInferenceStrategy] 步骤5: 执行后处理合并...")
        merged = self._merge_annotations(all_annotations, context.h, context.w)

        print(f"[SliceInferenceStrategy] 步骤6: 合并完成")
        print(f"[SliceInferenceStrategy]   - 合并后总数: {len(merged)} 个")
        print(f"[SliceInferenceStrategy]   - 减少数量: {len(all_annotations) - len(merged)} 个")
        print(f"[SliceInferenceStrategy] ====== 切片推理结束 ======")

        return merged
    
    def _create_slices(self, image: np.ndarray, rows: int, cols: int,
                      overlap_ratio: float) -> List[ImageSlice]:
        """创建图像切片（使用比例计算重叠）"""
        h, w = image.shape[:2]
        slices = []

        # 计算每个切片的尺寸（不考虑重叠）
        slice_w_base = w / cols if cols > 0 else w
        slice_h_base = h / rows if rows > 0 else h

        # 计算重叠像素数
        overlap_w = int(slice_w_base * overlap_ratio)
        overlap_h = int(slice_h_base * overlap_ratio)

        print(f"[SliceInferenceStrategy] 图像尺寸: {w}x{h}, 切片: {rows}x{cols}")
        print(f"[SliceInferenceStrategy] 基础切片尺寸: {slice_w_base:.1f}x{slice_h_base:.1f}")
        print(f"[SliceInferenceStrategy] 重叠像素: {overlap_w}x{overlap_h} (比例: {overlap_ratio:.1%})")

        for row in range(rows):
            for col in range(cols):
                # 计算切片的起始位置（考虑重叠）
                x_start = int(col * slice_w_base) - int(col * overlap_w)
                y_start = int(row * slice_h_base) - int(row * overlap_h)

                # 计算切片的结束位置（考虑重叠）
                x_end = min(int((col + 1) * slice_w_base) - int(col * overlap_w) + overlap_w, w)
                y_end = min(int((row + 1) * slice_h_base) - int(row * overlap_h) + overlap_h, h)

                # 确保起始位置不小于0
                x_start = max(0, x_start)
                y_start = max(0, y_start)

                slice_w = x_end - x_start
                slice_h = y_end - y_start

                if slice_w > 0 and slice_h > 0:
                    slice_img = image[y_start:y_end, x_start:x_end]
                    slices.append(ImageSlice(
                        image=slice_img,
                        bbox=[x_start, y_start, slice_w, slice_h],
                        row=row,
                        col=col
                    ))
                    print(f"[SliceInferenceStrategy] 切片 ({row},{col}): 位置=({x_start},{y_start}), 尺寸={slice_w}x{slice_h}")

        return slices
    
    def _merge_annotations(self, annotations: List[Dict], h: int, w: int) -> List[Dict]:
        """
        合并重叠区域的标注

        使用 SAHI 风格的后处理策略：
        1. 支持 NMS、NMM、GREEDYNMM 三种算法
        2. 支持 IOU 和 IOS 两种匹配度量
        3. 类别感知（只有相同类别才会合并）
        """
        if not annotations:
            print(f"[SliceInferenceStrategy]   没有标注需要合并")
            return []

        print(f"[SliceInferenceStrategy]   后处理配置:")
        print(f"[SliceInferenceStrategy]     - 算法: {self.postprocess_type.value}")
        print(f"[SliceInferenceStrategy]     - 匹配度量: {self.match_metric.value}")
        print(f"[SliceInferenceStrategy]     - 匹配阈值: {self.match_threshold}")

        # 统计各类别数量
        from collections import Counter
        categories = Counter([ann.get('label', 'unknown') for ann in annotations])
        print(f"[SliceInferenceStrategy]     - 类别分布: {dict(categories)}")

        # 创建 SAHI 风格的后处理器
        postprocessor = create_postprocess(
            postprocess_type=self.postprocess_type,
            match_threshold=self.match_threshold,
            match_metric=self.match_metric,
            class_agnostic=False  # 类别感知模式
        )

        # 执行后处理
        merged = postprocessor(annotations)

        # 显示合并后的置信度分布
        if merged:
            confidences = [ann.get('confidence', 0) for ann in merged]
            print(f"[SliceInferenceStrategy]     - 合并后置信度范围: [{min(confidences):.2f}, {max(confidences):.2f}]")

        return merged


class StrategyFactory:
    """策略工厂，用于创建对应的推理策略"""
    
    @staticmethod
    def create_strategy(mode: str, **kwargs) -> InferenceStrategy:
        """
        创建推理策略
        
        Args:
            mode: 模式名称 ('standard', 'roi', 'slice')
            **kwargs: 模式特定参数
            
        Returns:
            对应的推理策略实例
        """
        mode = mode.lower()
        
        if mode == 'standard':
            return StandardInferenceStrategy()
        
        elif mode == 'roi':
            roi = kwargs.get('roi')
            if roi is None:
                raise ValueError("ROI mode requires 'roi' parameter")
            return ROIInferenceStrategy(roi)
        
        elif mode == 'slice':
            rows = kwargs.get('rows', 2)
            cols = kwargs.get('cols', 2)
            overlap = kwargs.get('overlap', 50)
            return SliceInferenceStrategy(rows, cols, overlap)
        
        else:
            raise ValueError(f"Unknown inference mode: {mode}")


class InferenceEngine:
    """
    统一推理引擎
    使用策略模式，根据配置选择不同的推理策略
    """
    
    def __init__(self, strategy: Optional[InferenceStrategy] = None):
        """
        Args:
            strategy: 推理策略，如果为None则需要在infer时指定
        """
        self.strategy = strategy
    
    def infer(self, 
              image: np.ndarray, 
              inference_fn: Callable[[np.ndarray], Dict[str, Any]],
              strategy: Optional[InferenceStrategy] = None,
              **kwargs) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            image: 输入图像
            inference_fn: 推理函数
            strategy: 可选的推理策略（覆盖初始化时的策略）
            **kwargs: 额外参数
            
        Returns:
            推理结果
        """
        # 使用指定的策略或默认策略
        active_strategy = strategy or self.strategy
        if active_strategy is None:
            return {"status": "error", "message": "No inference strategy specified"}
        
        try:
            # 1. 创建上下文
            context = InferenceContext(image)
            
            # 2. 准备阶段：生成处理单元（ROI或切片）
            if not active_strategy.prepare(context, **kwargs):
                return {"status": "error", "message": f"Failed to prepare {active_strategy.name} inference"}
            
            # 3. 推理阶段：对每个处理单元执行推理
            raw_results = active_strategy.infer(context, inference_fn)
            
            # 检查原始结果中是否有错误
            for result in raw_results:
                if result.get('status') == 'error':
                    return {
                        "status": "error",
                        "message": result.get('message', 'Inference failed'),
                        "strategy": active_strategy.name
                    }
            
            # 4. 后处理阶段：结果映射或合并
            annotations = active_strategy.postprocess(context, raw_results, **kwargs)
            
            return {
                "status": "success",
                "annotations": annotations,
                "strategy": active_strategy.name
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}


# 便捷函数
def create_inference_engine(strategy: Optional[InferenceStrategy] = None) -> InferenceEngine:
    if strategy is None:
        strategy = StandardInferenceStrategy()
    """创建推理引擎实例"""
    return InferenceEngine(strategy)


def create_strategy(mode: str, **kwargs) -> InferenceStrategy:
    """创建推理策略"""
    return StrategyFactory.create_strategy(mode, **kwargs)
