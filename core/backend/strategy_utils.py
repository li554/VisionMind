from core.common.settings import settings
from core.common.project_settings import project_settings
from .inference_engine import StandardInferenceStrategy, ROIInferenceStrategy, SliceInferenceStrategy
from .postprocess import PostprocessType, MatchMetric


def build_inference_strategy(use_roi=True, use_slice=True, img_size=None, roi=None):
    """
    从 core_settings 构造推理策略

    Args:
        use_roi: 是否启用ROI策略
        use_slice: 是否启用切片策略
        img_size: 推理图像尺寸
        roi: 显式 ROI [x,y,w,h]；为 None 时回退项目配置中的 ROI。
             用于示例模式：按示例自身记录的 ROI 裁剪查询图，保证与示例裁剪图尺度一致。

    Returns:
        InferenceStrategy 实例
    """
    # 检查切片推理是否启用
    slice_enabled = settings.get("slice_inference_enabled", False)

    if use_slice and slice_enabled:
        rows = settings.get("slice_rows", 2)
        cols = settings.get("slice_cols", 2)
        overlap_ratio = settings.get("slice_overlap_ratio", 0.2)

        # 后处理配置
        postprocess_type_str = settings.get("slice_postprocess_type", "GREEDYNMM")
        postprocess_type = PostprocessType[postprocess_type_str] if hasattr(PostprocessType, postprocess_type_str) else PostprocessType.GREEDYNMM

        match_metric_str = settings.get("slice_match_metric", "IOS")
        match_metric = MatchMetric[match_metric_str] if hasattr(MatchMetric, match_metric_str) else MatchMetric.IOS

        match_threshold = settings.get("slice_match_threshold", 0.5)

        return SliceInferenceStrategy(rows, cols, overlap_ratio, postprocess_type, match_threshold, match_metric)

    elif use_roi:
        roi = roi or project_settings.get("roi")
        if roi:
            return ROIInferenceStrategy(roi)
        else:
            return StandardInferenceStrategy()
    else:
        return StandardInferenceStrategy()
