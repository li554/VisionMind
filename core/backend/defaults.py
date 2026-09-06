"""
推理参数默认值填充
从 core_settings 读取默认值，填充未指定的参数，消除各调用点的重复代码
"""
from typing import Any, Dict, Optional


def _resolve_refine_params() -> Dict[str, Any]:
    """从 core_settings 读取细化模型参数"""
    from core.common.settings import settings
    return {
        "refine_method": settings.get("refine_method", "vitmatte"),
        "use_onnx": settings.get("refine_use_onnx", True),
        "fast": settings.get("refine_fast", False),
        "L": settings.get("refine_image_size", 896),
        "trimap_thickness": settings.get("refine_trimap_thickness", 20),
        "use_roi": settings.get("refine_use_roi", True),
        "margin": settings.get("refine_roi_margin", 0.15),
    }


def resolve_example_params(
    model_type: str = None,
    refine: bool = None,
    smooth: bool = None,
    smooth_epsilon: float = None,
    img_size: int = None
) -> Dict[str, Any]:
    """
    填充自动标注推理参数的默认值（从 core_settings 读取）

    Args:
        model_type: 模型类型，None 则读 settings
        refine: 是否精细化，None 则读 settings
        smooth: 是否平滑，None 则读 settings
        smooth_epsilon: 平滑精度，None 则读 settings
        img_size: 推理图像尺寸，None 则读 settings

    Returns:
        参数字典，可直接 **kwargs 传给 example_predict
    """
    from core.common.settings import settings

    params = {
        "model_type": model_type or settings.get("example_model", "trtsam3"),
        "refine": refine if refine is not None else settings.get("high_precision", True),
        "smooth": smooth if smooth is not None else settings.get("mask_smooth_enabled", True),
        "smooth_epsilon": smooth_epsilon if smooth_epsilon is not None else settings.get("mask_smooth_epsilon", 0.001),
        "img_size": img_size or settings.get("example_img_size", 644),
    }
    params.update(_resolve_refine_params())
    return params


def resolve_interactive_params(
    model_type: str = None,
    refine: bool = None,
    smooth: bool = None,
    smooth_epsilon: float = None,
    img_size: int = None
) -> Dict[str, Any]:
    """
    填充交互式推理参数的默认值（从 core_settings 读取）

    Args:
        同 resolve_example_params

    Returns:
        参数字典
    """
    from core.common.settings import settings

    return {
        "model_type": model_type or settings.get("default_interactive_model", "mobilesam_onnx"),
        "refine": refine if refine is not None else settings.get("high_precision", True),
        "smooth": smooth if smooth is not None else settings.get("mask_smooth_enabled", True),
        "smooth_epsilon": smooth_epsilon if smooth_epsilon is not None else settings.get("mask_smooth_epsilon", 0.001),
        "img_size": img_size or settings.get("default_interactive_img_size", 1024),
        **_resolve_refine_params(),
    }
