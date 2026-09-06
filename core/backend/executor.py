"""
统一推理执行器
将"读图→构造策略→引擎执行→异常包装"合并为单次调用
"""
import traceback
from typing import Callable, Optional, Any, Dict

from core.common.image_utils import imread_unicode
from core.backend.strategy_utils import build_inference_strategy
from core.backend.inference_engine import create_inference_engine


def run_inference(
    image_path: str,
    inference_fn: Callable,
    use_roi: bool = True,
    use_slice: bool = True,
    img_size: int = None,
    roi: Any = None
) -> Dict[str, Any]:
    """
    统一推理执行入口

    自动完成：读图 → 构造策略 → 引擎执行 → 异常包装

    Args:
        image_path: 图像路径
        inference_fn: 推理函数，签名为 inference_fn(numpy_image) -> result_dict
                      调用者只需关注"对图像做什么推理"
        use_roi: 是否启用ROI策略
        use_slice: 是否启用切片策略
        img_size: 推理图像尺寸
        roi: 显式 ROI [x,y,w,h]；为 None 时回退 project_settings 中的项目 ROI。
             用于示例模式：查询图应裁剪到示例自身记录的 ROI，保证与示例裁剪图尺度一致。

    Returns:
        成功返回推理结果，失败返回 {"status": "error", "message": ...}
    """
    try:
        # 1. 读取图像
        image = imread_unicode(image_path)
        if image is None:
            return {"status": "error", "message": "Failed to read image"}

        # 2. 构造推理策略
        strategy = build_inference_strategy(
            use_roi=use_roi,
            use_slice=use_slice,
            img_size=img_size,
            roi=roi,
        )

        # 3. 创建引擎并执行
        engine = create_inference_engine(strategy)
        result = engine.infer(image, inference_fn)

        return result

    except Exception as e:
        from core.common.settings import settings
        if settings.get("DEBUG", False):
            raise
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def run_inference_with_image(
    image,
    inference_fn: Callable,
    use_roi: bool = True,
    use_slice: bool = True,
    img_size: int = None,
    roi: Any = None
) -> Dict[str, Any]:
    """
    统一推理执行入口（图像已加载）

    与 run_inference 相同，但跳过图像读取步骤，
    适用于图像已在内存中的场景（如交互式标注）

    Args:
        image: numpy 图像数组
        inference_fn: 推理函数
        use_roi/use_slice/img_size: 策略参数
        roi: 显式 ROI [x,y,w,h]；为 None 时回退项目 ROI

    Returns:
        成功返回推理结果，失败返回 {"status": "error", "message": ...}
    """
    try:
        if image is None:
            return {"status": "error", "message": "Image is None"}

        strategy = build_inference_strategy(
            use_roi=use_roi,
            use_slice=use_slice,
            img_size=img_size,
            roi=roi,
        )

        engine = create_inference_engine(strategy)
        result = engine.infer(image, inference_fn)

        return result

    except Exception as e:
        from core.common.settings import settings
        if settings.get("DEBUG", False):
            raise
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
