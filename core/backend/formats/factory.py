import os
from typing import List, Dict, Any, Optional, Type
from .base import AnnotationFormat, AnnotationData, SaveData
from .labelme_format import LabelMeFormat
from .coco_format import COCOFormat
from .yolo_format import YOLOFormat
from .voc_format import VOCFormat
from .mask_format import MaskFormat
from .sa1b_format import SA1BFormat


FORMAT_HANDLERS: List[Type[AnnotationFormat]] = [
    LabelMeFormat,
    COCOFormat,
    YOLOFormat,
    VOCFormat,
    MaskFormat,
    SA1BFormat,
]


def get_format(file_path: str) -> Optional[Type[AnnotationFormat]]:
    """
    根据文件路径自动检测标注格式

    Args:
        file_path: 标注文件路径

    Returns:
        对应的格式处理器类，如果无法识别则返回 None
    """
    for handler in FORMAT_HANDLERS:
        if handler.can_load(file_path):
            return handler
    return None


def load_annotations(file_path: str, **kwargs) -> AnnotationData:
    """
    自动检测格式并加载标注

    Args:
        file_path: 标注文件路径
        **kwargs: 额外参数
            - image_path: 图像文件路径（用于获取图像尺寸和匹配）

    Returns:
        AnnotationData: 统一格式的标注数据
        {
            "annotations": [...],
            "image_info": {"filename": str, "width": int, "height": int},
            "categories": [{"id": int, "name": str}, ...]
        }
    """
    handler = get_format(file_path)
    if handler:
        return handler.load(file_path, **kwargs)

    ext = os.path.splitext(file_path)[1].lower()
    print(f"[factory] Unsupported format: {ext}")
    return {"annotations": [], "image_info": {}, "categories": []}


def save_annotations(
    data: SaveData,
    output_path: str,
    format_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    保存标注到文件

    Args:
        data: 统一格式的保存数据
        {
            "annotations": [...],
            "image_info": {"filename": str, "width": int, "height": int},
            "categories": [{"id": int, "name": str}, ...]
        }
        output_path: 输出文件路径
        format_type: 指定格式类型 ('labelme', 'coco', 'yolo', 'yoloseg', 'yoloobb', 'voc', 'mask', 'sa1b')
        **kwargs: 额外参数
            - mode: 任务模式 ('det', 'seg', 'obb')，用于YOLO格式
            - mask_type: Mask类型 ('grayscale', 'colored', 'binary', 'both')
    """
    handler = _get_handler(output_path, format_type)

    if handler:
        handler.save(data, output_path, **kwargs)
    else:
        print(f"[factory] Cannot determine format for: {output_path}")


def update_category(dir_path: str, old_label: str, new_label: str) -> int:
    """
    更新目录中所有格式的类别名称

    Args:
        dir_path: 目录路径
        old_label: 旧类别名称
        new_label: 新类别名称

    Returns:
        处理的文件数量
    """
    total_processed = 0
    for handler in FORMAT_HANDLERS:
        total_processed += handler.update_category(dir_path, old_label, new_label)
    return total_processed


def remove_category(dir_path: str, category_name: str) -> int:
    """
    从目录中删除所有格式的指定类别标注

    Args:
        dir_path: 目录路径
        category_name: 要删除的类别名称

    Returns:
        处理的文件数量
    """
    total_processed = 0
    for handler in FORMAT_HANDLERS:
        total_processed += handler.remove_category(dir_path, category_name)
    return total_processed


def scan_categories(dir_path: str, format_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    扫描目录中的类别信息

    Args:
        dir_path: 目录路径
        format_type: 指定格式类型，如果为None则扫描所有格式

    Returns:
        类别列表 [{"id": int, "name": str}, ...]
    """
    if format_type:
        handler = _get_handler("", format_type)
        if handler:
            return handler.scan_categories(dir_path)
        return []
    
    all_categories = {}
    for handler in FORMAT_HANDLERS:
        categories = handler.scan_categories(dir_path)
        for cat in categories:
            name = cat.get("name", "")
            if name and name not in all_categories:
                all_categories[name] = cat
    
    return list(all_categories.values())


def _get_handler(output_path: str, format_type: Optional[str] = None) -> Optional[Type[AnnotationFormat]]:
    """获取格式处理器"""
    if format_type:
        format_map = {
            'labelme': LabelMeFormat,
            'json': LabelMeFormat,
            'coco': COCOFormat,
            'yolo': YOLOFormat,
            'yoloseg': YOLOFormat,
            'yoloobb': YOLOFormat,
            'txt': YOLOFormat,
            'voc': VOCFormat,
            'xml': VOCFormat,
            'mask': MaskFormat,
            'png': MaskFormat,
            'sa1b': SA1BFormat,
        }
        return format_map.get(format_type.lower())

    return get_format(output_path)
