from PySide6.QtGui import QColor

from ..services import CATEGORY_COLOR_PALETTE


def get_category_color(index_or_name):
    """根据索引或名称获取固定的类别颜色"""
    if isinstance(index_or_name, int):
        idx = index_or_name % len(CATEGORY_COLOR_PALETTE)
    else:
        idx = hash(index_or_name) % len(CATEGORY_COLOR_PALETTE)
    h, s, v = CATEGORY_COLOR_PALETTE[idx]
    return QColor.fromHsv(h, s, v)


__all__ = ["AnnotationInterface", "CATEGORY_COLOR_PALETTE", "get_category_color"]

from .annotation_interface import AnnotationInterface
