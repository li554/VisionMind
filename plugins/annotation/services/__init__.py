"""
标注服务工具函数 — 纯数据转换，不依赖类状态
"""

import numpy as np

# 类别颜色方案（HSV 元组列表，纯数据，供 Service 与 View 共同使用）
CATEGORY_COLOR_PALETTE = [
    (0, 200, 220),      # 红色
    (30, 200, 220),     # 橙色
    (60, 200, 220),     # 黄色
    (90, 200, 220),     # 绿色
    (120, 200, 220),    # 青色
    (150, 200, 220),    # 蓝色
    (180, 200, 220),    # 紫色
    (210, 200, 220),    # 粉色
    (240, 200, 220),    # 玫红
    (270, 200, 220),    # 深紫
    (300, 200, 220),    # 品红
    (330, 200, 220),    # 桃红
    (15, 200, 220),     # 金橙
    (45, 200, 220),     # 黄绿
    (75, 200, 220),     # 草绿
    (105, 200, 220),    # 青绿
    (135, 200, 220),    # 天蓝
    (165, 200, 220),    # 海蓝
    (195, 200, 220),    # 靛蓝
    (225, 200, 220),    # 紫罗兰
    (255, 200, 220),    # 紫红
    (285, 200, 220),    # 洋红
    (315, 200, 220),    # 玫瑰
    (345, 200, 220),    # 珊瑚
]


def _convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    return obj
