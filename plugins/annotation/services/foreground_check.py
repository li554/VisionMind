"""标注贴合度检查工具 — 经典CV前景提取 + 贴合度指标（后端实现转发）

算法实现已迁移至 core.backend.foreground_check，本模块仅为插件层兼容转发，
保持 `from .foreground_check import ...` 的既有导入路径可用。
"""

from core.backend.foreground_check import (
    SMALL_BOX_THRESHOLD,
    ExtractResult,
    ForegroundExtractor,
    choose_scale,
    upscale_patch,
    crop_roi,
    traditional_mask,
    fit_metrics,
    boundary_gradient,
    mask_to_polygon,
    mask_to_bbox,
)
