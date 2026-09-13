"""ManualAnnotationService — 手动标注领域服务

职责: 图片导航、文件列表、图片导航状态(原 NavigationVM)、类别管理、标注编辑、
     筛选审查、难样本、分割持久化、非项目配置、副标注加载、格式检测、目录扫描、
     搜索加载、IoU/匹配/标注构建算法
AI action: 本服务方法上的 @action(..., scope="agent") 为手动标注的核心逻辑
依赖: ProjectContext（共享状态）
"""

import copy
import hashlib
import json
import os
import re
import time
import cv2
import numpy as np
from PySide6.QtGui import QColor
from core.backend.core import interactive_predict
from core.backend.defaults import resolve_interactive_params
from core.backend.formats import load_annotations, save_annotations, update_category, remove_category
from core.backend.utils import calculate_iou
from core.common.action_registry import action
from core.common.image_utils import imread_unicode
from core.common.project_settings import (project_settings, RULES_SCHEMA_VERSION,
                                          migrate_legacy_rules)
from core.common.settings import settings
from core.event_bus import StateType
from core.common.atomic_io import atomic_open
from .annotation_writer import AnnotationWriter
from .foreground_check import (
    SMALL_BOX_THRESHOLD, traditional_mask, fit_metrics, boundary_gradient,
    mask_to_polygon, mask_to_bbox,
)
from .project_context import ProjectContext
from ..session import ImageSession


def _remove_with_retry(path, attempts=4, delay_s=0.03):
    """删除文件，遇 Windows 句柄占用时做**有界**短暂重试。

    Windows 上 `os.remove` 在文件仍被其它句柄占用时抛
    `PermissionError: [WinError 32] 另一个程序正在使用此文件`。本应用的图像解码
    （QImageReader 按路径打开文件）在解码期间会短暂持有句柄，而"打开目录"会为
    所有可见行一次性排入缩略图任务 —— 于是"刚打开目录就删图"原本必定失败
    （已用 scripts/_probe_delete_lock.py 实测：此刻缩略图池活跃 3、解码池活跃 1，
    删除失败；约 200ms 后成功）。

    重试上限刻意很小（默认 4 次 × 30ms ≈ 90ms），避免在 GUI 线程上制造可见卡顿；
    真正的长时间占用由调用方在删除前 `wait_idle()` 解决（见 interface.delete_image_file）。
    """
    last = None
    for i in range(max(1, attempts)):
        try:
            os.remove(path)
            return
        except PermissionError as e:      # 只对句柄占用重试
            last = e
            if i + 1 < attempts:
                time.sleep(delay_s)
        except FileNotFoundError:
            return                        # 已经不在了，视为成功
    raise last


class ManualAnnotationService:
    """手动标注领域服务：图片导航、类别、编辑、筛选、难样本、分割、格式检测/扫描/算法"""

    def __init__(self, context=None, project_service=None):
        self.context = context or ProjectContext()
        self.project_service = project_service
        self.auto = None  # AutoAnnotationService 交叉引用（Phase 2 注入）
        self._event_bus = None  # 事件总线，由 AnnotationPlugin.on_load 注入

        # ====== 状态变量: 文件导航 ======
        # 「当前图片 + 该图标注 + 已提交像素与度量」的唯一所有者。原先 service /
        # interface / widget 各存一份 current_image_path，再叠加 current_annotations
        # 的别名，使 "(图 N-1 的度量) + (图 N 的标注)" 这类不自洽状态可以在三处之间
        # 出现，并被任何一次自动保存写进磁盘（审计 W2）。现在真值只在 session 里。
        # 注意：service 是普通 Python 对象，不能当 QObject 的 parent。
        # session 由 self.session 强引用保活。
        self.session = ImageSession()
        # 标注写入的唯一出口：串行 + 同路径最新优先 + 可等待（"IO 全在后台"的最后一块）。
        # 生命周期由本 service 显式管理（drain() 排空并停止写入线程）。
        self._writer = AnnotationWriter()
        # 当前加载的图片文件列表、当前目录
        self.image_files = []
        self.current_dir = None
        self.current_dir_hash = ""

        # ====== 状态变量: 数据集划分 & 标注缓存 ======
        # image_path -> "train"|"val"|"test"；annotation_cache 加速筛选
        self.image_splits = {}
        self.annotation_cache = {}
        # image_path -> (width, height)，避免重复读图获取尺寸
        self._image_size_cache = {}

        # ====== 状态变量: 筛选审查 ======
        # 类别/尺寸/宽高筛选条件（None 表示未启用）
        self.filter_category = None
        self.filter_size_range = None
        self.filter_width_range = None
        self.filter_height_range = None
        # 标注状态筛选：None=全部，True=仅已标注，False=仅未标注
        self.filter_annotated = None
        # 数据集划分筛选：None=全部，train/val/test 之一，"" 表示「未划分」
        self.filter_split = None
        # 图片搜索/筛选后的可见性列表 [(file_path, hidden)]，供界面刷新时重新应用
        self.file_visibility = []

        # ====== 状态变量: 标注可见性 ======
        # 隐藏的标注索引集合（当前图片）。**权威源已并入 session**（审计收尾），
        # 这里保留同名 property 以便既有调用点继续工作，不再各存一份。
        # hidden_indices 由 visibility action 维护，界面经
        # annotation:annotations_changed 信号刷新；切图时由 session.begin_load 清空。

        # ====== 状态变量: 撤销/重做 ======
        # 撤销/重做栈同样并入 session（见下），_max_undo_steps 是策略不是状态，
        # 留在 service。
        self._max_undo_steps = 50
        self._is_undo_redo = False

        # ====== 状态变量: 副标注 & 参考标注 & 编辑器状态 ======
        # 副标注（secondary_*）是**按图作用域**的，权威源也已并入 session；
        # 目录/来源等"配置类"状态仍留在 service。
        self.reference_annotation = None
        self.secondary_loaded = False
        self.secondary_dir = None
        self._last_sam_annotation = None
        self.example_mode_enabled = False
        self._sam_active = False

        # ====== 状态变量: COCO 保存缓存 ======
        # save_image_annotations 保存 coco 格式时增量合并的缓存
        self._coco_cache = {}
        self._coco_cache_path = None

    # ====== 区域: 标注 I/O ======
    # 原 annotation_io: save_image_annotations / load_image_annotations / _collect_annotation_dirs

    def save_image_annotations(self, image_path, annotations, fmt="labelme",
                                output_dir=None, categories=None, project_mode=True):
        """保存图片标注到指定格式"""
        try:
            if not output_dir:
                output_dir = settings.get("output_dir", "")
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            if not annotations:
                fmt_config = settings.get("annotation_formats", {})
                if project_mode:
                    ann_path = os.path.join(output_dir, f"{base_name}.json")
                else:
                    cfg = fmt_config.get(fmt, {})
                    ext = cfg.get("ext", ".json")
                    ann_path = os.path.join(output_dir, f"{base_name}{ext}")
                if os.path.exists(ann_path):
                    os.remove(ann_path)
                return {"status": "success", "message": "Annotations cleared", "output_path": output_dir}

            os.makedirs(output_dir, exist_ok=True)
            image = imread_unicode(image_path)
            if image is None:
                return {"status": "error", "message": "Failed to read image"}
            h, w = image.shape[:2]

            id_to_name, name_to_id = self._normalize_categories(categories)

            instances = []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                label = ann.get('label', 'unknown')
                class_id = name_to_id.get(label, 0)
                if 'bbox' not in ann:
                    continue
                inst = {'bbox': ann['bbox'], 'polygons': ann.get('polygons', []),
                        'label': label, 'class_id': class_id}
                if 'shape_type' in ann:
                    inst['shape_type'] = ann['shape_type']
                instances.append(inst)

            save_data = {
                "annotations": instances,
                "image_info": {"filename": os.path.basename(image_path), "width": w, "height": h},
                "categories": [{"id": id, "name": name} for name, id in name_to_id.items()],
            }

            if project_mode:
                save_annotations(save_data, os.path.join(output_dir, f"{base_name}.json"), format_type="labelme")
                saved_path = os.path.join(output_dir, f"{base_name}.json")
            else:
                save_kwargs = {"mode": "det" if fmt in ["yolo", "voc"] else ("seg" if fmt == "yoloseg" else "obb"),
                               "mask_type": "both"}
                if fmt == "labelme":
                    save_annotations(save_data, os.path.join(output_dir, f"{base_name}.json"), format_type="labelme")
                    saved_path = os.path.join(output_dir, f"{base_name}.json")
                elif fmt == "coco":
                    fmt_config = settings.get("annotation_formats", {})
                    coco_cfg = fmt_config.get("coco", {"ext": ".json", "filename": "_annotations.coco.json"})
                    coco_filename = coco_cfg.get("filename", "_annotations.coco.json")
                    output_path = os.path.join(output_dir, coco_filename)
                    save_annotations(save_data, output_path, format_type="coco",
                                   coco_cache=self._coco_cache,
                                   coco_cache_path=self._coco_cache_path, **save_kwargs)
                    self._coco_cache_path = output_path
                    saved_path = output_path
                elif fmt in ("yolo", "yoloseg", "yoloobb"):
                    saved_path = os.path.join(output_dir, f"{base_name}.txt")
                    save_annotations(save_data, saved_path,
                                   format_type=fmt, **save_kwargs)
                else:
                    # 兜底分支必须按**格式自身的约定**决定落盘路径与扩展名
                    # （审计 D35）：原先硬编码 `<base>.json`，而 MaskFormat.save 写的是
                    # 图像（扩展名直接取自 output_path 交给 OpenCV），于是
                    # `imwrite_unicode(<...>.json.atomic-.., ext='.json')` 必然失败：
                    #   OpenCV: could not find encoder for the specified extension
                    # 也就是非项目模式下 mask/sa1b 保存 100% 失败（异常还被本函数的
                    # except 吞成 {"status":"error"}）。
                    target_dir, filename = self._resolve_annotation_output(
                        fmt, output_dir, base_name)
                    os.makedirs(target_dir, exist_ok=True)
                    saved_path = os.path.join(target_dir, filename)
                    save_annotations(save_data, saved_path, format_type=fmt, **save_kwargs)
            saved_summary = [{"label": i.get("label"), "bbox": i.get("bbox")} for i in instances]
            return {"status": "success", "output_path": output_dir, "file_path": saved_path,
                    "annotation_count": len(instances), "annotations": saved_summary}
        except Exception as e:
            if settings.DEBUG:
                raise
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _resolve_annotation_output(fmt, output_dir, base_name):
        """按格式自身的约定解析 (目标目录, 文件名) —— 审计 D35。

        每个格式"写什么文件、放哪个子目录"是与它自己的加载实现绑定的事实，因此
        这里从格式类读取，而不是在调用点猜：

          * `MaskFormat.EXTENSIONS = ['.png', ...]` 且约定放在 `masks/` 子目录
            （见 `MaskFormat.scan_categories` 的 `mask_subdirs = ['masks']`）——
            若把掩码写成 `<base>.json`，OpenCV 会因找不到编码器直接失败；
            若写成 `<dir>/<base>.png`，还会与**源图同名同目录**而覆盖别人的图。
          * `SA1BFormat.EXTENSIONS = ['.json']`，按 base 命名。
          * 其余（labelme/voc/未知）沿用 `<base>.json`。

        返回 (dir, filename)。
        """
        fmt_key = (fmt or "").lower()
        if fmt_key == "mask":
            try:
                from core.backend.formats.mask_format import MaskFormat
                ext = MaskFormat.EXTENSIONS[0]
            except Exception:
                ext = ".png"
            return os.path.join(output_dir, "masks"), f"{base_name}{ext}"
        if fmt_key == "sa1b":
            try:
                from core.backend.formats.sa1b_format import SA1BFormat
                ext = SA1BFormat.EXTENSIONS[0]
            except Exception:
                ext = ".json"
            return output_dir, f"{base_name}{ext}"
        # 项目内配置优先（与 load_image_annotations 用的是同一份约定）
        cfg = (settings.get("annotation_formats", {}) or {}).get(fmt_key) or {}
        ext = cfg.get("ext", ".json")
        filename = cfg.get("filename") or f"{base_name}{ext}"
        subdir = cfg.get("subdir", "") or ""
        return (os.path.join(output_dir, subdir) if subdir else output_dir), filename

    @staticmethod
    def _annotation_output_candidates(fmt, output_dir, base_name):
        """返回该格式**可能**的落盘路径候选（读取端按顺序探测）。

        与 `_resolve_annotation_output` 共用同一份约定，并额外覆盖"旧版本/项目配置"
        可能留下的路径，保证写入与读取不会漂移（审计 D35：写完读不回来同样是缺陷）。
        """
        primary_dir, primary_name = ManualAnnotationService._resolve_annotation_output(
            fmt, output_dir, base_name)
        candidates = [os.path.join(primary_dir, primary_name)]
        fmt_key = (fmt or "").lower()
        if fmt_key == "mask":
            # 兼容：旧版本把掩码直接放在目录根、以及把掩码误写成 .json 的历史情况
            for ext in ("",):
                candidates.append(os.path.join(output_dir, f"{base_name}{ext or '.png'}"))
        if fmt_key == "coco":
            cfg = (settings.get("annotation_formats", {}) or {}).get("coco") or {}
            candidates.append(os.path.join(
                output_dir, cfg.get("filename", "_annotations.coco.json")))
        return candidates

    def load_image_annotations(self, image_path, output_dir=None, fmt="mask"):
        """加载图片标注数据"""
        try:
            if not output_dir:
                output_dir = settings.get("output_dir", "")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            formats_to_try = ["labelme", "coco", "yolo", "voc", "mask"] if fmt == "all" else [fmt]
            for try_fmt in formats_to_try:
                # 与写入端共用同一个路径解析（审计 D35）：原先这里依赖
                # `settings["annotation_formats"]`，而该配置默认为空 -> mask 等格式被
                # 直接 `continue` 跳过，于是"写进去了却永远读不回来"。写入端已改为按
                # 格式类自身的约定解析，读取端必须用同一份约定，否则两边会再次漂移。
                candidates = self._annotation_output_candidates(try_fmt, output_dir, base_name)
                file_path = None
                for candidate in candidates:
                    if os.path.exists(candidate):
                        file_path = candidate
                        break
                if not file_path:
                    continue
                load_kwargs = {"image_path": image_path} if try_fmt in ("coco", "yolo", "yoloseg", "yoloobb") else {}
                result = load_annotations(file_path, **load_kwargs)
                if result and result.get("annotations"):
                    return {'status': 'success', 'annotations': result["annotations"],
                            'image_info': result.get("image_info", {}),
                            'categories': result.get("categories", []), 'format': try_fmt}
            return {'status': 'success', 'annotations': [], 'image_info': {}, 'categories': [], 'format': 'unknown'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'annotations': [], 'image_info': {}, 'categories': []}

    def _collect_annotation_dirs(self, output_dir, include_task_subdirs=False):
        """收集输出目录下所有可能包含标注文件的目录"""
        dirs_to_process = [output_dir]
        for subdir_name in ['seg', 'det', 'obb']:
            task_dir = os.path.join(output_dir, subdir_name)
            if os.path.exists(task_dir):
                dirs_to_process.append(task_dir)
                if include_task_subdirs:
                    for sub in ['labels', 'annotations']:
                        sd = os.path.join(task_dir, sub)
                        if os.path.exists(sd):
                            dirs_to_process.append(sd)
        for subdir_name in ['annotations', 'labels', 'coco_annotations', 'masks', 'voc_annotations']:
            sd = os.path.join(output_dir, subdir_name)
            if os.path.exists(sd):
                dirs_to_process.append(sd)
        parent_dir = os.path.dirname(output_dir)
        if parent_dir and parent_dir != output_dir:
            if os.path.exists(os.path.join(parent_dir, "classes.txt")):
                dirs_to_process.append(parent_dir)
        versions_dir = os.path.join(output_dir, "versions")
        if os.path.exists(versions_dir):
            for version in os.listdir(versions_dir):
                vdir = os.path.join(versions_dir, version)
                if os.path.isdir(vdir):
                    for split in ['train', 'val', 'test']:
                        sdir = os.path.join(vdir, split)
                        if os.path.exists(sdir):
                            if include_task_subdirs:
                                for sub in ['labels']:
                                    sd = os.path.join(sdir, sub)
                                    if os.path.exists(sd):
                                        dirs_to_process.append(sd)
                            dirs_to_process.append(sdir)
                    dirs_to_process.append(vdir)
        return dirs_to_process

    def batch_remove_category(self, output_dir, category_name):
        """从所有标注文件中批量移除指定类别"""
        try:
            total = 0
            for d in self._collect_annotation_dirs(output_dir, include_task_subdirs=False):
                total += remove_category(d, category_name)
            return {'status': 'success', 'count': total}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def batch_update_category_name(self, output_dir, old_name, new_name):
        """批量更新所有标注文件中的类别名称"""
        try:
            total = 0
            for d in self._collect_annotation_dirs(output_dir, include_task_subdirs=True):
                total += update_category(d, old_name, new_name)
            return {'status': 'success', 'count': total}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # ====== 区域: 图片导航 & 文件列表 ======
    # 原 annotation_io: change_path / scan_directory / delete_image / scan_image_files / validate_label_file

    def change_path(self, dir_path):
        from core.common.settings import settings
        settings.set("output_dir", dir_path)
        return {"output_dir": dir_path, "status": "success"}

    def scan_directory(self, dir_path):
        import glob
        result = {"dir_path": dir_path, "detected_format": None,
                  "detected_categories": [], "detected_task_type": None,
                  "has_images": False, "image_files": []}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
            result["image_files"].extend(glob.glob(os.path.join(dir_path, ext)))
            result["image_files"].extend(glob.glob(os.path.join(dir_path, "images", ext)))
        result["has_images"] = len(result["image_files"]) > 0
        detected_format, detected_categories, detected_task_type = self.auto_detect_format(dir_path)
        result["detected_format"] = detected_format
        result["detected_categories"] = detected_categories
        result["detected_task_type"] = detected_task_type
        return result

    def delete_image(self, image_path):
        """物理删除图片及其**专属**标注与缓存。

        修复的两处缺陷（审计 D1）：
        1. 原先的 `from core.backend.formats import find_label_file` 必定 ImportError
           （该符号只从 `core.backend.utils` 导出），而它在 `os.remove(image_path)`
           **之后** -> 图片已被删除、标注留在原地、调用方还拿到 success=False
           （于是 `delete_image_file` 直接 return error，列表里残留失效条目）。
        2. `find_label_file` 可能返回**共享**标注文件（如 `_annotations.coco.json`
           或共享掩码），直接 `os.remove` 会连带毁掉整个数据集里其它图片的标注。

        因此改为两阶段：先**只解析**待删清单（任一环节失败则一个文件都不删），
        再逐个删除并逐个记录；共享文件只报告、不删除。
        """
        result = {"success": True, "deleted_files": [], "errors": [], "warnings": []}
        if not os.path.exists(image_path):
            return {"success": False, "deleted_files": [], "warnings": [],
                    "errors": ["文件不存在: " + image_path]}

        # ---- 阶段一：解析（不产生任何副作用）----
        targets = [image_path]
        try:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            from core.backend.utils import find_label_file
            ann_path = find_label_file(image_path)
            if ann_path and os.path.exists(ann_path):
                if os.path.splitext(os.path.basename(ann_path))[0] == stem:
                    targets.append(ann_path)
                else:
                    result["warnings"].append(
                        "标注位于共享文件，已保留未删除（需按格式移除该图条目）: " + ann_path)
            base = os.path.splitext(image_path)[0]
            for ext in ['.npy', '.png', '.jpg', '.json']:
                cache_path = f"{base}_cache{ext}"
                if os.path.exists(cache_path):
                    targets.append(cache_path)
        except Exception as e:
            # 解析失败 -> 一个文件都不删，保证磁盘状态与调用方认知一致
            return {"success": False, "deleted_files": [], "warnings": [],
                    "errors": ["解析关联文件失败，未删除任何文件: " + str(e)]}

        # ---- 阶段二：先删图片本身 ----
        # 顺序很重要：图片是主对象。若它删不掉（Windows 上解码进行中会
        # WinError 32），就**不能**继续删标注/缓存 —— 否则会留下
        # "标注已删、图片还在"的反向孤儿状态（这是本测试实测到的情形）。
        try:
            _remove_with_retry(image_path)
            result["deleted_files"].append(image_path)
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"{image_path}: {e}")
            return result

        # ---- 阶段三：清理已解析出的关联文件（失败只记录，不回滚图片）----
        for target in targets[1:]:
            try:
                _remove_with_retry(target)
                result["deleted_files"].append(target)
            except Exception as e:
                result["success"] = False
                result["errors"].append(f"{target}: {e}")
        return result

    def scan_image_files(self, dir_path):
        all_files = []
        is_project = any(os.path.exists(os.path.join(dir_path, d)) for d in ['annotations', 'images'])
        if is_project:
            root_img_dir = os.path.join(dir_path, "images")
            if os.path.exists(root_img_dir):
                all_files = sorted([os.path.join(root_img_dir, f) for f in os.listdir(root_img_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        if not all_files:
            all_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
            if not all_files:
                all_files = sorted([os.path.join(root, f) for root, dirs, files in os.walk(dir_path)
                                    for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        return all_files

    def validate_label_file(self, file_path, mode):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    n = len(parts)
                    if mode == "det" and n == 5:
                        return True
                    elif mode == "seg" and n >= 11:
                        return True
                    elif mode == "obb" and n == 9:
                        return True
            return False
        except Exception:
            return False

    # ====== 区域: 图片导航 & 文件列表状态（原 NavigationVM）======
    # 原 navigation_view_model: set_image_files / set_current_image / set_image_split /
    #   set_filter (统一入口，mode=category/preset/custom_size/custom_wh) /
    #   clear_filters / clear / load_directory / save_last_index /
    #   load_annotations_for_image / delete_image_file / refresh_directory /
    #   set_current_image_with_split / clear_all / get_cached_annotations /
    #   search_images / navigate_relative

    # ====== 状态所有权：只读投影（真值在 self.session）======
    # 这两个 property **没有 setter**：任何 `self.current_image_path = ...` /
    # `self.current_annotations = ...` 都会立刻 AttributeError，而不是悄悄留下
    # 第二份真值。改写必须走 session.begin_load() / session.set_annotations()。

    @property
    def current_image_path(self):
        return self.session.path

    @property
    def current_annotations(self):
        return self.session.annotations

    # ---- 以下四项原先各有一份 service 侧副本，现统一委托给 session ----
    # 保留 property 是为了让既有调用点（约 40 处）无需改动，同时**消灭第二份真值**：
    # 任何 `self.hidden_indices = ...` / `self.secondary_annotations = ...` 都会写进
    # session，而不是在 service 上另建一个副本（那正是"两份状态不同步"的来源）。
    # 读 `self._undo_stack` 的地方一律经 undo_stack 属性，见下方说明。

    @property
    def hidden_indices(self):
        return self.session.hidden_indices

    @hidden_indices.setter
    def hidden_indices(self, value):
        self.session.hidden_indices = value

    @property
    def secondary_annotations(self):
        return self.session.secondary_annotations

    @secondary_annotations.setter
    def secondary_annotations(self, value):
        self.session.secondary_annotations = value

    @property
    def secondary_annotation_map(self):
        return self.session.secondary_annotation_map

    @secondary_annotation_map.setter
    def secondary_annotation_map(self, value):
        self.session.secondary_annotation_map = value

    @property
    def _undo_stack(self):
        return self.session.undo_stack

    @_undo_stack.setter
    def _undo_stack(self, value):
        self.session._undo_stack = list(value or ())

    @property
    def _redo_stack(self):
        return self.session.redo_stack

    @_redo_stack.setter
    def _redo_stack(self, value):
        self.session._redo_stack = list(value or ())

    def set_image_files(self, files):
        """设置当前图片文件列表"""
        self.image_files = files
        self.file_visibility = []

    def set_current_image(self, path):
        """设置当前图片路径：走 session 的唯一入口（换代次 + 清空已提交像素）。"""
        self.session.begin_load(path)

    def set_image_split(self, image_path, split):
        """记录图片的数据集划分（train/val/test）"""
        self.image_splits[image_path] = split

    @action("manual.set_filter",
            description="统一筛选入口（mode 参数驱动）。\n"
                       "- mode=category: 按类别筛选，category 参数为类别名（传空字符串或 None 清除）\n"
                       "- mode=preset: 预设尺寸筛选，index 0=全部 1=小目标(<32²) 2=中目标(32²-96²) 3=大目标(>96²)\n"
                       "- mode=custom_size: 自定义面积筛选（像素²），min_size/max_size 为范围，0 表示不限\n"
                       "- mode=custom_wh: 自定义宽高筛选（像素），min_width/max_width/min_height/max_height 为范围\n"
                       "- mode=annotated: 按标注状态筛选，annotated=true 仅显示已标注图片，annotated=false 仅显示未标注图片，annotated=None/省略 清除状态筛选\n"
                       "- mode=split: 按数据集划分筛选，split=train/val/test 显示对应划分图片，split=空字符串 显示未划分图片，split=None/省略 清除划分筛选\n"
                       "- 筛选后可用 manual.get_state 查看筛选状态",
            category="标注",
            params={"mode": "str", "category": "str", "index": "int",
                    "min_size": "int", "max_size": "int",
                    "min_width": "int", "max_width": "int",
                    "min_height": "int", "max_height": "int",
                    "annotated": "bool", "split": "str"},
            scope="agent")
    def set_filter(self, mode="category", category=None, index=0,
                   min_size=0, max_size=0,
                   min_width=0, max_width=0, min_height=0, max_height=0,
                   annotated=None, split=None):
        """统一筛选入口 — 按 mode 分发到不同的筛选逻辑。

        不同 mode 对应不同的筛选应用函数（lambda 输入统一为无参闭包，
        仅内部筛选逻辑不同），全部内联在本函数内。返回格式随 mode 变化：
        - category:    {"status": "success", "filter": "category", "category": category}
        - preset:      (size_range, is_custom, is_wh)，index 0=全部 1=小 2=中 3=大 4=自定义面积 5=自定义宽高
        - custom_size: (size_range, error)，min/max 均为 0 时清除面积筛选，单个 0 表示不限上界
        - custom_wh:   (wh_range, error)，维度 min/max 均为 0 时清除该维度，单个 0 表示不限上界
        - split:       {"status":"success","filter":"split","split":split}，split 传 None 清除，
                        传 "" 表示「未划分」，train/val/test 表示对应数据集划分
        """
        def _preset_filter():
            ranges = {0: None, 1: (0, 32 * 32), 2: (32 * 32, 96 * 96),
                      3: (96 * 96, float('inf'))}
            if index in ranges:
                self.filter_size_range = ranges[index]
                return ranges[index], False, False
            if index == 4:
                return None, True, False
            if index == 5:
                return None, False, True
            return None, False, False

        def _custom_size_filter():
            size_range, error = self._normalize_range(min_size, max_size)
            if error:
                return None, error
            self.filter_size_range = size_range
            return size_range, None

        def _custom_wh_filter():
            w_range, w_error = self._normalize_range(min_width, max_width)
            h_range, h_error = self._normalize_range(min_height, max_height)
            error = w_error or h_error
            if error:
                return None, error
            self.filter_width_range = w_range
            self.filter_height_range = h_range
            return self.filter_width_range, None

        handlers = {
            "category": lambda: (
                setattr(self, "filter_category", category),
                {"status": "success", "filter": "category", "category": category},
            )[1],
            "annotated": lambda: (
                setattr(self, "filter_annotated", annotated),
                {"status": "success", "filter": "annotated", "annotated": annotated},
            )[1],
            "split": lambda: (
                setattr(self, "filter_split", split),
                {"status": "success", "filter": "split", "split": split},
            )[1],
            "preset": lambda: _preset_filter(),
            "custom_size": lambda: _custom_size_filter(),
            "custom_wh": lambda: _custom_wh_filter(),
        }
        handler = handlers.get(mode)
        if handler is None:
            return f"错误: 未知筛选模式 '{mode}'，可选: category/annotated/preset/custom_size/custom_wh/split"
        result = handler()
        # 仅在筛选成功落定时发布事件；失败/错误分支不发
        if mode in ("category", "annotated", "preset", "split"):
            self._publish_filters_changed()
        elif mode in ("custom_size", "custom_wh"):
            error = (result[1] if isinstance(result, tuple) and len(result) >= 2 else None)
            if not error:
                self._publish_filters_changed()
        return result

    def _normalize_range(self, min_val, max_val):
        """将 (min_val, max_val) 归一化为内部范围元组。返回 (range, error)。

        - 任一为负数 → (None, "错误: 数值不能为负数")
        - min=0 且 max=0 → (None, None)，表示清除该维度
        - 单个 0 表示该边界不限上界（归一化为 None）
        - 其余情况返回 ((min_val, effective_max), None)
        """
        if min_val < 0 or max_val < 0:
            return None, "错误: 数值不能为负数"
        if min_val == 0 and max_val == 0:
            return None, None
        effective_max = max_val if max_val > 0 else None
        if min_val > 0 and effective_max is not None and min_val > effective_max:
            return None, "错误: 最小值不能大于最大值"
        return (min_val, effective_max), None

    @action("manual.clear_filters",
            description="清除全部筛选条件（类别/面积/宽高/标注状态）。\n- 返回清除结果描述字符串（如「已清除全部筛选条件」/「当前没有筛选条件，无需清除」）",
            category="标注", scope="agent")
    def clear_filters(self):
        """清空全部筛选状态。返回清除结果描述字符串。"""
        had = bool(self.filter_category or self.filter_size_range or
                   self.filter_width_range or self.filter_height_range or
                   self.filter_annotated is not None or self.filter_split is not None)
        self.filter_category = None
        self.filter_size_range = None
        self.filter_width_range = None
        self.filter_height_range = None
        self.filter_annotated = None
        self.filter_split = None
        if had:
            self._publish_filters_changed()
            return "已清除全部筛选条件"
        return "当前没有筛选条件，无需清除"

    def clear(self):
        """清空导航状态（图片文件、当前图片、目录、划分、缓存）"""
        self.image_files = []
        self.session.clear()
        self.current_dir = None
        self.current_dir_hash = ""
        self.image_splits = {}
        self.annotation_cache = {}
        self._image_size_cache = {}
        self.file_visibility = []

    def load_directory(self, dir_path):
        """加载目录：设置当前目录、扫描图片、重置筛选与缓存。

        返回 (image_files, last_idx)。
        """
        self.current_dir = dir_path
        self.current_dir_hash = hashlib.md5(dir_path.encode()).hexdigest()[:8]
        self.filter_category = None
        self.filter_size_range = None
        self.filter_width_range = None
        self.filter_height_range = None
        self.filter_annotated = None
        self.filter_split = None
        self.annotation_cache.clear()
        files = self.scan_image_files(dir_path)
        self.set_image_files(files)
        last_idx = settings.get_session(f"last_idx_{self.current_dir_hash}", 0)
        settings.set_session("last_opened_dir", dir_path)
        return files, last_idx

    def save_last_index(self, index):
        """保存当前目录最近查看的图片索引。

        用 `set_session`：这是**运行时状态**，写进受版本控制的 `core/core.json` 会让
        每次切图都落一次盘、并让该文件永久 dirty（审计 D30）。
        """
        if self.current_dir:
            settings.set_session(f"last_idx_{self.current_dir_hash}", index)

    def load_annotations_for_image(self, image_path, mode, strict=False,
                                   output_dir="", is_non_project_mode=False,
                                   non_project_format=None):
        """加载图片标注（自动降级尝试所有格式）。

        返回 (res, detected_format)。detected_format 在非项目模式下自动识别到
        新格式时返回该格式，否则为 None。
        """
        fmt = non_project_format or "labelme" if is_non_project_mode else "labelme"
        res = self.find_and_load_annotations(
            image_path, mode, fmt,
            output_dir=output_dir,
            is_non_project_mode=is_non_project_mode,
            non_project_format=non_project_format
        )

        detected_format = None
        if not strict and (res['status'] != 'success' or not res['annotations']):
            res_all = self.find_and_load_annotations(
                image_path, mode, "all",
                output_dir=output_dir,
                is_non_project_mode=is_non_project_mode,
                non_project_format=non_project_format
            )
            if res_all['status'] == 'success' and res_all['annotations']:
                res = res_all
                if is_non_project_mode:
                    detected_format = res.get('source_file_type')
        return res, detected_format

    def delete_image_file(self, path):
        """物理删除图片及关联标注。返回 (result, error)。"""
        if not path or not os.path.exists(path):
            return None, "错误: 文件不存在或路径为空"
        result = self.delete_image(path)
        if not result.get("success"):
            return None, "错误: " + "; ".join(result.get("errors", ["未知错误"]))
        if path in self.image_files:
            self.image_files.remove(path)
        if path in self.annotation_cache:
            del self.annotation_cache[path]
        return result, None

    def refresh_directory(self, current_image_path):
        """重新扫描当前目录。返回 (image_files, preserved_index, error)。"""
        if not self.current_dir or not os.path.exists(self.current_dir):
            return None, None, "错误: 当前目录不存在，请先打开图片目录"
        files = self.scan_image_files(self.current_dir)
        self.set_image_files(files)
        if current_image_path and current_image_path in files:
            return files, files.index(current_image_path), None
        return files, None, None

    def set_current_image_with_split(self, path):
        """设置当前图片并推断数据集划分。返回 split 值。"""
        self.set_current_image(path)
        norm_path = os.path.normpath(path)
        split = self.image_splits.get(norm_path, "")
        if not split:
            split = self.infer_split_from_path(norm_path)
            self.set_image_split(norm_path, split)
        return split

    def clear_all(self):
        """清空导航与编辑器状态（图片文件、目录、缓存、划分、撤销栈、副标注）。"""
        self.clear()
        self._undo_stack = []
        self._redo_stack = []
        self._is_undo_redo = False
        self.reference_annotation = None
        self.secondary_annotations = []
        self.secondary_annotation_map = {}
        self.secondary_loaded = False
        self.secondary_dir = None
        self._last_sam_annotation = None

    def get_cached_annotations(self, image_path, output_dir, is_non_project_mode, non_project_format):
        """读取（或加载并缓存）图片标注。返回 annotations 列表。"""
        if image_path in self.annotation_cache:
            return list(self.annotation_cache[image_path])

        annotations = []
        try:
            fmt = non_project_format or "labelme" if is_non_project_mode else "labelme"
            res = self.find_and_load_annotations(
                image_path, 'all', fmt,
                output_dir=output_dir,
                is_non_project_mode=is_non_project_mode,
                non_project_format=non_project_format
            )
            if res.get('status') != 'success' or not res.get('annotations'):
                res = self.find_and_load_annotations(
                    image_path, 'all', 'all',
                    output_dir=output_dir,
                    is_non_project_mode=is_non_project_mode,
                    non_project_format=non_project_format
                )
            if res.get('status') == 'success' and res.get('annotations'):
                annotations = res['annotations']
        except Exception:
            pass
        self.annotation_cache[image_path] = annotations
        return list(annotations)

    @action("manual.get_annotations_for_image", read_only=True,
            description="获取指定图片的标注列表(带缓存)。\n"
                        "- image_path: 必填，目标图片的绝对路径。调用前必须先通过 manual.get_state 获取当前图片路径（返回的 current_image_path 字段），将该路径作为 image_path 传入；不要空参调用\n"
                        "- 返回 {status: success/error, annotations: [{label, bbox, polygons}], message?}",
            category="导航", scope="agent", params={"image_path": {"type": "str", "required": True, "description": "目标图片绝对路径（必填）。调用前先通过 manual.get_state 获取 current_image_path，再作为本参数传入"}})
    def get_annotations_for_image(self, image_path: str = ""):
        """通过缓存读取图片标注,返回简洁标注列表。"""
        if not image_path:
            return {"status": "error", "annotations": [], "message": "image_path 必填：请先调用 manual.get_state 获取 current_image_path 后重试"}
        if not os.path.isfile(image_path):
            return {"status": "error", "annotations": [], "message": f"图片文件不存在: {image_path}"}
        try:
            anns = self.get_cached_annotations(
                image_path,
                self.context.output_dir,
                self.context._is_non_project_mode,
                getattr(self.context, '_non_project_format', None),
            )
            out = [{"label": a.get("label", ""), "bbox": a.get("bbox"), "polygons": a.get("polygons")}
                   for a in (anns or []) if isinstance(a, dict)]
            return {"status": "success", "annotations": out}
        except Exception as e:
            return {"status": "error", "annotations": [], "message": str(e)}

    @action("manual.search_images", read_only=True,
            description="按关键词搜索图片，返回每张图片的可见性（同时应用已设置的类别/尺寸/宽高筛选）。\n"
                        "- keyword: 图片名称关键词，支持 * 通配符（如 '*.png'）和子串匹配（如 '101' 匹配 'img_101.jpg'）；空字符串表示不按名称过滤\n"
                        "- 返回 dict：{\"summary\": {\"total\": 总数, \"visible\": 可见数, \"hidden\": 隐藏数}, \"files\": [[file_path, hidden], ...]}\n"
                        "- 注意：hidden=False 表示该图片可见（保留显示），hidden=True 表示应被隐藏；判断筛选结果请以 summary.visible 为准\n"
                        "- 刷新类操作已自动应用筛选结果到文件列表，无需手动调用本方法验证筛选生效\n"
                        "- 仅做名称过滤时，可先调用 manual.clear_filters 清除标注筛选",
            category="标注", params={"keyword": "str"}, scope="agent")
    def search_snapshot(self):
        """取一份搜索所需状态的**快照**（在 GUI 线程调用），供后台扫描使用。

        后台扫描必须基于快照：否则工作线程在迭代 `image_files` / 读取 `filter_*` /
        `image_splits` 时会被 GUI 线程的修改撕裂（列表被整体替换、筛选条件改到一半），
        得到自相矛盾的可见性结果。
        """
        return {
            "image_files": list(self.image_files),
            "image_splits": dict(self.image_splits),
            "filters": {
                "category": self.filter_category,
                "size_range": self.filter_size_range,
                "width_range": self.filter_width_range,
                "height_range": self.filter_height_range,
                "annotated": self.filter_annotated,
                "split": self.filter_split,
            },
        }

    def search_images(self, keyword, output_dir="", is_non_project_mode=False,
                      non_project_format=None, snapshot=None, apply_state=True):
        """计算文件列表中每个文件的可见性（搜索 + 标注筛选）。

        返回 dict：{"summary": {"total", "visible", "hidden"}, "files": [[file_path, hidden], ...]}。

        `snapshot` 非空时一律使用快照里的状态（供**后台**扫描）；`apply_state=False`
        时不回写 `self.file_visibility`（由调用方在 GUI 线程统一回写）。
        """
        snap = snapshot or {}
        files = snap.get("image_files")
        if files is None:
            files = self.image_files
        splits = snap.get("image_splits")
        if splits is None:
            splits = self.image_splits
        _f = snap.get("filters") or {}
        filter_category = _f.get("category", self.filter_category)
        filter_size_range = _f.get("size_range", self.filter_size_range)
        filter_width_range = _f.get("width_range", self.filter_width_range)
        filter_height_range = _f.get("height_range", self.filter_height_range)
        filter_annotated = _f.get("annotated", self.filter_annotated)
        filter_split = _f.get("split", self.filter_split)

        results = []
        search_text_lower = (keyword or "").lower()
        has_filter = bool(filter_category or filter_size_range or
                          filter_width_range or filter_height_range or
                          filter_annotated is not None or filter_split is not None)
        for file_path in files:
            if not isinstance(file_path, str):
                continue
            filename = os.path.basename(file_path).lower()
            if '*' in search_text_lower:
                pattern = '.*'.join(re.escape(part) for part in search_text_lower.split('*'))
                text_match = bool(re.fullmatch(pattern, filename))
            else:
                text_match = search_text_lower in filename

            annotation_match = True
            if has_filter:
                annotations = self.get_cached_annotations(
                    file_path, output_dir, is_non_project_mode, non_project_format
                )
                num_anns = len(annotations)
                # 数据集划分筛选 gate：train/val/test 精确匹配；"" 表示显示「未划分」图片
                if filter_split is not None:
                    actual_split = splits.get(os.path.normpath(file_path), "")
                    if filter_split == "":
                        split_ok = not actual_split
                    else:
                        split_ok = (actual_split == filter_split)
                else:
                    split_ok = True
                matched = False
                # 标注状态筛选优先：只判断是否有标注，再叠加类别/尺寸条件
                if filter_annotated is not None:
                    annotated_ok = (num_anns > 0) if filter_annotated else (num_anns == 0)
                    if not annotated_ok:
                        matched = False
                    elif filter_category or filter_size_range or \
                            filter_width_range or filter_height_range:
                        matched = any(
                            self._check_annotation_match(
                                ann,
                                filter_category=filter_category,
                                filter_size_range=filter_size_range,
                                filter_width_range=filter_width_range,
                                filter_height_range=filter_height_range
                            )
                            for ann in annotations
                        )
                    else:
                        matched = True
                elif filter_category or filter_size_range or \
                        filter_width_range or filter_height_range:
                    matched = any(
                        self._check_annotation_match(
                            ann,
                            filter_category=filter_category,
                            filter_size_range=filter_size_range,
                            filter_width_range=filter_width_range,
                            filter_height_range=filter_height_range
                        )
                        for ann in annotations
                    )
                else:
                    matched = True
                annotation_match = split_ok and matched
            results.append((file_path, not (text_match and annotation_match)))
        if apply_state:
            self.file_visibility = results
        total = len(results)
        visible = sum(1 for _, hidden in results if not hidden)
        return {
            "summary": {"total": total, "visible": visible, "hidden": total - visible},
            "files": results,
        }

    def visible_image_indices(self):
        """当前筛选结果下**可见**的图片下标集合（导航的唯一真值来源）。

        数据来自 `file_visibility`（界面在应用筛选/搜索后回写的计算结果）。
        没有筛选时等价于"全部可见"。

        审计阶段 3e：原先 service 侧的 `_navigate_ai` 硬编码
        `set(range(len(image_files)))`，即假设"没有筛选"，于是 agent 的
        `next_image` 会跳到被筛选隐藏的图片上，而界面按方向键会正确跳过 ——
        同一个动作、两条真值。现在两端都从这里取。
        """
        total = len(self.image_files)
        if not self.file_visibility:
            return set(range(total))
        visible = set()
        for i, item in enumerate(self.file_visibility):
            if i >= total:
                break
            try:
                _path, hidden = item
            except (TypeError, ValueError):
                continue
            if not hidden:
                visible.add(i)
        return visible if visible else set(range(total))

    def navigate_relative(self, current_idx, direction, visible_indices):
        """在可见图片列表中查找上一张(-1)或下一张(+1)。

        返回 (target_index, error)。未找到时 target_index 为 None。
        """
        if not self.image_files:
            return None, "错误: 当前没有加载图片目录"
        if not visible_indices:
            return None, None
        pos = max(current_idx, 0) if direction < 0 else max(current_idx, -1)
        if direction < 0:
            for i in range(pos - 1, -1, -1):
                if i in visible_indices:
                    return i, None
        else:
            for i in range(pos + 1, len(self.image_files)):
                if i in visible_indices:
                    return i, None
        return None, None

    # ====== 区域: 类别管理 ======
    # 原 annotation_io: add_category / delete_category / rename_category / merge_categories /
    #   import_categories_from_file / save_categories_to_project / save_categories_json /
    #   categories_from_serializable / _get_category_color / _load_categories_json /
    #   _normalize_categories / _categories_to_serializable / apply_category_deletion /
    #   apply_category_rename / apply_category_merge / load_categories_from_project_info /
    #   batch_remove_category / batch_update_category_name

    def _add_category(self, categories, name, color_value=None):
        if name in categories:
            return categories
        new_cats = dict(categories)
        new_cats[name] = color_value if color_value is not None else self._get_category_color(name)
        return new_cats

    def _delete_category(self, categories, name):
        if name not in categories:
            return categories
        new_cats = dict(categories)
        new_cats.pop(name)
        return new_cats

    def _rename_category(self, categories, old_name, new_name):
        if old_name not in categories or new_name in categories:
            return categories
        new_cats = dict(categories)
        color = new_cats.pop(old_name)
        new_cats[new_name] = color
        return new_cats

    def _merge_categories(self, categories, old_name, new_name):
        if old_name not in categories:
            return categories
        new_cats = dict(categories)
        new_cats.pop(old_name)
        return new_cats

    def import_categories_from_file(self, file_path):
        raw_cats = self._load_categories_json(file_path)
        result = {}
        if raw_cats:
            for k, v in raw_cats.items():
                if isinstance(v, list) and len(v) >= 3:
                    result[k] = v
                else:
                    result[k] = list(self._get_category_color(k))
        return result

    def save_categories_to_project(self, categories, project_name, project_service):
        if project_name and project_service:
            project_service.update_project_categories(project_name, categories)

    def save_categories_json(self, categories, config_path, config=None):
        merged = dict(config or {})
        merged["categories"] = self._categories_to_serializable(categories)
        try:
            with atomic_open(config_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[AnnotationService] 保存类别到 JSON 失败: {e}")

    def categories_from_serializable(self, data):
        """从 RGB list 格式恢复为 {name: [R, G, B]} 字典（供 View 层转换为 QColor）"""
        result = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) >= 3:
                result[k] = [int(v[0]), int(v[1]), int(v[2])]
            else:
                result[k] = [200, 200, 200]
        return result

    # ---------- 内部类别工具方法 ----------

    def _get_category_color(self, label):
        """从固定颜色表获取类别颜色（md5 哈希保证跨进程一致性，返回 list 格式用于序列化）"""
        import hashlib
        from . import CATEGORY_COLOR_PALETTE
        idx = int(hashlib.md5(label.encode()).hexdigest(), 16) % len(CATEGORY_COLOR_PALETTE)
        return list(CATEGORY_COLOR_PALETTE[idx])

    def _load_categories_json(self, path):
        """从 JSON 文件加载类别原始数据"""
        import json
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

    def _normalize_categories(self, categories):
        """将多种格式的类别数据归一化为 id_to_name 和 name_to_id 映射"""
        id_to_name = {}
        name_to_id = {}
        if not categories:
            return id_to_name, name_to_id
        if isinstance(categories, dict):
            is_id_map = all(isinstance(k, (int, str)) and str(k).isdigit() for k in categories)
            if is_id_map:
                sorted_ids = sorted(int(k) for k in categories)
                id_to_name = {i: categories[str(i)] if str(i) in categories else categories[i]
                              for i in sorted_ids}
                name_to_id = {v: k for k, v in id_to_name.items()}
            else:
                for i, name in enumerate(categories):
                    id_to_name[i] = name
                    name_to_id[name] = i
        elif isinstance(categories, (list, tuple)):
            for i, name in enumerate(categories):
                id_to_name[i] = name
                name_to_id[name] = i
        return id_to_name, name_to_id

    def _categories_to_serializable(self, categories):
        """将 QColor 值转为可序列化 RGB list 格式"""
        result = {}
        for k, v in categories.items():
            if hasattr(v, 'getRgb'):
                result[k] = list(v.getRgb()[:3])
            elif hasattr(v, '__len__') and len(v) >= 3:
                result[k] = list(v[:3])
            else:
                result[k] = [0, 200, 220]
        return result

    def apply_category_deletion(self, categories, annotations, category_name):
        if category_name not in categories:
            return {"categories": categories, "annotations": annotations,
                    "changed": False, "fallback_category": None}
        new_cats = dict(categories)
        new_cats.pop(category_name)
        new_anns = [ann for ann in annotations if ann.get('label') != category_name]
        changed = len(new_anns) != len(annotations)
        fallback = next(iter(new_cats.keys())) if new_cats else None
        return {"categories": new_cats, "annotations": new_anns,
                "changed": changed, "fallback_category": fallback}

    def apply_category_rename(self, categories, annotations, old_name, new_name):
        if old_name not in categories or new_name in categories:
            return {"success": False, "message": "无效的重命名操作"}
        new_cats = dict(categories)
        color = new_cats.pop(old_name)
        new_cats[new_name] = color
        changed = False
        new_anns = []
        for ann in annotations:
            new_ann = dict(ann)
            if new_ann.get('label') == old_name:
                new_ann['label'] = new_name
                changed = True
            new_anns.append(new_ann)
        return {"success": True, "categories": new_cats, "annotations": new_anns, "changed": changed}

    def apply_category_merge(self, categories, annotations, old_name, new_name):
        if old_name not in categories or new_name not in categories:
            return {"success": False, "message": "无效的合并操作"}
        new_cats = dict(categories)
        new_cats.pop(old_name)
        changed = False
        new_anns = []
        for ann in annotations:
            new_ann = dict(ann)
            if new_ann.get('label') == old_name:
                new_ann['label'] = new_name
                changed = True
            new_anns.append(new_ann)
        return {"success": True, "categories": new_cats, "annotations": new_anns, "changed": changed}

    def load_categories_from_project_info(self, raw_categories):
        """从项目信息解析类别数据"""
        if not raw_categories:
            return {}
        if all(isinstance(v, str) and not v.startswith('#') for v in raw_categories.values()):
            categories = {}
            sorted_indices = sorted(raw_categories.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            for idx in sorted_indices:
                categories[raw_categories[idx]] = None
            return categories
        else:
            return {name: color if isinstance(color, str) else None
                    for name, color in raw_categories.items()}

    # ====== 区域: 项目 & 类别管理状态（原 ProjectVM）======
    # 原 project_view_model: set_categories / set_current_category / select_category /
    #   touch_recent_category / cycle_category / apply_task_mode / apply_split /
    #   set_task_mode / set_project / _category_color / load_categories_from_project /
    #   setup_project / change_save_path / set_non_project_mode / load_project /
    #   enter_non_project_mode / load_non_project_config(已合并) / detect_and_set_export_format /
    #   apply_detected_task_type / add_category / delete_category / rename_category /
    #   merge_categories / import_categories / export_categories /
    #   save_categories_to_settings / save_non_project_config / toggle_hard_sample /
    #   save_hard_samples_config / load_hard_samples_config / get_hard_sample_state

    def set_categories(self, categories):
        """设置类别字典（写入共享 Context）"""
        self.context.categories = categories

    def set_current_category(self, category):
        """设置当前选中类别（写入共享 Context）"""
        self.context.current_category = category

    @action("manual.select_category",
            description="选择当前标注类别并更新最近使用顺序。\n- category_name：目标类别名称（必须已存在）\n- 返回 None 表示成功，否则为错误字符串",
            category="标注", params={"category_name": "str"}, scope="agent")
    def select_category(self, category_name=""):
        """选择当前标注类别并更新最近使用顺序。类别不存在时返回错误信息。"""
        if not category_name:
            return "错误: select_category 需要 category_name 参数"
        if category_name not in self.context.categories:
            return "错误: 类别不存在"
        self.context.current_category = category_name
        self.touch_recent_category(category_name)
        project_settings.set("current_category", category_name)
        self._publish_categories_changed(reason="select_category")
        return None

    def touch_recent_category(self, category_name):
        """将类别标记为最近使用（最近的在最前）"""
        if category_name in self.context._recent_categories:
            self.context._recent_categories.remove(category_name)
        self.context._recent_categories.insert(0, category_name)

    def cycle_category(self):
        """按最近使用顺序循环切换当前类别。返回 (next_category, error)。"""
        if not self.context.categories:
            return None, "错误: 没有可用的类别"
        recent = [c for c in self.context._recent_categories if c in self.context.categories]
        remaining = [c for c in self.context.categories if c not in recent]
        ordered = recent + remaining
        if len(ordered) <= 1:
            return None, None
        try:
            idx = ordered.index(self.context.current_category)
            next_idx = (idx + 1) % len(ordered)
        except ValueError:
            next_idx = 0
        next_cat = ordered[next_idx]
        self.select_category(next_cat)
        return next_cat, None

    @action("manual.set_task_mode",
            description="设置任务模式并同步项目 task_type。\n- mode：seg=分割, det=检测, obb=旋转框\n- 返回 (old_mode, error)",
            category="标注", params={"mode": "str"}, scope="agent")
    def apply_task_mode(self, mode=""):
        """应用任务模式并同步项目 task_type。返回 (old_mode, error)。"""
        if not mode:
            return None, "错误: apply_task_mode 需要 mode 参数"
        old_mode = self.context.task_mode
        if old_mode == mode:
            return old_mode, None
        self.set_task_mode(mode)
        project_settings.set("task_mode", mode)
        if not self.context._is_non_project_mode and self.project_service:
            p_name = self.context.current_project_name
            if p_name:
                # 任务类型统一为内部短名 det/seg/obb
                task_type_str = mode if mode in ("det", "seg", "obb") else "seg"
                self.project_service._update_project_info(p_name, {"task_type": task_type_str})
        return old_mode, None

    @action("manual.set_split",
            description="设置图片的数据集划分。\n- split：train/val/test 之一\n- image_path：图片路径，留空使用当前图片\n- 返回 (split, error)",
            category="标注", params={"split": "str", "image_path": "str"}, scope="agent")
    def apply_split(self, split="", image_path=""):
        """解析并应用图片的数据集划分。返回 (split, error)。"""
        if split:
            split_map = {"train": 0, "val": 1, "test": 2}
            index = split_map.get(split.lower(), -1)
            if index < 0:
                return None, "错误: 未知的分割类型: %s，可选: train/val/test" % split
            split = ["train", "val", "test"][index]
        if not split:
            return None, "错误: 未指定分割类型"
        if not image_path:
            image_path = self.current_image_path or ""
        if not image_path:
            return None, "错误: 当前没有打开任何图片"
        norm_path = os.path.normpath(image_path)
        self.set_image_split(norm_path, split)
        self._save_dataset_splits_to_file(self.context.output_dir, self.image_splits)
        self._publish_datasets_changed(op="apply_split", image_path=norm_path, split=split)
        return split, None

    def set_task_mode(self, mode):
        """记录任务模式状态（seg/det/obb），并发布 task_mode 信号驱动界面同步。"""
        self.context.task_mode = mode
        self._emit(StateType.TASK_MODE_CHANGED, {"task_mode": mode})

    def set_project(self, project_info):
        """将项目信息写入共享 Context"""
        self.context.current_project_name = project_info.get("name", self.context.current_project_name)
        self.context.output_dir = project_info.get("path", self.context.output_dir)
        # 规则以 project_info 持久化配置为准；无 rules/为空时不静默注入默认规则，
        # 与 setup_project 保持一致（预测只用配置里的规则，空则走各环节统一兜底）
        self.load_rules_from_project(project_info, keep_when_empty=True)

    def load_rules_from_project(self, project_info, keep_when_empty=False):
        """从项目信息装载规则，并把旧版（相对语义）规则一次性迁移到绝对语义版本。

        迁移仅在确有旧字段时写回 project_info.json（rules + rules_schema），
        保证文件、内存缓存、规则配置对话框、预测四处一致。
        """
        raw_rules = project_info.get('rules') or {}
        if not raw_rules and keep_when_empty:
            return self.context.current_project_rules

        rules, migrated = migrate_legacy_rules(raw_rules, project_info.get('rules_schema'))
        self.context.current_project_rules = rules
        if migrated:
            project_name = project_info.get('name')
            print(f"[AnnotationService] 项目 '{project_name}' 的旧版相对规则已迁移为绝对规则: {rules}")
            self._persist_rules_migration(project_name, rules)
        return rules

    def _persist_rules_migration(self, project_name, rules):
        """把迁移后的规则写回项目配置（update_project_rules 会一并写入
        rules_schema，并同步内存缓存）。"""
        ps = self.project_service
        if not project_name or ps is None:
            return
        try:
            ps.update_project_rules(project_name, rules)
        except Exception as e:
            print(f"[AnnotationService] 规则迁移写回失败: {e}")

    def _category_color(self, name):
        """根据类别名分配固定颜色（与 interfaces.get_category_color 同规则）。"""
        from . import CATEGORY_COLOR_PALETTE
        idx = hash(name) % len(CATEGORY_COLOR_PALETTE)
        h, s, v = CATEGORY_COLOR_PALETTE[idx]
        return QColor.fromHsv(h, s, v)

    def load_categories_from_project(self, project_info):
        """从项目信息解析类别。返回 categories dict。"""
        raw_categories = project_info.get('categories', {})
        self.context.project_has_own_categories = bool(raw_categories)
        parsed = self.load_categories_from_project_info(raw_categories)
        cats = {}
        for name, color_hex in parsed.items():
            cats[name] = QColor(color_hex) if color_hex else self._category_color(name)
        self.context.categories = cats
        return cats

    def setup_project(self, project_info):
        """装配项目上下文状态（非 UI 部分）。返回生效的 task_mode。"""
        self.context._is_non_project_mode = False
        self.context._non_project_format = None
        self.context.categories = {}
        self.context.project_has_own_categories = False
        self.context.output_dir = project_info['path']
        self.context.current_project_name = project_info.get('name')
        # 规则以项目已持久化的配置为准：无 rules 时保持空 dict，不静默注入
        # 默认规则。规则配置对话框（RuleConfigDialog）显示什么，预测就用什么；
        # 空规则下各环节兜底默认统一为 mode='reuse'（与对话框/项目创建种子一致）。
        # 旧项目（相对规则版本）在此一次性迁移为绝对规则。
        self.load_rules_from_project(project_info)

        # 任务类型统一为内部短名 det/seg/obb（兼容旧项目长名 detection/segmentation）
        from core.service.project_service import normalize_task_type
        task_mode = normalize_task_type(project_info.get('task_type', 'seg'))
        self.set_task_mode(task_mode)
        project_settings.set("task_mode", task_mode)

        self.load_categories_from_project(project_info)

        if self.context.categories:
            saved_cat = project_settings.get("current_category")
            if saved_cat and saved_cat in self.context.categories:
                self.set_current_category(saved_cat)
            else:
                self.set_current_category(next(iter(self.context.categories.keys())))
            project_settings.set("current_category", self.context.current_category)
            self.touch_recent_category(self.context.current_category)

        return task_mode

    def update_save_path(self, dir_path):
        """切换保存路径。成功返回 None，失败返回错误字符串。"""
        result = self.change_path(dir_path)
        if result.get("status") != "success":
            return "错误: 更新路径失败"
        self.context.output_dir = dir_path
        if self.context._is_non_project_mode:
            self.context._non_project_save_dir = dir_path
        return None

    def set_non_project_mode(self, output_dir):
        """同步非项目模式输出目录给 AutoService（示例库目录）。"""
        if self.auto is None:
            return
        if self.auto is self:
            from .auto_annotation_service import AutoAnnotationService
            AutoAnnotationService.set_non_project_mode(self, output_dir)
        else:
            self.auto.set_non_project_mode(output_dir)

    @action("manual.select_project",
        description="加载指定项目并载入其图片，供后续自动标注使用。\n- project_name：项目名（可用 get_state 的 projects 字段查看有哪些项目）\n- 加载后即设置当前项目、类别、任务模式并扫描 images 目录图片\n- 返回 {status, project_name, path, image_count, last_index}",
        category="导航", params={"project_name": "str"}, scope="agent")
    def select_project(self, project_name=""):
        """按项目名加载项目及其图片。项目不存在时返回错误。"""
        if not project_name:
            return {"status": "error", "message": "缺少 project_name 参数，可用 get_state.projects 查看项目列表"}
        if not self.project_service:
            return {"status": "error", "message": "项目服务不可用"}
        project_info = self.project_service.load_project(project_name)
        if not project_info:
            return {"status": "error", "message": "项目不存在: %s（可用 get_state.projects 查看）" % project_name}
        task_mode = self.setup_project(project_info)
        # 解析项目图片目录路径
        base_dir = getattr(self.project_service, 'base_dir', None)
        if not base_dir:
            base_dir = os.path.dirname(project_info.get("path", "")) if project_info.get("path") else ""
        images_dir = os.path.join(base_dir, project_name, "images")
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(project_info.get("path", ""), "images")
        files, last_idx = self.load_directory(images_dir)
        self._publish_project_selected(project_info)
        return {"status": "success",
                "project_name": project_name,
                "task_mode": task_mode,
                "image_count": len(files),
                "last_index": last_idx,
                "path": images_dir}

    def load_project(self, project_name):
        """加载项目信息并写入状态。项目不存在时返回 None。"""
        if not self.project_service or not project_name:
            return None
        project_info = self.project_service.load_project(project_name)
        if project_info:
            self.set_project(project_info)
        return project_info

    def enter_non_project_mode(self, dir_path):
        """进入非项目模式并扫描目录。返回 scan_result dict。

        `_non_project_save_dir` 记录"**当前**非项目目录的保存位置"，每次进入都必须
        无条件指向本次目录。原先写成 `if not self.context._non_project_save_dir:`
        （第一次进入才设置），于是**先开目录 A、再开目录 B** 时 `output_dir` 仍是 A，
        B 的标注会被写进 A 里（同名 basename 直接覆盖）—— 审计 D4。

        "上次打开过哪个目录"是另一回事，由 `last_opened_dir` 这类设置项承担，
        不复用这个字段。
        """
        self.context._is_non_project_mode = True
        self.context._non_project_format = None
        self.current_dir = dir_path
        self.context._non_project_save_dir = dir_path      # 无条件更新（D4）
        self.context.output_dir = dir_path
        self.context.categories = {}
        self.set_current_category(None)
        self.context.project_has_own_categories = False
        return self.scan_directory(dir_path)

    def detect_and_set_export_format(self, dir_path):
        """扫描非项目目录并检测导出格式。

        返回 (detected_task_type, new_categories, detected_format)，未检测到时返回 (None, None, None)。
        """
        scan_result = self.scan_directory(dir_path)
        if scan_result.get('status') != 'success':
            return None, None, None
        image_files = scan_result.get('image_files', [])
        annotations = scan_result.get('annotations', [])
        detected_task_type = None
        detected_format = None
        new_categories = {}
        for ann in annotations:
            if not ann.get('label'):
                continue
            label = ann.get('label')
            if label not in new_categories:
                new_categories[label] = self._category_color(label)
            cat = label
            bbox = ann.get('bbox') or {}
            points = ann.get('points') or {}
            if bbox and not points:
                detected_task_type = 'det'
            if points:
                detected_task_type = 'seg'
        if new_categories and not detected_task_type:
            detected_task_type = 'seg'
        if not new_categories:
            new_categories = None
        if detected_task_type and not self.context._non_project_format:
            detected_format = detected_task_type
        return detected_task_type, new_categories, detected_format

    def apply_detected_task_type(self, task_type):
        """应用检测到的任务类型并写入设置。"""
        task_mode = "det" if task_type == "det" else "seg"
        self.set_task_mode(task_mode)
        project_settings.set("task_mode", task_mode)

    @action("manual.add_category",
            description="添加新的标注类别并设为当前类别。\n- category_name：类别名称（必填）\n- color：颜色，如 '#FF0000'，留空自动分配\n- 若类别已存在则不重复创建，直接设为当前类别（幂等）\n- 返回 (categories, error)",
            category="标注", params={"category_name": "str", "color": "str"}, scope="agent")
    def add_category(self, category_name="", color=None):
        """添加类别并设为当前类别。返回 (categories, error)。"""
        if not category_name:
            return None, "错误: 类别名称为空"
        if category_name in self.context.categories:
            # 幂等：类别已存在时不报错，直接设为当前类别
            self.set_current_category(category_name)
            self.touch_recent_category(category_name)
            project_settings.set("current_category", category_name)
            self._publish_categories_changed(reason="add_category")
            return self.context.categories, None
        new_cats = self._add_category(
            self.context.categories, category_name, color_value=color
        )
        value = new_cats.get(category_name)
        if isinstance(value, str):
            new_cats[category_name] = QColor(value)
        elif isinstance(value, (list, tuple)):
            new_cats[category_name] = QColor(*[int(c) for c in value[:3]])
        self.context.categories = new_cats
        self.set_current_category(category_name)
        self.touch_recent_category(category_name)
        project_settings.set("current_category", category_name)
        self.save_categories_to_settings()
        self._publish_categories_changed(reason="add_category")
        return new_cats, None

    @action("manual.remove_category",
            description="删除类别并批量更新磁盘上的标注文件。\n- category_name：要删除的类别名称（必须已存在）\n- annotations：当前图片标注列表（可省略，省略时仅做磁盘批量更新）\n- 返回 (categories, annotations, info)",
            category="标注", params={"category_name": "str", "annotations": "list"}, scope="agent")
    def remove_category(self, category_name="", annotations=None):
        """删除类别并处理当前图片标注与批量更新。返回 (categories, annotations, info)。"""
        if not category_name:
            return None, None, {"error": "错误: remove_category 需要 category_name 参数"}
        if category_name not in self.context.categories:
            return None, None, {"error": "错误: 要删除的类别不存在"}
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        result = self.apply_category_deletion(
            self.context.categories, annotations, category_name
        )
        result["batch_status"] = self.batch_remove_category(
            self.context.output_dir, category_name
        )
        self.context.categories = result["categories"]
        if self.current_annotations:
            self.current_annotations[:] = result["annotations"]
        self.save_categories_to_settings()
        if self.context.current_category == category_name:
            self.set_current_category(result["fallback_category"] or "")
        self._publish_categories_changed(reason="remove_category")
        return result["categories"], result["annotations"], result

    @action("manual.rename_category",
            description="重命名类别（若新名称已存在则合并）。\n- old_name：旧类别名称\n- new_name：新类别名称\n- annotations：当前图片标注列表（可省略）\n- 返回 (categories, annotations, info)",
            category="标注", params={"old_name": "str", "new_name": "str", "annotations": "list"}, scope="agent")
    def rename_category(self, old_name="", new_name="", annotations=None):
        """重命名类别（若目标存在则合并）。返回 (categories, annotations, info)。"""
        if not old_name or not new_name:
            return None, None, {"error": "错误: rename_category 需要 old_name 和 new_name 参数"}
        if old_name not in self.context.categories:
            return None, None, {"error": "错误: 要重命名的类别不存在"}
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if new_name in self.context.categories:
            result = self.apply_category_merge(
                self.context.categories, annotations, old_name, new_name
            )
            result["batch_status"] = self.batch_update_category_name(
                self.context.output_dir, old_name, new_name
            )
            self.context.categories = result["categories"]
            if self.current_annotations:
                self.current_annotations[:] = result["annotations"]
            self.save_categories_to_settings()
            self._publish_categories_changed(reason="rename_category")
            return result["categories"], result["annotations"], result
        result = self.apply_category_rename(
            self.context.categories, annotations, old_name, new_name
        )
        result["batch_status"] = self.batch_update_category_name(
            self.context.output_dir, old_name, new_name
        )
        self.context.categories = result["categories"]
        if self.current_annotations:
            self.current_annotations[:] = result["annotations"]
        self.save_categories_to_settings()
        self._publish_categories_changed(reason="rename_category")
        return result["categories"], result["annotations"], result

    @action("manual.merge_categories",
            description="合并类别（旧类别并入新类别）。\n- old_name：被合并的旧类别名称\n- new_name：目标类别名称（必须已存在）\n- annotations：当前图片标注列表（可省略）\n- 返回 (categories, annotations, info)",
            category="标注", params={"old_name": "str", "new_name": "str", "annotations": "list"}, scope="agent")
    def merge_categories(self, old_name="", new_name="", annotations=None):
        """合并类别（旧类别并入新类别）。返回 (categories, annotations, info)。"""
        if not old_name or not new_name:
            return None, None, {"error": "错误: merge_categories 需要 old_name 和 new_name 参数"}
        if old_name not in self.context.categories:
            return None, None, {"error": "错误: 要合并的类别不存在"}
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        result = self.apply_category_merge(
            self.context.categories, annotations, old_name, new_name
        )
        result["batch_status"] = self.batch_update_category_name(
            self.context.output_dir, old_name, new_name
        )
        self.context.categories = result["categories"]
        if self.current_annotations:
            self.current_annotations[:] = result["annotations"]
        self.save_categories_to_settings()
        self._publish_categories_changed(reason="merge_categories")
        return result["categories"], result["annotations"], result

    @action("manual.import_categories",
            description="从 JSON 文件导入类别并替换当前类别列表。\n- file_path：JSON 文件路径\n- 返回 (categories, error)",
            category="标注", params={"file_path": "str"}, scope="agent")
    def import_categories(self, file_path=""):
        """从 JSON 文件导入类别并替换当前列表。返回 (categories, error)。"""
        if not file_path:
            return None, "错误: import_categories 需要 file_path 参数"
        try:
            raw_cats = self.import_categories_from_file(file_path)
            if not raw_cats:
                return None, "错误: 导入文件为空或格式无效"
            rgb_cats = self.categories_from_serializable(raw_cats)
            cats = {k: QColor(*v) for k, v in rgb_cats.items()}
            self.context.categories = cats
            if cats:
                self.set_current_category(next(iter(cats.keys())))
                project_settings.set("current_category", self.context.current_category)
            self._publish_categories_changed(categories=cats, reason="import_categories")
            return cats, None
        except Exception as e:
            return None, str(e)

    @action("manual.export_categories",
            description="导出当前类别列表到 JSON 文件。\n- file_path：目标 JSON 文件路径\n- 成功返回 None，失败返回错误字符串",
            category="标注", params={"file_path": "str"}, scope="agent")
    def export_categories(self, file_path=""):
        """导出类别到 JSON 文件。成功返回 None，失败返回错误字符串。"""
        if not file_path:
            return "错误: export_categories 需要 file_path 参数"
        try:
            self.save_categories_json(self.context.categories, file_path)
            self._publish_categories_changed(reason="export_categories")
            return None
        except Exception as e:
            return str(e)

    def save_categories_to_settings(self):
        """将类别持久化到项目配置或非项目配置。"""
        if self.context._is_non_project_mode:
            self.save_non_project_config(self.context.task_mode)
            return
        if self.context.current_project_name and self.project_service:
            self.save_categories_to_project(
                self.context.categories, self.context.current_project_name, self.project_service
            )

    def save_non_project_config(self, task_mode="seg"):
        """非项目模式下组装并保存配置文件。"""
        if not self.context._is_non_project_mode or not self.context.output_dir:
            return
        config = {
            "task_mode": task_mode,
            "categories": {
                name: color.name() if hasattr(color, "name") else color
                for name, color in self.context.categories.items()
            },
            "current_category": self.context.current_category or "",
            "export_format": self.context._non_project_format or "labelme",
            "rules": self.context.current_project_rules,
            "hard_samples": [
                {"path": p, "description": d} for p, d in self.context.hard_samples.items()
            ],
        }
        self._save_non_project_config_file(self.context.output_dir, config)

    @action("manual.set_hard_sample",
            description="设置图片为难样本或取消标记。\n- is_hard：True 标记为困难样本，False 取消标记\n- image_path：图片路径，留空使用当前图片\n- description：困难样本描述（可选）\n- 返回 (hard_samples, error)",
            category="标注", params={"is_hard": "bool", "image_path": "str", "description": "str"}, scope="agent")
    def toggle_hard_sample(self, is_hard=False, image_path="", description=""):
        """切换难样本标记。返回 (hard_samples, error)。"""
        if not image_path:
            image_path = self.current_image_path or ""
        if not image_path:
            return None, "错误: 当前未加载图片"
        if not is_hard:
            description = ""
        self.context.hard_samples = self._toggle_hard_sample(
            self.context.hard_samples, os.path.normpath(image_path), bool(is_hard), description
        )
        self._publish_hard_sample_changed(image_path=image_path, is_hard=bool(is_hard))
        return self.context.hard_samples, None

    def save_hard_samples_config(self):
        """持久化难样本标记。"""
        if self.context._is_non_project_mode:
            self.save_non_project_config(self.context.task_mode)
            return
        if not self.context.current_project_name or not self.project_service:
            return
        self.save_hard_samples(
            self.context.hard_samples, self.context.current_project_name, self.project_service
        )

    def load_hard_samples_config(self):
        """加载难样本标记。返回 hard_samples。"""
        self.context.hard_samples = self.load_hard_samples(
            self.context.current_project_name, self.project_service,
            is_non_project_mode=self.context._is_non_project_mode,
            output_dir=self.context.output_dir
        )
        return self.context.hard_samples

    def get_hard_sample_state(self, image_path):
        """查询图片是否为难样本。返回 (is_hard, description)。"""
        if not image_path:
            return False, ""
        norm_path = os.path.normpath(image_path)
        return norm_path in self.context.hard_samples, self.context.hard_samples.get(norm_path, "")

    # ====== 区域: 标注编辑（低层算法）======
    # 原 annotation_io: recalculate_bbox_from_polygons / _remove_polygon_points(原 _delete_polygon_points) /
    #   adjust_hidden_indices_after_deletion / add_secondary_annotation_to_primary /
    #   _set_annotation_label(原 change_annotation_label) / batch_update_annotation_labels /
    #   batch_remove_annotation_by_label

    def recalculate_bbox_from_polygons(self, annotation):
        """根据标注的多边形顶点重新计算 bbox（xywh 格式）。
        收集所有多边形的所有顶点，计算最小外接矩形。
        """
        polys = annotation.get('polygons', [])
        all_pts = []
        for poly in polys:
            for pt in poly:
                if hasattr(pt, '__len__') and len(pt) >= 2:
                    if hasattr(pt[0], '__len__'):
                        all_pts.append([pt[0][0], pt[0][1]])
                    else:
                        all_pts.append([pt[0], pt[1]])
        if all_pts:
            xs = [pt[0] for pt in all_pts]
            ys = [pt[1] for pt in all_pts]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            annotation['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        return annotation

    def _remove_polygon_points(self, annotations, batch_selected_points):
        """批量删除多边形顶点，并重新计算 bbox。
        batch_selected_points: [(ann_idx, poly_idx, pt_idx), ...]
        """
        points_by_ann = {}
        for ann_idx, poly_idx, pt_idx in batch_selected_points:
            if ann_idx not in points_by_ann:
                points_by_ann[ann_idx] = []
            points_by_ann[ann_idx].append((poly_idx, pt_idx))

        for ann_idx in sorted(points_by_ann.keys(), reverse=True):
            if 0 <= ann_idx < len(annotations):
                ann = annotations[ann_idx]
                polys = ann.get('polygons', [])

                points_by_poly = {}
                for poly_idx, pt_idx in points_by_ann[ann_idx]:
                    if poly_idx not in points_by_poly:
                        points_by_poly[poly_idx] = []
                    points_by_poly[poly_idx].append(pt_idx)

                for poly_idx in sorted(points_by_poly.keys(), reverse=True):
                    if poly_idx < len(polys):
                        poly = polys[poly_idx]
                        if hasattr(poly, 'tolist'):
                            poly_list = poly.tolist()
                        else:
                            poly_list = list(poly)

                        for pt_idx in sorted(points_by_poly[poly_idx], reverse=True):
                            if pt_idx < len(poly_list):
                                poly_list.pop(pt_idx)

                        polys[poly_idx] = poly_list

                # 重新计算 bbox
                self.recalculate_bbox_from_polygons(ann)

    # 旧名兼容别名（Phase 4 删除）
    _delete_polygon_points = _remove_polygon_points

    def adjust_hidden_indices_after_deletion(self, hidden_indices, deleted_index):
        """删除指定索引的标注后，调整 hidden_indices 集合中的索引值"""
        new_hidden = set()
        for i in hidden_indices:
            if i > deleted_index:
                new_hidden.add(i - 1)
            elif i < deleted_index:
                new_hidden.add(i)
        return new_hidden

    def add_secondary_annotation_to_primary(self, annotations, secondary_annotation):
        """将副标注复制为新的主标注，返回新标注和新索引"""
        new_ann = copy.deepcopy(secondary_annotation)
        new_index = len(annotations)
        annotations.append(new_ann)
        return annotations, new_index, new_ann

    def _set_annotation_label(self, annotations, index, new_label, valid_categories=None):
        if index < 0 or index >= len(annotations):
            return {"success": False, "message": "标注索引无效"}
        if valid_categories is not None and new_label not in valid_categories:
            return {"success": False, "message": "目标类别不存在于类别列表中"}
        ann = annotations[index]
        old_label = ann.get('label', '')
        if new_label == old_label:
            return {"success": False, "message": "类别未变化", "label": old_label}
        ann['label'] = new_label
        return {"success": True, "old_label": old_label, "new_label": new_label}

    def batch_update_annotation_labels(self, annotations, old_label, new_label):
        result = []
        changed = False
        for ann in annotations:
            new_ann = dict(ann)
            if new_ann.get('label') == old_label:
                new_ann['label'] = new_label
                changed = True
            result.append(new_ann)
        return result, changed

    def batch_remove_annotation_by_label(self, annotations, label):
        return [ann for ann in annotations if ann.get('label') != label]

    # ====== 区域: 撤销/重做 ======
    # 原 EditorVM: save_undo_state / undo / redo / clear_undo_history / has_undo / has_redo
    # （与 annotation_io 旧版 _undo_stack_impl/_redo_stack_impl 合并去重，保留本实现）

    def save_undo_state(self, annotations, image_path, hidden_indices=None):
        """保存当前标注快照到撤销栈。

        **可见性必须与标注同帧保存**（审计 D21）：`hidden_indices` 是按下标寻址的
        集合，删除/清空标注会让下标位移或失效。只快照标注、不快照可见性，撤销后就会
        出现"标注列表回来了、但可见性停留在删除后的状态"的不自洽组合 —— 例如
        `clear_annotations` 先把集合清空，撤销后所有标注仍然全部可见（原来的隐藏
        状态永久丢失）。

        `hidden_indices=None` 时取 service 当前值（copy 一份，避免后续原地修改
        污染已入栈的帧）。
        """
        if self._is_undo_redo:
            return
        source = self.hidden_indices if hidden_indices is None else hidden_indices
        state = {
            'annotations': copy.deepcopy(annotations),
            'image_path': image_path,
            'hidden_indices': set(source or ()),
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    @action("manual.step_history",
            description="撤销或重做标注操作。\n- direction：-1=撤销，1=重做\n- annotations：当前标注列表（可省略，缺省用 service 当前标注）\n- image_path：图片路径（可省略）\n- 返回 (restored_annotations, error)，恢复结果已写回 service 并保存",
            category="标注", params={"direction": "int", "annotations": "list", "image_path": "str"}, scope="agent")
    def step_history(self, direction=-1, annotations=None, image_path=""):
        """撤销或重做标注操作。direction=-1 撤销，1 重做。返回 (restored_annotations, error)。"""
        if direction > 0:
            return self.redo(annotations, image_path)
        return self.undo(annotations, image_path)

    @staticmethod
    def _same_image_path(a, b):
        """两个图片路径是否指向同一张图（大小写/相对路径归一化后比较）。

        路径信息缺失时返回 True（不阻断）：撤销栈的入栈点都会带 image_path，
        缺省值只用于兼容旧调用，不应因此拒绝正常撤销。
        """
        if not a or not b:
            return True
        try:
            return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        except Exception:
            return a == b

    def _pop_undo_frame(self, stack, kind):
        """弹出栈顶一帧，并**校验它属于当前图片**。返回 (frame, error)。

        撤销栈里存了 `image_path` 却从不比对：切图（或切项目）后按 Ctrl+Z，会把
        **旧图**的标注写进**当前图**的标注文件 —— 静默数据损坏（审计 D3）。
        这里在写回/落盘之前断言路径一致；不一致就**清栈并拒绝**（栈已跨图失效，
        留着只会在下一次撤销时再次误写）。
        """
        if not stack:
            return None, None
        frame = stack.pop()
        stored_path = frame.get('image_path') or ''
        if not self._same_image_path(stored_path, self.current_image_path):
            self.clear_undo_history()
            return None, ("错误: %s栈里的操作属于另一张图片(%s)，已清空撤销历史，"
                          "以免把旧图标注写入当前图的标注文件"
                          % (kind, os.path.basename(stored_path) or '未知'))
        return frame, None

    def undo(self, annotations=None, image_path=""):
        """撤销上次操作。返回 (restored_annotations, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致栈记录无效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not self._undo_stack:
            return None, "错误: 没有可撤销的操作"
        prev_state, err = self._pop_undo_frame(self._undo_stack, "撤销")
        if err:
            return None, err
        current_state = {
            'annotations': copy.deepcopy(annotations),
            'image_path': image_path,
            'hidden_indices': set(self.hidden_indices or ()),
        }
        self._redo_stack.append(current_state)
        restored = copy.deepcopy(prev_state['annotations'])
        # 可见性随标注一同回滚（审计 D21：只回滚标注会让下标集合与列表不自洽）
        self.hidden_indices = set(prev_state.get('hidden_indices') or ())
        # 写回共享数据并保存，确保刷新后可见。
        #
        # 注意这里**不能**用 `if self.current_annotations:` 做守卫 —— 那是拿
        # "即将被替换掉的状态"当条件：清除全部标注后 current_annotations 是空列表，
        # 守卫直接跳过写回，于是撤销只恢复了可见性、标注列表仍是空的
        # （H2 抓到的真实缺陷）。统一走 session 的受控入口，空列表也能正常写回。
        self.set_current_annotations(restored)
        self.save_current(annotations=restored)
        self._publish_annotations_changed(source="undo")
        return restored, None

    def redo(self, annotations=None, image_path=""):
        """重做被撤销的操作。返回 (restored_annotations, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致栈记录无效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not self._redo_stack:
            return None, "错误: 没有可重做的操作"
        next_state, err = self._pop_undo_frame(self._redo_stack, "重做")
        if err:
            return None, err
        current_state = {
            'annotations': copy.deepcopy(annotations),
            'image_path': image_path,
            'hidden_indices': set(self.hidden_indices or ()),
        }
        self._undo_stack.append(current_state)
        restored = copy.deepcopy(next_state['annotations'])
        self.hidden_indices = set(next_state.get('hidden_indices') or ())
        # 写回共享数据并保存，确保刷新后可见（同样不能用真值守卫，见 undo 的说明）
        self.set_current_annotations(restored)
        self.save_current(annotations=restored)
        self._publish_annotations_changed(source="redo")
        return restored, None

    def clear_undo_history(self):
        self._undo_stack = []
        self._redo_stack = []

    def has_undo(self):
        return len(self._undo_stack) > 0

    def has_redo(self):
        return len(self._redo_stack) > 0

    # ====== 区域: 标注编辑 ======
    # 原 EditorVM: delete_annotation / delete_annotation_points / change_annotation_label /
    #   clear_annotations / toggle_annotation_visibility / set_annotation_visibility /
    #   next_annotation_index / prev_annotation_index / resolve_tool_mode / save_current

    def delete_annotation_points(self, annotations, selected_points):
        """批量删除选中的多边形顶点。返回修改后的 annotations。"""
        self._remove_polygon_points(annotations, selected_points)
        return annotations

    @action("manual.delete_annotation",
            description="删除指定索引的标注。\n- index：标注索引（从0开始）\n- annotations：当前图片标注列表（可省略，缺省用 service 当前标注）\n- hidden_indices：已隐藏标注索引列表（可省略，缺省用 service 状态）\n- image_path：图片路径（可省略）\n- 优先基于 service 当前标注（与画布共享引用）操作，并自动保存\n- 返回 (annotations, hidden_indices, error)",
            category="标注", params={"index": "int", "annotations": "list"}, scope="agent")
    def delete_annotation(self, index=-1, annotations=None, hidden_indices=None, image_path=""):
        """删除指定索引的标注。返回 (annotations, hidden_indices, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致修改不生效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        hidden_indices = hidden_indices if hidden_indices is not None else self.hidden_indices
        if index < 0 or index >= len(annotations):
            return None, None, "错误: 标注索引无效"
        self.save_undo_state(annotations, image_path)
        new_annotations = [a for i, a in enumerate(annotations) if i != index]
        new_hidden = self.adjust_hidden_indices_after_deletion(hidden_indices, index)
        # 同步 service 可见性状态（数据权威源在 service）
        self.hidden_indices = new_hidden
        # 写回共享数据并保存，确保刷新后可见
        if self.current_annotations:
            self.current_annotations[:] = new_annotations
        self.save_current(annotations=new_annotations)
        self._publish_annotations_changed(source="delete_annotation")
        return new_annotations, new_hidden, None

    @action("manual.clear_annotations",
            description="清除当前图片的所有标注。\n- annotations：当前图片标注列表（可省略，缺省用 service 当前标注）\n- image_path：图片路径（可省略）\n- 优先基于 service 当前标注（与画布共享引用）操作，并自动保存\n- 返回 (annotations, error)，annotations 为清空后的空列表",
            category="标注", params={"annotations": "list", "image_path": "str"}, scope="agent")
    def clear_annotations(self, annotations=None, image_path=""):
        """清除当前图片所有标注。返回 (annotations, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致清除不生效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not annotations:
            return None, "错误: 当前没有标注"
        self.save_undo_state(annotations, image_path)
        # 清空标注时同步清空可见性状态
        self.hidden_indices.clear()
        # 写回共享数据并保存，确保刷新后可见
        if self.current_annotations:
            self.current_annotations[:] = []
        self.save_current(annotations=[])
        self._publish_annotations_changed(source="clear_annotations")
        return [], None

    @action("manual.add_annotation",
            description="添加标注到当前图片标注列表（直接修改 current_annotations）。\n"
                        "- annotation：标注 dict（含 label/bbox/points/type 等字段）\n"
                        "- 返回 {status, message, index, annotation_count}",
            category="标注", params={"annotation": "dict"}, scope="agent")
    def add_annotation(self, annotation):
        """添加标注到 current_annotations，返回结果 dict。"""
        if not isinstance(annotation, dict):
            return {"status": "error", "message": "annotation 必须是 dict"}
        self.current_annotations.append(annotation)
        self._publish_annotations_changed(source="add_annotation")
        return {"status": "success", "message": "标注已添加",
                "index": len(self.current_annotations) - 1,
                "annotation_count": len(self.current_annotations)}

    @action("manual.remove_annotation",
            description="删除指定索引的标注（直接修改 current_annotations）。\n"
                        "- index：标注索引（从0开始）\n"
                        "- 返回 {status, message, annotation_count}",
            category="标注", params={"index": "int"}, scope="agent")
    def remove_annotation(self, index=-1):
        """从 current_annotations 删除指定索引的标注，返回结果 dict。"""
        if index < 0 or index >= len(self.current_annotations):
            return {"status": "error", "message": f"标注索引 {index} 超出范围（共 {len(self.current_annotations)} 个标注）"}
        self.current_annotations.pop(index)
        if index in self.hidden_indices:
            self.hidden_indices.discard(index)
        self.hidden_indices = {i - 1 if i > index else i for i in self.hidden_indices}
        self._publish_annotations_changed(source="remove_annotation")
        return {"status": "success", "message": "标注已删除",
                "annotation_count": len(self.current_annotations)}

    @action("manual.draw_rect",
            description="创建矩形标注并添加到当前标注列表。\n"
                        "- x, y：矩形左上角图像坐标（像素）\n"
                        "- w, h：矩形宽高（像素）\n"
                        "- label：类别名称，留空使用当前类别\n"
                        "- 返回 {status, message, index, bbox, annotation_count}",
            category="标注", params={"x": "int", "y": "int", "w": "int", "h": "int", "label": "str"}, scope="agent")
    def draw_rect(self, x=100, y=100, w=200, h=200, label=""):
        """创建矩形标注 dict 并添加到 current_annotations。"""
        if not self.current_image_path:
            return {"status": "error", "message": "当前未加载图片"}
        if w <= 0 or h <= 0:
            return {"status": "error", "message": "矩形宽高必须大于0"}
        label = label or self.context.current_category or "object"
        annotation = {
            'label': label,
            'bbox': [x, y, w, h],
            'points': [[x, y], [x + w, y + h]],
            'type': 'rect',
            'shape_type': 'rectangle',
        }
        return self.add_annotation(annotation)

    @action("manual.update_annotation_label",
            description="修改指定索引标注的类别。\n- index：标注索引\n- category：新的类别名称\n- annotations：当前图片标注列表（可省略，缺省使用 service 当前标注数据）\n- categories：类别字典（可省略，使用当前类别列表）\n- 注意：优先基于 service 当前标注数据（与画布共享引用）修改，避免传入 get_state 快照副本导致修改不生效\n- 修改后自动保存到磁盘，界面已通过 annotation:annotations_changed 信号自动刷新\n- 返回 (annotations, error)",
            category="标注", params={"index": "int", "category": "str", "annotations": "list"}, scope="agent")
    def update_annotation_label(self, index=-1, category="", annotations=None, categories=None):
        """修改标注类别并自动保存。返回 (annotations, error)。"""
        if not category:
            return None, "错误: update_annotation_label 需要 category 参数"
        # 优先使用 service 当前标注数据（与画布共享引用）；
        # 传入的 annotations 可能是 get_state 快照副本，直接修改副本不会生效
        target = self.current_annotations if self.current_annotations else (annotations or [])
        if index < 0 or index >= len(target):
            return None, "错误: 标注索引无效"
        if categories is None:
            categories = self.context.categories
        result = self._set_annotation_label(target, index, category, categories)
        if result.get("success"):
            # 自动保存到磁盘，save_current 发布 annotations_changed 信号驱动界面刷新
            self.save_current(annotations=target)
            self._publish_annotations_changed(source="update_annotation_label")
            return target, None
        return None, "错误: 类别修改失败"

    @action("manual.toggle_annotation_visibility",
            description="切换标注的显示/隐藏。\n"
                        "- index 省略或为 -1：切换全部标注（全部可见→全部隐藏；存在隐藏→全部可见）\n"
                        "- index 为有效索引：切换该索引单个标注的可见性（可见→隐藏，隐藏→可见）\n"
                        "- 无标注或索引无效时返回错误\n"
                        "- 操作 service 中的 hidden_indices 数据；变更已通过 annotation:annotations_changed 信号自动刷新画布\n"
                        "- 典型用法：隐藏当前图片所有标注 → toggle_annotation_visibility(index=-1)，界面自动更新\n"
                        "- 隐藏当前选中标注：先用 edit.select_annotation_at 选中，再传入选中索引",
            category="标注", params={"index": "int"}, scope="agent")
    def toggle_annotation_visibility(self, index=-1):
        """切换标注可见性（操作 service 数据）。index=-1 时切换全部，否则切换单个。返回 error。"""
        count = len(self.current_annotations)
        if count == 0:
            return "错误: 当前没有标注"
        if index < 0:
            # 切换全部：全部可见→全部隐藏；存在隐藏→全部显示
            if len(self.hidden_indices) == 0:
                self.hidden_indices = set(range(count))
            else:
                self.hidden_indices.clear()
        else:
            # 切换单个
            if index >= count:
                return "错误: 标注索引无效"
            if index in self.hidden_indices:
                self.hidden_indices.discard(index)
            else:
                self.hidden_indices.add(index)
        self._publish_annotations_changed(source="toggle_annotation_visibility")
        return None

    @action("manual.set_annotation_visibility",
            description="设置指定标注为隐藏或显示。\n"
                        "- index：标注索引（从 0 开始）\n"
                        "- visible：True=显示该标注，False=隐藏该标注\n"
                        "- 索引无效时返回错误\n"
                        "- 操作 service 中的 hidden_indices 数据，变更已通过 annotation:annotations_changed 信号自动刷新画布",
            category="标注", params={"index": "int", "visible": "bool"}, scope="agent")
    def set_annotation_visibility(self, index=-1, visible=True):
        """设置指定标注可见性（操作 service 数据）。返回 error。"""
        if index < 0 or index >= len(self.current_annotations):
            return "错误: 标注索引无效"
        if visible:
            self.hidden_indices.discard(index)
        else:
            self.hidden_indices.add(index)
        self._publish_annotations_changed(source="set_annotation_visibility")
        return None

    def set_current_annotations(self, annotations, for_path=None, expected_generation=None):
        """整体替换当前图片标注列表（真值在 session，与 draw_area.annotations 同源）。

        `for_path` / `expected_generation` 非空时校验"这批标注是不是当前图的、代次是否
        仍然匹配"：不匹配则**拒绝装载**。这是审计 D2/D3 那类"把 A 图标注装进 B 图
        当前状态、随后被自动保存写进 B 图文件"的静默数据损坏的直接拦截点。
        """
        return self.session.set_annotations(annotations, for_path=for_path,
                                           expected_generation=expected_generation)

    def clear_visibility_state(self):
        """清空可见性状态（切图时由 interface 调用）。

        `loaded=False`：此刻我们**还不知道**新图有哪些标注（标注装载已改为后台
        进行），因此必须禁止保存 —— 否则会把上一张图的标注写进新图的文件。
        """
        self.hidden_indices.clear()
        self.session.set_annotations([], loaded=False)

    def next_annotation_index(self, count, current):
        """计算下一个标注索引（循环）。count=0 时返回错误。"""
        if count == 0:
            return None, "错误: 没有标注可切换"
        return (current + 1) % count if current >= 0 else 0, None

    def prev_annotation_index(self, count, current):
        """计算上一个标注索引（循环）。count=0 时返回错误。"""
        if count == 0:
            return None, "错误: 没有标注可切换"
        return (current - 1) % count if current >= 0 else count - 1, None

    def resolve_tool_mode(self, tool):
        """将工具字符串解析为绘制模式。返回 (mode, auto_task_mode, error)。"""
        from plugins.annotation.widgets.drawing_widget import DrawingMode
        mode_map = {
            'sam': DrawingMode.SAM, 'ai_rect': DrawingMode.AI_RECT,
            'rect': DrawingMode.RECT, 'rectangle': DrawingMode.RECT,
            'polygon': DrawingMode.POLYGON, 'poly': DrawingMode.POLYGON,
            'obb': DrawingMode.OBB, 'edit': DrawingMode.EDIT, 'roi': DrawingMode.ROI,
        }
        mode = mode_map.get(tool.lower())
        if mode is None:
            return None, None, "错误: 未知的工具类型 '%s'，可选: sam, ai_rect, rect, polygon, obb, edit, roi" % tool
        task_mode_map = {
            DrawingMode.RECT: 'det', DrawingMode.POLYGON: 'seg',
            DrawingMode.OBB: 'obb', DrawingMode.SAM: 'seg',
            DrawingMode.AI_RECT: 'seg',
        }
        return mode, task_mode_map.get(mode), None

    @action("manual.save_current",
            description="保存当前图片的标注到磁盘。\n- annotations：当前图片标注列表（可省略，缺省用当前画布/当前标注）\n- image_path：图片路径，留空使用当前图片\n- task_mode：seg/det/obb，留空使用当前任务模式\n- 返回 service 结果 dict（含 status/message）",
            category="标注", params={"annotations": "list", "image_path": "str", "task_mode": "str"}, scope="agent")
    def schedule_save_current(self, annotations=None, image_path="", task_mode=None):
        """把「保存当前标注」排入后台写入队列（调用方**不**等待结果）。

        与 save_current 共用同一套守卫/路径解析，区别只在于**写盘发生在后台线程**：

        * 同一路径**最新优先**：连续修改只会落一次盘（取最后一次快照），
          避免"拖一次鼠标写一次盘"；
        * 快照在调用线程（GUI）取好（deepcopy），工作线程只做纯 I/O，不碰 GUI 状态；
        * 标注装载在开始读之前调用 `wait_for_pending_writes`，保证"切走再切回"
          读到的是最新写入 —— 这是异步化**必须**配套的排序保证；
        * 事件在这里（提交时、GUI 线程）发布：语义是"标注集合变了"（触发刷新），
          而不是"磁盘写好了"。save_current 成功后发布的是同一个事件。
        """
        if task_mode is None:
            task_mode = self.context.task_mode
        if not image_path:
            image_path = self.current_image_path or ""
        if not image_path:
            return {"status": "error", "message": "当前未加载图片，无法保存"}
        # 与 save_current 完全一致的装载失败保护（审计 D8）
        if self.session.belongs_to_current(image_path) and not self.session.is_save_allowed:
            return {"status": "error",
                    "message": "当前图片的标注未成功装载（文件可能损坏或编码错误），"
                               "已拒绝保存以防覆盖原始标注文件"}

        source = annotations if annotations is not None else self.current_annotations
        snapshot = copy.deepcopy(list(source)) if source else []
        non_project = bool(self.context._is_non_project_mode)
        output_dir = self.context.output_dir
        categories = copy.deepcopy(self.context.categories) if self.context.categories else None
        export_fmt = self.context._non_project_format or "labelme"

        def _write():
            if non_project:
                os.makedirs(output_dir, exist_ok=True)
                return self.save_image_annotations(
                    image_path, snapshot, fmt=export_fmt,
                    output_dir=output_dir, categories=categories, project_mode=False)
            ann_dir = os.path.join(output_dir, "annotations")
            os.makedirs(ann_dir, exist_ok=True)
            return self.save_image_annotations(
                image_path, snapshot, fmt="labelme",
                output_dir=ann_dir, categories=categories, project_mode=True)

        if not self._writer.submit(image_path, _write):
            return {"status": "error", "message": "写入队列已关闭，无法保存"}
        if non_project:
            # 配置记录仍在 GUI 线程（settings 是共享状态，不能从工作线程写）
            self.save_non_project_config(task_mode)
        self._publish_annotations_changed(image_path=image_path,
                                         source="schedule_save_current")
        return {"status": "queued", "message": "已排入后台写入队列"}

    def wait_for_pending_writes(self, path, timeout_ms=5000):
        """等到该路径没有在途/排队的写入。供标注装载与显式保存用（读之前写完）。"""
        return self._writer.wait_for_path(path, timeout_ms)

    def set_save_failure_handler(self, handler):
        """设置后台保存失败的回调（在**后台线程**被调用，实现方需自行 marshal 到 GUI）。"""
        self._writer.set_failure_handler(handler)

    def drain(self, timeout_ms=5000):
        """Drainable 契约：排空后台写入队列并停止写入线程。"""
        return self._writer.drain(timeout_ms)

    def save_current(self, annotations=None, image_path="", task_mode=None):
        """保存当前图片标注到磁盘。返回 service 结果 dict。"""
        if task_mode is None:
            task_mode = self.context.task_mode
        if not image_path:
            image_path = self.current_image_path or ""
        if not image_path:
            return {"status": "error", "message": "当前未加载图片，无法保存"}
        # 装载失败保护（审计 D8）：标注文件损坏/编码错误时，界面会保留内存里的旧
        # 标注（以免误覆盖），但**绝不能落盘** —— 否则一次自动保存就把空标注写进
        # 那张图的标注文件，把"可恢复的损坏"变成"确定的数据丢失"。
        if self.session.belongs_to_current(image_path) and not self.session.is_save_allowed:
            return {"status": "error",
                    "message": "当前图片的标注未成功装载（文件可能损坏或编码错误），"
                               "已拒绝保存以防覆盖原始标注文件"}
        annotations = annotations if annotations is not None else self.current_annotations
        if self.context._is_non_project_mode:
            os.makedirs(self.context.output_dir, exist_ok=True)
            export_fmt = self.context._non_project_format or "labelme"
            res = self.save_image_annotations(
                image_path, annotations, fmt=export_fmt,
                output_dir=self.context.output_dir, categories=self.context.categories,
                project_mode=False
            )
            self.save_non_project_config(task_mode)
        else:
            output_dir = os.path.join(self.context.output_dir, "annotations")
            os.makedirs(output_dir, exist_ok=True)
            res = self.save_image_annotations(
                image_path, annotations, fmt="labelme",
                output_dir=output_dir, categories=self.context.categories,
                project_mode=True
            )
        if res.get("status") == "success":
            self._publish_annotations_changed(image_path=image_path, source="save_current")
        return res

    # ====== 区域: 副标注 ======
    # 原 EditorVM: load_secondary_annotations / clear_secondary_annotations /
    #   accept_secondary_annotation / add_secondary_as_primary
    #   （clear_all 合并入导航区，set_reference_annotation 为死代码未迁移）

    @action("manual.load_secondary_annotations",
            description="加载副标注目录并匹配主标注。\n- directory：副标注目录路径\n- image_path：图片路径，留空使用当前图片\n- primary_annotations：当前主标注列表\n- 返回结果 dict（status: success/empty/error）",
            category="标注", params={"directory": "str", "image_path": "str", "primary_annotations": "list"}, scope="agent")
    def load_secondary_annotations(self, directory="", image_path="", primary_annotations=None):
        """加载副标注并匹配主标注。返回结果 dict。"""
        if not image_path:
            image_path = self.current_image_path or ""
        if not image_path:
            return {"status": "error", "message": "当前未加载图像"}
        if not directory or not os.path.isdir(directory):
            return {"status": "error", "message": "副标注目录不存在或无效"}
        primary_annotations = primary_annotations or []
        self.secondary_dir = directory
        self.secondary_loaded = True
        secondary_file = self.find_secondary_annotation_file(
            image_path, self.secondary_dir
        )
        if not secondary_file:
            self.secondary_annotations = []
            self.secondary_annotation_map = {}
            self._publish_annotations_changed(source="load_secondary_annotations")
            return {"status": "empty", "message": "当前图像无对应标注文件"}
        try:
            annotations = self._load_secondary_annotations_file(
                secondary_file, image_path
            )
            if not annotations:
                self.secondary_annotations = []
                self.secondary_annotation_map = {}
                self._publish_annotations_changed(source="load_secondary_annotations")
                return {"status": "empty", "message": "标注文件为空"}
            self.secondary_annotations = annotations
            self.secondary_annotation_map = self._match_annotations_by_iou(
                primary_annotations, self.secondary_annotations
            )
            self._publish_annotations_changed(source="load_secondary_annotations")
            return {
                "status": "success",
                "matched_count": len(self.secondary_annotation_map),
                "unmatched": len(self.secondary_annotations) - len(self.secondary_annotation_map),
            }
        except Exception as e:
            self.secondary_annotations = []
            self.secondary_annotation_map = {}
            return {"status": "error", "message": str(e)}

    @action("manual.clear_secondary_annotations",
            description="清除已加载的副标注数据。\n- clear_dir：是否同时清除副标注目录信息（默认 False）\n- 成功返回 None，失败返回错误字符串",
            category="标注", params={"clear_dir": "bool"}, scope="agent")
    def clear_secondary_annotations(self, clear_dir=False):
        """清除副标注数据。成功返回 None，失败返回错误字符串。"""
        if not self.secondary_annotations:
            return "错误: 没有加载副标注"
        self.secondary_annotations = []
        self.secondary_annotation_map = {}
        if clear_dir:
            self.secondary_loaded = False
            self.secondary_dir = None
        self._publish_annotations_changed(source="clear_secondary_annotations")
        return None

    @action("manual.accept_secondary_annotation",
            description="将副标注结果写入指定主标注。\n- primary_idx：主标注索引\n- annotations：当前主标注列表（可省略，缺省用 service 当前标注）\n- image_path：图片路径（可省略）\n- 优先基于 service 当前标注（与画布共享引用）操作，并自动保存\n- 返回 (annotations, error)",
            category="标注", params={"primary_idx": "int", "annotations": "list"}, scope="agent")
    def accept_secondary_annotation(self, primary_idx=-1, annotations=None, image_path=""):
        """将副标注写入主标注。返回 (annotations, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致修改不生效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not self.secondary_loaded:
            return None, "错误: 未加载副标注"
        if primary_idx not in self.secondary_annotation_map:
            return None, "错误: 该主标注没有对应的副标注"
        if primary_idx < 0 or primary_idx >= len(annotations):
            return None, "错误: 主标注索引无效"
        secondary_idx = self.secondary_annotation_map[primary_idx]
        secondary_ann = self.secondary_annotations[secondary_idx]
        self.save_undo_state(annotations, image_path)
        new_annotations = list(annotations)
        new_annotations[primary_idx] = copy.deepcopy(secondary_ann)
        del self.secondary_annotation_map[primary_idx]
        # 写回共享数据并保存，确保刷新后可见
        if self.current_annotations:
            self.current_annotations[:] = new_annotations
        self.save_current(annotations=new_annotations)
        self._publish_annotations_changed(source="accept_secondary_annotation")
        return new_annotations, None

    @action("manual.add_secondary_as_primary",
            description="将指定副标注追加为主标注。\n- secondary_idx：副标注索引\n- annotations：当前主标注列表（可省略，缺省用 service 当前标注）\n- image_path：图片路径（可省略）\n- 优先基于 service 当前标注（与画布共享引用）操作，并自动保存\n- 返回 (annotations, new_primary_idx, error)",
            category="标注", params={"secondary_idx": "int", "annotations": "list"}, scope="agent")
    def add_secondary_as_primary(self, secondary_idx=-1, annotations=None, image_path=""):
        """将副标注复制为主标注。返回 (annotations, new_primary_idx, error)。"""
        # 优先基于 service 当前标注（与画布共享引用），避免 agent 传入快照副本导致修改不生效
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not self.secondary_loaded:
            return None, None, "错误: 未加载副标注，请先调用 load_secondary_annotations"
        if secondary_idx < 0 or secondary_idx >= len(self.secondary_annotations):
            return None, None, "错误: 副标注索引无效"
        secondary_ann = self.secondary_annotations[secondary_idx]
        self.save_undo_state(annotations, image_path)
        new_annotations, new_primary_idx, _ = self.add_secondary_annotation_to_primary(
            annotations, secondary_ann
        )
        self.secondary_annotation_map[new_primary_idx] = secondary_idx
        # 写回共享数据并保存，确保刷新后可见
        if self.current_annotations:
            self.current_annotations[:] = new_annotations
        self.save_current(annotations=new_annotations)
        self._publish_annotations_changed(source="add_secondary_as_primary")
        return new_annotations, new_primary_idx, None

    # ====== 区域: AI 辅助调用（薄封装）======
    # 原 EditorVM: predict_mask_with_bbox / predict_mask_with_points / predict_similar_in_image /
    #   convert_single_ann_to_obb / convert_single_ann_to_det / filter_duplicate_annotations /
    #   refine_annotation
    # build_annotation_from_bbox / build_annotation_from_polygon 已位于「筛选匹配 & 标注构建算法」区域
    # 算法实现位于 AutoAnnotationService；门面模式下 self.auto is self，
    # 需显式取 Auto 类方法（与 set_non_project_mode 同款守卫）。

    def _auto_invoke(self, name, *args, **kwargs):
        """委托 Auto 侧实现：独立实例直接调用，门面模式显式取 Auto 类方法。"""
        if self.auto is None:
            return {"status": "error", "message": "AutoAnnotationService 未注入"}
        if self.auto is self:
            from .auto_annotation_service import AutoAnnotationService
            return getattr(AutoAnnotationService, name)(self, *args, **kwargs)
        return getattr(self.auto, name)(*args, **kwargs)

    @action("manual.predict_mask_with_bbox",
            description="使用交互式分割模型根据矩形框预测掩码。\n- image_path：图片路径\n- bbox：矩形框 [x, y, width, height]（图像坐标）\n- model_type：交互模型名称，留空使用当前选中的交互模型\n- 返回 service 结果 dict（status=success 时 annotations[0] 为新标注）",
            category="标注", params={"image_path": "str", "bbox": "list", "model_type": "str"}, scope="agent")
    def predict_mask_with_bbox(self, image_path="", bbox=None, model_type=None):
        """根据 bbox 预测掩码。返回 service 结果 dict。"""
        if not image_path:
            return {"status": "error", "message": "缺少 image_path 参数"}
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return {"status": "error", "message": "bbox 参数无效，需为 [x, y, width, height]"}
        result = self._auto_invoke(
            "predict_mask_with_bbox", image_path, bbox, model_type=model_type
        )
        if result.get("status") == "success":
            self._publish_annotations_changed(image_path=image_path, source="predict_mask_with_bbox")
        return result

    def convert_single_ann_to_obb(self, ann):
        """将单个标注转换为 OBB 表示（就地修改）。"""
        self._auto_invoke("_convert_single_ann_to_obb", ann)

    def convert_single_ann_to_det(self, ann):
        """将单个标注转换为检测框表示（就地修改）。"""
        self._auto_invoke("_convert_single_ann_to_det", ann)

    def filter_duplicate_annotations(self, annotations, existing_annotations):
        """过滤与已有标注重复的新标注。返回过滤后的标注列表。"""
        return self._auto_invoke(
            "_filter_duplicate_annotations_hungarian", annotations, existing_annotations
        )

    @action("manual.predict_mask_with_points",
            description="使用交互式分割模型（SAM）根据点击点预测掩码。\n- image_path：图片路径\n- points_data：点击点列表，每项为 [x, y, label]，label 1=前景 0=背景\n- model_type：交互模型名称，留空使用当前选中的交互模型\n- 返回 service 结果 dict（status=success 时 annotations[0] 为新标注）",
            category="标注", params={"image_path": "str", "points_data": "list", "model_type": "str"}, scope="agent")
    def predict_mask_with_points(self, image_path="", points_data=None, model_type=None, refine=None, use_roi=True, img_size=None):
        """根据点位预测掩码。返回 service 结果 dict。"""
        if not image_path:
            return {"status": "error", "message": "缺少 image_path 参数"}
        if not points_data or not isinstance(points_data, list):
            return {"status": "error", "message": "points_data 参数无效，需为 [[x, y, label], ...]"}
        result = self._auto_invoke(
            "predict_mask_with_points", image_path, points_data, model_type=model_type,
            refine=refine, use_roi=use_roi, img_size=img_size
        )
        if result.get("status") == "success":
            self._publish_annotations_changed(image_path=image_path, source="predict_mask_with_points")
        return result

    @action("manual.predict_similar_in_image",
            description="在当前图片中查找与参考标注相似的目标。\n- image_path：图片路径\n- reference_annotations：参考标注列表（每个标注需包含 bbox 字段）\n- 返回 service 结果 dict（status=success 时 annotations 为找到的相似目标标注列表）",
            category="标注", params={"image_path": "str", "reference_annotations": "list"}, scope="agent")
    def predict_similar_in_image(self, image_path="", reference_annotations=None, rules=None, model_type=None, refine=None, use_roi=True, use_slice=True, img_size=None):
        """在当前图片中寻找相似目标。返回 service 结果 dict。"""
        if not image_path:
            return {"status": "error", "message": "缺少 image_path 参数"}
        if not reference_annotations or not isinstance(reference_annotations, list):
            return {"status": "error", "message": "reference_annotations 参数无效，需为标注列表"}
        result = self._auto_invoke(
            "predict_similar_in_image", image_path, reference_annotations, rules=rules,
            model_type=model_type, refine=refine, use_roi=use_roi,
            use_slice=use_slice, img_size=img_size
        )
        if result.get("status") == "success":
            self._publish_annotations_changed(image_path=image_path, source="predict_similar_in_image")
        return result

    @action("manual.refine_annotation",
            description="对指定索引的标注调用细化模型（vitmatte/cascadepsp/hybrid）精细化分割边缘。\n- index：标注在 annotations 列表中的索引\n- annotations：当前图片标注列表\n- image_path：图片路径\n- refine_method：细化模型，可选 vitmatte/cascadepsp/hybrid，留空使用当前选中的细化模型；hybrid 为混合分割（小目标经典CV、中大目标交互分割），可将矩形框标注细化为贴合轮廓的分割标注\n- 注意：det 模式下标注也可能携带 polygons（分割标注），本方法对分割/矩形框标注均适用\n- 成功后自动保存撤销状态并就地替换该标注的 polygons/bbox",
            category="标注", params={"index": "int", "annotations": "list", "image_path": "str", "refine_method": "str"}, scope="agent")
    def refine_annotation(self, index=-1, annotations=None, image_path="", refine_method=None):
        """调用 refine 模型细化指定标注。返回 service 结果 dict。"""
        annotations = self.current_annotations if self.current_annotations else (annotations or [])
        if not annotations:
            return {"status": "error", "message": "标注列表为空，无法细化"}
        if index < 0 or index >= len(annotations):
            return {"status": "error", "message": "标注索引无效"}
        ann = annotations[index]
        refine_method = refine_method or settings.get("refine_method", "vitmatte")
        if not ann.get('polygons') and not (refine_method == "hybrid" and self._annotation_bbox(ann)):
            return {"status": "error", "message": "标注没有多边形数据，无法细化"}
        if not image_path:
            return {"status": "error", "message": "当前未加载图片"}
        result = self._auto_invoke(
            "refine_annotation_polygons", image_path, ann.get('polygons', []),
            refine_method=refine_method, bbox=ann.get("bbox")
        )
        if result.get("status") == "success":
            self.save_undo_state(annotations, image_path)
            ann['polygons'] = result["polygons"]
            ann['bbox'] = result["bbox"]
            self.save_current(annotations=annotations)
            self._publish_annotations_changed(source="refine_annotation")
        return result

    # ====== 区域: 标注贴合度检查 ======
    # 混合分割方案：小目标（max(w,h)<30）用超分+经典CV前景提取，中大目标用交互式分割模型

    def _annotation_bbox(self, ann):
        """从标注 dict 提取图像坐标 bbox [x, y, w, h]，无 bbox 时用多边形包围盒。"""
        bbox = ann.get("bbox")
        if bbox and len(bbox) >= 4:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            try:
                w, h = float(w), float(h)
            except (TypeError, ValueError):
                w, h = 0.0, 0.0
            if w > 0 and h > 0:
                return [int(round(float(x))), int(round(float(y))), int(round(w)), int(round(h))]
        polygons = ann.get("polygons")
        if polygons:
            pts = polygons[0] if isinstance(polygons[0], list) else polygons
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if xs and ys:
                x, y = min(xs), min(ys)
                return [int(round(x)), int(round(y)), int(round(max(xs) - x)), int(round(max(ys) - y))]
        return None

    def _interactive_mask(self, img, box, model_type):
        """交互式分割（sam2 等）返回与原图等大的 0/255 掩码；失败返回 None。"""
        try:
            params = resolve_interactive_params(model_type, refine=False, img_size=1024)
            results = interactive_predict(img, bboxes=[list(box)], **params)
            if not results or results[0].masks is None:
                return None
            masks = results[0].masks.data.cpu().numpy()
            if masks.ndim != 3 or len(masks) == 0:
                return None
            m = masks[0]
            H, W = img.shape[:2]
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return (m > 0.5).astype(np.uint8)
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    @action("manual.fit_check_current", read_only=True,
            description="检查当前图片所有标注与实际目标轮廓的贴合度（混合分割）。\n- image_path：图片路径，留空使用当前图片\n- model_type：中大目标交互分割模型名，默认 sam2_l\n- 小目标（max(w,h)<30）使用超分+经典CV前景提取，中大目标使用交互式分割\n- 注意：det/obb 模式下标注也可能携带 polygons（分割标注），results 中 shape_type 字段区分 polygon（分割标注）与 bbox（矩形框标注），bbox 字段由标注的多边形外接框或矩形框得到\n- mask_iou 语义：shape_type=polygon 时表示检测掩码与标注多边形掩码的交并比（标注贴合度，通常应接近 bbox_iou 甚至更高）；shape_type=bbox 时表示检测掩码与标注框的交并比（框内填充度）\n- 返回每个标注的检出结果与贴合度指标（shape_type/area_ratio/bbox_iou/mask_iou/centroid_offset/overflow/deficit/boundary_gradient）及分割结果 polygons",
            category="标注", params={"image_path": "str", "model_type": "str"}, scope="agent")
    def fit_check_current(self, image_path="", model_type="sam2_l"):
        """检查当前图片标注贴合度。返回 {status, image_path, total, results}。"""
        image_path = image_path or self.current_image_path
        if not image_path:
            return {"status": "error", "message": "当前未加载图片，请先打开目录并选择图片"}
        annotations = self.current_annotations or []
        if not annotations:
            return {"status": "error", "message": "当前图片没有标注，无需检查"}
        img = imread_unicode(image_path)
        if img is None:
            return {"status": "error", "message": "图片读取失败: %s" % image_path}

        results = []
        for idx, ann in enumerate(annotations):
            item = {"idx": idx, "label": str(ann.get("label", ""))}
            item["shape_type"] = "polygon" if ann.get("polygons") else "bbox"
            box = self._annotation_bbox(ann)
            if box is None or box[2] <= 0 or box[3] <= 0:
                item["detected"] = 0
                item["error"] = "invalid bbox"
                results.append(item)
                continue
            x, y, w, h = box
            item["bbox"] = [x, y, w, h]
            mask = None
            method = ""
            if max(w, h) < SMALL_BOX_THRESHOLD:
                mask = traditional_mask(img, (x, y, w, h))
                method = "CV"
            else:
                mask = self._interactive_mask(img, (x, y, w, h), model_type)
                method = "SAM2"
            item["method"] = method
            item["boundary_gradient"] = round(boundary_gradient(img, (x, y, w, h)), 1)
            if mask is None or int(np.count_nonzero(mask)) <= 0:
                item["detected"] = 0
                results.append(item)
                continue
            item["detected"] = 1
            ann_mask = None
            if ann.get("polygons"):
                ann_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for poly in ann["polygons"]:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(ann_mask, [pts], 255)
            metrics = fit_metrics(mask, (x, y, w, h), ann_mask=ann_mask)
            if metrics:
                item.update(metrics)
            polygon = mask_to_polygon(mask)
            hybrid_bbox = mask_to_bbox(mask)
            if polygon is not None and hybrid_bbox is not None:
                item["polygons"] = polygon
                item["hybrid_bbox"] = hybrid_bbox
            results.append(item)

        self._publish_annotations_changed(image_path=image_path, source="fit_check_current")
        return {"status": "success", "image_path": image_path, "total": len(results), "results": results}

    # ====== 区域: 难样本 ======
    # 原 annotation_io: toggle_hard_sample / save_hard_samples / load_hard_samples

    def _toggle_hard_sample(self, hard_samples, norm_path, is_hard, description=""):
        result = dict(hard_samples)
        if is_hard:
            result[norm_path] = description
        else:
            result.pop(norm_path, None)
        return result

    def save_hard_samples(self, hard_samples, project_name, project_service,
                          is_non_project_mode=False, config_path=None, config=None):
        hard_list = [{"path": p, "description": d} for p, d in hard_samples.items()]
        if is_non_project_mode:
            if config_path and config is not None:
                merged = dict(config)
                merged["hard_samples"] = hard_list
                try:
                    with atomic_open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(merged, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"[AnnotationService] 保存难样本失败: {e}")
        else:
            if project_name and project_service:
                project_service.update_project_field(project_name, "hard_samples", hard_list)

    def load_hard_samples(self, project_name, project_service,
                          is_non_project_mode=False, output_dir=None):
        hard_samples = {}
        raw_list = []
        if is_non_project_mode:
            config_path = os.path.join(output_dir, "project_info.json") if output_dir else None
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8-sig') as f:
                        cfg = json.load(f)
                    raw_list = cfg.get("hard_samples", [])
                except Exception:
                    pass
        else:
            if project_name and project_service:
                try:
                    info = project_service.load_project(project_name)
                    if info:
                        raw_list = info.get("hard_samples", [])
                except Exception:
                    pass
        for item in raw_list:
            if isinstance(item, dict):
                p = os.path.normpath(item.get("path", ""))
                if p:
                    hard_samples[p] = item.get("description", "")
            elif isinstance(item, str):
                hard_samples[os.path.normpath(item)] = ""
        return hard_samples

    # ====== 区域: 分割 & 非项目配置 ======
    # 原 annotation_io: save_dataset_splits / load_dataset_splits / save_non_project_config / load_non_project_config

    def _save_dataset_splits_to_file(self, output_dir, image_splits):
        split_path = os.path.join(output_dir, "dataset_split.json")
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        try:
            normalized = {os.path.normpath(k): v for k, v in image_splits.items()}
            with atomic_open(split_path, 'w', encoding='utf-8') as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)
        except Exception:
            print(f"[AnnotationService] Failed to save dataset splits")

    @action("manual.dataset_splits",
            description="保存或加载数据集划分（train/val/test）。\n- op：save=保存到磁盘，load=从磁盘加载\n- output_dir：输出目录，留空使用当前项目/输出目录\n- save 返回 True/False，load 返回划分字典 {图片路径: 划分名}",
            category="标注", params={"op": "str", "output_dir": "str"}, scope="agent")
    def dataset_splits(self, op="load", output_dir=""):
        if not output_dir:
            output_dir = self.context.output_dir or ""
        if op == "save":
            if not output_dir:
                return False
            self._save_dataset_splits_to_file(output_dir, self.image_splits)
            self._publish_datasets_changed(op="save", image_path=self.current_image_path)
            return True
        if not output_dir:
            return {}
        split_path = os.path.join(output_dir, "dataset_split.json")
        self.image_splits = {}
        if os.path.exists(split_path):
            try:
                with open(split_path, 'r', encoding='utf-8-sig') as f:
                    raw = json.load(f)
                self.image_splits = {os.path.normpath(k): v for k, v in raw.items()}
            except Exception:
                pass
        self._publish_datasets_changed(op="load", image_path=self.current_image_path)
        return self.image_splits

    def _save_non_project_config_file(self, output_dir, config):
        config_path = os.path.join(output_dir, "project_info.json")
        merged = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8-sig') as f:
                    merged = json.load(f)
            except Exception:
                pass
        merged.update(config)
        merged["is_non_project"] = True
        try:
            with atomic_open(config_path, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[AnnotationService] 非项目模式保存配置失败: {e}")

    def load_non_project_config(self, output_dir):
        """加载非项目配置文件并写入共享状态。返回 config dict，无配置返回 None。

        （原 ProjectVM.load_non_project_config 合并版）
        """
        config_path = os.path.join(output_dir, "project_info.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        except Exception:
            return None
        if not config.get("is_non_project"):
            return None

        task_mode = config.get("task_mode", "seg")
        self.set_task_mode(task_mode)
        project_settings.set("task_mode", task_mode)

        categories_data = config.get("categories", {})
        if categories_data:
            cats = {}
            for name, color in categories_data.items():
                try:
                    cats[name] = QColor(color)
                except Exception:
                    cats[name] = self._category_color(name)
            self.context.categories = cats

        current_cat = config.get("current_category", "")
        if current_cat and current_cat in self.context.categories:
            self.set_current_category(current_cat)
            project_settings.set("current_category", current_cat)
            self.touch_recent_category(current_cat)

        export_fmt = config.get("export_format", "labelme")
        if export_fmt:
            self.context._non_project_format = export_fmt

        rules = config.get("rules")
        if rules:
            rules, migrated = migrate_legacy_rules(rules, config.get("rules_schema"))
            self.context.current_project_rules = rules
            if migrated:
                print(f"[AnnotationService] 非项目目录的旧版相对规则已迁移为绝对规则: {rules}")
                self._save_non_project_config_file(output_dir, {
                    "rules": rules,
                    "rules_schema": RULES_SCHEMA_VERSION,
                })

        hard_samples = {}
        for item in config.get("hard_samples", []):
            if isinstance(item, dict):
                p = os.path.normpath(item.get("path", ""))
                if p:
                    hard_samples[p] = item.get("description", "")
            elif isinstance(item, str):
                hard_samples[os.path.normpath(item)] = ""
        self.context.hard_samples = hard_samples
        return config

    # ====== 区域: 副标注加载 ======
    # 原 annotation_io: find_secondary_annotation_file / load_secondary_annotations

    def find_secondary_annotation_file(self, image_path, secondary_dir):
        if not image_path or not secondary_dir:
            return None
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for ext in ['.json', '.txt', '.xml']:
            candidate = os.path.join(secondary_dir, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        for subdir in ['annotations', 'labels', 'labelme', 'coco_annotations', 'voc_annotations']:
            for ext in ['.json', '.txt', '.xml']:
                candidate = os.path.join(secondary_dir, subdir, base_name + ext)
                if os.path.exists(candidate):
                    return candidate
        return None

    def _load_secondary_annotations_file(self, secondary_file, image_path=None):
        """按扩展名加载副标注文件，返回标注列表（无标注时返回空列表）"""
        ext = os.path.splitext(secondary_file)[1].lower()
        if ext == '.json':
            return load_annotations(secondary_file).get("annotations", [])
        if ext in ('.txt', '.xml'):
            result = load_annotations(secondary_file, image_path=image_path)
            return result.get("annotations", [])
        return []

    # ====== 区域: 搜索路径 / 加载 ======
    # 原 annotation_io: get_search_dirs / find_and_load_annotations / infer_split_from_path

    def get_search_dirs(self, output_dir, is_non_project_mode):
        if not output_dir:
            return []
        if is_non_project_mode:
            return [output_dir] if os.path.exists(output_dir) else []
        ann_dir = os.path.join(output_dir, "annotations")
        return [ann_dir] if os.path.exists(ann_dir) else []

    def find_and_load_annotations(self, image_path, mode, fmt,
                                   output_dir, is_non_project_mode, non_project_format=None):
        search_dirs = self.get_search_dirs(output_dir, is_non_project_mode)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        fmt_config = settings.get("annotation_formats", {})
        coco_cfg = fmt_config.get("coco", {"ext": ".json", "subdir": "coco_annotations", "filename": "_annotations.coco.json"})
        coco_filename = coco_cfg.get("filename", "_annotations.coco.json")

        load_failed = []      # 文件存在但装载失败（解析/编码错误）
        empty_files = []      # 文件存在且可解析，但里面没有标注
        for d in search_dirs:
            if is_non_project_mode:
                files_to_check = (
                    [os.path.join(d, f"{base_name}.txt"), os.path.join(d, f"{base_name}.json"), os.path.join(d, coco_filename)]
                    if fmt in ("yolo", "yoloseg", "yoloobb") else
                    [os.path.join(d, coco_filename), os.path.join(d, f"{base_name}.json"), os.path.join(d, f"{base_name}.txt")]
                    if fmt == "coco" else
                    [os.path.join(d, f"{base_name}.json"), os.path.join(d, f"{base_name}.txt"), os.path.join(d, coco_filename)]
                )
                for file_path in files_to_check:
                    if os.path.exists(file_path):
                        try:
                            ext = os.path.splitext(file_path)[1].lower()
                            load_kwargs = {"image_path": image_path} if (os.path.basename(file_path) == coco_filename or ext == '.txt') else {}
                            result = load_annotations(file_path, **load_kwargs)
                        except Exception as exc:
                            # 文件存在但装载抛错：这是**错误**，必须与"还没有标注文件"区分
                            load_failed.append("%s (%s: %s)" % (file_path, type(exc).__name__, exc))
                            continue
                        if result.get("annotations"):
                            fmt_type = 'coco' if os.path.basename(file_path) == coco_filename else (
                                self.detect_yolo_format(file_path) if ext == '.txt' else 'labelme')
                            return {'status': 'success', 'annotations': result["annotations"], 'source_file_type': fmt_type}
                        empty_files.append(file_path)
            else:
                labelme_path = os.path.join(d, f"{base_name}.json")
                if os.path.exists(labelme_path):
                    try:
                        result = load_annotations(labelme_path)
                    except Exception as exc:
                        load_failed.append("%s (%s: %s)" % (labelme_path, type(exc).__name__, exc))
                        continue
                    if result.get("annotations"):
                        return {'status': 'success', 'annotations': result["annotations"], 'source_file_type': 'labelme'}
                    empty_files.append(labelme_path)
        if load_failed:
            # 必须与"这张图还没有标注文件"（下面的 'empty'）区分开：界面据此**禁止保存**，
            # 避免把空标注覆盖到那张可恢复的坏文件上（审计 D8）。原先两者共用 'error'，
            # 于是"全新的图"也被判定为装载失败 -> **永远无法保存** —— 这是阶段 2 引入
            # 的严重回归，直到阶段 9 补上保存路径的回归测试才被抓到。
            return {'status': 'error', 'annotations': [],
                    'message': '标注文件存在但装载失败: %s' % "; ".join(load_failed[:3])}
        # 'empty'：这张图**还没有**标注（或标注文件本来就是空的）—— 不是错误，
        # 界面应允许保存，否则新数据集根本无法标注。
        return {'status': 'empty', 'annotations': [], 'message': 'No annotations found'}

    def infer_split_from_path(self, norm_path):
        path_lower = norm_path.lower().replace("\\", "/")
        if "/train/" in path_lower:
            return "train"
        elif "/val/" in path_lower:
            return "val"
        elif "/test/" in path_lower:
            return "test"
        return "train"

    # ====== 区域: 格式检测 & 目录扫描 ======
    # 原 annotation_io: detect_yolo_format / detect_coco_task_type / auto_detect_format

    def detect_yolo_format(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return "yolo"
                parts = first_line.split()
                if len(parts) >= 5:
                    if len(parts) == 9:
                        return "yoloobb"
                    elif len(parts) == 5:
                        return "yolo"
                    return "yoloseg"
            return "yolo"
        except Exception:
            return "yolo"

    def detect_coco_task_type(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            for ann in data.get('annotations', []):
                seg = ann.get('segmentation')
                if seg and ((isinstance(seg, list) and len(seg) > 0) or
                            (isinstance(seg, dict) and 'counts' in seg)):
                    return 'seg'
            return 'det'
        except Exception:
            return 'det'

    def auto_detect_format(self, dir_path):
        import glob
        format_patterns = [
            ("coco", "coco_annotations", "*_annotations.coco.json"),
            ("labelme", "annotations", "*.json"),
            ("yoloseg", "labels", "*.txt"),
            ("yolo", "labels", "*.txt"),
            ("yoloobb", "labels", "*.txt"),
            ("voc", "voc_annotations", "*.xml"),
            ("mask", "masks", "*.png"),
        ]
        detected_format = None
        detected_categories = []
        detected_task_type = None
        for fmt, subdir, pattern in format_patterns:
            search_dir = os.path.join(dir_path, subdir)
            if not os.path.exists(search_dir):
                search_dir = dir_path
            files = glob.glob(os.path.join(search_dir, pattern))
            if files:
                if fmt == "labelme":
                    is_labelme = False
                    for f in files:
                        try:
                            with open(f, 'r', encoding='utf-8-sig') as fp:
                                data = json.load(fp)
                            if 'shapes' in data:
                                is_labelme = True
                                break
                        except Exception:
                            pass
                    if not is_labelme:
                        continue
                    detected_format, detected_task_type = "labelme", 'seg'
                elif fmt in ("yoloseg", "yolo", "yoloobb"):
                    detected_format = self.detect_yolo_format(files[0])
                    detected_task_type = {'yoloobb': 'obb', 'yoloseg': 'seg'}.get(detected_format, 'det')
                else:
                    detected_format = fmt
                    detected_task_type = 'seg' if fmt == "mask" else ('det' if fmt == "voc" else self.detect_coco_task_type(files[0]))
                try:
                    from core.backend.formats.factory import scan_categories
                    categories = scan_categories(dir_path, detected_format)
                    detected_categories = [cat.get("name", "") for cat in categories if cat.get("name")]
                except Exception:
                    pass
                break
        return detected_format, detected_categories, detected_task_type

    # ====== 区域: 筛选匹配 & 标注构建算法 ======
    # 原 annotation_io: _check_annotation_match / _check_annotation_size / build_annotation_from_bbox /
    #   build_annotation_from_polygon / check_annotation_in_roi / _match_annotations_by_iou / _calculate_annotations_iou

    def _check_annotation_match(self, annotation,
                               filter_category=None,
                               filter_size_range=None,
                               filter_width_range=None,
                               filter_height_range=None):
        """检查单个标注是否满足类别/尺寸/宽高筛选条件"""
        if filter_category is not None and annotation.get('label') != filter_category:
            return False
        bbox = annotation.get('bbox')
        if not bbox:
            return False
        _, _, w, h = bbox
        area = w * h
        if filter_size_range is not None:
            min_a, max_a = filter_size_range
            if area < min_a or (max_a is not None and area > max_a):
                return False
        if filter_width_range is not None:
            min_w, max_w = filter_width_range
            if w < min_w or (max_w is not None and w > max_w):
                return False
        if filter_height_range is not None:
            min_h, max_h = filter_height_range
            if h < min_h or (max_h is not None and h > max_h):
                return False
        return True

    def _check_annotation_size(self, annotation, min_area, max_area):
        """检查标注面积是否在给定范围内"""
        bbox = annotation.get('bbox')
        if not bbox:
            return False
        _, _, w, h = bbox
        area = w * h
        if area < min_area:
            return False
        if max_area is not None and area > max_area:
            return False
        return True

    def build_annotation_from_bbox(self, bbox, label, shape_type='rectangle', mask=None):
        ann = {'bbox': bbox, 'label': label, 'shape_type': shape_type}
        if mask is not None:
            ann['mask'] = mask
        return ann

    def build_annotation_from_polygon(self, polygons, label, shape_type='polygon', image_path=None):
        data = polygons[0] if isinstance(polygons[0], list) else polygons
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        ann = {'polygons': [data],
               'bbox': [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
               'label': label, 'shape_type': shape_type}
        if shape_type == 'polygon' and image_path:
            img = imread_unicode(image_path)
            if img is not None:
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(data, dtype=np.int32)], 255)
                ann['mask'] = mask.tolist()
        return ann

    def check_annotation_in_roi(self, ann_bbox, roi_bbox):
        if not ann_bbox or not roi_bbox:
            return False
        ax, ay, aw, ah = ann_bbox
        rx, ry, rw, rh = roi_bbox
        return (ax + aw / 2 >= rx and ay + ah / 2 >= ry and
                ax + aw / 2 <= rx + rw and ay + ah / 2 <= ry + rh)

    def _match_annotations_by_iou(self, primary, secondary, iou_threshold=0.3):
        """基于 IoU 匹配主标注和副标注，返回 {primary_idx: secondary_idx}"""
        result = {}
        used_secondary = set()
        for i, p in enumerate(primary):
            best_idx = -1
            best_iou = iou_threshold
            for j, s in enumerate(secondary):
                if j in used_secondary:
                    continue
                iou = self._calculate_annotations_iou(p, s)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0:
                result[i] = best_idx
                used_secondary.add(best_idx)
        return result

    def _calculate_annotations_iou(self, ann1, ann2):
        """计算两个标注之间的 IoU：优先 polygon mask 精确计算，退化到 bbox IoU"""
        polygons1 = ann1.get('polygons')
        polygons2 = ann2.get('polygons')
        if polygons1 and polygons2:
            try:
                arr1 = np.asarray(polygons1[0])
                arr2 = np.asarray(polygons2[0])
            except (TypeError, ValueError, IndexError):
                pass
            else:
                if arr1.size > 0 and arr2.size > 0:
                    bbox1 = ann1.get('bbox')
                    bbox2 = ann2.get('bbox')
                    if bbox1 and bbox2:
                        x1, y1, w1, h1 = bbox1
                        x2, y2, w2, h2 = bbox2
                        ux = float(min(x1, x2))
                        uy = float(min(y1, y2))
                        ux2 = float(max(x1 + w1, x2 + w2))
                        uy2 = float(max(y1 + h1, y2 + h2))
                        uw, uh = int(np.ceil(ux2 - ux)), int(np.ceil(uy2 - uy))
                        if uw > 0 and uh > 0:
                            mask1 = np.zeros((uh, uw), dtype=np.uint8)
                            mask2 = np.zeros((uh, uw), dtype=np.uint8)
                            for poly in polygons1:
                                pts = np.asarray(poly, dtype=np.float64)
                                if pts.ndim == 3 and pts.shape[-1] >= 2:
                                    pts = pts.reshape(-1, pts.shape[-1])
                                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
                                    continue
                                pts = pts[:, :2].copy()
                                pts[:, 0] -= ux
                                pts[:, 1] -= uy
                                cv2.fillPoly(mask1, [np.round(pts).astype(np.int32)], 1)
                            for poly in polygons2:
                                pts = np.asarray(poly, dtype=np.float64)
                                if pts.ndim == 3 and pts.shape[-1] >= 2:
                                    pts = pts.reshape(-1, pts.shape[-1])
                                if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
                                    continue
                                pts = pts[:, :2].copy()
                                pts[:, 0] -= ux
                                pts[:, 1] -= uy
                                cv2.fillPoly(mask2, [np.round(pts).astype(np.int32)], 1)
                            inter = np.logical_and(mask1, mask2).sum()
                            union = np.logical_or(mask1, mask2).sum()
                            if union > 0:
                                return float(inter) / float(union)
        bbox1, bbox2 = ann1.get('bbox'), ann2.get('bbox')
        if bbox1 and bbox2:
            return calculate_iou(bbox1, bbox2)
        return 0.0

    # ====== 区域: AI action（导航/状态，agent 可调用）======

    @action("manual.get_state", read_only=True,
            description="获取手动标注服务当前状态：图片总数、当前图片索引/路径、图片尺寸、当前标注摘要（类别与坐标）、筛选条件、类别列表、当前类别、任务模式、数据集划分等。",
            category="导航", scope="agent")
    def get_state(self):
        """返回手动标注服务的当前状态 dict。"""
        current_index = -1
        if self.current_image_path and self.image_files:
            try:
                current_index = self.image_files.index(os.path.normpath(self.current_image_path))
            except ValueError:
                current_index = -1
        # 当前图片尺寸（带缓存，避免反复读图）
        image_width = image_height = None
        if self.current_image_path:
            cached = self._image_size_cache.get(self.current_image_path)
            if cached:
                image_width, image_height = cached
            else:
                try:
                    img = imread_unicode(self.current_image_path)
                    if img is not None:
                        image_height, image_width = img.shape[:2]
                        self._image_size_cache[self.current_image_path] = (image_width, image_height)
                except Exception:
                    pass
        # 当前标注摘要（label + bbox），便于 Agent 确认已有标注无需重复绘制
        annotations_summary = []
        for ann in (self.current_annotations or []):
            if not isinstance(ann, dict):
                continue
            annotations_summary.append({
                "label": ann.get("label", ""),
                "bbox": ann.get("bbox", []),
            })
        # 当前图片的划分/难样本状态（只保留当前图片，避免 get_state 把全项目路径
        # 映射混入对话，造成大量噪音与路径隐私泄露）
        current_image_split = None
        if self.current_image_path:
            current_image_split = self.image_splits.get(
                os.path.normpath(self.current_image_path)
            )
        is_hard_sample = False
        if self.current_image_path:
            is_hard_sample = (
                os.path.normpath(self.current_image_path)
                in self.context.hard_samples
            )
        return {
            "status": "success",
            "image_count": len(self.image_files),
            "current_index": current_index,
            "current_image_path": self.current_image_path,
            "current_dir": self.current_dir,
            "image_width": image_width,
            "image_height": image_height,
            "annotation_count": len(self.current_annotations or []),
            "annotations": annotations_summary,
            "task_mode": self.context.task_mode,
            "current_category": self.context.current_category,
            "categories": list(self.context.categories.keys()),
            "recent_categories": list(self.context._recent_categories),
            "filter_category": self.filter_category,
            "filter_size_range": list(self.filter_size_range) if self.filter_size_range else None,
            "filter_width_range": list(self.filter_width_range) if self.filter_width_range else None,
            "filter_height_range": list(self.filter_height_range) if self.filter_height_range else None,
            "filter_annotated": self.filter_annotated,
            "filter_split": self.filter_split,
            "is_non_project_mode": self.context._is_non_project_mode,
            "output_dir": self.context.output_dir,
            "current_image_split": current_image_split,
            "is_hard_sample": is_hard_sample,
        }

    @action("manual.open_directory",
            description="打开图片目录作为当前工作目录。\n- dir_path：项目目录或普通图片目录的路径\n- 自动扫描目录中的图片（jpg/png/jpeg/bmp 等）并设置当前目录\n- 返回 {status, image_count, image_files}",
            category="导航", params={"dir_path": "str"}, scope="agent")
    def open_directory(self, dir_path=""):
        """扫描并加载目录图片，作为 AI 手动标注的工作集。"""
        if not dir_path:
            return {"status": "error", "message": "缺少 dir_path 参数"}
        if not os.path.isdir(dir_path):
            return {"status": "error", "message": "目录不存在或无效: %s" % dir_path}
        files, last_idx = self.load_directory(dir_path)
        if not files:
            return {"status": "error", "message": "目录中没有找到图片", "image_files": []}
        return {"status": "success", "image_count": len(files), "last_index": last_idx, "image_files": files}

    @action("manual.select_image",
            description="跳转到指定索引的图片并设置当前图片。\n- image_index：从0开始的图片索引\n- 图片须先通过 manual.open_directory 加载\n- 返回 {status, image_path, image_index, split}",
            category="导航", params={"image_index": "int"}, scope="agent")
    def select_image(self, image_index=0):
        """设置当前图片索引，返回该图片路径与推断的数据集划分。"""
        if not self.image_files:
            return {"status": "error", "message": "没有已加载的图片，请先调用 manual.open_directory"}
        if not isinstance(image_index, int) or image_index < 0 or image_index >= len(self.image_files):
            return {"status": "error", "message": "图片索引越界: %s，共 %d 张" % (image_index, len(self.image_files)),
                    "image_count": len(self.image_files)}
        # 切换前自动保存当前图片标注（与界面 switch_image/on_file_item_clicked 行为一致，
        # 受 auto_save_enabled 控制）。此时 current_image_path/current_annotations 仍指向旧图，
        # 若不保存，切图后旧图内存标注会被新图标注替换而丢失，或残留旧标注随
        # current_image_path 变更被误写入新图片文件造成覆盖。
        if getattr(self.context, "auto_save_enabled", False) and self.current_image_path:
            try:
                self.save_current()
            except Exception as e:
                print(f"[ManualAnnotationService] 切换图片前自动保存失败: {e}")
        path = self.image_files[image_index]
        split = self.set_current_image_with_split(path)
        self.save_last_index(image_index)
        self._publish_image_changed(image_index)
        return {"status": "success", "image_path": path, "image_index": image_index,
                "split": split, "image_count": len(self.image_files)}

    def set_event_bus(self, bus):
        """注入事件总线，用于 service 数据变更后通知 interface 层刷新 UI。"""
        self._event_bus = bus

    def _publish_image_changed(self, image_index):
        """发布图片切换事件，由 interface 层订阅后调用 load_image 刷新画布/文件列表。"""
        self._emit(StateType.IMAGE_CHANGED, {"image_index": image_index})

    def _publish_project_selected(self, project_info):
        """发布项目已选中事件，通知 annotation/version 等界面刷新（agent 切项目后自动刷新 UI）。"""
        self._emit(StateType.PROJECT_SELECTED, project_info)

    def _publish_annotations_changed(self, image_path=None, source=""):
        """发布标注集合变更事件，由 interface 层订阅后刷新标注列表/画布。"""
        self._emit(StateType.ANNOTATIONS_CHANGED, {
            "image_path": image_path or self.current_image_path,
            "source": source or "",
        })

    def _publish_categories_changed(self, categories=None, reason=""):
        """发布类别集合变更事件，由 interface 层订阅后刷新类别列表/颜色。"""
        self._emit(StateType.CATEGORIES_CHANGED, {
            "categories": categories if categories is not None else self.context.categories,
            "reason": reason or "",
        })

    def _publish_filters_changed(self):
        """发布筛选条件变更事件，由 interface 层订阅后刷新文件可见性/筛选控件。"""
        self._emit(StateType.FILTERS_CHANGED, {
            "filter_category": self.filter_category,
            "size_range": self.filter_size_range,
            "width_range": self.filter_width_range,
            "height_range": self.filter_height_range,
        })

    def _publish_datasets_changed(self, op="", image_path=None, split=""):
        """发布数据集划分变更事件，由 interface 层订阅后刷新分割下拉/文件徽标。"""
        self._emit(StateType.DATASETS_CHANGED, {
            "op": op or "",
            "image_path": image_path or self.current_image_path,
            "split": split or "",
        })

    def _publish_hard_sample_changed(self, image_path=None, is_hard=False):
        """发布难样本标记变更事件，由 interface 层订阅后刷新难样本勾选框/计数。"""
        self._emit(StateType.HARD_SAMPLE_CHANGED, {
            "image_path": image_path or self.current_image_path,
            "is_hard": bool(is_hard),
        })

    def _emit(self, type_, data=None):
        """统一状态事件发布：经全局 StateRouter 信号分发。"""
        if self._event_bus:
            self._event_bus.publish_state(type_, data)

    @action("manual.navigate",
            description="按方向切换当前图片。\n- direction：1=下一张，-1=上一张\n- 已在边界时返回 noop\n- 返回 {status, image_path, image_index}。",
            category="导航", params={"direction": "int"}, scope="agent")
    def navigate(self, direction: int = 1):
        return self._navigate_ai(direction)

    def _navigate_ai(self, direction):
        """按方向切换当前图片（内部辅助）。"""
        if not self.image_files:
            return {"status": "error", "message": "没有已加载的图片，请先调用 manual.open_directory"}
        current_idx = -1
        if self.current_image_path:
            try:
                current_idx = self.image_files.index(os.path.normpath(self.current_image_path))
            except ValueError:
                current_idx = -1
        visible_indices = self.visible_image_indices()   # 唯一真值来源（审计 3e）
        target, error = self.navigate_relative(current_idx, direction, visible_indices)
        if error:
            return {"status": "error", "message": error}
        if target is None:
            return {"status": "noop", "message": "已在边界，无更多图片",
                    "image_index": current_idx, "image_count": len(self.image_files)}
        return self.select_image(target)
