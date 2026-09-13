import copy
import json
import os
import re
import threading
import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, Slot, Signal, QSize, QPoint, QEvent, QThread, QTimer
from PySide6.QtGui import QPixmap, QIcon, QColor, QCursor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QListWidgetItem,
                               QSplitter, QApplication, QSizePolicy, QDialog, QProgressDialog, QLabel,
                               QListWidget)
from qfluentwidgets import (PrimaryPushButton, PushButton, TransparentToolButton, StrongBodyLabel, CaptionLabel,
                            LineEdit, SearchLineEdit, FluentIcon as FIF, InfoBar, ListWidget,
                            SwitchButton, ComboBox, CheckBox, MessageBox, RoundMenu, Action)

from core.common.background_task import BackgroundTaskRunner
from core.common.main_thread_dispatcher import run_on_main
from core.common.widgets.base import Interface, ProgressDialog
from plugins.annotation.widgets.drawing_widget import DrawingWidget, DrawingMode
from plugins.annotation.widgets.thumbnail_manager import ThumbnailManager
from core.common.widgets.rule_config_dialog import RuleConfigDialog
from core.common.widgets.example_selection_dialog import ExampleLibraryDialog
from core.common.widgets.prompt_library_dialog import PromptLibraryDialog
from core.common.icons import AppIcon
from core.common.settings import settings
from core.common.project_settings import project_settings
from core.service.dataset_service import DatasetService
from core.service.project_service import ProjectService
from core.service.common_service import PromptManager
from core.common.action_registry import action
from .annotation_dialogs import (CategorySelectionDialog, ConversionDialog,
                                 AddCustomModelDialog, ModelSelectionDialog,
                                 FitCheckDialog, CustomSizeFilterDialog)
from . import get_category_color

# Project root calculation
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class _ReferencePopup(QWidget):
    """搜索框 @ 引用候选弹层（Tool 无焦点窗口，键盘事件仍留在搜索框）

    不使用 QLineEdit/QCompleter：QCompleter 的 complete() 在本项目环境
    （PySide6 6.10 + qfw SearchLineEdit）下会触发原生崩溃，且其补全只认
    行首前缀，无法支持行中 token。此处自管弹层，焦点保持输入框。

    多选：点击候选即把 @token 追加进搜索框并实时搜索，弹层保持打开；
    底部「完成筛选」按钮统一收起弹层收尾（避免逐条 setText 触发弹层重建闪烁）。
    """

    activated = Signal(str)  # 点击一个候选（追加/替换 @token）
    applyRequested = Signal()  # 点「完成筛选」按钮（收起弹层）

    def __init__(self):
        super().__init__()
        # WindowDoesNotAcceptFocus: 点击弹层不抢焦点(Windows WS_EX_NOACTIVATE)。
        # 否则按下瞬间搜索框 FocusOut → eventFilter 直接 hide 弹层,
        # itemClicked 永远不会触发(表现为点击无法填充 + 弹层闪烁)
        self.setWindowFlags(
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setObjectName("SearchRefPopup")
        self.setStyleSheet(
            "QWidget#SearchRefPopup { background: #2b2b2b;"
            " border: 1px solid #444; border-radius: 6px; }"
            "QListWidget { background: transparent; color: #e5e7eb; border: none;"
            " outline: none; font-size: 13px; }"
            "QListWidget::item { padding: 5px 12px; border-radius: 4px; }"
            "QListWidget::item:hover { background: #3d3d3d; }"
            "QListWidget::item:selected { background: rgba(0, 122, 255, 0.35); color: #ffffff; }"
        )
        self._list = QListWidget(self)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(lambda item: self.activated.emit(item.text()))

        self._apply_btn = PrimaryPushButton("完成筛选", self)
        self._apply_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._apply_btn.clicked.connect(self.applyRequested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 6, 5, 7)
        layout.setSpacing(6)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._apply_btn)

    # ---- 代理到内部列表，保持对外接口不变 ----
    def clear(self):
        self._list.clear()

    def addItems(self, entries):
        self._list.addItems(entries)

    def setCurrentRow(self, row):
        self._list.setCurrentRow(row)

    def currentRow(self):
        return self._list.currentRow()

    def currentItem(self):
        return self._list.currentItem()

    def item(self, row):
        return self._list.item(row)

    def count(self):
        return self._list.count()


def _route(path):
    """生成转发到领域服务指定属性的 property（保持测试断言语法兼容）

    path 前缀映射：context.* → self.context；manual.* → self.manual；auto.* → self.auto
    """
    parts = path.split(".")

    def getter(self):
        obj = getattr(self, parts[0])
        for p in parts[1:]:
            obj = getattr(obj, p)
        return obj

    def setter(self, value):
        obj = getattr(self, parts[0])
        for p in parts[1:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    return property(getter, setter)


class CategoryItemWidget(QWidget):
    """类别列表项：颜色方块 + 名称 + 重命名/删除图标"""

    rename_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, name: str, color, parent=None):
        super().__init__(parent)
        self._name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # 颜色方块
        self._color_label = QLabel()
        self._color_label.setFixedSize(14, 14)
        pixmap = QPixmap(14, 14)
        if isinstance(color, QColor):
            c = color
        elif isinstance(color, str):
            c = QColor(color)
            if not c.isValid():
                c = QColor(0, 255, 255)
        elif hasattr(color, '__len__') and len(color) >= 3:
            try:
                c = QColor(*[int(v) for v in color[:3]])
            except (TypeError, ValueError):
                c = QColor(0, 255, 255)
        else:
            c = QColor(0, 255, 255)
        pixmap.fill(c)
        self._color_label.setPixmap(pixmap)
        layout.addWidget(self._color_label)

        # 类别名称
        self._name_label = QLabel(name)
        self._name_label.setObjectName("CategoryItemName")
        layout.addWidget(self._name_label, 1)

        # 重命名图标
        self._rename_btn = TransparentToolButton(FIF.EDIT)
        self._rename_btn.setFixedSize(22, 22)
        self._rename_btn.setToolTip("重命名")
        self._rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rename_btn.clicked.connect(lambda: self.rename_requested.emit(self._name))
        layout.addWidget(self._rename_btn)

        # 删除图标
        self._delete_btn = TransparentToolButton(FIF.DELETE)
        self._delete_btn.setFixedSize(22, 22)
        self._delete_btn.setToolTip("删除")
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._name))
        layout.addWidget(self._delete_btn)


class AnnotationInterface(Interface):
    # 信号定义：用于延迟执行项目设置，避免 showEvent 竞争条件
    project_setup_requested = Signal(dict)  # project_info

    # ---------- 状态属性路由（读写转发到领域服务） ----------
    categories = _route("context.categories")  # 类别字典
    _recent_categories = _route("context._recent_categories")  # 最近使用的类别顺序
    current_category = _route("context.current_category")  # 当前选中类别
    project_has_own_categories = _route("context.project_has_own_categories")
    current_image_path = _route("manual.current_image_path")  # 当前图片路径
    image_files = _route("manual.image_files")  # 图片文件列表
    auto_save_enabled = _route("context.auto_save_enabled")  # 自动保存开关
    outputs_root = _route("context.outputs_root")
    output_dir = _route("context.output_dir")  # 输出目录
    model_type = _route("auto.model_type")  # 自动标注模型类型
    interactive_model_type = _route("auto.interactive_model_type")
    reference_annotation = _route("manual.reference_annotation")  # 参考标注
    example_mode_enabled = _route("manual.example_mode_enabled")  # 示例模式开关
    one_click_running = _route("auto.one_click_running")  # 一键标注运行中
    one_click_paused = _route("auto.one_click_paused")  # 一键标注已暂停
    batch_image_list = _route("auto.batch_image_list")  # 批量图片列表
    batch_current_idx = _route("auto.batch_current_idx")  # 批量当前索引
    batch_mode = _route("auto.batch_mode")  # "text" or "example"
    batch_prompt_data = _route("auto.batch_prompt_data")
    batch_rules = _route("auto.batch_rules")
    image_splits = _route("manual.image_splits")  # 数据集划分
    current_project_name = _route("context.current_project_name")  # 当前项目名称
    current_project_rules = _route("context.current_project_rules")  # 当前项目规则
    filter_category = _route("manual.filter_category")  # 筛选类别
    filter_size_range = _route("manual.filter_size_range")  # 目标大小范围
    filter_width_range = _route("manual.filter_width_range")  # 宽度筛选范围
    filter_height_range = _route("manual.filter_height_range")  # 高度筛选范围
    annotation_cache = _route("manual.annotation_cache")  # 标注缓存
    prompt_library = _route("context.prompt_library")  # 提示词库
    _is_non_project_mode = _route("context._is_non_project_mode")  # 非项目标注模式
    _non_project_format = _route("context._non_project_format")
    _non_project_save_dir = _route("context._non_project_save_dir")
    hard_samples = _route("context.hard_samples")  # 难样本标记
    _undo_stack = _route("manual._undo_stack")  # 撤销栈
    _redo_stack = _route("manual._redo_stack")  # 重做栈
    _max_undo_steps = _route("manual._max_undo_steps")
    _is_undo_redo = _route("manual._is_undo_redo")  # 正在执行撤销/重做
    secondary_annotations = _route("manual.secondary_annotations")  # 副标注列表
    secondary_annotation_map = _route("manual.secondary_annotation_map")
    secondary_loaded = _route("manual.secondary_loaded")  # 是否已加载副标注
    secondary_dir = _route("manual.secondary_dir")  # 副标注文件夹路径
    plain_model_type = _route("auto.plain_model_type")  # 当前普通模型
    _sam_active = _route("manual._sam_active")  # SAM 交互状态
    _last_sam_annotation = _route("manual._last_sam_annotation")
    current_dir = _route("manual.current_dir")  # 当前目录

    def __init__(self, project_service=None, parent=None):
        super().__init__('AnnotationInterface', parent)
        # Phase 2: 直接实例化领域服务（共享 ProjectContext）
        from ..services.project_context import ProjectContext
        from ..services.manual_annotation_service import ManualAnnotationService
        from ..services.auto_annotation_service import AutoAnnotationService
        self.context = ProjectContext()  # 手动/自动共享的项目状态
        self.manual = ManualAnnotationService(self.context, project_service)  # 手动标注领域服务
        self.auto = AutoAnnotationService(self.context)  # 自动标注领域服务
        self.manual.auto = self.auto  # service 间互访
        self.auto.manual = self.manual
        self.auto.project_service = project_service
        self.project_service = project_service  # 项目服务
        # We'll use the same service as AutoAnnotation for SAM
        self.model_manager = self.auto.model_manager  # 模型管理器（异步加载 + 模型列表）
        # 自动保存（silent=True）已改为后台写盘，调用方拿不到返回值 —— 因此写盘失败
        # **必须**显式上报到界面，否则就是"静默丢数据"（审计 D35 那类编码失败会被吞掉）。
        self.manual.set_save_failure_handler(self._on_background_save_failed)
        self.dataset_service = DatasetService()  # 数据集服务
        saved_plain_model = settings.get("plain_model", "")
        if saved_plain_model:
            self.auto.plain_model_type = saved_plain_model
        
        # Category state
        self.categories = {}  # 类别字典
        self._recent_categories = []  # 最近使用的类别顺序（用于快捷切换）
        self.current_category = project_settings.get("current_category") or settings.get("current_category", "")  # 当前选中类别
        self.project_has_own_categories = False
        # 注意：current_image_path 是**只读 property**，真值在 manual.session
        # （唯一状态所有者）。原先 interface 自己存一份，是审计 W2 的三份真值之一。
        self._event_bus = None  # 由 AnnotationPlugin.on_load 注入
        self.image_files = []  # 图片文件列表

        # 标注装载的后台执行器（"IO 全在后台"：逐图标注的读盘 + 解析不再占用 GUI
        # 线程）。单任务 + 最新优先：正在跑时新请求只覆盖
        # `_pending_annotation_request`，因此连按方向键不会积压一串读盘任务。
        # 实现 Drainable 契约，关闭时由 ShutdownCoordinator 排空。
        self._annotation_runner = BackgroundTaskRunner(parent=self, max_threads=1,
                                                       name="annotation.load")
        self._pending_annotation_request = None

        # 搜索扫描的后台执行器（审计 D15）：带筛选的搜索要逐图读盘解析标注，
        # 原先在每次击键的 GUI 线程上同步执行。现在击键只置脏 + 去抖，扫描在后台。
        self.SEARCH_DEBOUNCE_MS = 150
        self._search_runner = BackgroundTaskRunner(parent=self, max_threads=1,
                                                   name="annotation.search")
        self._pending_search_request = None
        self._search_generation = 0
        self._search_debounce = QTimer(self)
        self._search_debounce.setSingleShot(True)
        self._search_debounce.setInterval(self.SEARCH_DEBOUNCE_MS)
        self._search_debounce.timeout.connect(self._on_search_debounce_timeout)
        # 列表重建闸门（审计 D16）：重建会销毁列表行及其子控件，绝不能发生在
        # 某个子控件的信号发射栈上
        self._list_rebuilding = False
        
        # Save settings
        self.auto_save_enabled = True  # 自动保存开关
        self.outputs_root = settings.get("outputs_root", os.path.join(PROJECT_ROOT, "projects"))
        # 如果 outputs_root 为空，使用默认值
        if not self.outputs_root:
            self.outputs_root = os.path.join(PROJECT_ROOT, "projects")
        self.output_dir = settings.get("output_dir", os.path.join(self.outputs_root, "default_project"))  # 输出目录
        self.model_type = settings.get("example_model", "trtsam3")  # 自动标注模型类型
        self.interactive_model_type = settings.get("default_interactive_model", "sam_b")
        
        # Ensure outputs_root exists
        os.makedirs(self.outputs_root, exist_ok=True)
        
        self.reference_annotation = None  # 参考标注
        self.example_mode_enabled = False  # 示例模式开关
        
        # Batch processing state
        self.one_click_running = False  # 一键标注运行中
        self.one_click_paused = False  # 一键标注已暂停
        self.batch_image_list = []  # 批量图片列表
        self.batch_current_idx = 0  # 批量当前索引
        self.batch_mode = None # "text" or "example"
        self.batch_prompt_data = None
        self.batch_rules = None
        self._batch_progress_dialog = None  # 批量自动标注进度对话框（BATCH_PROGRESS 事件驱动）
        
        self.image_splits = {} # image_path -> "train"|"val"|"test"  # 数据集划分
        self.load_dataset_splits()
        
        # 项目规则
        self.current_project_name = None  # 当前项目名称
        self.current_project_rules = {}  # 当前项目规则
        
        # 筛选状态
        self.filter_category = None  # 筛选类别
        self.filter_size_range = None  # 当前筛选的目标大小范围 (min_area, max_area)
        self.filter_width_range = None  # 宽度筛选范围 (min_w, max_w)
        self.filter_height_range = None  # 高度筛选范围 (min_h, max_h)
        self.annotation_cache = {}  # 标注缓存

        # 提示词库
        self.prompt_library = []  # 提示词库
        self.load_prompt_library()

        # 项目切换防重入标志
        self._is_setting_project = False
        self._pending_project_info = None

        # 非项目标注模式标志
        self._is_non_project_mode = False  # True表示非项目标注模式
        self._non_project_format = None   # 非项目模式下自动识别的标注格式

        # Initialize UI first so widgets exist
        self.init_ui()
        self.apply_persistent_settings()

        # Then check for current project
        self.init_project()

        # 连接模型加载信号
        self.model_manager.model_load_started.connect(self._on_model_load_started)
        self.model_manager.model_load_finished.connect(self._on_model_load_finished)

        # 连接项目设置信号（使用 Qt.QueuedConnection 避免 showEvent 竞争条件）
        self.project_setup_requested.connect(self._do_set_project, Qt.QueuedConnection)

        # 注意：全局快捷键通过 QShortcut 在 setup_global_shortcuts() 中设置

    def set_event_bus(self, bus):
        self._event_bus = bus

    def on_annotation_image_changed(self, data):
        """service 层 agent 切图后，同步刷新画布与文件列表（配合 load_image 的幂等保护）。

        必须 force_reload：service 的 select_image 在发事件前已先行更新
        current_image_path（接口经 _route 与该属性共享同一数据），默认的
        “同一图片即跳过”幂等判断会误判并提前返回，导致画布/对象列表
        停留在旧图片、旧标注残留。
        """
        try:
            index = int((data or {}).get("image_index", -1))
            if index >= 0:
                self.load_image(index, force_reload=True)
        except (TypeError, ValueError):
            pass

    def _annotations_state_signature(self):
        """计算当前标注/可见性/副标注的轻量签名，用于 annotations_changed 事件的防重入去重。

        service 的嵌套调用（如 delete_annotation -> save_current）会连续发布多次
        annotations_changed，签名未变化时跳过刷新，避免重复刷新同一状态。

        **签名必须覆盖"真实编辑"的维度**（审计 D33）：原实现只记录
        `len(a.get("polygons"))` —— 移动一个多边形顶点、改一条多边形的形状，
        标注元素个数与标签、bbox 都没变 -> 签名不变 -> **刷新被抑制**，
        画布与列表停留在旧几何上。因此这里把几何也纳入签名：
          * bbox 用数值元组（而不是 `str()`，避免格式差异造成假变化）；
          * 每条多边形的**顶点坐标**参与；
          * SAM 点、rotated/obb 的额外字段一并覆盖。
        签名只用于"是否跳过同一次事件的重复刷新"，因此宁可灵敏也不要漏判。
        """
        anns = getattr(self.manual, "current_annotations", None) or []

        def _geom(a):
            polys = []
            for poly in (a.get("polygons") or []):
                try:
                    polys.append(tuple((float(p[0]), float(p[1])) for p in poly))
                except (TypeError, ValueError, IndexError):
                    polys.append(("?",))
            return tuple(polys)

        def _bbox(a):
            bbox = a.get("bbox")
            try:
                return tuple(float(v) for v in bbox) if bbox else ()
            except (TypeError, ValueError):
                return (str(bbox),)

        brief = []
        for a in anns:
            if isinstance(a, dict):
                brief.append((str(a.get("label", "")), _bbox(a), _geom(a),
                              str(a.get("shape_type", ""))))
        hidden = frozenset(getattr(self.manual, "hidden_indices", None) or ())
        sec = len(getattr(self.manual, "secondary_annotations", None) or [])
        return (len(brief), tuple(brief), hidden, sec)

    def reset_annotations_refresh_state(self):
        """清掉"上次已刷新过的标注签名"缓存（审计 D33）。

        这份缓存是**派生状态**（用于同一事件内的重复刷新去重），不是标注真值本身。
        但它必须在"标注真值被整体替换"的时刻失效 —— 原先 `clear_project` /
        `_do_set_project` 不重置它，切换项目后若新项目的标注恰好与旧的同签名，
        刷新就会被跳过，画布/列表停留在旧内容上。
        """
        self._last_ann_path = None
        self._last_ann_sig = None

    def on_annotations_changed(self, data):
        """订阅 annotation:annotations_changed：标注集合变更后刷新标注列表与画布。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            changed_path = (data or {}).get("image_path")
            if changed_path and os.path.normpath(str(changed_path)) != os.path.normpath(self.current_image_path or ""):
                return  # 非当前图片的标注变更，无需刷新本界面
            sig = self._annotations_state_signature()
            if (getattr(self, "_last_ann_path", None), getattr(self, "_last_ann_sig", None)) == (self.current_image_path, sig):
                return
            self._last_ann_path = self.current_image_path
            self._last_ann_sig = sig
            # 同步 service 最新当前标注到画布（共享引用下通常已是同一列表，此处为保险）
            self.manual.set_current_annotations(self.manual.current_annotations)
            self.refresh_label_list()
            self.draw_area.update()
        except Exception:
            if settings.DEBUG:
                raise

    def on_categories_changed(self, data):
        """订阅 annotation:categories_changed：类别集合/当前类别变更后刷新类别列表与颜色。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            self._refresh_category_state(update_list=True, update_colors=True, save=False)
        except Exception:
            if settings.DEBUG:
                raise

    def on_filters_changed(self, data):
        """订阅 annotation:filters_changed：筛选条件变更后刷新文件可见性并同步筛选控件。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            self._refresh_file_visibility(force=True)
        except Exception:
            if settings.DEBUG:
                raise

    def on_rules_changed(self, data):
        """订阅 annotation:rules_changed：agent set_rule 后,若规则配置对话框
        正在显示则实时刷新其控件值;未打开则无操作。"""
        try:
            dialog = getattr(self, "_rule_dialog", None)
            if dialog is not None and dialog.isVisible():
                rules = (data or {}).get("rules") or {}
                dialog.set_rules_to_ui(rules)
        except Exception:
            if settings.DEBUG:
                raise

    def on_datasets_changed(self, data):
        """订阅 annotation:datasets_changed：数据集划分变更后同步分割下拉与文件徽标。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            split = ""
            if self.current_image_path:
                split = self.manual.image_splits.get(os.path.normpath(self.current_image_path), "") or ""
            split_idx = {"train": 0, "val": 1, "test": 2}.get(split, 0)
            self.split_combo.blockSignals(True)
            self.split_combo.setCurrentIndex(split_idx)
            self.split_combo.blockSignals(False)
        except Exception:
            if settings.DEBUG:
                raise

    def on_hard_sample_changed(self, data):
        """订阅 annotation:hard_sample_changed：难样本标记变更后刷新勾选框与计数。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            changed_path = (data or {}).get("image_path")
            if changed_path and os.path.normpath(str(changed_path)) != os.path.normpath(self.current_image_path or ""):
                return  # 非当前图片的难样本变更，仅刷新计数即可
            self._update_hard_sample_checkbox()
            hard_samples = getattr(self.manual.context, "hard_samples", None) or {}
            self.hard_sample_count_label.setText(str(len(hard_samples)))
        except Exception:
            if settings.DEBUG:
                raise

    def on_task_mode_changed(self, data):
        """订阅 annotation:task_mode_changed：同步画布任务模式。"""
        try:
            if getattr(self, "_is_setting_project", False):
                return
            task_mode = (data or {}).get("task_mode")
            if task_mode:
                self.draw_area.task_mode = task_mode
        except Exception:
            if settings.DEBUG:
                raise

    # ---------------- 批量自动标注进度对话框 ----------------

    _BATCH_MODE_TITLES = {
        "text": "基于文本自动标注",
        "example": "基于示例自动标注",
        "model": "基于普通模型自动标注",
    }

    def on_batch_progress(self, data):
        """订阅 annotation:batch_progress：批量自动标注时显示/更新/关闭进度对话框。

        两种发起路径都必须把事件 marshal 回主线程，因此这里更新控件是安全的：
        菜单/快捷键发起时批量跑在 BackgroundTaskRunner 的线程上，agent 发起时跑在
        LLMWorker 线程上（service 侧 `batch.one_click_by_all` 标了 background=True）。
        两条路径都不再需要 processEvents —— GUI 线程在批量期间始终空闲。
        """
        try:
            data = data or {}
            phase = data.get("phase")
            if phase == "start":
                self._show_batch_progress(data)
            elif phase == "progress":
                self._update_batch_progress(data)
            elif phase == "finished":
                self._close_batch_progress()
        except Exception:
            if settings.DEBUG:
                raise

    def _show_batch_progress(self, data):
        """创建并显示批量标注进度对话框（带取消按钮）。

        批量标注可能由菜单在主线程同步执行（此时用 WindowModal 阻止用户在批量
        过程中触发其它标注操作），也可能由 agent 在后台线程执行（此时沿用导出
        等长任务的 NonModal 样式，用户仍可操作界面）。两种情况下 EventBus 都会
        保证该回调在主线程执行。
        """
        total = max(1, int(data.get("total") or 0))
        self._close_batch_progress()
        title = self._BATCH_MODE_TITLES.get(data.get("mode"), "自动标注")
        dialog = ProgressDialog(title, f"共 {total} 张图片，正在自动标注...",
                                parent=self, has_cancel=True)
        app = QApplication.instance()
        on_gui_thread = app is None or QThread.currentThread() is app.thread()
        dialog.setWindowModality(Qt.WindowModal if on_gui_thread else Qt.NonModal)
        dialog.setRange(0, total)
        dialog.setValue(0)
        dialog.canceled.connect(self._on_batch_progress_canceled)
        # 点关闭/按 Esc 也视为取消，避免对话框消失后批量标注在无可见进度下继续跑
        dialog.rejected.connect(self._on_batch_progress_canceled)
        dialog.show()
        self._batch_progress_dialog = dialog

    def _update_batch_progress(self, data):
        """更新批量标注进度（进度条 + 当前图片）。"""
        dialog = getattr(self, "_batch_progress_dialog", None)
        if dialog is None:
            return
        total = max(1, int(data.get("total") or 0))
        processed = max(0, int(data.get("processed") or 0))
        dialog.setRange(0, total)
        dialog.setValue(min(processed, total))
        image = data.get("image") or ""
        if image:
            dialog.setLabelText(f"正在标注 ({processed}/{total}): {os.path.basename(image)}")
        # 这里**不再**调用 QApplication.processEvents()：批量标注已由
        # BackgroundTaskRunner 放到后台线程执行，GUI 线程本来就空闲，Qt 会自然重绘。
        # 在状态变更处理器内重入事件循环会让用户在批量进行中触发切图/保存/删除，
        # 那正是审计列的 P1 重入缺陷。

    def _close_batch_progress(self):
        """关闭批量标注进度对话框（程序化关闭，不再触发取消信号）。"""
        dialog = getattr(self, "_batch_progress_dialog", None)
        self._batch_progress_dialog = None
        if dialog is not None:
            try:
                dialog.blockSignals(True)
                dialog.close()
                dialog.deleteLater()
            except RuntimeError:
                pass

    def _on_batch_progress_canceled(self):
        """用户取消（取消按钮 / 关闭 / Esc）：请求中止批量自动标注。"""
        self._close_batch_progress()
        try:
            self.auto.cancel_batch()
        except Exception:
            pass

    def _send_reference(self, ref_text: str):
        """把引用文本通过统一状态信号广播到 Agent 对话输入框(不自动发送)。"""
        if self._event_bus:
            self._event_bus.publish_state("annotation.reference_added", ref_text)

    def _on_model_load_started(self, model_type):
        """模型加载开始回调"""
        self.status_label.setText(f"正在后台加载模型: {model_type}...")
        # 可以添加进度条等 UI 反馈

    def _on_model_load_finished(self, success, message):
        """模型加载结束回调"""
        if success:
            self.status_label.setText(f"模型加载成功: {message}")
            InfoBar.success("加载成功", message, duration=2000, parent=self)
        else:
            self.status_label.setText(f"模型加载失败: {message}")
            InfoBar.error("加载失败", message, duration=3000, parent=self)

    def init_project(self):
        """初始化当前项目上下文"""
        project_name = settings.get("current_project_name")
        if project_name:
            # Try to load from project service indirectly
            project_path = os.path.join(PROJECT_ROOT, "projects", project_name)
            info_path = os.path.join(project_path, "project_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8-sig') as f:
                        info = json.load(f)
                        info['name'] = project_name
                        info['path'] = project_path
                        self.set_project(info)
                        return True
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    if settings.DEBUG:
                        raise
                    print(f"Warning: Corrupted project_info.json for {project_name}: {e}")
                    # If corrupted, try to use project_service to fix/reload it
                    if self.project_service:
                        try:
                            info = self.manual.load_project(project_name)
                            if info:
                                return True
                        except:
                            if settings.DEBUG:
                                raise
                            pass
        return False

    def save_project_rules(self, rules=None):
        """保存规则到当前项目（委托 VM 处理规则持久化）"""
        self.auto.save_rules(rules)
        if rules is not None or self.current_project_rules:
            print(f"AnnotationInterface: Saved rules for project {self.current_project_name}")

    def _refresh_category_state(self, update_list=True, update_colors=True, save=True):
        """统一刷新类别相关 UI 和持久化状态"""
        if update_list:
            self.refresh_category_list()
        if update_colors:
            self.draw_area.set_category_colors(self.categories)
        if save:
            self.save_categories_to_settings()

    def set_project(self, project_info):
        """切换项目并更新所有路径和配置 - 通过信号队列延迟执行"""
        new_project_name = project_info.get('name', 'Unknown')
        print(f"[AnnotationInterface] Queueing project setup: {new_project_name}")
        
        # 防重入检查：如果正在设置项目，记录待处理的项目
        if self._is_setting_project:
            print(f"[AnnotationInterface] Project setup in progress, queuing: {new_project_name}")
            self._pending_project_info = project_info
            return
        
        # 发射信号，使用 QueuedConnection 确保在当前事件循环结束后执行
        self.project_setup_requested.emit(project_info)

    def _do_set_project(self, project_info):
        """实际执行项目设置（在信号槽中调用）"""
        new_project_name = project_info.get('name', 'Unknown')
        
        if self._is_setting_project:
            print(f"[AnnotationInterface] Rejected duplicate project setup: {new_project_name}")
            return
        
        self._is_setting_project = True
        
        # 项目切换会整体替换标注真值：派生签名缓存必须作废（D33）
        self.reset_annotations_refresh_state()
        
        try:
            print(f"[AnnotationInterface] Executing project setup: {new_project_name}")

            # 退出非项目模式
            self._is_non_project_mode = False
            self._non_project_format = None

            # 恢复示例库为项目模式（从当前项目加载）
            self.manual.set_non_project_mode(None)

            self.one_click_running = False
            self.one_click_paused = False

            # 立即作废所有在途图像解码（非阻塞）。解码线程由 DrawingWidget 内部的
            # QThreadPool 持有，所以这里不需要也不能 wait/terminate —— 递增代次后
            # 在途任务会自行发现过期并丢弃结果。放在清列表之前，避免旧项目的解码
            # 结果在这段窗口里被提交到画布上。
            self.draw_area.cancel_pending_load()
            
            # 清掉"当前图"（走 session 的唯一入口；interface 不再自己存一份）
            self.manual.set_current_image(None)
            self.image_files = []
            
            self.file_list.blockSignals(True)
            try:
                while self.file_list.count() > 0:
                    self.file_list.takeItem(0)
            except Exception as e:
                print(f"[AnnotationInterface] Error clearing file_list: {e}")
                traceback.print_exc()
            self.file_list.blockSignals(False)
            
            try:
                self.draw_area.clear()
            except Exception as e:
                print(f"[AnnotationInterface] Error clearing draw_area: {e}")
                traceback.print_exc()
            
            if project_info.get('name'):
                project_settings.load_project(project_info['name'])
            task_mode = self.manual.setup_project(project_info)
            if task_mode:
                self.draw_area.task_mode = task_mode

            try:
                self._refresh_category_state(save=False)
            except Exception as e:
                print(f"[AnnotationInterface] Error updating UI components: {e}")
                traceback.print_exc()

            if os.path.exists(self.output_dir):
                try:
                    self.load_directory_images(self.output_dir)
                except Exception as e:
                    print(f"[AnnotationInterface] Error loading directory images: {e}")
                    traceback.print_exc()

            try:
                self.load_dataset_splits()
            except Exception as e:
                print(f"[AnnotationInterface] Error loading dataset splits: {e}")
                traceback.print_exc()

            # 更新UI状态（退出非项目模式）
            self._update_non_project_mode_ui()

            print(f"[AnnotationInterface] Project set to {project_info['name']}")

            try:
                self._refresh_category_state(update_list=False, save=False)
            except Exception as e:
                print(f"[AnnotationInterface] Error setting category colors: {e}")
                traceback.print_exc()

            try:
                self.update_mode_buttons(self.draw_area.mode)
            except Exception as e:
                print(f"[AnnotationInterface] Error updating mode buttons: {e}")
                traceback.print_exc()

            try:
                example_count = self.auto.load_support_sets()
                if example_count > 0:
                    self.status_label.setText(f"示例库已切换，当前项目有 {example_count} 个示例")
                else:
                    self.status_label.setText("示例库已切换，当前项目暂无示例")
            except Exception as e:
                print(f"[AnnotationInterface] Error loading support sets: {e}")
                traceback.print_exc()

            # 重新加载当前项目的提示词库
            try:
                self.load_prompt_library()
                prompt_count = len(self.prompt_library)
                if prompt_count > 0:
                    self.status_label.setText(f"{self.status_label.text()}，提示词库有 {prompt_count} 个提示词")
                else:
                    self.status_label.setText(f"{self.status_label.text()}，提示词库为空")
            except Exception as e:
                print(f"[AnnotationInterface] Error loading prompt library: {e}")
                traceback.print_exc()

            try:
                self.draw_area.reset_view()
            except Exception as e:
                print(f"[AnnotationInterface] Error resetting view: {e}")
                traceback.print_exc()

            try:
                self.draw_area.update()
            except Exception as e:
                print(f"[AnnotationInterface] Error updating draw_area: {e}")
                traceback.print_exc()

            print(f"[AnnotationInterface] Project setup completed successfully")

            # 加载难样本列表
            self._load_hard_samples_config()

            self.refresh_project_list()

        except Exception as e:
            print(f"[AnnotationInterface] Critical error in _do_set_project: {e}")
            traceback.print_exc()
        finally:
            self._is_setting_project = False

            if self._pending_project_info is not None:
                pending = self._pending_project_info
                self._pending_project_info = None
                self.set_project(pending)

    def _load_categories_for_project(self, project_info):
        """从项目信息加载类别（唯一数据源：project_info.json）"""
        raw_categories = project_info.get('categories', {})
        print(f"[DEBUG] _load_categories_for_project: project={project_info.get('name')}, raw_categories={raw_categories}")
        cats = self.manual.load_categories_from_project(project_info)
        print(f"[DEBUG] _load_categories_for_project: loaded categories = {list(cats.keys())}")

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_project_list()
        try:
            self.draw_area.reset_view()
        except Exception as e:
            print(f"[AnnotationInterface] Error resetting view in showEvent: {e}")
    def eventFilter(self, obj, event):
        """事件过滤器：标注列表键盘事件 + 搜索框 @ 弹层键盘导航"""
        if obj == self.label_list and event.type() == event.Type.KeyPress:
            key = event.key()
            current_row = self.label_list.currentRow()

            if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
                # Delete/Backspace：删除当前选中的标注
                if current_row >= 0 and current_row < len(self.draw_area.annotations):
                    self.delete_annotation(current_row)
                return True

        if obj == self.search_box and self._ref_popup.isVisible() \
                and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            popup = self._ref_popup
            count = popup.count()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and count:
                step = 1 if key == Qt.Key.Key_Down else -1
                popup.setCurrentRow((popup.currentRow() + step) % count)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                item = popup.currentItem()
                if item is not None:
                    self._on_reference_clicked(item.text())
                return True
            if key == Qt.Key.Key_Escape:
                popup.hide()
                return True

        if obj == self.search_box and event.type() == QEvent.Type.FocusIn:
            # 聚焦即弹出候选列表（无需先输入 @）；已有 @token 时按片段过滤。
            # 仅在窗口激活时弹出，避免界面初始化抢焦点时的闪弹。
            if self.isActiveWindow():
                self._update_reference_popup(self.search_box.text())
            return False

        if obj == self.search_box and event.type() == QEvent.Type.FocusOut:
            self._ref_popup.hide()

        return super().eventFilter(obj, event)


    @Slot()
    @action("nav.change_save_path", description="修改标注结果保存路径。\n- path：目标保存路径（可选），留空时弹出文件夹选择对话框\n- 非项目模式下会自动扫描新目录中的标注文件并更新示例库\n- 非项目模式下会自动检测标注格式和类别\n- 修改后自动重新加载当前图片的标注", category="标注", params={"path": "str"}, scope="ui")
    def update_save_path(self, path: str = ""):
        if path:
            if not os.path.isdir(path):
                return "错误: 指定的保存路径不存在"
            dir_path = path
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "选择保存路径", self.output_dir)
        if dir_path:
            err = self.manual.update_save_path(dir_path)
            if err:
                return err

            # 非项目模式下扫描标注文件可能较慢，显示进度条
            if self._is_non_project_mode:
                progress = ProgressDialog(
                    "正在加载标注",
                    "正在扫描目录中的标注文件...",
                    parent=self, has_cancel=False
                )
                progress.setWindowModality(Qt.WindowModal)
                progress.setRange(0, 3)
                progress.setValue(0)
                progress.show()
                try:
                    # 用 repaint() 而不是 processEvents()：只同步重绘这个（不可取消的）
                    # 进度对话框，**不派发任何输入/定时器/网络事件**，因此不存在重入。
                    # 进度对话框没有取消按钮，也就不需要事件循环来处理点击。
                    # 步骤1：更新示例库目录
                    progress.setLabelText(f"步骤 (1/3): 更新示例库...")
                    progress.setValue(1)
                    progress.repaint()
                    self.manual.set_non_project_mode(self.output_dir)

                    # 步骤2：自动检测已有标注格式并设置（可能较慢，需要扫描大量文件）
                    progress.setLabelText(f"步骤 (2/3): 检测标注格式和类别...")
                    progress.setValue(2)
                    progress.repaint()
                    self._auto_detect_and_set_export_format(dir_path)

                    # 步骤3：重新加载当前图片的标注
                    progress.setLabelText(f"步骤 (3/3): 加载当前图片标注...")
                    progress.setValue(3)
                    progress.repaint()
                    self.load_image_annotations()

                    InfoBar.success("保存路径已更改", f"非项目模式 - 新路径: {self.output_dir}\n标签将保存到此目录", parent=self)
                finally:
                    progress.close()
            else:
                InfoBar.success("保存路径已更改", f"新路径: {self.output_dir}", parent=self)

            # Reload current image annotations if any exist in the new path
            if not self._is_non_project_mode:
                self.load_image_annotations()

    def _auto_detect_and_set_export_format(self, dir_path):
        """自动检测目录中已有的标注格式并设置导出格式，同时扫描已有类别

        Returns:
            str: 检测到的任务类型 ('seg', 'det', 'obb') 或 None
        """
        detected_task_type, new_categories, detected_format = self.manual.detect_and_set_export_format(dir_path)

        if detected_format:
            settings.set("export_format", detected_format)
            if self._is_non_project_mode:
                InfoBar.info("格式已锁定", f"检测到已有标注格式: {detected_format}（保存将使用同一格式）", duration=3000, parent=self)

        if new_categories:
            self._refresh_category_state(save=False)
            self.refresh_label_list()
            InfoBar.info("类别已扫描", f"检测到 {len(new_categories)} 个类别: {', '.join(new_categories)}", duration=3000, parent=self)

        return detected_task_type

    def _update_non_project_mode_ui(self):
        """根据非项目模式状态更新UI"""
        if self._is_non_project_mode:
            # 非项目模式：禁用项目选择
            self.project_combo.setEnabled(False)

            # 禁用数据集划分（非项目模式不使用）
            self.split_combo.setEnabled(False)

            # 导出格式：非项目模式自动识别后锁定不可修改

            # 更新提示文本
            self.btn_set_save_path.setToolTip("设置标签保存路径 (非项目模式)")
        else:
            # 项目模式：启用控件
            self.project_combo.setEnabled(True)
            self.split_combo.setEnabled(True)
            self.btn_set_save_path.setToolTip("设置保存路径")
            # 项目模式下隐藏导出格式（固定labelme，在版本管理页面选择导出格式）

    def on_roi_selected(self, bbox):
        """Called when an ROI is selected on the canvas"""
        if not self.current_image_path:
            return

        # Save as persistent ROI to project configuration via project_settings
        self.draw_area.persistent_roi = bbox
        self.status_label.setText(f"已设置项目 ROI: {bbox}")
        self.draw_area.update()

    # ====== 区域: UI 构建 & 导航 & 文件列表（原 nav_mixin） ======
    def load_default_directory(self):
        # 1. Try output_dir (current project) first
        default_dir = self.output_dir
        current_project = settings.get("current_project_name", "")
        last_dir = settings.get("last_opened_dir", "")
        
        search_paths = []
        if default_dir and os.path.exists(default_dir):
            search_paths.append(default_dir)
        
        # 2. Only fall back to last_dir when没有绑定具体项目
        # 避免在项目切换时把上一项目或外部目录的图片带入新项目
        if not current_project and last_dir and os.path.exists(last_dir):
            search_paths.append(last_dir)
            
        for target_dir in search_paths:
            # Check if this directory or its project structure has images
            is_project = os.path.exists(os.path.join(target_dir, "annotations"))
            has_images = False

            if is_project:
                # 统一从images/目录读取图片
                if os.path.exists(os.path.join(target_dir, "images")):
                    has_images = True
            
            if not has_images:
                # Check for direct images or recursive images
                for root, dirs, files in os.walk(target_dir):
                    if any(f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) for f in files):
                        has_images = True
                        break
            
            if has_images:
                self.load_directory_images(target_dir)
                return


    def update_mode_buttons(self, mode):
        # Update checked state
        self.btn_mode_sam.setChecked(mode == DrawingMode.SAM)
        self.btn_mode_ai_rect.setChecked(mode == DrawingMode.AI_RECT)
        self.btn_mode_rect.setChecked(mode == DrawingMode.RECT)
        self.btn_mode_poly.setChecked(mode == DrawingMode.POLYGON)
        self.btn_mode_obb.setChecked(mode == DrawingMode.OBB)
        self.btn_mode_edit.setChecked(mode == DrawingMode.EDIT)
        self.btn_mode_roi.setChecked(mode == DrawingMode.ROI)

        # 图标按钮默认纯透明（无边框无背景），仅选中时给蓝色底；
        # 未选中的按钮清空内联样式，交由全局 QSS 的 hover/pressed 处理
        for btn in self.mode_group:
            if btn.isChecked():
                btn.setStyleSheet(
                    "TransparentToolButton { background-color: rgba(0, 122, 255, 0.18); }")
            else:
                btn.setStyleSheet("")
            btn.update()


    # ====== 状态所有权：只读投影（真值在 manual.session）======
    # 没有 setter：任何 `self.current_image_path = ...` 都会立刻 AttributeError，
    # 而不是悄悄留下第二份真值。改写必须走 session.begin_load() / session.clear()。

    @property
    def current_image_path(self):
        return self.manual.current_image_path

    def drain(self, timeout_ms=5000):
        """确定性排空本界面的后台工作（关闭 / 插件卸载路径）。

        实现 `core.lifecycle.Drainable` 契约，由 ShutdownCoordinator 调用。
        顺序：先排空图像解码（画布，作废在途代次），再排空缩略图线程池
        （纯装饰性，放后面）。总预算由调用方给出，内部不叠加超时。
        """
        deadline = time.perf_counter() + max(0, int(timeout_ms)) / 1000.0
        ok = True

        # 先请求取消批量标注：批量可能长达数分钟，直接等必然超时。
        # cancel_batch() 只置一个标志，服务在每张图之间检查，因此很快返回。
        try:
            runner = getattr(self, "_batch_runner", None)
            if runner is not None and runner.busy():
                self.auto.cancel_batch()
                print("[AnnotationInterface] 已请求取消批量标注（关闭流程）")
        except Exception as e:
            print(f"[AnnotationInterface] 请求取消批量标注失败: {e}")

        for name, target in (("batch_runner", getattr(self, "_batch_runner", None)),
                             ("annotation_loader", getattr(self, "_annotation_runner", None)),
                             ("annotation_writer", getattr(self, "manual", None)),
                             ("search_scanner", getattr(self, "_search_runner", None)),
                             ("image_decoder", getattr(self, "draw_area", None)),
                             ("thumbnails", getattr(self, "_thumb_manager", None)),
                             ("model_loader", getattr(self, "model_manager", None))):
            if target is None or not callable(getattr(target, "drain", None)):
                continue
            remaining = int((deadline - time.perf_counter()) * 1000.0)
            if remaining <= 0:
                print(f"[AnnotationInterface] 排空 {name} 时总预算已耗尽")
                ok = False
                continue
            try:
                if not target.drain(remaining):
                    print(f"[AnnotationInterface] {name} 未在 {remaining}ms 内排空")
                    ok = False
            except Exception as e:
                print(f"[AnnotationInterface] 排空 {name} 失败: {e}")
                ok = False
        return ok

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Initialize draw_area first as other components depend on it
        self.draw_area = DrawingWidget(self.manual, self)
        self.draw_area.setObjectName("DrawingWidget")

        # 1. Top Toolbar
        self.setup_toolbar()
        self.main_layout.addWidget(self.toolbar)

        # 2. Middle Content
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("AnnotationSplitter")
        # 接缝宽度：QSS ::handle 的 width 无法改变 QSplitter 手柄宽度属性，
        # 需在代码级设为 1px 才是真正的细线
        self.splitter.setHandleWidth(1)
        
        # Left sidebar for image list
        self.sidebar = QWidget()
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setObjectName("AnnotationSidebar")
        self.sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.sidebar_layout = QVBoxLayout(self.sidebar)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_title = StrongBodyLabel('图片列表')
        header_layout.addWidget(header_title)
        header_layout.addStretch(1)
        self.index_label = CaptionLabel('0 / 0')
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.index_label.setObjectName("AnnotationIndexLabel")
        
        # Manual refresh button
        self.btn_refresh = TransparentToolButton(FIF.SYNC, self)
        self.btn_refresh.setToolTip("刷新图片列表 (检测新文件)")
        self.btn_refresh.setFixedSize(28, 28)
        self.btn_refresh.clicked.connect(self.on_manual_refresh)
        self.btn_refresh.setObjectName("AnnotationRefreshButton")

        # Close button (hides left panel)
        self.btn_close_left_panel = TransparentToolButton(FIF.CLOSE, self)
        self.btn_close_left_panel.setToolTip("隐藏左面板 (Ctrl+B)")
        self.btn_close_left_panel.setFixedSize(28, 28)
        self.btn_close_left_panel.clicked.connect(lambda: self._toggle_panel("left", False))

        header_layout.addWidget(self.index_label)
        header_layout.addWidget(self.btn_refresh)
        header_layout.addWidget(self.btn_close_left_panel)
        self.sidebar_layout.addWidget(header_row)
        
        self.dir_label = CaptionLabel('未选择目录')
        self.dir_label.setObjectName("AnnotationDirLabel")
        self.sidebar_layout.addWidget(self.dir_label)
        
        self.search_box = SearchLineEdit()
        self.search_box.setPlaceholderText("搜索图片(*模糊)/序号跳转… 输入 @ 按类别/大小/状态筛选")
        self.search_box.searchButton.setIcon(AppIcon.SEARCH.icon())
        self.search_box.textChanged.connect(self._on_search_text_changed)
        self.sidebar_layout.addWidget(self.search_box)

        # @ 引用补全（在搜索框输入 @ 引用类别 / 大小进行筛选）
        self._setup_search_reference()

        # 旧筛选控件（类别/大小 combo 与自定义范围输入）已移除，
        # 类别/大小筛选统一由搜索框 @类别:/@大小: token 引用实现

        self.file_list = ListWidget()
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        self.file_list.setIconSize(QSize(44, 44))
        self._thumb_manager = ThumbnailManager(parent=self)
        self._thumb_manager.loaded.connect(self._on_thumbnail_loaded)
        self._thumb_item_index = {}  # path -> QListWidgetItem，缩略图完成时 O(1) 定位

        # 批量自动标注的后台运行器。批量**绝不**再在 GUI 线程同步执行 —— 原先必须靠
        # QApplication.processEvents() 才能让进度条重绘、让取消按钮可点，而那正是
        # "在状态变更处理器内部重入事件循环"的根源：批量运行期间用户还能触发切图/
        # 保存/删除，导致界面状态与磁盘状态错乱。任务移出 GUI 线程后事件循环本就空闲，
        # 既不需要 processEvents，取消按钮也自然可用。
        self._batch_runner = BackgroundTaskRunner(parent=self, max_threads=1,
                                                  name="annotation.batch")
        self._batch_result = None

        # 登记到进程级关闭协调器。为什么必须登记：本项目的关闭链路只有
        # MainWindow.closeEvent -> sys.exit(app.exec())，全仓没有 aboutToQuit、
        # 没有线程排空；若不登记，两个 QThreadPool 只能靠 C++ 析构期的隐式无超时
        # 等待，那发生在解释器收尾阶段 —— 正是退出期
        # "QThread: Destroyed while thread '' is still running"（56/68 份 crash log）
        # 的温床。登记后由 ShutdownCoordinator 在对象树析构前显式排空。
        try:
            from core.lifecycle import register_drainable
            register_drainable("annotation.interface", self)
        except Exception as e:
            print(f"[AnnotationInterface] 登记关闭排空失败: {e}")
        self.file_list.verticalScrollBar().valueChanged.connect(self._on_file_list_scrolled)
        self.sidebar_layout.addWidget(self.file_list)
        
        self.splitter.addWidget(self.sidebar)

        # 3. Canvas + Horizontal Mode Toolbar
        self.canvas_container = QWidget()
        self.canvas_container.setObjectName("AnnotationCanvasContainer")
        self.canvas_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        canvas_vlayout = QVBoxLayout(self.canvas_container)
        canvas_vlayout.setContentsMargins(0, 0, 0, 0)
        canvas_vlayout.setSpacing(0)

        # 横向模式工具栏（画布顶部居中，透明底色）
        self.setup_mode_toolbar()
        toolbar_wrapper = QHBoxLayout()
        toolbar_wrapper.setContentsMargins(0, 8, 0, 2)
        toolbar_wrapper.addStretch(1)
        toolbar_wrapper.addWidget(self.mode_toolbar)
        toolbar_wrapper.addStretch(1)
        canvas_vlayout.addLayout(toolbar_wrapper)

        # 画布
        self.canvas_layout = QHBoxLayout()
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(0)
        self.canvas_layout.addWidget(self.draw_area, 1)
        canvas_vlayout.addLayout(self.canvas_layout, 1)

        self.splitter.addWidget(self.canvas_container)
        
        # Right sidebar for labels and categories
        self.right_sidebar = QWidget()
        self.right_sidebar.setMinimumWidth(250)
        self.right_sidebar.setObjectName("AnnotationRightSidebar")
        self.right_sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.right_layout = QVBoxLayout(self.right_sidebar)

        # 右面板标题栏 + 关闭按钮
        right_header = QWidget()
        right_header_layout = QHBoxLayout(right_header)
        right_header_layout.setContentsMargins(0, 0, 0, 0)
        right_header_layout.setSpacing(8)
        right_header_layout.addWidget(StrongBodyLabel('标注管理'))
        right_header_layout.addStretch(1)
        self.btn_close_right_panel = TransparentToolButton(FIF.CLOSE, self)
        self.btn_close_right_panel.setToolTip("隐藏右面板 (Ctrl+\\)")
        self.btn_close_right_panel.setFixedSize(28, 28)
        self.btn_close_right_panel.clicked.connect(lambda: self._toggle_panel("right", False))
        right_header_layout.addWidget(self.btn_close_right_panel)
        self.right_layout.addWidget(right_header)

        # 2.1 Display Controls
        self.right_layout.addWidget(StrongBodyLabel('显示控制'))
        self.display_group = QWidget()
        self.display_layout = QHBoxLayout(self.display_group)
        self.cb_show_bbox = CheckBox('显示标注')
        self.cb_show_label = CheckBox('显示标签')
        self.cb_show_bbox.setChecked(True)
        self.cb_show_label.setChecked(True)
        self.cb_show_bbox.stateChanged.connect(lambda: self.on_display_changed())
        self.cb_show_label.stateChanged.connect(lambda: self.on_display_changed())
        self.display_layout.addWidget(self.cb_show_bbox)
        self.display_layout.addWidget(self.cb_show_label)
        self.right_layout.addWidget(self.display_group)

        # 2.2 Dataset Splitting
        self.right_layout.addWidget(StrongBodyLabel('数据集划分'))
        self.split_combo = ComboBox()
        self.split_combo.addItems(["训练集 (Train)", "验证集 (Val)", "测试集 (Test)"])
        self.split_combo.setCurrentIndex(0) # Default to Train
        self.split_combo.currentIndexChanged.connect(self.on_split_changed)
        self.right_layout.addWidget(self.split_combo)
        
        # 2.3 Category Management (内联操作按钮)
        self.right_layout.addWidget(StrongBodyLabel('类别管理'))
        self.category_list = ListWidget()
        self.category_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.category_list.customContextMenuRequested.connect(self.show_category_context_menu)
        self.refresh_category_list()
        self.right_layout.addWidget(self.category_list, 2)
        
        # 2.3 Object List
        self.right_layout.addWidget(StrongBodyLabel('对象列表'))
        self.label_list = ListWidget()
        self.right_layout.addWidget(self.label_list, 3)

        # 2.4 Hard Sample Marking
        self.right_layout.addWidget(StrongBodyLabel('难样本标记'))
        hard_sample_group = QWidget()
        hard_sample_layout = QVBoxLayout(hard_sample_group)
        hard_sample_layout.setContentsMargins(0, 2, 0, 2)
        hard_sample_layout.setSpacing(4)
        hard_sample_row = QHBoxLayout()
        self.cb_hard_sample = CheckBox('标记为难样本')
        self.cb_hard_sample.stateChanged.connect(self._on_hard_sample_toggled)
        hard_sample_row.addWidget(self.cb_hard_sample)
        hard_sample_row.addStretch(1)
        self.hard_sample_count_label = CaptionLabel('0')
        hard_sample_row.addWidget(self.hard_sample_count_label)
        hard_sample_layout.addLayout(hard_sample_row)
        self.hard_sample_desc_edit = LineEdit()
        self.hard_sample_desc_edit.setPlaceholderText('难样本描述（可选）')
        self.hard_sample_desc_edit.setFixedHeight(28)
        hard_sample_layout.addWidget(self.hard_sample_desc_edit)
        self.right_layout.addWidget(hard_sample_group)

        self.splitter.addWidget(self.right_sidebar)

        # 禁止拖动折叠面板：拖动到最小宽度即停住，而非突然隐藏
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(2, False)

        # Set initial sizes for splitter: left=250, middle=expand, right=300
        self.splitter.setSizes([250, 800, 300])
        self.main_layout.addWidget(self.splitter)

        # 3. Bottom Bar（放入画布容器内，仅占画布宽度，右侧面板可竖向占满）
        self.setup_bottom_bar()
        canvas_vlayout.addWidget(self.bottom_bar)

        # Connect signals
        self.btn_set_save_path.clicked.connect(self.update_save_path)
        
        self.draw_area.annotation_finished.connect(self.on_annotation_finished)
        self.draw_area.roi_selected.connect(self.on_roi_selected)
        self.draw_area.roi_menu_requested.connect(self.show_roi_context_menu)
        self.draw_area.sam_point_added.connect(self.on_sam_point_added)
        self.draw_area.delete_annotation.connect(self.refresh_label_list)
        self.draw_area.set_as_example.connect(self._on_set_as_example_with_dialog)
        self.draw_area.context_menu_requested.connect(self.show_draw_context_menu)
        self.draw_area.secondary_context_menu_requested.connect(self.show_secondary_context_menu)
        self.draw_area.annotation_double_clicked.connect(self.on_update_annotation_category)
        self.draw_area.mouse_pos_changed.connect(self.update_mouse_pos_label)
        self.draw_area.image_loaded.connect(self.update_resolution_label)
        self.file_list.itemClicked.connect(lambda item: self.on_file_item_clicked(image_index=self.file_list.row(item)))
        self.label_list.itemDoubleClicked.connect(self.on_label_item_double_clicked)
        self.category_list.itemClicked.connect(self.on_category_clicked)
        self.category_list.itemDoubleClicked.connect(self.on_category_double_clicked)
        self.label_list.itemClicked.connect(self.on_label_item_clicked)
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self.show_label_context_menu)
        self.label_list.installEventFilter(self)  # 安装事件过滤器用于键盘事件
        self.search_box.installEventFilter(self)  # 搜索框 @ 引用弹层键盘导航
        
        self.draw_area.mode_changed.connect(self.update_mode_buttons)
        self.draw_area.annotation_selected.connect(self.on_annotation_selected)
        self.draw_area.annotation_modified.connect(self._on_annotation_modified)
        self.draw_area.annotation_modify_started.connect(self._save_undo_state)
        self.draw_area.enhance_toggled.connect(self._on_enhance_state_changed)

        # 设置全局快捷键
        self.setup_global_shortcuts()


    def setup_global_shortcuts(self):
        """设置全局快捷键，确保在界面任何位置都能触发"""
        from core.common.shortcut_manager import ShortcutManager
        sm = ShortcutManager.instance()

        # action_id -> slot 映射
        action_slots = {
            "tool_sam":       self.btn_mode_sam.click,
            "tool_ai_rect":   self.btn_mode_ai_rect.click,
            "tool_rect":      self.btn_mode_rect.click,
            "tool_poly":      self.btn_mode_poly.click,
            "tool_obb":       self.btn_mode_obb.click,
            "tool_edit":      self.btn_mode_edit.click,
            "tool_roi":       self.btn_mode_roi.click,
            "num_sam":        self.btn_mode_sam.click,
            "num_ai_rect":    self.btn_mode_ai_rect.click,
            "num_rect":       self.btn_mode_rect.click,
            "num_poly":       self.btn_mode_poly.click,
            "num_obb":        self.btn_mode_obb.click,
            "num_edit":       self.btn_mode_edit.click,
            "num_roi":        self.btn_mode_roi.click,
            "prev_image":     lambda: self.switch_image(-1),
            "next_image":     lambda: self.switch_image(1),
            "toggle_view":    self._on_toggle_view,
            "toggle_enhance": self._on_toggle_enhance,
            "reset_view":     self.reset_view,
            "example_annotate": lambda: self.one_click_by_current("example"),
            "text_annotate":    lambda: self.one_click_by_current("text"),
            "refine":           self._on_refine_shortcut,
            "select_prev":      lambda: self.select_annotation(-1),
            "select_next":      lambda: self.select_annotation(1),
            "switch_category":  self._switch_category,
            "toggle_visibility": lambda: self.toggle_annotations_visibility("current"),
            "toggle_all_visibility": lambda: self.toggle_annotations_visibility("all"),
            "delete_annotation": self._on_delete_shortcut,
            "undo":            lambda: self.step_history(-1),
            "redo":            lambda: self.step_history(1),
            "escape":          self._on_escape_shortcut,
            "confirm":         self._on_return_shortcut,
            "backspace":       self._on_delete_shortcut,
        }

        self._shortcut_objects = sm.apply_shortcuts(self, action_slots)


    @action("interact.toggle_view", description="切换视图状态(V键)。\n- 有聚焦标注时：取消聚焦，恢复到完整视图\n- 有ROI区域时：在ROI缩放和全图之间切换\n- 有选中标注时：聚焦到该标注（放大并居中）\n- 以上都无时：重置视图", category="画布", scope="ui")
    def _on_toggle_view(self):
        if not self.draw_area.image or not self.draw_area.pixmap:
            return "错误: 没有加载图像"
        if self.draw_area.focused_index >= 0:
            self.draw_area.unfocus()
            return

        if self.draw_area.persistent_roi:
            self.draw_area.toggle_roi_zoom()
            return

        idx = self.draw_area.selected_index
        if idx >= 0:
            self.draw_area.focus_on_annotation(idx)
        else:
            self.draw_area.reset_view()


    def _on_toggle_enhance(self):
        if not self.draw_area.image or not self.draw_area.pixmap:
            return "错误: 没有加载图像"
        self.draw_area.toggle_enhance()


    def _on_enhance_btn_clicked(self):
        """增强按钮点击回调"""
        self.draw_area.toggle_enhance()


    def _on_enhance_state_changed(self, enhanced):
        """增强状态变更回调，同步按钮状态"""
        self.btn_enhance.blockSignals(True)
        self.btn_enhance.setChecked(enhanced)
        self.btn_enhance.blockSignals(False)


    @action("interact.cancel_current", description="取消当前正在进行的操作(Esc键)。\n- 正在 SAM 交互时：取消 SAM 操作\n- 正在绘制多边形时：清空已添加的顶点\n- 操作被取消后画布状态恢复", category="标注", scope="ui")
    def _on_escape_shortcut(self):
        if not self._sam_active and not self.draw_area.current_poly and not self.draw_area.sam_points:
            return "错误: 当前没有正在进行的操作"
        self._sam_active = False
        self.draw_area.current_poly = []
        self.draw_area.sam_points = []
        self.draw_area.update()


    @action("annotation.refresh_files", description="手动重新扫描当前目录的图片文件（可检测外部新增/删除的图片）并重建文件列表、重载当前图片。\n- 无参数\n- 用于目录内容在原界面打开期间发生了变化（外部新增/删除图片）时，重建文件列表并重载当前图片", category="标注", scope="ui")
    @Slot()
    def on_manual_refresh(self):
        """手动刷新图片列表，并检查未保存的标注。"""
        files, preserved_idx, error = self.manual.refresh_directory(self.current_image_path)
        if error:
            return error
        self.image_files = files
        self.file_list.blockSignals(True)
        while self.file_list.count() > 0:
            self.file_list.takeItem(0)
        self.file_list.blockSignals(False)
        for f in self.image_files:
            item = QListWidgetItem(os.path.basename(f))
            item.setData(Qt.UserRole, f)
            item.setSizeHint(QSize(0, 52))
            self.file_list.addItem(item)
        self._populate_file_thumbnails()

        if preserved_idx is not None:
            self.load_image(preserved_idx, force_reload=False)

        return "success"

    def _populate_file_thumbnails(self):
        """为文件列表请求缩略图（按可见行懒加载）。

        列表重建时重建 path->item 索引；只对当前可见行发起缩略图请求，
        命中缓存即时设置，其余显示占位图标，滚动时再动态加载。
        """
        placeholder = FIF.PHOTO.icon()
        self._thumb_item_index = {}
        first, last = self._visible_row_range()
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            path = item.data(Qt.UserRole)
            if not path:
                continue
            self._thumb_item_index[path] = item
            # 只在可见行做缓存读取/请求，离屏行一律占位（滚动时再加载），
            # 避免对几千张图逐个做磁盘 stat 阻塞主线程进入标注界面
            if first <= i <= last:
                cached = self._thumb_manager.get_cached(path)
                if cached is not None:
                    item.setIcon(QIcon(cached))
                else:
                    item.setIcon(placeholder)
                    self._thumb_manager.request(path)
            else:
                item.setIcon(placeholder)

    def _visible_row_range(self, margin: int = 2):
        """计算当前可见行区间（含上下 margin 行，用于提前加载）。"""
        count = self.file_list.count()
        if count == 0:
            return 0, -1
        row_h = self.file_list.sizeHintForRow(0)
        if row_h <= 0:
            row_h = 52
        value = self.file_list.verticalScrollBar().value()
        vh = max(1, self.file_list.viewport().height())
        first = max(0, value // row_h - margin)
        last = min(count - 1, (value + vh) // row_h + margin)
        return first, last

    def _on_file_list_scrolled(self, value=None):
        """滚动时动态加载进入视野的缩略图；命中缓存则立刻显示。"""
        first, last = self._visible_row_range()
        for i in range(first, last + 1):
            item = self.file_list.item(i)
            path = item.data(Qt.UserRole)
            if not path:
                continue
            cached = self._thumb_manager.get_cached(path)
            if cached is not None:
                item.setIcon(QIcon(cached))
            else:
                # request 内部会做缓存与在途请求去重，重复调用安全
                self._thumb_manager.request(path)

    @Slot(str, QPixmap)
    def _on_thumbnail_loaded(self, path: str, pixmap: QPixmap):
        """缩略图加载完成：按 path 索引 O(1) 定位行设置图标（行已重建/不存在时忽略）"""
        item = self._thumb_item_index.get(path)
        if item is not None:
            item.setIcon(QIcon(pixmap))


    def apply_persistent_settings(self):
        # Task mode - use project_settings first, fallback to settings
        task_mode = project_settings.get("task_mode") or settings.get("task_mode", "seg")
        self.draw_area.task_mode = task_mode
        self.manual.set_task_mode(task_mode)
        
        # Export format - use project_settings first, fallback to settings
        fmt = project_settings.get("export_format") or settings.get("export_format", "labelme")
        settings.set("export_format", fmt)
            
        # Category - use project_settings first, fallback to settings
        cat = project_settings.get("current_category") or settings.get("current_category", "")
        if cat in self.categories:
            self.current_category = cat
            # Update selection in UI
            self._select_category_in_list(cat)
        
        # Persistent ROI is handled automatically by project_settings
        pass
        
        # Initialize mode button background colors
        self.update_mode_buttons(self.draw_area.mode)


    def show_file_context_menu(self, pos):
        item = self.file_list.itemAt(pos)
        if not item:
            return
            
        path = item.data(Qt.UserRole)
        if not path:
            return

        menu = RoundMenu(parent=self)
        
        open_folder_action = Action(FIF.FOLDER, "打开文件夹")
        open_folder_action.triggered.connect(lambda: self.open_in_explorer("file", path))
        menu.addAction(open_folder_action)
        
        copy_path_action = Action(FIF.COPY, "复制路径")
        copy_path_action.triggered.connect(lambda: self.copy_file_path(path))
        menu.addAction(copy_path_action)

        copy_image_action = Action(FIF.PHOTO, "复制图片")
        copy_image_action.triggered.connect(lambda: self.copy_image_to_clipboard(path))
        menu.addAction(copy_image_action)
        
        menu.addSeparator()

        send_menu = RoundMenu("添加到对话", parent=self)
        filename = os.path.basename(path)
        act_orig = Action(FIF.PHOTO, "原图")
        act_orig.triggered.connect(lambda: self._send_reference(f"@原图:{filename}"))
        send_menu.addAction(act_orig)
        act_overlay = Action(FIF.PHOTO, "渲染图")
        act_overlay.triggered.connect(lambda: self._send_reference(f"@渲染图:{filename}"))
        send_menu.addAction(act_overlay)
        menu.addMenu(send_menu)

        delete_action = Action(FIF.DELETE, "删除图片")
        delete_action.triggered.connect(lambda: self.delete_image_file(path))
        menu.addAction(delete_action)

        menu.exec(QCursor.pos())


    @action("nav.open_in_explorer", description="在系统文件管理器中打开文件或目录。\n- target：file=定位到指定文件（需传 path），output=打开当前输出目录\n- path：target=file 时的文件路径\n- file 模式会自动选中该文件，output 模式目录不存在时弹出警告提示\n- 可通过 change_save_path 修改输出路径", category="导航", params={"target": "str", "path": "str"})
    def open_in_explorer(self, target="file", path=""):
        if target == "output":
            if not self.output_dir:
                return "错误: 输出目录未设置"
            if os.path.exists(self.output_dir):
                os.startfile(self.output_dir)
            else:
                InfoBar.warning("目录不存在", f"无法找到路径: {self.output_dir}", parent=self)
                return "错误: 输出目录不存在"
            return
        if not path:
            return "错误: path 参数为空"
        if not os.path.exists(path):
            return f"错误: 文件不存在: {path}"
        import subprocess
        subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')


    @action("nav.copy_file_path", read_only=True, description="复制文件路径到系统剪贴板。\n- path: 要复制的文件路径", category="导航", params={"path": "str"})
    def copy_file_path(self, path: str = ""):
        if not path:
            return "错误: path 参数为空"
        QApplication.clipboard().setText(path)
        return "success"


    @action("nav.copy_image_to_clipboard", read_only=True, description="将图片复制到系统剪贴板。\n- path: 图片文件路径\n- 复制后可粘贴到其他软件（如 Word、微信等）\n- 失败时会提示原因", category="导航", params={"path": "str"})
    def copy_image_to_clipboard(self, path: str = ""):
        if not path:
            return "错误: path 参数为空"
        if not os.path.exists(path):
            return f"错误: 文件不存在: {path}"
        qimg = self.draw_area._qimage if hasattr(self.draw_area, '_qimage') and self.draw_area._qimage else None
        if qimg is None:
            from PySide6.QtGui import QImage
            qimg = QImage(path)
        if qimg and not qimg.isNull():
            QApplication.clipboard().setImage(qimg)
            return "success"
        return "错误: 无法读取图片"


    @action("nav.delete_image_file", description="删除图片文件（不可恢复）。\n- path: 要删除的图片路径\n- 自动处理文件列表和标注缓存\n- 谨慎使用，物理删除不可恢复", category="导航", params={"path": "str"})
    def delete_image_file(self, path: str = "", skip_confirm: bool = False):
        if not path:
            return "错误: 路径为空"
        # skip_confirm 参数此前是死参数：action 描述写着"物理删除不可恢复"，但代码
        # 从不询问，于是 agent 可以在无确认的情况下删掉用户图片。现在默认必须确认。
        if not skip_confirm:
            box = MessageBox("确认删除",
                             f"将物理删除该图片及其专属标注，删除后不可恢复：\n{path}",
                             self)
            if not box.exec():
                return "已取消删除"

        # 删除前必须等在途解码结束：Windows 上解码进行中 os.remove 会
        # WinError 32。打开目录会一次性为所有可见行排入缩略图任务，
        # 因此"刚打开目录就删图"原本必定失败（已实测复现）。
        # wait_idle 没有副作用（不作废代次、不取消请求）。
        try:
            self._thumb_manager.wait_idle(3000)
            self.draw_area.wait_io_idle(3000)
        except Exception as e:
            print(f"[AnnotationInterface] 删除前等待解码空闲失败（继续删除）: {e}")

        result, error = self.manual.delete_image_file(path)
        if error:
            return error

        for warning in (result or {}).get("warnings", []):
            print(f"[AnnotationInterface] delete_image_file: {warning}")

        target_item = None
        row = -1
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.data(Qt.UserRole) == path:
                target_item = it
                row = i
                break
        if target_item:
            self.file_list.takeItem(row)
        self.image_files = self.manual.image_files
        self.update_index_label()

        if path == self.current_image_path:
            if self.file_list.count() > 0:
                new_idx = min(max(0, row), self.file_list.count() - 1)
                self.load_image(new_idx)
            else:
                # draw_area.clear() 内部已经 session.clear()（path/标注/像素一起清空）
                self.draw_area.clear()
        warnings = (result or {}).get("warnings", [])
        if warnings:
            return "success（注意: " + "; ".join(warnings) + "）"
        return "success"


    def update_index_label(self):
        count = self.file_list.count()
        if count == 0:
            self.index_label.setText("0 / 0")
            return
            
        current = self.file_list.currentRow() + 1
        # If no row selected but we have items
        if current <= 0 and count > 0:
            current = 0
            
        self.index_label.setText(f"{current} / {count}")


    def update_resolution_label(self):
        """更新图片分辨率显示"""
        if self.draw_area.original_image_size and not self.draw_area.original_image_size.isNull():
            w = self.draw_area.original_image_size.width()
            h = self.draw_area.original_image_size.height()
            self.resolution_label.setText(f"分辨率: {w} x {h}")
        else:
            self.resolution_label.setText("分辨率: -")


    def update_mouse_pos_label(self, img_pos):
        """更新鼠标坐标、灰度值和标注宽高显示"""
        if not img_pos:
            self.mouse_pos_label.setText("坐标: -, -")
            return

        x, y = int(img_pos.x()), int(img_pos.y())
        parts = [f"坐标: {x}, {y}"]

        # 灰度值
        try:
            image = self.draw_area.image
            if image and 0 <= x < image.width() and 0 <= y < image.height():
                color = image.pixelColor(x, y)
                if color.isValid():
                    # 灰度 = 0.299*R + 0.587*G + 0.114*B
                    gray = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())
                    parts.append(f"灰度: {gray}")
        except:
            pass

        # 标注框宽高
        try:
            hover_idx = self.draw_area.hover_index
            if hover_idx >= 0 and hover_idx < len(self.draw_area.annotations):
                ann = self.draw_area.annotations[hover_idx]
                bbox = ann.get('bbox')
                if bbox and len(bbox) >= 4:
                    w, h = int(bbox[2]), int(bbox[3])
                    parts.append(f"目标: {w}×{h}")
        except:
            pass

        self.mouse_pos_label.setText(" | ".join(parts))


    def setup_toolbar(self):
        """设置顶部菜单栏"""
        self.menu_bar = None  # will be created in _setup_menu_bar

        self.toolbar = QWidget()
        self.toolbar.setObjectName("AnnotationToolbar")
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        toolbar_vlayout = QVBoxLayout(self.toolbar)
        toolbar_vlayout.setContentsMargins(0, 0, 0, 0)
        toolbar_vlayout.setSpacing(0)

        # ========== 菜单栏 ==========
        self._setup_menu_bar(toolbar_vlayout)
        toolbar_vlayout.addStretch(0)  # 防止菜单栏被多余撑开

        # ========== 创建隐藏的必要 widget（其他代码引用） ==========
        self._setup_hidden_widgets()


    def get_menubar_config(self):
        """返回标注界面的菜单配置，供主窗口标题栏使用"""
        return [
            {
                'title': '文件(&F)',
                'actions': [
                    ('打开目录', self.open_directory),
                    ('打开保存目录', lambda: self.open_in_explorer("output")),
                    ('设置保存路径', lambda: self.btn_set_save_path.click() if hasattr(self, 'btn_set_save_path') else None),
                    None,
                    ('加载副标注', self.on_load_secondary_annotations),
                ]
            },
            {
                'title': '标注(&A)',
                'actions': [
                    ('基于示例标注全部图片', lambda: self.one_click_by_all("example")),
                    ('基于示例标注当前图片', lambda: self.one_click_by_current("example")),
                    None,
                    ('基于文本标注全部图片', lambda: self.one_click_by_all("text")),
                    ('基于文本标注当前图片', lambda: self.one_click_by_current("text")),
                    None,
                    ('基于普通模型标注当前图片', lambda: self.one_click_by_current("model")),
                    ('基于普通模型标注全部图片', lambda: self.one_click_by_all("model")),
                    None,
                    ('清除所有标注', lambda: self._clear_all_annotations_ui()),
                ]
            },
            {
                'title': '模型(&M)',
                'actions': [
                    ('交互模型 >', None),
                    ('示例模型 >', None),
                    ('普通模型 >', None),
                    ('细化模型 >', None),
                ]
            },
            {
                'title': '模式(&O)',
                'actions': [
                    ('高精度模式', None),
                ]
            },
            {
                'title': '视图(&V)',
                'actions': [
                    ('切换左栏', lambda: self._toggle_panel("left")),
                    ('切换右栏', lambda: self._toggle_panel("right")),
                    None,
                    ('重置视图', self.reset_view),
                    ('图像增强', None),
                    None,
                    ('显示边界框', None),
                    ('显示标签', None),
                ]
            },
            {
                'title': '设置(&S)',
                'actions': [
                    ('示例库管理', self.open_example_library_manager),
                    ('提示词管理', self.open_prompt_library_manager),
                    ('规则配置', self.open_rule_config),
                ]
            }
        ]


    def hide_internal_menubar(self):
        """隐藏内部菜单栏（由主窗口管理菜单时调用）"""
        if hasattr(self, 'toolbar') and self.toolbar:
            self.toolbar.setVisible(False)
        if self.menu_bar:
            self.menu_bar.setVisible(False)
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.setVisible(False)
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.setVisible(False)


    def _setup_menu_bar(self, parent_layout):
        """创建 VSCode 风格的水平菜单栏（内部使用）"""
        from PySide6.QtWidgets import QMenuBar

        self.menu_bar = QMenuBar()
        self.menu_bar.setObjectName("AnnotationMenuBar")
        self.menu_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        # --- 左面板切换按钮（菜单栏最左端） ---
        self.btn_toggle_left_panel = TransparentToolButton(FIF.MENU, self)
        self.btn_toggle_left_panel.setToolTip("切换左栏显示 (Ctrl+B)")
        self.btn_toggle_left_panel.setCheckable(True)
        self.btn_toggle_left_panel.setChecked(True)
        self.btn_toggle_left_panel.setFixedSize(28, 28)
        self.btn_toggle_left_panel.clicked.connect(lambda: self._toggle_panel("left"))

        # 从配置文件恢复自定义模型
        self._restore_custom_models_from_config()

        # 构建菜单到内部菜单栏
        self._build_menus(self.menu_bar)

        # --- 右面板切换按钮（菜单栏最右端） ---
        self.btn_toggle_right_panel = TransparentToolButton(FIF.LAYOUT, self)
        self.btn_toggle_right_panel.setToolTip("切换右栏显示 (Ctrl+\\)")
        self.btn_toggle_right_panel.setCheckable(True)
        self.btn_toggle_right_panel.setChecked(True)
        self.btn_toggle_right_panel.setFixedSize(28, 28)
        self.btn_toggle_right_panel.clicked.connect(lambda: self._toggle_panel("right"))

        # 用 QHBoxLayout 包裹按钮 + QMenuBar + 按钮
        menu_bar_layout = QHBoxLayout()
        menu_bar_layout.setContentsMargins(4, 2, 4, 2)
        menu_bar_layout.setSpacing(0)
        menu_bar_layout.addWidget(self.btn_toggle_left_panel)
        menu_bar_layout.addWidget(self.menu_bar)
        menu_bar_layout.addStretch(0)
        menu_bar_layout.addWidget(self.btn_toggle_right_panel)

        parent_layout.addLayout(menu_bar_layout)


    def build_menus(self, target_menu_bar):
        """将完整菜单构建到指定的菜单栏（供主窗口全局菜单栏使用）"""
        self._build_menus(target_menu_bar)


    @staticmethod
    def _menu_label_with_shortcut(label, action_id):
        """给菜单项文字追加快捷键提示（右侧对齐显示）。

        以 ``\\t`` 分隔：Qt 的菜单样式会把 \\t 之后的部分按快捷键列右对齐绘制，
        与 ``QAction.setShortcut`` 的显示效果完全一致（视图菜单的 Ctrl+B / Ctrl+\\
        就是这种显示方式）。这里只负责「显示提示」，真正的按键绑定仍由
        ShortcutManager 在 setup_global_shortcuts() 里创建的 QShortcut 负责，
        两者读取同一份快捷键配置（含用户在「设置 → 快捷键」里的自定义键位），
        因此不会重复绑定导致快捷键歧义。没有配置快捷键时返回原文字。
        """
        try:
            from core.common.shortcut_manager import ShortcutManager
            key = ShortcutManager.instance().get_key(action_id)
        except Exception:
            key = ""
        return f"{label}\t{key}" if key else label


    def _build_menus(self, target_menu_bar):
        """创建标注界面的完整菜单"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction

        # 持有全部菜单引用：QMenuBar.addMenu 返回的对象在部分 PySide6 环境
        # 下可能被 GC 回收（offscreen 复现），挂到 self 上确保与菜单栏同生命周期
        self._menus = {}

        # --- 文件菜单 ---
        file_menu = target_menu_bar.addMenu("文件(&F)")
        self._menus["文件"] = file_menu

        self._project_menu = file_menu.addMenu("项目库")
        self._project_menu.aboutToShow.connect(self._rebuild_project_menu)
        file_menu.addSeparator()

        act_open = QAction("打开目录", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_directory_ui)
        file_menu.addAction(act_open)

        act_explore = QAction("打开保存目录", self)
        act_explore.triggered.connect(lambda: self.open_in_explorer("output"))
        file_menu.addAction(act_explore)

        act_set_save_path = QAction("设置保存路径", self)
        act_set_save_path.triggered.connect(
            lambda: self.btn_set_save_path.click() if hasattr(self, 'btn_set_save_path') else None
        )
        file_menu.addAction(act_set_save_path)

        file_menu.addSeparator()

        # Ctrl+S 保存快捷键保留（"保存"菜单项已按需求移除）
        from PySide6.QtGui import QShortcut, QKeySequence
        save_sc = QShortcut(QKeySequence.StandardKey.Save, self)
        save_sc.activated.connect(self._save_current_ui)

        act_secondary = QAction("加载副标注", self)
        act_secondary.triggered.connect(self.on_load_secondary_annotations)
        file_menu.addAction(act_secondary)

        # --- 标注菜单 ---
        anno_menu = target_menu_bar.addMenu("标注(&A)")
        self._menus["标注"] = anno_menu

        act_example_all = QAction("基于示例标注全部图片", self)
        act_example_all.triggered.connect(lambda: self.one_click_by_all("example"))
        anno_menu.addAction(act_example_all)

        act_example_curr = QAction(
            self._menu_label_with_shortcut("基于示例标注当前图片", "example_annotate"), self)
        act_example_curr.triggered.connect(lambda: self.one_click_by_current("example"))
        anno_menu.addAction(act_example_curr)

        anno_menu.addSeparator()

        act_text_all = QAction("基于文本标注全部图片", self)
        act_text_all.triggered.connect(lambda: self.one_click_by_all("text"))
        anno_menu.addAction(act_text_all)

        act_text_curr = QAction(
            self._menu_label_with_shortcut("基于文本标注当前图片", "text_annotate"), self)
        act_text_curr.triggered.connect(lambda: self.one_click_by_current("text"))
        anno_menu.addAction(act_text_curr)

        anno_menu.addSeparator()

        act_model_curr = QAction("基于已训练模型标注当前图片", self)
        act_model_curr.triggered.connect(lambda: self.one_click_by_current("model"))
        anno_menu.addAction(act_model_curr)

        act_model_all = QAction("基于已训练模型标注全部图片", self)
        act_model_all.triggered.connect(lambda: self.one_click_by_all("model"))
        anno_menu.addAction(act_model_all)

        anno_menu.addSeparator()

        act_clear = QAction("清除所有标注", self)
        act_clear.triggered.connect(self._clear_all_annotations_ui)
        anno_menu.addAction(act_clear)

        # --- 模型菜单 ---
        model_menu = target_menu_bar.addMenu("模型(&M)")
        self._menus["模型"] = model_menu

        # 交互模型子菜单
        self._interactive_model_menu = model_menu.addMenu("交互模型")
        self._rebuild_interactive_model_menu()
        self._interactive_model_menu.aboutToShow.connect(self._rebuild_interactive_model_menu)

        # 示例模型子菜单
        self._auto_model_menu = model_menu.addMenu("示例模型")
        self._rebuild_auto_model_menu()
        self._auto_model_menu.aboutToShow.connect(self._rebuild_auto_model_menu)

        # 普通模型子菜单
        self._plain_model_menu = model_menu.addMenu("普通模型")
        self._rebuild_plain_model_menu()
        self._plain_model_menu.aboutToShow.connect(self._rebuild_plain_model_menu)

        # 细化模型子菜单
        self._refine_model_menu = model_menu.addMenu("细化模型")
        self._rebuild_refine_model_menu()
        self._refine_model_menu.aboutToShow.connect(self._rebuild_refine_model_menu)

        # --- 模式菜单 ---
        mode_menu = target_menu_bar.addMenu("模式(&O)")
        self._menus["模式"] = mode_menu

        self._act_example_mode = QAction("示例模式", self)
        self._act_example_mode.setCheckable(True)
        self._act_example_mode.setChecked(False)
        self._act_example_mode.triggered.connect(lambda checked: self.on_example_mode_changed(checked))
        mode_menu.addAction(self._act_example_mode)

        self._act_high_precision = QAction("高精度模式", self)
        self._act_high_precision.setCheckable(True)
        self._act_high_precision.setChecked(settings.get("high_precision", True))
        self._act_high_precision.triggered.connect(lambda checked: self._on_menu_high_precision(checked))
        mode_menu.addAction(self._act_high_precision)

        # --- 视图菜单 ---
        view_menu = target_menu_bar.addMenu("视图(&V)")
        self._menus["视图"] = view_menu

        act_toggle_left = QAction("切换左栏", self)
        act_toggle_left.setShortcut("Ctrl+B")
        act_toggle_left.triggered.connect(lambda: self._toggle_panel("left"))
        view_menu.addAction(act_toggle_left)

        act_toggle_right = QAction("切换右栏", self)
        act_toggle_right.setShortcut("Ctrl+\\")
        act_toggle_right.triggered.connect(lambda: self._toggle_panel("right"))
        view_menu.addAction(act_toggle_right)

        view_menu.addSeparator()

        act_reset_view = QAction("重置视图", self)
        act_reset_view.setShortcut("Ctrl+0")
        act_reset_view.triggered.connect(self.reset_view)
        view_menu.addAction(act_reset_view)

        self._act_enhance = QAction("图像增强", self)
        self._act_enhance.setCheckable(True)
        self._act_enhance.setChecked(
            self.draw_area._show_enhanced if hasattr(self, 'draw_area') else False
        )
        self._act_enhance.triggered.connect(lambda checked: self._on_menu_enhance(checked))
        view_menu.addAction(self._act_enhance)

        view_menu.addSeparator()

        self._act_show_bbox = QAction("显示边界框", self)
        self._act_show_bbox.setCheckable(True)
        self._act_show_bbox.setChecked(
            self.cb_show_bbox.isChecked() if hasattr(self, 'cb_show_bbox') else True
        )
        self._act_show_bbox.triggered.connect(lambda checked: self._on_menu_show_bbox(checked))
        view_menu.addAction(self._act_show_bbox)

        self._act_show_label = QAction("显示标签", self)
        self._act_show_label.setCheckable(True)
        self._act_show_label.setChecked(
            self.cb_show_label.isChecked() if hasattr(self, 'cb_show_label') else True
        )
        self._act_show_label.triggered.connect(lambda checked: self._on_menu_show_label(checked))
        view_menu.addAction(self._act_show_label)

        # --- 设置菜单 ---
        settings_menu = target_menu_bar.addMenu("设置(&S)")
        self._menus["设置"] = settings_menu

        act_example_lib = QAction("示例库管理", self)
        act_example_lib.triggered.connect(self.open_example_library_manager)
        settings_menu.addAction(act_example_lib)

        act_prompt_lib = QAction("提示词管理", self)
        act_prompt_lib.triggered.connect(self.open_prompt_library_manager)
        settings_menu.addAction(act_prompt_lib)

        act_rule_config = QAction("规则配置", self)
        act_rule_config.triggered.connect(self.open_rule_config)
        settings_menu.addAction(act_rule_config)


    def _rebuild_project_menu(self, projects=None):
        """重建项目库子菜单"""
        from PySide6.QtGui import QAction
        if not hasattr(self, '_project_menu'):
            return
        try:
            self._project_menu.clear()
            if projects is None:
                service = self.project_service or ProjectService()
                projects = service.get_all_projects()
            current_name = getattr(self, 'current_project_name', '')
            for p in projects:
                act = QAction(p['name'], self)
                act.setCheckable(True)
                act.setChecked(p['name'] == current_name)
                act.triggered.connect(lambda checked, n=p['name']: self.on_project_combo_changed(project_name=n))
                self._project_menu.addAction(act)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"重建项目菜单失败: {e}")


    @action("interact.toggle_panel", description="切换左栏/右栏（sidebar/right_sidebar）的显示/隐藏。\n- side：left=左栏，right=右栏\n- visible：True=显示，False=隐藏，None=切换当前状态\n- 切换后自动同步对应切换按钮的选中状态\n- 适用于需要控制界面布局的场景", category="视图", params={"side": "str", "visible": "bool"})
    def _toggle_panel(self, side: str = "left", visible: bool = None):
        """切换左栏/右栏显示/隐藏"""
        targets = {
            "left": (self.sidebar, 'btn_toggle_left_panel'),
            "right": (self.right_sidebar, 'btn_toggle_right_panel'),
        }
        panel, btn_name = targets.get(side, targets["left"])
        if visible is None:
            visible = not panel.isVisible()
        panel.setVisible(visible)
        if hasattr(self, btn_name):
            getattr(self, btn_name).setChecked(visible)
        # 广播统一 ui_state 状态（供录制/回放/其他插件观测面板显隐）
        if self._event_bus:
            self._event_bus.publish_state("annotation.ui_state",
                {"widget": "panel", "side": side, "visible": bool(visible)})


    @action("interact.toggle_recording", description="切换 Action Recorder 的录制状态。\n- checked：True=开始录制，False=停止录制，None=切换当前状态\n- 开始录制时自动插入 switch_to 步骤确保回放时先切到当前界面\n- 停止录制时弹出保存对话框，可将录制结果保存为 JSON 测试流程文件\n- 保存的文件可用于测试流程回放", category="录制")
    def _toggle_recording(self, checked: bool = None):
        """切换录制状态"""
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()

        if checked is None:
            checked = not recorder.is_recording

        if checked and not recorder.is_recording:
            # 开始录制
            recorder.start(scenario_name="录制流程")
            # 自动插入 switch_to 步骤，确保回放时先切到当前界面
            recorder.record_raw_step({
                "action": "switch_to",
                "target": "annotation",
                "description": "切换到标注界面"
            })
            self.act_record.setChecked(True) if hasattr(self, 'act_record') else None
        elif not checked and recorder.is_recording:
            # 停止录制 → 弹出保存对话框
            scenario = recorder.stop()
            self.act_record.setChecked(False) if hasattr(self, 'act_record') else None
            # 弹出保存对话框
            self._save_recording(scenario)


    def _save_recording(self, scenario: dict):
        """保存录制结果到文件"""
        from datetime import datetime
        default_name = f"test_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_dir = os.path.join(PROJECT_ROOT, "config")
        default_path = os.path.join(config_dir, default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存录制流程", default_path,
            "测试流程 (*.json);;所有文件 (*)"
        )
        if file_path:
            from core.common.action_recorder import ActionRecorder
            recorder = ActionRecorder.instance()
            recorder.save_to_file(file_path, scenario)
            InfoBar.success("录制已保存", f"流程已保存到 {os.path.basename(file_path)}", parent=self)


    def _setup_hidden_widgets(self):
        """创建隐藏的必要 widget（被其他代码引用，但不在菜单栏中显示）"""
        # 项目库 ComboBox
        self.project_combo = ComboBox()
        self.project_combo.currentIndexChanged.connect(self.on_project_combo_changed)
        self.project_combo.setVisible(False)

        # 模型 ComboBox（用于模型切换的内部状态跟踪）
        self.interactive_model_combo = ComboBox()
        interactive_models = self.model_manager.get_interactive_models()
        self.interactive_model_combo.addItems(interactive_models)
        if self.interactive_model_type in interactive_models:
            self.interactive_model_combo.setCurrentText(self.interactive_model_type)
        elif interactive_models:
            self.interactive_model_combo.setCurrentIndex(0)
            self.interactive_model_type = interactive_models[0]
        self.interactive_model_combo.currentIndexChanged.connect(self.on_interactive_model_changed)
        self.interactive_model_combo.setVisible(False)

        self.auto_model_combo = ComboBox()
        auto_models = self.model_manager.get_auto_models()
        self.auto_model_combo.addItems(auto_models)
        if self.model_type in auto_models:
            self.auto_model_combo.setCurrentText(self.model_type)
        elif auto_models:
            self.auto_model_combo.setCurrentIndex(0)
            self.model_type = auto_models[0]
        self.auto_model_combo.currentIndexChanged.connect(self.on_auto_model_changed)
        self.auto_model_combo.setVisible(False)

        # 文件操作按钮
        self.btn_open_dir = PushButton(FIF.FOLDER, "打开", self)
        self.btn_open_dir.clicked.connect(self._open_directory_ui)
        self.btn_open_dir.setVisible(False)
        self.btn_save = PushButton(FIF.SAVE, "保存", self)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save.setVisible(False)
        self.btn_explore = PushButton(FIF.FOLDER_ADD, "目录", self)
        self.btn_explore.clicked.connect(lambda: self.open_in_explorer("output"))
        self.btn_explore.setVisible(False)
        self.btn_load_secondary = PushButton(FIF.ADD_TO, "副标注", self)
        self.btn_load_secondary.clicked.connect(self.on_load_secondary_annotations)
        self.btn_load_secondary.setVisible(False)

        # 一键标注按钮
        self.btn_one_click = PrimaryPushButton(FIF.LABEL, "一键标注", self)
        self.btn_one_click.clicked.connect(self.on_one_click_clicked)
        self.btn_one_click.setVisible(False)

        # 精度模式开关
        self.cb_high_precision = SwitchButton()
        self.cb_high_precision.setOnText('高精度')
        self.cb_high_precision.setOffText('普通')
        self.cb_high_precision.setFixedWidth(80)
        self.cb_high_precision.setChecked(settings.get("high_precision", True))
        self.cb_high_precision.checkedChanged.connect(lambda c: settings.set("high_precision", c))
        self.cb_high_precision.setVisible(False)

        # 示例模式开关
        self.example_mode_switch = SwitchButton()
        self.example_mode_switch.setOnText('')
        self.example_mode_switch.setOffText('')
        self.example_mode_switch.setChecked(False)
        self.example_mode_switch.checkedChanged.connect(self.on_example_mode_changed)
        self.example_mode_switch.setVisible(False)

        # 视图按钮
        self.btn_reset_view = TransparentToolButton(AppIcon.FIT_VIEW, self)
        self.btn_reset_view.clicked.connect(self.reset_view)
        self.btn_reset_view.setVisible(False)
        self.btn_enhance = TransparentToolButton(FIF.BRIGHTNESS, self)
        self.btn_enhance.setCheckable(True)
        self.btn_enhance.clicked.connect(self._on_enhance_btn_clicked)
        self.btn_enhance.setVisible(False)
        self.btn_set_save_path = TransparentToolButton(FIF.SETTING, self)
        self.btn_set_save_path.setVisible(False)

        # 工具栏内容容器（隐藏，兼容旧代码引用）
        self.toolbar_content = QWidget()
        self.toolbar_content.setObjectName("AnnotationToolbarContent")
        self.toolbar_content.setVisible(False)
        self.toolbar_layout = QHBoxLayout(self.toolbar_content)


    def setup_mode_toolbar(self):
        """横向模式工具栏（画布顶部居中，dock 底色）"""
        self.mode_toolbar = QWidget()
        self.mode_toolbar.setObjectName("AnnotationModeToolbar")
        self.mode_toolbar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self.mode_toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        
        # AI Tools Group
        self.btn_mode_sam = TransparentToolButton(AppIcon.POINT, self)
        self.btn_mode_sam.setToolTip("AI 点工具 (Q/1)")
        self.btn_mode_sam.setCheckable(True)
        self.btn_mode_sam.setAutoExclusive(True)
        self.btn_mode_sam.setChecked(True)

        self.btn_mode_ai_rect = TransparentToolButton(AppIcon.AI_RECT, self)
        self.btn_mode_ai_rect.setToolTip("AI 矩形框工具 (W/2)")
        self.btn_mode_ai_rect.setCheckable(True)
        self.btn_mode_ai_rect.setAutoExclusive(True)

        # 添加 AI 工具按钮
        for btn in [self.btn_mode_sam, self.btn_mode_ai_rect]:
            btn.clicked.connect(self.on_mode_btn_clicked)
            btn.setFixedSize(30, 30)
            layout.addWidget(btn)

        # 添加竖向分割线
        from qfluentwidgets import VerticalSeparator
        separator1 = VerticalSeparator()
        separator1.setFixedHeight(24)
        layout.addWidget(separator1)

        # 普通工具 Group
        self.btn_mode_rect = TransparentToolButton(AppIcon.RECT, self)
        self.btn_mode_rect.setToolTip("矩形标注 (R/3)")
        self.btn_mode_rect.setCheckable(True)
        self.btn_mode_rect.setAutoExclusive(True)

        self.btn_mode_poly = TransparentToolButton(AppIcon.POLYGON, self)
        self.btn_mode_poly.setToolTip("多边形标注 (P/4)")
        self.btn_mode_poly.setCheckable(True)
        self.btn_mode_poly.setAutoExclusive(True)

        self.btn_mode_obb = TransparentToolButton(AppIcon.OBB, self)
        self.btn_mode_obb.setToolTip("旋转检测 (X/5)")
        self.btn_mode_obb.setCheckable(True)
        self.btn_mode_obb.setAutoExclusive(True)

        self.btn_mode_edit = TransparentToolButton(AppIcon.CURSOR, self)
        self.btn_mode_edit.setToolTip("编辑模式 (E/6)")
        self.btn_mode_edit.setCheckable(True)
        self.btn_mode_edit.setAutoExclusive(True)

        self.btn_mode_roi = TransparentToolButton(AppIcon.ROI, self)
        self.btn_mode_roi.setToolTip("ROI 区域选择 (S/7)")
        self.btn_mode_roi.setCheckable(True)
        self.btn_mode_roi.setAutoExclusive(True)

        self.mode_group = [self.btn_mode_sam, self.btn_mode_ai_rect, self.btn_mode_rect, self.btn_mode_poly, self.btn_mode_obb, self.btn_mode_edit, self.btn_mode_roi]
        for btn in self.mode_group[2:]:
            btn.clicked.connect(self.on_mode_btn_clicked)
            btn.setFixedSize(30, 30)
            layout.addWidget(btn)

        # 分割线 + 重置视图按钮
        separator2 = VerticalSeparator()
        separator2.setFixedHeight(24)
        layout.addWidget(separator2)

        self.btn_reset_view_toolbar = TransparentToolButton(AppIcon.FIT_VIEW, self)
        self.btn_reset_view_toolbar.setToolTip("重置画布 (Ctrl+0)")
        self.btn_reset_view_toolbar.setFixedSize(30, 30)
        self.btn_reset_view_toolbar.clicked.connect(self.reset_canvas_view)
        layout.addWidget(self.btn_reset_view_toolbar)


    def setup_bottom_bar(self):
        self.bottom_bar = QWidget()
        self.bottom_bar.setFixedHeight(30)
        self.bottom_bar.setObjectName("AnnotationBottomBar")
        self.bottom_layout = QHBoxLayout(self.bottom_bar)
        self.bottom_layout.setContentsMargins(10, 0, 10, 0)
        
        # 状态信息
        self.status_label = CaptionLabel("就绪")
        self.bottom_layout.addWidget(self.status_label)
        
        self.bottom_layout.addSpacing(20)
        
        # 图片分辨率
        self.resolution_label = CaptionLabel("分辨率: -")
        self.resolution_label.setObjectName("AnnotationResolutionLabel")
        self.bottom_layout.addWidget(self.resolution_label)
        
        self.bottom_layout.addSpacing(20)
        
        # 鼠标坐标
        self.mouse_pos_label = CaptionLabel("坐标: -, -")
        self.mouse_pos_label.setObjectName("AnnotationMousePosLabel")
        self.mouse_pos_label.setMinimumWidth(280)
        self.bottom_layout.addWidget(self.mouse_pos_label)
        
        self.bottom_layout.addStretch(1)

    @Slot()
    @action("interact.set_display_options", description="设置画布显示选项。\n- show_bbox：是否显示边界框（True=显示，False=隐藏，None=保持当前）\n- show_label：是否显示标签（True=显示，False=隐藏，None=保持当前）\n- 控制 draw_area 的 display_controls 显示状态", category="标注", params={"show_bbox": "bool", "show_label": "bool"})
    def on_display_changed(self, show_bbox: bool = None, show_label: bool = None):
        """Called when display checkboxes are toggled."""
        # action调用路径：直接设置checkbox状态
        if show_bbox is not None:
            self.cb_show_bbox.setChecked(show_bbox)
        if show_label is not None:
            self.cb_show_label.setChecked(show_label)
        show_ann = self.cb_show_bbox.isChecked()
        show_label = self.cb_show_label.isChecked()
        self.draw_area.set_display_controls(show_ann, show_label)
        # 广播统一 ui_state 状态（供录制/回放/其他插件观测显示选项）
        if self._event_bus:
            self._event_bus.publish_state("annotation.ui_state",
                {"widget": "display", "show_bbox": bool(show_ann), "show_label": bool(show_label)})


    def reset_view(self):
        """重置全部默认布局：面板显示、显示选项、画布缩放、splitter 比例"""
        # 恢复面板显示
        self._toggle_panel("left", True)
        self._toggle_panel("right", True)
        # 恢复显示选项
        if hasattr(self, 'cb_show_bbox') and not self.cb_show_bbox.isChecked():
            self.cb_show_bbox.setChecked(True)
        if hasattr(self, 'cb_show_label') and not self.cb_show_label.isChecked():
            self.cb_show_label.setChecked(True)
        # 恢复 splitter 比例
        if hasattr(self, 'splitter'):
            self.splitter.setSizes([250, 800, 300])
        # 重置画布视图
        if self.draw_area.image:
            self.draw_area.reset_view()

    @action("interact.reset_canvas_view", description="仅重置画布视图（缩放/平移回到初始状态）。\n- 不影响面板显示、显示选项、splitter 布局", category="画布", scope="ui")
    def reset_canvas_view(self):
        """仅重置画布视图，不恢复面板等其他布局"""
        if self.draw_area.image:
            self.draw_area.reset_view()


    def load_image_annotations(self, strict=False):
        """请求加载当前图的标注 —— **后台 I/O**，结果带代次校验后才提交。

        原先这里是同步读盘 + 解析（非项目模式还会降级二次扫描），发生在每一次切图的
        GUI 线程上，是"IO 全在后台"这条要求里最后一块同步 I/O。现在交给
        `_annotation_runner`：

        * **带代次校验**：结果必须与 `(generation, path)` 都匹配才允许提交，
          快速切图时过期结果一律丢弃；
        * **最新优先**：正在跑时只记住最后一个请求，跑完再发起，所以连按方向键
          不会积压一串读盘任务；
        * 提交前 `session.is_annotations_loaded` 为 False，因此 `save_current` 会被拒
          （不会把上一张图的标注写进这张图的文件），交互闸门也是关的。
        """
        if not self.current_image_path:
            return
        session = self.manual.session
        request = {
            "generation": session.generation,
            "path": self.current_image_path,
            "mode": self.draw_area.task_mode,
            "strict": bool(strict),
            "output_dir": self.output_dir,
            "is_non_project_mode": self._is_non_project_mode,
            "non_project_format": self._non_project_format,
        }
        runner = getattr(self, "_annotation_runner", None)
        if runner is None:
            # 兜底：runner 尚未建立（极早期调用）时退回同步路径
            self._apply_loaded_annotations(self._load_annotations_blocking(request), request)
            return
        if runner.busy():
            self._pending_annotation_request = request      # 最新优先
            return
        self._start_annotation_load(request)

    def _load_annotations_blocking(self, request):
        """同步装载（在**工作线程**上执行）。"""
        # 读之前先等这张图在途的写入落盘：标注保存现在是异步的，若不等，
        # "切走再切回同一张图"时读可能跑在写之前 -> 用户会看到刚改的标注消失。
        if not self.manual.wait_for_pending_writes(request["path"], 5000):
            print("[AnnotationInterface] 等待在途标注写入超时: %s" % request["path"])
        return self.manual.load_annotations_for_image(
            request["path"], request["mode"], request["strict"],
            output_dir=request["output_dir"],
            is_non_project_mode=request["is_non_project_mode"],
            non_project_format=request["non_project_format"])

    def _start_annotation_load(self, request):
        runner = self._annotation_runner
        started = runner.start(
            lambda: self._load_annotations_blocking(request),
            on_result=lambda loaded: self._on_annotations_loaded(request, loaded),
            on_error=lambda message: self._on_annotations_load_failed(request, message),
        )
        if not started:
            self._pending_annotation_request = request
        return started

    def _on_annotations_loaded(self, request, loaded):
        try:
            self._apply_loaded_annotations(loaded, request)
        except Exception as e:
            print(f"[AnnotationInterface] 提交标注装载结果失败: {e}")
            traceback.print_exc()
        finally:
            self._drain_pending_annotation_request()

    def _on_annotations_load_failed(self, request, message):
        print(f"[AnnotationInterface] 标注装载异常: {request['path']}: {message}")
        self.manual.session.mark_annotations_unloaded()
        self._drain_pending_annotation_request()

    def _drain_pending_annotation_request(self):
        pending = self._pending_annotation_request
        self._pending_annotation_request = None
        if pending is not None:
            self._start_annotation_load(pending)

    def _apply_loaded_annotations(self, loaded, request):
        """把后台装载结果提交到 session —— **代次与路径双重校验**通过才落地。"""
        session = self.manual.session
        path = request["path"]
        generation = request["generation"]
        if generation != session.generation or path != session.path:
            print(f"[AnnotationInterface] 丢弃过期标注装载结果: {os.path.basename(path)} "
                  f"(代次 {generation} != {session.generation})")
            return

        res, detected_format = loaded if isinstance(loaded, tuple) else (loaded, None)
        res = res or {}
        if detected_format:
            self._non_project_format = detected_format

        status = res.get("status")
        if status == "success":
            annotations = res.get("annotations") or []
            if request["mode"] == "obb":
                for ann in annotations:
                    self.manual.convert_single_ann_to_obb(ann)
            # for_path + expected_generation 双重校验：这批标注必须属于**当前**图。
            # 这是审计 D2/D3 那类"把 A 图标注装进 B 图当前状态、随后被自动保存
            # 写进 B 图文件"的静默数据损坏的直接拦截点。
            if not self.manual.set_current_annotations(
                    annotations, for_path=path, expected_generation=generation):
                print("[AnnotationInterface] 标注装载结果不属于当前图，已丢弃")
                return
            self._add_new_categories_from_annotations(annotations)
            self.draw_area.update()
            self.refresh_label_list()
            self.status_label.setText(f"已加载现有标注: {len(annotations)} 个")
        elif status == "error":
            # 装载失败：标记为"未成功装载" -> 禁止保存，避免把空标注覆盖到那张
            # 可恢复的坏文件上（审计 D8）。内存里的标注保持为空。
            session.mark_annotations_unloaded()
            print(f"[Warning] 标注加载失败: {res.get('message', '未知错误')}")
            self.status_label.setText(f"标注加载失败: {res.get('message', '未知错误')}")
            self.draw_area.update()
            self.refresh_label_list()
        else:
            self.manual.set_current_annotations([], for_path=path,
                                                expected_generation=generation)
            self.draw_area.update()
            self.refresh_label_list()


    @action("nav.switch_image", description="按方向切换图片。\n- direction：1=下一张，-1=上一张\n- 无图片时无操作\n- 如果 auto_save_enabled 为 True，切换前自动保存当前标注\n- 跳过被筛选隐藏的图片，只加载可见图片\n- 已处于第一张/最后一张可见图片时无操作", category="导航", params={"direction": "int"}, scope="ui")
    def visible_image_indices(self):
        """当前**可见**（未被筛选隐藏）的图片下标集合 —— 导航的唯一真值来源。

        审计阶段 3e：可见性原先有两个来源 —— 界面从 `file_list` 的隐藏状态推导
        （`switch_image`），而 service 侧的 `_navigate_ai` 直接假设"全部可见"
        （`range(len(image_files))`）。两条来源在启用筛选后必然不一致：agent 走
        `next_image` 会跳到**被筛选隐藏**的图片上，而界面按方向键会正确跳过。

        这里统一以 `file_list` 的实际隐藏状态为准，并提供给 service —— 它是界面上
        筛选结果的可视真值，而 `manual.file_visibility` 只是计算中间产物。
        """
        try:
            return {i for i in range(self.file_list.count())
                    if not self.file_list.item(i).isHidden()}
        except Exception:
            return set(range(len(self.image_files)))

    def switch_image(self, direction: int = 1):
        if self.auto_save_enabled:
            self.save_current(silent=True)

        current_idx = self.file_list.currentRow()
        target, error = self.manual.navigate_relative(
            current_idx, direction, self.visible_image_indices())
        if error:
            return error
        if target is not None:
            self.load_image(target)


    @action("interact.toggle_tool", description="切换当前使用的标注工具。\n- tool 参数可选值见 set_drawing_mode 的 mode 说明\n- 切换后 get_state 中的 drawing_mode 会更新\n- 建议切换后调用 get_state 确认状态", category="标注",
            params={"tool": {"type": "str", "enum": ["sam", "ai_rect", "rect", "polygon", "obb", "edit", "roi"],
                             "description": "工具类型: sam=SAM交互, ai_rect=AI矩形, rect=矩形, polygon=多边形, obb=旋转框, edit=编辑, roi=ROI"}}, scope="ui")
    @Slot()
    def on_mode_btn_clicked(self, tool: str = ""):
        # action调用路径：直接设置模式，确保可靠切换
        if tool and isinstance(tool, str):
            mode, auto_task, error = self.manual.resolve_tool_mode(tool)
            if error:
                return error
            # Reset SAM state
            self._sam_active = False
            self.draw_area.sam_points = []
            # 设置绘制模式
            self.draw_area.set_mode(mode)
            if auto_task:
                self.on_task_changed(auto_task)
            # 同步按钮状态
            btn_map = {
                DrawingMode.SAM: self.btn_mode_sam, DrawingMode.AI_RECT: self.btn_mode_ai_rect,
                DrawingMode.RECT: self.btn_mode_rect, DrawingMode.POLYGON: self.btn_mode_poly,
                DrawingMode.OBB: self.btn_mode_obb, DrawingMode.EDIT: self.btn_mode_edit,
                DrawingMode.ROI: self.btn_mode_roi,
            }
            btn = btn_map.get(mode)
            if btn and not btn.isChecked():
                btn.setChecked(True)
            self.update_mode_buttons(mode)
            return

        # 信号调用路径：通过sender判断
        sender = self.sender()
        if not sender: return

        # Reset SAM state when switching modes
        self._sam_active = False
        self.draw_area.sam_points = []

        if sender == self.btn_mode_sam: self.draw_area.set_mode(DrawingMode.SAM)
        elif sender == self.btn_mode_ai_rect: self.draw_area.set_mode(DrawingMode.AI_RECT)
        elif sender == self.btn_mode_rect:
            self.draw_area.set_mode(DrawingMode.RECT)
            self.on_task_changed('det')
        elif sender == self.btn_mode_poly:
            self.draw_area.set_mode(DrawingMode.POLYGON)
            self.on_task_changed('seg')
        elif sender == self.btn_mode_obb:
            self.draw_area.set_mode(DrawingMode.OBB)
            self.on_task_changed('obb')
        elif sender == self.btn_mode_edit: self.draw_area.set_mode(DrawingMode.EDIT)
        elif sender == self.btn_mode_roi: self.draw_area.set_mode(DrawingMode.ROI)

        self.update_mode_buttons(self.draw_area.mode)


    @action("nav.switch_project", description="切换到指定项目。\n- project_name：项目名称\n- 切换后自动刷新文件列表\n- 可在项目界面调用，无需先切换到标注界面\n- 切换后文件列表随事件自动刷新，可观察确认", category="导航", params={"project_index": "int", "project_name": "str"}, scope="ui")
    def on_project_combo_changed(self, index=None, project_index: int = -1, project_name: str = ""):
        # action调用路径：通过project_name或project_index定位项目
        if project_name:
            # 根据项目名找到combo中对应项
            for i in range(self.project_combo.count()):
                if self.project_combo.itemText(i) == project_name:
                    if self.project_combo.currentIndex() != i:
                        self.project_combo.setCurrentIndex(i)
                    else:
                        # 已经选中，直接加载
                        project_info = self.manual.load_project(project_name)
                        if project_info:
                            self.set_project(project_info)
                    return
            return "错误: 项目 '%s' 不存在" % project_name
        elif project_index >= 0:
            # 根据项目索引找到项目名
            all_projects = self.project_service.get_all_projects()
            if project_index < len(all_projects):
                name = all_projects[project_index]['name']
                for i in range(self.project_combo.count()):
                    if self.project_combo.itemText(i) == name:
                        if self.project_combo.currentIndex() != i:
                            self.project_combo.setCurrentIndex(i)
                        else:
                            project_info = self.manual.load_project(name)
                            if project_info:
                                self.set_project(project_info)
                        return
            return "错误: 项目索引 %d 超出范围（共 %d 个项目）" % (project_index, len(all_projects))

        # 信号调用路径：index参数来自currentIndexChanged信号
        if index is None or index < 0:
            return

        project_name = self.project_combo.itemText(index)
        project_info = self.manual.load_project(project_name)
        if project_info:
            self.set_project(project_info)


    def refresh_project_list(self):
        service = self.project_service or ProjectService()
        projects = service.get_all_projects()

        self.project_combo.blockSignals(True)
        try:
            self.project_combo.clear()
            # 使用 self.current_project_name 而不是 settings.get，保持一致性
            current_name = self.current_project_name

            current_idx = -1
            for i, p in enumerate(projects):
                self.project_combo.addItem(p['name'])
                if p['name'] == current_name:
                    current_idx = i

            if current_idx >= 0:
                self.project_combo.setCurrentIndex(current_idx)
        finally:
            self.project_combo.blockSignals(False)

        # 同步更新菜单栏中的项目子菜单
        self._rebuild_project_menu(projects)


    @action("nav.open_directory", description="打开图片目录加载图片进行标注。\n- directory: 目录路径\n- 支持标准项目结构（images/annotations 子目录）和非规范化目录\n- 非规范化目录以非项目模式打开", category="导航", params={"directory": "str"}, scope="ui")
    @Slot()
    def open_directory(self, directory: str = ""):
        if not directory:
            return "错误: open_directory 需要 directory 参数"
        dir_path = directory
        if not os.path.isdir(dir_path):
            return f"错误: 目录不存在: {dir_path}"

        is_normalized = (
            os.path.exists(os.path.join(dir_path, "images")) and
            os.path.exists(os.path.join(dir_path, "annotations"))
        )

        if not is_normalized:
            scan_result = self.manual.enter_non_project_mode(dir_path)
            self._update_non_project_mode_ui()
            config_loaded = self._load_non_project_config(dir_path)
            if config_loaded:
                detected_task_type = self.draw_area.task_mode
            else:
                detected_task_type = scan_result.get("detected_task_type")
                if scan_result.get("detected_format"):
                    self._non_project_format = scan_result["detected_format"]
                    settings.set("export_format", scan_result["detected_format"])
            if detected_task_type:
                self.draw_area.task_mode = self.manual.apply_detected_task_type(detected_task_type)
            if not self.categories:
                self.draw_area.set_category_colors(self.categories)
            self.manual.set_non_project_mode(self.output_dir)
            self.load_directory_images(dir_path)
            return

        self._is_non_project_mode = False
        self._update_non_project_mode_ui()
        self.load_directory_images(dir_path)

    def _open_directory_ui(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.open_directory(directory=dir_path)
    
    
    
    
    
    


    

    # ---- 列表重建的"发射栈安全"闸门（审计 D16）----

    def _begin_list_rebuild(self):
        """进入列表重建区：告知派发器"此刻正在重建列表"。

        重建会销毁并重建列表行（及其上的复选框/探测控件）。若这个动作发生在某个
        子控件**正在发射信号**的调用栈里（例如 `stateChanged` 的回调里触发重建），
        就会在发射栈上销毁发起者 —— Qt 下这是未定义行为（崩溃或静默错乱）。
        """
        self._list_rebuilding = True

    def _end_list_rebuild(self):
        self._list_rebuilding = False

    def request_list_rebuild(self, fn, from_signal=False):
        """请求重建列表；若当前正处于"子控件发射栈"上则改为**延后**执行。

        为什么不能靠 `self.sender()` 判断：在 lambda / 嵌套辅助函数里调用时
        `sender()` 返回 None（Qt 只保证在**直接**槽函数里有效），实测因此漏判 ——
        T4 显示重建仍在发射栈内同步执行。

        因此改为**显式声明**：凡是从信号槽（按钮 clicked、列表行控件 stateChanged、
        快捷键触发等）里发起的重建，调用方传 `from_signal=True`，这里就把重建推迟到
        事件循环的下一轮 —— 那时发射栈已经展开完毕，销毁发起控件是安全的。
        """
        if self._list_rebuilding:
            # 已经在一个重建流程里 —— 直接执行即可（不会再嵌套销毁）
            return fn()
        if from_signal:
            QTimer.singleShot(0, fn)
            return None
        return fn()

    def load_directory_images(self, dir_path):
        """Loads all images from a directory (or project root) and updates UI."""
        try:
            print(f"[AnnotationInterface] Loading directory images from: {dir_path}")
            files, last_idx = self.manual.load_directory(dir_path)
            self.image_files = files

            # 重建列表时**必须屏蔽所有信号**（审计 D16 第一部分）：
            # 原先只屏蔽了 file_list，而 `takeItem`/`addItem` 期间若有探测器或
            # 复选框的 stateChanged 正在发射，就会在**发射栈上销毁发起控件**；
            # `_populate_file_thumbnails()` 也会触发 itemChanged/滚动等信号。
            # 这里统一把 file_list 与 search_box 都关掉，重建完再打开。
            self._begin_list_rebuild()
            try:
                # 信号必须**贯穿整个重建过程**保持屏蔽（审计 D16）：原先只在
                # `takeItem` 循环前后屏蔽，而 `addItem` 之后的
                # `_populate_file_thumbnails()` 会设置 item 数据并触发
                # `itemChanged`（实测 5 行 = 5 次），也就是说重建期间仍有信号
                # 打到外部 —— 那正是"在发射栈上重建"的入口。
                self.file_list.blockSignals(True)
                try:
                    while self.file_list.count() > 0:
                        self.file_list.takeItem(0)
                    for f in self.image_files:
                        item = QListWidgetItem(os.path.basename(f))
                        item.setData(Qt.UserRole, f)
                        item.setSizeHint(QSize(0, 52))
                        self.file_list.addItem(item)
                    self._populate_file_thumbnails()

                    # 清空搜索框必然会发射 textChanged；不屏蔽就会在"目录刚重建、
                    # currentRow 尚未确定"时启动一次搜索/去抖，造成状态被并发改写
                    # （审计 D16：原先这里没有 blockSignals）。
                    self.search_box.blockSignals(True)
                    try:
                        self.search_box.clear()
                    finally:
                        self.search_box.blockSignals(False)
                finally:
                    self.file_list.blockSignals(False)
            finally:
                self._end_list_rebuild()

            self.dir_label.setText(os.path.basename(dir_path))
            self.dir_label.setToolTip(dir_path)

            if self.image_files:
                if 0 <= last_idx < len(self.image_files):
                    self.load_image(last_idx)
                else:
                    self.load_image(0)
            else:
                self.manual.set_current_image(None)
                self.draw_area.clear()
                self.status_label.setText("当前目录无图片")
                self.update_index_label()

            self.update_index_label()

            print(f"[AnnotationInterface] Directory images loaded: {len(files)} files")
        except Exception as e:
            print(f"[AnnotationInterface] Error in load_directory_images: {e}")
            traceback.print_exc()


    @Slot()
    def clear_project(self):
        """清空当前项目相关的路径、图片和画布显示，用于项目被删除后恢复到空状态。"""
        self.output_dir = ""
        self.manual.clear_all()
        # 项目被清空：派生签名缓存必须作废，否则"清空后再打开同签名数据"会被跳过（D33）
        self.reset_annotations_refresh_state()
        self.image_files = self.manual.image_files
        # 使用 takeItem 逐个删除，避免 QFluentWidgets ListWidget 的 clear() 崩溃问题
        self.file_list.blockSignals(True)
        while self.file_list.count() > 0:
            self.file_list.takeItem(0)
        self.file_list.blockSignals(False)
        self.search_box.clear()
        self.dir_label.setText('未选择目录')
        self.dir_label.setToolTip("")
        self.index_label.setText('0 / 0')
        self.draw_area.clear()
        
    def on_projects_changed(self, data):
        """订阅 dashboard:projects_changed：当前项目被删除时清空标注界面。"""
        if not data or data.get("action") != "deleted":
            return
        deleted = data.get("project_name")
        current = getattr(self, 'current_project_name', None)
        if current and deleted == current:
            print(f"[AnnotationInterface] 当前项目已删除: {deleted}，清空界面")
            self.clear_project()
            self.current_project_name = None

    def load_dataset_splits(self):
        self.image_splits = self.manual.dataset_splits("load", self.output_dir)

    def save_dataset_splits(self):
        self.manual.dataset_splits("save", self.output_dir)


    def load_image(self, index, force_reload=False):
        try:
            if 0 <= index < len(self.image_files):
                new_image_path = self.image_files[index]

                # If it's the same image and not forced, don't clear interaction state
                if not force_reload and self.current_image_path == new_image_path:
                    self.file_list.setCurrentRow(index)
                    return

                # 路径由 service.set_current_image_with_split() 走 session 提交，
                # interface 不再自己赋值一份（唯一状态所有者）
                split = self.manual.set_current_image_with_split(new_image_path)
                norm_path = os.path.normpath(self.current_image_path)

                self.manual.clear_undo_history()
                self.manual.clear_visibility_state()
                # 标注真值已被整体替换（换图）：派生出来的"上次刷新签名"必须作废，
                # 否则新图标注恰好同签名时刷新会被跳过（D33）
                self.reset_annotations_refresh_state()

                # 安全地调用 set_image
                try:
                    self.draw_area.set_image(self.current_image_path)
                except Exception as e:
                    print(f"[AnnotationInterface] Error in draw_area.set_image: {e}")
                    traceback.print_exc()
                    return

                self.file_list.setCurrentRow(index)
                self.status_label.setText(f"当前: {os.path.basename(self.current_image_path)} ({index+1}/{len(self.image_files)})")
                self.update_index_label()
                
                # 更新分辨率显示
                self.update_resolution_label()

                # Persistence: Save last index for this directory
                self.manual.save_last_index(index)

                # Load split selection
                split_idx = {"train": 0, "val": 1, "test": 2}.get(split, 0)
                self.split_combo.blockSignals(True)
                self.split_combo.setCurrentIndex(split_idx)
                self.split_combo.blockSignals(False)

                # Reset SAM state when switching images
                self._sam_active = False
                self.draw_area.sam_points = []

                # Load existing annotations if they exist
                try:
                    self.load_image_annotations()
                except Exception as e:
                    print(f"[AnnotationInterface] Error in load_image_annotations: {e}")
                    traceback.print_exc()

                # 这里**不再**重复调用 refresh_label_list()：标注装载现在在后台完成，
                # 由 _apply_loaded_annotations 在提交后刷新一次即可。原先同步路径下
                # 这里会再重建一遍列表（每次切图两次全量重建，20 条标注约 240ms）。

                # 更新难样本标记状态
                self._update_hard_sample_checkbox()

                # 如果已设置副标注文件夹，自动加载当前图像的副标注
                if self.manual.secondary_loaded and self.manual.secondary_dir:
                    try:
                        self._load_secondary_for_current_image()
                    except Exception as e:
                        print(f"[AnnotationInterface] Error loading secondary annotations: {e}")
                        traceback.print_exc()
        except Exception as e:
            print(f"[AnnotationInterface] Error in load_image: {e}")
            traceback.print_exc()


    @action("nav.select_image", description="跳转到指定索引的图片。\n- image_index：从 0 开始的图片索引\n- 可用 get_state 获取当前图片索引和总图片数\n- 也可用 switch_image(direction=±1) 逐张切换", category="标注", scope="ui",
            params={"image_index": {"type": "int", "description": "图片索引，从0开始"}})
    def on_file_item_clicked(self, item=None, image_index: int = -1):
        # 兼容信号调用(item为QListWidgetItem)和action调用(image_index为int)
        if image_index >= 0:
            row = image_index
        else:
            row = self.file_list.row(item)
        # 确保通过 @action 调用时传入正确的 image_index
        if image_index < 0 and row >= 0:
            image_index = row
        if self.auto_save_enabled:
            self.save_current(silent=True)
        self.load_image(row)


    def show_roi_context_menu(self, global_pos):
        """Shows a simplified menu for the persistent ROI"""
        if isinstance(global_pos, QPointF):
            global_pos = global_pos.toPoint()

        menu = RoundMenu(parent=self)

        send_action = Action(FIF.PHOTO, "添加到对话")
        send_action.triggered.connect(lambda: self._send_reference("@ROI"))
        menu.addAction(send_action)

        clear_action = Action(FIF.DELETE, "清除持久化 ROI")
        clear_action.triggered.connect(self.clear_persistent_roi)
        menu.addAction(clear_action)

        cancel_action = Action(FIF.CLOSE, "取消")
        menu.addAction(cancel_action)

        menu.exec(global_pos)


    @action("interact.clear_persistent_roi", description="清除当前设置的持久化 ROI 区域。\n- 持久化 ROI 由 set_persistent_roi 设置\n- 清除后画布恢复完整视图\n- 不会影响已有标注数据", category="标注", scope="ui")
    @Slot()
    def clear_persistent_roi(self):
        """Clears the persistent ROI"""
        if not self.draw_area.persistent_roi:
            return "错误: 没有设置持久化 ROI"
        self.draw_area.persistent_roi = None
        self.draw_area.update()
        self.status_label.setText("项目 ROI 已清除")

    # ====== 区域: 类别管理 & 标注编辑 & 筛选审查（原 edit_mixin） ======
    # ============================================================
    # edit.*  action methods
    # ============================================================

    @action("edit.delete_selected", description="删除当前选中的标注或点(Delete键)。\n- 有批量选中标注时：删除所有选中的标注\n- 有批量选中点时：删除选中的多边形顶点\n- 有单个选中标注时：删除该标注\n- 优先使用 edit.select_in_rect 批量选择后再调用此 action", category="标注", scope="ui")
    def _on_delete_shortcut(self):
        draw_area = self.draw_area
        if not draw_area.annotations:
            return "错误: 没有标注可删除"

        has_selected_annotations = hasattr(draw_area, 'batch_selected_indices') and draw_area.batch_selected_indices
        has_selected_points = hasattr(draw_area, 'batch_selected_points') and draw_area.batch_selected_points

        if has_selected_annotations or has_selected_points:
            self.manual.save_undo_state(draw_area.annotations, self.current_image_path)

            if has_selected_annotations:
                for idx in sorted(draw_area.batch_selected_indices, reverse=True):
                    if 0 <= idx < len(draw_area.annotations):
                        self.manual.remove_annotation(idx)
                draw_area.batch_selected_indices = []
                draw_area.batch_selected_points = []

            elif has_selected_points:
                self.manual.delete_annotation_points(
                    draw_area.annotations, draw_area.batch_selected_points
                )
                draw_area.batch_selected_points = []

            # 经延后闸门重建列表（审计 D16）：本函数由 label_list 上的删除按钮
            # 触发，直接重建会在**发射栈上销毁发起控件**
            self.request_list_rebuild(self.refresh_label_list, from_signal=True)
            draw_area.update()

        elif draw_area.selected_index != -1:
            new_anns, new_hidden, err = self.manual.delete_annotation(
                draw_area.selected_index, draw_area.annotations,
                draw_area.hidden_indices, self.current_image_path
            )
            if err:
                return err
            self.manual.set_current_annotations(new_anns)
            draw_area.selected_index = -1
            self.request_list_rebuild(self.refresh_label_list, from_signal=True)

        elif draw_area.hover_index != -1:
            new_anns, new_hidden, err = self.manual.delete_annotation(
                draw_area.hover_index, draw_area.annotations,
                draw_area.hidden_indices, self.current_image_path
            )
            if err:
                return err
            self.manual.set_current_annotations(new_anns)
            draw_area.hover_index = -1
            self.request_list_rebuild(self.refresh_label_list, from_signal=True)

    @action("edit.select_annotation", description="按方向选择标注并自动聚焦（下键/上键）。\n- direction：1=下一个，-1=上一个\n- 无标注时无操作\n- 循环选择：最后一个之后回到第一个，反之亦然\n- 选中后自动聚焦放大标注", category="标注", params={"direction": "int"}, scope="ui")
    def select_annotation(self, direction: int = 1):
        if direction >= 0:
            idx, error = self.manual.next_annotation_index(
                len(self.draw_area.annotations), self.draw_area.selected_index
            )
        else:
            idx, error = self.manual.prev_annotation_index(
                len(self.draw_area.annotations), self.draw_area.selected_index
            )
        if error:
            return error
        self.draw_area.selected_index = idx
        self.label_list.setCurrentRow(idx)
        self.draw_area.focus_on_annotation(idx)
        self.draw_area.update()

    def _on_return_shortcut(self):
        if hasattr(self, '_sam_active') and self._sam_active:
            self._sam_active = False
            self.draw_area.sam_points = []
            self.draw_area.update()
        elif self.draw_area.mode == DrawingMode.POLYGON and len(self.draw_area.current_poly) > 2:
            pts = [[int(p.x()), int(p.y())] for p in self.draw_area.current_poly]
            self.on_annotation_finished(pts)
            self.draw_area.current_poly = []
            self.draw_area.update()

    def _save_undo_state(self):
        self.manual.save_undo_state(self.draw_area.annotations, self.current_image_path)

    def _on_annotation_modified(self):
        self.save_current(silent=True)

    @action("edit.step_history", description="撤销/重做标注操作（Ctrl+Z/Ctrl+Y）。\n- direction：-1=撤销，1=重做\n- 支持多次撤销，最多可撤销 50 步\n- 撤销后可通过 direction=1 重做恢复\n- 不会撤销对标注类别/可见性的修改", category="标注", params={"direction": "int"}, scope="ui")
    def step_history(self, direction=-1):
        restored, err = self.manual.step_history(direction, self.draw_area.annotations, self.current_image_path)
        if err:
            return err
        # 标注数据刷新由 annotation:annotations_changed 事件处理器完成，此处仅重置 UI 选中态
        self.draw_area.selected_index = -1
        self.draw_area.hover_index = -1

    def show_draw_context_menu(self, index, global_pos):
        menu = RoundMenu(parent=self)

        if index >= 0:
            set_example_action = Action(FIF.LABEL, "设为示例")
            set_example_action.triggered.connect(lambda: self._on_set_as_example_with_dialog(index))
            menu.addAction(set_example_action)

            change_cat_action = Action(FIF.TAG, "切换类别")
            change_cat_action.triggered.connect(lambda: self.on_update_annotation_category(index))
            menu.addAction(change_cat_action)

            if self.manual.secondary_loaded and index in self.manual.secondary_annotation_map:
                accept_secondary_action = Action(FIF.ACCEPT, "接收副标注")
                accept_secondary_action.triggered.connect(lambda: self.on_accept_secondary_annotation(index))
                menu.addAction(accept_secondary_action)

            delete_action = Action(FIF.DELETE, "删除标注")
            delete_action.triggered.connect(lambda: self.delete_annotation(index))
            menu.addAction(delete_action)

            img_name = os.path.basename(self.current_image_path) if self.current_image_path else ""
            if img_name:
                send_menu = RoundMenu("添加到对话", parent=self)
                act_local = Action(FIF.PHOTO, "标注区域")
                act_local.triggered.connect(
                    lambda: self._send_reference(f"@局部:{img_name}标注{index}"))
                act_ov_local = Action(FIF.PHOTO, "渲染局部")
                act_ov_local.triggered.connect(
                    lambda: self._send_reference(f"@渲染局部:{img_name}标注{index}"))
                send_menu.addAction(act_local)
                send_menu.addAction(act_ov_local)
                menu.addMenu(send_menu)

            ctx_vertex = getattr(self.draw_area, "context_vertex_info", None)
            ctx_edge = getattr(self.draw_area, "context_edge_info", None)
            if self.draw_area.task_mode in ["seg", "obb"]:
                if ctx_vertex and ctx_vertex[0] == index:
                    ann_idx, poly_idx, pt_idx = ctx_vertex
                    vertex_del_action = Action(FIF.DELETE, "删除顶点")
                    vertex_del_action.triggered.connect(
                        lambda checked, a=ann_idx, p=poly_idx, t=pt_idx: self.draw_area.delete_poly_point(a, p, t)
                    )
                    menu.addAction(vertex_del_action)
                if ctx_edge and ctx_edge[0] == index:
                    ann_idx, poly_idx, edge_idx, x, y = ctx_edge
                    edge_add_action = Action(FIF.ADD, "添加顶点")
                    edge_add_action.triggered.connect(
                        lambda checked, a=ann_idx, p=poly_idx, e=edge_idx, xx=x, yy=y: self.draw_area.add_poly_point(a, p, e, img_pos=QPointF(xx, yy))
                    )
                    menu.addAction(edge_add_action)

            menu.addSeparator()

        if self.manual.secondary_loaded:
            clear_current_secondary_action = Action(FIF.CANCEL, "清除当前副标注")
            clear_current_secondary_action.triggered.connect(lambda: self.clear_secondary_annotations(clear_dir=False))
            menu.addAction(clear_current_secondary_action)

            close_secondary_mode_action = Action(FIF.CLOSE, "关闭副标注模式")
            close_secondary_mode_action.triggered.connect(lambda: self.clear_secondary_annotations(clear_dir=True))
            menu.addAction(close_secondary_mode_action)
            menu.addSeparator()

        undo_action = Action(FIF.LEFT_ARROW, "撤销 (Ctrl+Z)")
        undo_action.triggered.connect(lambda: self.step_history(-1))
        undo_action.setEnabled(self.manual.has_undo())
        menu.addAction(undo_action)

        redo_action = Action(FIF.RIGHT_ARROW, "重做 (Ctrl+Y)")
        redo_action.triggered.connect(lambda: self.step_history(1))
        redo_action.setEnabled(self.manual.has_redo())
        menu.addAction(redo_action)

        menu.addSeparator()

        clear_action = Action(FIF.REMOVE, "清除所有标注")
        clear_action.triggered.connect(self.clear_all_annotations)
        menu.addAction(clear_action)

        menu.exec(global_pos)
        if hasattr(self, "draw_area"):
            self.draw_area.ignore_next_left_press = False

    def show_secondary_context_menu(self, secondary_idx, global_pos):
        menu = RoundMenu(parent=self)

        add_as_primary_action = Action(FIF.ADD, "添加为主标注")
        add_as_primary_action.triggered.connect(lambda: self.on_add_secondary_as_primary(secondary_idx))
        menu.addAction(add_as_primary_action)

        menu.addSeparator()

        if self.manual.secondary_loaded:
            clear_current_secondary_action = Action(FIF.CANCEL, "清除当前副标注")
            clear_current_secondary_action.triggered.connect(lambda: self.clear_secondary_annotations(clear_dir=False))
            menu.addAction(clear_current_secondary_action)

            close_secondary_mode_action = Action(FIF.CLOSE, "关闭副标注模式")
            close_secondary_mode_action.triggered.connect(lambda: self.clear_secondary_annotations(clear_dir=True))
            menu.addAction(close_secondary_mode_action)

        menu.exec(global_pos)
        if hasattr(self, "draw_area"):
            self.draw_area.ignore_next_left_press = False

    @action("edit.add_secondary_as_primary", description="将副标注添加为主标注。\n- secondary_idx：副标注在 secondary_annotations 列表中的索引\n- 前提条件：已加载副标注（load_secondary_annotations），且索引有效\n- 操作后：副标注会被复制并添加到主标注列表末尾，同时建立映射关系显示为 [主+副]\n- 自动保存 undo 状态，可通过 undo 撤销", category="标注", params={"index": "int"}, scope="ui")
    def on_add_secondary_as_primary(self, secondary_idx: int = -1):
        new_annotations, new_primary_idx, err = self.manual.add_secondary_as_primary(
            secondary_idx, self.draw_area.annotations, self.current_image_path
        )
        if err:
            return err

        # 标注列表/画布刷新由 annotation:annotations_changed 事件处理器完成
        InfoBar.success("添加成功", f"已将副标注添加为主标注 #{new_primary_idx + 1}，显示为 [主+副]", parent=self)

    def show_label_context_menu(self, pos):
        item = self.label_list.itemAt(pos)
        menu = RoundMenu(parent=self)

        if item:
            idx = self.label_list.row(item)

            set_example_action = Action(FIF.LABEL, "设为示例")
            set_example_action.triggered.connect(lambda: self._on_set_as_example_with_dialog(idx))
            menu.addAction(set_example_action)

            change_cat_action = Action(FIF.TAG, "切换类别")
            change_cat_action.triggered.connect(lambda: self.on_update_annotation_category(idx))
            menu.addAction(change_cat_action)

            delete_action = Action(FIF.DELETE, "删除标注")
            delete_action.triggered.connect(lambda: self.delete_annotation(idx))
            menu.addAction(delete_action)

            img_name = os.path.basename(self.current_image_path) if self.current_image_path else ""
            if img_name:
                send_menu = RoundMenu("添加到对话", parent=self)
                act_local = Action(FIF.PHOTO, "标注区域")
                act_local.triggered.connect(
                    lambda: self._send_reference(f"@局部:{img_name}标注{idx}"))
                act_ov_local = Action(FIF.PHOTO, "渲染局部")
                act_ov_local.triggered.connect(
                    lambda: self._send_reference(f"@渲染局部:{img_name}标注{idx}"))
                send_menu.addAction(act_local)
                send_menu.addAction(act_ov_local)
                menu.addMenu(send_menu)

            menu.addSeparator()

        clear_action = Action(FIF.REMOVE, "清除所有标注")
        clear_action.triggered.connect(self.clear_all_annotations)
        menu.addAction(clear_action)

        menu.exec(self.label_list.mapToGlobal(pos))

    def save_categories_to_settings(self):
        self.manual.save_categories_to_settings()

    @action("edit.change_annotation_category", description="修改指定标注的类别。\n- index：标注在 annotations 列表中的索引（从 0 开始）\n- category：目标类别名称，为空时弹出类别选择对话框\n- 前提条件：index 有效\n- 操作后自动保存当前标注", category="标注", params={"index": "int", "category": "str"}, scope="ui")
    def on_update_annotation_category(self, index: int = -1, category: str = ""):
        if index < 0 or index >= len(self.draw_area.annotations):
            return "错误: 标注索引无效"
        if not category:
            if not self.categories:
                return "错误: 没有可用的类别"
            current_label = self.draw_area.annotations[index].get('label', '')
            dialog = CategorySelectionDialog(self.categories, current_label, self, recent_order=self._recent_categories)
            if not dialog.exec_():
                return None
            category = dialog.get_selected_category()
            if not category or category == current_label:
                return None
        new_annotations, err = self.manual.update_annotation_label(
            index, category, self.draw_area.annotations, self.categories
        )
        if err:
            return err
        # service 已保存并发布 annotations_changed，标注列表/画布刷新交给事件处理器
        self._touch_recent_category(category)
        return True

    @action("edit.clear_annotations", description="清除当前图片的所有标注。\n- 静默清除并自动保存\n- 清除后可通过画布右键「撤销」恢复", category="标注", scope="ui")
    def clear_all_annotations(self):
        new_annotations, err = self.manual.clear_annotations(
            self.draw_area.annotations, self.current_image_path
        )
        if err:
            return err
        # service 已保存并发布 annotations_changed，标注列表/画布刷新交给事件处理器

    def _clear_all_annotations_ui(self):
        if not self.draw_area.annotations:
            InfoBar.warning("提示", "当前没有标注", parent=self)
            return
        w = MessageBox("确认清除", "是否确定清除当前图片的所有标注？清除后仍可通过画布右键「撤销」恢复。", self)
        if not w.exec():
            return
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if recorder.is_recording:
            recorder.record_step(
                action_name="clear_annotations",
                params={},
                description="清除所有标注",
                expected_result="弹出确认清除对话框"
            )
        new_annotations, err = self.manual.clear_annotations(
            self.draw_area.annotations, self.current_image_path
        )
        if err:
            InfoBar.warning("提示", err, parent=self)
            return
        # service 已保存并发布 annotations_changed，标注列表/画布刷新交给事件处理器
        InfoBar.success("已清除", "当前图片的所有标注已清除", parent=self)

    def on_label_item_double_clicked(self, item):
        idx = self.label_list.row(item)
        self._on_change_annotation_category_with_dialog(idx)

    def _on_change_annotation_category_with_dialog(self, index: int):
        if index < 0 or index >= len(self.draw_area.annotations):
            return
        ann = self.draw_area.annotations[index]
        current_label = ann.get('label', 'default')
        dialog = CategorySelectionDialog(self.categories, current_label, self, recent_order=self._recent_categories)
        if dialog.exec_():
            new_cat = dialog.get_selected_category()
            if new_cat and new_cat != current_label:
                result = self.on_update_annotation_category(index=index, category=new_cat)
                if result is True:
                    InfoBar.success("类别已修改", f"已将目标 {index + 1} 的类别修改为 '{new_cat}'", parent=self)
                    from core.common.action_recorder import ActionRecorder
                    recorder = ActionRecorder.instance()
                    if recorder.is_recording:
                        recorder.record_step(
                            action_name="change_annotation_category",
                            params={"index": index},
                            description=f"修改标注类别(标注#{index+1})",
                            expected_result="弹出类别选择对话框"
                        )

    @action("edit.delete_annotation", description="删除指定索引的标注。\n- index：要删除的标注在 annotations 列表中的索引（从 0 开始）\n- 操作前自动保存 undo 状态，可通过 undo 撤销\n- 删除后自动更新 hidden_indices 索引集合\n- 如果 auto_save_enabled 为 True，自动保存当前标注", category="标注", scope="ui",
            params={"index": {"type": "int", "description": "要删除的标注索引"}})
    def delete_annotation(self, index: int = -1):
        new_annotations, new_hidden, err = self.manual.delete_annotation(
            index, self.draw_area.annotations,
            self.draw_area.hidden_indices, self.current_image_path
        )
        if err:
            return err
        # service 已保存并发布 annotations_changed，标注列表/画布刷新交给事件处理器

    def delete_annotation_by_index(self, index: int):
        self.delete_annotation(index)

    @action("interact.refine_single_annotation", description="对指定标注调用 refine 模型进行精细化分割。\n- index：标注在 annotations 列表中的索引\n- 前提条件：标注包含多边形数据（polygons）或 refine 模型为 hybrid 时含矩形框，当前图片已加载\n- 使用配置的 refine_method（vitmatte/cascadepsp/hybrid）进行细化\n- 细化后自动替换原标注的多边形轮廓", category="标注", params={"index": "int"}, scope="ui")
    def refine_single_annotation(self, index: int = -1):
        result = self.manual.refine_annotation(
            index, self.draw_area.annotations, self.current_image_path
        )
        if result.get("status") != "success":
            msg = result.get("message", "")
            if "无有效轮廓" in msg:
                return "细化后无有效轮廓，保持原标注"
            return f"错误: 细化失败: {msg}"
        # service 已保存并发布 annotations_changed，标注列表/画布刷新交给事件处理器

    def _on_refine_shortcut(self):
        idx = self.draw_area.selected_index
        if idx < 0 and self.draw_area.hover_index >= 0:
            idx = self.draw_area.hover_index
        if idx >= 0:
            result = self.refine_single_annotation(idx)
            if isinstance(result, str) and result.startswith("错误"):
                InfoBar.error("细化失败", result, parent=self)
            elif isinstance(result, str) and "无有效轮廓" in result:
                self.status_label.setText(result)
            else:
                self.status_label.setText("标注细化完成")
        else:
            InfoBar.info("提示", "请先选择一个标注再进行细化", parent=self)

    @action("edit.toggle_hard_sample", description="切换当前图片的难样本标记。\n- state：标记状态（True=标记为难样本，False=取消标记，None=切换当前状态）\n- 标记为 hard sample 时，描述输入框会被启用\n- 状态自动保存到项目配置（project_info.json）或非项目配置文件中\n- 先调用 nav.select_image 切换到目标图片后再执行此操作", category="标注", scope="ui")
    def _on_hard_sample_toggled(self, state=None):
        if state is None:
            is_hard = self.cb_hard_sample.isChecked()
        else:
            is_hard = bool(state)

        desc = self.hard_sample_desc_edit.text().strip() if is_hard else ""

        hard_samples, err = self.manual.toggle_hard_sample(
            is_hard, self.current_image_path, desc
        )
        if err:
            return err

        # 勾选框/描述/计数由 annotation:hard_sample_changed 事件处理器统一刷新，此处仅持久化
        self.manual.save_hard_samples_config()

    def _save_hard_samples_config(self):
        self.manual.save_hard_samples_config()

    def _load_hard_samples_config(self):
        hard_samples = self.manual.load_hard_samples_config()
        self.hard_sample_count_label.setText(str(len(hard_samples)))

    def _update_hard_sample_checkbox(self):
        self.cb_hard_sample.blockSignals(True)
        if self.current_image_path:
            is_hard, desc = self.manual.get_hard_sample_state(self.current_image_path)
            self.cb_hard_sample.setChecked(is_hard)
            self.hard_sample_desc_edit.setText(desc if is_hard else "")
            self.hard_sample_desc_edit.setEnabled(is_hard)
        else:
            self.cb_hard_sample.setChecked(False)
            self.hard_sample_desc_edit.setText("")
            self.hard_sample_desc_edit.setEnabled(False)
        self.cb_hard_sample.blockSignals(False)

    def on_label_item_clicked(self, item):
        idx = self.label_list.row(item)

        if self.draw_area.focused_index == idx:
            self.draw_area.unfocus()
        else:
            self.draw_area.selected_index = idx
            self.draw_area.focus_on_annotation(idx)

        self.draw_area.update()

    @action("edit.toggle_annotations_visibility", description="切换标注的显示/隐藏（UI 包装：调用 manual.toggle_annotation_visibility 后刷新画布与复选框）。\n- scope：all=切换全部标注（全部显示→全部隐藏，有隐藏→全部显示），current=切换当前选中标注\n- scope=all 无标注时返回错误\n- scope=current 需先用 edit.select_annotation_at 选中标注\n- 用于菜单/快捷键入口；agent 直接调用 service 层 manual.toggle_annotation_visibility(index=-1 切换全部 / index=具体值 切换单个)", category="标注", params={"scope": "str"}, scope="ui")
    def toggle_annotations_visibility(self, scope="current"):
        # 合并调用：scope=all 传 -1 切换全部，scope=current 传选中索引切换单个
        idx = -1 if scope == "all" else self.draw_area.selected_index
        err = self.manual.toggle_annotation_visibility(idx)
        if err:
            return err
        # 标注列表/画布刷新由 annotation:annotations_changed 事件处理器完成

    def _sync_visibility_checkbox(self, index, visible):
        item = self.label_list.item(index)
        if item is None:
            return
        widget = self.label_list.itemWidget(item)
        if widget is None:
            return
        cb = widget.findChild(CheckBox)
        if cb:
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    @action("edit.set_annotation_visibility", description="设置指定标注的可见性（UI 包装：调用 manual.set_annotation_visibility 后刷新画布）。\n- index：标注索引（从 0 开始）\n- visible：True=显示，False=隐藏\n- 索引无效时返回错误\n- 用于标注列表复选框交互；agent 直接调用 service 层 manual.set_annotation_visibility", category="标注", params={"index": "int", "visible": "bool"}, scope="ui")
    def on_annotation_visibility_changed(self, state=None, index: int = -1, visible: bool = None):
        # 重建标注列表时跳过信号处理，防止 stateChanged 干扰 hidden_indices
        if getattr(self, '_is_rebuilding_labels', False):
            return
        if index >= 0 and visible is not None:
            err = self.manual.set_annotation_visibility(index, visible)
            if err:
                return err
            return  # 画布/复选框刷新由 annotation:annotations_changed 事件处理器完成
        try:
            cb = self.sender()
            if cb is None:
                return
            idx = cb.property("annotation_idx")
            if idx is None:
                return
            visible = state == Qt.CheckState.Checked.value
            err = self.manual.set_annotation_visibility(idx, visible)
            # 画布/复选框刷新由 annotation:annotations_changed 事件处理器完成
            return err
        except RuntimeError:
            pass

    def on_annotation_selected(self, index):
        if index == -1:
            self.label_list.clearSelection()
            if self.draw_area.focused_index != -1:
                self.draw_area.unfocus()
        else:
            if index >= len(self.draw_area.annotations):
                return
            self.label_list.setCurrentRow(index)
            if self.draw_area.focused_index != -1 and not self.draw_area.dragging_point and self.draw_area.dragging_annotation == -1:
                self.draw_area.focus_on_annotation(index)

    # ============================================================
    # category.*  action methods
    # ============================================================

    def on_export_format_changed(self, index=None, format: str = ""):
        if format:
            project_settings.set("export_format", format)
            settings.set("export_format", format)
            return

    @action("category.select_category", description="选择当前使用的标注类别。\n- category_name：目标类别名称\n- 前提条件：category_name 必须存在于 categories 列表中\n- 选择后自动记录到最近使用列表\n- 影响新标注的默认类别（current_category）\n- 可通过 get_state 获取当前选中的类别", category="标注", params={"category_name": "str"}, scope="ui")
    def on_category_clicked(self, item=None, category_name: str = ""):
        if category_name:
            selected = category_name
        else:
            widget = self.category_list.itemWidget(item) if item else None
            selected = widget._name if widget else (item.text() if item else "")
        error = self.manual.select_category(selected)
        if error:
            return error
        self.status_label.setText(f"当前类别: {self.current_category}")

    def _touch_recent_category(self, category_name: str):
        self.manual.touch_recent_category(category_name)

    @action("category.switch_category", description="切换类别（Tab键）。\n- 有选中标注时：弹出类别选择对话框修改标注类别\n- 无选中标注时：按最近使用顺序循环切换当前工具类别\n- 前提条件：至少有一个类别可用\n- 类别只有 1 个时无法切换", category="标注", scope="ui")
    def _switch_category(self):
        if not self.categories:
            return "错误: 没有可用的类别"

        selected_indices = getattr(self.draw_area, 'batch_selected_indices', [])
        if not selected_indices:
            current_row = self.label_list.currentRow()
            if current_row >= 0 and current_row < len(self.draw_area.annotations):
                selected_indices = [current_row]
        if selected_indices:
            idx = selected_indices[0]
            ann = self.draw_area.annotations[idx]
            current_label = ann.get('label', '')
            dialog = CategorySelectionDialog(self.categories, current_label, self, recent_order=self._recent_categories)
            if dialog.exec_():
                new_cat = dialog.get_selected_category()
                if new_cat and new_cat != current_label:
                    new_annotations, err = self.manual.update_annotation_label(
                        idx, new_cat, self.draw_area.annotations, self.categories
                    )
                    if err:
                        return
                    self.manual.set_current_annotations(new_annotations)
                    self._touch_recent_category(new_cat)
                    # 本函数由类别按钮/双击信号驱动，重建必须延后（D16）
                    self.request_list_rebuild(self.refresh_label_list, from_signal=True)
                    self.draw_area.update()
                    self.save_current(silent=True)
                    InfoBar.success("类别已修改", f"类别已切换为 '{new_cat}'", duration=1500, parent=self)
            return

        next_cat, error = self.manual.cycle_category()
        if error:
            return error
        if next_cat:
            self.on_category_clicked(category_name=next_cat)
            self._select_category_in_list(next_cat)

    @action("category.add_category", description='添加新的标注类别。\n- category_name：类别名称，不能与已有类别重名\n- color：十六进制颜色值，如 "#ff0000"（红色）\n- 添加后自动选中该类别', category="标注", params={"category_name": "str", "color": "str"}, scope="ui")
    @Slot()
    def add_category(self, category_name: str = "", color: str = ""):
        if category_name:
            new_cats, err = self.manual.add_category(category_name, color or None)
            if err:
                return err
            self._refresh_category_state()
            self._select_category_in_list(category_name)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("添加类别")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)
        edit = LineEdit()
        edit.setPlaceholderText("输入类别名称")
        layout.addWidget(edit)
        btn = PrimaryPushButton("确定")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        if dialog.exec_():
            name = edit.text().strip()
            if name and name not in self.categories:
                new_cats, err = self.manual.add_category(name, None)
                if err:
                    return err
                self._refresh_category_state()
                self._select_category_in_list(name)
                from core.common.action_recorder import ActionRecorder
                recorder = ActionRecorder.instance()
                if recorder.is_recording:
                    recorder.record_step(
                        action_name="add_category",
                        params={},
                        description="添加类别",
                        expected_result="弹出类别输入对话框"
                    )

    def on_category_double_clicked(self, item):
        widget = self.category_list.itemWidget(item)
        name = widget._name if widget else (item.text() if item else "")
        self._rename_category_ui(old_name=name)

    def show_category_context_menu(self, pos):
        item = self.category_list.itemAt(pos)
        if not item: return

        name = item.text()
        menu = RoundMenu(parent=self)

        rename_action = Action(FIF.EDIT, "重命名")
        rename_action.triggered.connect(lambda: self._rename_category_ui(old_name=name))
        menu.addAction(rename_action)

        delete_action = Action(FIF.DELETE, "删除")
        delete_action.triggered.connect(lambda: self.remove_category(name))
        menu.addAction(delete_action)

        if name:
            send_menu = RoundMenu("添加到对话", parent=self)
            act_cat = Action(FIF.TAG, "引用类别")
            act_cat.triggered.connect(
                lambda: self._send_reference(f"@类别:{name}"))
            send_menu.addAction(act_cat)
            menu.addMenu(send_menu)

        menu.exec(self.category_list.mapToGlobal(pos))

    @Slot()
    @action("category.import_categories", description='从 JSON 文件导入标注类别。\n- file_path：JSON 文件路径（可选），留空时弹出文件选择对话框\n- JSON 格式：{"类别名": [R, G, B]}，颜色值为 0-255 整数\n- 导入后完全替换当前类别列表\n- 自动选中第一个导入的类别\n- 导入后自动保存类别配置', category="标注", params={"file_path": "str"}, scope="ui")
    def import_categories(self, file_path: str = ""):
        if file_path:
            if not os.path.isfile(file_path):
                return "错误: 导入文件不存在"
            path = file_path
        else:
            path, _ = QFileDialog.getOpenFileName(self, "导入类别", "", "JSON Files (*.json)")
        if path:
            cats, err = self.manual.import_categories(path)
            if err:
                if settings.DEBUG:
                    raise Exception(err)
                InfoBar.error("导入失败", err, parent=self)
                return f"错误: 导入失败: {err}"
            self._refresh_category_state()
            InfoBar.success("导入成功", f"已导入 {len(cats)} 个类别", parent=self)

    @action("category.export_categories", description="将当前标注类别列表导出为 JSON 文件。\n- file_path: 导出目标路径（可选，留空则弹出保存对话框）\n- 导出文件包含类别名称和颜色信息\n- 导出的文件可用 import_categories 导入", category="标注", params={"file_path": "str"}, scope="ui")
    @Slot()
    def export_categories(self, file_path: str = ""):
        if file_path:
            path = file_path
        else:
            path, _ = QFileDialog.getSaveFileName(self, "导出类别", "categories.json", "JSON Files (*.json)")
        if not path:
            return "错误: 未选择导出路径"
        err = self.manual.export_categories(path)
        if err:
            if settings.DEBUG:
                raise Exception(err)
            InfoBar.error("导出失败", err, parent=self)
            return f"错误: 导出失败: {err}"
        InfoBar.success("导出成功", f"已保存至 {path}", parent=self)

    # ============================================================
    # Secondary annotation methods
    # ============================================================

    def on_load_secondary_annotations(self, directory: str = None):
        if not self.current_image_path:
            InfoBar.warning("提示", "请先加载图像", parent=self)
            return "错误: 当前未加载图像"

        if directory is not None:
            if not directory or not os.path.isdir(directory):
                return "错误: 副标注目录不存在或无效"
            dir_path = directory
        else:
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择副标注文件夹",
                os.path.dirname(self.current_image_path)
            )
            if not dir_path:
                return

        self.manual.secondary_dir = dir_path
        self.manual.secondary_loaded = True

        self._load_secondary_for_current_image()

        InfoBar.success("设置成功", f"副标注文件夹已设置: {dir_path}", parent=self)

    def _load_secondary_for_current_image(self):
        ed = self.manual
        if not self.current_image_path or not ed.secondary_loaded or not ed.secondary_dir:
            return

        result = ed.load_secondary_annotations(
            ed.secondary_dir, self.current_image_path, self.draw_area.annotations
        )

        # 标注列表/画布刷新由 annotation:annotations_changed 事件处理器完成，此处仅更新状态栏
        status = result.get("status")
        if status == "success":
            matched_count = result.get("matched_count", 0)
            unmatched_secondary = result.get("unmatched", 0)
            self.status_label.setText(f"副标注: {len(ed.secondary_annotations)} 个 (匹配: {matched_count}, 未匹配: {unmatched_secondary})")
        elif status == "empty":
            self.status_label.setText(f"副标注: {result.get('message', '')}")
        else:
            print(f"[AnnotationInterface] Error loading secondary annotations: {result.get('message')}")
            traceback.print_exc()
            self.status_label.setText(f"副标注加载失败: {result.get('message')}")

    @action("edit.clear_secondary_annotations", description="清除当前加载的副标注数据。\n- clear_dir：是否同时清除副标注文件夹设置（默认 False）\n- clear_dir=True 时同时重置 secondary_loaded 和 secondary_dir\n- 不影响主标注数据\n- 清除后可通过 load_secondary_annotations 重新加载", category="标注", params={"clear_dir": "bool"})
    def clear_secondary_annotations(self, clear_dir=False):
        err = self.manual.clear_secondary_annotations(clear_dir)
        if err:
            return err
        # 标注列表/画布刷新由 annotation:annotations_changed 事件处理器完成

    @action("edit.accept_secondary_annotation", description="接受副标注，将副标注数据写入主标注。\n- index：主标注索引（从 0 开始）\n- 前提条件：已加载副标注，且该主标注有对应的副标注映射\n- 操作后自动保存 undo 状态，可通过 undo 撤销\n- 接受后副标注映射被清除，不再显示 [主+副] 状态", category="标注", params={"index": "int"})
    def on_accept_secondary_annotation(self, primary_idx, index: int = -1):
        if index >= 0:
            primary_idx = index

        new_annotations, err = self.manual.accept_secondary_annotation(
            primary_idx, self.draw_area.annotations, self.current_image_path
        )
        if err:
            return err
        # service 已写回共享数据并发布 annotations_changed，标注列表/画布刷新交给事件处理器

        InfoBar.success("接收成功", f"已将副标注写入主标注 #{primary_idx + 1}", parent=self)

    # ============================================================
    # batch.set_split method
    # ============================================================

    @action("batch.set_split", description="设置当前图片的数据集分割（train/val/test）。\n- split: 分割类型，可选值: train（训练集）, val（验证集）, test（测试集）\n- 在 action 调用时通过 split 参数指定\n- 设置后自动保存到 dataset_split.json\n- 调用前需先加载图片", category="标注", params={"split": "str"}, scope="ui")
    def on_split_changed(self, index=None, split: str = ""):
        if split is None and index is None:
            return "错误: 未指定分割类型"
        if split:
            resolved, error = self.manual.apply_split(split, self.current_image_path)
            if error:
                return error
            return f"当前图片已标记为: {resolved}"
        if index is None:
            return "错误: 未指定分割类型"
        splits = ["train", "val", "test"]
        if 0 <= index < len(splits):
            resolved, error = self.manual.apply_split(splits[index], self.current_image_path)
            if error:
                return error
            return f"当前图片已标记为: {resolved}"
        return "错误: 分割索引无效"

    # ============================================================
    # interact.set_task_mode method
    # ============================================================

    @action("interact.set_task_mode", description="设置标注任务类型。\n- mode 可选值：seg(分割任务，多边形标注), det(检测任务，矩形框标注), obb(旋转框任务)\n- 影响自动标注的推理方式和类别管理的行为", category="标注",
            params={"mode": {"type": "str", "enum": ["seg", "det", "obb"], "description": "任务模式: seg=分割, det=检测, obb=旋转框"}}, scope="ui")
    def on_task_changed(self, mode: str = ""):
        old_mode, error = self.manual.apply_task_mode(mode)
        if error:
            return error
        if old_mode is not None and old_mode != mode:
            self.draw_area.task_mode = mode

    def convert_all_with_ai(self, image_paths, source_mode='det', target_mode='seg'):
        if not image_paths:
            return

        progress = None
        try:
            self.setCursor(Qt.WaitCursor)

            title = "AI 全量转换"
            if source_mode == 'obb' and target_mode == 'det':
                title = "旋转框 -> 水平框 (AI 辅助)"

            progress = QProgressDialog(f"正在进行 {title}...", "取消", 0, len(image_paths), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setWindowTitle("批量处理中")
            progress.show()

            # 把"取消"按钮接到真正的取消逻辑（审计 D9 余项）：原先它没连任何东西，
            # 按下去毫无作用。这里用一个 threading.Event 传递取消意图，服务在
            # **图片边界**检查并提前收尾 —— 因此不会打断正在写盘的那张图，
            # 已落盘数据始终完整。
            cancel_event = threading.Event()

            def _request_cancel():
                cancel_event.set()
                progress.setLabelText("正在取消…（将在当前图片处理完后停止）")

            progress.canceled.connect(_request_cancel)

            def progress_callback(processed, total):
                progress.setValue(processed)
                if cancel_event.is_set():
                    progress.setLabelText("正在取消…（将在当前图片处理完后停止）")
                else:
                    progress.setLabelText(
                        f"正在处理 ({processed+1}/{total}): "
                        f"{os.path.basename(image_paths[min(processed, len(image_paths)-1)])}")
                # 只重绘本控件，不派发事件（见 update_save_path 的说明），
                # 避免在批量进行中重入事件循环。
                progress.repaint()

            result = self.auto.convert_all_with_ai(
                image_paths, source_mode, target_mode,
                progress_callback=progress_callback,
                cancel_event=cancel_event
            )

            if not cancel_event.is_set():
                progress.setValue(len(image_paths))

        except Exception as e:
            if settings.DEBUG:
                raise
            traceback.print_exc()
            InfoBar.error("全量转换出错", f"处理过程中出现错误: {str(e)}", parent=self)
            return
        finally:
            self.unsetCursor()
            if progress is not None:
                progress.close()
                # 同上：close() 已足够，不再 processEvents()

        self.load_image_annotations()

        # 取消路径必须给出明确反馈，且汇报的是**实际处理数**而不是计划总数
        if result.get("cancelled"):
            done = result.get("processed_count", 0)
            InfoBar.warning(
                "已取消全量转换",
                f"已在第 {done}/{result.get('total', 0)} 张停止（已处理的标注均已完整落盘）",
                parent=self)
            return

        empty_results = result.get("empty_results_count", 0)
        images_with_empty = result.get("images_with_empty_results", 0)
        converted = result.get("converted_count", 0)
        total = result.get("total", 0)

        if empty_results > 0:
            msg = f"已成功处理 {total} 张图片的标注（其中 {converted} 个目标 AI 转换成功）。\n\n"
            msg += f"注意：有 {images_with_empty} 张图片（共 {empty_results} 个目标）AI 转换未生成精确掩码，已自动回退为矩形多边形。"
            MessageBox("转换完成 (包含回退)", msg, self).exec()
        else:
            InfoBar.success("全量转换完成", f"已成功处理 {total} 张图片的标注", parent=self)

    # ============================================================
    # Review / filter methods
    # ============================================================

    @action("review.search_images", read_only=True,
            description="按关键词筛选图片列表，只显示名称匹配的图片。\n- keyword: 图片名称关键词，支持 * 通配符（如 '*.png'）和子串匹配（如 '101' 匹配 'img_101.jpg'）；传空字符串清除名称筛选\n- 纯数字会跳转到对应序号的图片（如 '5' 跳转到第 5 张）\n- 支持在关键词中内联引用筛选：'@类别:xxx' 按类别筛选、'@大小:小目标/中目标/大目标' 按预设大小筛选、'@大小:面积min-max' 与 '@大小:宽高minW-maxWxminH-maxH' 自定义范围、'@状态:已标注/@状态:未标注' 按标注状态筛选；不带 token 时清除对应筛选\n- 筛选后自动更新文件列表显示，跳转到第一张可见图片\n- 不传参数时读取搜索框当前文本\n- 同时应用已设置的类别/尺寸筛选，可用 review.clear_filters 清除",
            category="标注", params={"keyword": "str"}, scope="ui")
    @Slot(str)
    def search_images(self, keyword: str = None):
        # 任何**显式**搜索都作废在途的后台搜索（递增代次），避免旧结果回来覆盖新状态
        self._search_debounce.stop()
        self._search_generation += 1
        if keyword is None:
            search_text = self.search_box.text().strip()
        else:
            search_text = str(keyword).strip()
            if self.search_box.text().strip() != search_text:
                self.search_box.blockSignals(True)
                self.search_box.setText(search_text)
                self.search_box.blockSignals(False)

        # 解析 @类别:/@大小: token 并与筛选器对账，剩余文本作为名称关键词
        keyword_text = self._apply_search_tokens(search_text)

        if keyword_text.isdigit():
            target_idx = int(keyword_text) - 1
            if 0 <= target_idx < self.file_list.count():
                self.load_image(target_idx)
            return

        visibility_result = self.manual.search_images(
            keyword_text,
            output_dir=self.output_dir,
            is_non_project_mode=self._is_non_project_mode,
            non_project_format=self._non_project_format
        )
        self._apply_file_visibility(visibility_result["files"])

    def _on_search_text_changed(self, text):
        """搜索框每次击键：只**置脏 + 去抖**，真正的扫描在后台线程进行（审计 D15）。

        原先 `textChanged` 直接连到 `search_images`，于是每次击键都在 GUI 线程做一次
        全量扫描；一旦启用了类别/尺寸/标注状态筛选，扫描会**逐图读盘解析标注**
        （`get_cached_annotations`）—— 目录大一点就是秒级到分钟级的冻结。
        """
        stripped = (text or "").strip()
        # 纯数字是"跳转到第 N 张"，属于导航而非扫描，仍走同步路径保持原有手感
        if stripped.isdigit():
            self.search_images(stripped)
            return
        self._search_debounce.start(self.SEARCH_DEBOUNCE_MS)

    def _on_search_debounce_timeout(self):
        """去抖到点：把当前搜索文本交给后台扫描。"""
        search_text = self.search_box.text().strip()
        keyword_text = self._apply_search_tokens(search_text)
        if keyword_text.isdigit():
            return
        self._request_background_search(keyword_text)

    def _request_background_search(self, keyword_text):
        """提交一次**后台**搜索：GUI 线程只取快照，扫描在工作线程。

        代次守卫 + 最新优先：正在跑时新请求只覆盖待发请求；结果代次不匹配则丢弃。
        """
        self._search_generation += 1
        request = {
            "generation": self._search_generation,
            "keyword": keyword_text,
            "output_dir": self.output_dir,
            "is_non_project_mode": self._is_non_project_mode,
            "non_project_format": self._non_project_format,
            "snapshot": self.manual.search_snapshot(),
        }
        if self._search_runner.busy():
            self._pending_search_request = request      # 最新优先
            return
        self._start_search(request)

    def _start_search(self, request):
        started = self._search_runner.start(
            lambda: self.manual.search_images(
                request["keyword"],
                output_dir=request["output_dir"],
                is_non_project_mode=request["is_non_project_mode"],
                non_project_format=request["non_project_format"],
                snapshot=request["snapshot"],
                apply_state=False,          # 状态由 GUI 线程在提交结果时回写
            ),
            on_result=lambda result: self._on_search_done(request, result),
            on_error=lambda message: print(f"[AnnotationInterface] 后台搜索失败: {message}"),
        )
        if not started:
            self._pending_search_request = request

    def _on_search_done(self, request, result):
        """后台搜索完成（GUI 线程）：代次仍匹配才应用。"""
        try:
            if request["generation"] != self._search_generation:
                print("[AnnotationInterface] 丢弃过期搜索结果（代次 %d != %d）"
                      % (request["generation"], self._search_generation))
                return
            files = (result or {}).get("files") or []
            self.manual.file_visibility = files
            self._apply_file_visibility(files)
        except Exception:
            traceback.print_exc()
        finally:
            self._drain_pending_search()

    def _drain_pending_search(self):
        pending = self._pending_search_request
        self._pending_search_request = None
        if pending is not None:
            self._start_search(pending)

    # ---- @ 引用（搜索框内引用类别/大小进行筛选） ----

    _TOKEN_RE = re.compile(r'@(类别|大小|状态|划分):([^\s@]+)')
    _TOKEN_TAIL_RE = re.compile(r'@([^\s@]*)$')
    _CUSTOM_TOKEN = "@大小:自定义范围…"

    def _setup_search_reference(self):
        """为搜索框配置 @ 引用候选弹层：输入 @ 后在搜索框下方弹出类别/大小候选

        键盘事件过滤器在 init_ui 末尾统一安装（label_list 一并），避免
        过滤器在控件尚未创建完成时被触发。
        """
        self._ref_popup = _ReferencePopup()
        self._ref_popup.activated.connect(self._on_reference_clicked)
        self._ref_popup.applyRequested.connect(self._on_reference_apply)
        self.search_box.textChanged.connect(self._update_reference_popup)
        # 多选追加填充期间临时抑制弹层重建，避免 setText 触发失焦/闪烁
        self._suppress_popup_refresh = False

    def _current_reference_token(self, text):
        """返回文本末尾的 @token（含未完成输入），没有则返回 None"""
        match = self._TOKEN_TAIL_RE.search(text)
        return match.group(0) if match else None

    def _reference_entries(self, token):
        """根据 @token 片段生成候选列表（类别 / 大小预设 / 自定义范围）

        head 支持前缀匹配：输入 '@'、'@大'、'@大小' 均能给出大小候选。
        """
        head, _, value = token[1:].partition(":")
        value = value.strip()
        entries = []
        if not head or "类别".startswith(head):
            entries += [f"@类别:{c}" for c in sorted(self.categories.keys())
                        if c.lower().startswith(value.lower())]
        if not head or "大小".startswith(head):
            entries += [f"@大小:{p}" for p in ("小目标", "中目标", "大目标")
                        if p.startswith(value)]
            if not value or "自定义范围".startswith(value):
                entries.append(self._CUSTOM_TOKEN)
        if not head or "状态".startswith(head):
            entries += [f"@状态:{p}" for p in ("已标注", "未标注") if p.startswith(value)]
        if not head or "划分".startswith(head):
            entries += [f"@划分:{p}" for p in ("train", "val", "test", "未划分")
                        if p.startswith(value)]
        return entries

    def _reference_all_entries(self):
        """全部候选（搜索框获得焦点但尚未输入 @ 时展示）"""
        entries = [f"@类别:{c}" for c in sorted(self.categories.keys())]
        entries += ["@大小:小目标", "@大小:中目标", "@大小:大目标", self._CUSTOM_TOKEN]
        entries += ["@状态:已标注", "@状态:未标注"]
        entries += ["@划分:train", "@划分:val", "@划分:test", "@划分:未划分"]
        return entries

    def _show_reference_popup(self, entries):
        """定位并显示候选弹层"""
        popup = self._ref_popup
        target_pos = self.search_box.mapToGlobal(QPoint(0, self.search_box.height() + 2))
        target_w = max(240, self.search_box.width())
        target_h = min(len(entries), 6) * 30 + 52
        # 候选与几何都未变化时跳过重建(clear/addItems/show 每键重建会造成闪烁)
        if (popup.isVisible() and popup.pos() == target_pos
                and popup.width() == target_w and popup.height() == target_h
                and popup.count() == len(entries)
                and all(popup.item(i).text() == entries[i] for i in range(len(entries)))):
            return
        popup.clear()
        popup.addItems(entries)
        popup.setCurrentRow(0)
        # 定位到搜索框正下方，宽度对齐搜索框
        popup.move(target_pos)
        popup.setFixedWidth(target_w)
        # 高度 = 列表条目(≤6 行) + 底部「完成筛选」按钮与留白
        popup.setFixedHeight(target_h)
        popup.show()

    def _update_reference_popup(self, text):
        """更新候选弹层：@token 片段过滤候选；无 token 且搜索框持有焦点时展示全部候选"""
        # 多选追加填充期间不做任何重建/hide/show，保证不失焦、不闪烁
        if getattr(self, "_suppress_popup_refresh", False):
            return
        token = self._current_reference_token(str(text))
        if token is not None:
            entries = self._reference_entries(token)
        elif self.search_box.hasFocus():
            entries = self._reference_all_entries()
        else:
            entries = None
        if not entries:
            self._ref_popup.hide()
            return
        self._show_reference_popup(entries)

    def _on_reference_clicked(self, completion):
        """候选被选中（鼠标点击 / 键盘 Enter、Tab 触发）。

        - 末尾存在未完成的 @token（用户正在输入 @...）→ 替换该 token（单条件语义，向后兼容）
        - 末尾无 token（多选态）→ 追加 @token，弹层保持打开继续多选
        """
        completion = str(completion)
        token = self._current_reference_token(self.search_box.text())
        if token is not None:
            self._on_reference_selected(completion)
        else:
            self._insert_reference(completion)

    def _on_reference_selected(self, completion):
        """替换末尾未完成的 token（用户输入 @token 时选中候选）；自定义范围弹出输入对话框"""
        completion = str(completion)
        text = self.search_box.text()
        token = self._current_reference_token(text)
        if token is None:
            return
        base = text[: len(text) - len(token)].rstrip()
        if completion == self._CUSTOM_TOKEN:
            self.search_box.setText(base)
            self._open_custom_size_reference()
            return
        new_text = f"{base} {completion} " if base else f"{completion} "
        self.search_box.setText(new_text)

    def _on_reference_apply(self):
        """点击「完成筛选」：收起弹层并做一次最终搜索（文本已随每次追加实时生效）"""
        self._ref_popup.hide()
        self.search_images()

    def _insert_reference(self, token):
        """在搜索框末尾追加一条完整的 @token（多选累加）。

        - 先剔除末尾未完成的半截 @token，避免拼成 '@xxx@'
        - 已存在相同 token 时跳过，避免重复条件
        - 追加期间临时抑制弹层重建（_suppress_popup_refresh），使 setText 不失焦、不闪烁
        """
        text = self.search_box.text()
        # 剔除末尾未完成 token
        tail = self._current_reference_token(text)
        if tail:
            text = text[: len(text) - len(tail)].rstrip()
        else:
            text = text.rstrip()
        # 去重：整条 token 已存在则跳过
        pattern = re.compile(r"(^|\s)" + re.escape(token) + r"(\s|$)")
        if pattern.search(text):
            return
        new_text = f"{text} {token} " if text else f"{token} "
        self._suppress_popup_refresh = True
        try:
            self.search_box.setText(new_text)
        finally:
            self._suppress_popup_refresh = False

    def _open_custom_size_reference(self):
        """弹出自定义范围对话框，确认后生成 @大小 token"""
        dialog = CustomSizeFilterDialog(self)
        if dialog.exec_() == QDialog.DialogCode.Accepted and dialog.result_token:
            self._insert_reference(dialog.result_token)

    def _apply_search_tokens(self, search_text):
        """解析 @类别:/@大小: token 并与筛选器对账，返回剔除 token 后的名称关键词。

        - 类别：取最后一个 @类别 token（未知类别忽略）；无 token 时清除类别筛选
        - 大小：支持 小目标/中目标/大目标 预设、@大小:面积min-max、
          @大小:宽高minW-maxWxminH-maxH；无 token（或 token 尚未输完/无法解析）时
          才按"全部"对账，避免输入过程中反复清除筛选
        搜索框是筛选的唯一 UI 来源；agent 仍可直接调用 review.set_filter。
        """
        manual = self.manual
        category = None
        has_category = False
        size_preset = None
        size_area = None
        size_wh = None
        annotated_state = None
        has_annotated_token = False
        split_state = None
        has_split_token = False
        for typ, value in self._TOKEN_RE.findall(search_text):
            if typ == "类别":
                if value in self.categories:
                    category, has_category = value, True
                continue
            if typ == "状态":
                if value in ("已标注", "未标注"):
                    annotated_state = (value == "已标注")
                    has_annotated_token = True
                continue
            if typ == "划分":
                if value == "未划分":
                    split_state = ""  # 内部用 "" 表示「未划分」
                    has_split_token = True
                elif value in ("train", "val", "test"):
                    split_state = value
                    has_split_token = True
                continue
            if value == "全部":
                size_preset = 0
            elif value.startswith("小"):
                size_preset = 1
            elif value.startswith("中"):
                size_preset = 2
            elif value.startswith("大"):
                size_preset = 3
            else:
                match = re.fullmatch(r"面积(\d+)-(\d+)", value)
                if match:
                    size_area = (int(match.group(1)), int(match.group(2)))
                    continue
                match = re.fullmatch(r"宽高(\d+)-(\d+)x(\d+)-(\d+)", value)
                if match:
                    size_wh = tuple(int(g) for g in match.groups())

        # 类别筛选对账（无 token 但存在遗留筛选时清除）
        if has_category or manual.filter_category is not None:
            manual.set_filter(mode="category", category=category if has_category else None)

        # 大小筛选对账（面积/预设/宽高互斥，未指定的维度清除）
        has_size_token = bool(re.search(r"@大小:[^\s@]+", search_text))
        if not has_size_token or size_preset is not None or size_area is not None:
            if size_area is not None:
                manual.set_filter(mode="custom_size",
                                  min_size=size_area[0], max_size=size_area[1])
            elif size_preset is not None:
                manual.set_filter(mode="preset", index=size_preset)
            else:
                manual.set_filter(mode="preset", index=0)
        if size_wh is not None:
            manual.set_filter(mode="custom_wh",
                              min_width=size_wh[0], max_width=size_wh[1],
                              min_height=size_wh[2], max_height=size_wh[3])
        elif not has_size_token:
            manual.set_filter(mode="custom_wh",
                              min_width=0, max_width=0, min_height=0, max_height=0)

        # 标注状态筛选对账（无 token 但存在遗留状态筛选时清除）
        if has_annotated_token or manual.filter_annotated is not None:
            manual.set_filter(mode="annotated",
                              annotated=annotated_state if has_annotated_token else None)

        # 数据集划分筛选对账（无 token 但存在遗留划分筛选时清除）
        if has_split_token or manual.filter_split is not None:
            manual.set_filter(mode="split",
                              split=split_state if has_split_token else None)

        keyword_text = self._TOKEN_RE.sub(" ", search_text)
        tail = self._current_reference_token(search_text)
        if tail:
            keyword_text = keyword_text.replace(tail, " ")
        return " ".join(keyword_text.split())

    def _apply_file_visibility(self, visibility):
        """将可见性列表应用到文件列表，当前项被隐藏时跳转到第一张可见图片"""
        first_visible_idx = -1
        current_item_still_visible = False
        current_idx = self.file_list.currentRow()

        for i, (file_path, hidden) in enumerate(visibility):
            if i >= self.file_list.count():
                break
            item = self.file_list.item(i)
            item.setHidden(hidden)

            if not hidden:
                if first_visible_idx == -1:
                    first_visible_idx = i
                if i == current_idx:
                    current_item_still_visible = True

        if not current_item_still_visible and first_visible_idx != -1:
            self.load_image(first_visible_idx)

    def _refresh_file_visibility(self, force=False):
        """根据当前筛选/搜索条件重算文件可见性并应用到文件列表。

        当存在生效的筛选条件（类别/面积/宽高）或已有搜索可见性时，
        基于当前筛选状态重新调用 manual.search_images 计算可见性并应用，
        确保筛选在事件自动刷新 / refresh_view 后真正生效。
        force=True 时跳过早退（供筛选动作在清除筛选后强制刷新）。
        """
        manual = self.manual
        has_filter = bool(manual.filter_category or manual.filter_size_range or
                          manual.filter_width_range or manual.filter_height_range or
                          manual.filter_annotated is not None or manual.filter_split is not None)
        if not force and not has_filter and not manual.file_visibility:
            return
        search_text = self.search_box.text().strip()
        search_text = self._TOKEN_RE.sub(" ", search_text)  # 仅剔除 token，不回写筛选器
        search_text = " ".join(search_text.split())
        if search_text.isdigit():
            search_text = ""
        # 筛选条件变化后的重算同样是"逐图读盘"级别的开销 -> 走后台（审计 D15）
        self._request_background_search(search_text)

    def _get_cached_annotations(self, image_path):
        return self.manual.get_cached_annotations(
            image_path, self.output_dir,
            self._is_non_project_mode, self._non_project_format
        )

    @action("review.set_filter",
            description="统一筛选入口（mode 参数驱动）。\n"
                       "- mode=category: 按类别筛选，category 传类别名（传空字符串清除）\n"
                       "- mode=preset: 预设尺寸筛选，index 0=全部 1=小目标(<32²) 2=中目标(32²-96²) 3=大目标(>96²) 4=自定义面积 5=自定义宽高\n"
                       "- mode=custom_size: 自定义面积筛选（像素²），min_size/max_size 传范围，-1 表示从 UI 输入读取\n"
                       "- mode=custom_wh: 自定义宽高筛选（像素），min_width/max_width/min_height/max_height 传范围，-1 表示从 UI 输入读取\n"
                       "- mode=annotated: 按标注状态筛选，annotated=true 仅显示已标注，annotated=false 仅显示未标注，annotated=None/-1 清除\n"
                       "- 筛选后自动更新文件列表显示\n"
                       "- 用 review.clear_filters 清除所有筛选",
            category="标注", scope="ui",
            params={"mode": "str", "category": "str", "index": "int",
                    "min_size": "int", "max_size": "int",
                    "min_width": "int", "max_width": "int",
                    "min_height": "int", "max_height": "int",
                    "annotated": "bool"})
    def set_filter(self, mode="category", category=None, index=None,
                   min_size=-1, max_size=-1,
                   min_width=-1, max_width=-1, min_height=-1, max_height=-1,
                   annotated=None):
        """统一筛选入口 — 按 mode 分发并同步 UI 状态。"""
        if mode == "category":
            return self._apply_category_filter_ui(category, index)
        if mode == "preset":
            return self._apply_preset_filter_ui(index)
        if mode == "custom_size":
            return self._apply_custom_size_filter_ui(min_size, max_size)
        if mode == "custom_wh":
            return self._apply_custom_wh_filter_ui(
                min_width, max_width, min_height, max_height)
        if mode == "annotated":
            self.manual.set_filter(mode="annotated", annotated=annotated)
            self._refresh_file_visibility(force=True)
            return "已按标注状态筛选" if annotated is not None else "已清除标注状态筛选"
        return f"错误: 未知筛选模式 '{mode}'，可选: category/preset/custom_size/custom_wh/annotated"

    def _apply_category_filter_ui(self, category=None, index=None):
        """mode=category 的 UI 实现：index/category 二选一解析为类别并设置"""
        if category:
            categories = sorted(self.categories.keys())
            for i, cat in enumerate(categories):
                if cat == category:
                    index = i + 1
                    break
            else:
                InfoBar.warning("提示", f"未找到类别: {category}", parent=self)
                return "错误: 未找到类别: %s" % category
        if index is None:
            return "错误: 未指定类别"
        if index == -1 or index == 0:
            applied = self.manual.set_filter(mode="category", category=None)
        else:
            categories = sorted(self.categories.keys())
            if 0 <= index - 1 < len(categories):
                applied = self.manual.set_filter(mode="category", category=categories[index - 1])
            else:
                applied = None
        # 文件可见性刷新由 annotation:filters_changed 事件处理器（on_filters_changed）完成

    def _apply_preset_filter_ui(self, index=None):
        """mode=preset 的 UI 实现：解析预设索引并应用"""
        if index is None:
            return "错误: 未指定尺寸筛选类型"
        _, is_custom, is_wh = self.manual.set_filter(mode="preset", index=index)
        if is_custom or is_wh:
            return
        # 文件可见性刷新由 annotation:filters_changed 事件处理器（on_filters_changed）完成

    def _apply_custom_size_filter_ui(self, min_size=-1, max_size=-1):
        """mode=custom_size 的 UI 实现：直接应用传入的面积范围"""
        if min_size < 0 or max_size < 0:
            return "错误: 请提供有效的 min_size/max_size（-1 表示未指定）"
        size_range, error = self.manual.set_filter(
            mode="custom_size", min_size=min_size, max_size=max_size)
        if error:
            InfoBar.warning("参数错误", error, parent=self)
            return error
        # 文件可见性刷新由 annotation:filters_changed 事件处理器（on_filters_changed）完成

    def _apply_custom_wh_filter_ui(self, min_width=-1, max_width=-1, min_height=-1, max_height=-1):
        """mode=custom_wh 的 UI 实现：直接应用传入的宽高范围"""
        if min_width < 0 or max_width < 0 or min_height < 0 or max_height < 0:
            return "错误: 请提供有效的 min/max 宽高（-1 表示未指定）"
        wh_range, error = self.manual.set_filter(
            mode="custom_wh",
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
        )
        if error:
            InfoBar.warning("参数错误", error, parent=self)
            return error
        # 文件可见性刷新由 annotation:filters_changed 事件处理器（on_filters_changed）完成

    @action("review.clear_filters",
            description="清除筛选条件。\n- target: 清除范围，可选值 all(全部)/category(类别)/size(尺寸面积宽高)/width(宽度)/height(高度)\n- 清除后自动更新文件列表显示",
            category="标注", params={"target": "str"}, scope="ui")
    def clear_filters(self, target="all"):
        """统一清除筛选入口。target 控制清除范围：all=全部, category=类别, size=尺寸面积宽高, width=宽度, height=高度。"""
        nav = self.manual
        if target in ("all", "category"):
            if nav.filter_category is not None:
                nav.set_filter(mode="category", category=None)
        if target in ("all", "size"):
            if nav.filter_size_range is not None or nav.filter_width_range is not None or nav.filter_height_range is not None:
                nav.set_filter(mode="preset", index=0)
                nav.set_filter(mode="custom_wh", min_width=0, max_width=0, min_height=0, max_height=0)
        if target in ("all", "width"):
            if nav.filter_width_range is not None:
                if nav.filter_height_range:
                    min_h, max_h = nav.filter_height_range
                    max_h = max_h or 0
                else:
                    min_h = max_h = 0
                nav.set_filter(mode="custom_wh",
                               min_width=0, max_width=0,
                               min_height=min_h, max_height=max_h)
        if target in ("all", "height"):
            if nav.filter_height_range is not None:
                if nav.filter_width_range:
                    min_w, max_w = nav.filter_width_range
                    max_w = max_w or 0
                else:
                    min_w = max_w = 0
                nav.set_filter(mode="custom_wh",
                               min_width=min_w, max_width=max_w,
                               min_height=0, max_height=0)
        # 各分支的 nav.set_filter 已发布 annotation:filters_changed，文件可见性刷新交由 on_filters_changed 完成
        return "已清除筛选条件"

    # ============================================================
    # Interaction methods (on_annotation_finished, sam, etc.)
    # ============================================================

    def on_annotation_finished(self, data):
        if not self.current_image_path:
            return

        label = self.current_category
        if not label:
            if self.categories:
                label = list(self.categories.keys())[0]
            else:
                label = "default"

        is_bbox = isinstance(data, list) and len(data) == 4 and all(isinstance(x, (int, float)) for x in data)

        new_ann = None

        self._save_undo_state()

        use_ai = (self.draw_area.mode == DrawingMode.SAM or self.draw_area.mode == DrawingMode.AI_RECT)

        if use_ai:
            bbox = None
            if is_bbox:
                bbox = data
            elif isinstance(data, list) and len(data) > 0:
                pts_np = np.array(data, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts_np)
                bbox = [x, y, bw, bh]

            if bbox:
                res = self.manual.predict_mask_with_bbox(
                    self.current_image_path, bbox,
                    model_type=self.interactive_model_type
                )

                if res['status'] == 'success':
                    res['label'] = label
                    if self.draw_area.mode == DrawingMode.OBB:
                        self.manual.convert_single_ann_to_obb(res)
                        res['shape_type'] = 'rotation'
                    else:
                        res['shape_type'] = 'rectangle' if not res.get('polygons') else 'polygon'
                    new_ann = res
                    self.manual.add_annotation(res)
                else:
                    pass

        if new_ann is None:
            if is_bbox:
                ann = self.manual.build_annotation_from_bbox(data, label, 'rectangle')
                if self.draw_area.mode == DrawingMode.OBB:
                    self.manual.convert_single_ann_to_obb(ann)
                new_ann = ann
                self.manual.add_annotation(ann)
            elif isinstance(data, list) and len(data) > 0:
                shape_type = 'rotation' if self.draw_area.mode == DrawingMode.OBB else 'polygon'
                ann = self.manual.build_annotation_from_polygon(
                    [data], label,
                    shape_type=shape_type,
                    image_path=self.current_image_path if shape_type == 'polygon' else None
                )
                if self.draw_area.mode == DrawingMode.OBB:
                    self.manual.convert_single_ann_to_obb(ann)
                new_ann = ann
                self.manual.add_annotation(ann)

        # 标注列表/画布刷新由 manual.add_annotation 发布的 annotation:annotations_changed 事件处理器完成，无需手动刷新

        if self.example_mode_enabled and new_ann:
            self.reference_annotation = new_ann
            self.find_similar_in_current_image()

        if self.auto_save_enabled:
            self.save_current(silent=True)

    def find_similar_in_current_image(self):
        if not self.reference_annotation or not self.current_image_path:
            return

        if self.model_manager.is_model_loading():
            InfoBar.warning("模型加载中", "模型正在后台加载，请稍候...", parent=self)
            return

        self.status_label.setText("正在寻找相似目标...")

        refs = self.reference_annotation
        if not isinstance(refs, list):
            refs = [refs]

        res = self._predict_similar_in_image(
            self.current_image_path,
            refs,
            model_type=self.model_type
        )
        if res['status'] == 'success':
            filtered_annotations = self.manual.filter_duplicate_annotations(
                res['annotations'],
                self.draw_area.annotations
            )

            added_count = 0
            for ann in filtered_annotations:
                ann['label'] = refs[0].get('label', self.current_category or 'default')

                if self.draw_area.task_mode == 'obb':
                    self.manual.convert_single_ann_to_obb(ann)
                elif self.draw_area.task_mode == 'det':
                    self.manual.convert_single_ann_to_det(ann)

                self.manual.add_annotation(ann)
                added_count += 1

            self.refresh_label_list()
            self.status_label.setText(f"已自动找出 {added_count} 个相似目标")
        else:
            self.status_label.setText(f"未找到相似目标: {res.get('message', '')}")

    def _predict_mask_with_points(self, image_path, points_data, model_type=None, refine=None, use_roi=True, img_size=None):
        return self.manual.predict_mask_with_points(
            image_path, points_data, model_type=model_type,
            refine=refine, use_roi=use_roi, img_size=img_size
        )

    def _predict_similar_in_image(self, image_path, reference_annotations, rules=None, model_type=None, refine=None, use_roi=True, use_slice=True, img_size=None):
        return self.manual.predict_similar_in_image(
            image_path, reference_annotations, rules=rules,
            model_type=model_type, refine=refine, use_roi=use_roi,
            use_slice=use_slice, img_size=img_size
        )

    def on_sam_point_added(self, points, label_type):
        if not self.draw_area.ai_enabled or not self.current_image_path:
            return

        if self.model_manager.is_model_loading():
            InfoBar.warning("模型加载中", "模型正在后台加载，请稍候...", parent=self)
            if self.draw_area.sam_points:
                self.draw_area.sam_points.pop()
                self.draw_area.update()
            return

        res = self._predict_mask_with_points(
            self.current_image_path,
            points,
            model_type=self.interactive_model_type
        )

        if res.get('status') != 'success':
            self.refresh_label_list()
            return

        res['label'] = self.current_category

        if self.draw_area.task_mode == 'obb':
            self.manual.convert_single_ann_to_obb(res)
        elif self.draw_area.task_mode == 'det':
            self.manual.convert_single_ann_to_det(res)

        self._save_undo_state()

        if hasattr(self, '_sam_active') and self._sam_active and self.draw_area.annotations:
            self.manual.current_annotations[-1] = res
            self.draw_area.update()
        else:
            self.manual.add_annotation(res)
            self._sam_active = True

        self._last_sam_annotation = res
        self.refresh_label_list()

    # ============================================================
    # Update methods
    # ============================================================

    @staticmethod
    def _clear_list_widget(list_widget):
        """清空 QListWidget，并**销毁**通过 setItemWidget 挂上去的子控件。

        Qt 语义：`takeItem()` 只把 item 从列表里摘下来，**不会销毁** itemWidget
        —— 那些控件仍是 viewport 的子对象，会一直常驻。原先 label_list /
        category_list 都用 `while count(): takeItem(0)` 清空，于是每次重建都泄漏
        「每个 item 6 个控件」。实测（tests/annotation/test_label_list_leak.py，
        20 个标注）：单次 refresh_label_list 泄漏 120 个控件，重建 30 次后
        viewport 子控件从 120 涨到 3720 —— 而 refresh_label_list 每次切图都会
        被调用一次，界面因此逐步变慢直到卡死。

        其它列表（version_list / prompt_library_dialog / example_selection_dialog）
        用的是 `clear()`，Qt 会连同 indexWidget 一起销毁，所以不受影响。
        """
        while list_widget.count() > 0:
            item = list_widget.item(0)
            widget = list_widget.itemWidget(item)
            if widget is not None:
                list_widget.removeItemWidget(item)
                widget.setParent(None)
                widget.deleteLater()
            list_widget.takeItem(0)

    def refresh_label_list(self):
        # 重建标志：防止旧复选框销毁/新复选框创建时 stateChanged 信号干扰 hidden_indices
        self._is_rebuilding_labels = True
        try:
            self.label_list.blockSignals(True)
            self._clear_list_widget(self.label_list)
            self.label_list.blockSignals(False)

            for i, ann in enumerate(self.draw_area.annotations):
                label = ann.get('label', 'unknown')

                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, i)

                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 2, 5, 2)
                item_layout.setSpacing(4)

                visible_cb = CheckBox()
                visible_cb.setChecked(i not in self.draw_area.hidden_indices)
                visible_cb.setFixedSize(18, 18)
                visible_cb.setProperty("annotation_idx", i)
                visible_cb.stateChanged.connect(self.on_annotation_visibility_changed)
                item_layout.addWidget(visible_cb)

                color = self.categories.get(label, QColor(0, 255, 255))
                if not isinstance(color, QColor):
                    try:
                        if isinstance(color, str):
                            c = QColor(color)
                            color = c if c.isValid() else QColor(0, 255, 255)
                        elif hasattr(color, '__len__') and len(color) >= 3:
                            color = QColor(*[int(c) for c in color[:3]])
                        else:
                            color = QColor(0, 255, 255)
                    except Exception:
                        color = QColor(0, 255, 255)
                pixmap = QPixmap(12, 12)
                pixmap.fill(color)
                color_icon = QLabel()
                color_icon.setPixmap(pixmap)
                item_layout.addWidget(color_icon)

                info_label = CaptionLabel(f"{i+1}: {label}")
                item_layout.addWidget(info_label)
                item_layout.addStretch(1)

                refine_btn = TransparentToolButton(FIF.BRUSH)
                refine_btn.setFixedSize(22, 22)
                refine_btn.setToolTip("细化该标注 (C)")
                refine_btn.setProperty("annotation_idx", i)
                refine_btn.clicked.connect(lambda checked, idx=i: self.refine_single_annotation(idx))
                item_layout.addWidget(refine_btn)

                delete_btn = TransparentToolButton(FIF.DELETE)
                delete_btn.setFixedSize(22, 22)
                delete_btn.setToolTip("删除该标注")
                delete_btn.setProperty("annotation_idx", i)
                delete_btn.clicked.connect(lambda checked, idx=i: self.delete_annotation_by_index(idx))
                item_layout.addWidget(delete_btn)

                item.setSizeHint(item_widget.sizeHint())

                self.label_list.addItem(item)
                self.label_list.setItemWidget(item, item_widget)
        finally:
            self._is_rebuilding_labels = False

    @action("edit.save_current", description="保存当前图片的标注结果到磁盘。\n- silent: 是否静默保存（不显示提示，默认 False）\n- 非项目模式下自动识别格式并保存到 output_dir\n- 项目模式下固定使用 labelme 格式保存到 annotations/ 目录\n- 保存时同时更新 dataset_split.json 分割信息\n- 保存后刷新文件列表状态图标\n\nLabelMe JSON 格式说明：标注文件为 JSON 格式，包含 shapes 数组（每个 shape 有 label/points/bbox/shape_type）、imagePath/imageWidth/imageHeight 等字段。shape_type 可选 rectangle/polygon/obb。bbox 格式为 [x, y, width, height]，x,y 为左上角图像坐标。", category="标注", params={"silent": "bool"})
    def save_current(self, silent=False):
        if not self.current_image_path:
            return "错误: 当前没有打开任何图片"

        task_mode = self.draw_area.task_mode

        if silent:
            # 自动保存路径（全部调用方都不使用返回值）：**标注文件写入在后台**。
            # GUI 线程只做状态更新（切分记录 + 标注缓存），不再被
            # imread + json 写盘阻塞 —— 这是"拖一次鼠标写一次盘"的热路径。
            res = self.manual.schedule_save_current(
                self.draw_area.annotations, self.current_image_path, task_mode
            )
            if res.get("status") == "error":
                return f"错误: 保存失败: {res.get('message', '未知错误')}"
            if not self._is_non_project_mode:
                split_idx = self.split_combo.currentIndex()
                split = ["train", "val", "test"][split_idx]
                norm_path = os.path.normpath(self.current_image_path)
                self.manual.set_image_split(norm_path, split)
                self.save_dataset_splits()
            self.annotation_cache[self.current_image_path] = copy.deepcopy(self.draw_area.annotations)
            return "queued"

        # 显式保存（Ctrl+S / agent action）：调用方需要真实结果。
        # 先等同一路径的在途自动保存落盘，保证**写入顺序**（否则排队中的旧快照
        # 可能后落地，把这次显式保存覆盖掉）。
        self.manual.wait_for_pending_writes(self.current_image_path, 5000)

        res = self.manual.save_current(
            self.draw_area.annotations, self.current_image_path, task_mode
        )

        if not self._is_non_project_mode:
            split_idx = self.split_combo.currentIndex()
            split = ["train", "val", "test"][split_idx]
            norm_path = os.path.normpath(self.current_image_path)
            self.manual.set_image_split(norm_path, split)
            self.save_dataset_splits()

        if res['status'] == 'success':
            self.annotation_cache[self.current_image_path] = copy.deepcopy(self.draw_area.annotations)
            if not silent:
                mode_name = "分割" if task_mode == 'seg' else "检测" if task_mode == 'det' else "旋转检测"
                return f"保存成功: 当前 {mode_name} 标注已保存"
            else:
                return "success"
        else:
            return f"错误: 保存失败: {res.get('message', '未知错误')}"

    def _on_background_save_failed(self, path, message):
        """后台自动保存失败（**在写入线程上被调用**）：marshal 回 GUI 线程再提示。

        Qt 控件只能在 GUI 线程访问，所以这里不能直接操作 status_label / InfoBar。
        """
        print(f"[AnnotationInterface] 后台自动保存失败: {path}: {message}")

        def _show():
            try:
                self.status_label.setText(f"自动保存失败: {message}")
                InfoBar.error("自动保存失败",
                              f"{os.path.basename(path)}: {message}", parent=self)
            except Exception:
                traceback.print_exc()

        try:
            run_on_main(_show)
        except Exception:
            # 应用正在关闭时 dispatch 会被拒（这是设计好的闸门），此时只需记录日志
            print("[AnnotationInterface] 无法把保存失败提示投递到主线程（可能正在关闭）")

    def _save_current_ui(self):
        result = self.save_current(silent=False)
        if isinstance(result, str) and result.startswith("保存成功"):
            mode_name = result.split(": ", 1)[1] if ": " in result else ""
            InfoBar.success("保存成功", mode_name, parent=self)
            self.status_label.setText(f"已保存 ({self.draw_area.task_mode})")
        elif isinstance(result, str) and result.startswith("错误"):
            InfoBar.error("保存失败", result, parent=self)
            self.status_label.setText(f"保存失败")

    def _save_non_project_config(self):
        self.manual.save_non_project_config(self.draw_area.task_mode)

    def _load_non_project_config(self, dir_path):
        config = self.manual.load_non_project_config(dir_path)
        if not config:
            return False

        self.draw_area.task_mode = self.context.task_mode
        if self.context.categories:
            self._refresh_category_state(save=False)
        self.hard_sample_count_label.setText(str(len(self.context.hard_samples)))

        print(f"[AnnotationInterface] 非项目模式配置已加载 ({dir_path})")
        return True

    def _select_category_in_list(self, name: str):
        """在类别列表中选中指定名称的类别"""
        for i in range(self.category_list.count()):
            item = self.category_list.item(i)
            w = self.category_list.itemWidget(item)
            if w and w._name == name:
                self.category_list.setCurrentRow(i)
                break

    def refresh_category_list(self):
        self.category_list.blockSignals(True)
        self._clear_list_widget(self.category_list)
        self.category_list.blockSignals(False)
        recent = [c for c in self._recent_categories if c in self.categories]
        remaining = [c for c in self.categories if c not in recent]
        ordered_cats = recent + remaining
        for cat in ordered_cats:
            color = self.categories[cat]
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 30))
            self.category_list.addItem(item)
            widget = CategoryItemWidget(cat, color)
            widget.rename_requested.connect(self._rename_category_ui)
            widget.delete_requested.connect(lambda name: self.remove_category(name))
            self.category_list.setItemWidget(item, widget)
            if cat == self.current_category:
                self.category_list.setCurrentItem(item)

    @action("category.delete_category", description="删除指定的标注类别。\n- category_name：要删除的类别名称（可选），留空时弹出确认对话框\n- skip_confirm：是否跳过确认对话框（默认 False）\n- 删除类别后：从 categories 列表中移除\n- 自动移除当前图片中属于该类别的标注\n- 批量更新所有标注文件中该类别的标注\n- 如果删除的类别是 current_category，自动切换到第一个可用类别", category="标注", params={"category_name": "str", "skip_confirm": "bool"}, scope="ui")
    def remove_category(self, target=None, category_name: str = "", skip_confirm: bool = False):
        if category_name:
            cat_to_del = category_name
        elif target is None:
            if self.current_category:
                cat_to_del = self.current_category
            else:
                target = self.category_list.currentItem()
                if target:
                    w = self.category_list.itemWidget(target)
                    cat_to_del = w._name if w else target.text()
                else:
                    cat_to_del = None
        else:
            cat_to_del = target if isinstance(target, str) else target.text()

        if not cat_to_del:
            if self.category_list.count() > 0:
                InfoBar.warning("提示", "请先选择要删除的类别", parent=self)
            else:
                InfoBar.warning("提示", "没有可删除的类别", parent=self)
            return

        if not skip_confirm and not category_name:
            w = MessageBox("删除类别", f"确定要删除类别 '{cat_to_del}' 吗？\n删除后，当前图片中属于该类别的标注将被设为 'default'。", self)
            if not w.exec():
                return
            from core.common.action_recorder import ActionRecorder
            recorder = ActionRecorder.instance()
            if recorder.is_recording:
                recorder.record_step(
                    action_name="delete_category",
                    params={},
                    description="删除类别",
                    expected_result="弹出删除确认对话框"
                )

        categories, annotations, result = self.manual.remove_category(
            cat_to_del, self.draw_area.annotations
        )
        if result.get("error"):
            return result["error"]
        self.categories = categories
        self.manual.set_current_annotations(annotations)

        res = result.get("batch_status", {})
        if res.get('status') == 'success':
            count = res.get('count', 0)
            InfoBar.success("类别删除成功", f"已从 {count} 个文件中移除了类别 '{cat_to_del}' 并更新了索引", parent=self)
        elif res.get('status') == 'error':
            InfoBar.error("删除失败", res.get('message', '未知错误'), parent=self)

        if result["changed"]:
            # 标注列表/画布刷新由 save_current 发布的 annotation:annotations_changed 事件处理器完成
            self.save_current(silent=True)
        # 类别列表刷新由 annotation:categories_changed 事件处理器完成；service 已持久化，save=False 避免重复持久化
        self._refresh_category_state(update_list=True, update_colors=True, save=False)

        if self.current_category:
            project_settings.set("current_category", self.current_category)
            self._select_category_in_list(self.current_category)

    @action("category.rename_category", description="重命名或合并标注类别。\n- old_name：原类别名称\n- new_name：新类别名称\n- 如果 new_name 已存在，则合并：将 old_name 的所有标注改为 new_name 后删除 old_name\n- 如果 new_name 不存在，则直接重命名\n- 自动批量更新所有标注文件中的类别名称\n- 如果 old_name 是 current_category，自动更新", category="标注", params={"old_name": "str", "new_name": "str"}, scope="ui")
    def rename_category(self, old_name: str = "", new_name: str = ""):
        if not old_name or not new_name:
            return "错误: rename_category 需要 old_name 和 new_name 参数"
        categories, annotations, result = self.manual.rename_category(
            old_name, new_name, self.draw_area.annotations
        )
        if result.get("skipped"):
            return
        if result.get("error"):
            return result["error"]
        self.categories = categories
        self.manual.set_current_annotations(annotations)
        if result.get("changed"):
            # 标注列表/画布刷新由 save_current 发布的 annotation:annotations_changed 事件处理器完成
            self.save_current(silent=True)
        res = result.get("batch_status", {})
        if res.get('status') == 'error':
            InfoBar.error("更新失败", res.get('message', '未知错误'), parent=self)
        # 类别列表刷新由 annotation:categories_changed 事件处理器完成；service 已持久化，save=False 避免重复持久化
        self._refresh_category_state(update_list=True, update_colors=True, save=False)
        return True

    def _merge_categories_core(self, old_name, new_name):
        categories, annotations, result = self.manual.merge_categories(
            old_name, new_name, self.draw_area.annotations
        )
        if categories is not None:
            self.categories = categories
        if annotations is not None:
            self.manual.set_current_annotations(annotations)
        if result.get("changed"):
            # 标注列表/画布刷新由 save_current 持久化时发布的 annotation:annotations_changed 事件处理器完成
            # save_current 需保留：service 的 merge_categories 仅批量更新其他标注文件，当前文件标注需在此持久化
            self.save_current(silent=True)
        res = result.get("batch_status", {})
        if res.get('status') == 'error':
            InfoBar.error("更新失败", res.get('message', '未知错误'), parent=self)
        # 类别列表/颜色刷新由 annotation:categories_changed 事件处理器（on_categories_changed）完成；
        # service 已持久化类别，save=False 避免重复持久化
        self._refresh_category_state(update_list=True, update_colors=True, save=False)
        return True

    def _rename_category_ui(self, target=None, old_name: str = ""):
        if old_name and target is None:
            target = old_name
        if not target:
            return
        old_name = target if isinstance(target, str) else target.text()
        if old_name not in self.categories:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("重命名类别")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)
        edit = LineEdit()
        edit.setText(old_name)
        edit.setPlaceholderText("输入新类别名称")
        layout.addWidget(edit)
        btn = PrimaryPushButton("确定")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        if dialog.exec_():
            new_name = edit.text().strip()
            if not new_name or new_name == old_name:
                return

            from core.common.action_recorder import ActionRecorder
            recorder = ActionRecorder.instance()
            if recorder.is_recording:
                recorder.record_step(
                    action_name="rename_category",
                    params={"old_name": old_name},
                    description=f"重命名类别 '{old_name}'",
                    expected_result="弹出重命名对话框"
                )

            if new_name in self.categories:
                from qfluentwidgets import MessageBox
                content = f"类别 '{new_name}' 已存在。\n\n是否将 '{old_name}' 合并到 '{new_name}'？\n\n合并后，所有 '{old_name}' 的标注将变为 '{new_name}'，且 '{old_name}' 将被删除。"
                msg_box = MessageBox('类别已存在', content, self)
                msg_box.yesButton.setText('合并')
                msg_box.cancelButton.setText('取消')

                if not msg_box.exec():
                    return

                self._merge_categories_core(old_name, new_name)
                InfoBar.success("合并成功", f"已将 '{old_name}' 合并到 '{new_name}'", parent=self)
                return

            categories, annotations, result = self.manual.rename_category(
                old_name, new_name, self.draw_area.annotations
            )
            if result.get("skipped"):
                return
            if result.get("error"):
                InfoBar.warning("重命名失败", result["error"], parent=self)
                return
            self.categories = categories
            self.manual.set_current_annotations(annotations)
            if result.get("changed"):
                # 标注列表/画布刷新由 save_current 持久化时发布的 annotation:annotations_changed 事件处理器完成
                # save_current 需保留：service 的 rename_category 仅批量更新其他标注文件，当前文件标注需在此持久化
                self.save_current(silent=True)

            res = result.get("batch_status", {})
            if res.get('status') == 'success' and res.get('count', 0) > 0:
                InfoBar.success("批量更新成功", f"已将 {res['count']} 个文件中的类别 '{old_name}' 重命名为 '{new_name}'", parent=self)

            # 类别列表/颜色刷新由 annotation:categories_changed 事件处理器（on_categories_changed）完成；
            # service 已持久化类别，save=False 避免重复持久化
            self._refresh_category_state(update_list=True, update_colors=True, save=False)

            InfoBar.success("重命名成功", f"已将 '{old_name}' 重命名为 '{new_name}'", parent=self)

    # ====== 区域: 自动标注 UI（原 batch_mixin） ======
    def _restore_custom_models_from_config(self):
        from core.backend.core import ModelFactory
        from core.backend.path_resolver import get_custom_models
        for role in ("interactive", "auto"):
            models = get_custom_models(role)
            for info in models:
                ModelFactory.register_custom_model(info["name"], info["category"], info["weight_path"], role=role)

    def _rebuild_model_menu(self, menu, models, enabled, current_model, on_change, role):
        """通用重建模型选择菜单：内置/自定义模型 + enabled 过滤 + 管理入口"""
        from PySide6.QtGui import QAction
        from core.backend.core import ModelFactory
        menu.clear()
        builtin = [m for m in models if m not in ModelFactory._custom_weight_paths]
        custom = [m for m in models if m in ModelFactory._custom_weight_paths]
        for name in builtin:
            if enabled is not None and name not in enabled:
                continue
            act = QAction(name, self)
            act.setCheckable(True)
            act.setChecked(name == current_model)
            act.triggered.connect(lambda checked, n=name: on_change(model_name=n))
            menu.addAction(act)
        if custom:
            menu.addSeparator()
            for name in custom:
                act = QAction(name, self)
                act.setCheckable(True)
                act.setChecked(name == current_model)
                act.triggered.connect(lambda checked, n=name: on_change(model_name=n))
                menu.addAction(act)
        menu.addSeparator()
        act_manage = QAction("管理模型...", self)
        act_manage.triggered.connect(lambda: self._open_model_selection_dialog(role))
        menu.addAction(act_manage)

    def _rebuild_interactive_model_menu(self):
        enabled = settings.get("enabled_interactive_models", None)
        interactive_models = self.model_manager.get_interactive_models()
        if enabled is None:
            from core.backend.core import ModelFactory
            enabled = [m for m in interactive_models if m not in ModelFactory._custom_weight_paths]
            settings.set("enabled_interactive_models", enabled)
        self._rebuild_model_menu(
            self._interactive_model_menu, interactive_models, enabled,
            getattr(self, 'interactive_model_type', ''),
            self.on_interactive_model_changed, "interactive"
        )

    def _rebuild_auto_model_menu(self):
        enabled = settings.get("enabled_auto_models", None)
        auto_models = self.model_manager.get_auto_models()
        if enabled is None:
            from core.backend.core import ModelFactory
            enabled = [m for m in auto_models if m not in ModelFactory._custom_weight_paths]
            settings.set("enabled_auto_models", enabled)
        self._rebuild_model_menu(
            self._auto_model_menu, auto_models, enabled,
            getattr(self, 'model_type', ''),
            self.on_auto_model_changed, "auto"
        )

    def _rebuild_plain_model_menu(self):
        enabled = settings.get("enabled_plain_models", None)
        plain_models = self.model_manager.get_plain_models()
        if enabled is None:
            enabled = list(plain_models)
            settings.set("enabled_plain_models", enabled)
        self._rebuild_model_menu(
            self._plain_model_menu, plain_models, enabled,
            getattr(self, 'plain_model_type', ''),
            self.on_plain_model_changed, "plain"
        )

    def _rebuild_refine_model_menu(self):
        from core.backend.core import ModelFactory
        refine_models = ModelFactory.get_refine_models()
        self._rebuild_model_menu(
            self._refine_model_menu, refine_models, None,
            settings.get("refine_method", "vitmatte"),
            self.on_refine_model_changed, "refine"
        )

    def on_refine_model_changed(self, model_name: str = ""):
        if model_name:
            settings.set("refine_method", model_name)
            print(f"Refine model changed to: {model_name}")

    def _on_menu_high_precision(self, checked):
        settings.set("high_precision", checked)
        if hasattr(self, 'cb_high_precision'):
            self.cb_high_precision.blockSignals(True)
            self.cb_high_precision.setChecked(checked)
            self.cb_high_precision.blockSignals(False)

    def _on_menu_enhance(self, checked):
        if hasattr(self, 'btn_enhance'):
            self.btn_enhance.setChecked(checked)
        self._on_enhance_btn_clicked()

    def _on_menu_show_bbox(self, checked):
        if hasattr(self, 'cb_show_bbox'):
            self.cb_show_bbox.setChecked(checked)

    def _on_menu_show_label(self, checked):
        if hasattr(self, 'cb_show_label'):
            self.cb_show_label.setChecked(checked)

    def _change_model(self, model_name, index, combo, settings_key, predictor_getter, is_interactive, role=None):
        """通用模型切换：按名称或索引更新 combo、当前模型、设置并异步加载"""
        if model_name:
            for i in range(combo.count()):
                if model_name == combo.itemText(i):
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    break
            else:
                return f"错误: 未知的模型 '{model_name}'"
        elif index is None or index < 0 or index >= combo.count():
            return
        else:
            model_name = combo.itemText(index)

        attr_map = {
            "interactive": "interactive_model_type",
            "example": "model_type",
            "plain": "plain_model_type",
            "refine": "model_type",
        }
        current_model = getattr(self.auto, attr_map.get(role, "model_type" if not is_interactive else "interactive_model_type"))
        if model_name == current_model:
            return
        try:
            predictor_getter().release_models()
        except Exception:
            pass
        changed = self.auto.change_model(model_name, is_interactive=is_interactive, role=role)
        if changed:
            settings.set(settings_key, model_name)
            print(f"DEBUG: Model changed to: {model_name}")

    def on_interactive_model_changed(self, index=None, model_name: str = ""):
        from core.backend.core import get_interactive_predictor
        self._change_model(
            model_name, index, self.interactive_model_combo,
            "default_interactive_model",
            get_interactive_predictor, True, role="interactive"
        )

    def on_auto_model_changed(self, index=None, model_name: str = ""):
        from core.backend.core import get_example_predictor
        self._change_model(
            model_name, index, self.auto_model_combo,
            "example_model",
            get_example_predictor, False, role="example"
        )

    def on_plain_model_changed(self, index=None, model_name: str = ""):
        if model_name:
            result = self.auto.set_model("plain", model_name)
            if result is not True:
                return result
            settings.set("plain_model", model_name)
            print(f"DEBUG: Plain model changed to: {model_name}")
            return
        plain_models = self.model_manager.get_plain_models()
        if index is None or index < 0 or index >= len(plain_models):
            return
        self.on_plain_model_changed(model_name=plain_models[index])

    def _open_model_selection_dialog(self, role: str):
        dialog = ModelSelectionDialog(role, self)
        dialog.exec_()

    def _refresh_model_combos(self):
        self.interactive_model_combo.blockSignals(True)
        current_interactive = self.interactive_model_type
        self.interactive_model_combo.clear()
        interactive_models = self.model_manager.get_interactive_models()
        self.interactive_model_combo.addItems(interactive_models)
        idx = self.interactive_model_combo.findText(current_interactive)
        if idx >= 0:
            self.interactive_model_combo.setCurrentIndex(idx)
        self.interactive_model_combo.blockSignals(False)

        self.auto_model_combo.blockSignals(True)
        current_auto = self.model_type
        self.auto_model_combo.clear()
        auto_models = self.model_manager.get_auto_models()
        self.auto_model_combo.addItems(auto_models)
        idx = self.auto_model_combo.findText(current_auto)
        if idx >= 0:
            self.auto_model_combo.setCurrentIndex(idx)
        self.auto_model_combo.blockSignals(False)

        if hasattr(self, '_interactive_model_menu'):
            self._rebuild_interactive_model_menu()
        if hasattr(self, '_auto_model_menu'):
            self._rebuild_auto_model_menu()

    @Slot()
    def on_sam_commit(self):
        if hasattr(self, '_sam_active') and self._sam_active:
            self._sam_active = False
            self.draw_area.sam_points = []
            self.draw_area.update()
            if self.auto_save_enabled:
                self.save_current(silent=True)
            print("DEBUG: SAM session committed via right click")
            if self.example_mode_enabled:
                ref = getattr(self, "_last_sam_annotation", None)
                if ref and ref.get("bbox"):
                    self.reference_annotation = ref
                    self.find_similar_in_current_image()

    def _set_as_example_core(self, index: int, example_name: str = ""):
        """纯数据操作：将标注加入示例库。返回示例名称"""
        return self.auto.set_as_example_by_index(
            index, self.draw_area.annotations, self.current_image_path,
            example_name, getattr(self.draw_area, 'persistent_roi', None)
        )

    def on_set_as_example(self, index: int = -1, example_name: str = ""):
        if not example_name:
            return "错误: on_set_as_example 需要 example_name 参数"
        name = self._set_as_example_core(index, example_name)
        if name is None:
            return "错误: 标注索引无效"
        return True

    def _on_set_as_example_with_dialog(self, index: int = -1):
        if index < 0 or index >= len(self.draw_area.annotations):
            return
        ann = self.draw_area.annotations[index]
        dialog = QDialog(self)
        dialog.setWindowTitle("添加到示例库")
        dialog.setMinimumWidth(350)
        v_layout = QVBoxLayout(dialog)
        v_layout.addWidget(CaptionLabel("为该示例命名 (方便后续选择):"))
        name_edit = LineEdit()
        name_edit.setText(f"示例_{len(self.auto.get_example_library()) + 1}")
        v_layout.addWidget(name_edit)
        btn_box = QHBoxLayout()
        ok_btn = PrimaryPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        btn_box.addStretch(1)
        btn_box.addWidget(ok_btn)
        v_layout.addLayout(btn_box)
        if dialog.exec_():
            name = name_edit.text().strip()
            saved_name = self._set_as_example_core(index, name)
            if saved_name:
                from core.common.action_recorder import ActionRecorder
                recorder = ActionRecorder.instance()
                if recorder.is_recording:
                    recorder.record_step(
                        action_name="set_as_example",
                        params={"index": index},
                        description=f"设为示例(标注#{index+1})",
                        expected_result="弹出命名对话框"
                    )
                InfoBar.success("示例库更新", f"已将 '{saved_name}' 添加到示例库。现在可以在批量标注时勾选使用。", parent=self)
            else:
                err = getattr(self.auto, "_last_example_error", "") or "标注索引无效"
                self.auto._last_example_error = ""
                InfoBar.error("添加到示例库失败", err, parent=self)

    def on_example_mode_changed(self, checked):
        self.example_mode_enabled = checked
        if checked:
            self.btn_mode_rect.click()

            if not self.reference_annotation and self.draw_area.annotations:
                self.reference_annotation = self.draw_area.annotations[-1]
                InfoBar.info("自动设定示例", "已自动将最后一个标注设为示例。", parent=self)

        status = "开启" if checked else "关闭"
        self.status_label.setText(f"示例模式: {status}")

    def list_examples(self):
        return self.auto.list_examples()

    def set_examples(self, example_names: list = None):
        return self.auto.set_examples(example_names)

    def get_selected_examples(self):
        return self.auto.get_selected_examples()

    def clear_examples_selection(self):
        return self.auto.clear_examples_selection()

    def remove_example(self, index: int):
        return self.auto.remove_example(index)

    def clear_example_library(self):
        return self.auto.clear_example_library()

    def list_prompts(self):
        return self.auto.list_prompts()

    def add_prompt(self, text: str):
        return self.auto.add_prompt(text)

    def remove_prompt(self, index: int):
        return self.auto.remove_prompt(index)

    def clear_prompts(self):
        return self.auto.clear_prompts()

    def get_rules(self):
        return self.auto.get_rules()

    def set_rule(self, key: str, value: str):
        return self.auto.set_rule(key, value)

    @action("resource.open_example_library", description="打开示例库管理对话框。\n- 查看和管理支持集示例库\n- 前提条件：示例库非空（需先用 set_as_example 添加示例）\n- 在对话框中可选择示例用于批量标注", category="自动标注")
    def open_example_library_manager(self):
        library = self.auto.get_example_library()
        if not library:
            InfoBar.info("提示", "当前示例库为空，请先添加示例。", parent=self)
            return "错误: 示例库为空，请先使用 set_as_example 添加示例"

        dialog = ExampleLibraryDialog(
            list(library), self,
            selected_names=self.auto.get_selected_examples(),
            delete_callback=lambda idx, item: self.auto.remove_example(idx)
        )
        if dialog.exec_():
            # 把勾选结果持久化到示例选择管理器，供 one_click_by_current/all 使用
            names = dialog.get_selected_names()
            self.auto.set_examples(names)
            InfoBar.success("示例选择已保存", f"已选择 {len(names)} 个示例", parent=self)
        return None

    def load_prompt_library(self):
        self.auto.load_prompt_library()

    def save_prompt_library(self):
        self.auto.save_prompt_library()

    def get_selected_prompts_from_library(self):
        return PromptManager.get_selected_prompts(self.prompt_library)

    def get_prompt_text_for_dialog(self):
        selected_prompts = self.get_selected_prompts_from_library()
        if selected_prompts:
            return ", ".join(selected_prompts)
        return ""

    @action("resource.open_prompt_library", description="打开提示词管理对话框。\n- 查看、添加和编辑提示词\n- 提示词用于 one_click_by_text 自动标注\n- 编辑后自动保存到项目设置", category="自动标注")
    def open_prompt_library_manager(self):
        if self.prompt_library is None:
            self.prompt_library = []

        dialog = PromptLibraryDialog(list(self.prompt_library), self)
        if dialog.exec_():
            self.prompt_library = dialog.get_all_prompts()
            self.save_prompt_library()
            InfoBar.success("提示词库已保存", f"已保存 {len(self.prompt_library)} 个提示词", parent=self)

    @action("resource.open_rule_config", description="打开规则配置对话框。\n- 配置自动标注的规则和参数\n- 规则包括文本提示词等配置\n- 保存后自动应用新规则", category="自动标注")
    def open_rule_config(self):
        if not hasattr(self, 'current_project_rules'):
            return "错误: 未加载项目规则配置"
        rule_dialog = RuleConfigDialog(self, initial_rules=self.current_project_rules)
        # 保留引用:agent set_rule 修改规则时,若对话框正打开则实时刷新其控件
        self._rule_dialog = rule_dialog
        rule_dialog.finished.connect(lambda _: setattr(self, "_rule_dialog", None))
        if rule_dialog.exec_():
            rules = rule_dialog.get_rules()
            self.save_project_rules(rules)
            if rules.get('text'):
                self.add_prompt(rules['text'])
            InfoBar.success("规则已保存", "规则配置已成功保存", parent=self)

    @action("batch.one_click_annotate", description="一键标注入口，显示标注操作菜单。\n- 展示包含示例库管理、提示词管理、规则配置等功能的菜单\n- 需要先打开图片目录\n- 无图片时提示打开目录", category="自动标注")
    @Slot()
    def on_one_click_clicked(self):
        if not self.image_files:
            InfoBar.warning("提示", "请先打开包含图片的目录", parent=self)
            return "错误: 请先打开包含图片的目录"

        menu = RoundMenu(parent=self)

        lib_action = Action(FIF.LIBRARY, "示例库管理")
        lib_action.triggered.connect(self.open_example_library_manager)
        menu.addAction(lib_action)
        menu.addSeparator()

        prompt_lib_action = Action(FIF.DICTIONARY, "提示词管理")
        prompt_lib_action.triggered.connect(self.open_prompt_library_manager)
        menu.addAction(prompt_lib_action)
        menu.addSeparator()

        rule_action = Action(FIF.SETTING, "规则配置")
        rule_action.triggered.connect(self.open_rule_config)
        menu.addAction(rule_action)
        menu.addSeparator()

        select_model_action = Action(AppIcon.MODEL, "普通模型管理")
        select_model_action.triggered.connect(lambda: self._open_model_selection_dialog("plain"))
        menu.addAction(select_model_action)

        if self.plain_model_type:
            model_name = self.plain_model_type
            model_status_action = Action(FIF.ACCEPT, f"当前普通模型: {model_name}")
            model_status_action.setEnabled(False)
            menu.addAction(model_status_action)

        menu.addSeparator()

        text_action = Action(FIF.EDIT, "基于文本标注全部图片")
        text_action.triggered.connect(lambda: self.one_click_by_all("text"))
        menu.addAction(text_action)

        text_curr_action = Action(FIF.EDIT, "基于文本标注当前图片")
        text_curr_action.triggered.connect(lambda: self.one_click_by_current("text"))
        menu.addAction(text_curr_action)

        menu.addSeparator()

        library = self.auto.get_example_library()
        if library or self.reference_annotation:
            example_action = Action(FIF.LABEL, "基于示例标注全部图片")
            example_action.triggered.connect(lambda: self.one_click_by_all("example"))
            menu.addAction(example_action)

            example_curr_action = Action(FIF.LABEL, "基于示例标注当前图片")
            example_curr_action.triggered.connect(lambda: self.one_click_by_current("example"))
            menu.addAction(example_curr_action)

        menu.addSeparator()

        model_all_action = Action(FIF.VIDEO, "基于普通模型标注全部图片")
        model_all_action.triggered.connect(lambda: self.one_click_by_all("model"))
        menu.addAction(model_all_action)

        model_curr_action = Action(FIF.VIDEO, "基于普通模型标注当前图片")
        model_curr_action.triggered.connect(lambda: self.one_click_by_current("model"))
        menu.addAction(model_curr_action)

        menu.addSeparator()

        fit_check_action = Action(FIF.SEARCH, "检查当前图片标注贴合度")
        fit_check_action.triggered.connect(self.fit_check_current_ui)
        menu.addAction(fit_check_action)

        menu.exec(QCursor.pos())

    @action("batch.one_click_by_all_ui", description="按模式对全部图片进行批量自动标注（**立即返回，任务在后台线程执行**）。\n- mode: 标注模式，text=基于文本 / example=基于示例 / model=基于普通模型\n- text: mode=text 时的描述文本（可选），留空则使用提示词库中选中的提示词\n- 本方法只负责启动：返回「已在后台开始」。进度由进度对话框展示，完成/取消由界面提示\n- 需要**同步拿到处理结果**（processed/total/errors）时请用 service 层的 batch.one_click_by_all\n- 已有批量在跑时返回错误，不会并发执行", category="自动标注", params={"mode": "str", "text": "str"}, scope="ui")
    def one_click_by_all(self, mode: str = "", text: str = "", selected_examples: list = None):
        """UI 包装：把批量标注交给**后台线程**执行并立即返回。

        刻意不再同步执行：批量可能是成百上千张图，同步跑就必须靠 processEvents
        才能重绘进度条、让取消按钮可点，而那会在状态变更处理器内部重入事件循环
        （用户在批量进行中还能切图/保存/删除 -> 界面与磁盘状态错乱）。
        现在批量跑在 BackgroundTaskRunner 上：进度走 annotation:batch_progress 事件，
        取消走 `auto.cancel_batch()` 标志，结果由 `_on_batch_job_finished` 在 GUI 线程提示。
        """
        mode = (mode or "").strip()
        if mode == "text" and not text:
            selected_prompts = self.get_selected_prompts_from_library()
            if not selected_prompts:
                InfoBar.warning("提示", "提示词库中没有选中的提示词，请先添加并勾选提示词", parent=self)
                return "错误: 提示词库中没有选中的提示词，请先添加并勾选提示词"
            text = ", ".join(selected_prompts)

        if self._batch_runner.busy():
            InfoBar.warning("提示", "已有批量标注正在进行中", parent=self)
            return "错误: 已有批量标注正在进行中，请等待结束或先取消"

        total = len(self.manual.image_files or [])
        started = self._batch_runner.start(
            lambda: self.auto.one_click_by_all(
                mode=mode, text=text, selected_examples=selected_examples),
            on_result=self._on_batch_job_finished,
            on_error=self._on_batch_job_failed,
        )
        if not started:
            return "错误: 已有批量标注正在进行中，请等待结束或先取消"
        return f"批量标注已在后台开始（共 {total} 张图片），进度见进度对话框"

    def _on_batch_job_finished(self, result):
        """批量任务结束（GUI 线程）：刷新类别状态并提示结果。"""
        self._batch_result = result
        try:
            self._refresh_category_state()
        except Exception as e:
            print(f"[AnnotationInterface] 批量完成后刷新类别状态失败: {e}")
        if isinstance(result, dict) and result.get("status") == "success":
            total = result.get("total", 0)
            processed = result.get("processed", 0)
            errors = result.get("errors") or []
            if result.get("canceled"):
                InfoBar.warning("批量标注已取消",
                                f"已处理 {processed}/{total} 张图片", parent=self)
            elif errors:
                InfoBar.warning("批量标注完成（部分失败）",
                                f"共 {total} 张，失败 {len(errors)} 张；详见控制台日志",
                                parent=self)
                for err in errors[:10]:
                    print(f"[AnnotationInterface] 批量标注失败项: {err}")
            else:
                InfoBar.success("批量标注完成", f"共处理 {total} 张图片", parent=self)
        elif isinstance(result, str) and result.startswith("错误"):
            InfoBar.error("批量标注失败", result, parent=self)

    def _on_batch_job_failed(self, message):
        """批量任务抛异常（GUI 线程）。"""
        print(f"[AnnotationInterface] 批量标注异常: {message}")
        InfoBar.error("批量标注失败", message, parent=self)

    @action("batch.one_click_by_current_ui", description="按模式对当前图片进行自动标注并刷新画布。\n- mode: 标注模式，text=基于文本 / example=基于示例 / model=基于普通模型\n- text: mode=text 时的描述文本，留空则使用提示词库中选中的提示词\n- 数据由 service 层写入磁盘，成功后自动重载并刷新画布/列表/类别", category="自动标注", params={"mode": "str", "text": "str"}, scope="ui")
    def one_click_by_current(self, mode: str = "", text: str = ""):
        """UI 包装：调用 service 层 one_click_by_current 后通过 refresh_view 刷新界面。"""
        mode = (mode or "").strip()
        self.status_label.setText("正在自动标注当前图片...")

        res = self.auto.one_click_by_current(
            mode=mode, text=text,
            reference_annotation=self.reference_annotation,
            task_mode=self.draw_area.task_mode,
        )

        if res['status'] != 'success':
            InfoBar.warning("提示", res.get('message', ''), parent=self)
            self.status_label.setText(f"标注失败: {res.get('message', '')}")
            return f"错误: {res.get('message', '')}"

        if not res.get('annotations'):
            detected = res.get('detected', 0)
            duplicates = res.get('duplicates', 0)
            if detected:
                msg = (f"检测到 {detected} 个目标，但均与现有标注重重（已去重），未新增。"
                       f"如需重新标注请先删除对应标注")
                self.status_label.setText(msg)
                return f"提示: {msg}"
            self.status_label.setText("未检测到新目标，可能被规则过滤或模型未命中")
            return "提示: 未检测到新目标，可能被规则过滤（如 area_range/conf_threshold）或模型未命中，可先查看规则配置"

        # 数据已由 service 层写入并发布 annotations_changed 信号，界面自动刷新
        added = len(res.get('annotations', []))
        if mode == "text":
            self.status_label.setText(f"已添加 {added} 个新标注 (基于提示词: {res['prompt']})")
        elif mode == "example":
            self.status_label.setText(f"已添加 {added} 个相似标注")
        else:
            new_categories = res.get('new_categories', [])
            if new_categories:
                cat_names = [label for label, _ in new_categories]
                self.status_label.setText(f"已添加 {added} 个新标注，新增类别: {', '.join(cat_names)}")
            else:
                self.status_label.setText(f"已添加 {added} 个新标注")

    @action("manual.fit_check_current_ui",
            description="检查当前图片所有标注框与实际目标轮廓的贴合度，弹窗提示概览并允许选择将标注替换为 hybrid 分割结果。\n- 小目标（max(w,h)<30）使用超分+经典CV前景提取，中大目标使用SAM2交互分割\n- 检查完成后弹出对话框，用户可勾选要替换的标注并应用，替换后自动保存并刷新画布",
            category="标注", scope="ui")
    def fit_check_current_ui(self):
        """UI 包装：调用 service 层贴合度检查，弹窗汇总结果并支持应用 hybrid 结果替换标注。"""
        if not self.current_image_path:
            InfoBar.warning("提示", "请先打开图片目录并选择图片", parent=self)
            return "错误: 请先打开图片目录并选择图片"
        self.status_label.setText("正在检查标注贴合度...")
        result = self.manual.fit_check_current(image_path=self.current_image_path)
        if result.get('status') != 'success':
            InfoBar.warning("提示", result.get('message', ''), parent=self)
            self.status_label.setText(f"贴合度检查失败: {result.get('message', '')}")
            return f"错误: {result.get('message', '')}"
        rows = result.get('results', [])
        poor = []
        ok_count = 0
        for r in rows:
            if not r.get('detected'):
                poor.append(f"{r.get('label', '')}#{r.get('idx', '')}(未检出)")
                continue
            area_ratio = r.get('area_ratio', 1.0)
            bbox_iou = r.get('bbox_iou', 1.0)
            centroid_offset = r.get('centroid_offset', 0.0)
            if area_ratio < 0.15 or bbox_iou < 0.4 or centroid_offset > 0.4:
                poor.append(f"{r.get('label', '')}#{r.get('idx', '')}(ar={area_ratio},iou={bbox_iou},off={centroid_offset})")
            else:
                ok_count += 1
        msg = f"共检查 {len(rows)} 个标注，贴合良好 {ok_count} 个"
        if poor:
            preview = "、".join(poor[:5])
            if len(poor) > 5:
                preview += f" 等 {len(poor)} 个"
            msg += f"，需关注: {preview}"
            InfoBar.warning("贴合度检查", msg, parent=self)
        else:
            InfoBar.success("贴合度检查", msg, parent=self)
        self.status_label.setText(msg)

        replaceable = [r for r in rows if r.get('detected') and r.get('polygons')]
        if not replaceable:
            return result
        dialog = FitCheckDialog(rows, parent=self)
        if dialog.exec() != FitCheckDialog.DialogCode.Accepted:
            return result
        indices = dialog.selected_indices()
        if not indices:
            return result
        applied = 0
        for i in indices:
            res = self.manual.refine_annotation(
                index=i,
                annotations=self.draw_area.annotations,
                image_path=self.current_image_path,
                refine_method="hybrid",
            )
            if res.get('status') == 'success':
                applied += 1
        if applied <= 0:
            InfoBar.warning("提示", "没有标注被替换，请检查标注是否有效", parent=self)
            return result
        # refine_annotation 已发布 annotations_changed，画布/列表自动刷新
        InfoBar.success("贴合度检查", f"已用 hybrid 细化替换 {applied} 个标注并保存", parent=self)
        self.status_label.setText(f"已用 hybrid 细化替换 {applied} 个标注并保存")
        return {"status": "success", "applied": applied}

    def _ensure_yolo_model_loaded(self):
        # 普通模型由模型菜单管理，无需文件对话框；直接校验当前已选普通模型
        return self.auto.ensure_yolo_model_loaded()

    def _ensure_plain_model_loaded(self):
        """校验当前已选择可用的普通模型（模型菜单管理，无文件对话框）。"""
        return self.auto.is_plain_model_available()

    def select_trained_model(self, model_path: str = ""):
        if not model_path:
            return "错误: 需要 model_path 参数"
        return self.auto.select_trained_model(model_path)

    def list_custom_models(self, role: str = ""):
        return self.auto.list_custom_models(role)

    def list_builtin_models(self, role: str = ""):
        return self.auto.list_builtin_models(role)

    def add_custom_model(self, name: str, category: str, weight_path: str, role: str = "auto"):
        result = self.auto.add_custom_model(name, category, weight_path, role)
        rebuild_map = {
            "interactive": "_rebuild_interactive_model_menu",
            "auto": "_rebuild_auto_model_menu",
            "plain": "_rebuild_plain_model_menu",
            "refine": "_rebuild_refine_model_menu",
        }
        rebuild_method = rebuild_map.get(role)
        if rebuild_method and hasattr(self, rebuild_method):
            getattr(self, rebuild_method)()
        return result

    def remove_custom_model(self, name: str, role: str = "auto"):
        result = self.auto.remove_custom_model(name, role)
        rebuild_map = {
            "interactive": "_rebuild_interactive_model_menu",
            "auto": "_rebuild_auto_model_menu",
            "plain": "_rebuild_plain_model_menu",
            "refine": "_rebuild_refine_model_menu",
        }
        rebuild_method = rebuild_map.get(role)
        if rebuild_method and hasattr(self, rebuild_method):
            getattr(self, rebuild_method)()
        return result

    def _add_new_categories_from_annotations(self, annotations):
        new_labels = self.auto.discover_categories(annotations)

        if new_labels:
            self._refresh_category_state()

        return new_labels

    def _merge_and_add_annotations(self, new_annotations):
        """将新标注与当前标注去重合并后添加到画布并刷新列表"""
        existing_annotations = list(self.draw_area.annotations)
        added_annotations = self.auto.merge_annotations(
            new_annotations, existing_annotations
        )
        for ann in added_annotations:
            self.manual.add_annotation(ann)
        self.refresh_label_list()
        return added_annotations

    def bbox_to_poly(self, bbox):
        x, y, w, h = bbox
        return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
