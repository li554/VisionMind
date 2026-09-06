import json
import os

from PySide6.QtCore import Qt, Signal, QThread, QObject, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QListWidgetItem, QSizePolicy
from qfluentwidgets import (PrimaryPushButton, BodyLabel, CaptionLabel,
                            StrongBodyLabel, DoubleSpinBox, CheckBox,
                            FluentIcon, InfoBar, ListWidget, ComboBox, IconWidget, SpinBox, MessageBox,
                            SingleDirectionScrollArea, TransparentToolButton, ScrollArea)
from core.common.custom_card import SolidCardWidget as CardWidget

from core.common.widgets.base import ProgressDialog
from core.common.widgets.example_selection_dialog import ExampleLibraryDialog

from core.service.common_service import ExampleSelectionManager
from core.common.action_registry import action
from ..services.version_service import VersionService

class VersionExportWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, service, config):
        super().__init__()
        self.service = service
        self.config = config

    def run(self):
        try:
            res = self.service.export_version_v2(
                self.config,
                progress_callback=self.progress.emit
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class DatasetSplitWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, service, project_name, split_ratios):
        super().__init__()
        self.service = service
        self.project_name = project_name
        self.split_ratios = split_ratios

    def run(self):
        try:
            res = self.service.random_split_dataset(
                self.project_name,
                self.split_ratios,
                progress_callback=self.progress.emit
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))

class AugmentationCard(CardWidget):
    def __init__(self, title, description, params=None, presets=None, parent=None, has_select=False):
        super().__init__(parent)
        self.params = params or {} # {name: (min, max, default)}
        self.presets = presets or {} # {name: [val1, val2, ...]}
        self.widgets = {}
        self.has_select = has_select
        self.selected_data = [] # Stores custom selection data (e.g., folder names)

        # Modern look
        self.setObjectName("AugmentationCard")

        # Important: Allow the card to expand vertically
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(6)

        # Header: checkbox + title + description on one line
        header = QHBoxLayout()
        header.setSpacing(8)
        self.checkbox = CheckBox(title)
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        header.addWidget(self.checkbox)
        self.desc_label = CaptionLabel(description)
        header.addWidget(self.desc_label)
        header.addStretch()
        layout.addLayout(header)

        # Selection info label (if has_select)
        if self.has_select:
            self.select_info = CaptionLabel("已选择: 0 个")
            self.select_info.setProperty("class", "AugmentationSelectInfo")
            self.select_info.setVisible(False)
            layout.addWidget(self.select_info)

            # Use PushButton with explicit icon size and padding to avoid overlap
            from qfluentwidgets import PushButton
            self.select_btn = PushButton(FluentIcon.FOLDER_ADD, "选择示例")
            self.select_btn.setVisible(False)
            self.select_btn.setFixedHeight(32)
            self.select_btn.setMinimumWidth(120) # Ensure enough space for icon and text
            layout.addWidget(self.select_btn)

        # Parameters Container
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)
        self.param_layout.setContentsMargins(0, 5, 0, 5)
        self.param_layout.setSpacing(12)

        if self.params:
            for pname, pinfo in self.params.items():
                pmin, pmax, pdef = pinfo

                row = QHBoxLayout()
                label = BodyLabel(pname)
                label.setFixedWidth(70)
                row.addWidget(label)

                if pname in self.presets:
                    # Use ComboBox for presets
                    combo = ComboBox()
                    choices = [str(x) for x in self.presets[pname]]
                    combo.addItems(choices)
                    if str(pdef) in choices:
                        combo.setCurrentText(str(pdef))
                    else:
                        # If default not in choices, add it
                        combo.addItem(str(pdef))
                        combo.setCurrentText(str(pdef))
                    combo.setFixedWidth(120)

                    row.addWidget(combo)
                    self.widgets[pname] = combo
                elif isinstance(pdef, float):
                    spin = DoubleSpinBox()
                    spin.setRange(pmin, pmax)
                    spin.setValue(pdef)
                    spin.setSingleStep(0.01) # Smaller step for better precision
                    spin.setDecimals(3)     # Show more decimals
                    spin.setFixedWidth(120)  # Standardize width

                    row.addWidget(spin)
                    self.widgets[pname] = spin
                else:
                    spin = SpinBox()
                    spin.setRange(pmin, pmax)
                    spin.setValue(pdef)
                    spin.setFixedWidth(120)

                    row.addWidget(spin)
                    self.widgets[pname] = spin

                self.param_layout.addLayout(row)

            layout.addWidget(self.param_widget)
            self.param_widget.setVisible(False)
            self.setMinimumHeight(50)
        else:
            self.setFixedHeight(50)

    def _on_checkbox_changed(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        if self.has_select:
            self.select_info.setVisible(is_checked)
            self.select_btn.setVisible(is_checked)

        if self.params:
            self.param_widget.setVisible(is_checked)
            # Update minimum height based on visibility to help the layout
            if is_checked:
                # Approximate height calculation: 50 (header) + params * row_height
                h = 70 + len(self.params) * 40
                if self.has_select: h += 50
                self.setMinimumHeight(h)
            else:
                self.setMinimumHeight(50)
            self.updateGeometry()

    def get_settings(self):
        if not self.checkbox.isChecked():
            return None

        res = {}
        for name, w in self.widgets.items():
            if isinstance(w, ComboBox):
                try:
                    # Try to convert to int or float if possible
                    val = w.currentText()
                    if '.' in val: res[name] = float(val)
                    else: res[name] = int(val)
                except:
                    res[name] = w.currentText()
            else:
                res[name] = w.value()

        if self.has_select:
            res['Selected'] = self.selected_data

        return res

class VersionItemWidget(QWidget):
    """自定义版本列表项"""
    delete_requested = Signal(str)
    open_requested = Signal(str)

    def __init__(self, version_name, info=None, parent=None):
        super().__init__(parent)
        self.version_name = version_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(15)

        # Icon
        self.icon_widget = IconWidget(FluentIcon.FOLDER)
        self.icon_widget.setFixedSize(20, 20)
        layout.addWidget(self.icon_widget)

        # Version Name & Info
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        self.name_label = StrongBodyLabel(version_name)
        text_layout.addWidget(self.name_label)

        # Subtitle info (e.g. date or image count)
        sub_text = info if info else "数据集快照"
        self.info_label = CaptionLabel(sub_text)
        self.info_label.setProperty("class", "VersionInfoLabel")
        text_layout.addWidget(self.info_label)

        layout.addLayout(text_layout)
        layout.addStretch()

        # Action Buttons
        self.open_btn = TransparentToolButton(FluentIcon.FOLDER_ADD, self)
        self.open_btn.setToolTip("在资源管理器中打开")
        self.open_btn.clicked.connect(lambda: self.open_requested.emit(self.version_name))

        self.del_btn = TransparentToolButton(FluentIcon.DELETE, self)
        self.del_btn.setToolTip("删除版本")
        self.del_btn.clicked.connect(lambda: self.delete_requested.emit(self.version_name))

        layout.addWidget(self.open_btn)
        layout.addWidget(self.del_btn)

class VersionInterface(QFrame):
    def __init__(self, project_service, parent=None):
        super().__init__(parent)
        self.version_service = VersionService(project_service)
        self.setObjectName("VersionInterface")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._is_exporting = False

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 15, 30, 30)
        self.main_layout.setSpacing(20)

        # Main Content - Horizontal Split
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)

        # Left Panel: Project Selector + History and Statistics (Visual)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)

        # Project Selector (moved from header to left panel top)
        project_sel_layout = QHBoxLayout()
        self.project_label = CaptionLabel("切换项目:")
        self.project_combo = ComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.currentIndexChanged.connect(self.on_project_combo_changed)
        project_sel_layout.addWidget(self.project_label)
        project_sel_layout.addWidget(self.project_combo)
        project_sel_layout.addStretch()
        left_panel.addLayout(project_sel_layout)

        # Stats Card
        self.stats_card = CardWidget()
        self.stats_card.setFixedHeight(130)
        self.stats_card.setObjectName("VersionStatsCard")
        stats_layout = QHBoxLayout(self.stats_card)
        stats_layout.setContentsMargins(20, 15, 20, 15)

        def create_stat_item(title, value, icon, color="#3b82f6"):
            item = QVBoxLayout()
            item.setSpacing(8)
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)

            icon_lbl = IconWidget(icon)
            icon_lbl.setFixedSize(28, 28)
            item.addWidget(icon_lbl, 0, Qt.AlignmentFlag.AlignCenter)

            val_lbl = StrongBodyLabel(str(value))
            val_lbl.setProperty("class", "VersionStatValue")
            val_lbl.setProperty("statColor", color)
            item.addWidget(val_lbl, 0, Qt.AlignmentFlag.AlignCenter)

            tit_lbl = CaptionLabel(title)
            tit_lbl.setProperty("class", "VersionStatTitle")
            item.addWidget(tit_lbl, 0, Qt.AlignmentFlag.AlignCenter)
            return item

        self.stat_img_count = create_stat_item("总图片数", "0", FluentIcon.PHOTO, "#10b981") # Green
        self.stat_ver_count = create_stat_item("版本总数", "0", FluentIcon.SYNC, "#3b82f6")  # Blue
        self.stat_cls_count = create_stat_item("类别数量", "0", FluentIcon.TAG, "#f59e0b")   # Amber

        stats_layout.addLayout(self.stat_img_count)
        stats_layout.addLayout(self.stat_ver_count)
        stats_layout.addLayout(self.stat_cls_count)
        left_panel.addWidget(self.stats_card)

        # History List
        self.history_card = CardWidget()
        self.history_card.setObjectName("HistoryCard")
        history_layout = QVBoxLayout(self.history_card)
        history_layout.setContentsMargins(15, 15, 15, 15)

        history_header = QHBoxLayout()
        history_header.addWidget(StrongBodyLabel("版本历史"))
        history_header.addStretch()
        self.btn_refresh_history = TransparentToolButton(FluentIcon.SYNC, self.history_card)
        self.btn_refresh_history.setToolTip("刷新列表")
        self.btn_refresh_history.clicked.connect(self.refresh_versions)
        history_header.addWidget(self.btn_refresh_history)
        history_layout.addLayout(history_header)

        self.version_list = ListWidget()
        self.version_list.setObjectName("VersionHistoryList")
        history_layout.addWidget(self.version_list)
        left_panel.addWidget(self.history_card)

        content_layout.addLayout(left_panel, 5)

        # Right Panel: Export Settings (Config) with ScrollArea
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        self.scroll_area = SingleDirectionScrollArea(orient=Qt.Orientation.Vertical)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("VersionSettingsScroll")

        self.settings_group = QFrame()
        self.settings_group.setObjectName("VersionSettingsGroup")
        self.settings_group.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.scroll_area.setWidget(self.settings_group)

        right_panel.addWidget(self.scroll_area)

        settings_layout = QVBoxLayout(self.settings_group)
        settings_layout.setContentsMargins(20, 5, 20, 40) # 减少上方空白
        settings_layout.setSpacing(15)
        settings_layout.setSizeConstraint(QVBoxLayout.SetMinimumSize) # Force container to grow with children

        settings_layout.addWidget(StrongBodyLabel("导出新版本"))
        settings_layout.addSpacing(10)

        # 1. Dataset Splitting
        settings_layout.addWidget(StrongBodyLabel("1. 数据划分 (Dataset Splitting)"))

        # Split mode toggle
        self.use_existing_split = CheckBox("优先使用手动标注界面的划分结果")
        self.use_existing_split.setChecked(True)
        self.use_existing_split.stateChanged.connect(self.on_split_mode_changed)
        settings_layout.addWidget(self.use_existing_split)

        self.ratio_container = QWidget()
        ratio_layout = QHBoxLayout(self.ratio_container)
        ratio_layout.setContentsMargins(0, 0, 0, 0)

        self.train_ratio = DoubleSpinBox()
        self.train_ratio.setRange(0, 1)
        self.train_ratio.setValue(0.7)
        self.val_ratio = DoubleSpinBox()
        self.val_ratio.setRange(0, 1)
        self.val_ratio.setValue(0.2)
        self.test_ratio = DoubleSpinBox()
        self.test_ratio.setRange(0, 1)
        self.test_ratio.setValue(0.1)

        ratio_layout.addWidget(CaptionLabel("Train:"))
        ratio_layout.addWidget(self.train_ratio)
        ratio_layout.addWidget(CaptionLabel("Val:"))
        ratio_layout.addWidget(self.val_ratio)
        ratio_layout.addWidget(CaptionLabel("Test:"))
        ratio_layout.addWidget(self.test_ratio)
        settings_layout.addWidget(self.ratio_container)
        self.ratio_container.setEnabled(False) # Default disabled as we use existing by default

        settings_layout.addSpacing(15)

        # 1.5 导出设置
        settings_layout.addWidget(StrongBodyLabel("1.5 导出设置"))

        fmt_layout = QHBoxLayout()
        self.export_format_label = CaptionLabel("导出格式:")
        fmt_layout.addWidget(self.export_format_label)

        self.export_format_combo = ComboBox()
        self.export_format_combo.addItems(["yoloseg", "coco", "voc", "labelme"])
        self.export_format_combo.setCurrentIndex(0)
        self.export_format_combo.setMinimumWidth(120)
        self.export_format_combo.currentIndexChanged.connect(self.on_export_format_changed)
        fmt_layout.addWidget(self.export_format_combo)

        self.export_task_type_label = CaptionLabel("导出任务类型:")
        fmt_layout.addWidget(self.export_task_type_label)

        self.export_task_type_combo = ComboBox()
        self.export_task_type_combo.addItems(["检测", "分割", "旋转"])
        self.export_task_type_combo.setCurrentIndex(1)
        self.export_task_type_combo.setMinimumWidth(100)
        self.export_task_type_combo.currentIndexChanged.connect(self.on_export_task_type_changed)
        # 任务类型变更时更新可选导出格式
        self._update_export_format_options()
        fmt_layout.addWidget(self.export_task_type_combo)

        fmt_layout.addStretch()
        settings_layout.addLayout(fmt_layout)

        # 是否导出未标注图片（默认勾选，保留现状）
        export_opt_layout = QHBoxLayout()
        self.export_unannotated_check = CheckBox("导出未标注图片")
        self.export_unannotated_check.setChecked(True)
        self.export_unannotated_check.setToolTip(
            "勾选则导出所有图片（含未标注）；取消勾选则跳过没有标注的图片，仅导出已标注图片")
        export_opt_layout.addWidget(self.export_unannotated_check)
        export_opt_layout.addStretch(1)
        settings_layout.addLayout(export_opt_layout)

        # 任务类型提示标签
        self.task_type_hint = CaptionLabel("")
        self.task_type_hint.setProperty("class", "VersionTaskHint")
        settings_layout.addWidget(self.task_type_hint)

        settings_layout.addSpacing(15)

        # 2. Preprocessing
        settings_layout.addWidget(StrongBodyLabel("2. 预处理策略 (Preprocessing)"))

        # Define presets for size
        size_presets = [224, 256, 384, 640, 644, 768, 1024]
        self.pre_resize = AugmentationCard("Resize (调整尺寸)", "统一缩放图片尺寸",
                                         params={"Width": (32, 2048, 640), "Height": (32, 2048, 640)},
                                         presets={"Width": size_presets, "Height": size_presets})
        self.pre_auto_orient = AugmentationCard("Auto-Orient", "根据 EXIF 自动校准方向")

        pre_layout = QHBoxLayout()
        pre_layout.addWidget(self.pre_resize)
        pre_layout.addWidget(self.pre_auto_orient)
        settings_layout.addLayout(pre_layout)

        settings_layout.addSpacing(15)

        # 3. Augmentations
        settings_layout.addWidget(StrongBodyLabel("3. 数据增强策略 (Augmentations)"))

        self.aug_grid = QGridLayout()
        self.aug_grid.setSpacing(10)

        self.aug_hsv = AugmentationCard("HSV 增强", "随机调整色调、饱和度",
                                       params={"H": (0.0, 1.0, 0.015), "S": (0.0, 1.0, 0.7), "V": (0.0, 1.0, 0.4)})
        self.aug_rotate = AugmentationCard("随机旋转", "随机旋转图片角度",
                                          params={"Degrees": (0, 180, 15)})
        self.aug_flip = AugmentationCard("水平翻转", "随机进行左右水平翻转")
        self.aug_blur = AugmentationCard("模糊噪声", "添加高斯模糊强度",
                                        params={"Sigma": (0.0, 5.0, 1.0)})
        self.aug_copypaste = AugmentationCard("Copy-Paste", "随机跨图复制粘贴目标",
                                             params={"Count": (1, 10, 3)})
        self.aug_example_copypaste = AugmentationCard("示例库 Copy-Paste", "从示例库随机抽取目标粘贴",
                                                    params={"Count": (1, 10, 3)},
                                                    has_select=True)
        self.aug_example_copypaste.select_btn.clicked.connect(self.select_examples)

        self.aug_grid.addWidget(self.aug_hsv, 0, 0)
        self.aug_grid.addWidget(self.aug_rotate, 0, 1)
        self.aug_grid.addWidget(self.aug_flip, 1, 0)
        self.aug_grid.addWidget(self.aug_blur, 1, 1)
        self.aug_grid.addWidget(self.aug_copypaste, 2, 0)
        self.aug_grid.addWidget(self.aug_example_copypaste, 2, 1)
        settings_layout.addLayout(self.aug_grid)

        settings_layout.addSpacing(20)

        # Export Button
        self.export_btn = PrimaryPushButton("立即生成新版本", self, FluentIcon.SYNC)
        self.export_btn.setFixedHeight(40)
        self.export_btn.clicked.connect(self.generate_version)
        settings_layout.addWidget(self.export_btn)

        content_layout.addLayout(right_panel, 6) # Slightly wider

        self.main_layout.addLayout(content_layout, 1)

    @action("resource.set_example_sets_ui", description="选择用于版本生成的示例数据集（UI 包装）。\n- example_names: 要使用的示例名称列表（可用 list_example_sets 获取）\n- 选择后会同步更新界面示例选择状态，影响后续版本生成的 Copy-Paste 示例来源", category="版本管理", params={"example_names": "list"}, scope="ui")
    def set_example_sets_ui(self, example_names: list = None):
        project_name = self.project_combo.currentText() if hasattr(self, 'project_combo') else None
        if not example_names:
            self.version_service.example_selector.clear_selected_examples()
            self.aug_example_copypaste.selected_data = []
            self.aug_example_copypaste.select_info.setText("未选择示例")
            return {"status": "success", "message": "已清空示例选择"}

        example_sets = self.version_service.list_example_sets(project_name)
        available_names = ExampleSelectionManager.get_available_names(example_sets)
        res = self.version_service.example_selector.set_selected_examples(example_names, available_names)
        if res["status"] == "success":
            self.aug_example_copypaste.selected_data = list(res["selected"])
            self.aug_example_copypaste.select_info.setText(f"已选择：{res['count']} 个")
        return res

    @action("resource.select_examples", description="打开示例图片选择对话框，为版本生成选择参考示例。\n- 无参数\n- 从示例库中选择图片\n- 选择的示例会用于版本生成", category="版本管理", scope="ui")

    def select_examples(self):
        project_name = self.project_combo.currentText() if hasattr(self, 'project_combo') else None
        example_sets = self.version_service.list_example_sets(project_name)
        if not example_sets:
            support_sets_dir = self.version_service.get_support_sets_dir(project_name)
            if not support_sets_dir:
                support_sets_dir = os.path.join(self.version_service.base_dir, "support_sets")
            return "错误: 示例库为空" if not os.path.exists(support_sets_dir) else \
                   "错误: 示例目录存在但未找到有效示例集"

        selected_folders = getattr(self.aug_example_copypaste, 'selected_data', [])
        dialog = ExampleLibraryDialog(
            example_sets, self,
            title="选择 Copy-Paste 示例",
            selected_names=selected_folders
        )
        if dialog.exec_():
            result_names = dialog.get_selected_names()
            self.aug_example_copypaste.selected_data = result_names
            self.aug_example_copypaste.select_info.setText(f"已选择：{len(result_names)} 个")
            example_sets = self.version_service.list_example_sets(project_name)
            available_names = ExampleSelectionManager.get_available_names(example_sets)
            self.version_service.example_selector.set_selected_examples(result_names, available_names)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_project_list()
        self.refresh_versions()
        self.update_stats()
    @action("nav.project_combo_changed", description="切换到指定项目的版本管理界面。\n- 切换后会加载该项目的版本列表和统计信息", category="版本管理")

    def on_project_combo_changed(self, index):
        if index < 0:
            return "错误: 无效的索引"
        project_name = self.project_combo.itemText(index)
        if not project_name:
            return "错误: 项目名称为空"
        project_info = self.version_service.load_project(project_name)
        if project_info:
            # 清空示例选择，因为示例是项目级别的
            self.aug_example_copypaste.selected_data = []
            self.aug_example_copypaste.select_info.setText("未选择示例")
            self.version_service.example_selector.clear_selected_examples()

            # 根据项目任务类型自动设置导出任务类型和格式（内部统一短名 det/seg/obb）
            from core.service.project_service import normalize_task_type
            task_type = normalize_task_type(project_info.get('task_type', 'seg'))
            task_index_map = {'det': 0, 'seg': 1, 'obb': 2}
            task_idx = task_index_map.get(task_type, 1)
            self.export_task_type_combo.blockSignals(True)
            self.export_task_type_combo.setCurrentIndex(task_idx)
            self.export_task_type_combo.blockSignals(False)
            self._update_export_format_options()
            self._update_task_type_hint()

            self.refresh_versions()
            self.update_stats()
        else:
            return "错误: 项目不存在"
    @action("project.refresh_project_list", description="刷新版本管理界面的项目下拉列表。\n- 无参数\n- 从数据库重新加载所有项目名称", category="版本管理", scope="ui")
    def refresh_project_list(self):
        projects = self.version_service.get_all_projects()
        if not projects:
            return "错误: 没有找到项目"
        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        # 1. Try to get name from current_project
        current_name = self.version_service.current_project.get('name', "") if self.version_service.current_project else ""

        # 2. If not found, try to get from settings
        if not current_name:
            from core.common.settings import settings
            current_name = settings.get("current_project_name", "")
            if current_name:
                # Proactively load it
                self.version_service.load_project(current_name)

        current_idx = -1
        for i, p in enumerate(projects):
            self.project_combo.addItem(p['name'])
            if p['name'] == current_name:
                current_idx = i

        if current_idx >= 0:
            self.project_combo.setCurrentIndex(current_idx)
        else:
            self.project_combo.setCurrentIndex(-1)

        self.project_combo.blockSignals(False)
    @action("version.update_stats", description="更新当前项目的标注统计信息。\n- 无参数\n- 统计内容包括标注总数、类别分布等\n- 更新后会刷新显示", category="版本管理")

    def update_stats(self):
        if not self.version_service.current_project:
            if self.project_combo.currentIndex() >= 0:
                project_name = self.project_combo.currentText()
                self.version_service.load_project(project_name)
            else:
                return "错误: 没有加载项目"

        # Reload project info to get latest stats from disk
        project_name = self.version_service.current_project.get('name')
        project = self.version_service.load_project(project_name)

        if project:
            # Update UI labels
            self.stat_img_count.itemAt(1).widget().setText(str(project.get('image_count', 0)))
            self.stat_ver_count.itemAt(1).widget().setText(str(project.get('version_count', 0)))
            # Update category count
            categories = project.get('categories', {})
            self.stat_cls_count.itemAt(1).widget().setText(str(len(categories)))
    @action("version.refresh_versions", description="刷新当前项目的版本列表。\n- 无参数\n- 从磁盘重新扫描已导出的版本", category="版本管理", scope="ui")
    def refresh_versions(self):
        self.version_list.clear()
        project_name = self.version_service.current_project.get('name') if self.version_service.current_project else None
        if not project_name:
            return "错误: 没有加载项目"

        versions = self.version_service.scan_versions(project_name)
        for v in versions:
            item = QListWidgetItem(self.version_list)
            widget = VersionItemWidget(v["name"], f"创建于: {v['date']}")

            widget.open_requested.connect(self.open_version)
            widget.delete_requested.connect(self.delete_version_ui)

            item.setSizeHint(widget.sizeHint())
            self.version_list.addItem(item)
            self.version_list.setItemWidget(item, widget)
    @action("version.open_version", description="在系统资源管理器中打开指定版本的文件夹。\n- 无参数（基于当前选中的版本）\n- 方便用户查看导出的文件", category="版本管理", scope="ui")
    def open_version(self, version_name):
        if not self.version_service.current_project:
            return "错误: 没有加载项目"
        if not version_name:
            return "错误: 版本名称为空"
        project_name = self.version_service.current_project.get('name')
        path = self.version_service.get_version_path(project_name, version_name)
        if os.path.exists(path):
            os.startfile(path)
        else:
            return "错误: 版本文件夹不存在"
    @action("version.delete_version_ui", description="删除指定的版本及其所有导出文件（UI 包装）。\n- version_name: 要删除的版本名称\n- 弹出确认对话框，操作不可恢复", category="版本管理", params={"version_name": "str"}, scope="ui")
    def delete_version_ui(self, version_name):
        if not version_name:
            return "错误: 版本名称为空"
        if not self.version_service.current_project:
            return "错误: 没有加载项目"
        project_name = self.version_service.current_project.get('name')
        if not self.version_service.version_exists(project_name, version_name):
            return "错误: 版本不存在"
        msg = MessageBox(
            "确认删除",
            f"确定要删除版本 {version_name} 吗？此操作不可恢复。",
            self
        )
        if msg.exec():
            try:
                if self.version_service.delete_version_folder(project_name, version_name):
                    InfoBar.success("删除成功", f"版本 {version_name} 已删除", parent=self)
                    self.refresh_versions()
                    self.update_stats()
                else:
                    InfoBar.error("删除失败", "无法删除版本文件夹", parent=self)
            except Exception as e:
                InfoBar.error("删除失败", str(e), parent=self)
    @action("version.split_mode_changed", description="切换数据集划分模式。\n- 影响版本生成时的数据集划分方式", category="版本管理")

    def on_split_mode_changed(self, state):
        self.ratio_container.setEnabled(state == Qt.CheckState.Unchecked.value)

    @action("version.export_format_changed", description="切换版本导出的标注格式。\n- 可选格式取决于项目配置\n- 切换后会影响生成版本时的导出格式", category="版本管理")
    def on_export_format_changed(self, index):
        """导出格式变更时自动切换任务类型（仅对专用格式切换，多任务格式如coco不切换）"""
        fmt = self.export_format_combo.currentText()
        fmt_to_task_index = {
            "yolo": 0,      # 检测
            "yoloseg": 1,   # 分割
            "yoloobb": 2,   # 旋转
            "voc": 0,       # 检测
            "labelme": 0,   # 检测（默认）
        }
        task_index = fmt_to_task_index.get(fmt, None)
        if task_index is not None:
            self.export_task_type_combo.blockSignals(True)
            self.export_task_type_combo.setCurrentIndex(task_index)
            self.export_task_type_combo.blockSignals(False)
        self._update_export_format_options()
        self._update_task_type_hint()

    def on_export_task_type_changed(self, index):
        """导出任务类型变更时更新可选导出格式"""
        self._update_export_format_options()
        self._update_task_type_hint()

    def _update_export_format_options(self):
        """根据任务类型更新可选的导出格式"""
        task_type = self.export_task_type_combo.currentText()
        # 任务类型到专属格式和通用格式的映射
        task_format_map = {
            "检测": ["yolo", "coco", "voc", "labelme"],
            "分割": ["yoloseg", "coco", "voc", "labelme"],
            "旋转": ["yoloobb", "coco", "voc", "labelme"],
        }
        formats = task_format_map.get(task_type, ["coco", "voc", "labelme"])
        current_fmt = self.export_format_combo.currentText()

        self.export_format_combo.blockSignals(True)
        self.export_format_combo.clear()
        self.export_format_combo.addItems(formats)
        # 尝试保持当前选中的格式
        idx = self.export_format_combo.findText(current_fmt)
        if idx >= 0:
            self.export_format_combo.setCurrentIndex(idx)
        self.export_format_combo.blockSignals(False)

    def _update_task_type_hint(self):
        """更新任务类型提示标签"""
        fmt = self.export_format_combo.currentText()
        task_text = self.export_task_type_combo.currentText()
        fmt_hints = {
            "yolo": "YOLO格式",
            "yoloseg": "YOLO-Seg格式",
            "yoloobb": "YOLO-OBB格式",
            "coco": "COCO格式",
            "voc": "VOC格式",
            "labelme": "labelme格式",
        }
        task_hints = {
            "检测": "所有标注将转为水平矩形 (rectangle)",
            "分割": "所有标注将转为多边形 (polygon)",
            "旋转": "所有标注将转为旋转矩形 (rotation)",
        }
        hint = f"→ {fmt_hints.get(fmt, '')} | {task_hints.get(task_text, '')}"
        self.task_type_hint.setText(hint)

    def _build_example_copypaste_config(self):
        selected = self.version_service.example_selector.get_selected_examples()
        if selected:
            return {"Selected": selected, "Count": 3}
        return self.aug_example_copypaste.get_settings()

    @action("version.generate_version", description="基于当前项目数据生成一个标注版本。\n- 按照配置的导出格式和划分方式导出标注\n- 生成过程可能需要一些时间\n- 生成完成后版本列表会自动刷新", category="版本管理", scope="ui")
    def generate_version(self):
        from core.common.project_settings import project_settings
        # Proactively ensure project is loaded if combo has a selection
        if not self.version_service.current_project and self.project_combo.currentIndex() >= 0:
            self.version_service.load_project(self.project_combo.currentText())

        project = self.version_service.current_project
        if not project:
            InfoBar.error("提示", "请先在项目大厅选择一个项目", parent=self)
            return "错误: 请先在项目大厅选择一个项目"

        use_manual_split = self.use_existing_split.isChecked()
        ratios = (self.train_ratio.value(), self.val_ratio.value(), self.test_ratio.value())

        if not use_manual_split:
            if sum(ratios) != 1.0:
                InfoBar.warning("比例错误", "划分比例之和必须为 1.0", parent=self)
                return "错误: 划分比例之和必须为 1.0"

            # 检查是否存在已有的手动划分
            project_path = project['path']
            has_existing = False
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(project_path, split)
                if os.path.exists(split_path) and os.listdir(split_path):
                    has_existing = True
                    break

            if has_existing:
                msg = MessageBox(
                    "确认重新划分",
                    "检测到该项目已有手动划分的结果。如果继续随机划分，系统将覆盖之前的划分关系。是否继续？",
                    self
                )
                msg.yesButton.setText("覆盖并继续")
                msg.cancelButton.setText("取消")
                if not msg.exec():
                    return

        # 收集预处理设置
        preprocessing = {
            "resize": self.pre_resize.get_settings(),
            "auto_orient": self.pre_auto_orient.checkbox.isChecked()
        }

        # 收集增强设置
        augmentations = {
            "hsv": self.aug_hsv.get_settings(),
            "rotate": self.aug_rotate.get_settings(),
            "flip": self.aug_flip.checkbox.isChecked(),
            "blur": self.aug_blur.get_settings(),
            "copypaste": self.aug_copypaste.get_settings(),
            "example_copypaste": self._build_example_copypaste_config(),
        }

        # 构造导出配置
        export_config = {
            "project_name": project['name'],
            "use_manual_split": use_manual_split,
            "ratios": ratios if not use_manual_split else None,
            "preprocessing": preprocessing,
            "augmentations": augmentations,
            "roi_bbox": project_settings.get('roi'),
            "export_format": self.export_format_combo.currentText(),
            "export_task_type": ["det", "seg", "obb"][self.export_task_type_combo.currentIndex()],
            "export_unannotated": self.export_unannotated_check.isChecked()
        }

        # 创建进度对话框
        self.progress_dialog = ProgressDialog("正在导出版本...", "请稍候，正在处理并导出数据...", self)
        self.progress_dialog.show()

        # 创建工作线程
        self.export_thread = QThread()
        self.export_worker = VersionExportWorker(
            self.version_service,
            export_config
        )
        self.export_worker.moveToThread(self.export_thread)

        # 连接信号 - 使用 @Slot 装饰器的方法而不是 lambda
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.progress.connect(self.progress_dialog.setValue)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_worker.error.connect(self.on_export_error)

        # 清理 - 确保线程正确退出
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.error.connect(self.export_thread.quit)
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.export_thread.finished.connect(self.export_thread.deleteLater)

        self.export_thread.start()
        self._is_exporting = True
    @action("version.export_finished", description="版本导出完成后的处理。\n- 无参数\n- 内部回调，通常在 generate_version 完成后自动触发", category="版本管理")

    @Slot(dict)
    def on_export_finished(self, res):
        self._is_exporting = False
        self.progress_dialog.close()
        if res['status'] == 'success':
            InfoBar.success("导出成功", f"已生成版本 {res['version']}", parent=self)
            self.refresh_versions()
            self.update_stats()
        else:
            InfoBar.error("导出失败", res.get('message', '未知错误'), parent=self)
    @action("version.export_error", description="版本导出失败后的处理。\n- 无参数\n- 内部回调，通常在 generate_version 失败时自动触发", category="版本管理")

    @Slot(str)
    def on_export_error(self, error_msg):
        self._is_exporting = False
        self.progress_dialog.close()
        InfoBar.error("导出出错", error_msg, parent=self)
