"""
标注插件对话框类

包含：类别选择、转换确认、添加自定义模型、模型管理对话框。
示例库管理 (ExampleLibraryDialog) 和 提示词管理 (PromptLibraryDialog) 已迁移到 core/common/widgets/。
"""

import os
import re

from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QFileDialog,
                             QDialog, QWidget, QLabel, QTabWidget,
                             QListWidgetItem, QPushButton)
from qfluentwidgets import (PrimaryPushButton, PushButton, TransparentToolButton,
                            SubtitleLabel, CaptionLabel, BodyLabel, LineEdit,
                            FluentIcon as FIF, IconWidget,
                            InfoBar, ListWidget, CheckBox, MessageBox,
                            MessageBoxBase, ComboBox)

from core.common.widgets.base import BaseDialog, ProgressDialog
from core.backend.path_resolver import get_custom_models, save_custom_models
from core.backend.core import ModelFactory
from core.common.settings import settings


class CategorySelectionDialog(BaseDialog):
    def __init__(self, categories, current_category=None, parent=None, recent_order=None):
        super().__init__(parent)
        self.setWindowTitle("选择类别")
        self.setMinimumWidth(300)
        self.selected_category = None

        layout = QVBoxLayout(self)
        self.list_widget = ListWidget()

        if recent_order:
            recent_cats = [c for c in recent_order if c in categories]
            remaining_cats = sorted([c for c in categories if c not in recent_order])
            ordered_cats = recent_cats + remaining_cats
        else:
            ordered_cats = sorted(categories.keys())

        for cat in ordered_cats:
            item = QListWidgetItem(cat)
            color = categories[cat]
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))
            self.list_widget.addItem(item)
            if cat == current_category:
                self.list_widget.setCurrentItem(item)

        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.ok_btn = PrimaryPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = PushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def on_item_double_clicked(self, item):
        self.accept()

    def get_selected_category(self):
        item = self.list_widget.currentItem()
        return item.text() if item else None


class ConversionDialog(MessageBox):
    def __init__(self, title, content, parent=None, mode='seg'):
        super().__init__(title, content, parent)
        self.yesButton.setText("AI 转换 (精准)")
        self.cancelButton.setText("取消 (不转换)")

        self.geo_button = PushButton("直接转换 (几何)", self)
        self.buttonLayout.insertWidget(1, self.geo_button)

        self.result_type = "cancel"

        self.yesButton.clicked.connect(self._on_ai_clicked)
        self.geo_button.clicked.connect(self._on_geo_clicked)
        self.cancelButton.clicked.connect(self.reject)

        if mode == 'seg':
            self.geo_button.setText("直接转换 (矩形)")
        else:
            self.geo_button.setText("直接转换 (几何)")

    def _on_ai_clicked(self):
        self.result_type = "agent"
        self.accept()

    def _on_geo_clicked(self):
        self.result_type = "geo"
        self.accept()


class AddCustomModelDialog(BaseDialog):
    def __init__(self, model_role: str = "auto", parent=None):
        super().__init__(parent)
        self.model_role = model_role
        self.setWindowTitle("添加自定义模型")
        self.setMinimumWidth(450)
        self.result_data = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title_label = SubtitleLabel("添加自定义模型")
        title_label.setStyleSheet("font-size: 16px; font-weight: 600; background: transparent;")
        layout.addWidget(title_label)

        desc_label = CaptionLabel("配置自定义模型的参数，添加到可用模型列表")
        desc_label.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 12px; background: transparent;")
        layout.addWidget(desc_label)

        if model_role == "interactive":
            categories = ModelFactory.get_interactive_categories()
        elif model_role == "refine":
            categories = ModelFactory.get_refine_categories()
        elif model_role == "plain":
            categories = ModelFactory.get_plain_categories()
        else:
            categories = ModelFactory.get_auto_categories()

        category_label = CaptionLabel("模型类别")
        category_label.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.7); background: transparent;")
        layout.addWidget(category_label)
        self.category_combo = ComboBox()
        self.category_combo.addItems(categories)
        if categories:
            self.category_combo.setCurrentIndex(0)
        self.category_combo.setFixedHeight(34)
        layout.addWidget(self.category_combo)

        name_label = CaptionLabel("模型名称")
        name_label.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.7); background: transparent;")
        layout.addWidget(name_label)
        self.name_edit = LineEdit()
        self.name_edit.setPlaceholderText("输入模型名称，不可与已有模型重名")
        self.name_edit.setFixedHeight(34)
        layout.addWidget(self.name_edit)

        path_label = CaptionLabel("权重文件路径")
        path_label.setStyleSheet("font-size: 12px; color: rgba(255, 255, 255, 0.7); background: transparent;")
        layout.addWidget(path_label)
        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)
        self.path_edit = LineEdit()
        self.path_edit.setPlaceholderText("选择权重文件 (.pt/.pth/.onnx/.engine)")
        self.path_edit.setFixedHeight(34)
        self.browse_btn = PushButton(FIF.FOLDER, " 浏览")
        self.browse_btn.setFixedHeight(34)
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.clicked.connect(self._browse_file)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        layout.addStretch(1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self.cancel_btn = PushButton("取消")
        self.cancel_btn.setFixedSize(80, 36)
        self.cancel_btn.setStyleSheet("font-size: 13px;")
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = PrimaryPushButton("确定")
        self.ok_btn.setFixedSize(80, 36)
        self.ok_btn.setStyleSheet("font-size: 13px;")
        self.ok_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
        layout.addLayout(btn_layout)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择权重文件", "",
            "模型文件 (*.pt *.pth *.onnx *.engine);;所有文件 (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _on_accept(self):
        name = self.name_edit.text().strip()
        category = self.category_combo.currentText()
        weight_path = self.path_edit.text().strip()

        if not name:
            InfoBar.warning("提示", "模型名称不能为空", parent=self)
            return

        all_models = (ModelFactory.get_interactive_models() +
                      ModelFactory.get_auto_models() +
                      ModelFactory.get_plain_models() +
                      ModelFactory.get_refine_models())
        if name in all_models:
            InfoBar.warning("提示", f"模型名称 '{name}' 已存在，请使用其他名称", parent=self)
            return

        if not weight_path or not os.path.exists(weight_path):
            InfoBar.warning("提示", "权重文件路径无效或文件不存在", parent=self)
            return

        self.result_data = {
            "name": name,
            "category": category,
            "weight_path": weight_path,
        }
        self.accept()


class ModelSelectionDialog(BaseDialog):
    def __init__(self, model_role: str = "auto", parent=None):
        super().__init__(parent)
        self.model_role = model_role
        role_titles = {"interactive": "管理交互模型", "auto": "管理示例模型", "plain": "管理普通模型", "refine": "管理细化模型"}
        self.setWindowTitle(role_titles.get(model_role, "管理模型"))
        self.setMinimumSize(560, 520)
        self._selected_model = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title_label = SubtitleLabel(self.windowTitle())
        title_label.setStyleSheet("font-size: 16px; font-weight: 600; background: transparent;")
        layout.addWidget(title_label)

        role_descs = {
            "interactive": "选择要在菜单中显示的模型，或管理自定义模型",
            "auto": "选择要在菜单中显示的示例模型，或管理自定义模型",
            "plain": "选择要在菜单中显示的普通模型，或管理自定义模型",
            "refine": "选择要在菜单中显示的细化模型，或管理自定义模型",
        }
        desc_text = role_descs.get(model_role, "选择要在菜单中显示的模型")
        desc_label = CaptionLabel(desc_text)
        desc_label.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 12px; background: transparent;")
        layout.addWidget(desc_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: transparent;
                padding: 8px 16px;
                margin: 0 2px;
                border-bottom: 2px solid transparent;
                color: rgba(255, 255, 255, 0.6);
                font-size: 13px;
            }
            QTabBar::tab:selected {
                color: #0078d4;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover {
                color: rgba(255, 255, 255, 0.9);
            }
        """)

        builtin_tab = QWidget()
        builtin_layout = QVBoxLayout(builtin_tab)
        builtin_layout.setContentsMargins(0, 8, 0, 0)
        builtin_layout.setSpacing(8)

        builtin_toolbar = QHBoxLayout()
        builtin_toolbar.setSpacing(12)
        self.builtin_select_all = CheckBox("全选")
        self.builtin_select_all.setChecked(True)
        self.builtin_select_all.setStyleSheet("font-size: 13px; background: transparent;")
        self.builtin_select_all.stateChanged.connect(self._on_builtin_select_all)
        builtin_toolbar.addWidget(self.builtin_select_all)
        builtin_toolbar.addStretch(1)

        count_label = CaptionLabel()
        count_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 11px; background: transparent;")
        self._builtin_count_label = count_label
        builtin_toolbar.addWidget(count_label)
        builtin_layout.addLayout(builtin_toolbar)

        self.builtin_list = ListWidget()
        self.builtin_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                padding: 6px 8px;
            }
            QListWidget::item:selected {
                background: rgba(0, 120, 212, 0.15);
            }
            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.05);
            }
        """)
        builtin_layout.addWidget(self.builtin_list)

        self.tab_widget.addTab(builtin_tab, "内置模型")

        custom_tab = QWidget()
        custom_layout = QVBoxLayout(custom_tab)
        custom_layout.setContentsMargins(0, 8, 0, 0)
        custom_layout.setSpacing(8)

        self.custom_list = ListWidget()
        self.custom_list.setMouseTracking(True)  # itemEntered 需要鼠标追踪
        self.custom_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                border: none;
            }
            QListWidget::item:selected {
                background: rgba(0, 120, 212, 0.15);
            }
            QListWidget::item:hover {
                background: transparent;
            }
        """)
        self.custom_list.itemEntered.connect(self._on_custom_item_entered)
        self.custom_list.installEventFilter(self)
        custom_layout.addWidget(self.custom_list)

        self.tab_widget.addTab(custom_tab, "自定义模型")

        layout.addWidget(self.tab_widget, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.cancel_btn = PushButton("取消")
        self.cancel_btn.setFixedSize(80, 36)
        self.cancel_btn.setStyleSheet("font-size: 13px;")
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = PrimaryPushButton("确定")
        self.ok_btn.setFixedSize(80, 36)
        self.ok_btn.setStyleSheet("font-size: 13px;")
        self.ok_btn.clicked.connect(self._on_accept)
        bottom_layout.addWidget(self.cancel_btn)
        bottom_layout.addWidget(self.ok_btn)
        layout.addLayout(bottom_layout)

        self._refresh_builtin_list()
        self._refresh_custom_list()

    def _refresh_builtin_list(self):
        self.builtin_list.clear()
        if self.model_role == "interactive":
            all_models = ModelFactory.get_interactive_models()
        elif self.model_role == "refine":
            all_models = ModelFactory.get_refine_models()
        elif self.model_role == "plain":
            all_models = ModelFactory.get_plain_models()
        else:
            all_models = ModelFactory.get_auto_models()
        builtin = [m for m in all_models if m not in ModelFactory._custom_weight_paths]

        settings_key_map = {"interactive": "enabled_interactive_models", "refine": "enabled_refine_models", "plain": "enabled_plain_models", "auto": "enabled_auto_models"}
        settings_key = settings_key_map.get(self.model_role, "enabled_auto_models")
        enabled = settings.get(settings_key, None)
        if enabled is None:
            enabled = builtin[:]

        enabled_count = len([m for m in builtin if m in enabled])
        self._builtin_count_label.setText(f"已选 {enabled_count}/{len(builtin)}")

        for i, name in enumerate(builtin):
            item = QListWidgetItem()
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(8, 6, 8, 6)
            h_layout.setSpacing(12)

            checkbox = CheckBox()
            checkbox.setChecked(name in enabled)
            checkbox.setProperty("model_name", name)
            checkbox.setStyleSheet("background: transparent;")
            def on_checkbox_changed(state, item_ref=item):
                data = item_ref.data(Qt.ItemDataRole.UserRole)
                data["enabled"] = (state == Qt.CheckState.Checked.value)
                item_ref.setData(Qt.ItemDataRole.UserRole, data)
                self._update_select_all_state()
            checkbox.stateChanged.connect(on_checkbox_changed)
            h_layout.addWidget(checkbox)

            icon_label = QLabel()
            icon_label.setFixedSize(28, 28)
            icon_label.setStyleSheet("""
                QLabel {
                    background: rgba(0, 120, 212, 0.15);
                    border-radius: 6px;
                    color: #0078d4;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setText(name[0].upper() if name else "M")
            h_layout.addWidget(icon_label)

            info_layout = QVBoxLayout()
            info_layout.setSpacing(2)
            info_layout.setContentsMargins(0, 0, 0, 0)

            name_label = BodyLabel(name)
            name_label.setStyleSheet("font-size: 13px; background: transparent;")
            info_layout.addWidget(name_label)

            type_label = CaptionLabel("内置模型")
            type_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 11px; background: transparent;")
            info_layout.addWidget(type_label)

            h_layout.addLayout(info_layout, 1)

            item.setSizeHint(container.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, {"name": name, "enabled": name in enabled})
            self.builtin_list.addItem(item)
            self.builtin_list.setItemWidget(item, container)

        self._update_select_all_state()

    def _refresh_custom_list(self):
        self.custom_list.clear()

        # 顶部"假模型行"：与模型行同构（相同容器/边距/sizeHint 计算方式），
        # 保证左右对齐与行高一致；新添加的模型出现在它下方
        add_container = QWidget()
        add_container.setStyleSheet("background: transparent;")
        add_layout = QHBoxLayout(add_container)
        add_layout.setContentsMargins(8, 6, 8, 6)
        add_layout.setSpacing(12)
        add_layout.addWidget(self._build_add_custom_row())

        add_item = QListWidgetItem()
        add_item.setSizeHint(add_container.sizeHint())
        add_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.custom_list.addItem(add_item)
        self.custom_list.setItemWidget(add_item, add_container)

        custom_models = get_custom_models(self.model_role)
        for i, info in enumerate(custom_models):
            item = QListWidgetItem()
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(8, 6, 8, 6)
            h_layout.setSpacing(12)

            icon_label = QLabel()
            icon_label.setFixedSize(28, 28)
            icon_label.setStyleSheet("""
                QLabel {
                    background: rgba(16, 185, 129, 0.15);
                    border-radius: 6px;
                    color: #10b981;
                    font-size: 12px;
                    font-weight: bold;
                }
            """)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setText(info['name'][0].upper() if info.get('name') else "C")
            h_layout.addWidget(icon_label)

            info_layout = QVBoxLayout()
            info_layout.setSpacing(2)
            info_layout.setContentsMargins(0, 0, 0, 0)

            name_label = BodyLabel(info['name'])
            name_label.setStyleSheet("font-size: 13px; background: transparent;")
            info_layout.addWidget(name_label)

            category_label = CaptionLabel(info['category'])
            category_label.setStyleSheet("color: rgba(255, 255, 255, 0.4); font-size: 11px; background: transparent;")
            info_layout.addWidget(category_label)

            h_layout.addLayout(info_layout, 1)

            del_btn = TransparentToolButton(FIF.DELETE, container)
            del_btn.setFixedSize(24, 24)
            del_btn.setToolTip("删除该模型")
            del_btn.setVisible(False)  # 悬浮该行时显示
            del_btn.setStyleSheet("""
                TransparentToolButton {
                    border-radius: 4px;
                    background: transparent;
                }
                TransparentToolButton:hover {
                    background: rgba(255, 80, 80, 0.15);
                }
            """)
            del_btn.clicked.connect(lambda checked, idx=i: self._on_delete_custom_at(idx))
            h_layout.addWidget(del_btn)

            item.setSizeHint(container.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, info)
            self.custom_list.addItem(item)
            self.custom_list.setItemWidget(item, container)

    def _build_add_custom_row(self):
        """假模型添加行按钮：透明虚线框，居中加号图标 + 文字说明"""
        button = QPushButton("  添加自定义模型")
        button.setObjectName("AddCustomModelRow")
        button.setIcon(FIF.ADD.icon())
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(44)
        button.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.03);
                border: 1px dashed rgba(255, 255, 255, 0.22);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.55);
                font-size: 13px;
                padding: 8px;
            }
            QPushButton:hover {
                background: rgba(0, 120, 212, 0.10);
                border-color: rgba(0, 120, 212, 0.55);
                color: #4da2ff;
            }
            QPushButton:pressed {
                background: rgba(0, 120, 212, 0.18);
            }
        """)
        button.clicked.connect(self._on_add_custom)
        return button

    def _on_custom_item_entered(self, item):
        """悬浮某行时仅显示该行的删除图标按钮"""
        self._set_delete_buttons_visible(item)

    def _set_delete_buttons_visible(self, active_item):
        for i in range(self.custom_list.count()):
            widget = self.custom_list.itemWidget(self.custom_list.item(i))
            if widget is None:
                continue
            btn = widget.findChild(TransparentToolButton)
            if btn is not None:
                btn.setVisible(self.custom_list.item(i) is active_item)

    def eventFilter(self, obj, event):
        """鼠标离开自定义模型列表时隐藏所有删除按钮"""
        if obj is self.custom_list and event.type() == QEvent.Type.Leave:
            self._set_delete_buttons_visible(None)
        return super().eventFilter(obj, event)

    def _on_builtin_select_all(self, state):
        checked = state == Qt.CheckState.Checked.value
        for i in range(self.builtin_list.count()):
            item = self.builtin_list.item(i)
            container = self.builtin_list.itemWidget(item)
            checkbox = container.findChild(CheckBox)
            if checkbox:
                checkbox.blockSignals(True)
                checkbox.setChecked(checked)
                checkbox.blockSignals(False)
            data = item.data(Qt.ItemDataRole.UserRole)
            data["enabled"] = checked
            item.setData(Qt.ItemDataRole.UserRole, data)
        self._update_builtin_count()

    def _update_select_all_state(self):
        all_checked = True
        for i in range(self.builtin_list.count()):
            item = self.builtin_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data.get("enabled", True):
                all_checked = False
                break
        self.builtin_select_all.blockSignals(True)
        self.builtin_select_all.setChecked(all_checked)
        self.builtin_select_all.blockSignals(False)
        self._update_builtin_count()

    def _update_builtin_count(self):
        enabled_count = 0
        total_count = self.builtin_list.count()
        for i in range(total_count):
            item = self.builtin_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data.get("enabled", True):
                enabled_count += 1
        self._builtin_count_label.setText(f"已选 {enabled_count}/{total_count}")

    def _on_accept(self):
        enabled = []
        for i in range(self.builtin_list.count()):
            item = self.builtin_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data.get("enabled", True):
                enabled.append(data["name"])

        settings_key_map = {"interactive": "enabled_interactive_models", "refine": "enabled_refine_models", "plain": "enabled_plain_models", "auto": "enabled_auto_models"}
        settings_key = settings_key_map.get(self.model_role, "enabled_auto_models")
        settings.set(settings_key, enabled)

        rebuild_map = {
            "interactive": "_rebuild_interactive_model_menu",
            "refine": "_rebuild_refine_model_menu",
            "plain": "_rebuild_plain_model_menu",
            "auto": "_rebuild_auto_model_menu",
        }
        rebuild_method = rebuild_map.get(self.model_role)
        if rebuild_method and hasattr(self.parent(), rebuild_method):
            getattr(self.parent(), rebuild_method)()

        self.accept()

    def _on_add_custom(self):
        dialog = AddCustomModelDialog(self.model_role, self)
        if dialog.exec_() == QDialog.DialogCode.Accepted and dialog.result_data:
            data = dialog.result_data

            ModelFactory.register_custom_model(data["name"], data["category"], data["weight_path"], role=self.model_role)

            models = get_custom_models(self.model_role)
            models.append(data)
            save_custom_models(self.model_role, models)

            self._refresh_custom_list()

            rebuild_map = {
                "interactive": "_rebuild_interactive_model_menu",
                "auto": "_rebuild_auto_model_menu",
                "plain": "_rebuild_plain_model_menu",
                "refine": "_rebuild_refine_model_menu",
            }
            rebuild_method = rebuild_map.get(self.model_role)
            if rebuild_method and hasattr(self.parent(), rebuild_method):
                getattr(self.parent(), rebuild_method)()

            InfoBar.success("成功", f"已添加自定义模型: {data['name']}", parent=self)

    def _on_delete_custom_at(self, index):
        models = get_custom_models(self.model_role)
        if index >= len(models):
            return
        info = models[index]

        ModelFactory.unregister_custom_model(info["name"], role=self.model_role)

        models = [m for m in models if m["name"] != info["name"]]
        save_custom_models(self.model_role, models)

        self._refresh_custom_list()

        rebuild_map = {
            "interactive": "_rebuild_interactive_model_menu",
            "auto": "_rebuild_auto_model_menu",
            "plain": "_rebuild_plain_model_menu",
            "refine": "_rebuild_refine_model_menu",
        }
        rebuild_method = rebuild_map.get(self.model_role)
        if rebuild_method and hasattr(self.parent(), rebuild_method):
            getattr(self.parent(), rebuild_method)()

        InfoBar.success("成功", f"已删除自定义模型: {info['name']}", parent=self)

    def get_selected_model(self):
        return self._selected_model


class CustomSizeFilterDialog(MessageBoxBase):
    """自定义大小范围筛选对话框（供搜索框 @大小 引用使用）

    支持两种范围：
    - 面积范围：如 100-5000（像素²），生成 token "@大小:面积100-5000"
    - 宽高范围：如 100-800x50-600（像素），生成 token "@大小:宽高100-800x50-600"
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result_token = None

        self.titleLabel = SubtitleLabel("自定义大小范围")
        self.viewLayout.addWidget(self.titleLabel)

        self.type_combo = ComboBox()
        self.type_combo.addItems(["面积范围 (像素²)", "宽高范围 (像素)"])
        self.viewLayout.addWidget(self.type_combo)

        self.area_edit = LineEdit()
        self.area_edit.setPlaceholderText("面积范围，如 100-5000")
        self.viewLayout.addWidget(self.area_edit)

        self.wh_edit = LineEdit()
        self.wh_edit.setPlaceholderText("宽高范围，如 100-800x50-600")
        self.wh_edit.setVisible(False)
        self.viewLayout.addWidget(self.wh_edit)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

        self.yesButton.setText("确定")
        self.cancelButton.setText("取消")
        widget = getattr(self, "widget", None)
        if widget is not None:
            widget.setMinimumWidth(380)

    def _on_type_changed(self, index):
        self.area_edit.setVisible(index == 0)
        self.wh_edit.setVisible(index == 1)

    def validate(self) -> bool:
        """点击确定时校验输入，通过则生成 @大小 token"""
        if self.type_combo.currentIndex() == 0:
            text = self.area_edit.text().strip()
            match = re.fullmatch(r"(\d+)-(\d+)", text)
            if not match:
                InfoBar.warning("格式错误", "面积范围格式应为 最小-最大，如 100-5000", parent=self)
                return False
            self.result_token = f"@大小:面积{match.group(1)}-{match.group(2)}"
        else:
            text = self.wh_edit.text().strip()
            match = re.fullmatch(r"(\d+)-(\d+)x(\d+)-(\d+)", text)
            if not match:
                InfoBar.warning("格式错误", "宽高范围格式应为 宽最小-宽最大x高最小-高最大，如 100-800x50-600",
                                parent=self)
                return False
            self.result_token = (f"@大小:宽高{match.group(1)}-{match.group(2)}"
                                 f"x{match.group(3)}-{match.group(4)}")
        return True


class FitCheckDialog(BaseDialog):
    """贴合度检查结果应用对话框：勾选要替换为 hybrid 分割结果的标注。

    results 为 manual.fit_check_current 返回的 results 列表；
    已检出的标注默认勾选（可替换），未检出的禁用。
    selected_indices() 返回用户勾选的标注序号列表。
    """

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标注贴合度检查")
        self.setMinimumWidth(520)
        self._results = results or []

        layout = QVBoxLayout(self)
        title = SubtitleLabel("选择要替换为 Hybrid 分割结果的标注")
        layout.addWidget(title)

        self.list_widget = ListWidget()
        self._item_indices = []
        for r in self._results:
            label = str(r.get('label', ''))
            idx = r.get('idx', '')
            detected = bool(r.get('detected'))
            if detected:
                method = r.get('method', '')
                ar = r.get('area_ratio', '')
                iou = r.get('bbox_iou', '')
                off = r.get('centroid_offset', '')
                text = f"{label} #{idx}  [{method}]  area_ratio={ar}  bbox_iou={iou}  centroid_offset={off}"
            else:
                text = f"{label} #{idx}  [未检出]"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if detected else Qt.CheckState.Unchecked)
            if not detected:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._item_indices.append(idx)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        tip = CaptionLabel("提示：已检出的标注将用 hybrid 分割轮廓替换原标注框，替换后自动保存")
        layout.addWidget(tip)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self.cancel_btn = PushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        self.apply_btn = PrimaryPushButton("应用")
        self.apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)

    def _on_apply(self):
        self.accept()

    def selected_indices(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked and i < len(self._item_indices):
                selected.append(self._item_indices[i])
        return selected
