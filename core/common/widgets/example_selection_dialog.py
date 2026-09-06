"""
统一的示例库选择对话框

同时满足标注插件和版本插件的示例选择需求。
各插件将自身数据转为统一格式后传入即可。
"""

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QCheckBox, QLabel)
from qfluentwidgets import (PrimaryPushButton, PushButton, SubtitleLabel, CaptionLabel,
                            StrongBodyLabel, CheckBox, TransparentToolButton,
                            FluentIcon as FIF, InfoBar, MessageBox)

from core.common.widgets.base import BaseDialog


class ExampleLibraryDialog(BaseDialog):
    """统一的示例库选择对话框

    适用于所有需要选择示例的场景，包括：
    - 标注插件的自动标注示例选择
    - 版本插件的 Copy-Paste 示例选择

    items 格式: [{"name": str, "category": str, ...}, ...]
    delete_callback: 可选，删除某个示例时的回调，签名 delete_callback(index, item)
                     如果不提供则不显示删除按钮
    """

    def __init__(self, items, parent=None, title="示例库 - 选择要使用的示例",
                 selected_names=None, delete_callback=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 500)
        self.items = items
        self._delete_callback = delete_callback
        self.selected_names = set(selected_names or [])
        self.selected_indices = []

        self.init_ui()
        self._update_select_all_state()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(SubtitleLabel("请勾选想要依赖的示例："))

        toolbar_layout = QHBoxLayout()

        self.checkbox_select_all = CheckBox("全选")
        self.checkbox_select_all.stateChanged.connect(self.on_select_all_changed)
        toolbar_layout.addWidget(self.checkbox_select_all)

        toolbar_layout.addStretch(1)

        self.btn_clear_all = PushButton(FIF.DELETE, "清空")
        self.btn_clear_all.setToolTip("删除所有示例")
        self.btn_clear_all.clicked.connect(self.on_clear_all_clicked)
        toolbar_layout.addWidget(self.btn_clear_all)

        layout.addLayout(toolbar_layout)

        self.list_widget = QListWidget()
        self.refresh_list()
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.btn_cancel = PushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = PrimaryPushButton("确定")
        self.btn_ok.clicked.connect(self.on_accept)

        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def refresh_list(self):
        self.list_widget.clear()
        for i, item in enumerate(self.items):
            row = QListWidgetItem()
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(5, 4, 5, 4)
            h_layout.setSpacing(10)

            # 示例缩略图（同步读取，列表量小）
            thumb_label = QLabel()
            thumb_label.setFixedSize(64, 48)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet(
                "QLabel { background-color: rgba(255, 255, 255, 0.04); border-radius: 4px; }"
            )
            image_path = item.get('image_path', '')
            if image_path and os.path.isfile(image_path):
                reader = QImageReader(image_path)
                reader.setAutoTransform(True)
                img = reader.read()
                if not img.isNull():
                    thumb_label.setPixmap(QPixmap.fromImage(img).scaled(
                        64, 48, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
            h_layout.addWidget(thumb_label)

            name = item.get('name', '未命名')
            category = item.get('category', '')
            label_parts = [name]
            if category:
                label_parts.append(f"[{category}]")
            if image_path and os.path.exists(image_path):
                label_parts.append(f"({os.path.basename(image_path)})")

            checkbox = QCheckBox(" ".join(label_parts))
            is_checked = self.selected_names and name in self.selected_names
            if not self.selected_names:
                is_checked = True
            checkbox.setChecked(is_checked)
            checkbox.setProperty("index", i)
            h_layout.addWidget(checkbox)

            h_layout.addStretch(1)

            if self._delete_callback is not None:
                del_btn = TransparentToolButton(FIF.DELETE, container)
                del_btn.setToolTip("删除该示例")
                del_btn.clicked.connect(lambda checked, idx=i: self.on_delete_clicked(idx))
                h_layout.addWidget(del_btn)

            row.setSizeHint(container.sizeHint())
            self.list_widget.addItem(row)
            self.list_widget.setItemWidget(row, container)

    def on_delete_clicked(self, index):
        if index < 0 or index >= len(self.items):
            return
        item = self.items[index]
        name = item.get('name', '')
        w = MessageBox("删除确认", f"确定要删除示例 '{name}' 吗？", self)
        if w.exec():
            if self._delete_callback:
                self._delete_callback(index, item)
            self.items.pop(index)
            self.selected_names.discard(name)
            self.refresh_list()

    def on_select_all_changed(self, state):
        is_select_all = (state == Qt.CheckState.Checked.value)
        for i in range(self.list_widget.count()):
            row_item = self.list_widget.item(i)
            container = self.list_widget.itemWidget(row_item)
            checkbox = container.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(is_select_all)

    def on_clear_all_clicked(self):
        if not self.items:
            return

        w = MessageBox("清空确认", f"确定要删除所有 {len(self.items)} 个示例吗？此操作不可恢复。", self)
        if w.exec():
            if self._delete_callback:
                for i in range(len(self.items) - 1, -1, -1):
                    self._delete_callback(i, self.items[i])
            self.items.clear()
            self.selected_names.clear()
            self.refresh_list()
            InfoBar.success("已清空", "所有示例已删除", parent=self)

    def on_accept(self):
        self.selected_indices = []
        for i in range(self.list_widget.count()):
            row_item = self.list_widget.item(i)
            container = self.list_widget.itemWidget(row_item)
            checkbox = container.findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                self.selected_indices.append(checkbox.property("index"))

        if not self.selected_indices:
            InfoBar.warning("提示", "请至少选择一个示例", parent=self)
            return

        self.accept()

    def get_selected_examples(self):
        return [self.items[i] for i in self.selected_indices]

    def get_selected_names(self):
        return [self.items[i].get('name', '') for i in self.selected_indices]

    def _update_select_all_state(self):
        all_checked = True
        for i in range(self.list_widget.count()):
            row_item = self.list_widget.item(i)
            container = self.list_widget.itemWidget(row_item)
            checkbox = container.findChild(QCheckBox)
            if checkbox and not checkbox.isChecked():
                all_checked = False
                break
        self.checkbox_select_all.blockSignals(True)
        self.checkbox_select_all.setChecked(all_checked)
        self.checkbox_select_all.blockSignals(False)
