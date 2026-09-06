"""
统一的提示词管理对话框

适用于所有需要管理提示词的场景，包括标注插件、推理插件等。
各插件将自身提示词数据转为统一格式后传入即可。
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QCheckBox
from qfluentwidgets import (PrimaryPushButton, PushButton, SubtitleLabel, CaptionLabel,
                            StrongBodyLabel, CheckBox, TransparentToolButton, LineEdit,
                            FluentIcon as FIF, InfoBar, MessageBox,
                            MessageBoxBase)

from core.common.widgets.base import BaseDialog


class PromptLibraryDialog(BaseDialog):
    """统一的提示词管理对话框

    适用于所有需要管理提示词的场景。

    prompts 格式: [{"text": str, "checked": bool}, ...]
    对话框会直接修改传入的 prompts 列表。
    """

    def __init__(self, prompts, parent=None, title="提示词管理"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        self.prompts = prompts
        self.selected_indices = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(SubtitleLabel("请勾选要使用的提示词："))

        toolbar_layout = QHBoxLayout()

        self.checkbox_select_all = CheckBox("全选")
        self.checkbox_select_all.setChecked(True)
        self.checkbox_select_all.stateChanged.connect(self.on_select_all_changed)
        toolbar_layout.addWidget(self.checkbox_select_all)

        toolbar_layout.addStretch(1)

        self.btn_add = PushButton(FIF.ADD, "添加")
        self.btn_add.setToolTip("添加新提示词")
        self.btn_add.clicked.connect(self.on_add_clicked)
        toolbar_layout.addWidget(self.btn_add)

        self.btn_clear_all = PushButton(FIF.DELETE, "清空")
        self.btn_clear_all.setToolTip("删除所有提示词")
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
        for i, p in enumerate(self.prompts):
            item = QListWidgetItem()
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(5, 2, 5, 2)

            checkbox = QCheckBox(p.get('text', ''))
            checkbox.setChecked(p.get('checked', True))
            checkbox.setProperty("index", i)
            h_layout.addWidget(checkbox)

            h_layout.addStretch(1)

            del_btn = TransparentToolButton(FIF.DELETE, container)
            del_btn.setToolTip("删除该提示词")
            del_btn.clicked.connect(lambda checked, idx=i: self.on_delete_clicked(idx))
            h_layout.addWidget(del_btn)

            item.setSizeHint(container.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, container)

    def on_add_clicked(self):
        class AddPromptDialog(MessageBoxBase):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.titleLabel = StrongBodyLabel('添加新提示词')
                self.viewLayout.addWidget(self.titleLabel)

                self.edit = LineEdit()
                self.edit.setPlaceholderText('请输入提示词（例如: defect, scratch）')
                self.viewLayout.addWidget(self.edit)

                self.yesButton.setText('添加')
                self.cancelButton.setText('取消')

            def get_text(self):
                return self.edit.text().strip()

        dialog = AddPromptDialog(self)
        if dialog.exec():
            text = dialog.get_text()
            if text:
                self.prompts.append({'text': text, 'checked': True})
                self.refresh_list()

    def on_delete_clicked(self, index):
        prompt_text = self.prompts[index].get('text', '')
        w = MessageBox("删除确认", f"确定要删除提示词 '{prompt_text}' 吗？", self)
        if w.exec():
            self.prompts.pop(index)
            self.refresh_list()

    def on_select_all_changed(self, state):
        is_select_all = (state == Qt.CheckState.Checked.value)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            container = self.list_widget.itemWidget(item)
            checkbox = container.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(is_select_all)

    def on_clear_all_clicked(self):
        if not self.prompts:
            return

        w = MessageBox("清空确认", f"确定要删除所有 {len(self.prompts)} 个提示词吗？此操作不可恢复。", self)
        if w.exec():
            self.prompts.clear()
            self.refresh_list()
            InfoBar.success("已清空", "所有提示词已删除", parent=self)

    def on_accept(self):
        self.selected_indices = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            container = self.list_widget.itemWidget(item)
            checkbox = container.findChild(QCheckBox)
            if checkbox:
                idx = checkbox.property("index")
                if idx < len(self.prompts):
                    self.prompts[idx]['checked'] = checkbox.isChecked()
                if checkbox.isChecked():
                    self.selected_indices.append(idx)

        if not self.selected_indices:
            InfoBar.warning("提示", "请至少选择一个提示词", parent=self)
            return

        self.accept()

    def get_selected_prompts(self):
        return [self.prompts[i]['text'] for i in self.selected_indices]

    def get_all_prompts(self):
        return self.prompts
