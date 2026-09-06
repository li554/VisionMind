"""
自动交互录制器

基于 Qt 信号/事件拦截的通用自动录制机制。
自动录制所有 QDialog 的交互（确认/取消/内部按钮），
以及 ComboBox 切换等 UI 操作，零侵入，不需要修改任何业务代码。

核心原理：
1. QApplication 事件过滤器检测 QDialog Show 事件
2. 自动连接 accepted/rejected 信号
3. 自动收集对话框内控件值（LineEdit, ComboBox, ListWidget 等）
4. 自动识别按钮语义（确定/取消/删除/清空等）
5. 与 @action 机制去重，避免重复录制

关键设计：
- dialog accepted/rejected 使用 QTimer.singleShot(0) 延迟录制，
  确保触发 action 先被录制，对话框响应后录制
- 每个步骤包含 timestamp 字段记录绝对时间
"""

import time
from PySide6.QtCore import QObject, QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QListWidget, QSpinBox, QDoubleSpinBox, QToolButton,
    QWidget,
)
from typing import Dict, Set, Optional


class AutoInteractionRecorder(QObject):
    """自动交互录制器

    使用方式：
    - 录制开始时调用 activate()
    - 录制结束时调用 deactivate()
    - 自动拦截所有 QDialog 交互，无需手动录制代码
    """

    # 确认按钮关键词（这些按钮的 clicked 信号由 accepted/rejected 处理，不重复录制）
    ACCEPT_KEYWORDS = {"确定", "是", "ok", "yes", "确认", "应用", "apply", "保存", "添加"}
    # 取消按钮关键词
    REJECT_KEYWORDS = {"取消", "否", "cancel", "no", "关闭", "close"}
    # 操作按钮关键词（需要单独录制的对话框内部操作）
    ACTION_KEYWORDS = {"删除", "清空", "delete", "clear", "remove"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self._tracked_dialogs: Set[int] = set()  # 已追踪的对话框 id
        self._action_recorded_steps: Set[str] = set()  # 本轮已通过 @action 录制的步骤，用于去重

    def activate(self):
        """激活自动录制"""
        if not self._active:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().installEventFilter(self)
            self._active = True

    def deactivate(self):
        """停用自动录制"""
        if self._active:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().removeEventFilter(self)
            self._active = False
            self._tracked_dialogs.clear()

    def notify_action_recorded(self, action_name: str):
        """通知 @action 已录制的步骤，用于去重

        当 @action wrapper 自动录制了某个操作后调用此方法，
        AutoInteractionRecorder 在相同操作时不再重复录制。
        """
        self._action_recorded_steps.add(action_name)

    def clear_action_records(self):
        """清除去重记录（每个步骤执行后调用）"""
        self._action_recorded_steps.clear()

    # ==================== 事件过滤器 ====================

    def eventFilter(self, obj, event) -> bool:
        if not self._active:
            return False

        # 检测 QDialog Show 事件
        if event.type() == QEvent.Type.Show and isinstance(obj, QDialog):
            dialog_id = id(obj)
            if dialog_id not in self._tracked_dialogs:
                self._tracked_dialogs.add(dialog_id)
                self._on_dialog_shown(obj)
                # 对话框关闭时清理
                obj.finished.connect(lambda: self._on_dialog_finished(obj))

        return False

    # ==================== 对话框事件处理 ====================

    def _on_dialog_shown(self, dialog: QDialog):
        """对话框显示时自动连接信号"""
        # 1. 连接 accepted/rejected 信号
        dialog.accepted.connect(lambda: self._on_dialog_accepted(dialog))
        dialog.rejected.connect(lambda: self._on_dialog_rejected(dialog))

        # 2. 连接对话框内操作按钮的 clicked 信号
        self._connect_dialog_buttons(dialog)

    def _on_dialog_finished(self, dialog: QDialog):
        """对话框关闭时清理"""
        dialog_id = id(dialog)
        self._tracked_dialogs.discard(dialog_id)

    def _connect_dialog_buttons(self, dialog: QDialog):
        """连接对话框内所有操作按钮的 clicked 信号"""
        # QPushButton
        for btn in dialog.findChildren(QPushButton):
            text = btn.text().strip().lower()
            # 跳过确认/取消按钮（由 accepted/rejected 处理）
            if any(kw in text for kw in self.ACCEPT_KEYWORDS | self.REJECT_KEYWORDS):
                continue
            # 连接操作按钮
            btn.clicked.connect(
                lambda checked, b=btn, d=dialog: self._on_action_button_clicked(d, b)
            )

        # QToolButton（如列表项中的删除按钮）
        for btn in dialog.findChildren(QToolButton):
            btn.clicked.connect(
                lambda checked, b=btn, d=dialog: self._on_tool_button_clicked(d, b)
            )

    # ==================== 录制逻辑 ====================

    def _on_dialog_accepted(self, dialog: QDialog):
        """对话框确认时自动录制（延迟到下一个事件循环，确保触发 action 先录制）"""
        dialog_type = type(dialog).__name__
        if self._is_action_duplicate(f"dialog_accept_{dialog_type}"):
            return

        # 延迟录制：等当前事件处理完毕（@action wrapper 可能还没录制触发 action）
        QTimer.singleShot(0, lambda: self._do_record_dialog_accept(dialog, dialog_type))

    def _do_record_dialog_accept(self, dialog: QDialog, dialog_type: str):
        """延迟执行：录制 dialog_accept"""
        # 再次检查录制状态（可能录制已停止）
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if not recorder.is_recording:
            return

        # 自动收集对话框内所有输入控件的值
        values = self._collect_dialog_values(dialog)

        # 将收集的值合并到 params
        params = {"op": "accept", "dialog_type": dialog_type}
        params.update(values)

        # 生成描述
        title = dialog.windowTitle() or dialog_type
        value_desc = ""
        if values:
            value_items = list(values.items())[:2]
            value_desc = " → " + ", ".join(f"{v}" for _, v in value_items)

        recorder.record_step(
            action_name="dialog.op",
            params=params,
            description=f"确认对话框: {title}{value_desc}",
            expected_result="操作执行成功"
        )

    def _on_dialog_rejected(self, dialog: QDialog):
        """对话框取消时自动录制（延迟到下一个事件循环）"""
        dialog_type = type(dialog).__name__
        if self._is_action_duplicate(f"dialog_reject_{dialog_type}"):
            return

        QTimer.singleShot(0, lambda: self._do_record_dialog_rejected(dialog, dialog_type))

    def _do_record_dialog_rejected(self, dialog: QDialog, dialog_type: str):
        """延迟执行：录制 dialog_reject"""
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if not recorder.is_recording:
            return

        title = dialog.windowTitle() or dialog_type
        recorder.record_step(
            action_name="dialog.op",
            params={"op": "reject", "dialog_type": dialog_type},
            description=f"取消对话框: {title}"
        )

    def _on_action_button_clicked(self, dialog: QDialog, button: QPushButton):
        """对话框内操作按钮点击自动录制"""
        text = button.text().strip()
        if not text:
            return

        # 推断操作类型
        op = self._infer_action_from_button(text)
        if not op:
            return

        # 收集按钮上下文参数
        context = self._collect_button_context(dialog, button, text)
        context["op"] = op

        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if recorder.is_recording:
            title = dialog.windowTitle() or type(dialog).__name__
            recorder.record_step(
                action_name="dialog.op",
                params=context,
                description=f"{title}: {text}",
                expected_result=f"执行{text}操作"
            )

    def _on_tool_button_clicked(self, dialog: QDialog, button: QToolButton):
        """对话框内工具按钮点击自动录制（如列表项中的删除按钮）"""
        tooltip = button.toolTip() or ""
        text = button.text().strip()

        # 识别删除按钮
        if "删除" in tooltip or "删除" in text or "delete" in tooltip.lower():
            # 尝试从父 widget 中获取标识信息
            parent_widget = button.parentWidget()
            item_info = self._get_item_info_from_container(parent_widget)

            from core.common.action_recorder import ActionRecorder
            recorder = ActionRecorder.instance()
            if recorder.is_recording:
                params = {"op": "delete_item", "dialog_type": type(dialog).__name__}
                if item_info:
                    params["item"] = item_info

                title = dialog.windowTitle() or type(dialog).__name__
                recorder.record_step(
                    action_name="dialog.op",
                    params=params,
                    description=f"{title}: 删除 {item_info or '项目'}",
                    expected_result="弹出删除确认"
                )

    # ==================== 参数收集 ====================

    def _collect_dialog_values(self, dialog: QDialog) -> Dict:
        """自动收集对话框内所有输入控件的值

        Returns:
            字典 {控件标识: 值}
        """
        values = {}

        # LineEdit
        for le in dialog.findChildren(QLineEdit):
            if not le.isVisible():
                continue
            text = le.text().strip()
            if not text:
                continue
            # 用 objectName 或 placeholderText 作为 key
            key = le.objectName()
            if not key or key.startswith("qt_"):
                key = le.placeholderText().replace("请输入", "").replace("，", "").strip()
            if not key:
                key = f"text_{len(values)}"
            # 避免路径键过长
            if len(key) > 30:
                key = key[:30]
            values[key] = text

        # ComboBox
        for cb in dialog.findChildren(QComboBox):
            if not cb.isVisible():
                continue
            key = cb.objectName()
            if not key or key.startswith("qt_"):
                key = f"combo_{len(values)}"
            values[key] = cb.currentText()

        # CheckBox（列表项中的勾选状态）
        checkboxes = dialog.findChildren(QCheckBox)
        checked_items = []
        for cb in checkboxes:
            if cb.isChecked() and cb.isVisible():
                text = cb.text().strip()
                if text:
                    checked_items.append(text)
        if checked_items:
            values["selected"] = checked_items

        # SpinBox
        for sb in dialog.findChildren(QSpinBox) + dialog.findChildren(QDoubleSpinBox):
            if not sb.isVisible():
                continue
            key = sb.objectName() or f"spin_{len(values)}"
            values[key] = sb.value()

        return values

    def _collect_button_context(self, dialog: QDialog, button: QPushButton, text: str) -> Dict:
        """收集按钮点击的上下文参数"""
        context = {"dialog_type": type(dialog).__name__}

        # 尝试从按钮的 parent 获取行项目信息
        parent = button.parentWidget()
        item_info = self._get_item_info_from_container(parent)
        if item_info:
            context["item"] = item_info

        return context

    def _get_item_info_from_container(self, container: QWidget) -> str:
        """从列表项的容器 widget 中获取标识信息"""
        if not container:
            return ""
        # 查找 CheckBox（通常包含项目名称）
        checkboxes = container.findChildren(QCheckBox)
        for cb in checkboxes:
            text = cb.text().strip()
            if text:
                return text
        # 查找 Label
        from PySide6.QtWidgets import QLabel
        labels = container.findChildren(QLabel)
        for label in labels:
            text = label.text().strip()
            if text:
                return text
        return ""

    # ==================== 辅助方法 ====================

    def _infer_action_from_button(self, text: str) -> str:
        """从按钮文本推断对话框操作类型（对应 dialog.op 的 op 参数）"""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ("清空", "clear")):
            return "clear_all"
        if any(kw in text_lower for kw in ("删除", "delete", "remove")):
            return "delete_item"
        if any(kw in text_lower for kw in ("添加", "add")):
            return "add_item"
        if any(kw in text_lower for kw in ("刷新", "refresh")):
            return "refresh"
        if any(kw in text_lower for kw in ("全选", "select_all")):
            return "select_all"
        # 未识别的操作按钮，仍然录制
        return f"unknown"

    def _is_action_duplicate(self, key: str) -> bool:
        """检查是否已被 @action 录制"""
        return key in self._action_recorded_steps

    @property
    def is_active(self) -> bool:
        return self._active
