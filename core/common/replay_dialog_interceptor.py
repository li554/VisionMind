"""
回放对话框拦截器

回放时拦截弹出的对话框，等待 dialog.op（op=accept/reject/delete_item 等）
action 触发后填入录制值并自动确认/取消。让用户能看到对话框内容。

核心机制：
1. TestEngine 在执行可能弹对话框的 action 前，把后续连续的对话框操作步骤
   （dialog.op 及旧格式 dialog_accept, dialog_reject, dialog_delete_item 等）
   通过 set_pending_ops 传给 interceptor。
2. 当 action 内部调用 dialog.exec_() 阻塞事件循环时，interceptor 在对话框
   Show 事件中自动从 pending_ops 取出操作执行（填值、点击按钮）。
3. 对于链式操作（如 delete_item 弹出确认 MessageBox → accept），
   interceptor 在子对话框 Show 时继续消费 pending_ops，并在对话框关闭后
   检查是否有剩余操作需要在父对话框上执行。
"""

from PySide6.QtCore import QObject, QEvent, QTimer, Qt
from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton, QListWidget
from typing import Dict, List, Optional


class ReplayDialogInterceptor(QObject):
    """回放时对话框拦截器

    核心逻辑：
    1. 回放时检测 QDialog 的 Show 事件，记录弹出的对话框
    2. 当 dialog_accept action 被调用时，填入录制值并点击确认
    3. 当 dialog_reject action 被调用时，点击取消
    4. 对话框正常弹出，用户可看到内容，由 action 控制关闭时机
    5. 当 action 内部 exec_() 阻塞时，通过 pending_ops 自动执行后续对话框操作
    """

    # 对话框操作 action 集合（含旧录制格式名称，兼容历史测试文件）
    DIALOG_OP_ACTIONS = {
        'dialog.op',
        'dialog_accept', 'dialog_reject',
        'dialog_delete_item', 'dialog_clear_all', 'dialog_add_item',
    }

    # 旧录制格式 action 名 → op 映射（兼容历史测试文件）
    LEGACY_OP_MAP = {
        'dialog_accept': 'accept',
        'dialog_reject': 'reject',
        'dialog_delete_item': 'delete_item',
        'dialog_clear_all': 'clear_all',
        'dialog_add_item': 'add_item',
    }

    def __init__(self, auto_accept_delay: int = 500, parent=None):
        """
        Args:
            auto_accept_delay: 点击确认按钮前的延迟（毫秒），让用户能看到填入过程
        """
        super().__init__(parent)
        self._auto_accept_delay = auto_accept_delay
        self._current_action = ""
        self._current_params: Dict = {}
        self._active = False
        self._active_dialogs: List[QDialog] = []  # 当前显示的对话框列表（栈结构）
        self._current_dialog_type = ""  # 当前等待处理的对话框类型
        # pending_ops: TestEngine 预读的后续对话框操作步骤
        # 当 action 内部 exec_() 阻塞时，interceptor 自动消费这些操作
        self._pending_ops: List[Dict] = []

    def activate(self):
        """激活拦截器，安装到 QApplication"""
        if not self._active:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().installEventFilter(self)
            self._active = True

    def deactivate(self):
        """停用拦截器，从 QApplication 移除"""
        if self._active:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().removeEventFilter(self)
            self._active = False
            self._active_dialogs.clear()
            self._pending_ops.clear()

    def set_current_action(self, action_name: str, params: Dict):
        """设置当前正在回放的 action 及其参数"""
        self._current_action = action_name
        self._current_params = params or {}

    def clear_current_action(self):
        """清除当前 action 信息"""
        self._current_action = ""
        self._current_params = {}

    def set_pending_ops(self, ops: List[Dict]):
        """设置待执行的对话框操作队列

        由 TestEngine 在执行可能弹对话框的 action 前调用。
        interceptor 在对话框 Show 时自动消费这些操作。

        Args:
            ops: 操作列表，每项格式 {'action': str, 'params': dict, 'description': str}
        """
        self._pending_ops = list(ops)  # 拷贝

    # ==================== 事件过滤器 ====================

    def eventFilter(self, obj, event) -> bool:
        """事件过滤器：检测 QDialog 的 Show 事件"""
        if not self._active:
            return False

        if event.type() == QEvent.Type.Show and isinstance(obj, QDialog):
            if obj not in self._active_dialogs:
                self._active_dialogs.append(obj)
                obj.finished.connect(lambda: self._on_dialog_finished(obj))
            # 如果有 pending ops，自动执行下一个
            if self._pending_ops:
                self._auto_execute_next_op(obj)

        return False

    def _on_dialog_finished(self, dialog):
        """对话框关闭时处理"""
        if dialog in self._active_dialogs:
            self._active_dialogs.remove(dialog)

        # 如果还有 pending ops 和活跃对话框（如子对话框关闭后回到父对话框），
        # 继续在父对话框上执行剩余操作
        if self._pending_ops and self._active_dialogs:
            parent_dialog = self._active_dialogs[-1]
            QTimer.singleShot(self._auto_accept_delay,
                              lambda: self._auto_execute_next_op(parent_dialog))

    # ==================== pending_ops 自动执行 ====================

    def _auto_execute_next_op(self, dialog: QDialog):
        """自动执行 pending_ops 中的下一个操作"""
        if not self._pending_ops or not dialog.isVisible():
            return

        op = self._pending_ops.pop(0)
        action = op.get('action', '')
        params = op.get('params', {})
        desc = op.get('description', '')
        print(f"    [Replay] 自动执行对话框操作: {action} ({desc})")

        QTimer.singleShot(self._auto_accept_delay,
                          lambda: self._do_dialog_op(dialog, action, params))

    def _do_dialog_op(self, dialog: QDialog, action: str, params: Dict):
        """在指定对话框上执行操作"""
        if not dialog.isVisible():
            return

        # 统一入口 dialog.op 由 params.op 指定操作；
        # 旧录制格式由 action 名推断 op
        op = params.get('op', '') or self.LEGACY_OP_MAP.get(action, '')
        if not op:
            print(f"    [Replay] 未知对话框操作: {action}")
            return

        try:
            if op == 'accept':
                self._current_params = params
                self._fill_values(dialog)
                self._do_accept(dialog)
            elif op == 'reject':
                self._do_reject(dialog)
            elif op == 'delete_item':
                item = params.get('item', params.get('example_name', ''))
                self._click_delete_button_for_item(dialog, item)
            elif op == 'clear_all':
                self._click_button_by_text(dialog, ["清空"])
            elif op == 'add_item':
                self._click_button_by_text(dialog, ["添加"])
            else:
                print(f"    [Replay] 未知对话框操作: {op}")
        except Exception as e:
            print(f"    [Replay] 对话框操作执行失败: {action} - {e}")

    # ==================== 由 dialog_accept/dialog_reject action 调用 ====================

    def accept_current_dialog(self, dialog_type: str = ""):
        """由 dialog_accept action 调用：确认当前弹出的对话框

        Args:
            dialog_type: 对话框类型标识
        """
        if not self._active or not self._active_dialogs:
            return

        dialog = self._active_dialogs[-1]  # 取最后弹出的对话框
        if not dialog.isVisible():
            return

        self._current_dialog_type = dialog_type

        try:
            # 1. 填入录制参数值
            self._fill_values(dialog)

            # 2. 延迟后点击确认按钮
            QTimer.singleShot(self._auto_accept_delay, lambda: self._do_accept(dialog))
        except Exception as e:
            print(f"[ReplayDialogInterceptor] 确认对话框失败: {e}")
            try:
                dialog.accept()
            except:
                pass

    def reject_current_dialog(self, dialog_type: str = ""):
        """由 dialog_reject action 调用：取消当前弹出的对话框

        Args:
            dialog_type: 对话框类型标识
        """
        if not self._active or not self._active_dialogs:
            return

        dialog = self._active_dialogs[-1]
        if not dialog.isVisible():
            return

        try:
            QTimer.singleShot(self._auto_accept_delay, lambda: self._do_reject(dialog))
        except Exception as e:
            print(f"[ReplayDialogInterceptor] 取消对话框失败: {e}")
            try:
                dialog.reject()
            except:
                pass

    def click_dialog_button(self, button_type: str, identifier: str = ""):
        """由对话框内部操作 action 调用：在当前对话框中点击指定按钮

        Args:
            button_type: 按钮类型（"delete", "clear_all", "add"）
            identifier: 标识符（如示例名称、提示词文本）
        """
        if not self._active or not self._active_dialogs:
            return

        dialog = self._active_dialogs[-1]
        if not dialog.isVisible():
            return

        try:
            if button_type == "clear_all":
                self._click_button_by_text(dialog, ["清空"])
            elif button_type == "add":
                self._click_button_by_text(dialog, ["添加"])
            elif button_type == "delete":
                self._click_delete_button_for_item(dialog, identifier)
        except Exception as e:
            print(f"[ReplayDialogInterceptor] 点击对话框按钮失败: {e}")

    # ==================== 内部实现 ====================

    def _fill_values(self, dialog: QDialog):
        """根据录制参数填入对话框的输入值"""
        params = self._current_params
        if not params:
            return

        # 1. 填入 LineEdit（输入对话框）
        line_edits = [le for le in dialog.findChildren(QLineEdit) if le.isVisible()]
        if line_edits:
            # 标准 key 优先匹配
            standard_keys = ["example_name", "category_name", "new_name", "name"]
            standard_values = [params.get(k) for k in standard_keys if params.get(k)]

            if standard_values:
                # 用标准 key 填值（按顺序填入 LineEdit）
                for le, val in zip(line_edits, standard_values):
                    if val:
                        le.setText(str(val))
            else:
                # 通用填值：把 params 中非 "dialog_type" 的值按顺序填入 LineEdit
                # 适用于 AutoInteractionRecorder 自动收集的 text_0, combo_0 等 key
                other_values = [v for k, v in params.items()
                                if k != "dialog_type" and v and not isinstance(v, (list, dict))]
                for le, val in zip(line_edits, other_values):
                    le.setText(str(val))

        # 2. 选中 ListWidget 项（选择对话框）
        list_widgets = dialog.findChildren(QListWidget)
        if list_widgets:
            category = params.get("category", "")
            if category:
                for list_widget in list_widgets:
                    items = list_widget.findItems(category, Qt.MatchFlag.MatchExactly)
                    if items:
                        list_widget.setCurrentItem(items[0])
                        break

    def _do_accept(self, dialog: QDialog):
        """点击确认按钮"""
        if not dialog.isVisible():
            return

        # 优先点击"确定"/"是"按钮
        if not self._click_accept_button(dialog):
            dialog.accept()

    def _do_reject(self, dialog: QDialog):
        """点击取消按钮"""
        if not dialog.isVisible():
            return

        # 优先点击"取消"/"否"按钮
        if not self._click_reject_button(dialog):
            dialog.reject()

    def _click_accept_button(self, dialog: QDialog) -> bool:
        """查找并点击确认按钮"""
        buttons = dialog.findChildren(QPushButton)
        accept_keywords = ["确定", "是", "OK", "Yes", "确认", "应用", "Apply", "保存"]

        for btn in buttons:
            text = btn.text().strip()
            if any(kw in text for kw in accept_keywords):
                btn.click()
                return True
        return False

    def _click_reject_button(self, dialog: QDialog) -> bool:
        """查找并点击取消按钮"""
        buttons = dialog.findChildren(QPushButton)
        reject_keywords = ["取消", "否", "Cancel", "No", "关闭", "Close"]

        for btn in buttons:
            text = btn.text().strip()
            if any(kw in text for kw in reject_keywords):
                btn.click()
                return True
        return False

    def _click_button_by_text(self, dialog: QDialog, keywords: list) -> bool:
        """查找并点击包含指定关键词的按钮"""
        buttons = dialog.findChildren(QPushButton)
        for btn in buttons:
            text = btn.text().strip()
            if any(kw in text for kw in keywords):
                btn.click()
                return True
        return False

    def _click_delete_button_for_item(self, dialog: QDialog, identifier: str):
        """在列表对话框中找到指定项的删除按钮并点击

        通过 QListWidget 中的项文本匹配 identifier，然后点击该行的删除按钮。
        """
        from PySide6.QtWidgets import QToolButton

        list_widgets = dialog.findChildren(QListWidget)
        if not list_widgets:
            return

        for list_widget in list_widgets:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                container = list_widget.itemWidget(item)
                if not container:
                    continue

                # 检查该项的文本是否包含 identifier
                from PySide6.QtWidgets import QCheckBox
                checkboxes = container.findChildren(QCheckBox)
                for cb in checkboxes:
                    if identifier and identifier in cb.text():
                        # 找到了对应的项，点击其删除按钮
                        del_buttons = container.findChildren(QToolButton)
                        if del_buttons:
                            QTimer.singleShot(self._auto_accept_delay, lambda b=del_buttons[0]: b.click())
                            return
                        # 回退：查找 ToolButton 类型的按钮
                        all_buttons = container.findChildren(QPushButton)
                        for btn in all_buttons:
                            if "删除" in btn.toolTip() or "delete" in btn.toolTip().lower():
                                QTimer.singleShot(self._auto_accept_delay, lambda b=btn: b.click())
                                return

    @property
    def is_active(self) -> bool:
        return self._active
