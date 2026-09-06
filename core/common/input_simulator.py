"""
底层输入模拟 Action

提供鼠标/键盘的底层模拟操作，基于 QTest API。
"""

from typing import List, Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from core.common.action_registry import action


class InputSimulator:
    """底层输入模拟器"""

    def __init__(self, main_window=None):
        self._main_window = main_window

    def _get_focus_widget(self) -> Optional[QWidget]:
        """获取当前焦点 widget"""
        if self._main_window:
            return self._main_window.focusWidget() or self._main_window
        return None

    # ==================== 鼠标操作 ====================

    @action("input.mouse", description="统一鼠标操作入口（op 参数驱动）。\n- op=click: 在指定位置点击\n- op=dbl_click: 在指定位置双击\n- op=press: 按下不释放（与 release 配合模拟拖拽）\n- op=release: 释放鼠标按键\n- op=move: 移动到指定位置\n- op=drag: 从起点拖拽到终点\n- x, y: 点击/双击/按下/释放/移动的坐标（像素）\n- x1, y1, x2, y2: 拖拽的起点与终点坐标（像素）\n- button: 鼠标按钮，可选 left/right/middle（默认 left）\n- widget_name: 目标 widget 的 objectName（可选，默认焦点 widget）\n- 坐标为控件坐标，非图像坐标", category="输入模拟",
            params={"op": "str", "x": "int", "y": "int", "x1": "int", "y1": "int", "x2": "int", "y2": "int", "button": "str", "widget_name": "str"})
    def mouse(self, op: str = "click", x: int = 0, y: int = 0, x1: int = 0, y1: int = 0,
              x2: int = 100, y2: int = 100, button: str = "left", widget_name: str = ""):
        """统一鼠标操作入口 — 按 op 分发到不同鼠标模拟逻辑。"""
        def _click():
            if x < 0 or y < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                print(f"    [Error] 未找到目标 widget: {widget_name}")
                return "错误: 未找到目标 widget"
            btn = self._resolve_button(button)
            pos = QPoint(x, y)
            QTest.mouseClick(widget, btn, Qt.KeyboardModifier.NoModifier, pos)
            print(f"    鼠标点击: ({x}, {y}) {button} @ {widget.objectName() or widget.__class__.__name__}")
            return True

        def _dbl_click():
            if x < 0 or y < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                return "错误: 未找到目标 widget"
            btn = self._resolve_button(button)
            QTest.mouseDblClick(widget, btn, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
            print(f"    鼠标双击: ({x}, {y}) {button}")
            return True

        def _press():
            if x < 0 or y < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                return "错误: 未找到目标 widget"
            btn = self._resolve_button(button)
            QTest.mousePress(widget, btn, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
            print(f"    鼠标按下: ({x}, {y}) {button}")
            return True

        def _release():
            if x < 0 or y < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                return "错误: 未找到目标 widget"
            btn = self._resolve_button(button)
            QTest.mouseRelease(widget, btn, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
            print(f"    鼠标释放: ({x}, {y}) {button}")
            return True

        def _move():
            if x < 0 or y < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                return "错误: 未找到目标 widget"
            QTest.mouseMove(widget, QPoint(x, y))
            print(f"    鼠标移动: ({x}, {y})")
            return True

        def _drag():
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return "错误: 坐标不能为负数"
            widget = self._find_widget(widget_name)
            if not widget:
                return "错误: 未找到目标 widget"
            btn = self._resolve_button(button)
            QTest.mousePress(widget, btn, Qt.KeyboardModifier.NoModifier, QPoint(x1, y1))
            QTest.mouseMove(widget, QPoint(x2, y2))
            QTest.mouseRelease(widget, btn, Qt.KeyboardModifier.NoModifier, QPoint(x2, y2))
            print(f"    鼠标拖拽: ({x1}, {y1}) → ({x2}, {y2}) {button}")
            return True

        handlers = {
            "click": _click,
            "dbl_click": _dbl_click,
            "press": _press,
            "release": _release,
            "move": _move,
            "drag": _drag,
        }
        handler = handlers.get(op)
        if handler is None:
            return f"错误: 未知鼠标操作 '{op}'，可选: click/dbl_click/press/release/move/drag"
        return handler()

    # ==================== 键盘操作 ====================

    @action("input.key_press", description="模拟按下并释放一个键盘按键。\n- key: 按键名称，如 'enter', 'tab', 'escape', 'a', 'ctrl' 等\n- modifiers: 修饰键列表，如 ['ctrl', 'shift']\n- 按键名称参考 Qt.Key 枚举", category="输入模拟",
            params={"key": "str", "modifiers": "list", "widget_name": "str"})
    def key_press(self, key: str = "", modifiers: list = None, widget_name: str = ""):
        """
        模拟键盘按键

        Args:
            key: 按键名称 (如 "A", "Return", "Delete", "Ctrl+S")
            modifiers: 修饰键列表 (如 ["Ctrl", "Shift"])
            widget_name: 目标 widget
        """
        if not key:
            return "错误: 按键名称不能为空"
        widget = self._find_widget(widget_name)
        if not widget:
            return "错误: 未找到目标 widget"

        widget.setFocus()
        mod_flags = self._resolve_modifiers(modifiers or [])

        # 处理组合键 (如 "Ctrl+S")
        if "+" in key:
            parts = key.split("+")
            if len(parts) == 2:
                mod_flags = self._resolve_modifiers([parts[0]])
                qt_key = self._resolve_key(parts[1])
                QTest.keyClick(widget, qt_key, mod_flags)
                print(f"    按键: {key}")
                return True

        qt_key = self._resolve_key(key)
        QTest.keyClick(widget, qt_key, mod_flags)
        print(f"    按键: {key}")
        return True

    @action("input.key_type", description="模拟键盘逐字输入一段文本。\n- text: 要输入的文本内容\n- 支持普通字符和特殊符号\n- 输入前确保目标输入框已获得焦点", category="输入模拟",
            params={"text": "str", "widget_name": "str"})
    def key_type(self, text: str = "", widget_name: str = ""):
        """模拟键盘逐字符输入文本"""
        if not text:
            return "错误: 输入文本不能为空"
        widget = self._find_widget(widget_name)
        if not widget:
            return "错误: 未找到目标 widget"

        widget.setFocus()
        QTest.keyClicks(widget, text)
        print(f"    输入文本: {text[:20]}{'...' if len(text) > 20 else ''}")
        return True

    @action("input.key_shortcut", description="执行键盘快捷键组合。\n- keys: 快捷键组合列表，如 ['ctrl', 's'] 表示保存\n- 按顺序依次按下并释放所有按键\n- 常用快捷键: ctrl+c(复制), ctrl+v(粘贴), ctrl+s(保存)", category="输入模拟",
            params={"shortcut": "str", "widget_name": "str"})
    def key_shortcut(self, shortcut: str = "", widget_name: str = ""):
        """
        执行快捷键组合

        支持格式: "Ctrl+S", "Ctrl+Shift+Z", "Delete", "Escape" 等
        """
        if not shortcut:
            return "错误: 快捷键不能为空"
        widget = self._find_widget(widget_name)
        if not widget:
            return "错误: 未找到目标 widget"

        widget.setFocus()
        parts = shortcut.split("+")
        if len(parts) == 1:
            qt_key = self._resolve_key(parts[0])
            QTest.keyClick(widget, qt_key)
        else:
            mod_flags = self._resolve_modifiers(parts[:-1])
            qt_key = self._resolve_key(parts[-1])
            QTest.keyClick(widget, qt_key, mod_flags)

        print(f"    快捷键: {shortcut}")
        return True

    # ==================== 辅助方法 ====================

    def _find_widget(self, widget_name: str = "") -> Optional[QWidget]:
        """查找目标 widget"""
        if widget_name and self._main_window:
            widget = self._main_window.findChild(QWidget, widget_name)
            if widget:
                return widget
        return self._get_focus_widget() or self._main_window

    @staticmethod
    def _resolve_button(button: str) -> Qt.MouseButton:
        """解析鼠标按键名称"""
        mapping = {
            "left": Qt.MouseButton.LeftButton,
            "right": Qt.MouseButton.RightButton,
            "middle": Qt.MouseButton.MiddleButton,
        }
        return mapping.get(button.lower(), Qt.MouseButton.LeftButton)

    @staticmethod
    def _resolve_key(key: str) -> Qt.Key:
        """解析键盘按键名称"""
        key_map = {
            "return": Qt.Key.Key_Return,
            "enter": Qt.Key.Key_Return,
            "escape": Qt.Key.Key_Escape,
            "esc": Qt.Key.Key_Escape,
            "tab": Qt.Key.Key_Tab,
            "backtab": Qt.Key.Key_Backtab,
            "backspace": Qt.Key.Key_Backspace,
            "delete": Qt.Key.Key_Delete,
            "insert": Qt.Key.Key_Insert,
            "home": Qt.Key.Key_Home,
            "end": Qt.Key.Key_End,
            "pageup": Qt.Key.Key_PageUp,
            "pagedown": Qt.Key.Key_PageDown,
            "up": Qt.Key.Key_Up,
            "down": Qt.Key.Key_Down,
            "left": Qt.Key.Key_Left,
            "right": Qt.Key.Key_Right,
            "space": Qt.Key.Key_Space,
            "f1": Qt.Key.Key_F1, "f2": Qt.Key.Key_F2, "f3": Qt.Key.Key_F3,
            "f4": Qt.Key.Key_F4, "f5": Qt.Key.Key_F5, "f6": Qt.Key.Key_F6,
            "f7": Qt.Key.Key_F7, "f8": Qt.Key.Key_F8, "f9": Qt.Key.Key_F9,
            "f10": Qt.Key.Key_F10, "f11": Qt.Key.Key_F11, "f12": Qt.Key.Key_F12,
        }
        lower = key.lower()
        if lower in key_map:
            return key_map[lower]
        # 单字符
        if len(key) == 1:
            return Qt.Key(ord(key.upper()))
        return Qt.Key(key)

    @staticmethod
    def _resolve_modifiers(modifiers: list) -> Qt.KeyboardModifier:
        """解析修饰键"""
        mod_map = {
            "ctrl": Qt.KeyboardModifier.ControlModifier,
            "control": Qt.KeyboardModifier.ControlModifier,
            "shift": Qt.KeyboardModifier.ShiftModifier,
            "alt": Qt.KeyboardModifier.AltModifier,
            "meta": Qt.KeyboardModifier.MetaModifier,
        }
        result = Qt.KeyboardModifier.NoModifier
        for mod in modifiers:
            result |= mod_map.get(mod.lower(), Qt.KeyboardModifier.NoModifier)
        return result
