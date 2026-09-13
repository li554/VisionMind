"""
Agent 对话侧栏 — VisionMind Agent 交互界面

基于 docs/agent_dialog_prototype.html 原型的 PySide6 实现，
作为现有 chat_sidebar.py 的功能升级版（API 向下兼容并扩展新增能力）。

公共 API：
    - 兼容 ChatSidebar: message_sent / debug_requested 信号
    - 兼容方法: add_message / start_assistant_message / append_to_assistant /
                finalize_assistant / show_thinking / clear_messages
    - 新增信号: stop_requested
    - 新增方法: add_tool_call / update_tool_status / update_exec_status /
                show_exec_status / hide_exec_status / set_model / add_session
"""

import math
import os
import re
from typing import List, Optional

import markdown as _md

from PySide6.QtCore import (
    Qt, Signal, QPropertyAnimation, QEasingCurve, QTimer, QPoint, QRect, QSize,
    QEvent, QFile, QTextStream
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, QTextCharFormat, QSyntaxHighlighter,
    QTextDocument, QPalette, QLinearGradient
)
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QToolButton,
    QScrollArea, QWidget, QSizePolicy, QPushButton, QPlainTextEdit,
    QGraphicsDropShadowEffect, QComboBox, QFileDialog, QMenu, QInputDialog,
    QListWidgetItem
)
from qfluentwidgets import FluentIcon as FIF, TransparentToolButton, IconWidget

from core.common.theme_manager import theme_manager
from core.common.icons import AppIcon
from core.common.settings import settings
from core.common.config import Config


# ============================================================================
# 辅助函数
# ============================================================================

def _break_long_tokens(text: str, chunk: int = 12, threshold: int = 20) -> str:
    """给超长无空白 token（如 Windows 路径、base64）插入零宽空格断行机会。

    QLabel 富文本不会在长 token 中间断行，longest token 会把最小宽度撑到
    超出侧栏宽度，整条消息溢出屏幕右侧。U+200B 是 Qt 文本引擎认可的
    断行机会，视觉不可见（代价是复制文本会带上零宽字符）。
    """
    if not text:
        return text
    zwsp = "\u200b"

    def _insert(m):
        token = m.group(0)
        return zwsp.join(token[i:i + chunk] for i in range(0, len(token), chunk))

    return re.sub(r"[^\s<>]{%d,}" % threshold, _insert, text)


def _md_to_html(text: str) -> str:
    """将 Markdown 文本转换为 HTML"""
    if not text:
        return ""
    # 转换前打断长 token（此时还是纯文本,不会破坏 HTML 标签）
    text = _break_long_tokens(text)
    try:
        html = _md.markdown(
            text,
            extensions=["fenced_code", "codehilite", "nl2br", "tables"],
        )
        # 为代码块添加基本样式
        html = html.replace(
            "<pre>",
            '<pre style="background:rgba(0,0,0,0.25);padding:8px;border-radius:6px;overflow-x:auto;font-size:11px;">'
        )
        html = html.replace(
            "<code>",
            '<code style="background:rgba(0,0,0,0.15);padding:1px 4px;border-radius:3px;font-size:11px;">'
        )
        html = html.replace(
            "</pre><code>",
            "</pre>"
        )
        html = html.replace(
            "<table>",
            '<table style="border-collapse:collapse;width:100%;font-size:11px;">'
        )
        html = html.replace(
            "<td>",
            '<td style="border:1px solid rgba(255,255,255,0.15);padding:4px 8px;">'
        )
        html = html.replace(
            "<th>",
            '<th style="border:1px solid rgba(255,255,255,0.15);padding:4px 8px;font-weight:600;">'
        )
        # 添加 blockquote 样式
        html = html.replace(
            "<blockquote>",
            '<blockquote style="border-left:3px solid #2563eb;padding:4px 12px;margin:8px 0;color:#9ca3af;">'
        )
        return html
    except Exception:
        return text.replace("\n", "<br>")


def _strip_markdown(text: str) -> str:
    """粗略去除 Markdown 标记，用于预览文本"""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"!?\[.*?\]\(.*?\)", "", t)
    return t.strip()


class AutoHeightLabel(QLabel):
    """按当前宽度精确自适应高度的富文本标签。

    QLabel 对富文本 + 自动换行的 sizeHint 采用试探式估算：首次布局时
    宽度尚未落定，会得到严重偏大的高度，等后续重排才缩回——表现为
    消息框先巨大、agent 开始回复后才慢慢收缩。这里在文本/尺寸变化时
    把高度直接钉在 heightForWidth(当前宽度)，布局每轮都拿到确定值。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setText(self, text: str):
        super().setText(text)
        self._sync_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_height()

    def _sync_height(self):
        w = self.width()
        if w <= 0:
            return
        h = self.heightForWidth(w)
        if h > 0 and h != self.height():
            self.setFixedHeight(h)


# ============================================================================
# 辅助：JSON 语法高亮
# ============================================================================

class JsonHighlighter(QSyntaxHighlighter):
    """简单的 JSON 语法高亮器"""

    def __init__(self, document: QTextDocument):
        super().__init__(document)
        self._key_fmt = QTextCharFormat()
        self._key_fmt.setForeground(QColor("#c792ea"))
        self._string_fmt = QTextCharFormat()
        self._string_fmt.setForeground(QColor("#c3e88d"))
        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#f78c6c"))
        self._bool_fmt = QTextCharFormat()
        self._bool_fmt.setForeground(QColor("#ff5370"))

    def highlightBlock(self, text: str):
        import re
        for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:', text):
            f = QTextCharFormat(self._key_fmt)
            self.setFormat(m.start(1), m.end(1) - m.start(1), f)
        for m in re.finditer(r':\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text):
            f = QTextCharFormat(self._string_fmt)
            self.setFormat(m.start(1), m.end(1) - m.start(1), f)
        for m in re.finditer(r':\s*(-?\d+\.?\d*)', text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), self._number_fmt)
        for m in re.finditer(r':\s*(true|false|null)', text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), self._bool_fmt)


def _json_to_html(text: str) -> str:
    """将 JSON 文本转为带语法颜色高亮的 HTML"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # 先打断长 token 再包 span,避免零宽空格误插入 style 属性破坏标签
    text = _break_long_tokens(text)
    text = re.sub(
        r'(:\s*)(-?\d+\.?\d*)',
        r'\1<span style="color:#f78c6c;">\2</span>', text
    )
    text = re.sub(
        r'(:\s*)(true|false|null)',
        r'\1<span style="color:#ff5370;">\2</span>', text
    )
    text = re.sub(
        r'(:\s*)"((?:[^"\\]|\\.)*)"',
        r'\1<span style="color:#c3e88d;">"\2"</span>', text
    )
    text = re.sub(
        r'("(?:[^"\\]|\\.)*")(\s*:)',
        r'<span style="color:#c792ea;">\1</span>\2', text
    )
    # QLabel 富文本对 white-space:pre-wrap 支持不可靠，改为显式换行 + 硬折行
    text = text.replace("\n", "<br/>")
    return (
        f'<div style="font-family:JetBrains Mono, monospace;'
        f'font-size:11px;margin:0;word-wrap:break-word;">{text}</div>'
    )


# ============================================================================
# 左侧拖拽调整宽度的把手
# ============================================================================

class ResizeHandle(QFrame):
    """左侧 4px 宽的拖拽把手，用于调整 AgentDialog 宽度"""

    width_changed = Signal(int)
    resize_started = Signal()
    resize_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ResizeHandle")
        self.setFixedWidth(6)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self._dragging = False
        self._start_x = 0
        self._start_width = 0

        self._badge = QLabel(self)
        self._badge.setObjectName("ResizeBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedSize(56, 22)
        self._badge.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_x = event.globalPosition().toPoint().x()
            dialog = self.window() if False else self.parent()
            if dialog is not None:
                self._start_width = dialog.width()
            self.setProperty("active", True)
            self.setStyleSheet(self.styleSheet())
            self._show_badge(self._start_width)
            self.resize_started.emit()
            event.accept()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        delta = self._start_x - event.globalPosition().toPoint().x()
        new_width = self._start_width + delta
        new_width = max(340, min(700, new_width))
        self.width_changed.emit(new_width)
        self._show_badge(new_width)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.setProperty("active", False)
            self.setStyleSheet(self.styleSheet())
            self._badge.hide()
            dialog = self.parent()
            final = dialog.width() if dialog else 420
            self.resize_finished.emit(final)
            event.accept()

    def _show_badge(self, width_px: int):
        self._badge.setText(f"{width_px}px")
        x = (self.width() - self._badge.width()) // 2
        y = (self.parent().height() - self._badge.height()) // 2 if self.parent() else 100
        self._badge.move(max(0, x), max(0, y))
        self._badge.raise_()
        self._badge.show()


# ============================================================================
# 旋转 Spinner
# ============================================================================

class Spinner(QLabel):
    """旋转的加载圈"""

    def __init__(self, parent=None, size: int = 14, color: str = "#2563eb"):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._color = QColor(color)
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.translate(self.width() / 2, self.height() / 2)
        p.rotate(self._angle)
        pen = QPen(QBrush(self._color), 2)
        p.setPen(pen)
        rect = QRect(-self.width() / 2 + 2, -self.height() / 2 + 2,
                     self.width() - 4, self.height() - 4)
        p.drawArc(rect, 0, 270 * 16)
        p.end()


# ============================================================================
# 思考指示器（三点弹跳动画）
# ============================================================================

class ThinkingIndicator(QFrame):
    """三点弹跳动画"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ThinkingIndicator")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        avatar = IconWidget(AppIcon.AGENT)
        avatar.setFixedSize(26, 26)
        avatar.setObjectName("AssistantAvatar")
        layout.addWidget(avatar)

        dots_frame = QFrame()
        dots_frame.setObjectName("ThinkingDots")
        dots_layout = QHBoxLayout(dots_frame)
        dots_layout.setContentsMargins(14, 10, 14, 10)
        dots_layout.setSpacing(4)
        self._dots = []
        for i in range(3):
            d = QLabel(dots_frame)
            d.setFixedSize(6, 6)
            dots_layout.addWidget(d)
            self._dots.append(d)
        layout.addWidget(dots_frame)
        layout.addStretch()

        self._anim_offsets = [0.0, 0.0, 0.0]
        self._step = 0
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._step = 0
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._step += 1
        for i in range(3):
            phase = (self._step + i * 3) % 14
            if phase < 4:
                offset = -phase
            elif phase < 8:
                offset = phase - 8
            else:
                offset = 0
            self._dots[i].setStyleSheet(
                f"background-color: #6b7280; border-radius: 3px; "
                f"margin-bottom: {-offset}px;"
            )


# ============================================================================
# 工具步骤行
# ============================================================================

class ToolStepItem(QFrame):
    """单条工具执行步骤"""

    def __init__(self, index: int, label: str, status: str = "pending", parent=None):
        super().__init__(parent)
        self.setObjectName("ToolStepItem")
        self._status = status
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self._num = QLabel(f"{index}.")
        self._num.setObjectName("StepNum")
        layout.addWidget(self._num)

        self._label = QLabel(label)
        self._label.setObjectName("StepLabel")
        self._label.setWordWrap(True)
        layout.addWidget(self._label, 1)

        self._status_label = QLabel()
        self._status_label.setObjectName("StepStatus")
        layout.addWidget(self._status_label)

        self.set_status(status)

    def set_status(self, status: str):
        self._status = status
        icons = {"done": "✓", "active": "⟳", "pending": "○"}
        self._status_label.setText(icons.get(status, "○"))
        self._status_label.setProperty("status", status)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)
        self.setProperty("active", status == "active")
        self.setProperty("done", status == "done")
        self.style().unpolish(self)
        self.style().polish(self)
        if status == "done":
            # 完成步骤降低显示对比度
            self.setStyleSheet(
                "QFrame#ToolStepItem { opacity: 1; background: rgba(255,255,255,0.02); }"
                "QLabel { color: rgba(156, 163, 175, 0.55); }"
            )
        else:
            self.setStyleSheet("")


# ============================================================================
# 工具调用块（可折叠卡片）
# ============================================================================

class ToolCallBlock(QFrame):
    """可折叠的工具调用展示卡片"""

    def __init__(self, tool_name: str, status: str = "running", duration: str = "",
                 input_data: str = "", output_data: str = "",
                 steps: Optional[List[dict]] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolCallBlock")
        self._tool_name = tool_name
        self._status = status
        self._duration = duration
        self._expanded = (status != "success")
        self._steps_data = steps
        self._input_data = input_data
        self._output_data = output_data

        is_plan = "plan" in tool_name.lower()
        # 有输出数据时不自动折叠
        self._expanded = self._expanded or bool(output_data)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = QFrame()
        self._header.setObjectName("ToolCallHeader")
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)

        self._status_icon = QLabel()
        self._status_icon.setObjectName("ToolStatusIcon")
        self._status_icon.setFixedSize(18, 18)
        self._status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self._status_icon)

        display_name = tool_name
        if is_plan:
            display_name = "📋 执行计划"
        self._name_label = QLabel(display_name)
        self._name_label.setObjectName("ToolName")
        header_layout.addWidget(self._name_label)
        header_layout.addStretch()

        self._duration_label = QLabel(duration)
        self._duration_label.setObjectName("ToolDuration")
        if not duration:
            self._duration_label.hide()
        header_layout.addWidget(self._duration_label)

        self._chevron = QLabel("▸")
        self._chevron.setObjectName("ToolChevron")
        header_layout.addWidget(self._chevron)

        layout.addWidget(self._header)

        # Body
        self._body = QFrame()
        self._body.setObjectName("ToolCallBody")
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 10)
        body_layout.setSpacing(4)

        self._has_content = bool(steps) or bool(input_data) or bool(output_data)

        if self._has_content:
            content = QFrame()
            content.setObjectName("ToolCallContent")
            self._content_layout = QVBoxLayout(content)
            self._content_layout.setContentsMargins(12, 6, 12, 0)
            self._content_layout.setSpacing(6)

            if steps:
                steps_frame = QFrame()
                steps_frame.setObjectName("ToolSteps")
                sl = QVBoxLayout(steps_frame)
                sl.setContentsMargins(0, 0, 0, 0)
                sl.setSpacing(3)
                self._step_items: List[ToolStepItem] = []
                for i, s in enumerate(steps):
                    item = ToolStepItem(i + 1, s.get("label", ""), s.get("status", "pending"))
                    sl.addWidget(item)
                    self._step_items.append(item)
                self._content_layout.addWidget(steps_frame)
            else:
                if input_data:
                    lbl1 = QLabel("输入参数")
                    lbl1.setObjectName("ToolSectionLabel")
                    self._content_layout.addWidget(lbl1)
                    self._input_view = self._make_code_view(input_data)
                    self._content_layout.addWidget(self._input_view)
                if output_data:
                    lbl2 = QLabel("执行结果")
                    lbl2.setObjectName("ToolSectionLabel")
                    self._content_layout.addWidget(lbl2)
                    self._output_view = self._make_code_view(output_data)
                    self._content_layout.addWidget(self._output_view)

            body_layout.addWidget(content)
        else:
            self._content_layout = None

        layout.addWidget(self._body)

        self._anim = None
        self._header.mousePressEvent = self._on_header_click
        self._apply_status(status)
        self._apply_expanded(self._expanded, animate=False)

    @staticmethod
    def _format_code_text(text: str) -> str:
        stripped = text.strip()
        # 仅当是合法 JSON 时做美化高亮；非 JSON（如图片结构化信息
        # [图片: ...]、纯文本/markdown 结果）回退为 markdown 渲染+自动换行
        try:
            import json as _json
            if stripped.startswith("{") or stripped.startswith("["):
                parsed = _json.loads(stripped)
                text = _json.dumps(parsed, ensure_ascii=False, indent=2)
                return _json_to_html(text)
        except Exception:
            pass
        return _md_to_html(text)

    def _make_code_view(self, text: str) -> QLabel:
        label = AutoHeightLabel(self._format_code_text(text))
        label.setObjectName("ToolCode")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _on_header_click(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()

    def toggle(self):
        self._expanded = not self._expanded
        self._apply_expanded(self._expanded, animate=True)

    def _stop_anim(self):
        """停止当前动画并断开 finished 信号，避免快速连点时
        多个动画同时驱动 maximumHeight 互相拉扯造成抖动。"""
        if self._anim is not None:
            try:
                self._anim.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._anim.stop()
            self._anim.deleteLater()
            self._anim = None

    def _apply_expanded(self, expanded: bool, animate: bool):
        self._expanded = expanded
        self._chevron.setText("▾" if expanded else "▸")
        if self.property("expanded") != expanded:
            self.setProperty("expanded", expanded)
            self.style().unpolish(self)
            self.style().polish(self)
        self._stop_anim()
        if not animate:
            self._body.setMaximumHeight(16777215 if expanded else 0)
            return
        # 折叠与展开都做动画,从当前实际高度平滑过渡,避免展开瞬间跳变
        self._anim = QPropertyAnimation(self._body, b"maximumHeight", self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(self._body.height())
        if expanded:
            self._anim.setEndValue(max(1, self._body.sizeHint().height()))
            self._anim.finished.connect(self._on_expand_finished)
        else:
            self._anim.setEndValue(0)
            self._anim.finished.connect(self._on_collapse_finished)
        self._anim.start()

    def _on_expand_finished(self):
        # 展开完成后解除高度上限,后续内容更新(如工具输出到达)不被卡住
        if self._expanded:
            self._body.setMaximumHeight(16777215)

    def _on_collapse_finished(self):
        if not self._expanded:
            self._body.setMaximumHeight(0)

    def _apply_status(self, status: str):
        icons = {"running": "⟳", "success": "✓", "error": "✗", "planning": "◇"}
        self._status_icon.setText(icons.get(status, "○"))
        self._status = status
        self._status_icon.setProperty("status", status)
        self._status_icon.style().unpolish(self._status_icon)
        self._status_icon.style().polish(self._status_icon)
        # success 折叠（仅对无输出的纯步骤卡自动折叠）
        has_output = hasattr(self, "_output_view") and self._output_view is not None
        if status == "success" and self._expanded and not self._steps_data and not has_output:
            self._expanded = False
            self._apply_expanded(False, animate=False)

    def update_status(self, status: str, output: str = ""):
        # 先添加内容，再更新状态（避免 _apply_status 在添加内容前折叠正文）
        if output:
            if hasattr(self, "_output_view") and self._output_view is not None:
                self._output_view.setText(self._format_code_text(output))
            else:
                self._output_view = self._make_code_view(output)
                if self._content_layout is None:
                    content = QFrame()
                    content.setObjectName("ToolCallContent")
                    self._content_layout = QVBoxLayout(content)
                    self._content_layout.setContentsMargins(12, 6, 12, 0)
                    self._content_layout.setSpacing(6)
                    body_layout = self._body.layout()
                    body_layout.addWidget(content)
                    self._has_content = True
                lbl2 = QLabel("执行结果")
                lbl2.setObjectName("ToolSectionLabel")
                self._content_layout.addWidget(lbl2)
                self._content_layout.addWidget(self._output_view)
            # 确保正文展开以显示新增的内容
            if not self._expanded:
                self._expanded = True
                self._apply_expanded(True, animate=False)
            # 标记布局为脏，下一次事件循环重算
            self._body.updateGeometry()
            self.updateGeometry()
        self._apply_status(status)


# ============================================================================
# 消息气泡
# ============================================================================

class MessageBubble(QFrame):
    """单条消息（带头像、内容、可选工具调用块）"""

    def __init__(self, role: str, content: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("MessageBubble")
        self._role = role
        self._raw_text = content
        self._tool_blocks: List[ToolCallBlock] = []

        self._avatar = IconWidget(AppIcon.USER if role == "user" else AppIcon.AGENT)
        self._avatar.setFixedSize(26, 26)
        self._avatar.setObjectName(
            "UserAvatar" if role == "user" else "AssistantAvatar"
        )
        self._body = self._build_body(content)
        # 内容为空(仅换行/空白且无文本、无图片)时隐藏 body 本体,避免纯工具
        # 调用回复出现空白消息框
        _has_img = isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        )
        _txt = content if isinstance(content, str) else (
            " ".join(p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text")
            if isinstance(content, list) else ""
        )
        if not _txt.strip() and not _has_img:
            self._body.setVisible(False)
        self._body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # 外层纵向：上行=头像+内容气泡，下行=工具块+时间戳
        # avatar 在左列,右侧纵向依次为 文本 body / 工具块 / 时间戳。
        # 头像对整个右列垂直居中,当回复仅含工具调用(文本为空)时头像也能
        # 相对工具块居中,且空文本不再残留空白框。
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(6)

        body_col = QVBoxLayout()
        body_col.setContentsMargins(0, 0, 0, 0)
        body_col.setSpacing(0)
        body_col.addWidget(self._body)

        self._tools_widget = QFrame()
        self._tools_widget.setObjectName("MsgTools")
        self._tools_layout = QVBoxLayout(self._tools_widget)
        self._tools_layout.setContentsMargins(0, 6, 0, 0)
        self._tools_layout.setSpacing(6)
        self._tools_widget.setVisible(False)

        from datetime import datetime
        self._meta = QLabel(datetime.now().strftime("%H:%M"))
        self._meta.setObjectName("MsgMeta")
        if role == "user":
            self._meta.setAlignment(Qt.AlignmentFlag.AlignRight)

        if role == "user":
            right_col.addLayout(body_col)
            right_col.setStretch(0, 1)
            right_col.addWidget(self._meta, 0, Qt.AlignmentFlag.AlignRight)
            outer.addLayout(right_col, 1)
            outer.addWidget(self._avatar, 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            right_col.addLayout(body_col)
            right_col.addWidget(self._tools_widget)
            right_col.addWidget(self._meta)
            outer.addWidget(self._avatar, 0, Qt.AlignmentFlag.AlignVCenter)
            outer.addLayout(right_col, 1)

    def _build_body(self, content) -> QFrame:
        body = QFrame()
        body.setObjectName("MsgBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)

        # content 可为 str 或多模态 content list
        text_parts = []
        image_urls = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    image_urls.append(part.get("image_url", {}).get("url", ""))

        html = _md_to_html("\n".join(text_parts)) if self._role == "assistant" \
            else "\n".join(text_parts).replace("\n", "<br>")
        for url in image_urls:
            html = self._append_image_to_html(html, url)
        self._content_label = AutoHeightLabel(html)
        self._content_label.setObjectName("MsgContent")
        self._content_label.setWordWrap(True)
        self._content_label.setProperty("role", self._role)
        self._content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._content_label.setTextFormat(Qt.TextFormat.RichText)
        if self._role == "user":
            self._content_label.setStyleSheet(
                "background-color: rgba(37, 99, 235, 0.15);"
                "border: 1px solid rgba(37, 99, 235, 0.3);"
                "border-top-right-radius: 4px;"
                "border-radius: 12px;"
            )
            body.setObjectName("MsgBodyUser")
        else:
            self._content_label.setStyleSheet(
                "background-color: rgba(255, 255, 255, 0.05);"
                "border: 1px solid rgba(255, 255, 255, 0.1);"
                "border-top-left-radius: 4px;"
                "border-radius: 12px;"
            )
        body_layout.addWidget(self._content_label)
        if not html.strip():
            self._content_label.setVisible(False)
        return body

    def _append_image_to_html(self, html: str, url: str) -> str:
        """把 data_url 图片缩放后以 <img> 追加到 HTML(QLabel RichText 渲染)。"""
        import base64
        from PySide6.QtCore import QBuffer
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap()
        ok = False
        if url.startswith("data:image/"):
            ok = pixmap.loadFromData(base64.b64decode(url.split(",", 1)[1]))
        else:
            ok = pixmap.load(url)
        if not ok:
            return html
        pixmap = pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
        buf = QBuffer()
        buf.open(QBuffer.OpenModeFlag.WriteOnly)
        pixmap.save(buf, "PNG")
        b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
        img_html = (
            f'<div style="margin:6px 0;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'width="{pixmap.width()}"/></div>'
        )
        return html + img_html

    def append_text(self, text: str):
        self._raw_text += text
        if not (self._raw_text or "").strip():
            # 仍是纯空白(如单独成 token 的换行符),先不渲染,避免空消息框
            return
        if self._role == "assistant":
            html = _md_to_html(self._raw_text)
        else:
            html = self._raw_text.replace("\n", "<br>")
        self._content_label.setText(html)
        if not self._content_label.isVisible():
            self._content_label.setVisible(True)
        if not self._body.isVisible():
            self._body.setVisible(True)

    def add_tool_block(self, block: ToolCallBlock):
        self._tool_blocks.append(block)
        self._tools_layout.addWidget(block)
        block.show()
        self._tools_widget.show()
        self._tools_widget.setVisible(True)
        # 确保父级布局标记为脏，以便自适应高度
        self._tools_widget.updateGeometry()
        self.updateGeometry()

    def is_empty(self) -> bool:
        """判断气泡是否有可见内容(文本或工具块)。用于 finalize 时清理空框。"""
        return (not (self._raw_text or "").strip()) and (not self._tool_blocks)

    def find_tool_block(self, name: str) -> Optional[ToolCallBlock]:
        for b in self._tool_blocks:
            if b._tool_name == name:
                return b
        return None


# ============================================================================
# bash 解锁审批卡
# ============================================================================

class BashApprovalCard(QFrame):
    """工具解锁审批卡：展示 agent 的解锁理由，等待用户允许/拒绝。

    respond 回调由工作线程的 threading.Event 消费（见
    VisionMindAgent._request_bash_approval / _request_tool_approval），
    点击任一按钮即唤醒。超时/停止时无人再消费回调，卡片保持已答状态即可。
    tool 参数为 None 时按 bash 审批卡渲染（向后兼容）。
    """

    def __init__(self, reason: str, respond, tool: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("BashApprovalCard")
        self._respond = respond
        self._answered = False

        tool = tool or "bash"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title = QLabel(f"🔐 Agent 申请解锁 {tool}")
        title.setObjectName("BashApprovalTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        msg = QLabel(reason or "（未提供理由）")
        msg.setObjectName("BashApprovalReason")
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._allow_btn = QPushButton("允许本次执行")
        self._allow_btn.setObjectName("BashApproveBtn")
        self._allow_btn.setFixedHeight(26)
        self._allow_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._allow_btn.clicked.connect(lambda: self._decide(True))
        self._deny_btn = QPushButton("拒绝")
        self._deny_btn.setObjectName("BashDenyBtn")
        self._deny_btn.setFixedHeight(26)
        self._deny_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._deny_btn.clicked.connect(lambda: self._decide(False))
        btn_row.addWidget(self._allow_btn)
        btn_row.addWidget(self._deny_btn)
        btn_row.addStretch()

        self._status = QLabel()
        self._status.setObjectName("BashApprovalStatus")
        self._status.hide()
        btn_row.addWidget(self._status)
        layout.addLayout(btn_row)

    def _decide(self, ok: bool):
        if self._answered:
            return
        self._answered = True
        self._allow_btn.setEnabled(False)
        self._deny_btn.setEnabled(False)
        self._status.setText("✅ 已允许（仅本次任务）" if ok else "⛔ 已拒绝")
        self._status.show()
        try:
            self._respond(ok)
        except Exception as e:
            print(f"[BashApprovalCard] respond 回调失败: {e}")


# ============================================================================
# 意图分析建议卡片
# ============================================================================

class SuggestionCard(QFrame):
    """意图分析建议卡片（标题 + 消息 + 可选执行按钮）"""

    suggestion_clicked = Signal(str, dict)  # action_name, params

    def __init__(self, title: str, message: str, action_name: str = "",
                 params: dict = None, parent=None):
        super().__init__(parent)
        self.setObjectName("SuggestionCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("SuggestionTitle")
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._msg_label = AutoHeightLabel(message)
        self._msg_label.setObjectName("SuggestionMsg")
        self._msg_label.setWordWrap(True)
        self._msg_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._msg_label)

        if action_name:
            btn_row = QHBoxLayout()
            btn = QPushButton("执行建议")
            btn.setObjectName("SuggestionBtn")
            btn.setFixedHeight(26)
            btn.clicked.connect(
                lambda checked=False, a=action_name, p=params or {}:
                self.suggestion_clicked.emit(a, p)
            )
            btn_row.addWidget(btn)
            btn_row.addStretch()
            layout.addLayout(btn_row)

        self._action_name = action_name

    def set_style(self, light: bool):
        """根据主题设置样式"""
        if light:
            self.setStyleSheet("""
                SuggestionCard {
                    background-color: rgba(37, 99, 235, 0.06);
                    border: 1px solid rgba(37, 99, 235, 0.25);
                    border-radius: 8px;
                }
                #SuggestionTitle { color: #1d4ed8; font-weight: bold; font-size: 13px; }
                #SuggestionMsg { color: #4b5563; font-size: 12px; }
                #SuggestionBtn {
                    background-color: #2563eb; color: white; border: none;
                    border-radius: 4px; padding: 2px 12px; font-size: 12px;
                }
                #SuggestionBtn:hover { background-color: #1d4ed8; }
            """)
        else:
            self.setStyleSheet("""
                SuggestionCard {
                    background-color: rgba(74, 158, 255, 0.08);
                    border: 1px solid rgba(74, 158, 255, 0.25);
                    border-radius: 8px;
                }
                #SuggestionTitle { color: #4a9eff; font-weight: bold; font-size: 13px; }
                #SuggestionMsg { color: #d1d5db; font-size: 12px; }
                #SuggestionBtn {
                    background-color: #2563eb; color: white; border: none;
                    border-radius: 4px; padding: 2px 12px; font-size: 12px;
                }
                #SuggestionBtn:hover { background-color: #3b82f6; }
            """)


# ============================================================================
# 会话列表项
# ============================================================================

class ElidedLabel(QLabel):
    """自动省略号显示的标签：长文本按可用宽度截断为 "…" 结尾

    避免长文本撑宽布局导致兄弟控件（如删除按钮）被挤出可视区域
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = text
        super().setText("")
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

    def set_full_text(self, text: str):
        self._full_text = text or ""
        self._apply_elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elide()

    def _apply_elide(self):
        w = self.width()
        if w <= 0 or not self._full_text:
            return
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, w)
        super().setText(elided)


class SessionItem(QFrame):
    """会话历史条目"""

    clicked = Signal(int)
    delete_clicked = Signal(int)
    rename_requested = Signal(int)

    def __init__(self, index: int, title: str, time_str: str = "", preview: str = "",
                 active: bool = False, parent=None):
        super().__init__(parent)
        self._index = index
        self._title = title
        self.setObjectName("SessionItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("active", active)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(4)
        self._title_lbl = ElidedLabel(title)
        self._title_lbl.setObjectName("SessionTitle")
        self._title_lbl.setStyleSheet("font-size: 13px; font-weight: 500;")
        self._title_lbl.setToolTip(title)
        self._title_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title_row.addWidget(self._title_lbl, 1)
        self._del_btn = QToolButton()
        self._del_btn.setObjectName("SessionDelBtn")
        self._del_btn.setText("×")
        self._del_btn.setFixedSize(20, 20)
        self._del_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._del_btn.setToolTip("删除会话")
        self._del_btn.clicked.connect(self._on_delete)
        title_row.addWidget(self._del_btn)
        layout.addLayout(title_row)

        self._time_lbl = ElidedLabel(time_str) if time_str else None
        if self._time_lbl:
            self._time_lbl.setObjectName("SessionTime")
            self._time_lbl.setStyleSheet("font-size: 11px; color: #6b7280;")
            self._time_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self._time_lbl)
        self._prev_lbl = ElidedLabel(preview) if preview else None
        if self._prev_lbl:
            self._prev_lbl.setObjectName("SessionPreview")
            self._prev_lbl.setStyleSheet("font-size: 12px; color: #9ca3af;")
            self._prev_lbl.setToolTip(preview)
            self._prev_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self._prev_lbl)

    def set_index(self, idx: int):
        self._index = idx

    def update_title(self, title: str):
        self._title = title
        self._title_lbl.set_full_text(title)
        self._title_lbl.setToolTip(title)

    def _on_delete(self):
        self.delete_clicked.emit(self._index)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        rename_action = menu.addAction("重命名")
        delete_action = menu.addAction("删除")
        action = menu.exec(self.mapToGlobal(pos))
        if action == rename_action:
            self.rename_requested.emit(self._index)
        elif action == delete_action:
            self.delete_clicked.emit(self._index)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)


# ============================================================================
# 会话历史面板（从左侧滑入）
# ============================================================================

class SessionPanel(QFrame):
    """会话历史滑入面板"""

    session_selected = Signal(int)
    new_session_requested = Signal()
    close_requested = Signal()
    delete_requested = Signal(int)
    rename_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SessionPanel")
        self.setFixedWidth(280)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._items: List[SessionItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("SessionPanelHeader")
        header.setFixedHeight(48)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 0, 14, 0)
        title = QLabel("会话历史")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        h_layout.addWidget(title)
        h_layout.addStretch()
        close_btn = QToolButton()
        close_btn.setText("✕")
        close_btn.setObjectName("HeaderBtn")
        close_btn.setFixedSize(30, 30)
        close_btn.setToolTip("关闭")
        close_btn.clicked.connect(self.close_requested.emit)
        h_layout.addWidget(close_btn)
        layout.addWidget(header)

        new_btn = QPushButton("+ 新建会话")
        new_btn.setObjectName("NewSessionBtn")
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.clicked.connect(self.new_session_requested.emit)
        layout.addWidget(new_btn)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setObjectName("SessionListScroll")
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(8, 8, 8, 8)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        layout.addWidget(self._scroll, 1)

    def add_session(self, title: str, preview: str = "", time_str: str = "",
                    active: bool = False) -> int:
        idx = len(self._items)
        item = SessionItem(idx, title, time_str, preview, active)
        item.clicked.connect(self._on_select)
        item.delete_clicked.connect(self.delete_requested.emit)
        item.rename_requested.connect(self.rename_requested.emit)
        self._items.append(item)
        self._list_layout.insertWidget(self._list_layout.count() - 1, item)
        if active:
            self._set_active(idx)
        return idx

    def clear_sessions(self):
        """清空所有会话条目"""
        for item in self._items:
            self._list_layout.removeWidget(item)
            item.deleteLater()
        self._items.clear()

    def remove_session(self, idx: int):
        """移除指定索引的会话条目（UI 和数据层同步移除）"""
        if 0 <= idx < len(self._items):
            item = self._items.pop(idx)
            self._list_layout.removeWidget(item)
            item.deleteLater()
            # 更新剩余条目的索引
            for i, it in enumerate(self._items):
                it.set_index(i)

    def _on_select(self, idx: int):
        self._set_active(idx)
        self.session_selected.emit(idx)

    def _set_active(self, idx: int):
        for i, it in enumerate(self._items):
            it.setProperty("active", i == idx)
            it.style().unpolish(it)
            it.style().polish(it)


# ============================================================================
# 执行状态栏
# ============================================================================

class ExecutionStatusBar(QFrame):
    """Agent 执行中的状态栏"""

    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ExecutionStatusBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(8)

        self._spinner = Spinner(self)
        self._spinner.start()
        layout.addWidget(self._spinner)

        self._status_text = QLabel("正在执行...")
        self._status_text.setObjectName("ExecStatusText")
        layout.addWidget(self._status_text, 1)

        stop_btn = QPushButton("停止")
        stop_btn.setObjectName("StopBtn")
        stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        stop_btn.setFixedHeight(24)
        stop_btn.clicked.connect(self.stop_requested.emit)
        layout.addWidget(stop_btn)

    def show_status(self, text: str = ""):
        if text:
            self._status_text.setText(text)
        self.show()
        self._spinner.start()

    def update_status(self, text: str):
        self._status_text.setText(text)

    def hide_status(self):
        self._spinner.stop()
        self.hide()


# ============================================================================
# 欢迎屏
# ============================================================================

class WelcomeScreen(QFrame):
    """空会话欢迎屏"""

    quick_action_triggered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WelcomeScreen")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.addStretch()

        title = QLabel("VisionMind Agent")
        title.setObjectName("WelcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("我可以帮你完成标注任务、管理项目、<br>分析数据等各种操作。")
        desc.setObjectName("WelcomeDesc")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(desc)

        grid_container = QFrame()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(8)
        actions = [
            ("🏷️", "自动标注当前图片", "帮我自动标注当前图片"),
            ("📊", "分析数据集质量", "分析数据集质量"),
            ("⚡", "批量标注前10张", "批量标注前10张图片"),
            ("📤", "导出标注数据", "导出标注数据"),
        ]
        for i, (icon, text, prompt) in enumerate(actions):
            card = QFrame()
            card.setObjectName("QuickAction")
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 12, 12, 12)
            cl.setSpacing(6)
            ic = QLabel(icon)
            ic.setStyleSheet("font-size: 18px;")
            cl.addWidget(ic)
            tx = QLabel(text)
            tx.setStyleSheet("font-size: 12px; color: #9ca3af;")
            cl.addWidget(tx)
            card.mousePressEvent = lambda e, p=prompt: self.quick_action_triggered.emit(p)
            grid_layout.addWidget(card, i // 2, i % 2)
        layout.addWidget(grid_container)
        layout.addStretch()


# ============================================================================
# 输入区
# ============================================================================

class InputArea(QFrame):
    """消息输入区（Enter 发送，Shift+Enter 换行）"""

    send_requested = Signal(str)
    model_changed = Signal(str)
    model_selected = Signal(str, str)  # provider_id, model —— 跨提供商即时切换
    attachment_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("InputArea")
        self._models = []
        self._min_input_height = 40
        self._max_input_height = 150
        self._resize_guard = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(0)

        wrapper = QFrame()
        wrapper.setObjectName("InputWrapper")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        # top: 左侧按钮 + 输入框 + 发送按钮
        top = QFrame()
        top.setObjectName("InputTop")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.setSpacing(4)

        left_actions = QFrame()
        la_layout = QHBoxLayout(left_actions)
        la_layout.setContentsMargins(8, 0, 4, 0)
        la_layout.setSpacing(2)
        # 语义化图标按钮（附件/图片）——qfw TransparentToolButton 自绘图标
        for icon, tip in [(AppIcon.ATTACH, "附加文件"), (AppIcon.IMAGE, "附加图片")]:
            btn = TransparentToolButton(icon)
            btn.setObjectName("InputBtn")
            btn.setIconSize(QSize(18, 18))
            btn.setFixedSize(32, 32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            btn.clicked.connect(
                lambda checked, i=tip: self._on_attachment(i)
            )
            la_layout.addWidget(btn)
        top_layout.addWidget(left_actions)

        self._input = QPlainTextEdit()
        self._input.setObjectName("MessageInput")
        self._input.setPlaceholderText("输入消息，Enter 发送...")
        self._input.setFixedHeight(self._min_input_height)
        self._input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._input.installEventFilter(self)
        self._input.textChanged.connect(self._auto_resize)
        top_layout.addWidget(self._input, 1)

        from PySide6.QtWidgets import QListWidget
        self._suggest_list = QListWidget()
        # Popup 窗口标志: 独立弹出窗口不受 parent 几何约束,且能正常接收
        # 鼠标点击(ToolTip 窗口在按下瞬间即被关闭,itemClicked 不会触发,
        # 表现为点击选项无法填充输入框)
        self._suggest_list.setWindowFlags(Qt.WindowType.Popup)
        self._suggest_list.setMaximumHeight(200)
        self._suggest_list.hide()
        self._suggest_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._suggest_list.itemClicked.connect(self._insert_suggestion)
        self._input.textChanged.connect(self._update_suggestions)
        # 候选项回填输入框期间置 True,屏蔽 textChanged 重新弹出候选框
        self._inserting = False

        self._send_btn = QPushButton("↑")
        self._send_btn.setObjectName("SendBtn")
        self._send_btn.setFixedSize(34, 34)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setToolTip("发送")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._on_send)
        top_layout.addWidget(self._send_btn)

        wrapper_layout.addWidget(top)

        # bottom: 模型选择 + 提示
        bottom = QFrame()
        bottom.setObjectName("InputBottom")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 6)
        bottom_layout.setSpacing(0)

        self._model_selector = QComboBox()
        self._model_selector.setObjectName("ModelSelector")
        self._model_selector.currentTextChanged.connect(self._on_model_selected)
        # 监听全局配置变更，按提供商分组刷新模型列表
        settings.settings_changed.connect(self._on_ai_model_changed)
        self._reload_models()
        bottom_layout.addWidget(self._model_selector)
        bottom_layout.addStretch()

        hint = QLabel("Enter 发送 · Shift+Enter 换行")
        hint.setObjectName("InputHint")
        bottom_layout.addWidget(hint)
        wrapper_layout.addWidget(bottom)

        layout.addWidget(wrapper)

    def eventFilter(self, obj, event):
        if obj == self._input and event.type() == QEvent.Type.KeyPress:
            if self._suggest_list.isVisible():
                key = event.key()
                if key == Qt.Key.Key_Up or key == Qt.Key.Key_Down:
                    row = self._suggest_list.currentRow()
                    n = self._suggest_list.count()
                    self._suggest_list.setCurrentRow(
                        (row - 1) % n if key == Qt.Key.Key_Up else (row + 1) % n)
                    return True
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                    it = self._suggest_list.currentItem()
                    if it:
                        self._insert_suggestion(it)
                        return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._resize_guard:
            self._auto_resize()

    def _auto_resize(self):
        text = self._input.toPlainText()
        self._send_btn.setEnabled(bool(text.strip()))
        doc = self._input.document()
        doc.setTextWidth(self._input.viewport().width())
        lay = doc.documentLayout()
        total = 0.0
        block = doc.begin()
        while block.isValid():
            total += lay.blockBoundingRect(block).height()
            block = block.next()
        content_h = total + doc.documentMargin()
        vm = self._input.viewport().contentsMargins()
        extra = vm.top() + vm.bottom() + self._input.frameWidth() * 2
        new_h = int(max(
            self._min_input_height,
            min(self._max_input_height, math.ceil(content_h + extra) + 1),
        ))
        if new_h == self._input.height():
            return
        self._resize_guard = True
        try:
            self._input.setFixedHeight(new_h)
        finally:
            self._resize_guard = False

    def _on_send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._input.setFixedHeight(self._min_input_height)
        self.send_requested.emit(text)

    def _update_suggestions(self):
        if self._inserting:
            return
        text = self._input.toPlainText()
        cursor = self._input.textCursor()
        pos = cursor.position()
        idx = text.rfind("@", 0, pos)
        if idx == -1 or pos - idx > 64:
            self._suggest_list.hide()
            return
        prefix = text[idx + 1:pos]
        items = self._suggestion_items(prefix)
        if not items:
            self._suggest_list.hide()
            return
        self._suggest_list.clear()
        for it in items[:8]:
            self._suggest_list.addItem(it)
        # 父级为顶层窗口,用全局坐标定位到光标下方
        rect = self._input.cursorRect()
        global_pos = self._input.mapToGlobal(rect.bottomLeft() + QPoint(0, 4))
        self._suggest_list.move(global_pos)
        self._suggest_list.show()
        self._suggest_list.setCurrentRow(0)

    def _suggestion_items(self, prefix: str):
        """生成候选列表。

        优先级:
        1. 空 prefix 或 prefix 命中已知语法关键字 → 显示语法模板(子串匹配)
        2. prefix 不匹配任何语法关键字 → 按文件名搜索当前项目图像路径
        """
        templates = [
            "@原图:当前图", "@图:当前图索引", "@局部:当前图标注",
            "@渲染图:当前图", "@渲染局部:当前图标注",
            "@ROI", "@类别:", "@难样本", "@搜索:", "@报告:",
        ]
        if not prefix:
            return templates
        matched = [t for t in templates if prefix in t]
        if matched:
            return matched
        # prefix 不匹配语法关键字 → 文件名搜索
        keyword = prefix[3:] if prefix.startswith("文件:") else prefix
        paths = self._search_image_paths(keyword)
        items = []
        for p in paths:
            item = QListWidgetItem(p)
            item.setData(Qt.ItemDataRole.UserRole, "@文件:" + p)
            items.append(item)
        return items

    def _search_image_paths(self, keyword: str, limit: int = 20) -> list:
        """经 service 层 action 搜索当前项目图片路径,仅返回图像文件。

        未加载项目/非图像文件一律不提示,返回空列表。
        """
        try:
            from core.common.action_registry import ActionRegistry
            registry = ActionRegistry.instance()
            result = registry.call("annotation.manual.search_images", keyword=keyword)
        except Exception:
            return []
        files = (result.get("files") or []) if isinstance(result, dict) else []
        out = []
        for entry in files:
            p = entry[0] if isinstance(entry, (list, tuple)) else entry
            if isinstance(p, str) and p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                out.append(p)
            if len(out) >= limit:
                break
        return out

    def _insert_suggestion(self, item):
        text = self._input.toPlainText()
        cursor = self._input.textCursor()
        pos = cursor.position()
        idx = text.rfind("@", 0, pos)
        if idx != -1:
            insert_text = item.data(Qt.ItemDataRole.UserRole) or item.text()
            # 先隐藏候选框并屏蔽插入期间的 textChanged,
            # 否则 setPlainText 触发 _update_suggestions 按新文本重新弹框(闪烁)
            self._suggest_list.hide()
            self._inserting = True
            try:
                self._input.setPlainText(text[:idx] + insert_text + text[pos:])
                c = self._input.textCursor()
                c.setPosition(idx + len(insert_text))
                self._input.setTextCursor(c)
            finally:
                self._inserting = False
        else:
            self._suggest_list.hide()

    def append_reference(self, ref_text: str):
        """在输入框末尾追加引用文本(不发送)。"""
        text = self._input.toPlainText()
        self._input.setPlainText((text + " " + ref_text) if text else ref_text)
        self._input.setFocus()

    def set_model(self, name: str):
        # 只允许使用当前配置的 AI 模型，其他传入模型一律忽略
        if not self._models:
            self._reload_models()
        if name not in self._models:
            return
        self._model_selector.blockSignals(True)
        for i in range(self._model_selector.count()):
            data = self._model_selector.itemData(i) or ("", "")
            if data[1] == name:
                self._model_selector.setCurrentIndex(i)
                break
        self._model_selector.blockSignals(False)
        self.model_changed.emit(name)

    def get_model(self) -> str:
        return self._model_selector.currentText()

    def _reload_models(self):
        """从 ai_providers 按提供商刷新分组模型列表（「提供商 › 模型」）。"""
        from core.common.ai_providers import grouped_models, get_active_provider
        groups = grouped_models()
        self._models = [m for _, _, models in groups for m in models]
        self._model_selector.blockSignals(True)
        self._model_selector.clear()
        if not groups:
            self._model_selector.addItem("未配置模型", ("", ""))
        else:
            for pid, pname, models in groups:
                for m in models:
                    self._model_selector.addItem(f"{pname} › {m}", (pid, m))
            self._select_active_model()
        self._model_selector.blockSignals(False)

    def _select_active_model(self):
        """将下拉定位到当前激活提供商的默认模型。"""
        from core.common.ai_providers import get_active_provider
        active = get_active_provider()
        target = (active.get("id", ""), active.get("default_model", ""))
        for i in range(self._model_selector.count()):
            if tuple(self._model_selector.itemData(i) or ()) == target:
                self._model_selector.setCurrentIndex(i)
                return

    def _on_ai_model_changed(self, key: str, value):
        if key == "ai_providers":
            self._reload_models()

    def _on_model_selected(self, model: str):
        data = self._model_selector.itemData(self._model_selector.currentIndex()) or ("", "")
        pid, m = data
        if m:
            self.model_selected.emit(pid, m)
            self.model_changed.emit(m)

    def _on_attachment(self, mode: str):
        if mode == "附加文件":
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择附件", "", "所有文件 (*.*)"
            )
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)"
            )
        if files:
            self.attachment_requested.emit(files)


# ============================================================================
# AgentDialog 主容器
# ============================================================================

class AgentDialog(QFrame):
    """Agent 对话侧栏主容器"""

    # 兼容 ChatSidebar 的信号
    message_sent = Signal(object)  # 放宽以便传递多模态 content list
    debug_requested = Signal()
    # 新增信号
    stop_requested = Signal()
    attachment_requested = Signal(list)
    suggestion_clicked = Signal(str, dict)  # 意图分析建议：action_name, params
    model_selected = Signal(str, str)  # provider_id, model —— 跨提供商即时切换
    history_loaded = Signal(object)  # 会话加载后清理完毕(已去 base64)的历史消息

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AgentDialog")
        self.setMinimumWidth(280)
        self.setMaximumWidth(700)

        self._messages: list = []
        self._current_assistant_bubble: Optional[MessageBubble] = None
        self._thinking: Optional[ThinkingIndicator] = None
        self._executing = False
        self._session_open = False
        self._session_ids: list[str] = []
        self._current_session_id: str | None = None
        self._load_qss()
        self._setup_ui()
        self._connect_signals()
        self._load_sessions()
        # 等布局就绪后设置初始宽度（不限制后续拖动）
        QTimer.singleShot(0, lambda: self.setFixedWidth(420))

    # ----------------------- QSS -----------------------

    def _load_qss(self):
        qss_path = os.path.join(Config.QSS_DIR, "agent_dialog.qss")
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                self._qss_dark = f.read()
        except OSError:
            self._qss_dark = ""
        # 分离 light 段：QSS 文件中用注释标记 /* === LIGHT THEME === */
        light_qss = ""
        if "/* === LIGHT THEME === */" in self._qss_dark:
            parts = self._qss_dark.split("/* === LIGHT THEME === */", 1)
            self._qss_dark = parts[0].strip()
            light_qss = parts[1].strip()
        self._qss_light = light_qss
        self._apply_qss()

    def _apply_qss(self):
        if theme_manager.is_light_theme():
            self.setStyleSheet(self._qss_light or self._qss_dark)
        else:
            self.setStyleSheet(self._qss_dark)

    # ----------------------- UI -----------------------

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左侧拖拽把手
        self._resize_handle = ResizeHandle(self)
        root.addWidget(self._resize_handle)
        self._resize_handle.width_changed.connect(self._on_resize)

        # 主容器
        container = QFrame()
        container.setObjectName("AgentSidebar")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setObjectName("SidebarHeader")
        header.setFixedHeight(48)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 0, 14, 0)
        h_layout.setSpacing(8)

        left_section = QHBoxLayout()
        left_section.setSpacing(8)
        self._history_btn = TransparentToolButton(AppIcon.HISTORY)
        self._history_btn.setObjectName("HeaderBtn")
        self._history_btn.setFixedSize(30, 30)
        self._history_btn.setToolTip("会话历史")
        left_section.addWidget(self._history_btn)
        h_layout.addLayout(left_section)
        h_layout.addStretch()

        self._debug_btn = TransparentToolButton(AppIcon.DEBUG)
        self._debug_btn.setObjectName("HeaderBtn")
        self._debug_btn.setFixedSize(30, 30)
        self._debug_btn.setToolTip("系统提示词")
        h_layout.addWidget(self._debug_btn)

        self._clear_btn = TransparentToolButton(AppIcon.CLEAR)
        self._clear_btn.setObjectName("HeaderBtn")
        self._clear_btn.setFixedSize(30, 30)
        self._clear_btn.setToolTip("清空对话")
        h_layout.addWidget(self._clear_btn)
        container_layout.addWidget(header)

        # 会话面板（作为 AgentDialog 子控件浮动在最上层，脱离 layout 管理）
        self._session_panel = SessionPanel(self)
        self._HEADER_H = 48
        self._session_panel.setGeometry(-280, self._HEADER_H, 280, 0)
        self._session_panel.hide()

        # 消息滚动区
        self._messages_scroll = QScrollArea()
        self._messages_scroll.setObjectName("MessagesArea")
        self._messages_scroll.setWidgetResizable(True)
        # 极宽表格等内容仍可能超出视口最小宽度,允许必要时横向滚动,
        # 而不是把内容裁剪在侧栏之外
        self._messages_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._messages_widget = QWidget()
        self._messages_widget.setObjectName("MessagesContainer")
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(16, 16, 16, 16)
        self._messages_layout.setSpacing(16)
        self._messages_layout.addStretch()
        self._messages_scroll.setWidget(self._messages_widget)

        self._welcome = WelcomeScreen()
        self._messages_layout.insertWidget(0, self._welcome)
        self._welcome_visible = True
        container_layout.addWidget(self._messages_scroll, 1)

        # 执行状态栏
        self._exec_bar = ExecutionStatusBar()
        self._exec_bar.setObjectName("ExecutionStatusBar")
        self._exec_bar.hide()
        container_layout.addWidget(self._exec_bar)

        # 输入区
        self._input_area = InputArea()
        container_layout.addWidget(self._input_area)

        root.addWidget(container, 1)

        # 会话面板高度跟随容器
        container_layout.parent  # noop
        QTimer.singleShot(0, self._resize_session_panel)

    def _resize_session_panel(self):
        h = max(0, self.height() - self._HEADER_H)
        self._session_panel.setGeometry(-280 if not self._session_open else 0,
                                        self._HEADER_H, 280, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_session_panel()

    # ----------------------- 信号 -----------------------

    def _connect_signals(self):
        self._history_btn.clicked.connect(self.toggle_session_panel)
        self._clear_btn.clicked.connect(self.clear_messages)
        self._debug_btn.clicked.connect(self.debug_requested.emit)
        self._input_area.send_requested.connect(self._on_input_send)
        self._input_area.model_changed.connect(self._on_model_changed)
        self._input_area.attachment_requested.connect(self.attachment_requested.emit)
        self._input_area.model_selected.connect(self.model_selected.emit)
        self._exec_bar.stop_requested.connect(self._on_stop)
        self._session_panel.close_requested.connect(self.hide_session_panel)
        self._session_panel.new_session_requested.connect(self.new_session)
        self._session_panel.session_selected.connect(self._on_session_selected)
        self._session_panel.delete_requested.connect(self._on_session_delete)
        self._session_panel.rename_requested.connect(self._on_session_rename)
        self._welcome.quick_action_triggered.connect(self._on_input_send)
        if hasattr(theme_manager, "theme_changed"):
            try:
                theme_manager.theme_changed.connect(lambda *_: self._apply_qss())
            except Exception:
                pass

    def set_event_bus(self, bus):
        """注入事件总线并订阅「添加到对话」事件(幂等,可安全重复调用)。"""
        if bus is None or getattr(self, "_event_bus", None) is bus:
            return
        self._event_bus = bus
        # reference_added 现经统一状态信号 app:state_changed 广播，按 type 分流
        from core.event_bus import StateType, StateRouter
        self._agent_router = StateRouter(bus).register(StateType.REFERENCE_ADDED,
                                                       self._on_reference_added)

    def _on_reference_added(self, ref_text):
        if hasattr(self, "_input_area") and ref_text:
            self._input_area.append_reference(str(ref_text))

    def _on_input_send(self, text: str):
        from core.agent.multimodal import ReferenceResolver
        content = text
        try:
            content = ReferenceResolver().resolve_message(text)
        except Exception:
            content = text  # 解析失败时退回纯文本
        self.add_message("user", content)
        self.message_sent.emit(content)

    def _on_model_changed(self, model: str):
        pass

    def _on_stop(self):
        self._executing = False
        self.stop_requested.emit()
        self.hide_exec_status()

    def _on_resize(self, new_width: int):
        self.setFixedWidth(new_width)

    def _on_session_selected(self, idx: int):
        """点击会话历史后加载该会话的消息"""
        if idx < 0 or idx >= len(self._session_ids):
            self.hide_session_panel()
            return
        sid = self._session_ids[idx]
        try:
            from corecoder.session import load_session
            result = load_session(sid)
            if result is None:
                self.hide_session_panel()
                return
            messages, model = result
            # 加载后清理历史:删除 base64 图片 user 消息,将解读写回 tool 消息
            from core.agent.multimodal import replace_image_urls_with_captions
            messages = replace_image_urls_with_captions(messages)
        except Exception:
            self.hide_session_panel()
            return
        self._current_session_id = sid
        self.clear_messages()
        self.history_loaded.emit(messages)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content") or ""
            # list 内容(多模态 user 消息)直接透传给 add_message,气泡按 _build_body 渲染
            if isinstance(content, list):
                self.add_message(role, content)
                i += 1
                continue

            if role == "tool":
                i += 1
                continue
            elif role == "assistant":
                self.start_assistant_message(content)
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        args_str = fn.get("arguments", "")
                        # 预读后续 role: tool 消息找到对应结果
                        output_data = ""
                        for j in range(i + 1, len(messages)):
                            nxt = messages[j]
                            if nxt.get("role") != "tool":
                                break
                            if nxt.get("tool_call_id") == tc.get("id"):
                                output_data = nxt.get("content", "")
                                break
                        self.add_tool_call(
                            fn.get("name", ""),
                            "success",
                            input_data=args_str,
                            output_data=output_data,
                        )
                self.finalize_assistant()
            else:
                self.add_message(role, content)
            i += 1
        self.hide_session_panel()

    def _on_session_delete(self, idx: int):
        """删除一条会话记录"""
        if idx < 0 or idx >= len(self._session_ids):
            return
        sid = self._session_ids.pop(idx)
        if sid == self._current_session_id:
            self.clear_messages()
            self._current_session_id = None
        from corecoder.session import SESSIONS_DIR
        session_file = SESSIONS_DIR / f"{sid}.json"
        if session_file.exists():
            session_file.unlink()
        self._session_panel.remove_session(idx)

    def _on_session_rename(self, idx: int):
        """重命名一条会话记录"""
        if idx < 0 or idx >= len(self._session_ids):
            return
        sid = self._session_ids[idx]
        from corecoder.session import SESSIONS_DIR
        import json as _json
        session_file = SESSIONS_DIR / f"{sid}.json"
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            return
        old_title = data.get("id", sid)
        new_title, ok = QInputDialog.getText(
            self, "重命名会话", "请输入新名称:", text=old_title
        )
        if ok and new_title.strip():
            # CoreCoder 的 id 是文件名标识符，和显示标题无关
            # 所以我们直接修改面板中显示的条目文本即可
            self._session_panel._items[idx].update_title(new_title.strip())

    # ----------------------- 公共 API（向下兼容） -----------------------

    def add_message(self, role: str, content):
        """添加一条消息"""
        self._hide_welcome()
        bubble = MessageBubble(role, content)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, bubble
        )
        self._messages.append({"role": role, "content": content, "bubble": bubble})
        self._scroll_to_bottom()
        self._request_relayout()

    def start_assistant_message(self, initial_text: str = ""):
        """开始一个新的助手消息气泡（流式）"""
        self._hide_welcome()
        self._remove_thinking()
        self._current_assistant_bubble = MessageBubble("assistant", initial_text)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, self._current_assistant_bubble
        )
        self._messages.append(
            {"role": "assistant", "content": initial_text, "bubble": self._current_assistant_bubble}
        )
        self._scroll_to_bottom()
        self._request_relayout()

    def append_to_assistant(self, text: str):
        """向当前正在流式输出的助手消息追加文本"""
        if self._current_assistant_bubble is None:
            self.start_assistant_message(text)
        else:
            self._current_assistant_bubble.append_text(text)
            if self._messages:
                self._messages[-1]["content"] += text
        self._scroll_to_bottom()

    def finalize_assistant(self):
        """完成当前助手消息的流式输出。

        若该气泡既无可见文本也无工具块(如工具调用轮次只产生了空 token),
        将其从界面移除,避免残留空白消息框。
        """
        bubble = self._current_assistant_bubble
        self._current_assistant_bubble = None
        if bubble is not None and hasattr(bubble, "is_empty") and bubble.is_empty():
            self._remove_bubble(bubble)

    def _remove_bubble(self, bubble):
        """从消息列表与布局中移除指定气泡。"""
        self._messages[:] = [m for m in self._messages if m.get("bubble") is not bubble]
        idx = self._messages_layout.indexOf(bubble)
        if idx >= 0:
            item = self._messages_layout.takeAt(idx)
            if item and item.widget():
                item.widget().deleteLater()

    def show_thinking(self, thinking: bool):
        """显示/隐藏思考动画"""
        if thinking:
            self._show_thinking()
        else:
            self._remove_thinking()

    def clear_messages(self):
        """清空消息列表"""
        while self._messages_layout.count() > 1:
            item = self._messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._messages.clear()
        self._current_assistant_bubble = None
        self._welcome = WelcomeScreen()
        self._messages_layout.insertWidget(0, self._welcome)
        self._welcome_visible = True
        self._welcome.quick_action_triggered.connect(self._on_input_send)
        self.hide_exec_status()

    # ----------------------- 新增 API -----------------------

    def add_tool_call(self, tool_name: str, status: str, duration: str = "",
                      input_data: str = "", output_data: str = "", steps: list = None):
        """在当前助手消息中添加一个工具调用展示块"""
        block = ToolCallBlock(tool_name, status, duration, input_data, output_data, steps)
        if self._current_assistant_bubble is None:
            self.start_assistant_message("")
        self._current_assistant_bubble.add_tool_block(block)
        self._scroll_to_bottom()
        return block

    def update_tool_status(self, tool_name_or_block, status: str, output: str = ""):
        """更新工具调用的状态"""
        block = None
        if isinstance(tool_name_or_block, ToolCallBlock):
            block = tool_name_or_block
        elif self._current_assistant_bubble:
            block = self._current_assistant_bubble.find_tool_block(tool_name_or_block)
        if block is None:
            for m in reversed(self._messages):
                if m.get("bubble") and hasattr(m["bubble"], "find_tool_block"):
                    block = m["bubble"].find_tool_block(tool_name_or_block)
                    if block:
                        break
        if block:
            block.update_status(status, output)

    def update_exec_status(self, text: str):
        """更新执行状态栏文本"""
        self._exec_bar.update_status(text)

    def show_exec_status(self, text: str = ""):
        """显示执行状态栏"""
        self._executing = True
        self._exec_bar.show_status(text)

    def hide_exec_status(self):
        """隐藏执行状态栏"""
        self._executing = False
        self._exec_bar.hide_status()

    def set_model(self, model_name: str):
        """切换当前模型"""
        self._input_area._model_selector.blockSignals(True)
        try:
            self._input_area.set_model(model_name)
        finally:
            self._input_area._model_selector.blockSignals(False)

    def add_session(self, title: str, preview: str = "", time_str: str = ""):
        """添加一条会话历史记录"""
        return self._session_panel.add_session(title, preview, time_str, active=False)

    def add_suggestion(self, title: str, message: str, action_name: str = "", params: dict = None):
        """添加意图分析建议卡片（可执行建议动作）"""
        self._hide_welcome()
        card = SuggestionCard(title, message, action_name, params)
        card.set_style(theme_manager.is_light_theme())
        card.suggestion_clicked.connect(self.suggestion_clicked)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, card
        )
        self._scroll_to_bottom()
        return card

    def request_bash_approval(self, reason: str, respond) -> None:
        """展示 bash 解锁审批卡（跨线程调用，respond 由工作线程 Event 消费）"""
        card = BashApprovalCard(reason, respond)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, card
        )
        self._scroll_to_bottom()

    def request_tool_approval(self, tool: str, reason: str, respond) -> None:
        """展示高危工具解锁审批卡（跨线程调用，respond 由工作线程 Event 消费）"""
        card = BashApprovalCard(reason, respond, tool=tool)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, card
        )
        self._scroll_to_bottom()

    # ----------------------- 内部辅助 -----------------------

    def _show_thinking(self):
        self._hide_welcome()
        self._remove_thinking()
        self._thinking = ThinkingIndicator()
        self._thinking.start()
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, self._thinking
        )
        self._scroll_to_bottom()

    def _remove_thinking(self):
        if self._thinking is not None:
            self._thinking.stop()
            self._thinking.setParent(None)
            self._thinking.deleteLater()
            self._thinking = None

    def _hide_welcome(self):
        if self._welcome_visible:
            for i in range(self._messages_layout.count()):
                item = self._messages_layout.itemAt(i)
                if item and item.widget() is self._welcome:
                    self._messages_layout.takeAt(i)
                    break
            self._welcome.setParent(None)
            self._welcome.deleteLater()
            self._welcome = None
            self._welcome_visible = False

    def _scroll_to_bottom(self):
        sb = self._messages_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: sb.setValue(sb.maximum()))

    def _request_relayout(self):
        """在下一事件循环用已确定的实际宽度重新计算消息列表布局。

        新插入的富文本气泡在父级宽度尚未落定时 sizeHint 高度可能偏大，
        延时 activate 以真实宽度重算，避免“首条消息框过大、等 agent
        回复后才缩回正确高度”的问题。
        """
        self._messages_layout.invalidate()
        self._messages_widget.updateGeometry()
        QTimer.singleShot(0, self._messages_layout.activate)

    # ----------------------- 会话持久化 -----------------------

    def _load_sessions(self):
        """从 CoreCoder 会话目录加载会话历史"""
        try:
            from corecoder.session import list_sessions
            sessions = list_sessions()
            self._session_ids.clear()
            for s in sessions:
                sid = s.get("id", "")
                if not sid:
                    continue
                self._session_ids.append(sid)
                saved_at = s.get("saved_at", "")
                # 解析 saved_at 为显示时间
                time_str = ""
                if saved_at and len(saved_at) >= 16:
                    time_str = saved_at[11:16]
                title = f"对话 {saved_at[5:16]}" if saved_at else "会话"
                preview = s.get("preview", "")
                self._session_panel.add_session(
                    title, preview=preview, time_str=time_str, active=False,
                )
        except Exception:
            pass

    # ----------------------- 会话面板 -----------------------

    def toggle_session_panel(self):
        if self._session_open:
            self.hide_session_panel()
        else:
            self.show_session_panel()

    def show_session_panel(self):
        self._session_open = True
        # 打开前刷新列表（新保存的会话可能已出现）
        self._session_panel.clear_sessions()
        self._session_ids.clear()
        self._load_sessions()
        self._resize_session_panel()
        self._session_panel.show()
        self._session_panel.raise_()

    def hide_session_panel(self):
        self._session_open = False
        self._session_panel.hide()

    def new_session(self):
        """新建会话"""
        self._current_session_id = None
        self.clear_messages()
        self.hide_session_panel()