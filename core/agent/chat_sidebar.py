"""
AI Chat 侧栏 — 可折叠的 Agent 对话界面

位于主窗口右侧，通过标题栏按钮控制显示/隐藏。
支持：消息列表、输入框、建议卡片、工具调用显示。
"""

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTextEdit,
    QScrollArea, QWidget, QSizePolicy
)
from qfluentwidgets import (
    PrimaryPushButton, TransparentToolButton, FluentIcon as FIF,
    SimpleCardWidget, StrongBodyLabel, CaptionLabel,
    PlainTextEdit, ProgressBar
)


class MessageBubble(QFrame):
    """消息气泡"""

    def __init__(self, role: str, content: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MessageBubble")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        if role == "user":
            role_label = CaptionLabel("你")
            role_label.setStyleSheet("color: #888888; font-size: 11px;")
        elif role == "assistant":
            role_label = CaptionLabel("AI 助手")
            role_label.setStyleSheet("color: #4a9eff; font-size: 11px; font-weight: bold;")
        elif role == "tool":
            role_label = CaptionLabel("工具调用")
            role_label.setStyleSheet("color: #ff9800; font-size: 11px;")
        else:
            role_label = CaptionLabel("系统")
            role_label.setStyleSheet("color: #888888; font-size: 11px;")

        layout.addWidget(role_label)

        self._content_label = StrongBodyLabel(content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._content_label)

        # 根据角色设置背景色
        if role == "user":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: rgba(74, 158, 255, 0.15);
                    border-radius: 8px;
                    border: 1px solid rgba(74, 158, 255, 0.3);
                }
            """)
        elif role == "assistant":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """)
        elif role == "tool":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: rgba(255, 152, 0, 0.1);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 152, 0, 0.3);
                }
            """)

    def append_content(self, text: str):
        """追加文本（用于流式输出）"""
        current = self._content_label.text()
        self._content_label.setText(current + text)


class SuggestionCard(SimpleCardWidget):
    """建议卡片（模式检测触发）"""

    suggestion_clicked = Signal(str)

    def __init__(self, title: str, message: str, actions: list = None, parent=None):
        super().__init__(parent)
        self.setObjectName("SuggestionCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title_label = StrongBodyLabel(title)
        title_label.setStyleSheet("color: #4a9eff; font-weight: bold;")
        layout.addWidget(title_label)

        msg_label = CaptionLabel(message)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        if actions:
            btn_layout = QHBoxLayout()
            for text, callback in actions:
                btn = PrimaryPushButton(text)
                btn.setFixedHeight(28)
                btn.clicked.connect(lambda checked, t=text: self.suggestion_clicked.emit(t))
                btn_layout.addWidget(btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

        self.setStyleSheet("""
            SuggestionCard {
                background-color: rgba(74, 158, 255, 0.08);
                border: 1px solid rgba(74, 158, 255, 0.2);
                border-radius: 8px;
            }
        """)


class ChatSidebar(QFrame):
    """AI Chat 侧栏"""

    message_sent = Signal(str)  # 用户发送消息
    debug_requested = Signal()  # 用户请求打开调试窗口

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChatSidebar")
        self.setFixedWidth(380)
        self.setMinimumWidth(380)

        self._messages = []
        self._current_assistant_bubble = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        header = QFrame()
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 8, 0)

        title = StrongBodyLabel("AI 助手")
        header_layout.addWidget(title)
        header_layout.addStretch()

        # 调试按钮
        debug_btn = TransparentToolButton(FIF.DEVELOPER_TOOLS)
        debug_btn.setFixedSize(28, 28)
        debug_btn.setToolTip("系统提示词调试")
        debug_btn.clicked.connect(self.debug_requested.emit)
        header_layout.addWidget(debug_btn)

        # 清空按钮
        clear_btn = TransparentToolButton(FIF.DELETE)
        clear_btn.setFixedSize(28, 28)
        clear_btn.setToolTip("清空对话")
        clear_btn.clicked.connect(self.clear_messages)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: rgba(255,255,255,0.1);")
        layout.addWidget(separator)

        # 消息列表（可滚动）
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._messages_widget = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(8, 8, 8, 8)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()

        self._scroll_area.setWidget(self._messages_widget)
        layout.addWidget(self._scroll_area, 1)

        # 进度条（Agent 执行中时显示）
        self._progress = ProgressBar()
        self._progress.setFixedHeight(3)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # 输入区域
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(6)

        self._input = PlainTextEdit()
        self._input.setPlaceholderText("输入消息...")
        self._input.setFixedHeight(60)
        self._input.setMaximumHeight(60)
        self._input.setStyleSheet("""
            PlainTextEdit {
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
        """)
        # Ctrl+Enter 发送
        self._input.installEventFilter(self)
        input_layout.addWidget(self._input, 1)

        send_btn = PrimaryPushButton("发送")
        send_btn.setFixedWidth(60)
        send_btn.setFixedHeight(60)
        send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(send_btn)

        layout.addWidget(input_frame)

        # 全局样式
        self.setStyleSheet("""
            ChatSidebar {
                background-color: rgba(30, 30, 30, 0.95);
                border-left: 1px solid #333333;
            }
        """)

    def eventFilter(self, obj, event):
        """拦截 Ctrl+Enter 发送"""
        if obj == self._input and event.type() == event.Type.KeyPress:
            if (event.key() == Qt.Key.Key_Return and
                    event.modifiers() == Qt.KeyboardModifier.ControlModifier):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _on_send(self):
        """发送消息"""
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self.add_message("user", text)
        self.message_sent.emit(text)
        self._show_thinking(True)

    def add_message(self, role: str, content: str):
        """添加消息到列表"""
        bubble = MessageBubble(role, content)
        # 插入到 stretch 之前
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, bubble)
        self._messages.append({"role": role, "content": content})

        # 滚动到底部
        self._scroll_to_bottom()

    def start_assistant_message(self, initial_text: str = ""):
        """开始一个新的助手消息气泡（用于流式输出）"""
        self._current_assistant_bubble = MessageBubble("assistant", initial_text)
        self._messages_layout.insertWidget(
            self._messages_layout.count() - 1, self._current_assistant_bubble
        )
        self._messages.append({"role": "assistant", "content": initial_text})
        self._scroll_to_bottom()

    def append_to_assistant(self, text: str):
        """追加文本到当前正在流式输出的助手消息"""
        if self._current_assistant_bubble is None:
            self.start_assistant_message(text)
        else:
            self._current_assistant_bubble.append_content(text)
            if self._messages:
                self._messages[-1]["content"] += text
        self._scroll_to_bottom()

    def finalize_assistant(self):
        """完成当前助手消息，停止流式追加"""
        self._current_assistant_bubble = None

    def _scroll_to_bottom(self):
        """滚动消息列表到底部"""
        self._scroll_area.verticalScrollBar().setValue(
            self._scroll_area.verticalScrollBar().maximum()
        )

    def add_suggestion(self, title: str, message: str, actions: list = None):
        """添加建议卡片"""
        card = SuggestionCard(title, message, actions)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, card)

    def show_thinking(self, thinking: bool):
        """显示/隐藏思考状态"""
        self._show_thinking(thinking)

    def _show_thinking(self, thinking: bool):
        self._progress.setVisible(thinking)
        if thinking:
            self._progress.setRange(0, 0)  # 无限进度条

    def clear_messages(self):
        """清空消息列表"""
        while self._messages_layout.count() > 1:  # 保留 stretch
            item = self._messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._messages.clear()
