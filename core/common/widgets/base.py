from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QDialog, QHBoxLayout
from qfluentwidgets import IndeterminateProgressRing, BodyLabel, ProgressBar, SubtitleLabel, PushButton


class BaseDialog(QDialog):
    """对话框基类：按空格键触发 accept（等同点击确定）"""

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            # 当焦点在可编辑控件上时不拦截，允许输入空格
            focus_widget = self.focusWidget()
            if focus_widget and focus_widget.metaObject().className() in (
                'LineEdit', 'QLineEdit', 'TextEdit', 'QTextEdit', 'PlainTextEdit',
            ):
                super().keyPressEvent(event)
                return
            self.accept()
            return
        super().keyPressEvent(event)


class ProgressDialog(QDialog):
    canceled = Signal()

    def __init__(self, title, message, parent=None, has_cancel=False, cancel_text="取消"):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 190 if has_cancel else 150)
        # Remove context help button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._is_canceled = False

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)

        self.title_label = SubtitleLabel(title, self)
        self.message_label = BodyLabel(message, self)
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.message_label)
        self.layout.addWidget(self.progress_bar)

        if has_cancel:
            self.btn_layout = QHBoxLayout()
            self.btn_layout.addStretch(1)
            self.cancel_btn = PushButton(cancel_text, self)
            self.cancel_btn.clicked.connect(self._on_cancel)
            self.btn_layout.addWidget(self.cancel_btn)
            self.layout.addLayout(self.btn_layout)

    def setValue(self, value):
        self.progress_bar.setValue(value)

    def setLabelText(self, text):
        self.message_label.setText(text)

    def setRange(self, minimum, maximum):
        self.progress_bar.setRange(minimum, maximum)

    def setMinimumDuration(self, ms):
        pass

    @Slot()
    def _on_cancel(self):
        self._is_canceled = True
        self.canceled.emit()
        self.close()

    def wasCanceled(self):
        return self._is_canceled

class Interface(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

class LoadingMask(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("loadingMask")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(15)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_ring = IndeterminateProgressRing(self)
        self.progress_ring.setFixedSize(60, 60)
        self.progress_ring.setStrokeWidth(4)

        self.tip_label = BodyLabel("正在生成缺陷，请稍候...", self)
        self.tip_label.setProperty("class", "LoadingTip")

        self.layout.addWidget(self.progress_ring, 0, Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.tip_label, 0, Qt.AlignmentFlag.AlignCenter)

        # Initially hidden
        self.hide()

    def showEvent(self, event):
        super().showEvent(event)
        if self.parent():
            self.resize(self.parent().size())
        self.progress_ring.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.progress_ring.stop()
