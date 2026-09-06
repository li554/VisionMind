"""
自定义卡片组件 - 使用纯色边界线避免圆角透明问题
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QFrame

from qfluentwidgets.common.style_sheet import isDarkTheme


class SolidCardWidget(QFrame):
    """
    纯色卡片组件 - 使用纯色绘制边界线，避免圆角处透明问题
    暗色主题使用硬编码颜色，亮色主题使用QSS设置的颜色
    """
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._isClickEnabled = False
        self._borderRadius = 8
        self._isHover = False
        self._isPressed = False

    def setClickEnabled(self, isEnabled: bool):
        """设置是否启用点击效果"""
        self._isClickEnabled = isEnabled
        self.update()

    def isClickEnabled(self):
        return self._isClickEnabled

    def getBorderRadius(self):
        return self._borderRadius

    def setBorderRadius(self, radius: int):
        self._borderRadius = radius
        self.update()

    def enterEvent(self, e):
        """鼠标进入"""
        self._isHover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        """鼠标离开"""
        self._isHover = False
        self._isPressed = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        """鼠标按下"""
        self._isPressed = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        """鼠标释放"""
        self._isPressed = False
        self.update()
        self.clicked.emit()
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        """绘制卡片 - 使用纯色边界线"""
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if isDarkTheme():
            # 暗色主题：使用硬编码颜色
            bgColor = QColor(37, 37, 37)
            borderColor = QColor(51, 51, 51)
            hoverBgColor = QColor(45, 45, 45)
            hoverBorderColor = QColor(59, 130, 246)
        else:
            # 亮色主题：使用QSS设置的背景色
            qss_bg = self.palette().color(self.backgroundRole())
            bgColor = qss_bg if qss_bg.isValid() else QColor(240, 244, 248)
            borderColor = QColor(208, 224, 240)
            hoverBgColor = QColor(232, 240, 248)
            hoverBorderColor = QColor(37, 99, 235)

        # 确定当前颜色（悬停或按下状态）
        if (self._isPressed and self._isClickEnabled) or self._isHover:
            bgColor = hoverBgColor
            borderColor = hoverBorderColor

        r = self._borderRadius

        # 如果圆角为0，使用直角矩形绘制
        if r == 0:
            # 绘制边界线（纯色，不透明）
            painter.setPen(borderColor)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

            # 绘制背景（纯色）
            painter.setPen(Qt.NoPen)
            painter.setBrush(bgColor)
            painter.drawRect(self.rect().adjusted(2, 2, -2, -2))
        else:
            # 绘制边界线（纯色，不透明）
            painter.setPen(borderColor)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

            # 绘制背景（纯色）
            painter.setPen(Qt.NoPen)
            painter.setBrush(bgColor)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), max(0, r - 1), max(0, r - 1))


class SimpleSolidCardWidget(SolidCardWidget):
    """
    简单纯色卡片 - 无点击效果
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setClickEnabled(False)

    def mousePressEvent(self, e):
        """禁用点击效果"""
        super().mousePressEvent(e)
