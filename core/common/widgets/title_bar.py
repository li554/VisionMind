"""
自定义无边框窗口标题栏
- 包含菜单栏、窗口标题、窗口控制按钮
- 支持拖动、双击最大化、边缘调整大小
"""
import os
import sys
from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal, QTimer
from PySide6.QtGui import QCursor, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QToolButton,
    QMenuBar, QMenu, QSizeGrip, QApplication, QMainWindow, QFrame, QStatusBar
)

from core.common.theme_manager import theme_manager


class WindowButton(QPushButton):
    """窗口控制按钮（最小化、最大化、关闭）- 绘制 VSCode 风格图标"""

    def __init__(self, parent=None, btn_type="normal"):
        super().__init__(parent)
        self._btn_type = btn_type
        self.setFixedSize(46, 32)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setObjectName(f"WindowBtn_{btn_type}")
        self._is_maximized = False
        self._set_icon()

    def set_maximized(self, maximized):
        """更新最大化/还原图标状态"""
        if self._btn_type == "maximize":
            self._is_maximized = maximized
            self._set_icon()

    def _set_icon(self):
        from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon
        from PySide6.QtCore import Qt, QRectF

        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 根据当前主题选择图标颜色
        is_dark = theme_manager.is_dark_theme()
        color = QColor(200, 200, 200) if is_dark else QColor(80, 80, 80)
        pen = QPen(color, 4.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        margin = 18
        if self._btn_type == "minimize":
            painter.drawLine(margin, size // 2, size - margin, size // 2)
        elif self._btn_type == "maximize":
            if self._is_maximized:
                # 还原图标：两个重叠矩形
                outer = QRectF(margin - 4, margin + 4, size - 2 * margin, size - 2 * margin - 8)
                inner = QRectF(margin + 4, margin - 4, size - 2 * margin - 8, size - 2 * margin - 8)
                painter.drawRect(outer)
                painter.drawRect(inner)
            else:
                # 最大化图标：单个矩形
                rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
                painter.drawRect(rect)
        elif self._btn_type == "close":
            offset = margin - 2
            painter.drawLine(offset, offset, size - offset, size - offset)
            painter.drawLine(size - offset, offset, offset, size - offset)

        painter.end()
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(16, 16))


class CustomTitleBar(QWidget):
    """
    自定义标题栏
    包含：菜单栏 | 拖动区域 | 窗口控制按钮
    """
    menu_bar_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent
        self._is_maximized = False
        self._drag_pos = None
        self._drag_maximized = False
        self._restore_pending = False
        self._restore_target = None
        self._intent_widget = None
        self.setFixedHeight(40)
        self.setObjectName("CustomTitleBar")

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧：应用图标 + 应用名称 + 菜单栏
        left_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # 应用图标
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(20, 20)
        self._icon_label.setScaledContents(True)
        self._load_app_icon()
        left_layout.addWidget(self._icon_label)

        # 应用名称
        self._title_label = QLabel()
        self._title_label.setObjectName("TitleBarAppName")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if self._main_window:
            self._title_label.setText(self._main_window.windowTitle())
        left_layout.addWidget(self._title_label)

        # 菜单栏
        self._menu_bar = QMenuBar()
        self._menu_bar.setObjectName("GlobalMenuBar")
        self._menu_bar.setFixedHeight(40)
        left_layout.addWidget(self._menu_bar)

        layout.addWidget(left_widget, 0)

        # 中间：可拖动 + 意图胶囊区域
        self._drag_area = QWidget()
        self._drag_area.setObjectName("DragArea")
        self._drag_area.mousePressEvent = self._drag_area_mousePressEvent
        self._drag_area.mouseMoveEvent = self._drag_area_mouse_move_event
        self._drag_area.mouseReleaseEvent = self._drag_area_mouse_release_event
        self._drag_area.mouseDoubleClickEvent = self._drag_area_mouse_double_click
        layout.addWidget(self._drag_area, 1)

        # 定期检查容器尺寸变化，重新定位意图胶囊（确保居中）
        self._resize_observer = QTimer(self)
        self._resize_observer.setInterval(50)
        self._resize_observer.timeout.connect(self._reposition_intent_if_needed)
        self._resize_observer.start()
        self._last_drag_area_size = None

        # 录制状态指示器（位于窗口控制按钮左侧）
        self._recording_indicator = QLabel("  ● REC 0")
        self._recording_indicator.setObjectName("RecordingIndicator")
        self._recording_indicator.setVisible(False)
        self._recording_indicator.setToolTip("正在录制操作... 点击菜单 停止录制 停止")
        self._recording_indicator.setStyleSheet(
            "QLabel { color: #ff4444; font-weight: bold; padding: 2px 8px; "
            "background-color: rgba(255, 68, 68, 0.15); border-radius: 4px; }"
        )
        layout.addWidget(self._recording_indicator)

        # AI 侧栏切换按钮（qfw 按钮自绘图标，随主题自动切换颜色）
        from core.common.icons import AppIcon
        from qfluentwidgets import TransparentToolButton
        self._ai_toggle_btn = TransparentToolButton(AppIcon.AGENT)
        self._ai_toggle_btn.setFixedSize(32, 32)
        self._ai_toggle_btn.setToolTip("AI 助手")
        self._ai_toggle_btn.setCheckable(True)
        self._ai_toggle_btn.setObjectName("AiToggleBtn")
        self._ai_toggle_btn.setStyleSheet("""
            QToolButton {
                border: none;
                border-radius: 4px;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QToolButton:checked {
                background-color: rgba(74, 158, 255, 0.2);
                border: 1px solid rgba(74, 158, 255, 0.4);
            }
        """)
        self._ai_toggle_btn.clicked.connect(self._on_ai_toggle)
        layout.addWidget(self._ai_toggle_btn)

        # 右侧：窗口控制按钮
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.btn_minimize = WindowButton(self, "minimize")
        self.btn_minimize.clicked.connect(self._on_minimize)
        right_layout.addWidget(self.btn_minimize)

        self.btn_maximize = WindowButton(self, "maximize")
        self.btn_maximize.clicked.connect(self._on_maximize)
        right_layout.addWidget(self.btn_maximize)

        self.btn_close = WindowButton(self, "close")
        self.btn_close.clicked.connect(self._on_close)
        right_layout.addWidget(self.btn_close)

        layout.addWidget(right_widget, 0)

        # 主题切换时刷新按钮图标颜色
        theme_manager.theme_changed.connect(self._update_window_button_icons)

    def _update_window_button_icons(self, theme=None):
        """主题切换后重绘窗口控制按钮图标"""
        for btn in (self.btn_minimize, self.btn_maximize, self.btn_close):
            if btn:
                btn._set_icon()

    def _load_app_icon(self):
        """加载应用图标到左上角"""
        from PySide6.QtGui import QPixmap
        from core.common.config import Config

        icon_path = os.path.join(Config.RESOURCE_DIR, "icons", "app.png")
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)

        if os.path.exists(icon_path):
            source = QPixmap(icon_path)
            if not source.isNull():
                pixmap = source.scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

        self._icon_label.setPixmap(pixmap)

    def update_title(self, title):
        """更新标题栏应用名称"""
        if self._title_label:
            self._title_label.setText(title)

    # ------------------------------------------------------------------
    # 录制指示器 API
    # ------------------------------------------------------------------

    def show_recording(self, step_count=0):
        """显示录制指示器"""
        self._recording_indicator.setText(f"  ● REC {step_count}")
        self._recording_indicator.setVisible(True)

    def update_recording_count(self, step_count):
        """更新录制步数"""
        self._recording_indicator.setText(f"  ● REC {step_count}")

    def hide_recording(self):
        """隐藏录制指示器"""
        self._recording_indicator.setVisible(False)

    # ------------------------------------------------------------------
    # 菜单栏 API
    # ------------------------------------------------------------------

    def get_menu_bar(self):
        """获取菜单栏供插件添加菜单"""
        return self._menu_bar

    def clear_menus(self):
        """清空所有菜单"""
        self._menu_bar.clear()

    def add_menu(self, menu_title, actions):
        """
        添加一个菜单
        Args:
            menu_title: 菜单标题，如 "文件(&F)"
            actions: 列表，每项是 (action_text, callback) 或 None(分隔符)
        """
        menu = self._menu_bar.addMenu(menu_title)
        for item in actions:
            if item is None:
                menu.addSeparator()
            elif isinstance(item, tuple) and len(item) == 2:
                text, callback = item
                action = menu.addAction(text)
                if callback:
                    # 统一用 lambda 包裹，避免 triggered(bool) 签名不匹配
                    action.triggered.connect(lambda checked=False, cb=callback: cb())
            elif isinstance(item, dict):
                # 子菜单
                sub_menu = menu.addMenu(item['title'])
                for sub_item in item.get('actions', []):
                    if sub_item is None:
                        sub_menu.addSeparator()
                    elif isinstance(sub_item, tuple):
                        act = sub_menu.addAction(sub_item[0])
                        if sub_item[1]:
                            act.triggered.connect(lambda checked=False, cb=sub_item[1]: cb())
        self.menu_bar_changed.emit()

    def set_intent_widget(self, widget):
        """将意图胶囊嵌入标题栏（整个顶部栏）正中央"""
        self._intent_widget = widget
        widget.setParent(self)
        widget.setFixedHeight(30)
        # 用 QTimer 延迟定位，确保布局已完成
        from PySide6.QtCore import QTimer as _QTimer
        _QTimer.singleShot(0, self._reposition_intent_if_needed)

    def _reposition_intent_if_needed(self):
        """当标题栏尺寸变化时，重新将意图胶囊整体居中（含左右两侧控件）"""
        if not self._intent_widget:
            return
        current_size = self.size()
        if current_size == self._last_drag_area_size:
            return
        self._last_drag_area_size = current_size
        w = self.width()
        iw = self._intent_widget.width()
        x = max(0, (w - iw) // 2)
        self._intent_widget.move(x, (self.height() - self._intent_widget.height()) // 2)

    def recenter_intent(self):
        """胶囊内容尺寸变化后，强制重新居中（绕过拖拽区域尺寸变化检测）"""
        if not self._intent_widget:
            return
        self._last_drag_area_size = None
        self._reposition_intent_if_needed()

    # ------------------------------------------------------------------
    # 窗口控制
    # ------------------------------------------------------------------

    def _on_minimize(self):
        if self._main_window:
            self._main_window.showMinimized()

    def _on_maximize(self):
        if self._main_window:
            if self._main_window.isMaximized():
                self._main_window.showNormal()
                self.btn_maximize.set_maximized(False)
            else:
                # 最大化前先将窗口置于屏幕中心：showNormal 还原时恢复的 normalGeometry
                # 即为居中位置，双击退出全屏时窗口自然居中，不受异步几何恢复影响
                self._center_window_on_screen()
                self._main_window.showMaximized()
                self.btn_maximize.set_maximized(True)

    def _center_window_on_screen(self):
        """将窗口居中到其所在屏幕（双击还原时使用）"""
        win = self._main_window
        if not win:
            return
        screen = win.windowHandle().screen() if win.windowHandle() else QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        frame = win.frameGeometry()
        win.move(
            geo.center().x() - frame.width() // 2,
            geo.center().y() - frame.height() // 2,
        )

    def _on_close(self):
        if self._main_window:
            self._main_window.close()

    def _on_ai_toggle(self):
        """切换 AI 侧栏"""
        if self._main_window and hasattr(self._main_window, 'toggle_ai_sidebar'):
            self._main_window.toggle_ai_sidebar()

    # ------------------------------------------------------------------
    # 拖动 & 双击
    # ------------------------------------------------------------------

    def _drag_area_mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._main_window.pos()
            # 记录按下时是否处于最大化：拖动超过阈值才还原，单击不退出最大化
            self._drag_maximized = bool(self._main_window and self._main_window.isMaximized())

    def _drag_area_mouse_move_event(self, event):
        if not (self._drag_pos and self._main_window):
            return
        if self._drag_maximized:
            # 最大化状态下拖动超过阈值才还原窗口（避免单击/轻微移动误触发）
            if (event.globalPosition().toPoint() - self._drag_pos).manhattanLength() < QApplication.startDragDistance():
                return
            self._main_window.showNormal()
            self.btn_maximize.set_maximized(False)
            self._drag_maximized = False
            # 还原后沿用按下时鼠标相对窗口的偏移，保证起始抓点跟随鼠标不跳变；
            # showNormal 的几何恢复是异步的，同步 move 会被覆盖导致窗口跳到
            # normalGeometry 位置，故延迟到恢复完成后定位，后续拖动立即接管取消
            target = event.globalPosition().toPoint() - self._drag_pos
            self._restore_target = target
            self._restore_pending = True
            QTimer.singleShot(0, self._finish_restore)
            self._drag_pos = event.globalPosition().toPoint() - target
        else:
            self._restore_pending = False
            self._main_window.move(event.globalPosition().toPoint() - self._drag_pos)

    def _finish_restore(self):
        """在 showNormal 的几何恢复完成后，将窗口定位到拖拽抓点位置"""
        if not self._restore_pending or not self._main_window:
            return
        self._restore_pending = False
        target = self._restore_target
        self._restore_target = None
        if target is not None:
            self._main_window.move(target)

    def _drag_area_mouse_release_event(self, event):
        self._drag_pos = None
        self._drag_maximized = False

    def _drag_area_mouse_double_click(self, event):
        self._on_maximize()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class FramelessWindow(QMainWindow):
    """
    无边框主窗口
    - 自定义标题栏
    - 支持边缘调整大小
    - 支持拖动和双击最大化
    - 圆角窗口
    """

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 边缘调整大小的边距
        self._edge_margin = 6
        self._is_resizing = False
        self._resize_edge = None
        self._border_radius = 10

        # 外层容器（用于绘制圆角背景）
        self._outer_widget = QWidget()
        self._outer_widget.setObjectName("OuterWidget")
        self._update_outer_style()
        theme_manager.theme_changed.connect(self._update_outer_style)

        outer_layout = QVBoxLayout(self._outer_widget)
        # 留出 1px 边距，使 OuterWidget 的圆角边框完整可见
        outer_layout.setContentsMargins(1, 1, 1, 1)
        outer_layout.setSpacing(0)

        # 中央部件
        self._central_widget = QWidget()
        self._central_widget.setObjectName("MainWindowCentral")
        outer_layout.addWidget(self._central_widget)

        # 状态栏（放在 OuterWidget 内部，避免压住外层圆角边框）
        self._status_bar = QStatusBar()
        self._status_bar.setObjectName("CustomStatusBar")
        self._status_bar.setFixedHeight(24)
        outer_layout.addWidget(self._status_bar)

        self.setCentralWidget(self._outer_widget)

        self._main_layout = QVBoxLayout(self._central_widget)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # 标题栏
        self._title_bar = CustomTitleBar(self)
        self._main_layout.addWidget(self._title_bar)

        # 标题栏下方水平分隔线
        self._title_separator = QFrame()
        self._title_separator.setObjectName("TitleBarSeparator")
        self._title_separator.setFrameShape(QFrame.Shape.HLine)
        self._title_separator.setFrameShadow(QFrame.Shadow.Plain)
        self._title_separator.setFixedHeight(1)
        self._main_layout.addWidget(self._title_separator)

        # 内容区域（供子类添加）
        self._content_area = QWidget()
        self._content_area.setObjectName("ContentArea")
        self._content_layout = QVBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(self._content_area, 1)

        # 大小调整手柄
        self._size_grip = QSizeGrip(self)
        self._size_grip.setFixedSize(16, 16)
        self._size_grip.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        """绘制圆角窗口

        窗口启用 WA_TranslucentBackground,若此处不填充背景,
        子控件重排未覆盖到的区域会透出桌面(拖拽侧栏时可见闪烁)。
        按主题填充底色后再画圆角,保持圆角外仍是透明的。
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        try:
            bg = QColor("#ffffff") if theme_manager.is_light_theme() else QColor("#252525")
        except Exception:
            bg = QColor("#252525")
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), self._border_radius, self._border_radius)
        painter.end()

    def _update_outer_style(self, theme=None):
        """根据当前主题更新外层容器边框与背景"""
        if theme is None:
            theme = theme_manager.get_current_theme()
        if theme == 'light':
            bg_color = "#f8f8f8"
            border_color = "#b0b0b0"
        else:
            bg_color = "#181818"
            border_color = "#555555"
        self._outer_widget.setStyleSheet(f"""
            #OuterWidget {{
                background-color: {bg_color};
                border-radius: 10px;
                border: 1px solid {border_color};
            }}
        """)

    @property
    def menu_bar(self):
        """获取标题栏的菜单栏"""
        return self._title_bar.get_menu_bar()

    @property
    def content_layout(self):
        """获取内容区域的布局"""
        return self._content_layout

    @property
    def title_bar(self):
        return self._title_bar

    def statusBar(self):
        """返回自定义状态栏（位于 OuterWidget 内部）"""
        return self._status_bar

    def setWindowTitle(self, title):
        """同步更新自定义标题栏中的应用名称"""
        super().setWindowTitle(title)
        if hasattr(self, '_title_bar') and self._title_bar:
            self._title_bar.update_title(title)

    # ------------------------------------------------------------------
    # 边缘调整大小
    # ------------------------------------------------------------------

    def _get_resize_edge(self, pos):
        """判断鼠标在哪个边缘"""
        rect = self.rect()
        x, y = pos.x(), pos.y()
        m = self._edge_margin

        on_left = x < m
        on_right = x > rect.width() - m
        on_top = y < m
        on_bottom = y > rect.height() - m

        if on_top and on_left:
            return Qt.Edge.TopEdge | Qt.Edge.LeftEdge
        elif on_top and on_right:
            return Qt.Edge.TopEdge | Qt.Edge.RightEdge
        elif on_bottom and on_left:
            return Qt.Edge.BottomEdge | Qt.Edge.LeftEdge
        elif on_bottom and on_right:
            return Qt.Edge.BottomEdge | Qt.Edge.RightEdge
        elif on_top:
            return Qt.Edge.TopEdge
        elif on_bottom:
            return Qt.Edge.BottomEdge
        elif on_left:
            return Qt.Edge.LeftEdge
        elif on_right:
            return Qt.Edge.RightEdge
        return None

    def _get_resize_cursor(self, edge):
        """根据边缘返回对应的光标"""
        if edge is None:
            return Qt.CursorShape.ArrowCursor
        cursors = {
            Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
            Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
            Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
            Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
            Qt.Edge.TopEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeFDiagCursor,
            Qt.Edge.BottomEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeFDiagCursor,
            Qt.Edge.TopEdge | Qt.Edge.RightEdge: Qt.CursorShape.SizeBDiagCursor,
            Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: Qt.CursorShape.SizeBDiagCursor,
        }
        return cursors.get(edge, Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            edge = self._get_resize_edge(event.position().toPoint())
            if edge:
                self._is_resizing = True
                self._resize_edge = edge
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geo = self.geometry()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_resizing and self._resize_edge:
            delta = event.globalPosition().toPoint() - self._resize_start_pos
            geo = self._resize_start_geo

            new_geo = QRect(geo)
            if self._resize_edge & Qt.Edge.LeftEdge:
                new_geo.setLeft(geo.left() + delta.x())
            if self._resize_edge & Qt.Edge.RightEdge:
                new_geo.setRight(geo.right() + delta.x())
            if self._resize_edge & Qt.Edge.TopEdge:
                new_geo.setTop(geo.top() + delta.y())
            if self._resize_edge & Qt.Edge.BottomEdge:
                new_geo.setBottom(geo.bottom() + delta.y())

            if new_geo.width() >= self.minimumWidth() and new_geo.height() >= self.minimumHeight():
                self.setGeometry(new_geo)
        else:
            # 更新光标
            if not self.isMaximized():
                edge = self._get_resize_edge(event.position().toPoint())
                self.setCursor(self._get_resize_cursor(edge))
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_resizing = False
        self._resize_edge = None
        super().mouseReleaseEvent(event)
