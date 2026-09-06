"""
回放步骤进度面板

回放时显示一个浮动面板，实时展示当前执行的步骤，
让用户可以感知回放进度和当前操作。
"""

from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QScrollArea, QFrame,
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, ProgressBar,
    BodyLabel, CaptionLabel, StrongBodyLabel,
    TransparentPushButton, FluentIcon as FIF,
    isDarkTheme,
)
from typing import List, Dict, Optional


class StepItemWidget(QFrame):
    """单个步骤项"""

    def __init__(self, index: int, action: str, description: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._action = action
        self._description = description
        self._state = "pending"  # pending / running / completed / failed
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # 步骤序号
        self._index_label = CaptionLabel(f"{self._index + 1}.")
        self._index_label.setFixedWidth(24)
        layout.addWidget(self._index_label)

        # 步骤描述
        self._desc_label = BodyLabel(self._description or self._action)
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label, 1)

        # 状态图标
        self._state_label = CaptionLabel("")
        self._state_label.setFixedWidth(20)
        layout.addWidget(self._state_label)

        self._update_style()

    def set_state(self, state: str):
        """设置步骤状态: pending / running / completed / failed"""
        self._state = state
        self._update_style()

    def _update_style(self):
        dark = isDarkTheme()

        if self._state == "pending":
            color = "gray" if dark else "#999"
            self._desc_label.setStyleSheet(f"color: {color};")
            self._state_label.setText("")
        elif self._state == "running":
            color = "#00b4d8" if not dark else "#48cae4"
            self._desc_label.setStyleSheet(
                f"color: {color}; font-weight: bold;"
            )
            self._state_label.setText("▶")
            self._state_label.setStyleSheet(f"color: {color};")
        elif self._state == "completed":
            color = "#2d6a4f" if not dark else "#52b788"
            self._desc_label.setStyleSheet(f"color: {color};")
            self._state_label.setText("✓")
            self._state_label.setStyleSheet(f"color: {color};")
        elif self._state == "failed":
            color = "#e63946" if not dark else "#ff6b6b"
            self._desc_label.setStyleSheet(f"color: {color};")
            self._state_label.setText("✗")
            self._state_label.setStyleSheet(f"color: {color};")


class ReplayStepPanel(QWidget):
    """回放步骤进度面板"""

    stop_requested = Signal()
    pause_requested = Signal(bool)  # True=暂停, False=继续

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: List[Dict] = []
        self._step_widgets: List[StepItemWidget] = []
        self._current_index = -1
        self._is_paused = False
        self._drag_pos = QPoint()

        self._setup_ui()
        self._setup_window_flags()

    def _setup_window_flags(self):
        """设置窗口标志：无边框、置顶、工具窗口"""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def _setup_ui(self):
        dark = isDarkTheme()
        bg_color = "#1e1e2e" if dark else "#ffffff"
        border_color = "#45475a" if dark else "#e0e0e0"

        self.setStyleSheet(f"""
            ReplayStepPanel {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 标题栏（可拖拽）
        title_bar = QWidget()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet(f"background-color: {'#313244' if dark else '#f5f5f5'}; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 0, 8, 0)

        self._title_label = StrongBodyLabel("回放进度")
        title_layout.addWidget(self._title_label)

        title_layout.addStretch()

        self._close_btn = TransparentPushButton(FIF.CLOSE, "")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.clicked.connect(self.hide)
        title_layout.addWidget(self._close_btn)

        main_layout.addWidget(title_bar)

        # 内容区域
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(8)

        # 进度信息
        info_layout = QHBoxLayout()
        self._progress_label = BodyLabel("步骤: 0/0")
        info_layout.addWidget(self._progress_label)
        info_layout.addStretch()
        self._scenario_label = CaptionLabel("")
        info_layout.addWidget(self._scenario_label)
        content_layout.addLayout(info_layout)

        # 进度条
        self._progress_bar = ProgressBar()
        self._progress_bar.setFixedHeight(4)
        content_layout.addWidget(self._progress_bar)

        # 步骤列表（滚动区域）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(300)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(2)
        self._steps_layout.addStretch()

        scroll.setWidget(self._steps_container)
        self._scroll = scroll
        content_layout.addWidget(scroll, 1)

        # 控制按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._pause_btn = PushButton(FIF.PAUSE, "暂停")
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        btn_layout.addWidget(self._pause_btn)

        self._stop_btn = PushButton(FIF.CANCEL, "停止")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self._stop_btn)

        content_layout.addLayout(btn_layout)

        main_layout.addWidget(content)

    def set_steps(self, steps: List[Dict], scenario_name: str = ""):
        """设置步骤列表"""
        self._steps = steps
        self._scenario_label.setText(scenario_name)
        self._current_index = -1

        # 清空现有步骤
        for w in self._step_widgets:
            w.deleteLater()
        self._step_widgets.clear()

        # 创建步骤 widget
        for i, step in enumerate(steps):
            action = step.get("action", "")
            desc = step.get("description", action)
            item = StepItemWidget(i, action, desc)
            self._steps_layout.insertWidget(i, item)
            self._step_widgets.append(item)

        self._update_progress()

    def set_current_step(self, index: int):
        """设置当前执行步骤"""
        if self._current_index >= 0 and self._current_index < len(self._step_widgets):
            # 上一步未标记完成，标记为 completed
            pass

        self._current_index = index

        if 0 <= index < len(self._step_widgets):
            self._step_widgets[index].set_state("running")
            # 滚动到当前步骤
            self._scroll.ensureWidgetVisible(self._step_widgets[index])

        self._update_progress()

    def set_step_completed(self, index: int, success: bool = True):
        """标记步骤完成"""
        if 0 <= index < len(self._step_widgets):
            self._step_widgets[index].set_state("completed" if success else "failed")

    def _update_progress(self):
        """更新进度显示"""
        total = len(self._steps)
        current = self._current_index + 1 if self._current_index >= 0 else 0
        self._progress_label.setText(f"步骤: {current}/{total}")

        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))
        else:
            self._progress_bar.setValue(0)

    def _on_pause_clicked(self):
        """暂停/继续按钮"""
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_btn.setIcon(FIF.PLAY)
            self._pause_btn.setText("继续")
        else:
            self._pause_btn.setIcon(FIF.PAUSE)
            self._pause_btn.setText("暂停")
        self.pause_requested.emit(self._is_paused)

    def _on_stop_clicked(self):
        """停止按钮"""
        self.stop_requested.emit()
        self.hide()

    # ---- 拖拽支持 ----

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()

    def show_panel(self, x: int = 100, y: int = 100):
        """显示面板在指定位置"""
        self.move(x, y)
        self.show()
        self.raise_()

    def hide_panel(self):
        """隐藏面板"""
        self.hide()
