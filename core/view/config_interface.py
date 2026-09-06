"""
VS Code 风格的配置界面
- 左栏：可折叠的分类导航树
- 右栏：设置内容区域
- 搜索过滤
- 自动保存
"""
import os
import json
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QScrollArea,
    QStackedWidget, QFrame, QLabel, QLineEdit, QPushButton,
    QInputDialog, QMessageBox, QPlainTextEdit, QRadioButton, QButtonGroup
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QKeySequence
from qfluentwidgets import (
    StrongBodyLabel, CaptionLabel, BodyLabel,
    PushButton, PrimaryPushButton,
    ComboBox as FluentComboBox, SpinBox, LineEdit,
    CheckBox, Slider, InfoBar, FluentIcon as FIF, SwitchButton,
    SearchLineEdit, TransparentToolButton
)
from core.common.widgets.base import Interface
from core.common.icons import AppIcon
from core.common.settings import settings
from core.common.project_settings import project_settings
from core.common.config import Config
from core.common.theme_manager import theme_manager

from core.common.action_registry import action
from core.service.project_service import TASK_TYPE_DISPLAY, TASK_TYPE_DEFAULT_EXPORT


# ============================================================
# 设计令牌
# ============================================================
ROW_HEIGHT = 32
CATEGORY_WIDTH = 220
ITEM_PADDING_V = 12
ITEM_PADDING_H = 16
SEPARATOR_COLOR_LIGHT = (208, 224, 240)
SEPARATOR_COLOR_DARK = (51, 51, 51)


# ============================================================
# SettingItem - 单个设置项
# ============================================================
class SettingItem(QWidget):
    """
    VS Code 风格的设置项
    布局: 标题(类别: 名称) → 描述(灰色) → 控件（垂直堆叠）
    """
    setting_changed = Signal(str, object)

    def __init__(self, key, title, description, control, parent=None):
        super().__init__(parent)
        self._key = key
        self._control = control
        self._title_text = title
        self._description_text = description
        self.setObjectName('SettingItem')
        self._init_ui(title, description, control)

    def _init_ui(self, title, description, control):
        # 两栏布局：左侧标题+描述，右侧控件（垂直居中、右对齐）
        layout = QHBoxLayout(self)
        layout.setContentsMargins(ITEM_PADDING_H, 14, ITEM_PADDING_H, 14)
        layout.setSpacing(24)

        # 左侧：标题 + 描述
        text_column = QVBoxLayout()
        text_column.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName('SettingTitle')
        title_label.setWordWrap(True)
        text_column.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName('SettingDescription')
            desc_label.setWordWrap(True)
            text_column.addWidget(desc_label)

        text_column.addStretch(1)
        layout.addLayout(text_column, 1)

        # 右侧：控件（垂直居中）
        control_column = QVBoxLayout()
        control_column.setSpacing(0)
        control_column.addStretch(1)
        control_column.addWidget(control)
        control_column.addStretch(1)
        layout.addLayout(control_column, 0)

    def filter_text(self, query):
        """检查是否匹配搜索关键词"""
        if not query:
            return True
        query_lower = query.lower()
        return (query_lower in self._title_text.lower() or
                query_lower in self._description_text.lower() or
                query_lower in self._key.lower())


# ============================================================
# SettingSection - 可折叠的设置分组
# ============================================================
class SettingSection(QWidget):
    """
    可折叠的设置分组
    点击标题可展开/折叠
    """
    toggled = Signal(bool)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._is_expanded = True
        self._items = []
        self.setObjectName('SettingSection')
        self._init_ui(title)

    def _init_ui(self, title):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # 分组标题（可点击折叠）
        self._header = QWidget()
        self._header.setObjectName('SectionHeader')
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setFixedHeight(40)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(ITEM_PADDING_H, 0, ITEM_PADDING_H, 0)

        # 折叠箭头
        self._arrow_label = QLabel('▼')
        self._arrow_label.setObjectName('CollapseArrow')
        self._arrow_label.setFixedWidth(16)
        header_layout.addWidget(self._arrow_label)

        # 标题
        self._title_label = QLabel(title)
        self._title_label.setObjectName('SectionTitle')
        header_layout.addWidget(self._title_label)
        header_layout.addStretch(1)

        self._header.mousePressEvent = self._toggle
        self._layout.addWidget(self._header)

        # 内容容器
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._layout.addWidget(self._content)

    def _toggle(self, event=None):
        self._is_expanded = not self._is_expanded
        self._content.setVisible(self._is_expanded)
        self._arrow_label.setText('▼' if self._is_expanded else '▶')
        self.toggled.emit(self._is_expanded)

    def add_item(self, item):
        self._items.append(item)
        self._content_layout.addWidget(item)

    def items(self):
        return self._items

    def filter_items(self, query):
        """过滤设置项，返回是否有匹配"""
        has_match = False
        for item in self._items:
            if hasattr(item, 'filter_text'):
                match = item.filter_text(query)
            else:
                match = not query  # 无filter_text的项仅在无搜索词时可见
            item.setVisible(match)
            if match:
                has_match = True
        # 如果有匹配项，展开分组
        if has_match and query:
            if not self._is_expanded:
                self._toggle()
        return has_match


# ============================================================
# _SectionCard - 设置分组卡片容器
# ============================================================
class _SectionCard(QFrame):
    """把一个 SettingSection 包装成圆角卡片（现代设置页风格）"""

    def __init__(self, section, parent=None):
        super().__init__(parent)
        self.setObjectName('SectionCard')
        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(4, 0, 4, 4)
        card_layout.setSpacing(0)
        card_layout.addWidget(section)


# ============================================================
# SettingsContentPage - 可滚动的设置内容页
# ============================================================
class SettingsContentPage(QScrollArea):
    """一个分类的设置内容页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections = []
        self._init_ui()

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setObjectName('SettingsContentPage')

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(32, 8, 32, 32)
        self._container_layout.setSpacing(16)

        self._container_layout.addStretch(1)
        self.setWidget(self._container)

    def add_section(self, section):
        self._sections.append(section)
        idx = self._container_layout.count() - 1
        self._container_layout.insertWidget(idx, _SectionCard(section))

    def add_page_header(self, title, description=''):
        """在页面顶部插入大标题页头"""
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(32, 16, 32, 8)
        header_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName('PageTitle')
        header_layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setObjectName('PageDesc')
            desc_label.setWordWrap(True)
            header_layout.addWidget(desc_label)

        self._container_layout.insertWidget(0, header)

    def filter(self, query):
        """过滤所有设置项，返回是否有匹配"""
        has_match = False
        for section in self._sections:
            if section.filter_items(query):
                has_match = True
        return has_match


# ============================================================
# SettingsCategoryTree - 左侧分类导航树
# ============================================================
class SettingsCategoryTree(QWidget):
    """
    左侧分类导航树，支持可折叠分组
    """
    category_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('SettingsCategoryTree')
        self.setFixedWidth(CATEGORY_WIDTH)
        self._categories = {}
        self._groups = {}
        self._current_category = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setObjectName('CategoryTreeScroll')
        # viewport 默认用浅色调色板自填充，在暗色主题下形成白底；
        # 强制透明以融入面板背景
        viewport = self._scroll.viewport()
        viewport.setAutoFillBackground(False)
        viewport.setStyleSheet('background: transparent;')

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(0)
        self._container_layout.addStretch(1)

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

    def add_group(self, group_id, group_title):
        """添加可折叠分组"""
        group = _CategoryGroup(group_id, group_title, self)
        self._groups[group_id] = group
        idx = self._container_layout.count() - 1
        self._container_layout.insertWidget(idx, group)
        group.category_clicked.connect(self._on_category_clicked)
        return group

    def add_category(self, group_id, category_id, category_title):
        """向分组添加分类"""
        if group_id in self._groups:
            self._groups[group_id].add_category(category_id, category_title)
            self._categories[category_id] = group_id

    def _on_category_clicked(self, category_id):
        if self._current_category:
            prev_group = self._groups.get(self._categories.get(self._current_category))
            if prev_group:
                prev_group.set_category_selected(self._current_category, False)

        self._current_category = category_id
        group_id = self._categories.get(category_id)
        if group_id and group_id in self._groups:
            self._groups[group_id].set_category_selected(category_id, True)

        self.category_changed.emit(category_id)

    def select_category(self, category_id):
        """程序化选择分类"""
        if category_id in self._categories:
            self._on_category_clicked(category_id)

    def get_current_category(self):
        return self._current_category


class _CategoryGroup(QWidget):
    """导航树中的可折叠分组"""
    category_clicked = Signal(str)

    def __init__(self, group_id, title, parent=None):
        super().__init__(parent)
        self._group_id = group_id
        self._is_expanded = True
        self._categories = {}
        self.setObjectName('CategoryGroup')
        self._init_ui(title)

    def _init_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 分组标题
        self._header = QWidget()
        self._header.setObjectName('GroupHeader')
        self._header.setFixedHeight(32)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 0, 12, 0)

        self._arrow = QLabel('▼')
        self._arrow.setObjectName('GroupArrow')
        self._arrow.setFixedWidth(14)
        header_layout.addWidget(self._arrow)

        title_label = QLabel(title)
        title_label.setObjectName('GroupTitle')
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        self._header.mousePressEvent = self._toggle
        layout.addWidget(self._header)

        # 分类容器
        self._container = QWidget()
        self._container.setObjectName('GroupContainer')
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 4)
        self._container_layout.setSpacing(0)
        layout.addWidget(self._container)

    def _toggle(self, event=None):
        self._is_expanded = not self._is_expanded
        self._container.setVisible(self._is_expanded)
        self._arrow.setText('▼' if self._is_expanded else '▶')

    def add_category(self, category_id, category_title):
        btn = _CategoryButton(category_id, category_title)
        self._categories[category_id] = btn
        btn.clicked.connect(lambda: self.category_clicked.emit(category_id))
        self._container_layout.addWidget(btn)

    def set_category_selected(self, category_id, selected):
        if category_id in self._categories:
            self._categories[category_id].set_selected(selected)


class _CategoryButton(QWidget):
    """分类导航按钮"""
    clicked = Signal()

    def __init__(self, category_id, title, parent=None):
        super().__init__(parent)
        self._category_id = category_id
        self._is_selected = False
        self.setObjectName('CategoryButton')
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 0, 12, 0)

        self._title = QLabel(title)
        self._title.setObjectName('CategoryButtonTitle')
        layout.addWidget(self._title)

    def set_selected(self, selected):
        self._is_selected = selected
        self.setProperty('selected', selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if self._is_selected:
            from qfluentwidgets.common.style_sheet import isDarkTheme
            if isDarkTheme():
                painter.setPen(QPen(QColor(37, 99, 235), 2))
            else:
                painter.setPen(QPen(QColor(37, 99, 235), 2))
            painter.drawLine(0, 4, 0, self.height() - 4)

        super().paintEvent(event)


# ============================================================
# KeySequenceEdit - 快捷键录制控件
# ============================================================
class KeySequenceEdit(QLineEdit):
    """快捷键输入控件：点击后进入录制状态，按键后显示快捷键"""
    key_sequence_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recording = False
        self._key_text = ""
        self.setReadOnly(True)
        self.setPlaceholderText("点击设置快捷键")
        self.setFixedHeight(ROW_HEIGHT)
        self.setMinimumWidth(120)

    def set_key(self, key_text: str):
        """设置快捷键显示"""
        self._key_text = key_text
        self.setText(key_text)

    def mousePressEvent(self, event):
        """点击进入录制状态"""
        self._recording = True
        self.setText("请按键...")
        self.setStyleSheet("border: 2px solid #0078d4; border-radius: 4px;")
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if not self._recording:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # 忽略单独的修饰键
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt,
                   Qt.Key.Key_Meta, Qt.Key.Key_No):
            return

        # 构建快捷键字符串
        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")

        # 键名映射
        key_map = {
            Qt.Key.Key_Space: "Space",
            Qt.Key.Key_Up: "Up",
            Qt.Key.Key_Down: "Down",
            Qt.Key.Key_Left: "Left",
            Qt.Key.Key_Right: "Right",
            Qt.Key.Key_Return: "Return",
            Qt.Key.Key_Enter: "Enter",
            Qt.Key.Key_Escape: "Escape",
            Qt.Key.Key_Backspace: "Backspace",
            Qt.Key.Key_Delete: "Delete",
            Qt.Key.Key_Tab: "Tab",
            Qt.Key.Key_Home: "Home",
            Qt.Key.Key_End: "End",
            Qt.Key.Key_PageUp: "PageUp",
            Qt.Key.Key_PageDown: "PageDown",
            Qt.Key.Key_Insert: "Insert",
            Qt.Key.Key_F1: "F1", Qt.Key.Key_F2: "F2", Qt.Key.Key_F3: "F3",
            Qt.Key.Key_F4: "F4", Qt.Key.Key_F5: "F5", Qt.Key.Key_F6: "F6",
            Qt.Key.Key_F7: "F7", Qt.Key.Key_F8: "F8", Qt.Key.Key_F9: "F9",
            Qt.Key.Key_F10: "F10", Qt.Key.Key_F11: "F11", Qt.Key.Key_F12: "F12",
        }

        if key in key_map:
            parts.append(key_map[key])
        elif 32 <= key <= 127:
            # 普通可打印字符
            parts.append(chr(key).upper())
        else:
            # 不支持的键，取消录制
            self._cancel_recording()
            return

        key_str = "+".join(parts)
        self._recording = False
        self._key_text = key_str
        self.setText(key_str)
        self.setStyleSheet("")
        self.key_sequence_changed.emit(key_str)

    def focusOutEvent(self, event):
        """失去焦点时取消录制"""
        if self._recording:
            self._cancel_recording()
        super().focusOutEvent(event)

    def _cancel_recording(self):
        self._recording = False
        self.setText(self._key_text)
        self.setStyleSheet("")

    def key(self) -> str:
        return self._key_text


# ============================================================
# ShortcutItem - 单个快捷键设置项
# ============================================================
class ShortcutItem(QWidget):
    """快捷键设置项：描述 + 快捷键输入 + 重置按钮"""
    shortcut_changed = Signal(str, str)  # (action_id, new_key)

    def __init__(self, action_id, description, default_key, current_key, parent=None):
        super().__init__(parent)
        self._action_id = action_id
        self._default_key = default_key
        self.setObjectName('SettingItem')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(ITEM_PADDING_H, 10, ITEM_PADDING_H, 10)
        row.setSpacing(12)

        # 描述
        desc_label = QLabel(description)
        desc_label.setObjectName('SettingTitle')
        row.addWidget(desc_label, 1)

        # 默认键提示
        if current_key != default_key:
            default_hint = CaptionLabel(f"(默认: {default_key})")
            default_hint.setObjectName('SettingDescription')
            row.addWidget(default_hint)

        # 快捷键输入
        self._key_edit = KeySequenceEdit()
        self._key_edit.set_key(current_key)
        self._key_edit.key_sequence_changed.connect(self._on_key_changed)
        row.addWidget(self._key_edit)

        # 重置按钮
        self._reset_btn = PushButton("重置")
        self._reset_btn.setFixedHeight(28)
        self._reset_btn.setFixedWidth(60)
        self._reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self._reset_btn)

        outer.addLayout(row)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName('SettingSeparator')
        outer.addWidget(separator)

    def _on_key_changed(self, new_key):
        self.shortcut_changed.emit(self._action_id, new_key)

    def _on_reset(self):
        self._key_edit.set_key(self._default_key)
        self.shortcut_changed.emit(self._action_id, self._default_key)

    def set_key(self, key: str):
        self._key_edit.set_key(key)

    def filter_text(self, query):
        if not query:
            return True
        q = query.lower()
        return q in self._action_id.lower() or q in self._key_edit.key().lower()


# ============================================================
# AIProviderSection - 按提供商分组的多模型管理面板
# ============================================================
class ModelRowEdit(QFrame):
    """单个模型参数行：名称 / 输入上限 / 输出上限 / 支持图像 / 默认（无重复列标题）。"""

    remove_requested = Signal()

    def __init__(self, model=None, is_default=False, parent=None):
        super().__init__(parent)
        self.setObjectName('ModelRowEdit')
        name = str(model.get("name", "")) if isinstance(model, dict) else str(model or "")
        max_in = (model.get("max_input_tokens", 0) if isinstance(model, dict) else 0) or 16384
        max_out = (model.get("max_output_tokens", 0) if isinstance(model, dict) else 0) or 4096
        vision = bool(model.get("supports_image", False)) if isinstance(model, dict) else False

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(8)

        # 模型名称
        self._name = QLineEdit(name)
        self._name.setPlaceholderText('模型名称，如 gpt-4o')
        self._name.setFixedHeight(ROW_HEIGHT)
        row.addWidget(self._name, 3)

        # 输入 Token
        self._max_in = SpinBox()
        self._max_in.setRange(1, 16_000_000)
        self._max_in.setValue(max_in)
        self._max_in.setFixedHeight(ROW_HEIGHT)
        self._max_in.setToolTip('最大输入 token 数')
        row.addWidget(self._max_in, 1)

        # 输出 Token
        self._max_out = SpinBox()
        self._max_out.setRange(1, 16_000_000)
        self._max_out.setValue(max_out)
        self._max_out.setFixedHeight(ROW_HEIGHT)
        self._max_out.setToolTip('最大输出 token 数')
        row.addWidget(self._max_out, 1)

        # 图像（无文字标签，与表头对齐）
        self._vision = CheckBox()
        self._vision.setChecked(vision)
        self._vision.setToolTip('是否支持图像 / 多模态输入')
        self._vision.setFixedSize(20, ROW_HEIGHT)
        row.addWidget(self._vision)

        # 默认（无文字标签，与表头对齐）
        self._default = QRadioButton()
        self._default.setChecked(True if is_default else False)
        self._default.setToolTip('设为该提供商的默认模型')
        self._default.setFixedSize(20, ROW_HEIGHT)
        row.addWidget(self._default)

        # 删除按钮（小图标）
        self._remove_btn = TransparentToolButton(FIF.DELETE)
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setToolTip('删除该模型')
        self._remove_btn.clicked.connect(self.remove_requested.emit)
        row.addWidget(self._remove_btn)

    def is_default(self):
        return self._default.isChecked()

    def set_default(self, on: bool):
        self._default.blockSignals(True)
        self._default.setChecked(on)
        self._default.blockSignals(False)

    def to_model(self) -> dict:
        return {
            "name": self._name.text().strip(),
            "max_input_tokens": self._max_in.value(),
            "max_output_tokens": self._max_out.value(),
            "supports_image": self._vision.isChecked(),
        }


# ============================================================
# AIProviderSection - 按提供商分组的多模型管理面板
# ============================================================
class AIProviderSection(QFrame):
    """AI 大模型提供商管理：新增/编辑/删除/激活提供商，并为每个模型独立配置参数。"""

    providers_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('AIProviderSection')
        self._providers = []
        self._active_id = ''
        self._current_pid = ''
        self._default_group = QButtonGroup(self)
        self._default_group.setExclusive(True)
        self._model_rows = []
        self._build_ui()
        self._reload()

    # ----------------------- UI -----------------------
    def _add_field_row(self, title, description, control, extra_controls=None):
        """两栏字段行：左侧标题+描述文字，右侧控件固定宽度靠右。

        参考 SettingItem 布局，避免 label 与输入框被拉远。
        """
        row = QHBoxLayout()
        row.setSpacing(16)

        text = QVBoxLayout()
        text.setSpacing(3)
        t = QLabel(title)
        t.setObjectName('SettingTitle')
        t.setWordWrap(True)
        text.addWidget(t)
        d = QLabel(description)
        d.setObjectName('SettingDescription')
        d.setWordWrap(True)
        text.addWidget(d)
        text.addStretch(1)
        row.addLayout(text, 1)

        right = QHBoxLayout()
        right.setSpacing(6)
        if isinstance(control, (list, tuple)):
            for c in control:
                right.addWidget(c)
        else:
            right.addWidget(control)
        if extra_controls:
            for c in extra_controls:
                right.addWidget(c)
        right.addStretch(1)
        row.addLayout(right, 0)
        return row

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        # ---- 提供商 ----（含新增/激活/删除按钮）
        self._provider_combo = FluentComboBox()
        self._provider_combo.setFixedHeight(ROW_HEIGHT)
        self._provider_combo.setFixedWidth(320)
        self._provider_combo.currentIndexChanged.connect(self._load_provider_form)
        self._add_btn = PushButton('新增')
        self._add_btn.setFixedSize(64, ROW_HEIGHT)
        self._add_btn.clicked.connect(self._on_add)
        self._active_btn = PushButton('设为激活')
        self._active_btn.setFixedSize(80, ROW_HEIGHT)
        self._active_btn.clicked.connect(self._on_set_active)
        self._del_btn = PushButton('删除')
        self._del_btn.setFixedSize(64, ROW_HEIGHT)
        self._del_btn.clicked.connect(self._on_delete)
        layout.addLayout(self._add_field_row(
            '提供商',
            '选择当前激活的 AI 提供商。切换后，Agent 对话与意图分析将使用该提供商的默认模型。',
            self._provider_combo,
            [self._add_btn, self._active_btn, self._del_btn],
        ))

        # ---- 名称 ----
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText('提供商名称，如 Agnes')
        self._name_edit.setFixedHeight(ROW_HEIGHT)
        self._name_edit.setFixedWidth(320)
        layout.addLayout(self._add_field_row(
            '名称',
            '该提供商在界面中显示的名称。',
            self._name_edit,
        ))

        # ---- Base URL ----
        self._base_url_edit = QLineEdit()
        self._base_url_edit.setPlaceholderText('https://apihub.agnes-ai.com/v1')
        self._base_url_edit.setFixedHeight(ROW_HEIGHT)
        self._base_url_edit.setFixedWidth(320)
        layout.addLayout(self._add_field_row(
            'Base URL',
            'OpenAI 兼容的 API 端点地址，例如 https://api.deepseek.com。',
            self._base_url_edit,
        ))

        # ---- API Key ----
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText('API Key（仅保存在本地配置）')
        self._api_key_edit.setFixedHeight(ROW_HEIGHT)
        self._api_key_edit.setFixedWidth(320)
        layout.addLayout(self._add_field_row(
            'API Key',
            '调用 API 所需的密钥。仅保存在本地配置文件中，不会上传。',
            self._api_key_edit,
        ))

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName('Separator')
        layout.addSpacing(6)
        layout.addWidget(line)
        layout.addSpacing(6)

        # 模型区域标题 + 添加按钮
        model_head = QHBoxLayout()
        model_head.addWidget(StrongBodyLabel('模型'))
        model_head.addStretch(1)
        self._add_model_btn = PushButton('添加模型')
        self._add_model_btn.clicked.connect(self._on_add_model)
        model_head.addWidget(self._add_model_btn)
        layout.addLayout(model_head)

        # 统一表头行（与 ModelRowEdit 控件对齐）
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.setSpacing(8)
        header.addWidget(CaptionLabel('模型名称'), 3)
        header.addWidget(CaptionLabel('输入 Token'), 1)
        header.addWidget(CaptionLabel('输出 Token'), 1)
        header.addWidget(CaptionLabel('图像'), 0)
        header.addWidget(CaptionLabel('默认'), 0)
        header.addSpacing(24 + 8)  # 删除按钮宽度 + spacing
        layout.addLayout(header)

        # 模型行列表
        self._model_list = QWidget()
        self._model_layout = QVBoxLayout(self._model_list)
        self._model_layout.setContentsMargins(0, 0, 0, 0)
        self._model_layout.setSpacing(4)
        layout.addWidget(self._model_list)

        self._save_btn = PrimaryPushButton('保存更改')
        self._save_btn.clicked.connect(self._on_save)
        layout.addWidget(self._save_btn)

        tip = QLabel('每个模型可独立配置输入/输出 token 上限与是否支持图像；"●" 标记激活提供商，单选"默认"标出默认模型。')
        tip.setObjectName('PageDesc')
        tip.setWordWrap(True)
        layout.addWidget(tip)

    # ----------------------- 数据加载 -----------------------
    def _reload(self):
        from core.common.ai_providers import get_active_provider, get_providers
        self._providers = get_providers()
        self._active_id = get_active_provider().get('id', '')
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        for p in self._providers:
            mark = '● ' if p.get('id') == self._active_id else ''
            self._provider_combo.addItem(
                f"{mark}{p.get('name', '')} ({p.get('base_url', '')})", p.get('id'))
        self._provider_combo.blockSignals(False)
        if self._providers:
            self._provider_combo.setCurrentIndex(0)
            self._load_provider_form()
        else:
            self._clear_form()

    def _current_provider(self):
        idx = self._provider_combo.currentIndex()
        if idx < 0 or idx >= len(self._providers):
            return None
        return self._providers[idx]

    def _clear_form(self):
        self._current_pid = ''
        self._name_edit.clear()
        self._base_url_edit.clear()
        self._api_key_edit.clear()
        self._clear_model_rows()
        self._active_btn.setEnabled(False)
        self._save_btn.setEnabled(False)

    def _clear_model_rows(self):
        for row in self._model_rows:
            self._default_group.removeButton(row._default)
            self._model_layout.removeWidget(row)
            row.deleteLater()
        self._model_rows = []

    def _load_provider_form(self):
        p = self._current_provider()
        if p is None:
            self._clear_form()
            return
        self._current_pid = p['id']
        self._name_edit.setText(p.get('name', ''))
        self._base_url_edit.setText(p.get('base_url', ''))
        self._api_key_edit.setText(p.get('api_key', ''))
        self._clear_model_rows()
        default_model = p.get('default_model', '')
        for m in p.get('models', []):
            self._append_model_row(m, is_default=isinstance(m, dict) and m.get('name') == default_model)
        self._active_btn.setEnabled(p.get('id') != self._active_id)
        self._save_btn.setEnabled(True)

    def _append_model_row(self, model=None, is_default=False):
        row = ModelRowEdit(model, is_default=is_default)
        row.remove_requested.connect(
            lambda r=row: self._model_layout.removeWidget(r) or (self._model_rows.remove(r), r.deleteLater()))
        self._default_group.addButton(row._default)
        self._default_group.setId(row._default, len(self._model_rows))
        self._model_layout.addWidget(row)
        self._model_rows.append(row)
        self._sync_row_defaults()

    def _sync_row_defaults(self):
        default = None
        for row in self._model_rows:
            if row.is_default():
                default = row
                break
        for row in self._model_rows:
            if row is not default:
                row.set_default(False)

    # ----------------------- 模型增删 -----------------------
    def _on_add_model(self):
        self._append_model_row(model={'name': '', 'max_input_tokens': 16384,
                                      'max_output_tokens': 4096, 'supports_image': False})
        self._model_rows[-1]._name.setFocus()

    # ----------------------- 操作 -----------------------
    def _on_add(self):
        from core.common.ai_providers import add_provider
        name, ok = QInputDialog.getText(self, '新增提供商', '提供商名称：')
        if not ok or not name.strip():
            return
        add_provider(name.strip(), '', '', [])
        self.providers_changed.emit()
        self._reload()
        for i, p in enumerate(self._providers):
            if p.get('name') == name.strip():
                self._provider_combo.setCurrentIndex(i)
                break

    def _on_save(self):
        p = self._current_provider()
        if p is None:
            return
        models = []
        default_model = self._current_provider().get('default_model', '')
        for row in self._model_rows:
            m = row.to_model()
            if m['name']:
                if row.is_default():
                    default_model = m['name']
                models.append(m)
        from core.common.ai_providers import update_provider
        update_provider(
            p['id'],
            name=self._name_edit.text().strip(),
            base_url=self._base_url_edit.text().strip(),
            api_key=self._api_key_edit.text().strip(),
            models=models,
            default_model=default_model,
        )
        self.providers_changed.emit()
        self._reload()
        for i, prov in enumerate(self._providers):
            if prov.get('id') == p['id']:
                self._provider_combo.setCurrentIndex(i)
                break

    def _on_delete(self):
        p = self._current_provider()
        if p is None:
            return
        ret = QMessageBox.question(
            self, '删除提供商',
            f'确定要删除提供商「{p.get("name", "")}」及其全部模型吗？')
        if ret != QMessageBox.StandardButton.Yes:
            return
        from core.common.ai_providers import delete_provider
        delete_provider(p['id'])
        self.providers_changed.emit()
        self._reload()

    def _on_set_active(self):
        p = self._current_provider()
        if p is None:
            return
        from core.common.ai_providers import set_active_provider
        set_active_provider(p['id'])
        self.providers_changed.emit()
        self._reload()


# ============================================================
# ConfigInterface - 主配置界面
# ============================================================
class ConfigInterface(Interface):
    def __init__(self, parent=None):
        super().__init__('ConfigInterface', parent)
        self.plugin_manager = None
        self.project_service = None
        self._pages = {}
        self._all_widgets = {}
        self._init_ui()

    @action("config.set_plugin_manager", description="设置插件管理器实例。\n- 内部使用，用于配置界面的插件管理功能", category="配置")
    def set_plugin_manager(self, pm):
        self.plugin_manager = pm
        self._populate_plugin_settings()

    @action("config.set_project_service", description="设置项目服务实例。\n- 内部使用，用于配置界面的项目服务功能", category="配置")
    def set_project_service(self, service):
        self.project_service = service
        self._populate_project_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部搜索栏
        search_bar = self._create_search_bar()
        main_layout.addWidget(search_bar)

        # 主内容区域
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 左侧分类树
        self._category_tree = SettingsCategoryTree()
        self._category_tree.category_changed.connect(self._on_category_changed)
        content_layout.addWidget(self._category_tree)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setObjectName('CategorySeparator')
        content_layout.addWidget(separator)

        # 右侧内容区域
        self._stacked = QStackedWidget()
        self._stacked.setObjectName('SettingsStacked')
        content_layout.addWidget(self._stacked, 1)

        main_layout.addWidget(content_widget, 1)

        # 构建所有设置页面
        self._build_all_pages()

        # 默认选中第一个分类
        first_category = next(iter(self._pages.keys()), None)
        if first_category:
            self._category_tree.select_category(first_category)

    def _create_search_bar(self):
        """创建顶部搜索栏"""
        search_widget = QWidget()
        search_widget.setObjectName('SearchBar')
        search_widget.setFixedHeight(52)

        layout = QHBoxLayout(search_widget)
        layout.setContentsMargins(24, 8, 24, 8)

        self._search_input = SearchLineEdit()
        self._search_input.setPlaceholderText('搜索设置...')
        self._search_input.setFixedHeight(36)
        self._search_input.searchButton.setIcon(AppIcon.SEARCH.icon())
        self._search_input.textChanged.connect(self._on_search_changed)
        # 覆盖 qfw 默认的主题色下划线（组件级样式需用官方覆盖接口）
        from qfluentwidgets.common.style_sheet import setCustomStyleSheet
        setCustomStyleSheet(
            self._search_input,
            lightQss=("SearchLineEdit { background: transparent; border: none;"
                      " border-bottom: 1px solid #e5e5ea; border-radius: 0; }"
                      "SearchLineEdit:hover { border-bottom: 1px solid #9ca3af; }"
                      "SearchLineEdit:focus { border-bottom: 1px solid #2563eb; }"),
            darkQss=("SearchLineEdit { background: transparent; border: none;"
                     " border-bottom: 1px solid #333; border-radius: 0; }"
                     "SearchLineEdit:hover { border-bottom: 1px solid #4b5563; }"
                     "SearchLineEdit:focus { border-bottom: 1px solid #2563eb; }"),
        )
        layout.addWidget(self._search_input, 1)

        return search_widget

    def _on_search_changed(self, query):
        """搜索关键词变化时过滤"""
        for category_id, page in self._pages.items():
            has_match = page.filter(query)
            # 更新分类树的可见性
            group_id = self._category_tree._categories.get(category_id)
            if group_id and group_id in self._category_tree._groups:
                group = self._category_tree._groups[group_id]
                if category_id in group._categories:
                    btn = group._categories[category_id]
                    btn.setVisible(has_match or not query)

    def _on_category_changed(self, category_id):
        """分类切换"""
        if category_id in self._pages:
            self._stacked.setCurrentWidget(self._pages[category_id])

    def _build_all_pages(self):
        """构建所有设置页面"""
        # 从 schema 分组
        schema = settings.get_schema()

        # --- 界面与通用 ---
        general_keys = ['theme', 'outputs_root']
        general_items = [s for s in schema if s.get('key') in general_keys]

        # --- 模型配置 子分组 ---
        interactive_keys = ['default_interactive_model', 'default_interactive_img_size']
        example_keys = ['example_model', 'example_img_size']
        refine_keys = ['refine_method', 'refine_use_onnx', 'refine_fast',
                       'refine_image_size', 'refine_trimap_thickness',
                       'refine_use_roi', 'refine_roi_margin']
        paths_keys = ['weights_dir', 'sam_dir', 'sam2_dir', 'yolo_dir', 'trt_dir', 'hqsam_dir', 'refine_dir',
                      'sam_b_path', 'sam_l_path', 'sam_vit_h_path', 'mobile_sam_path',
                      'sam_hq_b_path', 'sam_hq_l_path', 'sam_hq_tiny_path',
                      'sam2_b_path', 'sam2_l_path', 'sam2_hq_b_path', 'sam2_hq_l_path',
                      'yoloe_26x_path', 'yoloe_26n_path', 'yoloe_26s_path', 'yolo11n_path', 'yolov8n_path',
                      'sam3_path', 'sam3_bpe_path',
                      'vision_encoder', 'text_encoder', 'geometry_encoder', 'decoder', 'tokenizer_path',
                      'vitmatte_onnx_path',
                      'sam_onnx_b_encoder_path', 'sam_onnx_b_decoder_path',
                      'sam_onnx_l_encoder_path', 'sam_onnx_l_decoder_path',
                      'sam_onnx_h_encoder_path', 'sam_onnx_h_decoder_path',
                      'mobile_sam_onnx_encoder_path', 'mobile_sam_onnx_decoder_path',
                      'hqsam_onnx_b_encoder_path', 'hqsam_onnx_b_decoder_path',
                      'hqsam_onnx_l_encoder_path', 'hqsam_onnx_l_decoder_path',
                      'hqsam_onnx_h_encoder_path', 'hqsam_onnx_h_decoder_path',
                      'sam2_onnx_b_encoder_path', 'sam2_onnx_b_decoder_path',
                      'sam2_onnx_l_encoder_path', 'sam2_onnx_l_decoder_path',
                      'sam2_onnx_s_encoder_path', 'sam2_onnx_s_decoder_path',
                      'sam2_onnx_t_encoder_path', 'sam2_onnx_t_decoder_path']

        interactive_items = [s for s in schema if s.get('key') in interactive_keys]
        example_items = [s for s in schema if s.get('key') in example_keys]
        refine_items = [s for s in schema if s.get('key') in refine_keys]
        paths_items = [s for s in schema if s.get('key') in paths_keys]

        # --- 推理配置 子分组 ---
        postprocess_keys = ['high_precision', 'mask_smooth_enabled', 'mask_smooth_epsilon']
        slice_keys = ['slice_inference_enabled', 'slice_rows', 'slice_cols', 'slice_overlap_ratio',
                      'slice_postprocess_type', 'slice_match_metric', 'slice_match_threshold']

        postprocess_items = [s for s in schema if s.get('key') in postprocess_keys]
        slice_items = [s for s in schema if s.get('key') in slice_keys]

        # ====== 分组 1: 界面与通用 ======
        self._category_tree.add_group('general', '界面与通用')
        self._build_schema_page('general', 'general', '主题与通用', '主题与通用', general_items,
                                description='应用外观与通用行为')

        # ====== 分组 2: 模型配置 ======
        self._category_tree.add_group('models', '模型配置')
        self._build_schema_page('models_interactive', 'models', '交互模型', '交互模型配置', interactive_items,
                                description='交互式分割默认模型与输入尺寸')
        self._build_schema_page('models_example', 'models', '示例模型', '示例模型配置', example_items,
                                description='示例标注（few-shot）模型配置')
        self._build_schema_page('models_refine', 'models', '细化模型', '细化模型配置', refine_items,
                                description='标注细化的方法与参数')

        # ====== 模型路径：按模型家族拆分为多张卡片 ======
        path_page = self._create_page('models_paths', 'models', '模型路径',
                                      '各模型权重与 ONNX 文件路径，留空时使用自动解析。')
        path_buckets = [
            ('SAM 系列权重与 ONNX',
             ('sam_dir', 'sam_b_path', 'sam_l_path', 'sam_vit_h_path', 'mobile_sam_path', 'sam_onnx_')),
            ('SAM2 权重与 ONNX', ('sam2_',)),
            ('SAM3 权重与组件', ('sam3_', 'vision_encoder', 'text_encoder', 'geometry_encoder',
                              'decoder', 'tokenizer_path')),
            ('HQ-SAM 与细化', ('sam_hq_', 'hqsam_', 'vitmatte_onnx_path', 'refine_dir')),
            ('YOLO 系列权重', ('yolo_dir', 'yoloe_', 'yolo11n_path', 'yolov8n_path')),
        ]
        buckets = {title: [] for title, _ in path_buckets}
        buckets['其他路径'] = []
        for s in paths_items:
            key = s.get('key', '')
            for title, prefixes in path_buckets:
                if key.startswith(prefixes):
                    buckets[title].append(s)
                    break
            else:
                buckets['其他路径'].append(s)
        for title, bucket in buckets.items():
            if bucket:
                self._add_schema_section(path_page, title, bucket)

        # ====== 分组 3: 推理配置 ======
        self._category_tree.add_group('inference', '推理配置')
        self._build_schema_page('inference_postprocess', 'inference', '细化处理', '细化处理配置', postprocess_items,
                                description='掩码后处理与精度模式')
        self._build_schema_page('inference_slice', 'inference', '切片推理', '切片推理配置', slice_items,
                                description='大图切片推理设置')

        # ====== 分组 4: 插件管理 ======
        self._category_tree.add_group('plugins', '插件管理')
        self._build_plugins_page()

        # ====== 分组 6: 项目配置 ======
        self._category_tree.add_group('project', '项目配置')
        self._build_project_page()

        # ====== 分组 7: 快捷键配置 ======
        self._category_tree.add_group('shortcuts', '快捷键')
        self._build_shortcuts_page()

        # ====== 分组 8: AI 助手 ======
        self._category_tree.add_group('agent', 'AI 助手')
        # AI 提供商管理面板（按提供商分组的多模型配置）
        ai_providers_page = self._create_page(
            'ai_providers', 'agent', 'AI 提供商',
            '按提供商分组管理大语言模型 API 配置。')
        self._add_ai_providers_section(ai_providers_page)
        # 其他 AI 相关设置（意图分析复用对话 Agent 的提供商/模型，无需独立配置 API Key）
        ai_other_keys = ['intent_analysis_cooldown']
        ai_other_items = [s for s in schema if s.get('key') in ai_other_keys]
        if ai_other_items:
            self._build_schema_page('ai_agent', 'agent', 'AI 助手配置', '意图分析配置', ai_other_items,
                                    description='意图分析复用对话 Agent 的提供商与模型，以下为分析频率设置')

    def _add_ai_providers_section(self, page):
        """在页面上挂载 AI 提供商管理面板，并在变更时重新初始化 Agent。"""
        section = AIProviderSection()
        section.providers_changed.connect(self._reinit_agent)
        page.add_section(section)

    def _build_schema_page(self, page_id, group_id, title, section_title, items, description=''):
        """从 schema 列表构建设置页面（description 用于页头说明）"""
        page = self._create_page(page_id, group_id, title, description)
        self._add_schema_section(page, section_title, items)

    def _add_schema_section(self, page, section_title, items):
        """把一组 schema 项构建为一个设置分组并装入页面"""
        section = SettingSection(section_title)
        for item in items:
            widget = self._create_widget_from_schema(item)
            if widget:
                self._connect_auto_save_widget(widget, item)
                self._all_widgets[item['key']] = widget
                section.add_item(SettingItem(item['key'], item['title'],
                                            item.get('description', ''), widget))
        page.add_section(section)

    def _create_widget_from_schema(self, item):
        """根据 schema 创建控件"""
        setting_type = item.get('type', 'string')
        if setting_type in ('object', 'hidden'):
            return None
        default = item.get('value', item.get('default'))

        if setting_type == 'combo':
            widget = FluentComboBox()
            options = item.get('options', [])
            widget.addItems(options)
            if default:
                widget.setCurrentText(str(default))
            widget.setFixedHeight(ROW_HEIGHT)
        elif setting_type == 'int':
            widget = SpinBox()
            widget.setRange(item.get('min', 0), item.get('max', 9999))
            if default is not None:
                widget.setValue(int(default))
            widget.setFixedHeight(ROW_HEIGHT)
        elif setting_type == 'float':
            widget = SpinBox()
            min_val = item.get('min', 0)
            max_val = item.get('max', 100)
            widget.setRange(int(min_val * 1000), int(max_val * 1000))
            if default is not None:
                widget.setValue(int(float(default) * 1000))
            widget.setSuffix('')
            widget.setFixedHeight(ROW_HEIGHT)
        else:
            widget = LineEdit()
            if default:
                widget.setText(str(default))
            widget.setFixedHeight(ROW_HEIGHT)
            widget.setMinimumWidth(400)
            # 敏感字段（如 API Key）用密码模式显示，避免明文泄露
            if item.get('key') in ('ai_api_key', 'api_key', 'secret', 'token', 'password'):
                from PySide6.QtWidgets import QLineEdit
                widget.setEchoMode(QLineEdit.EchoMode.Password)
        return widget

    def _connect_auto_save_widget(self, widget, item):
        """连接控件的自动保存信号"""
        key = item['key']
        setting_type = item.get('type', 'string')

        if setting_type == 'combo':
            widget.currentTextChanged.connect(
                lambda t, k=key: self._save_setting(k, t))
        elif setting_type == 'int':
            widget.valueChanged.connect(
                lambda v, k=key: self._save_setting(k, v))
        elif setting_type == 'float':
            widget.valueChanged.connect(
                lambda v, k=key: self._save_setting(k, v / 1000.0))
        else:
            widget.editingFinished.connect(
                lambda w=widget, k=key: self._save_setting(k, w.text()))

    def _create_page(self, category_id, group_id, category_title, description=''):
        """创建页面并注册到分类树"""
        page = SettingsContentPage()
        page.add_page_header(category_title, description)
        self._pages[category_id] = page
        self._stacked.addWidget(page)
        self._category_tree.add_category(group_id, category_id, category_title)
        return page

    # --------------------------------------------------------
    # 插件管理
    # --------------------------------------------------------
    def _build_plugins_page(self):
        """插件管理 - 顶层为已安装插件开关页，各插件配置为同级子项（动态添加）"""
        page = self._create_page('plugins_toggle', 'plugins', '已安装插件',
                                 '启用或禁用已安装的插件')

        # 插件开关
        section1 = SettingSection('已安装插件')
        self._plugin_section = section1
        self._plugin_page = section1
        page.add_section(section1)

        # 插件配置将在 set_plugin_manager 时动态创建为同级子页面
        self._plugin_configs_widgets = {}  # {plugin_id: {key: widget}}

    # --------------------------------------------------------
    # 项目配置
    # --------------------------------------------------------
    def _build_project_page(self):
        page = self._create_page('project', 'project', '项目设置',
                                 '当前项目的任务类型、导出格式与类别')

        # 项目基本信息
        section1 = SettingSection('项目基本信息')

        self.project_combo = FluentComboBox()
        self.project_combo.setFixedHeight(ROW_HEIGHT)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        section1.add_item(SettingItem('project', '选择项目',
                                      '选择当前活动的项目。', self.project_combo))

        self.task_type_combo = FluentComboBox()
        # 内部统一短名 det/seg/obb，界面显示中文名
        # 注意 qfw addItem 签名: addItem(text, icon=None, userData=None)
        for short, disp in (("det", "目标检测"), ("seg", "目标分割"), ("obb", "旋转目标检测")):
            self.task_type_combo.addItem(disp, None, short)
        self.task_type_combo.setFixedHeight(ROW_HEIGHT)
        self.task_type_combo.currentIndexChanged.connect(lambda _: self._save_task_type())
        section1.add_item(SettingItem('task_type', '任务类型',
                                      '标注任务的类型（目标检测 / 目标分割 / 旋转目标检测）。', self.task_type_combo))

        self.export_format_combo = FluentComboBox()
        self.export_format_combo.addItems(['yoloseg', 'yolodet', 'yoloobb', 'coco', 'voc'])
        self.export_format_combo.setFixedHeight(ROW_HEIGHT)
        self.export_format_combo.currentTextChanged.connect(
            lambda t: self._save_project_setting('export_format', t))
        section1.add_item(SettingItem('export_format', '导出格式',
                                      '标注数据的导出格式。', self.export_format_combo))

        page.add_section(section1)

        # 缺陷类别
        section2 = SettingSection('缺陷类别配置')
        self._categories_section = section2
        self._category_checkboxes = {}
        page.add_section(section2)

        # ROI配置
        section3 = SettingSection('ROI配置')

        roi_grid = QWidget()
        roi_layout = QGridLayout(roi_grid)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(8)

        self.roi_x_edit = LineEdit()
        self.roi_x_edit.setPlaceholderText('X坐标')
        self.roi_x_edit.setFixedHeight(ROW_HEIGHT)
        roi_layout.addWidget(QLabel('X:'), 0, 0)
        roi_layout.addWidget(self.roi_x_edit, 0, 1)

        self.roi_y_edit = LineEdit()
        self.roi_y_edit.setPlaceholderText('Y坐标')
        self.roi_y_edit.setFixedHeight(ROW_HEIGHT)
        roi_layout.addWidget(QLabel('Y:'), 0, 2)
        roi_layout.addWidget(self.roi_y_edit, 0, 3)

        self.roi_w_edit = LineEdit()
        self.roi_w_edit.setPlaceholderText('宽度')
        self.roi_w_edit.setFixedHeight(ROW_HEIGHT)
        roi_layout.addWidget(QLabel('宽度:'), 1, 0)
        roi_layout.addWidget(self.roi_w_edit, 1, 1)

        self.roi_h_edit = LineEdit()
        self.roi_h_edit.setPlaceholderText('高度')
        self.roi_h_edit.setFixedHeight(ROW_HEIGHT)
        roi_layout.addWidget(QLabel('高度:'), 1, 2)
        roi_layout.addWidget(self.roi_h_edit, 1, 3)

        # 输入完成即保存 ROI（编辑失焦/回车触发），并同步标注画布
        for edit in (self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit):
            edit.editingFinished.connect(self._save_roi)

        section3.add_item(SettingItem('roi', 'ROI坐标',
                                      '感兴趣区域的坐标，修改后自动保存。', roi_grid))

        page.add_section(section3)

        # 提示词配置（项目级，用于基于文本的一键标注）
        section4 = SettingSection('提示词配置')
        self._prompts_section = section4
        self._prompt_library_cache = []
        self._prompt_checkboxes = {}

        add_row = QWidget()
        add_layout = QHBoxLayout(add_row)
        add_layout.setContentsMargins(ITEM_PADDING_H, 10, ITEM_PADDING_H, 10)
        add_layout.setSpacing(8)
        self._prompt_input = LineEdit()
        self._prompt_input.setPlaceholderText('输入提示词，回车或点击添加')
        self._prompt_input.setFixedHeight(ROW_HEIGHT)
        self._prompt_input.returnPressed.connect(self._on_prompt_add)
        add_layout.addWidget(self._prompt_input, 1)
        prompt_add_btn = PushButton(FIF.ADD, '添加')
        prompt_add_btn.setFixedHeight(ROW_HEIGHT)
        prompt_add_btn.clicked.connect(self._on_prompt_add)
        add_layout.addWidget(prompt_add_btn)
        section4.add_item(add_row)

        self._prompt_list_box = QWidget()
        self._prompt_list_layout = QVBoxLayout(self._prompt_list_box)
        self._prompt_list_layout.setContentsMargins(0, 0, 0, 0)
        self._prompt_list_layout.setSpacing(0)
        section4.add_item(self._prompt_list_box)
        page.add_section(section4)

        # 示例选择（项目级，用于基于示例的一键标注）
        section5 = SettingSection('示例选择')
        self._examples_section = section5
        self._examples_cache = []
        self._selected_examples_cache = set()
        self._example_checkboxes = {}

        self._example_list_box = QWidget()
        self._example_list_layout = QVBoxLayout(self._example_list_box)
        self._example_list_layout.setContentsMargins(0, 0, 0, 0)
        self._example_list_layout.setSpacing(0)
        section5.add_item(self._example_list_box)
        page.add_section(section5)

    # --------------------------------------------------------
    # 快捷键配置
    # --------------------------------------------------------
    def _build_shortcuts_page(self):
        page = self._create_page('shortcuts', 'shortcuts', '快捷键设置',
                                 '自定义键盘快捷键')

        from core.common.shortcut_manager import ShortcutManager
        sm = ShortcutManager.instance()

        # 按分类创建 section
        self._shortcut_items = {}
        for category in sm.get_categories():
            section = SettingSection(category)
            infos = sm.get_by_category(category)
            for info in infos:
                item = ShortcutItem(
                    info.action_id, info.description,
                    info.default_key, info.key
                )
                item.shortcut_changed.connect(self._on_shortcut_changed)
                section.add_item(item)
                self._shortcut_items[info.action_id] = item
            page.add_section(section)

        # 全部重置按钮
        reset_section = SettingSection("操作")
        reset_widget = QWidget()
        reset_layout = QHBoxLayout(reset_widget)
        reset_layout.setContentsMargins(ITEM_PADDING_H, 12, ITEM_PADDING_H, 12)
        reset_all_btn = PrimaryPushButton("重置所有快捷键为默认")
        reset_all_btn.setFixedHeight(36)
        reset_all_btn.clicked.connect(self._on_reset_all_shortcuts)
        reset_layout.addWidget(reset_all_btn)
        reset_layout.addStretch(1)
        reset_section.add_item(reset_widget)
        page.add_section(reset_section)

    def _on_shortcut_changed(self, action_id: str, new_key: str):
        """快捷键变更回调"""
        from core.common.shortcut_manager import ShortcutManager
        sm = ShortcutManager.instance()

        # 检查冲突
        conflict = sm.is_conflict(new_key, exclude_action=action_id)
        if conflict:
            conflict_info = sm.get_all().get(conflict)
            conflict_desc = conflict_info.description if conflict_info else conflict
            InfoBar.warning(
                "快捷键冲突",
                f"快捷键 {new_key} 已被「{conflict_desc}」使用，请选择其他快捷键。",
                duration=3000, parent=self
            )
            # 恢复原值
            current_info = sm.get_all().get(action_id)
            if action_id in self._shortcut_items:
                self._shortcut_items[action_id].set_key(current_info.key)
            return

        sm.set_key(action_id, new_key)
        InfoBar.success("快捷键已更新", f"快捷键已保存，重新进入标注界面后生效。", duration=2000, parent=self)

    def _on_reset_all_shortcuts(self):
        """重置所有快捷键"""
        from core.common.shortcut_manager import ShortcutManager
        sm = ShortcutManager.instance()
        sm.reset_all()
        # 刷新UI
        for action_id, item in self._shortcut_items.items():
            info = sm.get_all().get(action_id)
            if info:
                item.set_key(info.key)
        InfoBar.success("已重置", "所有快捷键已恢复为默认值。", duration=2000, parent=self)

    # --------------------------------------------------------
    # 数据加载/保存
    # --------------------------------------------------------
    def _populate_plugin_settings(self):
        """填充插件管理设置"""
        if not self.plugin_manager:
            return

        # 清除现有内容
        while self._plugin_section._content_layout.count():
            item = self._plugin_section._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加提示标签
        hint_label = CaptionLabel('⚠ 修改后下次启动生效')
        hint_label.setObjectName('SettingDescription')
        self._plugin_section._content_layout.addWidget(hint_label)

        # 添加每个插件的开关
        for plugin_info in self.plugin_manager.get_plugin_list():
            switch_widget = QWidget()
            switch_layout = QHBoxLayout(switch_widget)
            switch_layout.setContentsMargins(ITEM_PADDING_H, 8, ITEM_PADDING_H, 8)

            name_label = BodyLabel(plugin_info['name'])
            name_label.setFixedWidth(120)
            switch_layout.addWidget(name_label)

            switch = SwitchButton()
            switch.setChecked(plugin_info['enabled'])
            switch.setOffText('禁用')
            switch.setOnText('启用')
            switch.setFixedHeight(28)
            switch.checkedChanged.connect(
                lambda checked, pid=plugin_info['id']: self._on_plugin_toggled(pid, checked))
            switch_layout.addWidget(switch)

            status = CaptionLabel('✓ 已加载' if plugin_info['loaded'] else '未加载')
            switch_layout.addWidget(status)
            switch_layout.addStretch(1)

            self._plugin_section._content_layout.addWidget(switch_widget)

        # 动态加载各插件的配置
        self._load_all_plugin_configs()

    def _populate_plc_settings(self):
        """填充 PLC 设置"""
        # PLC 设置已经在 _build_plugins_page 中创建

    def _populate_project_settings(self):
        """填充项目设置"""
        if not self.project_service:
            return

        # 刷新项目列表
        self.project_combo.clear()
        project_dicts = self.project_service.get_all_projects()
        for p in project_dicts:
            name = p.get('name', '')
            if name:
                self.project_combo.addItem(name)

        # 加载第一个项目的信息
        if project_dicts:
            first_project = project_dicts[0]
            name = first_project.get('name', '')
            if name:
                self._load_project_info(name)

    def _on_project_changed(self, project_name):
        """项目切换"""
        if not project_name:
            return
        projects_root = settings.get('outputs_root') or Config.PROJECTS_DIR
        project_path = os.path.join(projects_root, project_name)
        project_settings.load_project(project_name)
        self._load_project_info(project_name)

    def _load_project_info(self, project_name):
        """加载项目信息"""
        projects_root = settings.get('outputs_root') or Config.PROJECTS_DIR
        project_path = os.path.join(projects_root, project_name)
        info_path = os.path.join(project_path, 'project_info.json')

        project_info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8-sig') as f:
                    project_info = json.load(f)
            except Exception:
                project_info = {}

        # 任务类型：按内部短名 det/seg/obb 回填，界面显示中文（旧项目长名兼容归一）
        from core.service.project_service import normalize_task_type
        task_idx = self.task_type_combo.findData(
            normalize_task_type(project_info.get('task_type', 'seg')))
        self.task_type_combo.blockSignals(True)
        self.task_type_combo.setCurrentIndex(task_idx if task_idx >= 0 else 0)
        self.task_type_combo.blockSignals(False)

        export_fmt = project_settings.get('export_format') or project_info.get('export_format', 'yoloseg')
        self.export_format_combo.setCurrentText(export_fmt)

        # 加载缺陷类别
        self._load_categories_checkboxes(project_info)

        # 加载 ROI
        roi = project_info.get('roi')
        if roi and len(roi) >= 4:
            self.roi_x_edit.setText(str(roi[0]))
            self.roi_y_edit.setText(str(roi[1]))
            self.roi_w_edit.setText(str(roi[2]))
            self.roi_h_edit.setText(str(roi[3]))
        else:
            for edit in [self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit]:
                edit.setText('')

        # 提示词 / 示例选择
        self._reload_prompt_section(project_info)
        self._reload_examples_section(project_name, project_info)

    def _load_categories_checkboxes(self, project_info):
        """加载缺陷类别复选框"""
        # 清除现有内容
        while self._categories_section._content_layout.count():
            item = self._categories_section._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._category_checkboxes = {}
        categories = project_info.get('categories', {})
        defect_categories = project_info.get('defect_categories', [])

        if categories:
            cat_names = list(categories.values()) if isinstance(categories, dict) else categories
            for cat_name in cat_names:
                checkbox = CheckBox(cat_name)
                if cat_name in defect_categories:
                    checkbox.setChecked(True)
                checkbox.stateChanged.connect(
                    lambda s, n=cat_name: self._on_defect_category_changed(n, s == Qt.CheckState.Checked.value))
                self._category_checkboxes[cat_name] = checkbox

                # 包装在带边距的容器中
                wrapper = QWidget()
                wrapper_layout = QHBoxLayout(wrapper)
                wrapper_layout.setContentsMargins(ITEM_PADDING_H, 8, ITEM_PADDING_H, 8)
                wrapper_layout.addWidget(checkbox)
                wrapper_layout.addStretch(1)
                self._categories_section._content_layout.addWidget(wrapper)

    # --------------------------------------------------------
    # 提示词 / 示例选择（项目级配置）
    # --------------------------------------------------------

    def _annotation_auto(self):
        """获取标注插件的自动标注服务（用于同步运行时数据），不可用时返回 None。"""
        try:
            pm = getattr(self, 'plugin_manager', None)
            if pm is None:
                return None
            mw = getattr(getattr(pm, 'context', None), '_main_window', None)
            if mw is None:
                return None
            interface = mw.find_interface('AnnotationInterface')
            if interface is None:
                return None
            return getattr(interface, 'auto', None)
        except Exception:
            return None

    def _is_active_project(self) -> bool:
        """配置页当前选中的项目是否为应用当前活动项目（决定是否同步标注服务运行时）。"""
        return bool(project_settings.name) and \
            project_settings.name == settings.get('current_project_name', '')

    # ---- 提示词 ----

    def _reload_prompt_section(self, project_info=None):
        """按项目重建提示词勾选列表。"""
        box = getattr(self, '_prompt_list_box', None)
        if box is None:
            return
        while self._prompt_list_layout.count():
            item = self._prompt_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._prompt_checkboxes = {}
        if project_info is not None:
            prompts = list(project_info.get('prompt_library') or [])
        else:
            prompts = list(project_settings.get('prompt_library') or [])
        self._prompt_library_cache = prompts
        for p in prompts:
            text = p.get('text', '')
            if not text:
                continue
            checkbox = CheckBox(text)
            checkbox.setChecked(bool(p.get('checked', True)))
            checkbox.stateChanged.connect(
                lambda s, t=text: self._on_prompt_check_changed(t, s == Qt.CheckState.Checked.value))
            self._prompt_checkboxes[text] = checkbox
            self._prompt_list_layout.addWidget(
                self._make_checkbox_row(checkbox, lambda t=text: self._on_prompt_remove(t)))
        if not prompts:
            empty = CaptionLabel('暂无提示词，可在上方输入添加。')
            empty.setObjectName('SettingDescription')
            wrapper = QWidget()
            wl = QHBoxLayout(wrapper)
            wl.setContentsMargins(ITEM_PADDING_H, 10, ITEM_PADDING_H, 10)
            wl.addWidget(empty)
            self._prompt_list_layout.addWidget(wrapper)

    def _make_checkbox_row(self, checkbox, remove_cb):
        """构建 复选框 + 删除按钮 的行容器。"""
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(ITEM_PADDING_H, 6, ITEM_PADDING_H, 6)
        layout.setSpacing(8)
        layout.addWidget(checkbox)
        layout.addStretch(1)
        del_btn = TransparentToolButton(FIF.DELETE)
        del_btn.setToolTip('删除')
        del_btn.clicked.connect(remove_cb)
        layout.addWidget(del_btn)
        return wrapper

    def _persist_prompts(self, prompts):
        """保存提示词库到项目设置，并同步标注服务运行时数据。"""
        self._prompt_library_cache = list(prompts)
        project_settings.set('prompt_library', list(prompts))
        if self._is_active_project():
            auto = self._annotation_auto()
            if auto is not None:
                try:
                    auto.context.prompt_library = list(prompts)
                    auto.save_prompt_library()
                except Exception as e:
                    print(f'[ConfigInterface] 同步提示词到标注服务失败: {e}')
        self._reload_prompt_section()

    def _on_prompt_add(self):
        text = self._prompt_input.text().strip()
        if not text:
            return
        from core.service.common_service import PromptManager
        self._prompt_input.clear()
        self._persist_prompts(PromptManager.add_prompt(self._prompt_library_cache, text))
        InfoBar.success('已添加', f'提示词「{text}」已添加', duration=1500, parent=self)

    def _on_prompt_remove(self, text):
        prompts = [p for p in self._prompt_library_cache if p.get('text') != text]
        self._persist_prompts(prompts)

    def _on_prompt_check_changed(self, text, checked):
        prompts = self._prompt_library_cache
        for p in prompts:
            if p.get('text') == text:
                p['checked'] = bool(checked)
                break
        self._persist_prompts(prompts)

    # ---- 示例选择 ----

    def _load_examples_for_project(self, project_name):
        """扫描指定项目的 support_sets 目录，返回示例列表 [{name, category}]。"""
        examples = []
        if not project_name:
            return examples
        projects_root = settings.get('outputs_root') or Config.PROJECTS_DIR
        support_dir = os.path.join(projects_root, project_name, 'support_sets')
        if os.path.isdir(support_dir):
            for folder in sorted(os.listdir(support_dir)):
                info_path = os.path.join(support_dir, folder, 'info.json')
                if os.path.isfile(info_path):
                    try:
                        with open(info_path, 'r', encoding='utf-8-sig') as f:
                            info = json.load(f)
                        examples.append({
                            'name': info.get('name', folder),
                            'category': info.get('category', ''),
                        })
                    except Exception:
                        continue
        return examples

    def _reload_examples_section(self, project_name="", project_info=None):
        """按项目重建示例选择列表。"""
        box = getattr(self, '_example_list_box', None)
        if box is None:
            return
        while self._example_list_layout.count():
            item = self._example_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._example_checkboxes = {}
        if not project_name:
            project_name = project_settings.name or settings.get('current_project_name', '')
        examples = self._load_examples_for_project(project_name)
        selected = set((project_info or {}).get('selected_examples') or []) or \
            set(project_settings.get('selected_examples') or [])
        self._examples_cache = examples
        self._selected_examples_cache = selected
        if not examples:
            empty = CaptionLabel('当前项目没有示例，请在标注界面添加示例。')
            empty.setObjectName('SettingDescription')
            wrapper = QWidget()
            wl = QHBoxLayout(wrapper)
            wl.setContentsMargins(ITEM_PADDING_H, 10, ITEM_PADDING_H, 10)
            wl.addWidget(empty)
            self._example_list_layout.addWidget(wrapper)
            return
        for ex in examples:
            name = ex.get('name', '')
            if not name:
                continue
            label = f"{name} [{ex.get('category', '')}]" if ex.get('category') else name
            checkbox = CheckBox(label)
            checkbox.setChecked(name in selected)
            checkbox.stateChanged.connect(
                lambda s, n=name: self._on_example_check_changed(n, s == Qt.CheckState.Checked.value))
            self._example_checkboxes[name] = checkbox
            # 与缺陷类别卡片一致的样式：每个复选框独占一行，带上下边距
            wrapper = QWidget()
            wrapper_layout = QHBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(ITEM_PADDING_H, 8, ITEM_PADDING_H, 8)
            wrapper_layout.addWidget(checkbox)
            wrapper_layout.addStretch(1)
            self._example_list_layout.addWidget(wrapper)

    def _persist_examples(self, names):
        """保存示例选中状态到项目设置，并同步标注服务运行时数据。"""
        self._selected_examples_cache = set(names)
        project_settings.set('selected_examples', sorted(names))
        if self._is_active_project():
            auto = self._annotation_auto()
            if auto is not None:
                try:
                    auto.set_examples(sorted(names))
                except Exception as e:
                    print(f'[ConfigInterface] 同步示例选择到标注服务失败: {e}')

    def _on_example_check_changed(self, name, checked):
        current = set(self._selected_examples_cache)
        if checked:
            current.add(name)
        else:
            current.discard(name)
        self._persist_examples(sorted(current))

    def _on_plugin_toggled(self, plugin_id, enabled):
        """插件启用/禁用"""
        if enabled:
            self.plugin_manager.enable_plugin(plugin_id)
        else:
            self.plugin_manager.disable_plugin(plugin_id)

    def _load_all_plugin_configs(self):
        """动态加载所有插件的配置 - 每个插件创建独立的同级子页面"""
        if not self.plugin_manager:
            print("[ConfigInterface] No plugin_manager")
            return

        for plugin_info in self.plugin_manager.get_plugin_list():
            plugin_id = plugin_info['id']
            plugin_name = plugin_info['name']

            if not plugin_info.get('loaded', False):
                print(f"[ConfigInterface] Plugin {plugin_id} not loaded, skipping")
                continue

            # 尝试加载插件的配置定义
            config_schema = self._get_plugin_config_schema(plugin_id)
            if not config_schema:
                print(f"[ConfigInterface] No config schema for {plugin_id}")
                continue

            print(f"[ConfigInterface] Loading config for {plugin_id}: {len(config_schema)} settings")

            # 为每个插件创建独立的同级子页面
            page_id = f'plugins_{plugin_id}'
            page = self._create_page(page_id, 'plugins', f'{plugin_name} 配置')
            section = SettingSection(f'{plugin_name} 配置')
            self._plugin_configs_widgets[plugin_id] = {}

            for setting_def in config_schema:
                key = setting_def['key']
                title = setting_def['title']
                description = setting_def.get('description', setting_def.get('title', ''))
                setting_type = setting_def.get('type', 'string')
                default = setting_def.get('default', None)
                options = setting_def.get('options', None)
                min_val = setting_def.get('min', None)
                max_val = setting_def.get('max', None)

                # 创建控件
                widget = self._create_setting_widget(
                    setting_type, default, options, min_val, max_val)

                # 加载当前值
                current_value = self._load_plugin_setting(plugin_id, key, default)
                self._set_widget_value(widget, setting_type, current_value)

                # 连接自动保存
                self._connect_widget_auto_save(widget, setting_type,
                    lambda v, pid=plugin_id, k=key: self._save_plugin_setting(pid, k, v))

                # 添加到分组
                section.add_item(SettingItem(key, title, description, widget))
                self._plugin_configs_widgets[plugin_id][key] = widget

            page.add_section(section)
            print(f"[ConfigInterface] Added page for {plugin_id}")

    def _get_plugin_config_schema(self, plugin_id):
        """获取插件的配置定义 - 优先从 PluginConfig 读取"""
        from core.common.plugin_config import PluginConfig
        from core.common.config import Config
        import os

        plugin_dir = os.path.join(Config.ROOT_DIR, 'plugins', plugin_id)
        print(f"[ConfigInterface] Looking for schema in: {plugin_dir}")

        # 方式1：通过 PluginConfig 读取（支持 config_schema.json 和 config.json）
        try:
            config = PluginConfig(plugin_dir)
            schema = config.get_schema()
            if schema:
                print(f"[ConfigInterface] Loaded schema from PluginConfig: {len(schema)} items")
                return schema
        except Exception as e:
            print(f"[ConfigInterface] Error loading schema from PluginConfig: {e}")

        # 方式2：直接从 config_schema.json 读取
        schema_path = os.path.join(plugin_dir, 'config_schema.json')
        print(f"[ConfigInterface] Checking schema path: {schema_path}, exists: {os.path.exists(schema_path)}")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                    print(f"[ConfigInterface] Loaded schema: {len(schema)} items")
                    return schema
            except Exception as e:
                print(f"[ConfigInterface] Error loading schema: {e}")

        # 方式3：从插件的 config.py 中的 CONFIG_SCHEMA 读取
        config_py_path = os.path.join(plugin_dir, 'config.py')
        if os.path.exists(config_py_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"plugins.{plugin_id}.config", config_py_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'CONFIG_SCHEMA'):
                    return module.CONFIG_SCHEMA
            except:
                pass

        # 方式4：从已加载的插件实例中读取
        if self.plugin_manager and plugin_id in self.plugin_manager.plugins:
            plugin = self.plugin_manager.plugins[plugin_id]
            if hasattr(plugin, 'config_schema'):
                return plugin.config_schema

        return None

    def _create_setting_widget(self, setting_type, default, options, min_val, max_val):
        """根据类型创建设置控件"""
        if setting_type == 'bool':
            # bool 类型转换为 combo（打开/关闭）
            widget = FluentComboBox()
            widget.addItems(['打开', '关闭'])
            if default:
                widget.setCurrentText('打开')
            else:
                widget.setCurrentText('关闭')
            widget.setFixedHeight(ROW_HEIGHT)
        elif setting_type == 'combo':
            widget = FluentComboBox()
            if options:
                widget.addItems(options)
            if default:
                widget.setCurrentText(str(default))
            widget.setFixedHeight(ROW_HEIGHT)
        elif setting_type == 'int':
            widget = SpinBox()
            if min_val is not None:
                widget.setRange(min_val, max_val or 9999)
            if default is not None:
                widget.setValue(int(default))
            widget.setFixedHeight(ROW_HEIGHT)
        elif setting_type == 'float':
            widget = SpinBox()
            if min_val is not None:
                widget.setRange(int(min_val * 100), int((max_val or 100) * 100))
            if default is not None:
                widget.setValue(int(float(default) * 100))
            widget.setFixedHeight(ROW_HEIGHT)
        else:  # string
            widget = LineEdit()
            if default:
                widget.setText(str(default))
            widget.setFixedHeight(ROW_HEIGHT)
        return widget

    def _set_widget_value(self, widget, setting_type, value):
        """设置控件的值"""
        if setting_type == 'bool':
            # bool 类型转换为 combo（打开/关闭）
            if value:
                widget.setCurrentText('打开')
            else:
                widget.setCurrentText('关闭')
        elif setting_type == 'combo':
            widget.setCurrentText(str(value))
        elif setting_type in ('int', 'float'):
            widget.setValue(int(value))
        else:
            widget.setText(str(value))

    def _get_widget_value(self, widget, setting_type):
        """获取控件的值"""
        if setting_type == 'bool':
            # bool 类型转换为 combo（打开/关闭）
            return widget.currentText() == '打开'
        elif setting_type == 'combo':
            return widget.currentText()
        elif setting_type in ('int', 'float'):
            return widget.value()
        else:
            return widget.text()

    def _load_plugin_setting(self, plugin_id, key, default):
        """从 config.json 加载插件配置值，annotation 插件走全局 settings"""
        if plugin_id == 'annotation':
            return settings.get(key, default)
        try:
            from core.common.config import Config
            import os
            config_path = os.path.join(Config.ROOT_DIR, 'plugins', plugin_id, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for item in config:
                    if item.get('key') == key:
                        return item.get('value', item.get('default', default))
            return default
        except:
            return default

    def _save_plugin_setting(self, plugin_id, key, value):
        """保存插件配置值到 config.json，annotation 插件走全局 settings"""
        if plugin_id == 'annotation':
            settings.set(key, value)
            return
        try:
            from core.common.config import Config
            import os
            config_path = os.path.join(Config.ROOT_DIR, 'plugins', plugin_id, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for item in config:
                    if item.get('key') == key:
                        item['value'] = value
                        break
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ConfigInterface] Failed to save plugin setting: {e}")

    def _connect_widget_auto_save(self, widget, setting_type, save_callback):
        """连接控件的自动保存信号"""
        if setting_type == 'bool':
            # bool 类型转换为 combo（打开/关闭）
            widget.currentTextChanged.connect(lambda t: save_callback(t == '打开'))
        elif setting_type == 'combo':
            widget.currentTextChanged.connect(lambda t: save_callback(t))
        elif setting_type in ('int', 'float'):
            widget.valueChanged.connect(lambda v: save_callback(v))
        else:
            widget.editingFinished.connect(lambda: save_callback(widget.text()))

    def _save_setting(self, key, value):
        """保存设置（自动保存）"""
        # 主题切换特殊处理
        if key == 'theme':
            theme_value = 'dark' if value == '暗色主题' else 'light'
            settings.set(key, theme_value)
            self._on_theme_changed(value)
        else:
            settings.set(key, value)
            # AI 配置类设置修改后立即生效（无需重启）
            if key in ('ai_api_key', 'ai_base_url', 'ai_model', 'ai_providers'):
                self._reinit_agent()

    def _reinit_agent(self):
        """AI 配置变更后重新初始化 Agent，使 API Key / Base URL / 模型立即生效。"""
        try:
            from core.common.ai_providers import get_active_credentials
            api_key, base_url, model = get_active_credentials()
            from PySide6.QtWidgets import QApplication
            for widget in QApplication.topLevelWidgets():
                agent = getattr(widget, '_ai_agent', None)
                if agent is None:
                    continue
                if agent.is_busy():
                    print("[ConfigInterface] Agent 正在执行任务，暂不重新初始化，将在下次生效")
                    return
                agent.initialize(api_key, base_url, model, main_window=widget)
                print(f"[ConfigInterface] Agent 已按新配置重新初始化: {model}")
                return
        except Exception as e:
            print(f"[ConfigInterface] 重新初始化 Agent 失败: {e}")

    def _save_project_setting(self, key, value):
        """保存项目设置"""
        project_settings.set(key, value)

    def _save_task_type(self):
        """任务类型变更：保存内部短名 det/seg/obb，并联动默认导出格式。"""
        short = self.task_type_combo.currentData()
        if not short:
            return
        self._save_project_setting('task_type', short)
        fmt = TASK_TYPE_DEFAULT_EXPORT.get(short, 'yoloseg')
        self.export_format_combo.blockSignals(True)
        self.export_format_combo.setCurrentText(fmt)
        self.export_format_combo.blockSignals(False)
        self._save_project_setting('export_format', fmt)
        InfoBar.success('已更新', f'任务类型已切换为 {TASK_TYPE_DISPLAY.get(short, short)}', duration=1500, parent=self)

    def _on_defect_category_changed(self, cat_name, checked):
        """缺陷类别勾选变更：更新项目设置中的 defect_categories 列表。"""
        current = list(project_settings.get('defect_categories') or [])
        if checked and cat_name not in current:
            current.append(cat_name)
        elif not checked and cat_name in current:
            current.remove(cat_name)
        self._save_project_setting('defect_categories', current)

    def _save_roi(self):
        """保存 ROI 坐标 [x, y, w, h] 到项目设置；全部留空=清除 ROI，并同步标注画布。"""
        vals = []
        for edit in (self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit):
            text = edit.text().strip()
            try:
                vals.append(int(float(text)))
            except (TypeError, ValueError):
                vals.append(None)
        if all(v is None for v in vals):
            roi = None
        elif any(v is None for v in vals):
            InfoBar.warning('ROI 未保存', 'X/Y/宽度/高度需完整填写数字', duration=2000, parent=self)
            self._load_roi_to_edits()  # 回填当前值
            return
        else:
            roi = vals
        self._save_project_setting('roi', roi)
        InfoBar.success('ROI 已保存', f'ROI: {roi}', duration=1500, parent=self)
        self._refresh_annotation_roi()

    def _load_roi_to_edits(self):
        """把 project_settings 中的 ROI 回填到编辑框（用于校验失败后还原）。"""
        roi = project_settings.get('roi')
        edits = [self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit]
        if roi and len(roi) >= 4:
            for edit, v in zip(edits, roi[:4]):
                edit.setText(str(v))
        else:
            for edit in edits:
                edit.setText('')

    def _refresh_annotation_roi(self):
        """同步标注界面画布：ROI 变化后让 draw_area 重绘。"""
        if not self._is_active_project():
            return
        try:
            pm = getattr(self, 'plugin_manager', None)
            if pm is None:
                return
            mw = getattr(getattr(pm, 'context', None), '_main_window', None)
            if mw is None:
                return
            interface = mw.find_interface('AnnotationInterface')
            if interface is None:
                return
            draw_area = getattr(interface, 'draw_area', None)
            if draw_area is not None:
                draw_area.update()
        except Exception:
            pass

    def _on_theme_changed(self, text):
        """主题切换"""
        new_theme = 'dark' if text == '暗色主题' else 'light'
        if new_theme != theme_manager.get_current_theme():
            theme_manager.set_theme(new_theme)

    def _on_smoothness_changed(self, slider_value):
        """平滑精细度变化"""
        epsilon = 0.0001 * (0.01 / 0.0001) ** (slider_value / 100.0)
        self.smoothness_value_label.setText(f'{epsilon:.4f}')
        self._save_setting('mask_smooth_epsilon', round(epsilon, 6))

    def _update_img_size_from_model(self, model_type, is_interactive):
        """根据模型类型更新默认图片尺寸"""
        try:
            from core.backend.core import ModelFactory
            default_size = ModelFactory.get_default_img_size(model_type)
            if is_interactive:
                self.interactive_img_size_spin.setValue(default_size)
            else:
                self.example_img_size_spin.setValue(default_size)
        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"[ConfigInterface] Failed to get default img_size: {e}")
