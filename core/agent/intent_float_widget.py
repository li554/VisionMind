"""
意图分析状态胶囊 — 嵌入标题栏菜单栏中央

- 作为普通 QWidget 嵌入主窗口标题栏中央，无浮动窗口
- 胶囊：状态点 + 当前意图文本，随 IntentAgent.analysis_updated 实时刷新
- 点击胶囊 → 展开/收起详情卡片（意图徽章 / 意图描述 / 建议列表 + 执行按钮）
- 卡片头部提供意图分析开关
"""

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QWidget,
)
from qfluentwidgets import SwitchButton

from core.common.theme_manager import theme_manager


class _Pill(QFrame):
    """可点击的胶囊（点击 = 固定/取消固定）"""

    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class IntentSuggestionItem(QFrame):
    """详情卡片内的单条执行建议"""

    suggestion_clicked = Signal(str, dict)

    def __init__(self, title: str, message: str, action_name: str = "",
                 params: dict = None, parent=None):
        super().__init__(parent)
        self.setObjectName("IntentSugItem")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("IntentSugTitle")
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._msg_label = QLabel(message)
        self._msg_label.setObjectName("IntentSugMsg")
        self._msg_label.setWordWrap(True)
        layout.addWidget(self._msg_label)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._action_label = QLabel(action_name or "")
        self._action_label.setObjectName("IntentSugTag")
        if not action_name:
            self._action_label.hide()
        row.addWidget(self._action_label)
        row.addStretch()
        self._exec_btn = QToolButton()
        self._exec_btn.setText("执行")
        self._exec_btn.setObjectName("IntentSugBtn")
        self._exec_btn.setFixedHeight(24)
        self._exec_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._exec_btn.setToolTip("执行此建议")
        self._exec_btn.clicked.connect(
            lambda checked=False, a=action_name, p=params or {}:
            self.suggestion_clicked.emit(a, p)
        )
        row.addWidget(self._exec_btn)
        layout.addLayout(row)


class IntentFloatWidget(QFrame):
    """意图分析状态胶囊（嵌入菜单栏中间 + 下拉详情卡片）"""

    suggestion_clicked = Signal(str, dict)  # 点击执行建议
    enabled_toggled = Signal(bool)          # 开关切换
    geometry_changed = Signal()             # 尺寸变化（供外部重新定位卡片）

    CARD_WIDTH = 340

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("IntentFloatWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._expanded = True  # 兼容旧 API
        self._pinned = False
        self._pill_state = "idle"
        self._pulse_on = False
        self._suggestion_items = []

        self._build_ui()
        self._reparent_card()
        self._apply_theme()
        theme_manager.theme_changed.connect(self._apply_theme)

    def _reparent_card(self):
        """将详情卡片挂载到主窗口作为普通子控件（非独立窗口）

        - 不设置独立窗口标志 / 窗口级透明：避免主窗口与卡片两个分层窗口在
          Windows 上叠加触发渲染异常（点击标题栏导致整个窗口消失且无法恢复）
        - 卡片随主窗口自动移动，仅需在胶囊水平位置变化时重新定位
        """
        main_win = self.window()
        if main_win and main_win is not self:
            self._card.setParent(main_win)
            self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self._card.hide()
            # 卡片脱离胶囊子树后不再继承 QSS，需复制完整样式到卡片自身
            self._card.setStyleSheet(self.styleSheet())
            self.geometry_changed.connect(
                lambda: self._reposition_card() if self._card.isVisible() else None
            )

    # ----------------------- UI -----------------------

    def _build_ui(self):
        # 嵌入标题栏：布局紧凑贴合，无多余空白
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------- 状态胶囊 ----------
        self._pill = _Pill()
        self._pill.setObjectName("IntentPill")
        self._pill.setFixedHeight(30)
        self._pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pill.setToolTip("点击展开详情 · 再点收起")
        self._pill.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._pill.clicked.connect(self.toggle_expand)

        p = QHBoxLayout(self._pill)
        p.setContentsMargins(12, 0, 12, 0)
        p.setSpacing(8)

        self._pin_icon = QLabel("📌")
        self._pin_icon.setObjectName("IntentPillPin")
        self._pin_icon.setFixedWidth(14)
        self._pin_icon.hide()
        p.addWidget(self._pin_icon)

        self._dot = QLabel("●")
        self._dot.setObjectName("IntentPillDot")
        self._dot.setFixedSize(10, 10)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p.addWidget(self._dot)

        self._pill_text = QLabel("等待操作")
        self._pill_text.setObjectName("IntentPillText")
        p.addWidget(self._pill_text)
        root.addWidget(self._pill, 1)

        # ---------- 悬浮详情卡片 ----------
        self._card = QFrame()
        self._card.setObjectName("IntentCard")
        self._card.setFixedWidth(self.CARD_WIDTH)
        self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 嵌入模式下卡片脱离子树挂到主窗口顶层；不添加 QGraphicsDropShadowEffect，
        # 避免在无边框透明主窗口上触发 Windows 渲染异常导致整个窗口消失
        self._card.hide()

        cv = QVBoxLayout(self._card)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        # 卡片头部
        header = QFrame()
        header.setObjectName("IntentCardHeader")
        header.setFixedHeight(50)
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 0, 10, 0)
        h.setSpacing(8)

        self._card_icon = QLabel("🤖")
        self._card_icon.setObjectName("IntentCardIcon")
        self._card_icon.setFixedSize(28, 28)
        self._card_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(self._card_icon)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        self._card_title = QLabel("意图分析")
        self._card_title.setObjectName("IntentCardTitle")
        title_box.addWidget(self._card_title)
        self._card_sub = QLabel("基于最近操作自动判断")
        self._card_sub.setObjectName("IntentCardSub")
        title_box.addWidget(self._card_sub)
        h.addLayout(title_box)
        h.addStretch()

        self._switch = SwitchButton()
        self._switch.setFixedSize(44, 22)
        self._switch.setOnText("")
        self._switch.setOffText("")
        self._switch.setChecked(True)
        self._switch.setToolTip("开启/关闭意图分析")
        self._switch.checkedChanged.connect(self._on_switch_changed)
        h.addWidget(self._switch)

        self._pin_btn = QToolButton()
        self._pin_btn.setText("📌")
        self._pin_btn.setObjectName("IntentCardBtn")
        self._pin_btn.setFixedSize(26, 26)
        self._pin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pin_btn.setToolTip("固定/取消固定")
        self._pin_btn.clicked.connect(self.toggle_pin)
        h.addWidget(self._pin_btn)
        cv.addWidget(header)

        # 卡片主体
        body = QWidget()
        b = QVBoxLayout(body)
        b.setContentsMargins(12, 10, 12, 10)
        b.setSpacing(8)

        self._intent_badge = QFrame()
        self._intent_badge.setObjectName("IntentBadge")
        self._intent_badge.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        badge_l = QHBoxLayout(self._intent_badge)
        badge_l.setContentsMargins(10, 3, 10, 3)
        badge_l.setSpacing(6)
        self._badge_dot = QLabel("●")
        self._badge_dot.setObjectName("IntentBadgeDot")
        self._badge_dot.setFixedSize(8, 8)
        self._badge_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_l.addWidget(self._badge_dot)
        self._badge_text = QLabel("等待操作")
        self._badge_text.setObjectName("IntentBadgeText")
        badge_l.addWidget(self._badge_text)
        b.addWidget(self._intent_badge, 0, Qt.AlignmentFlag.AlignLeft)

        self._desc_label = QLabel("")
        self._desc_label.setObjectName("IntentCardDesc")
        self._desc_label.setWordWrap(True)
        b.addWidget(self._desc_label)

        self._section_label = QLabel("执行建议")
        self._section_label.setObjectName("IntentSectionLabel")
        b.addWidget(self._section_label)

        self._suggestions_box = QWidget()
        self._suggestions_layout = QVBoxLayout(self._suggestions_box)
        self._suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self._suggestions_layout.setSpacing(6)

        self._empty_label = QLabel("暂无新建议，继续操作试试")
        self._empty_label.setObjectName("IntentEmptyHint")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.hide()
        self._suggestions_layout.addWidget(self._empty_label)
        b.addWidget(self._suggestions_box)
        cv.addWidget(body, 1)

        # 卡片底部
        footer = QFrame()
        footer.setObjectName("IntentCardFooter")
        footer.setFixedHeight(30)
        f = QHBoxLayout(footer)
        f.setContentsMargins(12, 0, 12, 0)
        f.setSpacing(6)
        self._hint_label = QLabel("操作后将自动分析意图并给出建议")
        self._hint_label.setObjectName("IntentCardHint")
        f.addWidget(self._hint_label)
        cv.addWidget(footer)

        # 卡片独立于胶囊布局，由 _reparent_card 挂载到主窗口顶层

        # 分析中脉冲动画
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_timer.timeout.connect(self._on_pulse)

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ----------------------- 卡片定位（嵌入模式） -----------------------

    def _reposition_card(self):
        """将详情卡片定位到胶囊正下方（菜单栏下缘）"""
        if not self._card.isVisible():
            return
        win = self.window()
        if not win:
            return
        bar = getattr(win, "_title_bar", None)
        # 标题栏下缘在窗口坐标系中的 y（考虑 1px 外层边距）
        if bar:
            bar_bottom = bar.mapTo(win, bar.rect().bottomLeft()).y() + 1
        else:
            bar_bottom = 40
        # 卡片为主窗口子控件，使用主窗口本地坐标
        pill_center = self._pill.mapTo(win, self._pill.rect().center())
        x = max(0, pill_center.x() - self._card.width() // 2)
        self._card.setGeometry(x, bar_bottom, self._card.width(), self._card.height())
        self._card.raise_()

    # ----------------------- 事件 -----------------------

    def moveEvent(self, event):
        """胶囊在标题栏中位置变化时，通知外部重新定位卡片"""
        super().moveEvent(event)
        self._update_geometry()

    def _show_card(self):
        if self._card.isVisible():
            return
        # 卡片脱离布局管理，显示前按内容重新计算高度
        self._card.adjustSize()
        self._card.show()
        self._update_geometry()

    def _hide_card(self):
        if not self._card.isVisible():
            return
        self._card.hide()

    def toggle_pin(self):
        self._pinned = not self._pinned
        self._pin_icon.setVisible(self._pinned)
        self._pin_btn.setProperty("active", self._pinned)
        self._repolish(self._pin_btn)
        self._pill.setProperty("pinned", self._pinned)
        self._repolish(self._pill)
        if self._pinned:
            self._show_card()
        else:
            self._hide_card()
        self._resize_to_fit()

    def _update_geometry(self):
        """通知外部更新卡片位置"""
        self.geometry_changed.emit()

    def _resize_to_fit(self):
        """胶囊内容变化后按内容调整宽度，并通知外部重新居中 / 定位卡片

        直接基于字体度量实时计算文本宽度，绕开 Qt 布局 sizeHint 缓存
        （setText 后需事件循环处理 LayoutRequest 才刷新，无法同步跟随）
        """
        fm = self._pill_text.fontMetrics()
        text_w = fm.horizontalAdvance(self._pill_text.text())
        m = self._pill.layout().contentsMargins()
        sp = self._pill.layout().spacing()
        pin_w = self._pin_icon.width() if self._pin_icon.isVisible() else 0
        w = (m.left() + m.right()
             + (sp + pin_w if pin_w else 0)
             + self._dot.width() + sp + text_w + 2)
        self.resize(w, self.height())
        self._update_geometry()
        bar = getattr(self.window(), "_title_bar", None)
        if bar is not None:
            recenter = getattr(bar, "recenter_intent", None)
            if recenter is not None:
                recenter()

    # ----------------------- 状态 -----------------------

    def _set_pill_state(self, state: str):
        self._pill_state = state
        if state == "analyzing":
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
            self._pulse_on = False
        self._dot.setProperty("state", state)
        self._dot.setProperty("pulse", self._pulse_on)
        self._repolish(self._dot)

    def _on_pulse(self):
        self._pulse_on = not self._pulse_on
        self._dot.setProperty("pulse", self._pulse_on)
        self._repolish(self._dot)

    def update_analysis(self, result):
        if not result or not self.is_enabled():
            return
        intent = result.get("intent", "其他操作")
        if intent == "idle":
            intent = "等待操作"
        elif intent == "unknown":
            intent = "未知"
        desc = result.get("intent_description", "")
        suggestions = result.get("suggestions") or []
        self._pill_text.setText(intent)
        self._badge_text.setText(intent)
        self._desc_label.setText(desc)
        self._set_suggestions(suggestions)
        state = "active" if suggestions else "idle"
        self._set_pill_state(state)
        self._intent_badge.setProperty("state", state)
        self._repolish(self._intent_badge)
        self._resize_to_fit()
        self._reposition_card()

    def set_enabled(self, enabled: bool):
        self._switch.blockSignals(True)
        self._switch.setChecked(enabled)
        self._switch.blockSignals(False)
        self._apply_enabled_state(enabled)

    def is_enabled(self) -> bool:
        return self._switch.isChecked()

    def _on_switch_changed(self, checked: bool):
        self._apply_enabled_state(checked)
        self.enabled_toggled.emit(checked)

    def _apply_enabled_state(self, enabled: bool):
        self._clear_suggestions()
        if enabled:
            self._pill_text.setText("等待操作")
            self._badge_text.setText("等待操作")
            self._desc_label.setText("")
            self._hint_label.setText("操作后将自动分析意图并给出建议")
            self._section_label.setText("执行建议")
            self._set_pill_state("idle")
            self._intent_badge.setProperty("state", "idle")
        else:
            self._pill_text.setText("意图分析已关闭")
            self._badge_text.setText("已关闭")
            self._desc_label.setText("")
            self._hint_label.setText("意图分析已关闭，操作不再被分析")
            self._section_label.setText("执行建议")
            self._set_pill_state("off")
            self._intent_badge.setProperty("state", "off")
        self._repolish(self._intent_badge)
        self._resize_to_fit()
        self._reposition_card()

    def toggle_expand(self):
        """点击胶囊：展开详情卡片；再点收起。"""
        if self._card.isVisible():
            self._hide_card()
        else:
            self._show_card()

    # ----------------------- 建议 -----------------------

    def _set_suggestions(self, suggestions):
        self._clear_suggestions()
        for s in suggestions[:3]:
            item = IntentSuggestionItem(
                s.get("title", "建议"),
                s.get("message", ""),
                s.get("action", ""),
                s.get("params", {}),
            )
            item.suggestion_clicked.connect(self.suggestion_clicked)
            self._suggestions_layout.addWidget(item)
            self._suggestion_items.append(item)
        has_sug = bool(suggestions)
        self._empty_label.setVisible(not has_sug)
        self._hint_label.setText(
            "发现 %d 条建议，点击「执行」即可直接操作" % len(suggestions)
            if has_sug else "暂无新建议，继续操作试试"
        )

    def _clear_suggestions(self):
        for item in self._suggestion_items:
            self._suggestions_layout.removeWidget(item)
            item.deleteLater()
        self._suggestion_items.clear()

    # ----------------------- 主题 -----------------------

    def _apply_theme(self):
        if theme_manager.is_light_theme():
            qss = self._LIGHT_QSS
        else:
            qss = self._DARK_QSS
        self.setStyleSheet(qss)
        # 卡片已脱离胶囊子树，需同步应用完整样式
        if hasattr(self, "_card") and self._card.parent() is not self:
            self._card.setStyleSheet(qss)

    _DARK_QSS = """
    #IntentFloatWidget {
        background: transparent;
    }
    #IntentPill {
        background-color: rgba(30, 33, 40, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 15px;
    }
    #IntentPill:hover, #IntentPill[pinned="true"] {
        border: 1px solid #3b82f6;
    }
    #IntentPillDot { color: #9aa3b2; font-size: 7px; }
    #IntentPillDot[state="active"] { color: #3b82f6; }
    #IntentPillDot[state="off"] { color: #6b7280; }
    #IntentPillDot[state="analyzing"] { color: #3b82f6; }
    #IntentPillDot[state="analyzing"][pulse="true"] { color: #93c5fd; }
    #IntentPillText { color: #e7e9ee; font-size: 12px; font-weight: 500; }
    #IntentPillPin { color: #3b82f6; font-size: 11px; }

    #IntentCard {
        background-color: #23262e;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
    }
    #IntentCardHeader {
        background-color: rgba(0, 0, 0, 0.12);
        border-top-left-radius: 13px;
        border-top-right-radius: 13px;
    }
    #IntentCardIcon {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 #2563eb, stop:1 #34d399);
        border-radius: 8px;
        color: #ffffff;
        font-size: 14px;
    }
    #IntentCardTitle { color: #e7e9ee; font-size: 13px; font-weight: 600; }
    #IntentCardSub { color: #8b93a3; font-size: 11px; }
    #IntentCardBtn {
        background: transparent;
        border: none;
        border-radius: 6px;
        color: #6b7280;
        font-size: 12px;
    }
    #IntentCardBtn:hover { background: rgba(59, 130, 246, 0.16); color: #60a5fa; }
    #IntentCardBtn[active="true"] { color: #60a5fa; }

    #IntentBadge { border-radius: 999px; }
    #IntentBadge[state="active"] {
        background-color: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    #IntentBadge[state="active"] #IntentBadgeText,
    #IntentBadge[state="active"] #IntentBadgeDot { color: #60a5fa; }
    #IntentBadge[state="idle"] {
        background-color: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    #IntentBadge[state="idle"] #IntentBadgeText,
    #IntentBadge[state="idle"] #IntentBadgeDot { color: #9aa3b2; }
    #IntentBadge[state="off"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    #IntentBadge[state="off"] #IntentBadgeText,
    #IntentBadge[state="off"] #IntentBadgeDot { color: #6b7280; }
    #IntentBadgeDot { font-size: 5px; }
    #IntentBadgeText { font-size: 12px; font-weight: 600; }

    #IntentCardDesc { color: #9aa3b2; font-size: 12px; }
    #IntentSectionLabel { color: #6b7280; font-size: 11px; font-weight: 600; }
    #IntentSugItem {
        background-color: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 10px;
    }
    #IntentSugTitle { color: #e7e9ee; font-size: 13px; font-weight: 600; }
    #IntentSugMsg { color: #9aa3b2; font-size: 12px; }
    #IntentSugTag {
        color: #6b7280;
        background-color: #1a1d24;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10px;
        font-family: Consolas;
    }
    #IntentSugBtn {
        background-color: #3b82f6;
        color: #ffffff;
        border: none;
        border-radius: 7px;
        padding: 0 12px;
        font-size: 11px;
    }
    #IntentSugBtn:hover { background-color: #60a5fa; }
    #IntentEmptyHint {
        color: #6b7280;
        font-size: 12px;
        border: 1px dashed rgba(255, 255, 255, 0.18);
        border-radius: 10px;
        padding: 12px;
    }
    #IntentCardFooter {
        background-color: rgba(0, 0, 0, 0.12);
        border-bottom-left-radius: 13px;
        border-bottom-right-radius: 13px;
    }
    #IntentCardHint { color: #6b7280; font-size: 11px; }
    """

    _LIGHT_QSS = """
    #IntentFloatWidget {
        background: transparent;
    }
    #IntentPill {
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 15px;
    }
    #IntentPill:hover, #IntentPill[pinned="true"] {
        border: 1px solid #2563eb;
    }
    #IntentPillDot { color: #9ca3af; font-size: 7px; }
    #IntentPillDot[state="active"] { color: #2563eb; }
    #IntentPillDot[state="off"] { color: #9ca3af; }
    #IntentPillDot[state="analyzing"] { color: #2563eb; }
    #IntentPillDot[state="analyzing"][pulse="true"] { color: #93c5fd; }
    #IntentPillText { color: #1d1d1f; font-size: 12px; font-weight: 500; }
    #IntentPillPin { color: #2563eb; font-size: 11px; }

    #IntentCard {
        background-color: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 14px;
    }
    #IntentCardHeader {
        background-color: rgba(0, 0, 0, 0.04);
        border-top-left-radius: 13px;
        border-top-right-radius: 13px;
    }
    #IntentCardIcon {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 #2563eb, stop:1 #34d399);
        border-radius: 8px;
        color: #ffffff;
        font-size: 14px;
    }
    #IntentCardTitle { color: #1d1d1f; font-size: 13px; font-weight: 600; }
    #IntentCardSub { color: #9ca3af; font-size: 11px; }
    #IntentCardBtn {
        background: transparent;
        border: none;
        border-radius: 6px;
        color: #9ca3af;
        font-size: 12px;
    }
    #IntentCardBtn:hover { background: rgba(37, 99, 235, 0.1); color: #2563eb; }
    #IntentCardBtn[active="true"] { color: #2563eb; }

    #IntentBadge { border-radius: 999px; }
    #IntentBadge[state="active"] {
        background-color: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.25);
    }
    #IntentBadge[state="active"] #IntentBadgeText,
    #IntentBadge[state="active"] #IntentBadgeDot { color: #2563eb; }
    #IntentBadge[state="idle"] {
        background-color: rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    #IntentBadge[state="idle"] #IntentBadgeText,
    #IntentBadge[state="idle"] #IntentBadgeDot { color: #4b5563; }
    #IntentBadge[state="off"] {
        background-color: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    #IntentBadge[state="off"] #IntentBadgeText,
    #IntentBadge[state="off"] #IntentBadgeDot { color: #9ca3af; }
    #IntentBadgeDot { font-size: 5px; }
    #IntentBadgeText { font-size: 12px; font-weight: 600; }

    #IntentCardDesc { color: #4b5563; font-size: 12px; }
    #IntentSectionLabel { color: #9ca3af; font-size: 11px; font-weight: 600; }
    #IntentSugItem {
        background-color: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.22);
        border-radius: 10px;
    }
    #IntentSugTitle { color: #1d1d1f; font-size: 13px; font-weight: 600; }
    #IntentSugMsg { color: #4b5563; font-size: 12px; }
    #IntentSugTag {
        color: #9ca3af;
        background-color: #eef0f4;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10px;
        font-family: Consolas;
    }
    #IntentSugBtn {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 7px;
        padding: 0 12px;
        font-size: 11px;
    }
    #IntentSugBtn:hover { background-color: #1d4ed8; }
    #IntentEmptyHint {
        color: #9ca3af;
        font-size: 12px;
        border: 1px dashed rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        padding: 12px;
    }
    #IntentCardFooter {
        background-color: rgba(0, 0, 0, 0.04);
        border-bottom-left-radius: 13px;
        border-bottom-right-radius: 13px;
    }
    #IntentCardHint { color: #9ca3af; font-size: 11px; }
    """
