"""
无边框主窗口
- 自定义标题栏 + 菜单栏
- 左侧导航栏
- 中央内容区域
"""
import os

from PySide6.QtCore import Qt, Slot, QSize, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFrame, QSizePolicy, QScrollArea,
    QPushButton, QToolButton
)
from qfluentwidgets import FluentIcon as FIF, InfoBar

from core.common.config import Config
from core.common.icons import AppIcon
from core.common.settings import settings
from core.common.theme_manager import theme_manager
from core.common.action_registry import action
from core.common.widgets.title_bar import FramelessWindow


class NavigationBar(QWidget):
    """左侧垂直导航栏 - 图标 + tooltip"""

    item_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NavigationBar")
        self.setFixedWidth(48)
        self._items = {}
        self._current = None
        self._top_items = []
        self._bottom_items = []
        self._init_ui()

    def _init_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 8, 4, 8)
        self._layout.setSpacing(4)

        self._top_container = QWidget()
        self._top_layout = QVBoxLayout(self._top_container)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        self._top_layout.setSpacing(4)
        self._top_layout.addStretch(1)
        self._layout.addWidget(self._top_container, 1)

        self._bottom_container = QWidget()
        self._bottom_layout = QVBoxLayout(self._bottom_container)
        self._bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._bottom_layout.setSpacing(4)
        self._bottom_layout.addStretch(1)
        self._layout.addWidget(self._bottom_container, 0)

    def addItem(self, route_key, icon, text, position="top", onClick=None):
        """添加导航项"""
        btn = QToolButton(self)
        btn.setFixedSize(40, 40)
        btn.setToolTip(text)
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self._on_item_clicked(route_key, onClick))
        btn.setObjectName("NavButton")
        btn.setProperty("fluent_icon", icon)
        self._set_icon(btn, icon)

        self._items[route_key] = btn

        if position == "bottom":
            idx = self._bottom_layout.count() - 1
            self._bottom_layout.insertWidget(idx, btn)
            self._bottom_items.append(route_key)
        else:
            idx = self._top_layout.count() - 1
            self._top_layout.insertWidget(idx, btn)
            self._top_items.append(route_key)

    def insertItem(self, index, route_key, icon, text, position="top", onClick=None):
        """在指定位置插入导航项"""
        btn = QToolButton(self)
        btn.setFixedSize(40, 40)
        btn.setToolTip(text)
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self._on_item_clicked(route_key, onClick))
        btn.setObjectName("NavButton")
        btn.setProperty("fluent_icon", icon)
        self._set_icon(btn, icon)

        self._items[route_key] = btn

        if position == "bottom":
            self._bottom_layout.insertWidget(index, btn)
            self._bottom_items.insert(index, route_key)
        else:
            self._top_layout.insertWidget(index, btn)
            self._top_items.insert(index, route_key)

    def setCurrentItem(self, route_key):
        """设置当前选中项"""
        if self._current and self._current in self._items:
            self._items[self._current].setChecked(False)
        if route_key in self._items:
            self._items[route_key].setChecked(True)
            self._current = route_key

    def removeWidget(self, route_key):
        """移除导航项"""
        if route_key in self._items:
            btn = self._items.pop(route_key)
            btn.setParent(None)
            btn.deleteLater()
            if route_key in self._top_items:
                self._top_items.remove(route_key)
            if route_key in self._bottom_items:
                self._bottom_items.remove(route_key)

    def widget(self, route_key):
        """获取导航项widget"""
        return self._items.get(route_key)

    def _set_icon(self, btn, icon):
        """设置图标，根据当前主题自动着色"""
        from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon
        from PySide6.QtCore import Qt
        from core.common.theme_manager import theme_manager

        # 获取原始图标
        if hasattr(icon, 'icon'):
            original_icon = icon.icon()
        else:
            original_icon = icon

        # 根据主题选择颜色
        is_dark = theme_manager.is_dark_theme()
        color = QColor(200, 200, 200) if is_dark else QColor(80, 80, 80)

        # 创建高分辨率 pixmap
        size = 64  # 高分辨率
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(color)
        painter.setBrush(color)

        # 绘制原始图标（使用原始尺寸，让 Qt 自然缩放）
        icon_pixmap = original_icon.pixmap(size, size)
        painter.drawPixmap(0, 0, icon_pixmap)
        painter.end()

        btn.setIcon(QIcon(pixmap))

    def update_icons_for_theme(self):
        """主题切换后更新所有图标"""
        for btn in self._items.values():
            icon = btn.property("fluent_icon")
            if icon:
                self._set_icon(btn, icon)

    def _on_item_clicked(self, route_key, callback):
        self.setCurrentItem(route_key)
        if callback:
            callback()


class MainWindow(FramelessWindow):
    """无边框主窗口，所有子界面由插件动态注册。"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle(Config.APP_NAME)
        self.setMinimumSize(800, 600)

        self._init_icon()
        self._init_size()
        self._init_content()
        self._init_navigation()
        self._init_status_bar()

        self.load_qss()
        theme_manager.theme_changed.connect(self._on_theme_changed)

        # 插件界面注册表
        self._plugin_interfaces = {}
        self._ordered_widgets = []
        self._streaming_assistant = False
        self._pending_tool_blocks: dict = {}  # 工具名 -> 未回结果的工具块队列

    def closeEvent(self, event):
        """关闭窗口前保存当前 AI 会话并收尾 Web 侧栏"""
        if hasattr(self, '_ai_agent') and self._ai_agent.is_ready():
            self._ai_agent.save_session()
        sidebar = getattr(self, "_ai_sidebar", None)
        if sidebar is not None and hasattr(sidebar, "shutdown"):
            sidebar.shutdown()
        super().closeEvent(event)

    def _init_icon(self):
        # 多尺寸图标：Windows 任务栏/Alt-Tab 需要 16/32 等小尺寸，
        # 直接用 app.png 单尺寸图标偶尔会回落成默认程序图标
        from core.common.icons import build_app_icon
        icon = build_app_icon()
        if not icon.isNull():
            self._app_icon = icon  # 保持引用，供 show 后补设
            self.setWindowIcon(icon)

    def _init_size(self):
        screen = QGuiApplication.primaryScreen().availableGeometry()
        screen_w, screen_h = screen.width(), screen.height()

        if screen_w < 1280 or screen_h < 850:
            window_width = max(int(screen_w * 0.95), 800)
            window_height = max(int(screen_h * 0.95), 600)
        else:
            window_width, window_height = 1200, 800

        self.resize(window_width, window_height)
        self.move(screen_w // 2 - window_width // 2, screen_h // 2 - window_height // 2)

    def _init_content(self):
        """初始化主内容区域"""
        content = self.content_layout

        # 主体容器
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # 左侧导航栏
        self.navigationInterface = NavigationBar()
        body_layout.addWidget(self.navigationInterface)

        # 导航栏右侧垂直分隔线
        self._nav_separator = QFrame()
        self._nav_separator.setObjectName("NavBarSeparator")
        self._nav_separator.setFrameShape(QFrame.Shape.VLine)
        self._nav_separator.setFrameShadow(QFrame.Shadow.Plain)
        self._nav_separator.setFixedWidth(1)
        body_layout.addWidget(self._nav_separator)

        # 右侧内容区
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.setObjectName("stackedWidget")
        body_layout.addWidget(self.stackedWidget, 1)

        # AI Chat 侧栏（默认隐藏）
        self._ai_sidebar_separator = QFrame()
        self._ai_sidebar_separator.setObjectName("AgentSidebarSeparator")
        self._ai_sidebar_separator.setFrameShape(QFrame.Shape.VLine)
        self._ai_sidebar_separator.setFrameShadow(QFrame.Shadow.Plain)
        self._ai_sidebar_separator.setFixedWidth(1)
        self._ai_sidebar_separator.setVisible(False)
        body_layout.addWidget(self._ai_sidebar_separator)

        # AI 侧栏:优先用 Web 实现(DSH 视觉复刻前端 + QWebChannel 桥接
        # corecoder),QtWebEngine 不可用时回退经典 Qt 侧栏。两者同构 API。
        self._ai_sidebar_classic = None
        try:
            from core.agent.webui.bridge import WebAgentSidebar
            self._ai_sidebar = WebAgentSidebar()
        except Exception as e:
            print(f"[MainWindow] Web 侧栏不可用({e}),回退经典侧栏")
            from core.agent.agent_dialog import AgentDialog
            self._ai_sidebar = AgentDialog()
            self._ai_sidebar_classic = self._ai_sidebar
        self._ai_sidebar.setVisible(False)
        self._ai_sidebar.message_sent.connect(self._on_ai_message_sent)
        self._ai_sidebar.stop_requested.connect(self._on_ai_stop)
        self._ai_sidebar.model_selected.connect(self._on_sidebar_model_selected)
        self._ai_sidebar.history_loaded.connect(self._on_ai_history_loaded)
        body_layout.addWidget(self._ai_sidebar)

        # AI Agent 初始化
        from core.agent.visionmind_agent import VisionMindAgent
        self._ai_agent = VisionMindAgent(self)
        self._ai_agent.response_ready.connect(self._on_ai_response)
        self._ai_agent.error_occurred.connect(self._on_ai_error)
        self._ai_agent.tool_called.connect(self._on_ai_tool_called)
        self._ai_agent.tool_result.connect(self._on_ai_tool_result)
        self._ai_agent.token_received.connect(self._on_ai_token)
        self._ai_agent.stopped.connect(self._on_ai_stopped)
        # 上下文压缩报告（手动 /compact 或阈值自动触发）→ 以可展开工具卡片展示
        self._ai_agent.compression_report.connect(self._on_compression_report)
        # bash 解锁审批卡:工作线程的申请跨线程投递到侧栏 UI
        self._ai_agent.bash_approval_requested.connect(
            self._ai_sidebar.request_bash_approval
        )
        # ask_user 提问卡:Web 侧栏渲染提问 UI;经典侧栏回退 Qt 对话框
        self._ai_agent.ask_user_requested.connect(self._on_ask_user_requested)
        # 上下文用量统计 → Web 侧栏悬浮卡
        if hasattr(self._ai_sidebar, "push_context_stats"):
            self._ai_agent.context_stats.connect(
                self._ai_sidebar.push_context_stats
            )

        # AI 侧栏调试按钮
        self._ai_sidebar.debug_requested.connect(self._on_open_debug_window)

        # 意图分析状态胶囊：嵌入标题栏菜单栏中央，悬停显示详情卡片
        try:
            from core.agent.intent_agent import IntentAgent
            from core.agent.intent_float_widget import IntentFloatWidget
            self._intent_agent = IntentAgent.instance()
            self._intent_float = IntentFloatWidget(self)
            self._intent_float.suggestion_clicked.connect(self._on_suggestion_clicked)
            self._intent_float.enabled_toggled.connect(self._on_intent_enabled_toggled)
            self._intent_float.geometry_changed.connect(self._on_intent_geometry_changed)
            self._intent_agent.analysis_updated.connect(self._on_intent_analysis)
            self._intent_agent.enabled_changed.connect(self._on_intent_enabled_changed)
            self._ai_sidebar.suggestion_clicked.connect(self._on_suggestion_clicked)
            self._intent_float.set_enabled(self._intent_agent.is_enabled())
            # 嵌入标题栏中央
            self._title_bar.set_intent_widget(self._intent_float)
            self._intent_float.show()
            # LLM 尚未就绪（Agent 初始化后才有），先显示等待状态
            if not self._intent_agent.has_llm():
                self._on_intent_analysis({
                    "intent": "等待初始化",
                    "intent_description": "请先打开 Agent 对话栏配置 API Key，意图分析将在 Agent 就绪后自动启动",
                    "recent_actions": [],
                    "suggestions": [],
                })
        except Exception as e:
            print(f"[MainWindow] 意图分析悬浮框初始化失败: {e}")

        content.addWidget(body)

    def _init_navigation(self):
        """初始化导航（仅底部固定项）"""
        self.navigationInterface.addItem(
            route_key='theme_toggle',
            icon=AppIcon.THEME,
            text='切换主题',
            position='bottom',
            onClick=self.on_theme_toggle_clicked
        )

    def _init_status_bar(self):
        """初始化状态栏"""
        self.statusBar().showMessage("就绪")
        # 连接全局录制状态
        self._connect_recording_signals()

    def _connect_recording_signals(self):
        """连接全局录制状态信号"""
        try:
            from core.common.action_recorder import ActionRecorder
            recorder = ActionRecorder.instance()
            recorder.recording_changed.connect(self._on_recording_changed)
            recorder.step_recorded.connect(self._on_recording_step)
        except Exception as e:
            print(f"[MainWindow] 连接录制信号失败: {e}")

    @Slot(bool)
    def _on_recording_changed(self, recording: bool):
        """录制状态变化时更新标题栏指示器"""
        if recording:
            self._title_bar.show_recording(0)
        else:
            self._title_bar.hide_recording()

    @Slot(str, dict)
    def _on_recording_step(self, action_name: str, step: dict):
        """录制步骤更新时更新标题栏步数"""
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        self._title_bar.update_recording_count(recorder.step_count)

    # ------------------------------------------------------------------
    # 插件注册 API
    # ------------------------------------------------------------------

    def add_plugin_interface(self, widget, icon, text, order=99, position="top"):
        """添加插件界面"""
        if not widget.objectName():
            raise ValueError("插件界面的 objectName 不能为空")

        route_key = widget.objectName()

        # 计算插入位置
        insert_index = 0
        for i, (existing_order, _) in enumerate(self._ordered_widgets):
            if existing_order <= order:
                insert_index = i + 1
            else:
                break

        # 添加到 stackedWidget
        self.stackedWidget.addWidget(widget)

        # 添加到导航栏
        def on_click(w=widget):
            self.switchTo(w)

        # 底部项固定追加在底部（主题切换之后），顶部项按 order 排序
        if position == "bottom":
            nav_index = self.navigationInterface._bottom_layout.count() - 1
        else:
            nav_index = insert_index
        self.navigationInterface.insertItem(
            index=nav_index,
            route_key=route_key,
            icon=icon,
            text=text,
            position=position,
            onClick=on_click
        )

        if self.stackedWidget.count() == 1:
            self.navigationInterface.setCurrentItem(route_key)
            self.switchTo(widget)

        self._plugin_interfaces[widget] = (icon, text, order, route_key)
        self._ordered_widgets.insert(insert_index, (order, widget))

    @action("nav.switch_to", description="切换到应用程序的指定功能界面。\n- target: 目标界面名称，可选值：annotation(标注), dashboard(项目), version(版本管理), config(配置)\n- 切换后该界面的相关工具会自动加载\n- 跨界面调用 action 前需要先切换到此界面", category="导航",
            scope="agent",
            params={"target": {"type": "str", "dynamic_enum": "switch_to_targets", "description": "目标界面名称（objectName、显示名或快捷名）"}})
    def switchTo(self, interface=None, target=""):
        """切换界面"""
        if not target and interface is None:
            return "错误: target 参数不能为空"

        if target and interface is None:
            interface_map = {
                'dashboard': 'DashboardInterface', 'annotation': 'AnnotationInterface',
                'generation': 'GenerationInterface', 'training': 'TrainingInterface',
                'inference': 'InferenceInterface', 'config': 'ConfigInterface',
                'flow': 'FlowInterface', 'kachi': 'KachiInterface',
                'version': 'VersionInterface', 'unsupervised': 'UnsupervisedTrainingInterface',
                'test_interface': 'TestInterface',
            }
            object_name = interface_map.get(target, target)
            interface = self.find_interface(object_name)
            if interface is None:
                return f"错误: 无效的目标界面名称: {target}"

        if interface is None:
            return "错误: 无法找到目标界面"

        self.stackedWidget.setCurrentWidget(interface)
        route_key = interface.objectName()
        self.navigationInterface.setCurrentItem(route_key)

        # 更新菜单栏
        self._update_menubar(interface)

        return True

    def find_interface(self, object_name):
        """根据 objectName 查找界面"""
        for widget in self._plugin_interfaces:
            if widget.objectName() == object_name:
                return widget
        return None

    def remove_plugin_interface(self, widget):
        """移除插件界面"""
        if widget not in self._plugin_interfaces:
            return

        icon, text, order, route_key = self._plugin_interfaces.pop(widget)
        self.navigationInterface.removeWidget(route_key)
        self.stackedWidget.removeWidget(widget)
        widget.hide()
        self._ordered_widgets = [(o, w) for o, w in self._ordered_widgets if w is not widget]

    # ------------------------------------------------------------------
    # 菜单栏管理
    # ------------------------------------------------------------------

    def _update_menubar(self, interface):
        """根据当前界面更新菜单栏"""
        self._title_bar.clear_menus()

        # 隐藏插件内部菜单栏，统一使用顶部全局菜单栏
        if hasattr(interface, 'hide_internal_menubar'):
            interface.hide_internal_menubar()

        # 优先使用插件提供的 build_menus 直接构建完整菜单
        if hasattr(interface, 'build_menus'):
            interface.build_menus(self._title_bar.get_menu_bar())
            return

        # 兼容旧版配置化菜单
        menus = None
        if hasattr(interface, 'get_menubar_config'):
            menus = interface.get_menubar_config()

        if menus:
            for menu_config in menus:
                self._title_bar.add_menu(menu_config['title'], menu_config.get('actions', []))

    # ------------------------------------------------------------------
    # 主题 & 样式
    # ------------------------------------------------------------------

    def load_qss(self):
        if theme_manager.is_light_theme():
            qss_path = os.path.join(Config.QSS_DIR, "light_theme.qss")
        else:
            qss_path = os.path.join(Config.QSS_DIR, "dark_theme.qss")

        if os.path.exists(qss_path):
            with open(qss_path, 'r', encoding='utf-8') as f:
                content = f.read()
                QApplication.instance().setStyleSheet(content)

    def _on_theme_changed(self, theme):
        old_size = self.size()
        old_pos = self.pos()
        self.load_qss()
        self.resize(old_size)
        self.move(old_pos)

    @Slot()
    @action("nav.toggle_theme", description="切换应用程序的明暗主题。\n- 无参数\n- 在明亮主题和暗色主题之间切换\n- 切换后所有界面同步更新", category="导航", scope="agent")
    def on_theme_toggle_clicked(self):
        new_theme = 'light' if theme_manager.is_dark_theme() else 'dark'
        theme_manager.set_theme(new_theme)

        # 更新导航栏文字颜色
        self._update_nav_colors()

        theme_text = '亮色主题' if new_theme == 'light' else '暗色主题'
        InfoBar.success(
            title='主题已切换',
            content=f'已切换到{theme_text}',
            orient=Qt.Horizontal,
            isClosable=True,
            duration=2000,
            parent=self
        )

    def _update_nav_colors(self):
        """根据主题更新导航栏和标题栏颜色 - 通过重新加载QSS实现"""
        # 重新加载QSS（已由 theme_manager 处理）
        # 更新导航栏图标颜色
        self.navigationInterface.update_icons_for_theme()

    @action("nav.open_document", read_only=True, description="打开 Markdown 文档编辑器窗口，可阅读/修改 reports 目录下的文档。\n- path: 文档完整路径，省略时打开最近生成的文档或新建文档\n- 非模态窗口，编辑区 + 实时预览分栏展示", category="导航", scope="ui")
    def on_open_document_clicked(self, path=""):
        """打开文档编辑器窗口"""
        from core.common.widgets.document_editor_dialog import DocumentEditorDialog
        dlg = DocumentEditorDialog.get_or_create()
        dlg.open_file(path if path else "")

    # ------------------------------------------------------------------
    # AI Chat 侧栏
    # ------------------------------------------------------------------

    def toggle_ai_sidebar(self):
        """切换 AI 侧栏显示/隐藏"""
        visible = not self._ai_sidebar.isVisible()
        self._ai_sidebar.setVisible(visible)
        self._ai_sidebar_separator.setVisible(visible)

        # 更新标题栏按钮状态
        if hasattr(self._title_bar, '_ai_toggle_btn'):
            self._title_bar._ai_toggle_btn.setChecked(visible)

        # 如果首次打开，初始化 Agent
        if visible and not self._ai_agent.is_ready():
            self._init_ai_agent()

    def _init_ai_agent(self):
        """初始化 AI Agent（首次打开侧栏时）"""
        from core.common.ai_providers import get_active_credentials
        api_key, base_url, model = get_active_credentials()

        if not api_key:
            self._ai_sidebar.add_message("system", "请先在系统配置中设置 API Key。")
            return

        self._ai_agent.initialize(api_key, base_url, model, main_window=self)
        self._ai_sidebar.set_model(model)

        # 连接操作记录器
        from core.agent.operation_logger import OperationLogger
        self._op_logger = OperationLogger.instance()
        print("[MainWindow] AI Agent 初始化完成，action 队列定时器已启动")

    def _on_sidebar_model_selected(self, provider_id: str, model: str):
        """对话框模型下拉切换（跨提供商即时生效）：重建 Agent 并更新激活配置。"""
        from core.common.ai_providers import (
            get_provider, set_active_provider, update_provider,
        )
        p = get_provider(provider_id)
        if p is None:
            return
        set_active_provider(provider_id)
        update_provider(provider_id, default_model=model)
        if not p.get("api_key"):
            self._ai_sidebar.add_message(
                "system", f"提供商「{p.get('name', provider_id)}」未配置 API Key，"
                          f"请先在系统配置中填写。"
            )
            return
        if self._ai_agent.is_busy():
            self._ai_sidebar.add_message(
                "system", "Agent 正在执行任务，模型将在当前任务结束后切换。"
            )
            return
        self._ai_agent.initialize(
            p.get("api_key", ""), p.get("base_url", "https://api.deepseek.com"),
            model, main_window=self,
        )
        self._ai_sidebar.set_model(model)
        print(f"[MainWindow] 已切换提供商「{p.get('name')}」/ 模型 {model}")

    def set_event_bus(self, bus):
        """把事件总线转发给 AI 侧栏(供 chat.reference_added 订阅)。"""
        if hasattr(self, "_ai_sidebar"):
            self._ai_sidebar.set_event_bus(bus)

    @Slot(object)
    def _on_ai_message_sent(self, text: object):
        """用户在 AI 侧栏发送消息（支持 /plan、/compact 指令）"""
        print(f"[用户] {text}")
        self._pending_tool_blocks.clear()

        if isinstance(text, str):
            stripped = text.strip()
            if stripped.startswith("/plan"):
                rest = stripped[5:].strip()
                if not rest:
                    if self._ai_agent.plan_mode:
                        self._exit_plan_mode()
                        self._ai_sidebar.add_message("system", "已退出计划模式，全部工具已恢复可见。")
                    else:
                        self._enter_plan_mode()
                    return
                if not self._ai_agent.plan_mode:
                    self._enter_plan_mode()
                text = rest
            elif stripped.startswith("/compact"):
                self._run_manual_compact()
                return

        # 重置流式状态
        self._streaming_assistant = False
        self._ai_sidebar.show_exec_status("分析任务中...")
        self._ai_agent.chat(text)

    def _enter_plan_mode(self):
        self._ai_agent.set_plan_mode(True)
        self._ai_sidebar.add_message(
            "system",
            "已进入计划模式：仅开放只读工具（查状态/读知识段/搜索），"
            "执行类工具不可见。完成调研后将输出执行计划供你审批。"
            "再次发送 /plan 可退出。",
        )
        if hasattr(self._ai_sidebar, "set_plan_banner"):
            self._ai_sidebar.set_plan_banner(True)

    def _exit_plan_mode(self):
        self._ai_agent.set_plan_mode(False)
        if hasattr(self._ai_sidebar, "set_plan_banner"):
            self._ai_sidebar.set_plan_banner(False)

    def _run_manual_compact(self):
        """手动压缩上下文（压缩调用 LLM 摘要，可能耗时数秒）"""
        if self._ai_agent.is_busy():
            self._ai_sidebar.add_message("system", "Agent 正在执行任务，请稍后再压缩。")
            return
        self._ai_sidebar.show_exec_status("正在压缩上下文...")
        try:
            # 压缩报告经 compression_report 信号以可展开工具卡片展示
            self._ai_agent.compact_now()
        except Exception as e:
            self._ai_sidebar.add_message("system", f"压缩失败: {e}")
        self._ai_sidebar.hide_exec_status()

    @Slot(str)
    def _on_compression_report(self, report: str):
        """上下文压缩报告（手动 /compact 或阈值自动触发）→ 以可展开工具卡片展示。"""
        if not report:
            return
        try:
            if not self._streaming_assistant and self._ai_sidebar._current_assistant_bubble is None:
                self._ai_sidebar.start_assistant_message("")
            self._ai_sidebar.add_tool_call(
                tool_name="上下文压缩",
                status="success",
                input_data="压缩上下文",
                output_data=report,
            )
        except Exception as e:
            print(f"[MainWindow] 压缩报告展示失败: {e}")

    def _maybe_show_plan_approval(self):
        """计划模式下回复结束 → 弹计划审批卡（执行 / 继续修改）"""
        if not self._ai_agent.plan_mode:
            return

        def on_decide(execute: bool):
            if execute:
                self._exit_plan_mode()
                self._ai_sidebar.add_message("user", "计划已批准，请按执行计划开始实施。")
                self._on_ai_message_sent("计划已批准，请按执行计划开始实施。")
            # 继续修改:保持计划模式,等用户下一条输入

        if hasattr(self._ai_sidebar, "show_plan_approval"):
            self._ai_sidebar.show_plan_approval(on_decide)
        else:
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("执行计划已生成")
            box.setText("计划模式：执行计划已生成，是否开始执行？")
            execute_btn = box.addButton("▶ 执行计划", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("✎ 继续修改", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            on_decide(box.clickedButton() is execute_btn)

    @Slot(object)
    def _on_ai_history_loaded(self, messages: object):
        """加载会话历史时把清理后的消息写回 Agent，使后续对话带上历史"""
        try:
            if isinstance(messages, list) and hasattr(self, "_ai_agent"):
                self._ai_agent.set_history(messages)
        except Exception:
            pass

    @Slot(str)
    def _on_ai_token(self, token: str):
        """流式接收 LLM token"""
        if not token:
            return
        if not token.strip() and not self._streaming_assistant:
            # 尚未开始回复的孤立空白 token(如工具调用轮次的换行符)不建气泡
            return
        if not self._streaming_assistant:
            self._ai_sidebar.start_assistant_message("")
            self._streaming_assistant = True
        # 回复进行中的空白 token 必须追加:模型常把换行符单独成 token 发送,
        # 丢弃会破坏 markdown 结构(表格/标题/列表都依赖行首换行)
        self._ai_sidebar.append_to_assistant(token)

    @Slot(str)
    def _on_ai_response(self, response: str):
        """收到 AI 完整回复 — 仅用于最终完成"""
        self._streaming_assistant = False
        self._ai_sidebar.show_thinking(False)
        self._ai_sidebar.finalize_assistant()
        self._ai_sidebar.hide_exec_status()
        if hasattr(self._ai_sidebar, "mark_turn_finished"):
            self._ai_sidebar.mark_turn_finished()
        # 计划模式下回复结束 → 提交计划审批(执行 / 继续修改)
        self._maybe_show_plan_approval()

    @Slot()
    def _on_ai_stopped(self):
        """worker 因用户停止而退出 — 确保 UI 状态已清理"""
        self._streaming_assistant = False
        self._ai_sidebar.show_thinking(False)
        self._ai_sidebar.finalize_assistant()
        self._ai_sidebar.hide_exec_status()
        if hasattr(self._ai_sidebar, "mark_turn_finished"):
            self._ai_sidebar.mark_turn_finished()

    @Slot(dict)
    def _on_intent_analysis(self, result: dict):
        """收到实时意图分析结果 — 刷新悬浮状态框"""
        if not result:
            return
        if hasattr(self, "_intent_float"):
            self._intent_float.update_analysis(result)

    @Slot(str, dict)
    def _on_suggestion_clicked(self, action_name: str, params: dict):
        """用户点击建议执行按钮 — 发送到对话侧栏，由 Agent 处理执行"""
        if not action_name:
            return

        # 自动展开 Agent 对话框
        if not self._ai_sidebar.isVisible():
            self._ai_sidebar.setVisible(True)
            self._ai_sidebar_separator.setVisible(True)
            if hasattr(self._title_bar, '_ai_toggle_btn'):
                self._title_bar._ai_toggle_btn.setChecked(True)

        # 构建发送给 Agent 的消息
        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        display_name = action_name
        try:
            meta = registry.get_meta(action_name)
            if meta:
                display_name = meta.name
        except Exception:
            pass

        params_str = ""
        if params:
            import json
            params_str = f"，参数: {json.dumps(params, ensure_ascii=False)}"

        # 在侧栏显示用户消息（建议内容），然后通过 Agent 执行
        message = f"请执行建议: {display_name}{params_str}"
        self._ai_sidebar.add_message("user", message)

        # 确保 Agent 已初始化
        if not self._ai_agent.is_ready():
            self._init_ai_agent()
        if self._ai_agent.is_ready():
            self._on_ai_message_sent(message)
        else:
            self._ai_sidebar.add_message("system", "Agent 未初始化，请先配置 API Key 后再使用建议执行功能。")

    @Slot(bool)
    def _on_intent_enabled_toggled(self, enabled: bool):
        """悬浮框开关切换 — 同步意图分析启停"""
        if hasattr(self, "_intent_agent"):
            self._intent_agent.set_enabled(enabled)
        else:
            from core.agent.intent_agent import IntentAgent
            IntentAgent.instance().set_enabled(enabled)

    @Slot(bool)
    def _on_intent_enabled_changed(self, enabled: bool):
        """意图分析开关状态变化 — 同步悬浮框显示"""
        if hasattr(self, "_intent_float"):
            self._intent_float.set_enabled(enabled)

    @Slot()
    def _on_intent_geometry_changed(self):
        """意图胶囊/卡片位置变化时，重新定位详情卡片到胶囊正下方"""
        if not hasattr(self, "_intent_float"):
            return
        card = self._intent_float._card
        if not card.isVisible():
            return
        self._intent_float._reposition_card()

    @Slot(str)
    def _on_ai_error(self, error: str):
        """AI 调用出错"""
        print(f"[AI 错误] {error}")
        self._streaming_assistant = False
        self._ai_sidebar.show_thinking(False)
        self._ai_sidebar.finalize_assistant()
        self._ai_sidebar.hide_exec_status()
        if hasattr(self._ai_sidebar, "mark_turn_finished"):
            self._ai_sidebar.mark_turn_finished()
        self._ai_sidebar.add_message("system", f"错误: {error}")

    @Slot()
    def _on_ai_stop(self):
        """用户点击停止按钮"""
        # 请求 agent 停止当前 worker（在下一个工具调用前退出）
        if hasattr(self, '_ai_agent') and self._ai_agent:
            self._ai_agent.stop()
        # 立即更新 UI 状态（worker 退出后会通过 stopped 信号再次确认）
        self._ai_sidebar.hide_exec_status()
        self._ai_sidebar.show_thinking(False)
        self._ai_sidebar.finalize_assistant()
        self._ai_sidebar.add_message("system", "⚠️ 执行已停止。")

    @Slot(str, dict)
    def _on_ai_tool_called(self, tool_name: str, args: dict):
        """AI 调用工具 — 先完成流式消息，再显示工具调用"""
        # 完成当前正在流式输出的助手消息
        self._streaming_assistant = False
        self._ai_sidebar.finalize_assistant()

        parts = []
        for k, v in args.items():
            if isinstance(v, str):
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v}")
        args_str = ", ".join(parts) if parts else ""
        print(f"[AI 工具] {tool_name}({args_str})")
        # 使用新的工具调用展示块 API
        if not self._streaming_assistant and self._ai_sidebar._current_assistant_bubble is None:
            self._ai_sidebar.start_assistant_message("")
        block = self._ai_sidebar.add_tool_call(
            tool_name=tool_name, status="running", input_data=args_str
        )
        # 并行工具调用时 tool_called 全部先到、结果交错返回,
        # 单一 _current_tool_block 会把结果写错行(表现为工具行永远 running)。
        # 改为按工具名排队,结果按名字+顺序对号入座。
        self._pending_tool_blocks.setdefault(tool_name, []).append(block)

    @Slot(str, str)
    def _on_ai_tool_result(self, tool_name: str, result: str):
        """收到工具执行结果"""
        try:
            result = result if isinstance(result, str) else (
                "" if result is None else str(result))
            if tool_name == "view_image" and result.startswith("[IMG_VIEW:"):
                # 机制 B: 标记将由 llm.chat 包装层编码并注入 user 图片消息
                display = result.replace("[IMG_VIEW:", "已请求加载图片").rstrip("]")
            else:
                # 结果过长时截断显示
                display = result[:500] + "..." if len(result) > 500 else result
        except Exception as e:
            print(f"[AI 工具结果] {tool_name} 结果处理异常: {e}")
            display = str(result)
        queue = getattr(self, "_pending_tool_blocks", {}).get(tool_name)
        block = queue.pop(0) if queue else None
        print(f"[AI 工具结果] {tool_name}: {len(display)} chars")
        if block is not None:
            self._ai_sidebar.update_tool_status(block, status="success", output=display)
        else:
            self._ai_sidebar.update_tool_status(tool_name, status="success", output=display)

    @Slot(str, object)
    def _on_ask_user_requested(self, questions_json: str, respond):
        """ask_user 工具请求用户作答:Web 侧栏出提问卡,经典侧栏回退 Qt 对话框"""
        import json as _json
        try:
            qlist = _json.loads(questions_json)
        except Exception:
            qlist = []
        if hasattr(self._ai_sidebar, "request_ask_user"):
            self._ai_sidebar.request_ask_user(qlist, respond)
            return
        from PySide6.QtWidgets import QInputDialog
        answers = []
        for q in qlist or []:
            options = q.get("options") or []
            title = q.get("header") or "Agent 提问"
            if options:
                sel, ok = QInputDialog.getItem(
                    self, title, q.get("question", ""), options, 0, False)
                if not ok:
                    respond(None)
                    return
                answers.append({"id": q.get("id", ""), "selected": [sel]})
            else:
                text, ok = QInputDialog.getText(
                    self, title, q.get("question", ""))
                answers.append({
                    "id": q.get("id", ""),
                    "selected": [],
                    "custom": text if ok else "",
                })
        respond(answers)

    @Slot()
    def _on_open_debug_window(self):
        """打开系统提示词调试窗口"""
        from core.agent.debug_window import SystemPromptDebugDialog
        dialog = SystemPromptDebugDialog(self)
        dialog.exec()
