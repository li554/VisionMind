from core.plugin_manager import IPlugin, PluginContext


class DashboardPlugin(IPlugin):
    @staticmethod
    def plugin_id(): return "dashboard"

    @staticmethod
    def plugin_name(): return "项目大厅"

    @staticmethod
    def plugin_icon(): return "HOME"

    @staticmethod
    def dependencies(): return []

    def on_load(self, ctx: PluginContext):
        from core.event_bus import StateType
        from .interfaces.dashboard_interface import DashboardInterface

        self.interface = DashboardInterface(ctx.project_service)
        # 先注册 service，让 @action(scope="agent") 优先注册
        ctx.register_service(self.interface.dashboard_service, prefix="dashboard")
        ctx.register_interface(self.interface, "HOME", "项目大厅", order=1)

        # 保存上下文引用
        self._ctx = ctx

        # 声明响应的统一状态事件（核心 StateRouter 统一分发）
        ctx.register_state_handlers({
            StateType.PROJECT_SELECTED:  self.on_project_changed,
            StateType.PROJECTS_CHANGED:  self.on_projects_changed,
        })

        # 连接 DashboardInterface 的 project_selected 信号，发布事件到事件总线
        # 将项目名称转换为完整的项目信息 dict 后再发布，并切换到标注界面
        def _on_project_selected(name):
            project_info = ctx.project_service.load_project(name)
            if project_info:
                ctx.publish_state("project.project_selected", project_info)
                # 切换到标注界面
                self._navigate_to_annotation(ctx)

        self.interface.project_selected.connect(_on_project_selected)

    def on_unload(self, ctx: PluginContext):
        ctx.remove_interface(self.interface)
        self.interface = None

    def _navigate_to_annotation(self, ctx):
        """项目选中后切换到标注界面"""
        main_window = ctx._main_window
        annotation_widget = main_window.find_interface('AnnotationInterface')
        if annotation_widget:
            main_window.switchTo(annotation_widget)

    def on_project_changed(self, project_info):
        pass  # Dashboard publishes this event, doesn't need to respond

    def on_projects_changed(self, data):
        """订阅 dashboard:projects_changed：刷新项目卡片列表。"""
        try:
            if hasattr(self.interface, 'refresh_projects'):
                self.interface.refresh_projects()
        except Exception as e:
            print(f"[DashboardPlugin] 刷新项目列表失败: {e}")
