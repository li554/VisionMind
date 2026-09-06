from core.plugin_manager import IPlugin, PluginContext

class VersionPlugin(IPlugin):
    @staticmethod
    def plugin_id(): return "version"
    @staticmethod
    def plugin_name(): return "版本管理"
    @staticmethod
    def plugin_icon(): return "HISTORY"
    @staticmethod
    def dependencies(): return ["dashboard"]
    def on_load(self, ctx):
        from core.event_bus import StateType
        from .interfaces.version_interface import VersionInterface
        self.interface = VersionInterface(ctx.project_service, None)
        # 先注册 service，让 @action(scope="agent") 优先注册
        ctx.register_service(self.interface.version_service, prefix="version")
        ctx.register_interface(self.interface, "HISTORY", "版本管理", order=3)
        from .interfaces.settings_card import VersionSettingsCard
        ctx.register_config_card("版本设置", VersionSettingsCard, order=25)

        # 声明响应的统一状态事件（核心 StateRouter 统一分发）
        self._handlers = {
            StateType.PROJECT_SELECTED:  self.on_project_changed,
            StateType.PROJECTS_CHANGED:  self.on_projects_changed,
        }
        ctx.register_state_handlers(self._handlers)

    def on_unload(self, ctx):
        ctx.unregister_state_handlers(self._handlers)
        ctx.remove_interface(self.interface)
        self.interface = None
    def on_project_changed(self, project_info):
        if hasattr(self.interface, 'set_project'):
            self.interface.set_project(project_info)
    def on_projects_changed(self, data):
        """订阅 dashboard:projects_changed：刷新项目下拉列表。"""
        try:
            if hasattr(self.interface, 'refresh_project_list'):
                self.interface.refresh_project_list()
        except Exception as e:
            print(f"[VersionPlugin] 刷新项目列表失败: {e}")
