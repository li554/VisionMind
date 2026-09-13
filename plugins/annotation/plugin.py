from core.plugin_manager import IPlugin, PluginContext


class AnnotationPlugin(IPlugin):
    @staticmethod
    def plugin_id(): return "annotation"

    @staticmethod
    def plugin_name(): return "手动标注"

    @staticmethod
    def plugin_icon(): return "TAG"

    @staticmethod
    def dependencies(): return ["dashboard"]

    def on_load(self, ctx: PluginContext):
        from core.event_bus import StateType
        from .interfaces.annotation_interface import AnnotationInterface

        self.interface = AnnotationInterface(ctx.project_service, None)
        # 先注册领域服务，让 @action(scope="agent") 优先注册
        ctx.register_service(self.interface.manual, prefix="annotation")
        ctx.register_service(self.interface.auto, prefix="annotation")
        ctx.register_interface(self.interface, "TAG", "手动标注", order=2)
        self.interface.set_event_bus(ctx.event_bus)
        self.interface.manual.set_event_bus(ctx.event_bus)
        # 声明响应的统一状态事件（核心 StateRouter 统一接收 app:state_changed 后按 type 分发）
        self._handlers = {
            StateType.IMAGE_CHANGED:       self.interface.on_annotation_image_changed,
            StateType.ANNOTATIONS_CHANGED: self.interface.on_annotations_changed,
            StateType.RULES_CHANGED: self.interface.on_rules_changed,
            StateType.CATEGORIES_CHANGED:  self.interface.on_categories_changed,
            StateType.FILTERS_CHANGED:     self.interface.on_filters_changed,
            StateType.DATASETS_CHANGED:    self.interface.on_datasets_changed,
            StateType.HARD_SAMPLE_CHANGED: self.interface.on_hard_sample_changed,
            StateType.TASK_MODE_CHANGED:   self.interface.on_task_mode_changed,
            StateType.BATCH_PROGRESS:      self.interface.on_batch_progress,
            StateType.PROJECTS_CHANGED:    self.interface.on_projects_changed,
            StateType.PROJECT_SELECTED:    self.on_project_changed,
        }
        ctx.register_state_handlers(self._handlers)

        # 注册配置卡片
        from plugins.annotation.interfaces.settings_card import AnnotationSettingsCard
        ctx.register_config_card("标注设置", AnnotationSettingsCard, order=20)

    def on_unload(self, ctx: PluginContext):
        ctx.unregister_state_handlers(self._handlers)
        ctx.remove_interface(self.interface)
        self.interface = None

    def on_project_changed(self, project_info):
        if hasattr(self.interface, 'set_project'):
            self.interface.set_project(project_info)
