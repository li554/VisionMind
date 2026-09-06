"""
跨插件事件总线，支持发布/订阅模式
"""
from typing import Callable, Any, Dict, List


# 统一状态路由信号：全网唯一 topic，所有领域/UI 状态变更都经它发布，
# 由 StateRouter 按 payload["type"] 分发给各插件注册的处理器。
ROUTER_SIGNAL = "app:state_changed"


class StateType:
    """统一状态事件的 type 命名空间（开放）。

    命名规范：`{plugin_id}.{event}`，以插件/域为前缀，天然防跨插件重名。
    以下为本项目内置常用 type 的常量引用；插件自定义 type 直接用字符串
    `"{plugin_id}.{event}"` 即可（如 register_state_handlers / publish_state），
    无需修改本类 —— StateType 是开放命名空间的预设项，而非封闭枚举。
    """
    # —— project 域（项目，跨插件消费）——
    PROJECT_SELECTED = "project.project_selected"
    PROJECTS_CHANGED = "project.projects_changed"
    # —— annotation 域（手动标注）——
    IMAGE_CHANGED = "annotation.image_changed"
    ANNOTATIONS_CHANGED = "annotation.annotations_changed"
    RULES_CHANGED = "annotation.rules_changed"
    CATEGORIES_CHANGED = "annotation.categories_changed"
    FILTERS_CHANGED = "annotation.filters_changed"
    DATASETS_CHANGED = "annotation.datasets_changed"
    HARD_SAMPLE_CHANGED = "annotation.hard_sample_changed"
    # 纯 UI 状态（面板显隐/显示选项等）
    UI_STATE = "annotation.ui_state"
    # 任务模式（seg/det/obb）变更
    TASK_MODE_CHANGED = "annotation.task_mode_changed"
    # annotation → agent 单向引用
    REFERENCE_ADDED = "annotation.reference_added"


class EventBus:
    """事件总线，支持跨插件通信"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event: str, callback: Callable):
        """
        订阅事件
        
        Args:
            event: 事件名称
            callback: 回调函数，签名为 callback(data=None)
        """
        if event not in self._subscribers:
            self._subscribers[event] = []
        if callback not in self._subscribers[event]:
            self._subscribers[event].append(callback)
    
    def unsubscribe(self, event: str, callback: Callable):
        """
        取消订阅事件
        
        Args:
            event: 事件名称
            callback: 要移除的回调函数
        """
        if event in self._subscribers:
            try:
                self._subscribers[event].remove(callback)
            except ValueError:
                pass
            if not self._subscribers[event]:
                del self._subscribers[event]
    
    def publish(self, event: str, data: Any = None):
        """
        发布事件

        从非主线程发布时，监听器回调自动 marshal 到主线程执行——
        后台线程跑重计算 action（如 one_click 推理）后发出的 UI 刷新
        事件，其监听器里的 Qt 操作仍在主线程进行，避免跨线程操作控件。

        Args:
            event: 事件名称
            data: 事件数据
        """
        if event not in self._subscribers:
            return
        callbacks = self._subscribers[event][:]  # 复制列表避免迭代中修改
        # 延迟导入:避免插件在 QApplication 就绪前导入本模块的启动顺序问题
        try:
            from PySide6.QtCore import QCoreApplication, QThread
            from core.common.main_thread_dispatcher import run_on_main
            app = QCoreApplication.instance()
            off_main = app is not None and QThread.currentThread() is not app.thread()
        except Exception:
            off_main = False
        for callback in callbacks:
            try:
                if off_main:
                    run_on_main(lambda cb=callback: cb(data))
                else:
                    callback(data)
            except Exception as e:
                print(f"[EventBus] Error in callback for event '{event}': {e}")

    def publish_state(self, type_: str, data: Any = None, extra: Dict = None):
        """发布统一状态事件，payload 固定 {type, data, extra}。"""
        self.publish(ROUTER_SIGNAL, {"type": type_, "data": data, "extra": extra or {}})

    def clear(self):
        """清除所有订阅"""
        self._subscribers.clear()
    
    def has_subscribers(self, event: str) -> bool:
        """检查某事件是否有订阅者"""
        return event in self._subscribers and len(self._subscribers[event]) > 0


class StateRouter:
    """全局状态路由器：core 单例，唯一订阅 app:state_changed，按 type 分发给各插件注册的 handler。

    跨插件、可扩展：新插件在 on_load 调 PluginContext.register_state_handlers({...}) 即接入，
    无需接触本类内部。type 空间开放（方案 A），插件可发布/注册自定义 type。
    """
    def __init__(self, bus: EventBus):
        self._bus = bus
        self._handlers: Dict[str, List[Callable]] = {}   # type -> [handler(data), ...]
        bus.subscribe(ROUTER_SIGNAL, self._dispatch)

    def register(self, type_: str, handler: Callable) -> "StateRouter":
        """注册某 type 的处理函数（可多插件各注册，互不覆盖）。"""
        self._handlers.setdefault(type_, []).append(handler)
        return self

    def register_many(self, mapping: Dict[str, Callable]) -> "StateRouter":
        """批量注册 {type: handler}——插件的接入入口。"""
        for t, h in mapping.items():
            self.register(t, h)
        return self

    def unregister(self, type_: str = None, handler: Callable = None):
        """卸载处理器（插件 on_unload 清理，幂等）。"""
        if type_ is None:
            self._handlers.clear()
            return
        if handler is None:
            self._handlers.pop(type_, None)
            return
        lst = self._handlers.get(type_)
        if lst and handler in lst:
            lst.remove(handler)

    def _dispatch(self, payload: Any):
        if not isinstance(payload, dict):
            return
        for handler in self._handlers.get(payload.get("type"), ()):
            try:
                handler(payload.get("data"))
            except Exception as e:
                print(f"[StateRouter] handler error for '{payload.get('type')}': {e}")

    @property
    def handlers(self) -> Dict[str, List[Callable]]:
        """当前注册表（供调试/自省）。"""
        return self._handlers
