"""ModelManager — 模型管理器（Phase 2 自 view_models 迁移）

模型加载在**后台线程池**上执行，对外保留
`model_load_started` / `model_load_finished` 两个信号。
由 AutoAnnotationService 在 __init__ 中实例化，界面层通过
annotation_service.model_manager 访问。

为什么不再用「继承 QThread + `self.load_thread = ...`」（审计 D12）：

1. 那个类把 `finished = Signal(bool, str)` 定义在自己身上，**遮蔽了 QThread.finished**；
   而且是在 `run()` **内部** emit —— 也就是"线程还没真正结束就宣告完成"，于是
   `_model_loading` 被提前清成 False。
2. 清成 False 之后若再调一次 `load_model_async`，就会执行
   `self.load_thread = ModelLoadThread(...)`，**覆盖掉仍在运行的那个 QThread 的引用**
   -> 引用计数归零 -> `QThread: Destroyed while thread is still running` -> `abort()`。
   这与标注界面原先那个 `ImageLoader` 崩溃是同一类缺陷。

现在线程由 `QThreadPool` 持有，调用方**无法**丢掉线程引用；完成状态用代次跟踪，
过期的加载结果不会清掉当前这次加载的状态。
"""

from PySide6.QtCore import QObject, Signal

from core.common.background_task import BackgroundTaskRunner


class ModelManager(QObject):
    """
    模型管理器：负责标注模型的异步加载与可用模型列表查询。

    职责范围：
    - load_model_async / is_model_loading：后台线程异步加载模型，避免阻塞 UI
    - get_interactive_models / get_auto_models：向界面层提供可用的模型列表
    """

    # 模型加载相关信号
    model_load_started = Signal(str)
    model_load_finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model_loading = False
        # 单任务后台运行器：拒绝并发启动，因此不可能出现"两个加载线程同时存在"。
        self._loader = BackgroundTaskRunner(parent=self, max_threads=1,
                                            name="model.load")
        # 加载代次：过期的完成回调不得清掉当前这次加载的状态
        self._load_generation = 0

    def load_model_async(self, model_type, is_interactive=False):
        """异步加载模型（在后台线程池上执行，调用方不持有线程对象）。"""
        if self._model_loading:
            return

        self._model_loading = True
        self.model_load_started.emit(model_type)

        # 注意：代次只能在**启动成功之后**才提交。若在 start() 之前就自增，被拒绝的
        # 重复请求会把代次推进，导致真正在途的那个任务的完成回调被判为"过期"而丢弃
        # —— 那样 _model_loading 会永远停在 True（本模块的回归测试正是这么抓到的）。
        generation = self._load_generation + 1
        started = self._loader.start(
            lambda: self._load_model_blocking(model_type, is_interactive),
            on_result=lambda payload: self._on_load_done(generation, payload),
            on_error=lambda message: self._on_load_done(generation, (False, message)),
        )
        if not started:
            # 已经有一次加载在跑（_loader 拒绝并发启动）。此时必须**保持** _model_loading
            # 为 True —— 不能谎报"未在加载"，否则 is_model_loading() 的守卫会失效。
            print("[ModelManager] 已有模型加载在进行中，忽略本次请求")
            return
        self._load_generation = generation

    @staticmethod
    def _load_model_blocking(model_type, is_interactive):
        """在工作线程里执行的实际加载。返回 (success, message)。"""
        from core.backend.core import get_example_predictor, get_interactive_predictor
        if is_interactive:
            segmentor = get_interactive_predictor()
        else:
            segmentor = get_example_predictor()

        # get_model will trigger load() if not already loaded
        wrapper = segmentor.get_model(model_type)
        if wrapper:
            return True, f"模型 {model_type} 加载成功"
        return False, f"模型 {model_type} 加载失败"

    def _on_load_done(self, generation, payload):
        """加载完成（GUI 线程）。过期结果不得改动当前状态。"""
        success, message = payload if isinstance(payload, tuple) else (False, str(payload))
        if generation != self._load_generation:
            print(f"[ModelManager] 忽略过期的模型加载结果（代次 {generation} "
                  f"!= {self._load_generation}）")
            return
        self._model_loading = False
        self.model_load_finished.emit(success, message)

    def drain(self, timeout_ms=5000):
        """Drainable 契约：等待在途模型加载结束（关闭路径调用）。

        返回是否在超时内排空。加载大模型可能超过预算，此时如实返回 False 并让
        关闭流程继续，而不是像原先那样把仍在运行的 QThread 丢掉后 abort。
        """
        return self._loader.drain(timeout_ms)

    def is_model_loading(self):
        return self._model_loading

    def get_interactive_models(self):
        """获取可用的交互式模型列表"""
        from core.backend.core import ModelFactory
        return ModelFactory.get_interactive_models()

    def get_auto_models(self):
        """获取可用的自动标注模型列表"""
        from core.backend.core import ModelFactory
        return ModelFactory.get_auto_models()

    def get_plain_models(self):
        """获取可用的普通模型列表"""
        from core.backend.core import ModelFactory
        return ModelFactory.get_plain_models()

    def get_plain_categories(self):
        """获取普通模型可用类别"""
        from core.backend.core import ModelFactory
        return ModelFactory.get_plain_categories()
