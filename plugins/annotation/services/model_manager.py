"""ModelManager — 模型管理器（Phase 2 自 view_models 迁移）

保留 QThread + 信号（model_load_started / model_load_finished），
由 AutoAnnotationService 在 __init__ 中实例化，界面层通过
annotation_service.model_manager 访问。
"""

from PySide6.QtCore import QObject, Signal, QThread


class ModelLoadThread(QThread):
    """线程用于异步加载模型"""
    finished = Signal(bool, str)  # (success, message)

    def __init__(self, model_type, is_interactive=False):
        super().__init__()
        self.model_type = model_type
        self.is_interactive = is_interactive

    def run(self):
        try:
            from core.backend.core import get_example_predictor, get_interactive_predictor
            if self.is_interactive:
                segmentor = get_interactive_predictor()
            else:
                segmentor = get_example_predictor()

            # get_model will trigger load() if not already loaded
            wrapper = segmentor.get_model(self.model_type)
            if wrapper:
                self.finished.emit(True, f"模型 {self.model_type} 加载成功")
            else:
                self.finished.emit(False, f"模型 {self.model_type} 加载失败")
        except Exception as e:
            self.finished.emit(False, str(e))


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

    def load_model_async(self, model_type, is_interactive=False):
        """异步加载模型"""
        if self._model_loading:
            return

        self._model_loading = True
        self.model_load_started.emit(model_type)

        self.load_thread = ModelLoadThread(model_type, is_interactive)
        self.load_thread.finished.connect(self._on_model_load_finished)
        self.load_thread.start()

    def _on_model_load_finished(self, success, message):
        self._model_loading = False
        self.model_load_finished.emit(success, message)

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
