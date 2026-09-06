"""
主线程调度器 — 将任意可调用对象分发到 Qt 主线程执行

使用 QMetaObject.invokeMethod + QueuedConnection + queue.Queue，
支持**多线程并发调用**（ThreadPoolExecutor 中的多个工具可同时
等待主线程执行）。

不再使用 Q_ARG(object, func) 传递 callable，避免 QMetaType 错误。
历史缺陷修复: 等待侧原用 QEventLoop,主线程可能在 worker 进入
loop.exec() 之前就 quit()(对未启动的循环 quit 是 no-op),worker
随后永久阻塞——并行工具越多竞态窗口越大。已改用 threading.Event。
"""
from PySide6.QtCore import QObject, Slot, QMetaObject, Qt, QThread, QCoreApplication
from typing import Any, Callable
import queue
import threading


class _CallContext:
    """封装一次 run_on_main 调用的上下文"""
    def __init__(self, func: Callable):
        self.func = func
        self.result: Any = None
        self.error: Exception | None = None
        self.done = threading.Event()


class _MainThreadDispatcher(QObject):
    """在 Qt 主线程执行 callable 的单例调度器（线程安全）

    使用 queue.Queue 实现任务队列，支持多线程并发等待。
    """

    _instance = None
    _init_lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self._queue: queue.Queue[_CallContext] = queue.Queue()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    ins = cls()
                    app = QCoreApplication.instance()
                    if app and QThread.currentThread() != app.thread():
                        ins.moveToThread(app.thread())
                    cls._instance = ins
        return cls._instance

    @Slot()
    def _execute_all(self):
        """主线程：处理队列中所有待执行的任务"""
        while not self._queue.empty():
            ctx = self._queue.get_nowait()
            try:
                ctx.result = ctx.func()
            except Exception as e:
                ctx.error = e
            finally:
                ctx.done.set()

    def call(self, func: Callable) -> Any:
        """在 Qt 主线程同步执行 func 并返回结果（线程安全）"""
        app = QCoreApplication.instance()
        if not app:
            return func()

        if QThread.currentThread() == app.thread():
            return func()

        ctx = _CallContext(func)

        self._queue.put(ctx)

        QMetaObject.invokeMethod(
            self, "_execute_all",
            Qt.ConnectionType.QueuedConnection
        )

        # Event 无 quit-before-exec 竞态:set 先于 wait 也立即返回
        ctx.done.wait()

        if ctx.error is not None:
            raise ctx.error
        return ctx.result


def run_on_main(func: Callable) -> Any:
    """快捷方式：在 Qt 主线程执行 func 并返回结果"""
    return _MainThreadDispatcher.instance().call(func)
