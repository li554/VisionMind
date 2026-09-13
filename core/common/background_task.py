"""BackgroundTaskRunner — 把"原本在 GUI 线程同步跑 + processEvents 重绘"的长任务
搬到 QThreadPool，结果以**队列化信号**回到 GUI 线程。

为什么需要它：`annotation_interface` 里原本有 7 处 `QApplication.processEvents()`，
根源都是"长任务在 GUI 线程同步执行"——不主动让出事件循环，进度条就不重绘、取消
按钮就点不动。而在状态变更处理器内部重入事件循环，会让用户在批量标注进行中继续
触发切图/保存/删除，导致状态错乱（审计列为 P1）。把任务移出 GUI 线程后，
事件循环本来就空闲，既不需要 processEvents，取消按钮也自然可用。

设计约束（与 `ImageDecodeService` 同一套教训）：

* **不继承 `QThread`** —— 线程由 `QThreadPool` 持有与回收，调用方不可能"丢掉线程引用"
  而触发 `QThread: Destroyed while thread is still running`（实测可同步 abort）。
* **不遮蔽 Qt 内建信号名**（对外只有 `finished`/`failed`，且它们不是 QThread 的信号）。
* 单任务语义：`start()` 在已有任务在跑时返回 False，避免同一份状态被两次批量并发写。
* 实现 `drain(timeout_ms)`（Drainable 契约），宿主关闭时由 ShutdownCoordinator 排空。
* `_inflight` 只在 GUI 线程的槽里归零，因此"任务刚结束但信号还没派发"的窗口内
  `start()` 仍会拒绝 —— 不会出现两次任务同时操作界面状态。
"""
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal


class _BackgroundSignals(QObject):
    """任务结果信号的载体（独立对象，避免遮蔽 QRunnable/QThread 的任何名字）。"""

    finished = Signal(object)   # 任务返回值
    failed = Signal(str)        # 异常摘要


class _BackgroundTask(QRunnable):
    """跑一次函数调用；成功/失败都**必定**发信号，不会让调用方永远等不到结果。"""

    def __init__(self, fn, signals):
        super().__init__()
        self.setAutoDelete(True)
        self._fn = fn
        self._signals = signals

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:
            print(f"[BackgroundTask] 任务执行失败: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self._signals.finished.emit(result)


class BackgroundTaskRunner(QObject):
    """一次性后台任务运行器（单任务语义）。"""

    def __init__(self, parent=None, max_threads=1, name="background"):
        super().__init__(parent)
        self._name = name
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_threads)))
        self._signals = _BackgroundSignals()
        self._inflight = False
        self._on_result = None
        self._on_error = None
        # 显式 QueuedConnection：结果必须在 GUI 线程交付（调用方会碰控件）
        self._signals.finished.connect(self._deliver_result, Qt.ConnectionType.QueuedConnection)
        self._signals.failed.connect(self._deliver_error, Qt.ConnectionType.QueuedConnection)

    # ---- 查询 ----

    def busy(self) -> bool:
        """是否有任务在跑（含"已跑完但结果尚未派发"的窗口）。"""
        return self._inflight

    def active_count(self) -> int:
        return self._pool.activeThreadCount()

    @property
    def name(self) -> str:
        return self._name

    # ---- 启动 ----

    def start(self, fn, on_result=None, on_error=None) -> bool:
        """启动任务。已有任务在跑时返回 False（不排队、不并发）。"""
        if self._inflight:
            return False
        self._inflight = True
        self._on_result = on_result
        self._on_error = on_error
        self._pool.start(_BackgroundTask(fn, self._signals))
        return True

    # ---- 结果交付（GUI 线程）----

    def _deliver_result(self, result):
        self._inflight = False
        callback, self._on_result, self._on_error = self._on_result, None, None
        if callback is not None:
            try:
                callback(result)
            except Exception:
                traceback.print_exc()

    def _deliver_error(self, message):
        self._inflight = False
        callback, self._on_result, self._on_error = self._on_error, None, None
        if callback is not None:
            try:
                callback(message)
            except Exception:
                traceback.print_exc()

    # ---- 生命周期 ----

    def wait_idle(self, timeout_ms=5000) -> bool:
        """等待在途任务结束（不取消）。"""
        return bool(self._pool.waitForDone(int(timeout_ms)))

    def drain(self, timeout_ms=5000) -> bool:
        """Drainable 契约：等待在途任务结束，并**弃置**尚未派发的结果。

        排空意味着"不再接受任何在途结果"，所以这里除了等线程池结束，还要清掉
        在途标记与回调：否则
          * `busy()` 会一直为 True（`_inflight` 只在队列化的交付槽里归零，
            而 drain 是同步等待、不跑事件循环）；
          * 已经排在事件队列里的结果仍会在排空后派发，把过期状态写进正在被
            拆掉的界面。
        调用方应**先请求业务层取消**（例如批量标注的 cancel_batch），否则可能等满超时。
        """
        ok = self.wait_idle(timeout_ms)
        self._inflight = False
        self._on_result = None
        self._on_error = None
        return ok


__all__ = ["BackgroundTaskRunner"]
