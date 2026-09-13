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


class MainThreadCallAborted(Exception):
    """投递到主线程的调用被 abort_event 取消（见 `call` 的说明）。"""


class _CallContext:
    """封装一次 run_on_main 调用的上下文"""
    def __init__(self, func: Callable):
        self.func = func
        self.result: Any = None
        self.error: Exception | None = None
        self.done = threading.Event()
        self.abandoned = False      # 等待方已放弃：派发时必须跳过（D14）


# 关闭闸门：ShutdownCoordinator.shutdown() 期间，主线程正阻塞在排空流程里、并不在跑
# 事件循环。此时工作线程若通过 run_on_main 投递任务并等待，队列里的任务永远不会被
# 执行，而下面的 ctx.done.wait() 又没有超时 -> 永久挂起，解释器收尾卡死。
# 因此关闭期间让这类调用**快速失败**，而不是死等。
_shutting_down = False
_shutdown_refused = 0


def set_shutting_down(value: bool) -> None:
    """由 ShutdownCoordinator 在排空前后调用。"""
    global _shutting_down
    _shutting_down = bool(value)


def is_shutting_down() -> bool:
    return _shutting_down


def refused_count() -> int:
    """关闭期间被拒绝的投递次数（诊断/测试用）。"""
    return _shutdown_refused


def reset_refused_count() -> None:
    global _shutdown_refused
    _shutdown_refused = 0


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
        while True:
            try:
                ctx = self._queue.get_nowait()
            except queue.Empty:
                return
            if ctx.abandoned:
                # 等待方已经放弃（停止请求打断了互等）——绝不执行过期任务
                ctx.done.set()
                continue
            try:
                ctx.result = ctx.func()
            except Exception as e:
                ctx.error = e
            finally:
                ctx.done.set()

    def _discard_abandoned(self, ctx) -> None:
        """把已被放弃的调用从待执行队列里摘掉（审计 D14 的关键一环）。

        只是把 ctx 标记成 abandoned 还不够：那条 record 仍留在 `_queue` 里、
        对应的 `_execute_all` 也仍排在 Qt 事件队列里。它们会在**下一次**有人
        派发主线程任务时被一起取出执行 —— 而此时调用方早已放弃等待，于是
        "本该被丢弃的任务"照样占住主线程。若它恰好是阻塞型任务（例如
        `agent_chat` 触发的那类长任务），主线程就会一直卡在这里，界面彻底冻结。

        因此放弃等待时立刻摘除该 record；`_execute_all` 里对 abandoned 的
        判断保留为二重保险（可能已被取出、正要执行）。
        """
        kept = []
        while True:
            try:
                other = self._queue.get_nowait()
            except queue.Empty:
                break
            if other is ctx or other.abandoned:
                other.done.set()
                continue
            kept.append(other)
        for other in kept:
            self._queue.put(other)
        ctx.done.set()

    def call(self, func: Callable, abort_event=None, poll_ms: int = 50) -> Any:
        """在 Qt 主线程同步执行 func 并返回结果（线程安全）。

        `abort_event`：可选的 `threading.Event`。等待期间被 set 时立即抛出
        `MainThreadCallAborted`，而不再无限期等下去。

        为什么需要它（审计 D14）：主线程可能正阻塞在 `QThread.wait()` 上（例如
        `VisionMindAgent.stop_and_wait`），此时它**不跑事件循环**，投递过去的任务
        永远不会被执行。原先这里是无超时的 `ctx.done.wait()`，于是 worker 永久卡住、
        主线程冻结到 `wait` 超时（默认 10 分钟）。有了 abort 通道，停止请求可以立刻
        打断这个等待。
        """
        global _shutdown_refused

        app = QCoreApplication.instance()
        if not app:
            return func()

        if QThread.currentThread() == app.thread():
            return func()

        if _shutting_down:
            # 主线程此刻在排空里阻塞，投递出去也没人执行；快速失败优于死等。
            _shutdown_refused += 1
            if _shutdown_refused == 1:
                print("[MainThreadDispatcher] 关闭进行中：拒绝在主线程执行任务"
                      "（若继续等待会永久挂起关闭流程）")
            raise RuntimeError("application is shutting down: run_on_main refused")

        ctx = _CallContext(func)

        self._queue.put(ctx)

        QMetaObject.invokeMethod(
            self, "_execute_all",
            Qt.ConnectionType.QueuedConnection
        )

        if abort_event is None:
            # Event 无 quit-before-exec 竞态:set 先于 wait 也立即返回
            ctx.done.wait()
        else:
            step = max(1, int(poll_ms)) / 1000.0
            while True:
                if ctx.done.wait(step):
                    break
                if abort_event.is_set():
                    # 任务可能仍在队列里（也可能正在主线程执行）——标记作废，
                    # 并且**必须把它从队列里摘掉**，否则残留 record 会在下一次
                    # 派发时被执行，占住主线程（详见 _discard_abandoned 的说明）。
                    ctx.abandoned = True
                    self._discard_abandoned(ctx)
                    raise MainThreadCallAborted(
                        "main-thread call aborted by stop request")

        if ctx.error is not None:
            raise ctx.error
        return ctx.result


def run_on_main(func: Callable, abort_event=None, poll_ms: int = 50) -> Any:
    """快捷方式：在 Qt 主线程执行 func 并返回结果。

    `abort_event` 见 `_MainThreadDispatcher.call`（用于避免与主线程 `QThread.wait`
    互等，审计 D14）。
    """
    return _MainThreadDispatcher.instance().call(func, abort_event=abort_event,
                                                 poll_ms=poll_ms)
