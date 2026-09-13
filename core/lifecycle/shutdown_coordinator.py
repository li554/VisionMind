"""ShutdownCoordinator — 在 Qt 对象树析构之前，按 LIFO 顺序排空后台工作。

为什么必须有它：本项目的关闭链路原本是
    MainWindow.closeEvent -> sys.exit(app.exec())
而 `closeEvent` 只做了一件空转的事（保存 AI 会话 + 调一个 `pass` 的 sidebar.shutdown），
全仓没有 `aboutToQuit`、没有 `atexit` 排空。于是任何仍在运行的 QThread / QThreadPool
只能靠 C++ 析构期的隐式等待 —— 那发生在解释器收尾阶段，既没有超时，也没有作废代次，
正是 `logs/crash_log_*.txt` 里 `QThread: Destroyed while thread '' is still running`
（56/68 份）的温床 —— 而且这些 fatal **全部**出现在 `Application exiting normally`
之后，即退出期（已用脚本统计确认）。

契约（`Drainable` Protocol）::

    def drain(self, timeout_ms: int) -> bool:
        '''作废在途工作并等待其结束；返回是否在超时内排空。只能在 GUI 线程调用。'''

实现要点：

1. **总预算**而不是每项各一份超时：`shutdown(total_ms)` 内部按剩余时间分配给每一项，
   避免 N 个 drainable 各等 5s 把关闭拖成 N×5s。
2. **LIFO**：后注册的先排空（依赖关系上更安全——依赖方先于被依赖方收尾）。
3. **幂等**：`closeEvent` 与 `aboutToQuit` 都会调用，第二次直接返回首次的结果。
4. **排空期间禁止 `run_on_main`**：主线程此刻正阻塞在 `shutdown()` 里、并不在跑事件循环，
   若工作线程此时通过 `run_on_main` 投递任务并等待，就会永久挂起
   （`core/common/main_thread_dispatcher.py` 的 `ctx.done.wait()` 没有超时）。
   因此 `shutdown()` 一进入就打开一个全局闸门，让这类调用**快速失败**而不是死等。
"""
import time
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QObject

try:  # typing.Protocol 只在 3.8+ 可用；本项目是 3.12，但保持导入健壮
    from typing import Protocol

    class Drainable(Protocol):
        """可被关闭协调器排空的对象。"""

        def drain(self, timeout_ms: int) -> bool:  # pragma: no cover - 协议声明
            ...

except ImportError:  # pragma: no cover
    Drainable = object  # type: ignore


class ShutdownCoordinator(QObject):
    """进程级关闭协调器（单例）。"""

    _instance: Optional["ShutdownCoordinator"] = None

    def __init__(self):
        super().__init__()
        self._items: List[Tuple[str, object, int]] = []  # (name, obj, order)
        self._order = 0
        self._shutting_down = False
        self._results: Optional[List[Tuple[str, bool, float]]] = None

    # ---- 单例 ----

    @classmethod
    def instance(cls) -> "ShutdownCoordinator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls):
        """仅供测试：丢弃单例，避免测试之间互相污染。"""
        cls._instance = None

    # ---- 登记 ----

    def register(self, name: str, obj) -> None:
        """登记一个 drainable（实现 `drain(timeout_ms) -> bool`）。

        重复登记同名项会覆盖旧的（便于界面重建/插件重载）。
        """
        if obj is None:
            return
        if not callable(getattr(obj, "drain", None)):
            raise TypeError(f"{name} 未实现 drain(timeout_ms) 契约，无法登记到关闭协调器")
        self.unregister(name)
        self._order += 1
        self._items.append((name, obj, self._order))
        if self._shutting_down:
            # 关闭过程中新登记的东西也要被排空，但不能打断当前循环；
            # 记一条日志让这种"关闭后又创建后台工作"的行为可见。
            print(f"[ShutdownCoordinator] 关闭进行中仍登记了 {name}（将在本轮排空）")

    def unregister(self, name: str = None, obj=None) -> None:
        if name is None and obj is None:
            self._items.clear()
            return
        self._items = [
            it for it in self._items
            if not ((name is not None and it[0] == name)
                    or (obj is not None and it[1] is obj))
        ]

    @property
    def registered(self) -> List[str]:
        return [it[0] for it in self._items]

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    # ---- 排空 ----

    def shutdown(self, total_ms: int = 5000) -> List[Tuple[str, bool, float]]:
        """按 LIFO 排空全部已登记的 drainable。返回 [(name, ok, elapsed_ms)]。

        幂等：重复调用返回首次结果（`closeEvent` 与 `aboutToQuit` 都会调）。
        只能在 GUI 线程调用。
        """
        if self._results is not None:
            return self._results

        self._shutting_down = True
        _set_dispatcher_shutdown(True)

        results: List[Tuple[str, bool, float]] = []
        deadline = time.perf_counter() + max(0, int(total_ms)) / 1000.0

        try:
            for name, obj, _order in reversed(self._items):
                remaining_ms = int((deadline - time.perf_counter()) * 1000.0)
                if remaining_ms <= 0:
                    results.append((name, False, 0.0))
                    print(f"[ShutdownCoordinator] 总预算耗尽，{name} 未能排空")
                    continue
                started = time.perf_counter()
                try:
                    ok = bool(obj.drain(remaining_ms))
                except Exception as exc:  # 排空失败不能让关闭流程再抛异常
                    import traceback
                    print(f"[ShutdownCoordinator] {name}.drain 抛异常: {exc}")
                    traceback.print_exc()
                    ok = False
                elapsed = (time.perf_counter() - started) * 1000.0
                results.append((name, ok, elapsed))
                if not ok:
                    print(f"[ShutdownCoordinator] {name} 未在 {remaining_ms}ms 内排空"
                          f"（已用 {elapsed:.0f}ms）")
        finally:
            # 注意：`_shutting_down` 与 dispatcher 闸门都**保持开启**。关闭流程结束后
            # 不应该再有任何后台工作；保持闸门关闭可以让"退出期还去投递主线程任务"
            # 这类错误快速失败，而不是挂住解释器收尾。登记新 drainable 会打日志，
            # 让这类"关闭后又创建后台工作"的行为可见。
            self._results = results
        return results

    def summary(self) -> str:
        if not self._results:
            return "未执行关闭"
        parts = []
        for name, ok, ms in self._results:
            parts.append(f"{name}={'ok' if ok else 'TIMEOUT'}({ms:.0f}ms)")
        return ", ".join(parts)


def _set_dispatcher_shutdown(value: bool) -> None:
    """打开/关闭 `run_on_main` 的快速失败闸门（延迟导入，避免循环依赖）。"""
    try:
        from core.common import main_thread_dispatcher as mtd
        mtd.set_shutting_down(value)
    except Exception as exc:  # 调度器不可用时不应影响关闭
        print(f"[ShutdownCoordinator] 无法设置调度器闸门: {exc}")


def register_drainable(name: str, obj) -> None:
    """便捷函数：登记到进程级协调器。"""
    ShutdownCoordinator.instance().register(name, obj)


def unregister_drainable(name: str = None, obj=None) -> None:
    ShutdownCoordinator.instance().unregister(name, obj)


def shutdown_all(total_ms: int = 5000) -> List[Tuple[str, bool, float]]:
    return ShutdownCoordinator.instance().shutdown(total_ms)


__all__ = [
    "Drainable",
    "ShutdownCoordinator",
    "register_drainable",
    "unregister_drainable",
    "shutdown_all",
]
