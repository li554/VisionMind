"""D14：`stop_and_wait` 与 `run_on_main` 不得互等 —— 回归。

被测缺陷（审计 D14，两层）：

1. **等待侧无超时**：`run_on_main` 用 `ctx.done.wait()` 无限等待。主线程一旦被
   占住（线上形态：`agent_chat` 这类非 background action 本身就在主线程执行，
   又被 `QThread.wait(max_ms)` 挡住），投递过去的任务永远不会被执行 ->
   worker 永久卡住、主线程冻结到 `max_wait` 超时（默认 600000ms = 10 分钟）。

2. **放弃后仍残留待执行记录**（本轮新发现，比第 1 条更隐蔽）：abort 路径原本只把
   ctx 标记为 abandoned，却**没有把它从 `_queue` 里摘掉**。那条 record 以及对应的
   `_execute_all` 仍排在 Qt 事件队列里，会在下一次有人派发主线程任务时被一起取出执行
   —— 也就是说"调用方已经放弃的任务"照样会占住主线程（若它恰好是阻塞型任务，
   界面就真的卡死）。这直接破坏了"abandoned 任务绝不被执行"的保证。

为什么本测试**不**去真的占用自己的主线程：
  一旦阻塞型任务被派发，`processEvents()` 就不会返回 —— 测试自身也会卡死，无法再做
  任何断言（这一点是实测踩过的坑）。因此这里改为**直接断言调度器的契约**：
    * abort 后必须在 ~250ms 内返回（不是无限等）；
    * abort 后队列必须被摘干净（不得残留任何待执行记录）；
    * 被放弃的任务绝不在主线程上执行；
    * 未 abort 的调用仍能正常在主线程执行并拿到返回值；
    * `stop_and_wait` 在主线程分支里必须**泵事件循环**（否则就是互等），
      在工作线程分支里仍是阻塞等待且能等到 worker 退出。

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_stop_and_wait.py
"""
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from core.common.main_thread_dispatcher import (
        run_on_main, MainThreadCallAborted, _MainThreadDispatcher,
    )

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    def pump(seconds=0.3):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.002)

    def queue_size():
        return _MainThreadDispatcher.instance()._queue.qsize()

    print("== D14：stop_and_wait / run_on_main 互等回归 ==")

    # ---------------- D1: 正常路径仍然可用 ----------------
    seen = []
    got = run_on_main(lambda: seen.append("ran") or 42)
    check(got == 42 and seen == ["ran"],
          "D1 正常 run_on_main 未返回预期结果: %r / %r" % (got, seen))
    print("   D1 正常 run_on_main 返回 %r、执行 1 次  OK" % got)

    # ---------------- D2: abort 必须在有限时间内返回 ----------------
    # 构造"主线程不跑事件循环"的窗口：主线程在这里完全不 processEvents。
    abort = threading.Event()
    outcome = {}

    def waiter():
        t0 = time.perf_counter()
        try:
            run_on_main(lambda: "SHOULD_NOT_HAPPEN", abort_event=abort)
            outcome["result"] = "returned"
        except MainThreadCallAborted:
            outcome["result"] = "aborted"
        except BaseException as exc:
            outcome["result"] = "error:%r" % exc
        outcome["ms"] = (time.perf_counter() - t0) * 1000.0

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    time.sleep(0.15)                 # 让它进入等待；主线程此刻不处理队列
    abort.set()
    th.join(2.0)
    print("   D2 abort 通道：结果=%s 耗时=%.0f ms" % (outcome.get("result"), outcome.get("ms", -1)))
    check(outcome.get("result") == "aborted",
          "D2 abort 没有生效: %r（会无限等待，即互等的等待侧）" % outcome.get("result"))
    check(outcome.get("ms", 9e9) <= 400,
          "D2 abort 生效太慢: %.0fms" % outcome.get("ms", -1))

    # ---------------- D3: abort 后队列必须被摘干净（本轮修的缺陷）----------------
    size_after_abort = queue_size()
    print("   D3 abort 后的待执行记录数=%d（必须为 0）" % size_after_abort)
    check(size_after_abort == 0,
          "D3 abort 后队列仍残留 %d 条待执行记录 —— 它们会在下一次派发时被执行，"
          "占住主线程（D14 第 2 层缺陷）" % size_after_abort)

    # ---------------- D4: 被放弃的任务绝不被执行 ----------------
    marker = []
    abort2 = threading.Event()

    def waiter2():
        try:
            run_on_main(lambda: marker.append("RAN"), abort_event=abort2)
        except MainThreadCallAborted:
            pass

    th2 = threading.Thread(target=waiter2, daemon=True)
    th2.start()
    time.sleep(0.15)
    abort2.set()                     # 等待方放弃
    th2.join(1.0)
    pump(0.5)                        # 主线程恢复派发
    print("   D4 被放弃的任务是否被执行: %s（应为空）" % marker)
    check(not marker, "D4 被放弃的过期任务仍在主线程上执行了")

    # ---------------- D5: 多次放弃不得累积残留 ----------------
    abort3 = threading.Event()
    threads = []
    for _ in range(5):
        t = threading.Thread(
            target=lambda: _safe_call(abort3), daemon=True)
        t.start()
        threads.append(t)
    time.sleep(0.2)
    abort3.set()
    for t in threads:
        t.join(1.0)
    size_multi = queue_size()
    print("   5 次并发放弃后的待执行记录数=%d（必须为 0）" % size_multi)
    check(size_multi == 0, "D5 多次放弃后残留 %d 条记录" % size_multi)

    # ---------------- D6: stop_and_wait 的分支选择 ----------------
    import core.agent.visionmind_agent as vm
    import inspect
    src = inspect.getsource(vm.VisionMindAgent.stop_and_wait)
    check("_wait_pumping_events" in src,
          "D6 stop_and_wait 缺少主线程分支（主线程上仍会 QThread.wait 互等）")
    check("QThread.currentThread()" in src,
          "D6 stop_and_wait 未区分调用线程")
    print("   D6 stop_and_wait 已按调用线程分支（主线程 -> 泵事件循环）  OK")

    # ---------------- D7: 工作线程分支仍能正常等待 worker 退出 ----------------
    class SlowWorker(QThread):
        """worker 只有在**主线程**放行时才退出 —— 与线上"卡在 LLM 网络调用"同构。

        若让 run() 只看停止标志，stop_and_wait 会瞬间返回（worker 自己就结束了），
        测不出"阻塞等待"这件事。所以这里用 release 事件把退出时机交给测试。
        """

        def __init__(self):
            super().__init__()
            self._stop = threading.Event()
            self.release = threading.Event()
            self.exited = threading.Event()
            self.stop_requested = False

        def request_stop(self):
            self.stop_requested = True
            self._stop.set()

        def run(self):
            self._stop.wait(3.0)
            self.release.wait(3.0)      # 真正决定何时退出的是测试
            self.exited.set()

    w = SlowWorker()
    w.start()
    agent = vm.VisionMindAgent()
    agent._worker = w
    result = {}

    def off_main_caller():
        t = time.perf_counter()
        result["ok"] = agent.stop_and_wait(3000)
        result["ms"] = (time.perf_counter() - t) * 1000.0

    th3 = threading.Thread(target=off_main_caller, daemon=True)
    th3.start()
    # 等待期间主线程不跑事件循环：这正是"工作线程分支必须靠 QThread.wait"的场景
    time.sleep(0.5)
    still_waiting = th3.is_alive()
    check(w.stop_requested, "D7 停止请求没有传达给 worker")
    w.release.set()                     # 放行 worker 退出
    th3.join(4.0)
    print("   D7 工作线程 stop_and_wait：等待中=%s 返回=%s 耗时=%.0f ms worker 已退出=%s"
          % (still_waiting, result.get("ok"), result.get("ms", -1), w.exited.is_set()))
    check(still_waiting, "D7 工作线程分支立即返回 —— 没有真正等待 worker")
    check(result.get("ok") is True, "D7 工作线程分支应返回 True（worker 已退出）: %r" % result)
    check(result.get("ms", 0) >= 300, "D7 工作线程分支等待过短: %.0fms" % result.get("ms", -1))
    w.wait(1000)

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: D1~D7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


def _safe_call(abort_event):
    from core.common.main_thread_dispatcher import run_on_main, MainThreadCallAborted
    try:
        run_on_main(lambda: None, abort_event=abort_event)
    except MainThreadCallAborted:
        pass
    except BaseException:
        pass


if __name__ == "__main__":
    sys.exit(main())
