"""模型加载不得再出现 QThread 析构崩溃 + 状态机自洽 —— D12 回归。

被测缺陷（审计 D12，与标注界面原先那个 ImageLoader 崩溃同类）：
  1. `ModelLoadThread(QThread)` 把 `finished = Signal(bool, str)` 定义在自己身上，
     **遮蔽了 `QThread.finished`**；而且是在 `run()` **内部** emit —— 线程还没真正
     结束就宣告完成 -> `_model_loading` 被提前清成 False。
  2. 清成 False 后再调 `load_model_async`，会 `self.load_thread = ModelLoadThread(...)`
     **覆盖仍在运行的 QThread 的引用** -> 引用计数归零 ->
     `QThread: Destroyed while thread is still running` -> `abort()`。

断言：
  M0 静态：model_manager 模块内不得再出现 QThread
  M1 加载期间重复请求必须被拒绝（只有一个后台任务）
  M2 关键场景：强制把 _model_loading 清零后再请求（模拟原缺陷的路径）——
     不得启动第二个线程、不得 abort、状态必须保持自洽
  M3 完成状态由代次跟踪：过期的完成回调不得清掉当前加载的状态
  M4 完成之后可以再次发起加载
  M5 drain() 覆盖模型加载器
  M6 连发 10 次（每次都强制清零标志）仍不得 abort

退出码即断言：0 通过 / 1 断言失败 / -1073740791 = 进程被 abort。
用法: python -u tests/annotation/test_model_load_thread.py
"""
import inspect
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SLOW_S = 0.30


def main():
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    import plugins.annotation.services.model_manager as mm

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    print("== 模型加载线程生命周期回归 ==")

    # ---------------- M0: 静态不变量 ----------------
    # 只用正则剥掉三引号块再剥注释来判定：inspect.getsource 的行尾可能与
    # __doc__（LF）不同，直接用字符串 replace 去 docstring 并不可靠。
    src = inspect.getsource(mm)
    body = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
    if "QThread" in code:
        failures.append("M0 model_manager.py 的代码里仍出现 QThread（应改为后台线程池）"
                        ": %s" % [ln.strip() for ln in code.splitlines() if "QThread" in ln][:3])
    else:
        print("   M0 静态检查：model_manager 代码内 0 处 QThread  OK")
    if hasattr(mm, "ModelLoadThread"):
        failures.append("M0 ModelLoadThread 仍然存在（遮蔽 QThread.finished 的类）")

    # ---------------- 准备：慢加载 + 计数 ----------------
    state = {"calls": 0, "finished": []}
    mgr = mm.ModelManager()

    def slow_load(model_type, is_interactive=False):
        state["calls"] += 1
        time.sleep(_SLOW_S)
        return True, f"模型 {model_type} 加载成功"

    mgr._load_model_blocking = slow_load
    mgr.model_load_finished.connect(lambda ok, msg: state["finished"].append((ok, msg)))

    def pump(seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    # ---------------- M1: 加载期间重复请求被拒 ----------------
    mgr.load_model_async("m1")
    busy_after_first = mgr.is_model_loading()
    mgr.load_model_async("m1-dup")                    # 应被 _model_loading 拦掉
    # 工作线程是异步启动的，先给它一点时间真正跑起来再计数
    pump(0.1)
    refused_calls = state["calls"]
    if not check(busy_after_first, "M1 首次加载后 is_model_loading 应为 True"):
        pass
    if refused_calls != 1:
        failures.append("M1 加载期间的重复请求启动了第二个任务（calls=%d）" % refused_calls)
    else:
        print("   M1 加载期间重复请求被拒（只启动 1 个后台任务）  OK")
    pump(2.0, until=lambda: not mgr.is_model_loading())
    if mgr.is_model_loading():
        failures.append("M1 加载未在 2s 内完成")

    # ---------------- M2/M6: 强制清零标志后连发（模拟原缺陷路径）----------------
    state["calls"] = 0
    state["finished"].clear()
    mgr.load_model_async("m2")
    for i in range(10):                                # 每次都绕过守卫
        mgr._model_loading = False
        mgr.load_model_async("m2-dup-%d" % i)
    pump(0.1)                                          # 等工作线程真正起来
    calls_during = state["calls"]
    flag_after = mgr.is_model_loading()
    print("   M2 强制清零标志后连发 11 次：实际启动任务=%d  标志=%s" % (calls_during, flag_after))
    if calls_during != 1:
        failures.append("M2 启动了 %d 个并发加载任务（runner 应拒绝并发）" % calls_during)
    if not flag_after:
        failures.append("M2 被拒的请求把 is_model_loading 谎报成 False（守卫会失效）")
    pump(2.0, until=lambda: not mgr.is_model_loading())
    settled_calls = state["calls"]
    finish_count = len(state["finished"])
    print("   M6 连发结束：总任务=%d  完成信号=%d  标志=%s"
          % (settled_calls, finish_count, mgr.is_model_loading()))
    if settled_calls != 1:
        failures.append("M6 连发期间共启动了 %d 个任务" % settled_calls)
    if finish_count != 1:
        failures.append("M6 完成信号发了 %d 次（被拒的请求不应各发一次）" % finish_count)
    if mgr.is_model_loading():
        failures.append("M6 结束后标志仍为 True")

    # ---------------- M3: 过期完成回调不得清掉当前加载状态 ----------------
    state["calls"] = 0
    state["finished"].clear()
    mgr.load_model_async("m3")
    stale_gen = mgr._load_generation - 1
    mgr._on_load_done(stale_gen, (True, "stale"))
    kept = mgr.is_model_loading()
    stale_emitted = len(state["finished"])
    print("   M3 过期完成回调：标志保持=True？%s  完成信号数=%d" % (kept, stale_emitted))
    if not kept:
        failures.append("M3 过期完成回调清掉了当前加载的状态（原缺陷的「误判」）")
    if stale_emitted != 0:
        failures.append("M3 过期完成回调仍发出了 model_load_finished")
    pump(2.0, until=lambda: not mgr.is_model_loading())

    # ---------------- M4: 完成后可再次发起 ----------------
    state["calls"] = 0
    state["finished"].clear()
    mgr.load_model_async("m4")
    ok = pump(2.0, until=lambda: not mgr.is_model_loading())
    if not ok or state["calls"] != 1 or not state["finished"]:
        failures.append("M4 完成后无法再次发起加载（calls=%d finished=%s）"
                        % (state["calls"], state["finished"]))
    else:
        print("   M4 完成后可再次发起加载  OK")

    # ---------------- M5: drain 覆盖加载器 ----------------
    mgr.load_model_async("m5")
    drained = mgr.drain(5000)
    busy_after_drain = mgr._loader.busy()
    print("   M5 drain(5000)=%s  加载器仍忙碌=%s" % (drained, busy_after_drain))
    if not drained:
        failures.append("M5 drain(5000) 未能在超时内排空")
    if busy_after_drain:
        failures.append("M5 drain 后加载器仍在跑")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: M0~M6 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
