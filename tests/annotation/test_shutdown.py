"""可确定性关闭回归 — 关闭协调器 + 在途后台工作排空。

被测目标（objective 里的"可确定性关闭"）：
  * 关闭时必须在 **Qt 对象树析构之前** 显式排空后台工作；
  * 排空期间不得死锁（工作线程此时若通过 `run_on_main` 投递并等待会永久挂起，
    因为主线程正阻塞在排空里、并不在跑事件循环）；
  * `shutdown()` 必须幂等（`closeEvent` 与 `aboutToQuit` 都会调用）。

退出码即断言（同 test_switch_stress.py）：
  0 = 通过 / 1 = 断言失败 / -1073740791 = 进程被 abort / 其他 = 异常退出

用法: python -u tests/annotation/test_shutdown.py [switch_ms] [switch_rounds]
"""
import os
import shutil
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import build_fixture, isolate_state  # noqa: E402


class _MarshalProbe:
    """模拟"排空期间工作线程向主线程投递任务"。

    这是 `stop_and_wait` ↔ `run_on_main` 互等死锁（审计 D14）的最小化复现形态：
    主线程阻塞在 drain() 里，工作线程却在等主线程执行它投递的任务。
    正确行为是**快速失败**（RuntimeError），而不是挂住关闭流程。
    """

    def __init__(self):
        self.error_type = None
        self.hung = False
        self.executed = False

    def drain(self, timeout_ms):
        from core.common.main_thread_dispatcher import run_on_main

        box = {}

        def worker():
            try:
                run_on_main(lambda: "ok")
                box["executed"] = True
            except Exception as exc:  # 期望走到这里
                box["error"] = type(exc).__name__

        t = threading.Thread(target=worker, name="marshal-probe", daemon=True)
        t.start()
        t.join(timeout=2.0)
        self.hung = t.is_alive()
        self.error_type = box.get("error")
        self.executed = bool(box.get("executed"))
        return not self.hung


def main():
    switch_ms = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    switch_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from core.lifecycle import ShutdownCoordinator
    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    ShutdownCoordinator.reset_for_tests()

    w = AnnotationInterface(None, None)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.open_directory(build_fixture(state_dir))
    app.processEvents()

    dec = w.draw_area._decoder
    thumbs = w._thumb_manager
    print("== 关闭回归 ==")
    print("   已登记的 drainable: %s" % ShutdownCoordinator.instance().registered)

    st = {"i": 0, "dec_peak": 0, "thumb_peak": 0, "thumb_pending_peak": 0}
    failures = []

    def tick():
        st["i"] += 1
        if st["i"] > switch_rounds:
            timer.stop()
            QTimer.singleShot(0, phase_shutdown)
            return
        # 持续切图，制造在途解码
        w.switch_image(1 if st["i"] % 2 else -1)
        st["dec_peak"] = max(st["dec_peak"], dec.active_count())
        st["thumb_peak"] = max(st["thumb_peak"], thumbs.active_count())
        st["thumb_pending_peak"] = max(st["thumb_pending_peak"], thumbs.pending_count())

    def phase_shutdown():
        print("   切图 %d 次：解码并发峰值=%d  缩略图并发峰值=%d  缩略图在途峰值=%d"
              % (switch_rounds, st["dec_peak"], st["thumb_peak"], st["thumb_pending_peak"]))

        # 断言 A：关闭前确实有在途工作（否则本测试没有测到东西）
        if st["dec_peak"] == 0 and st["thumb_peak"] == 0:
            failures.append("A 未能在关闭前制造出任何在途后台工作，测试无效")

        # 断言 B：排空期间工作线程向主线程投递 -> 必须快速失败而不是挂住
        probe = _MarshalProbe()
        coordinator = ShutdownCoordinator.instance()
        coordinator.register("test.marshal_probe", probe)

        t0 = time.perf_counter()
        results = coordinator.shutdown(5000)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print("   排空耗时 %.0f ms -> %s" % (elapsed_ms, coordinator.summary()))

        if probe.hung:
            failures.append("B 排空期间工作线程投递主线程任务后**挂住**（互等死锁未解决）")
        elif probe.error_type != "RuntimeError":
            failures.append("B 期望快速失败为 RuntimeError，实际 %r（executed=%s）"
                            % (probe.error_type, probe.executed))

        # 断言 C：排空后线程池必须真的空了
        dec_left = dec.active_count()
        thumb_left = thumbs.active_count()
        if dec_left != 0:
            failures.append("C 排空后仍有 %d 个解码在跑" % dec_left)
        if thumb_left != 0:
            failures.append("C 排空后仍有 %d 个缩略图解码在跑" % thumb_left)

        # 断言 D：所有已登记项都报告成功
        bad = [name for name, ok, _ms in results if not ok]
        if bad:
            failures.append("D 以下项未排空成功: %s" % bad)

        # 断言 E：幂等（closeEvent + aboutToQuit 会各调一次）
        again = coordinator.shutdown(5000)
        if again != results:
            failures.append("E shutdown() 非幂等：第二次返回不同结果")

        # 断言 F：关闭闸门必须**保持关闭** —— 退出期不应再有任何后台工作；
        # 若此时还有工作线程向主线程投递任务（exit 钩子里很常见），闸门关闭
        # 才能保证它快速失败而不是把解释器收尾挂住。
        from core.common.main_thread_dispatcher import is_shutting_down
        if not is_shutting_down():
            failures.append("F 关闭后调度器闸门被复位：退出期的工作线程投递可能再次挂住")
        if not coordinator.is_shutting_down:
            failures.append("F 协调器未保持 shutting_down 状态")
        if failures:
            print("   FAIL:")
            for f in failures:
                print("     - %s" % f)
            st["code"] = 1
        else:
            print("   PASS: A/B/C/D/E 全部通过")
            st["code"] = 0

        # 收尾：关窗 -> 事件循环退出。退出码 0 且无 Qt Fatal 才算真正通过。
        QTimer.singleShot(100, w.close)
        QTimer.singleShot(300, app.quit)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(switch_ms)

    st["code"] = 1
    QTimer.singleShot(switch_rounds * switch_ms + 20000, app.quit)
    app.exec()

    code = st.get("code", 1)
    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
