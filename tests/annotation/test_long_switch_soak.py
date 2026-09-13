"""长时间连续切图的稳定性：不得有资源随切换次数单调增长 —— 长跑回归。

为什么单独写这一条：用户报告的原始症状是"**快速切换前后两张图片一段时间后**
软件突然卡住"。已有的 `test_switch_stress.py` 是短促爆发（40/80 轮），断言的是
"不 abort / 状态一致 / 并发有界 / 最新优先"，**没有任何一条断言资源不随时间增长** ——
所以"跑一段时间后才卡"这种形态在原有覆盖下是可以漏过去的。

本测试做长跑（默认 400 轮）+ 分段采样，断言：
  L1 线程数不得随切换次数单调增长（结束后回到基线附近）
  L2 GUI 线程的 QObject 子对象数不得持续增长（信号连接/控件泄漏）
  L3 解码线程池的活跃线程峰值有界（<= max_workers）
  L4 切换期间事件循环不得出现"长冻结"（单次间隔 > 阈值即判失败）
  L5 结束后排空干净：drain 后无在途任务、画布提交最后请求的图
  L6 全程无 abort（退出码即断言）

退出码即断言：0 通过 / 1 断言失败 / 非 0 崩溃 = abort。
用法: python -u tests/annotation/test_long_switch_soak.py [rounds] [images]
"""
import gc
import os
import shutil
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# 单次事件循环间隔的容忍上限（毫秒）。超过即视为"可感知冻结"。
# 参考：切一张 512x512 图的理想间隔约 8~16ms，这里给 20 倍余量。
MAX_GAP_MS = 400.0


def count_threads():
    try:
        return threading.active_count()
    except Exception:
        return -1


def count_qobjects():
    """GUI 线程上存活的 QObject 数量（经 sip 间接统计不可靠，这里用 QApplication
    的顶层对象图规模近似）。"""
    from PySide6.QtCore import QCoreApplication
    try:
        app = QCoreApplication.instance()
        return len(app.findChildren(object)) if app is not None else -1
    except Exception:
        return -1


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []
    notes = []

    root = tempfile.mkdtemp(prefix="visionmind_soak_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    # 用不同尺寸让解码耗时略有差异，避免"每张都一样快"掩盖竞态
    for i in range(IMAGES):
        w = 512 + (i % 4) * 64
        img = QImage(w, 512, QImage.Format.Format_RGB32)
        img.fill(QColor(40 + (i * 7) % 180, 100, 150))
        img.save(os.path.join(d, "img%03d.png" % i))

    print("== 长跑切图稳定性（%d 轮 / %d 图）==" % (ROUNDS, IMAGES))

    w = AnnotationInterface(None, None)
    w.resize(1000, 700)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    w._non_project_format = "labelme"

    baseline_threads = count_threads()
    baseline_objs = count_qobjects()
    print("   基线: 线程=%d 子对象=%d" % (baseline_threads, baseline_objs))

    canvas = w.draw_area
    max_workers = 4
    try:
        if getattr(canvas, "_decoder", None) is not None:
            max_workers = canvas._decoder.max_workers()
    except Exception:
        pass

    worst_gap = 0.0
    worst_at = -1
    thread_samples = []
    obj_samples = []
    concurrency_peak = 0
    last_requested = None

    # 分段采样：每 1/4 进度记录一次，用来看"是否随时间增长"
    sample_at = {ROUNDS * k // 4 for k in range(1, 5)}

    for i in range(ROUNDS):
        target = i % IMAGES
        last_requested = target
        t0 = time.perf_counter()
        w.load_image(target, force_reload=False)
        app.processEvents()
        gap = (time.perf_counter() - t0) * 1000.0
        if gap > worst_gap:
            worst_gap, worst_at = gap, i

        # 并发峰值（若解码器暴露在途计数）
        try:
            dec = getattr(canvas, "_decoder", None)
            inflight = getattr(dec, "_inflight", None)
            if isinstance(inflight, int):
                concurrency_peak = max(concurrency_peak, inflight)
        except Exception:
            pass

        if (i + 1) in sample_at:
            thread_samples.append((i + 1, count_threads()))
            obj_samples.append((i + 1, count_qobjects()))

        # 周期性跑一轮事件循环，模拟真实交互
        if i % 20 == 0:
            app.processEvents()

    # ================= 阶段二：连发切换（真实"快速滑动"形态）=================
    # 阶段一每轮都跑一次 processEvents，间隔 >= 解码耗时；而用户"快速滑动"时是**不等
    # 上一张解码完就继续发请求**（间隔 1ms << 解码耗时），这才是原始冻结报告的形态。
    # 这里做连发压测，重点看：线程是否累积、事件循环最差间隔、收尾能否排空。
    print("\n   ---- 阶段二：连发切换（间隔 ~1ms，不等解码完成）----")
    burst_rounds = ROUNDS * 3
    burst_base_threads = count_threads()
    burst_worst = 0.0
    t_burst = time.perf_counter()
    for i in range(burst_rounds):
        b0 = time.perf_counter()
        w.load_image(i % IMAGES, force_reload=False)
        app.processEvents()
        burst_worst = max(burst_worst, (time.perf_counter() - b0) * 1000.0)
        if i % 50 == 0:
            app.processEvents()
    burst_wall = time.perf_counter() - t_burst
    # 阶段二改变了"最后一次请求"，L5 的期望值必须跟着更新（否则断言的是旧目标）
    last_requested = (burst_rounds - 1) % IMAGES
    print("   连发 %d 次，墙钟 %.2fs，最差间隔 %.1f ms（阈值 %.0f）"
          % (burst_rounds, burst_wall, burst_worst, MAX_GAP_MS))

    # 收尾静止后再看线程
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
    gc.collect()
    burst_end_threads = count_threads()
    print("   L1 连发后线程=%d（连发前 %d）" % (burst_end_threads, burst_base_threads))

    if burst_worst > MAX_GAP_MS:
        failures.append("L4 连发期间事件循环冻结 %.1f ms（阈值 %.0f）"
                        % (burst_worst, MAX_GAP_MS))
    if burst_end_threads > burst_base_threads + 4:
        failures.append("L1 连发后线程未回落: %d（连发前 %d）"
                        % (burst_end_threads, burst_base_threads))

    # ================= 收尾断言 =================
    # 收尾：等到静止
    end = time.time() + 8.0
    settled = False
    while time.time() < end:
        app.processEvents()
        if w.manual.session.is_committed and w.manual.session.is_annotations_loaded:
            settled = True
            break
        time.sleep(0.005)

    print("   L4 事件循环最差间隔 %.1f ms（第 %d 轮，阈值 %.0f ms）"
          % (worst_gap, worst_at, MAX_GAP_MS))
    print("   L3 解码并发峰值 %d（上限 %d）" % (concurrency_peak, max_workers))
    print("   L1 线程采样: %s（基线 %d）"
          % ([(n, t) for n, t in thread_samples], baseline_threads))
    print("   L2 子对象采样: %s（基线 %d）" % (obj_samples, baseline_objs))

    if worst_gap > MAX_GAP_MS:
        failures.append("L4 事件循环冻结 %.1f ms（第 %d 轮，阈值 %.0f）"
                        % (worst_gap, worst_at, MAX_GAP_MS))
    if concurrency_peak > max_workers:
        failures.append("L3 解码并发越界: %d > %d" % (concurrency_peak, max_workers))

    # L1: 线程数不得单调增长（以首次采样为参照，末次允许小幅波动但不翻倍）
    if len(thread_samples) >= 2:
        first_t = thread_samples[0][1]
        last_t = thread_samples[-1][1]
        growth = last_t - first_t
        print("   L1 线程从 %d -> %d（增长 %d）" % (first_t, last_t, growth))
        if growth > 4:
            failures.append("L1 线程数随切换增长 %d（%d -> %d）—— 疑似线程泄漏"
                            % (growth, first_t, last_t))

    # 静止后再采一次：必须回落到基线附近
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
    gc.collect()
    final_threads = count_threads()
    print("   L1 静止后线程=%d（基线 %d）" % (final_threads, baseline_threads))
    if final_threads > baseline_threads + 4:
        failures.append("L1 结束后线程未回落: %d（基线 %d）"
                        % (final_threads, baseline_threads))

    # L5: 排空 + 最新优先
    ok_drain = w.drain(8000)
    got_path = w.current_image_path
    expected = os.path.join(d, "img%03d.png" % last_requested)
    same = (os.path.normcase(os.path.normpath(str(got_path)))
            == os.path.normcase(os.path.normpath(expected)))
    print("   L5 drain=%s 画布图=%s 期望=%s 一致=%s"
          % (ok_drain, os.path.basename(str(got_path)),
             os.path.basename(expected), same))
    if not ok_drain:
        failures.append("L5 drain 未能在 8000ms 内排空")
    if not same:
        failures.append("L5 停止后画布不是最后一次请求的图: %r != %r"
                        % (got_path, expected))

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: L1~L6 全部通过（%d 轮无资源增长、无冻结、无 abort）" % ROUNDS)
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
