"""搜索扫描必须去抖 + 在后台执行 —— D15 回归。

被测缺陷（审计 D15）：`search_box.textChanged` 直接连到 `search_images`，于是
**每次击键**都在 GUI 线程做一次全量扫描；一旦启用类别/尺寸/标注状态筛选，扫描会
逐图读盘解析标注（`get_cached_annotations`）—— 目录大一点就是秒级到分钟级冻结。

断言：
  Q1 击键不再触发扫描：连续 5 次输入必须 fast return，扫描次数被去抖/最新优先压到 <=2，
     且扫描期间事件循环照常运转
  Q2 最新优先：先按 "a" 起一次慢扫描，随即改成 "img0" —— 只允许应用 "img0" 的结果
  Q3 显式搜索作废在途后台搜索：显式同步搜索后，迟到的后台结果不得覆盖它
  Q4 drain() 覆盖搜索扫描器
  Q5 去抖窗口内 GUI 线程不读盘（逐图标注 I/O 计数不增长）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_search_async.py
"""
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_N_IMAGES = 12
_PER_IMAGE_S = 0.03          # 模拟"逐图读盘解析标注"


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_search_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    for i in range(_N_IMAGES):
        img = QImage(32, 24, QImage.Format.Format_RGB32)
        img.fill(QColor(40 + i, 90, 150))
        img.save(os.path.join(d, "img%02d.png" % i))

    w = AnnotationInterface(None, None)
    w.resize(1000, 660)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    # 等标注装载落地（异步），避免基线被迟到结果干扰
    _end = time.time() + 5
    while time.time() < _end:
        app.processEvents()
        if w.manual.session.is_annotations_loaded and not w._annotation_runner.busy():
            break
        time.sleep(0.005)

    def pump(seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    # 记录每次"应用可见性"的结果，用于判断哪次搜索结果真的落地了
    applied = []
    real_apply = w._apply_file_visibility

    def traced_apply(visibility):
        applied.append([os.path.basename(p) for p, hidden in visibility if not hidden])
        return real_apply(visibility)

    w._apply_file_visibility = traced_apply

    # 逐图读盘计数（模拟真实开销），并强制启用筛选让扫描必须读标注
    io = {"reads": 0}

    def slow_cached(image_path, output_dir, is_non_project_mode, non_project_format):
        io["reads"] += 1
        time.sleep(_PER_IMAGE_S)
        return []

    w.manual.get_cached_annotations = slow_cached
    # 触发"逐图读盘"的真实条件：搜索框里带 @划分: token 时筛选器保持有效
    # （_apply_search_tokens 在**没有**对应 token 时会清除筛选，所以纯文本输入不会读盘）
    TOKEN = "@划分:未划分 "

    # 关键词级别的记账：区分"结果回来了"与"结果被应用了"
    # （本测试里类别筛选匹配不到任何标注，可见集恒为空，不能靠内容区分关键词）
    search_calls = []
    done_calls = []
    real_search = w.manual.search_images

    def traced_search(keyword, **kw):
        search_calls.append(keyword)
        return real_search(keyword, **kw)

    w.manual.search_images = traced_search
    real_done = w._on_search_done

    def traced_done(request, result):
        done_calls.append(request["keyword"])
        return real_done(request, result)

    w._on_search_done = traced_done

    print("== 搜索扫描后台化回归 ==")

    # ---------------- Q1: 击键只置脏 + 去抖 ----------------
    ticks = {"n": 0}
    ticker = QTimer()
    ticker.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
    ticker.start(20)

    per_keystroke = []
    for text in ("i", "im", "img", "img0", "img00"):
        full = TOKEN + text
        t0 = time.perf_counter()
        w.search_box.setText(full)
        per_keystroke.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.01)                        # 模拟人类击键间隔
    reads_during_typing = io["reads"]           # 去抖窗口内不得读盘（Q5）
    # 注意：必须等"扫描真的发生并且跑完"，否则 until 一开始就成立会立刻返回
    pump(6.0, until=lambda: io["reads"] >= _N_IMAGES
         and not w._search_runner.busy())
    scans = io["reads"] / float(_N_IMAGES)
    ticks_during = ticks["n"]
    ticker.stop()
    worst_ms = max(per_keystroke)
    print("   Q1 5 次击键最慢一次 %.1f ms；去抖窗口内读盘=%d 次；实际扫描轮数=%.1f（应≈1）；"
          "扫描期间 tick=%d" % (worst_ms, reads_during_typing, scans, ticks_during))
    if worst_ms > 50:
        failures.append("Q1 击键处理耗时 %.0fms —— 仍在 GUI 线程同步扫描" % worst_ms)
    if not check(scans <= 1.5, "Q1 扫描轮数 %.1f（去抖失效，应只跑一轮）" % scans):
        pass
    if ticks_during < 3:
        failures.append("Q1 扫描期间事件循环只 tick %d 次 —— GUI 被阻塞" % ticks_during)
    if io["reads"] < _N_IMAGES:
        failures.append("Q1 去抖后没有任何扫描发生（测试无效）")

    # ---------------- Q5: 去抖窗口内 GUI 线程不读盘 ----------------
    print("   Q5 去抖窗口内（击键之间）读盘次数=%d" % reads_during_typing)
    if reads_during_typing != 0:
        failures.append("Q5 去抖窗口内 GUI 线程读了 %d 次盘（应只在去抖到点后由后台扫描读）"
                        % reads_during_typing)

    # ---------------- Q2: 最新优先 ----------------
    applied.clear()
    io["reads"] = 0
    search_calls.clear()
    done_calls.clear()
    w.search_box.setText(TOKEN + "a")           # 先起一次慢扫描
    pump(0.60, until=lambda: w._search_runner.busy())    # 等去抖到点、扫描开始
    started = w._search_runner.busy()
    w.search_box.setText(TOKEN + "img0")        # 扫描在途时改关键词
    pump(6.0, until=lambda: not w._search_runner.busy()
         and w._pending_search_request is None)
    print("   Q2 第一次扫描已启动=%s；返回的关键词=%s；被应用次数=%d"
          % (started, done_calls, len(applied)))
    if not started:
        failures.append("Q2 前置条件失败：第一次扫描没有启动")
    if "a" not in done_calls or "img0" not in done_calls:
        failures.append("Q2 两次扫描都应有结果返回（证明过期结果确实到达过）: %s" % done_calls)
    if len(applied) != 1:
        failures.append("Q2 只应应用最后一次（img0）的结果，实际应用了 %d 次" % len(applied))
    if search_calls and search_calls[-1] != "img0":
        failures.append("Q2 最后一次扫描的关键词不是 img0: %s" % search_calls)

    # ---------------- Q3: 显式搜索作废在途后台搜索 ----------------
    applied.clear()
    io["reads"] = 0
    search_calls.clear()
    done_calls.clear()
    w.search_box.setText(TOKEN + "a")
    pump(0.60, until=lambda: w._search_runner.busy())
    ret = w.search_images(TOKEN + "img02")      # 显式同步搜索（应立即应用）
    after_explicit = len(applied)
    pump(6.0, until=lambda: not w._search_runner.busy()
         and w._pending_search_request is None)
    print("   Q3 显式搜索返回=%r；显式后立即应用=%d 次；最终应用=%d 次；后台关键词=%s"
          % (ret, after_explicit, len(applied), done_calls))
    if after_explicit != 1:
        failures.append("Q3 显式同步搜索没有立即应用结果")
    if len(applied) != 1:
        failures.append("Q3 迟到的后台结果覆盖了显式搜索结果（共应用 %d 次）" % len(applied))
    if not done_calls and "a" not in done_calls:
        failures.append("Q3 前置条件失败：被作废的那次后台搜索没有返回")

    # ---------------- Q4: drain 覆盖搜索扫描器 ----------------
    w.search_box.setText("img")
    pump(0.20, until=lambda: w._search_runner.busy())
    ok = w.drain(5000)
    busy = w._search_runner.busy()
    print("   Q4 drain(5000)=%s  搜索扫描器仍忙碌=%s" % (ok, busy))
    if not ok:
        failures.append("Q4 drain(5000) 未能在超时内排空")
    if busy:
        failures.append("Q4 drain 后搜索扫描器仍在跑")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: Q1~Q5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
