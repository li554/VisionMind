"""标注界面切图压力回归 — 真实 AnnotationInterface 端到端。

为什么必须是这个形态（而不是普通单元测试）：
  被测缺陷是 `qFatal("QThread: Destroyed while thread '' is still running")`
  → `abort()`。这是进程级强杀，Python 层不可能捕获，**测试无法自报失败**。
  因此本脚本把"退出码"作为断言的一部分：
      0           = 全部断言通过
      1           = 断言失败（脚本自己 sys.exit(1)）
      -1073740791 = 0xC0000409，进程被 abort（QThread 析构崩溃）
      其他非零     = 其它异常退出

覆盖的架构不变量：
  I1 快速切图（间隔 < 单张解码耗时）不得使进程 abort      —— 线程生命周期
  I2 文件列表行、widget 当前路径、service 当前路径必须一致 —— 状态所有权
  I3 在途解码并发数必须有界（<= max_workers）             —— 资源上界
  I4 切换停止后，画布必须提交**最后一次请求**的那张图     —— 最新优先不丢结果

用法:
    python -u tests/annotation/test_switch_stress.py [rounds] [interval_ms]
    python -u tests/annotation/test_switch_stress.py 40 20      # 默认
    python -u tests/annotation/test_switch_stress.py 80 5       # 更极端

回归用法：把本脚本的退出码作为 pass/fail（见上）。
"""
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def isolate_state(state_dir):
    """把持久化存储全部重定向到临时目录，并从未知状态开始。

    必须在构造任何界面/服务之前调用。这个应用默认会把运行时状态写进真实文件：
      - core/common/settings.py        -> <repo>/core/core.json（受版本控制）
      - core/common/project_settings.py -> <repo>/projects/<name>/project_info.json
    没有这一步，跑测试会污染用户数据（曾经的探针就踩过这个坑）。
    """
    os.makedirs(state_dir, exist_ok=True)

    import copy
    import core.common.settings as cs
    cs.settings.filepath = os.path.join(state_dir, "core.json")
    cs.settings._schema_data = []
    cs.settings._data = {}

    import core.common.project_settings as ps
    ps.project_settings._project_name = None
    ps.project_settings._project_path = None
    ps.project_settings._settings = copy.deepcopy(ps.project_settings.DEFAULT_SETTINGS)

    return cs.settings.filepath


def build_fixture(state_dir, n=4):
    """n 份大图副本。用大图是刻意的：解码要几十毫秒，才能制造出
    "上一次解码还没回来就切下一张" 的窗口（小图会立刻解码完，测不出问题）。"""
    src = os.path.join(_ROOT, "resources", "img.png")
    if not os.path.exists(src):
        raise SystemExit("缺少测试图 %s" % src)
    imgs = os.path.join(state_dir, "dataset", "images")
    os.makedirs(imgs, exist_ok=True)
    for i in range(n):
        dst = os.path.join(imgs, "p_%02d.png" % i)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
    return os.path.join(state_dir, "dataset")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    interval_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    settings_path = isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    t0 = time.perf_counter()
    w = AnnotationInterface(None, None)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    build_ms = (time.perf_counter() - t0) * 1000.0

    dataset = build_fixture(state_dir)
    w.open_directory(dataset)
    if w.file_list.count() < 2:
        print("FAIL I0: 测试数据集未加载成功")
        return 1

    max_workers = w.draw_area._decoder.max_workers()
    dec = w.draw_area._decoder

    print("== 切图压力回归 ==")
    print("   rounds=%d interval=%dms  images=%d  max_workers=%d"
          % (rounds, interval_ms, w.file_list.count(), max_workers))
    print("   界面构建 %.0f ms  隔离设置=%s" % (build_ms, settings_path))

    st = {
        "i": 0, "i2_fail": [], "concurrency_peak": 0,
        "start": time.perf_counter(), "worst_gap": 0.0, "last_tick": time.perf_counter(),
    }

    def sample_concurrency():
        n = dec.active_count()
        if n > st["concurrency_peak"]:
            st["concurrency_peak"] = n

    def tick_switch():
        now = time.perf_counter()
        st["worst_gap"] = max(st["worst_gap"], (now - st["last_tick"]) * 1000.0)
        st["last_tick"] = now
        sample_concurrency()

        st["i"] += 1
        if st["i"] > rounds:
            timer.stop()
            QTimer.singleShot(2500, finish)
            return

        w.switch_image(1 if st["i"] % 2 else -1)

        # I2: 列表行 / 界面当前路径 / service 当前路径 三者必须指向同一张图
        row = w.file_list.currentRow()
        row_path = w.image_files[row] if 0 <= row < len(w.image_files) else None
        if not (row_path == w.current_image_path == w.manual.current_image_path):
            st["i2_fail"].append((st["i"], row, w.current_image_path,
                                  w.manual.current_image_path))

    def finish():
        sample_concurrency()
        elapsed = time.perf_counter() - st["start"]
        print("\n== 结果 ==")
        print("  切换 %d 次，墙钟 %.2fs（理想 %.2fs）"
              % (rounds, elapsed, rounds * interval_ms / 1000.0))
        print("  事件循环最差间隔 %.1f ms" % st["worst_gap"])
        print("  I3 解码并发峰值 %d（上限 %d）" % (st["concurrency_peak"], max_workers))

        failures = []
        if st["i2_fail"]:
            failures.append("I2 状态不一致 %d 例，例如 %s"
                            % (len(st["i2_fail"]), st["i2_fail"][:3]))
        if st["concurrency_peak"] > max_workers:
            failures.append("I3 并发越界 %d > %d"
                            % (st["concurrency_peak"], max_workers))

        # I4: 停止切换后，画布必须持有**最后一次请求**的图（最新优先不得丢结果）
        # 同时校验"唯一状态所有者"：三处投影都必须等于 session 里的真值。
        expected = w.current_image_path
        got = w.draw_area.current_image_path
        if got != expected:
            failures.append("I4 画布路径 %r != 最后请求 %r" % (got, expected))
        session_path = w.manual.session.path
        if not (session_path == w.manual.current_image_path == w.current_image_path
                == w.draw_area.current_image_path):
            failures.append("I4 唯一状态所有者不一致: session=%r service=%r interface=%r widget=%r"
                            % (session_path, w.manual.current_image_path,
                               w.current_image_path, w.draw_area.current_image_path))
        if not w.draw_area.is_committed:
            failures.append("I4 静置后画布仍未提交（is_committed=False）")
        if w.draw_area.pixmap is None:
            failures.append("I4 画布未提交 pixmap（最后一张图丢失）")
        if dec.active_count() != 0:
            failures.append("I4 静置后仍有 %d 个解码在跑" % dec.active_count())

        # 排空能力（关闭路径依赖它）
        if not w.draw_area.drain(5000):
            failures.append("I5 drain(5000) 未能在超时内排空解码线程池")

        if failures:
            print("  FAIL:")
            for f in failures:
                print("    - %s" % f)
        else:
            print("  PASS: I1/I2/I3/I4/I5 全部通过")

        st["code"] = 1 if failures else 0
        QTimer.singleShot(100, app.quit)

    timer = QTimer()
    timer.timeout.connect(tick_switch)
    timer.start(interval_ms)

    st["code"] = 1  # 若事件循环异常退出（含 abort 之外的意外），按失败处理
    QTimer.singleShot(rounds * interval_ms + 15000, app.quit)
    app.exec()

    code = st.get("code", 1)
    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
