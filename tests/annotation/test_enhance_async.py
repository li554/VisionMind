"""增强图生成必须在后台 + 结果校验 + 代次守卫 —— D17 回归。

被测缺陷（审计 D17）：
  1. `_generate_enhanced_pixmap()` 在 **GUI 线程**同步执行 LibLLIE + numpy（每次切图
     提交都会跑），开启增强后每次切图都冻结界面；
  2. 对增强器返回值**零校验**：直接 `h, w, ch = enhanced_arr.shape` 并用 `ch * w`
     当 bytesPerLine 构造 QImage —— 返回 float32/uint16 时行距与真实不符，会逐行
     越界读 numpy 缓冲（花屏或崩溃）。

断言：
  F1 后台执行：慢增强器（250ms）下 `_generate_enhanced_pixmap()` 必须 fast return，
     且事件循环在生成期间照常运转；完成后结果被安装
  F2 代次守卫：A 图增强在途时切到 B 图，A 的结果**不得**被安装（否则就是把 A 的
     增强图贴到 B 上）；B 的结果必须被安装
  F3 结果校验：float32 / RGBA / 非连续 / 2 维 等畸形返回值的规范化行为
  F4 drain 覆盖增强线程
  F5 增强器不可用（如缺 libllie）时不在每次切图重复尝试

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_enhance_async.py
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

_SLOW_S = 0.25


def make_images(d):
    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(3):
        path = os.path.join(d, "img%02d.png" % i)
        img = QImage(80, 60, QImage.Format.Format_RGB32)
        img.fill(QColor(30 + 50 * i, 70, 110))
        img.save(path)
        paths.append(path)
    return paths


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_enh_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    paths = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    canvas = w.draw_area
    sess = w.manual.session

    def pump(seconds=2.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    def wait_committed(t=3.0):
        pump(t, until=lambda: sess.is_committed)

    # 可调"慢/快"的假增强器：返回与输入同形状的 uint8 数组
    st = {"slow": False, "calls": 0}

    class _FakeEnhancer:
        def __call__(self, arr):
            st["calls"] += 1
            if st["slow"]:
                time.sleep(_SLOW_S)
            return np.ascontiguousarray(arr)

    # ---------------- F3: 结果校验（纯函数，先测最关键的）----------------
    norm = canvas._normalize_enhanced_array
    h, wd = 4, 5
    base = np.zeros((h, wd, 3), np.uint8)
    try:
        got = norm(base)
        check(got.shape == (h, wd, 3) and got.dtype == np.uint8 and got.flags["C_CONTIGUOUS"],
              "F3 uint8 RGB 应原样通过")
        got4 = norm(np.zeros((h, wd, 4), np.uint8))
        check(got4.shape == (h, wd, 3), "F3 RGBA 应降成 RGB")
        gotf = norm(np.full((h, wd, 3), 300.0, np.float32))
        check(gotf.dtype == np.uint8 and gotf.max() <= 255,
              "F3 float32 应 clip 成 uint8（否则行距不符会越界读）")
        noncontig = np.zeros((h, wd * 2, 3), np.uint8)[:, ::2, :]
        check(not noncontig.flags["C_CONTIGUOUS"], "F3 前置条件：样本应为非连续")
        check(norm(noncontig).flags["C_CONTIGUOUS"], "F3 非连续输入应转成连续数组")
        for bad, label in ((np.zeros((h, wd), np.uint8), "二维"),
                           (np.zeros((h, wd, 2), np.uint8), "两通道"),
                           (np.zeros((0, wd, 3), np.uint8), "空尺寸")):
            try:
                norm(bad)
                failures.append("F3 %s 畸形返回值未被拒绝" % label)
            except ValueError:
                pass
    except Exception as exc:
        failures.append("F3 规范化抛异常: %r" % exc)
    if not any(f.startswith("F3") for f in failures):
        print("   F3 结果校验：RGBA/float32/非连续/畸形形状 处理正确  OK")

    # ---------------- F1: 后台执行 ----------------
    w.load_image(0)
    wait_committed()
    canvas._llie_enhancer = _FakeEnhancer()
    st["slow"] = True
    ticks = {"n": 0}
    ticker = QTimer()
    ticker.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
    ticker.start(20)

    t0 = time.perf_counter()
    accepted = canvas._generate_enhanced_pixmap()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    busy = canvas._enhancer_runner.busy()
    pump(1.5, until=lambda: canvas._enhanced_pixmap is not None)
    ticks_during = ticks["n"]
    ticker.stop()
    print("   F1 _generate_enhanced_pixmap 返回 %.1f ms（慢增强器 %.0f ms），"
          "忙碌=%s，生成期间 tick=%d，结果已安装=%s"
          % (elapsed_ms, _SLOW_S * 1000, busy, ticks_during,
             canvas._enhanced_pixmap is not None))
    if elapsed_ms > 60:
        failures.append("F1 增强仍在 GUI 线程同步执行（返回耗时 %.0fms）" % elapsed_ms)
    if not busy:
        failures.append("F1 未启动后台增强任务")
    if ticks_during < 2:
        failures.append("F1 生成期间事件循环只 tick %d 次 —— GUI 线程被阻塞" % ticks_during)
    if not check(canvas._enhanced_pixmap is not None, "F1 后台生成的结果未被安装"):
        pass

    # ---------------- F2: 代次守卫（A 在途时切到 B）----------------
    st["slow"] = True
    installs = []
    stale = []

    real_ready = canvas._on_enhance_ready

    def traced_ready(payload):
        gen, path, _arr = payload
        same = (path == sess.path and gen == sess.generation)
        if same:
            installs.append(os.path.basename(path))
        else:
            stale.append(os.path.basename(path))
            record["pixmap_when_stale_arrived"] = canvas._enhanced_pixmap is not None
        return real_ready(payload)

    record = {}
    canvas._on_enhance_ready = traced_ready

    canvas._enhance_requested = True
    canvas._show_enhanced = False
    canvas._enhanced_pixmap = None
    w.load_image(1)
    wait_committed()
    # 此刻 A(=img01) 的增强在途；立刻切到 B(=img02)
    w.load_image(2)
    wait_committed()
    end = time.time() + 4
    while time.time() < end:
        app.processEvents()
        if canvas._enhanced_pixmap is not None and not canvas._enhancer_runner.busy() \
                and canvas._pending_enhance_request is None:
            break
        time.sleep(0.005)
    canvas._on_enhance_ready = real_ready

    print("   F2 安装的增强结果=%s  被丢弃的过期结果=%s  过期到达时是否误装=%s"
          % (installs, stale, record.get("pixmap_when_stale_arrived")))
    if not stale:
        failures.append("F2 未制造出「过期增强结果」（测试无效，前提不成立）")
    if record.get("pixmap_when_stale_arrived"):
        failures.append("F2 过期增强结果被安装（会把 A 的增强图贴到 B 上）")
    if installs != ["img02.png"]:
        failures.append("F2 只应安装当前图 img02.png 的增强结果，实测 %s" % installs)

    # ---------------- F4: drain 覆盖增强线程 ----------------
    st["slow"] = True
    canvas._enhance_requested = True
    canvas._generate_enhanced_pixmap()
    ok = canvas.drain(5000)
    busy_after = canvas._enhancer_runner.busy()
    print("   F4 drain(5000)=%s  增强线程仍忙碌=%s" % (ok, busy_after))
    if not ok:
        failures.append("F4 drain(5000) 未能在超时内排空")
    if busy_after:
        failures.append("F4 drain 后增强线程仍在跑")

    # ---------------- F5: 增强器不可用时不再重试 ----------------
    # 注意：drain() 会作废代次并清空画布，所以必须先重新提交一张图再测
    canvas._enhance_requested = False
    w.load_image(0)
    wait_committed()
    if not sess.is_committed:
        failures.append("F5 前置条件失败：无法重新提交图像")

    def failing_ensure():
        raise ModuleNotFoundError("No module named 'libllie'")

    canvas._llie_enhancer = None
    canvas._enhancer_unavailable = False
    canvas._ensure_enhancer = failing_ensure
    canvas._enhance_requested = True
    started = canvas._generate_enhanced_pixmap()
    pump(1.5, until=lambda: canvas._enhancer_unavailable)
    print("   F5 已受理=%s  增强器不可用标记=%s" % (started, canvas._enhancer_unavailable))
    if not canvas._enhancer_unavailable:
        failures.append("F5 缺失 libllie 的失败未被记录（会每次切图重复尝试并刷屏）")

    ticker.stop()
    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: F1~F5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
