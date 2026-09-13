"""逐图标注装载必须走后台 I/O，且结果带代次校验 —— 阶段 3 回归。

被测缺陷：`load_image_annotations()` 原先在 GUI 线程**同步**读盘 + 解析
（非项目模式还会降级二次扫描），发生在每一次切图上；紧随其后 `load_image` 又调用
一次 `refresh_label_list()`，于是每次切图要全量重建两遍标注列表
（实测 20 条标注约 240ms）。这是"IO 全在后台"这条要求里最后一块同步 I/O。

断言：
  P1 切图不在 GUI 线程做标注 I/O：慢 loader（250ms）下 load_image 必须快速返回，
     且事件循环在装载期间照常运转
  P2 代次校验 / 最新优先：慢 loader 下连续切 A→B→C，**任何时刻都不得**看到
     非当前图的标注落地；B 的中间请求应被合并掉
  P3 装载窗口内禁止保存与交互（避免把上一张的标注写进这张图的文件 / 把刚画的
     内容被随后的既有标注覆盖）
  P4 每次切图只重建一次标注列表（去掉重复 refresh_label_list）
  P5 drain() 覆盖标注装载器

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_annotation_io_async.py
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

_IMAGE_COUNT = 4
_SLOW_LOAD_S = 0.25


def make_images(d):
    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(_IMAGE_COUNT):
        path = os.path.join(d, "img%02d.png" % i)
        img = QImage(96, 64, QImage.Format.Format_RGB32)
        img.fill(QColor(40 * i % 255, 90, 150))
        img.save(path)
        paths.append(path)
    return paths


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_io_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    paths = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(1100, 700)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    sess = w.manual.session

    def tags_of(annotations):
        return {a.get("label") for a in annotations if isinstance(a, dict)}

    def tag_for(path):
        return "tag:" + os.path.basename(path)

    def pump(seconds=1.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    # 可切换"慢/快"的假 loader：慢的时候用于制造在途装载窗口
    state = {"slow": False, "calls": []}

    def fake_load(image_path, mode, strict=False, **kw):
        state["calls"].append(image_path)
        if state["slow"]:
            time.sleep(_SLOW_LOAD_S)
        return {"status": "success", "annotations": [
            {"label": tag_for(image_path), "bbox": [1, 2, 20, 30],
             "polygons": [[[1, 2], [21, 2], [21, 32], [1, 32]]],
             "shape_type": "polygon"}]}, "labelme"

    w.manual.load_annotations_for_image = fake_load

    print("== 标注装载后台 I/O 回归 ==")

    # ---------------- P1: GUI 线程不得做标注 I/O ----------------
    state["slow"] = True
    ticks = {"n": 0}
    ticker = QTimer()
    ticker.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
    ticker.start(20)

    t0 = time.perf_counter()
    w.load_image(1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # 装载在途时事件循环应当照常运转
    pump(0.6, until=lambda: sess.is_annotations_loaded)
    ticks_during = ticks["n"]
    ticker.stop()

    print("   P1 load_image 返回耗时 %.1f ms（慢 loader %.0f ms），装载期间 tick=%d"
          % (elapsed_ms, _SLOW_LOAD_S * 1000, ticks_during))
    if elapsed_ms > 80:
        failures.append("P1 load_image 耗时 %.0fms —— 标注 I/O 仍在 GUI 线程同步执行"
                        % elapsed_ms)
    if ticks_during < 2:
        failures.append("P1 装载期间事件循环只 tick 了 %d 次 —— GUI 线程被阻塞"
                        % ticks_during)
    if not sess.is_annotations_loaded:
        failures.append("P1 装载最终未完成")

    # ---------------- P2: 代次校验 / 最新优先 ----------------
    state["slow"] = True
    state["calls"].clear()          # 只统计本段发起的装载
    samples = []

    def sample():
        samples.append((sess.path, frozenset(tags_of(sess.annotations))))

    # 连续切 A→B→C，中间只跑少量事件循环（让第一次装载在途）
    w.load_image(0)
    sample()
    app.processEvents(); sample()
    w.load_image(1)
    sample()
    app.processEvents(); sample()
    w.load_image(2)
    sample()
    # 等到尘埃落定
    end = time.time() + 4
    while time.time() < end:
        app.processEvents()
        sample()
        if sess.is_annotations_loaded and not w._annotation_runner.busy() \
                and w._pending_annotation_request is None:
            break
        time.sleep(0.005)

    bad = []
    for path, tags in samples:
        if not tags:
            continue
        want = {tag_for(path)}
        if tags != want:
            bad.append((os.path.basename(path or ""), sorted(tags)))
    final_tags = tags_of(sess.annotations)
    print("   P2 采样 %d 次；最终图=%s 标注=%s；越权落地 %d 例"
          % (len(samples), os.path.basename(sess.path or ""), sorted(final_tags), len(bad)))
    if bad:
        failures.append("P2 出现过「标注不属于当前图」的落地: %s" % bad[:3])
    if final_tags != {tag_for(paths[2])}:
        failures.append("P2 最终标注 %s != 图 C 的 %s（最新优先失效）"
                        % (sorted(final_tags), tag_for(paths[2])))
    # 最新优先：B 的请求应当被合并掉，loader 调用次数 <= 3（A + C，B 被合并）
    print("   P2 loader 调用: %s" % [os.path.basename(p) for p in state["calls"]])
    if state["calls"].count(paths[1]) > 0:
        failures.append("P2 中间图 B 的装载未被合并（最新优先失效）")

    # ---------------- P3: 装载窗口内禁止保存与交互 ----------------
    state["calls"].clear()
    state["slow"] = True
    w.load_image(3)
    allowed = sess.is_save_allowed
    ready = w.draw_area._interaction_ready()
    files_before = set(os.listdir(d))
    res = w.manual.save_current()
    files_after = set(os.listdir(d))
    print("   P3 装载窗口内 is_save_allowed=%s is_ready=%s save_current=%s"
          % (allowed, ready, res.get("status")))
    if allowed:
        failures.append("P3 装载窗口内仍允许保存")
    if ready:
        failures.append("P3 装载窗口内交互闸门仍开放（会画了就被覆盖）")
    if res.get("status") != "error":
        failures.append("P3 装载窗口内 save_current 未拒绝: %r" % res)
    if files_after != files_before:
        failures.append("P3 装载窗口内仍写出了文件: %s" % sorted(files_after - files_before))
    pump(1.5, until=lambda: sess.is_annotations_loaded)

    # ---------------- P4: 每次切图只重建一次列表 ----------------
    state["slow"] = False
    real_refresh = w.refresh_label_list
    counter = {"n": 0}

    def counting_refresh():
        counter["n"] += 1
        return real_refresh()

    w.refresh_label_list = counting_refresh
    try:
        deltas = []
        for i in (0, 1, 2, 3):
            before = counter["n"]
            w.load_image(i)
            pump(1.0, until=lambda: sess.is_annotations_loaded
                 and not w._annotation_runner.busy())
            deltas.append(counter["n"] - before)
    finally:
        w.refresh_label_list = real_refresh
    print("   P4 每次切图的 refresh_label_list 次数: %s" % deltas)
    if any(n != 1 for n in deltas):
        failures.append("P4 每次切图应恰好重建一次标注列表，实测 %s（重复重建会随"
                        "标注数线性拖慢切图）" % deltas)

    # ---------------- P5: drain 覆盖标注装载器 ----------------
    state["slow"] = True
    w.load_image(0)
    ok = w.drain(5000)
    busy = w._annotation_runner.busy()
    print("   P5 drain(5000)=%s  装载器仍忙碌=%s" % (ok, busy))
    if not ok:
        failures.append("P5 drain(5000) 未能在超时内排空")
    if busy:
        failures.append("P5 drain 后标注装载器仍在跑")

    ticker.stop()
    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: P1~P5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
