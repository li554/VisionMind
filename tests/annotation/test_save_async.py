"""标注写入必须在后台，且不得因异步化引入"乱序覆盖 / 读到写之前" —— 阶段 3c 回归。

被测改动：`save_current(silent=True)`（每次标注修改、每次切图的自动保存，全部调用方都
不使用返回值）改为把**写盘**排入后台串行队列（`AnnotationWriter`）。原先
`save_image_annotations` 在 GUI 线程同步执行，内部还要 `imread_unicode` 读回整张图只为
取宽高，于是"拖一次鼠标写一次盘"。

异步化必须同时解决两件事，否则是拿新缺陷换旧缺陷：
  * 同一路径的两次写入乱序 -> 旧快照盖掉新内容（本测试用同路径最新优先 + 串行覆盖）
  * 切走（排入后台保存）再切回同一张图时"读跑在写之前" -> 刚改的标注消失

断言：
  S1 后台执行：慢写盘（300ms）下 `save_current(silent=True)` 必须 fast return，
     事件循环在写盘期间照常运转；最终文件内容正确
  S2 同路径最新优先：连续 5 次自动保存必须被合并（不是写 5 次），且最终落盘的是**最后一次**
  S3 读在写之后：排入保存后立刻装载同一张图，必须读到**刚保存**的内容（不能读到旧内容）
  S4 D8 守卫在异步路径同样生效：装载失败时拒绝排入写入
  S5 显式保存先等待在途写入落盘（写入顺序），且返回真实结果
  S6 drain() 覆盖写入队列（排空 + 停止写入线程）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_save_async.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_SLOW_WRITE_S = 0.30


def make_images(d):
    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(3):
        path = os.path.join(d, "img%02d.png" % i)
        img = QImage(64, 48, QImage.Format.Format_RGB32)
        img.fill(QColor(50, 90 + 20 * i, 140))
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

    root = tempfile.mkdtemp(prefix="visionmind_save_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    paths = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(1000, 660)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    sess = w.manual.session
    writer = w.manual._writer

    def pump(seconds=2.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    def wait_committed(t=4.0):
        # 必须等到"已装载"（只等像素提交会落在装载窗口里，此时保存被守卫拒绝是正确的）
        pump(t, until=lambda: sess.is_annotations_loaded and sess.is_committed)

    print("== 标注后台写入回归 ==")

    # ---------------- S1: 后台执行 ----------------
    st = {"slow": False, "writes": 0, "snapshots": []}
    real_save = w.manual.save_image_annotations

    def fake_save(image_path, annotations, **kw):
        st["writes"] += 1
        st["snapshots"].append(("%s|%s" % (os.path.basename(image_path),
                                           [a.get("label") for a in annotations])))
        if st["slow"]:
            time.sleep(_SLOW_WRITE_S)
        return real_save(image_path, annotations, **kw)

    w.manual.save_image_annotations = fake_save

    w.load_image(0)
    wait_committed()
    # 非项目模式的格式自动识别在"目录里没有任何标注文件"时会给出 mask，而
    # save_image_annotations 的兜底分支把 mask 也按 <base>.json 落盘 -> 掩码写入器
    # 拿到 .json 路径 -> OpenCV 必然编码失败（审计 D35，本轮只记录不修）。
    # 这里显式锁定 labelme，让本测试聚焦"异步写入"本身。
    w._non_project_format = "labelme"
    w.draw_area.annotations.append(
        {"label": "s1", "bbox": [1, 2, 10, 20], "shape_type": "rect", "polygons": []})

    st["slow"] = True
    ticks = {"n": 0}
    ticker = QTimer()
    ticker.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
    ticker.start(20)

    t0 = time.perf_counter()
    ret = w.save_current(silent=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pending = writer.pending_count()
    pump(2.0, until=lambda: writer.pending_count() == 0)
    ticks_during = ticks["n"]

    print("   S1 save_current(silent=True) 返回 %.1f ms（慢写盘 %.0f ms），ret=%s，"
          "在途=%s，写盘期间 tick=%d"
          % (elapsed_ms, _SLOW_WRITE_S * 1000, ret, pending, ticks_during))
    if elapsed_ms > 80:
        failures.append("S1 自动保存仍在 GUI 线程同步写盘（返回耗时 %.0fms）" % elapsed_ms)
    if ticks_during < 2:
        failures.append("S1 写盘期间事件循环只 tick %d 次 —— GUI 线程被阻塞" % ticks_during)
    ann_file = os.path.join(d, "img00.json")
    if not check(os.path.exists(ann_file), "S1 后台写入没有产生标注文件: %s" % ann_file):
        pass
    else:
        with open(ann_file, encoding="utf-8") as f:
            saved = json.load(f)
        # labelme 落盘的键是 shapes（save_data 的 annotations 由写入器转换）
        raw = saved.get("annotations") or saved.get("shapes") or []
        labels = [a.get("label") for a in raw]
        if labels != ["s1"]:
            failures.append("S1 落盘内容不正确: %s" % labels)
        else:
            print("   S1 落盘内容正确: %s  OK" % labels)

    # ---------------- S2: 同路径最新优先（合并且取最后一次）----------------
    st["slow"] = True
    st["writes"] = 0
    st["snapshots"] = []
    w.draw_area.annotations.append(
        {"label": "s2", "bbox": [1, 2, 10, 20], "shape_type": "rect", "polygons": []})
    for i in range(5):
        w.save_current(silent=True)
        time.sleep(0.01)
    pump(3.0, until=lambda: writer.pending_count() == 0)
    calls = st["writes"]
    last_snap = st["snapshots"][-1] if st["snapshots"] else ""
    print("   S2 连续 5 次自动保存 -> 实际写盘 %d 次；最后一次快照=%s" % (calls, last_snap))
    if calls >= 5:
        failures.append("S2 5 次自动保存写了 %d 次盘（同路径未合并/最新优先失效）" % calls)
    if calls < 1:
        failures.append("S2 没有任何写入发生")
    if "s2" not in last_snap:
        failures.append("S2 最后一次落盘的快照不是最新内容: %s" % last_snap)

    # ---------------- S3: 读必须在写之后 ----------------
    # 排入一次慢速保存（内容含 s3 标签），立刻装载同一张图 -> 必须读到 s3
    st["slow"] = True
    st["writes"] = 0
    w.draw_area.annotations.append(
        {"label": "s3", "bbox": [3, 4, 12, 22], "shape_type": "rect", "polygons": []})
    w.save_current(silent=True)
    pending_before = str(writer.is_path_pending(w.current_image_path))
    target = w.current_image_path
    w.load_image(0, force_reload=True)          # 立刻重新装载同一张图
    pump(5.0, until=lambda: sess.is_annotations_loaded
         and not w._annotation_runner.busy())
    loaded_labels = [a.get("label") for a in sess.annotations
                     if isinstance(a, dict)]
    print("   S3 装载前该路径有在途写入=%s；装载后标注=%s" % (pending_before, loaded_labels))
    if "s3" not in loaded_labels:
        failures.append("S3 装载读到了旧内容（异步写入未完成就读取）: %s" % loaded_labels)
    else:
        print("   S3 装载读到的是刚保存的内容（读等待写入）  OK")

    # ---------------- S4: D8 守卫在异步路径同样生效 ----------------
    st["slow"] = False
    st["writes"] = 0
    sess.mark_annotations_unloaded()
    res = w.manual.schedule_save_current(w.draw_area.annotations, w.current_image_path)
    pump(0.3)
    print("   S4 装载失败后 schedule_save_current=%s 写盘次数=%d"
          % (res.get("status"), st["writes"]))
    if res.get("status") != "error":
        failures.append("S4 装载失败时异步保存未被拒: %r" % res)
    if st["writes"] != 0:
        failures.append("S4 装载失败时仍写出了文件")

    # S4 故意把"已装载"标记清零了，必须重新装载回来（否则后续保存会被 D8 守卫拒绝）
    w.load_image(0, force_reload=True)
    wait_committed()
    if not sess.is_save_allowed:
        failures.append("S4 之后无法恢复可保存状态（前置条件失败，后续断言无效）")

    # ---------------- S5: 显式保存的写入顺序 ----------------
    st["slow"] = True
    st["writes"] = 0
    w.draw_area.annotations.append(
        {"label": "s5", "bbox": [5, 6, 14, 24], "shape_type": "rect", "polygons": []})
    w.save_current(silent=True)                  # 排入一个慢速后台写
    st["slow"] = False
    ret = w.save_current(silent=False)           # 显式保存：必须先等它落盘
    pump(0.3)
    with open(ann_file, encoding="utf-8") as f:
        saved_final = json.load(f)
    raw_final = saved_final.get("annotations") or saved_final.get("shapes") or []
    labels_after = [a.get("label") for a in raw_final]
    print("   S5 显式保存返回=%s；最终文件=%s" % (ret, labels_after))
    if not (isinstance(ret, str) and ret.startswith("保存成功")):
        failures.append("S5 显式保存未返回成功: %r" % ret)
    if "s5" not in labels_after:
        failures.append("S5 最终文件被排队中的旧快照覆盖: %s" % labels_after)

    # ---------------- S7: 后台保存失败必须上报（否则是静默丢数据）----------------
    reported = []
    real_handler = w.manual._writer._failure_handler
    w.manual._writer.set_failure_handler(lambda p, m: reported.append((p, m)))
    real_save2 = w.manual.save_image_annotations

    def failing_save(image_path, annotations, **kw):
        return {"status": "error", "message": "模拟编码失败"}

    w.manual.save_image_annotations = failing_save
    try:
        w.save_current(silent=True)
        pump(2.0, until=lambda: len(reported) > 0)
    finally:
        w.manual.save_image_annotations = real_save2
        w.manual._writer.set_failure_handler(real_handler)
    print("   S7 写盘失败上报次数=%d  内容=%s" % (len(reported), reported[:1]))
    if not reported:
        failures.append("S7 后台保存失败未被上报（异步路径会静默丢数据）")
    elif "模拟编码失败" not in reported[0][1]:
        failures.append("S7 上报的失败信息不含原因: %r" % (reported[0],))

    # ---------------- S6: drain 覆盖写入队列 ----------------
    st["slow"] = True
    w.save_current(silent=True)
    ok = w.manual.drain(5000)
    left = writer.pending_count()
    print("   S6 drain(5000)=%s  剩余在途=%d  写入线程存活=%s"
          % (ok, left, writer._thread.is_alive()))
    if not ok:
        failures.append("S6 drain 未能在超时内排空写入队列")
    if left != 0:
        failures.append("S6 drain 后仍有 %d 项在途" % left)
    if writer._thread.is_alive():
        failures.append("S6 drain 后写入线程仍在运行")

    ticker.stop()
    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: S1~S7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
