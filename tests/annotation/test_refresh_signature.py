"""标注刷新去重签名必须覆盖几何，且真值被替换时缓存必须作废 —— D33 回归。

被测缺陷（审计 D33，两层）：

1. **签名盲于几何**：`_annotations_state_signature()` 只记录
   `len(a.get("polygons"))` —— 移动一个多边形顶点、改变多边形形状，元素个数/标签/
   bbox 都没变 -> 签名不变 -> `on_annotations_changed` 提前 return -> **刷新被抑制**，
   画布与列表停留在旧几何上。
2. **私有第二份状态不清**：`_last_ann_sig` / `_last_ann_path` 在 `clear_project` /
   `_do_set_project` 时不重置 -> 切项目后若新数据恰好同签名，刷新被跳过。

断言：
  G1 移动多边形顶点必须改变签名（修复前签名完全相同 —— 实测 sig1 == sig2）
  G2 修改 bbox 必须改变签名
  G3 标签数量/内容变化必须改变签名
  G4 仅隐藏索引变化必须改变签名
  G5 顶点移动后 on_annotations_changed 必须**真的刷新**（不得被去重抑制）
  G6 真值被整体替换时缓存必须作废（切图 / 切项目 / 清项目三个入口）
  G7 同一状态重复发布仍然去重（不能因为修 G1 而丢掉防重入能力）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_refresh_signature.py
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


def make_images(d):
    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(3):
        path = os.path.join(d, "img%02d.png" % i)
        img = QImage(64, 48, QImage.Format.Format_RGB32)
        img.fill(QColor(50 + 25 * i, 90, 150))
        img.save(path)
        paths.append(path)
    return paths


def poly(coords):
    return [[list(c) for c in coords]]


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_sig_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    paths = make_images(d)

    w = AnnotationInterface(None, None)
    w.resize(1000, 660)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    w._non_project_format = "labelme"

    def pump(seconds=1.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    w.load_image(0, force_reload=True)
    pump(4.0, until=lambda: w.manual.session.is_annotations_loaded
         and w.manual.session.is_committed)

    # 统计真实刷新次数
    refreshes = {"n": 0}
    real_refresh = w.refresh_label_list

    def counting_refresh():
        refreshes["n"] += 1
        return real_refresh()

    w.refresh_label_list = counting_refresh

    ann_a = [{"label": "a", "bbox": [1, 2, 10, 20],
              "polygons": poly([[1, 2], [11, 2], [11, 22], [1, 22]]),
              "shape_type": "polygon"}]

    def set_anns(anns):
        w.manual.set_current_annotations(anns, for_path=w.current_image_path)

    print("== 标注刷新签名回归（D33）==")

    # ---------------- G1: 顶点移动必须改变签名 ----------------
    set_anns(ann_a)
    sig_a = w._annotations_state_signature()
    moved = [{"label": "a", "bbox": [1, 2, 10, 20],
              "polygons": poly([[1, 2], [99, 2], [99, 99], [1, 99]]),
              "shape_type": "polygon"}]
    set_anns(moved)
    sig_moved = w._annotations_state_signature()
    print("   G1 顶点移动前后签名是否不同: %s" % (sig_a != sig_moved))
    check(sig_a != sig_moved,
          "G1 顶点移动未改变签名 —— 真实编辑会被去重抑制（D33 第 1 层）")

    # ---------------- G2: bbox 变化必须改变签名 ----------------
    bbox_changed = [{"label": "a", "bbox": [5, 5, 30, 30],
                     "polygons": poly([[1, 2], [99, 2], [99, 99], [1, 99]]),
                     "shape_type": "polygon"}]
    set_anns(bbox_changed)
    sig_bbox = w._annotations_state_signature()
    check(sig_moved != sig_bbox, "G2 bbox 变化未改变签名")
    print("   G2 bbox 变化改变签名: %s" % (sig_moved != sig_bbox))

    # ---------------- G3: 标签变化必须改变签名 ----------------
    label_changed = [{"label": "b", "bbox": [5, 5, 30, 30],
                      "polygons": poly([[1, 2], [99, 2], [99, 99], [1, 99]]),
                      "shape_type": "polygon"}]
    set_anns(label_changed)
    sig_label = w._annotations_state_signature()
    check(sig_bbox != sig_label, "G3 标签变化未改变签名")
    set_anns([])
    sig_empty = w._annotations_state_signature()
    check(sig_label != sig_empty, "G3 清空标注未改变签名")
    print("   G3 标签/数量变化改变签名: %s" % (sig_bbox != sig_label != sig_empty))

    # ---------------- G4: 可见性变化必须改变签名 ----------------
    set_anns(ann_a)
    sig_vis_before = w._annotations_state_signature()
    w.manual.hidden_indices = {0}
    sig_vis_after = w._annotations_state_signature()
    w.manual.hidden_indices = set()
    check(sig_vis_before != sig_vis_after, "G4 可见性变化未改变签名")
    print("   G4 可见性变化改变签名: %s" % (sig_vis_before != sig_vis_after))

    # ---------------- G5: 顶点移动必须真的触发刷新 ----------------
    # 注意：独立构造的 interface 没有 event_bus（由 AnnotationPlugin 注入），
    # 因此这里直接驱动 on_annotations_changed —— 被测的是它内部的去重判定，
    # 而不是事件总线本身。
    def publish(path=None, source="test"):
        handler = w.on_annotations_changed
        handler({"image_path": path if path is not None else w.current_image_path,
                 "source": source})
        app.processEvents()

    set_anns(ann_a)
    w._last_ann_path = None
    w._last_ann_sig = None
    refreshes["n"] = 0
    publish(source="baseline")
    baseline = refreshes["n"]
    # 只移动顶点，其余全同 -> 必须再刷新一次
    set_anns(moved)
    publish(source="moved")
    after_move = refreshes["n"]
    print("   G5 发布两次后的刷新次数: baseline=%d after_move=%d" % (baseline, after_move))
    check(baseline >= 1, "G5 前置条件：基线发布没有触发刷新")
    check(after_move >= baseline + 1,
          "G5 顶点移动后的发布被去重抑制（未刷新，画布/列表会停在旧几何）")

    # ---------------- G7: 同状态重复发布仍应去重 ----------------
    # 先制造一个新的签名（未缓存过），再连续发布同一状态 —— 期望恰好刷新一次
    w._last_ann_path = None
    w._last_ann_sig = None
    set_anns(ann_a)
    refreshes["n"] = 0
    for _ in range(4):
        publish(source="dup")
    dup_refreshes = refreshes["n"]
    print("   G7 同一状态连续发布 4 次的刷新次数: %d（应只刷新一次）" % dup_refreshes)
    check(dup_refreshes == 1,
          "G7 去重能力异常：同一状态连续发布刷新了 %d 次（应为 1）" % dup_refreshes)

    # ---------------- G6: 真值被替换时缓存必须作废 ----------------
    w._last_ann_path = w.current_image_path
    w._last_ann_sig = w._annotations_state_signature()
    cached_before = (w._last_ann_path, w._last_ann_sig)
    w.load_image(1, force_reload=True)
    pump(4.0, until=lambda: w.manual.session.is_annotations_loaded)
    cleared_by_switch = (w._last_ann_path, w._last_ann_sig) != cached_before
    print("   G6a 切图后派生签名缓存已作废: %s" % cleared_by_switch)
    check(cleared_by_switch, "G6a 切图未作废派生签名缓存（D33 第 2 层）")

    w._last_ann_path = w.current_image_path
    w._last_ann_sig = w._annotations_state_signature()
    snapshot = (w._last_ann_path, w._last_ann_sig)
    w.reset_annotations_refresh_state()
    cleared_by_api = (w._last_ann_path, w._last_ann_sig) != snapshot
    print("   G6b reset_annotations_refresh_state() 生效: %s" % cleared_by_api)
    check(cleared_by_api, "G6b reset_annotations_refresh_state 没有清除缓存")

    # 三个入口都必须调用它（静态检查，避免将来新增入口漏掉）
    import inspect
    src_clear = inspect.getsource(AnnotationInterface.clear_project)
    src_set = inspect.getsource(AnnotationInterface._do_set_project)
    check("reset_annotations_refresh_state" in src_clear,
          "G6c clear_project 未作废派生签名缓存")
    check("reset_annotations_refresh_state" in src_set,
          "G6c _do_set_project 未作废派生签名缓存")
    print("   G6c clear_project / _do_set_project 均已作废缓存: %s"
          % ("reset_annotations_refresh_state" in src_clear
             and "reset_annotations_refresh_state" in src_set))

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: G1~G7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
