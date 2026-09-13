"""撤销/重做必须与可见性同帧回滚 —— D21 回归。

被测缺陷（审计 D21）：`hidden_indices` 是**按下标寻址**的集合，而撤销栈原先只快照
`annotations`。于是：

  * `clear_annotations` 先把 `hidden_indices` 清空、再把标注置空并**只快照标注** ——
    撤销后标注列表回来了，但可见性集合停在"空"，**原来的隐藏状态永久丢失**；
  * `delete_annotation` 会让下标位移（service 用 `adjust_hidden_indices_after_deletion`
    维护），若撤销只回滚列表、不回滚集合，就会出现
    "标注列表 = 删除前、可见性下标 = 删除后" 的不自洽组合 —— 复选框勾选状态与
    实际画布显示对不上，而且下次再删会删错对象。

断言：
  H1 撤销帧必须携带可见性快照（且与标注同帧、不被后续原地修改污染）
  H2 clear_annotations + 撤销：标注恢复，**隐藏状态也恢复**
  H3 delete_annotation + 撤销：标注恢复，可见性下标回到删除前的值
  H4 重做同样回滚可见性（undo/redo 对称）
  H5 撤销越界/跨图拒绝路径仍然工作（不能因为加字段而破坏已有的 D3 保护）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_undo_visibility.py
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
        img.fill(QColor(45 + 25 * i, 95, 155))
        img.save(path)
        paths.append(path)
    return paths


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

    root = tempfile.mkdtemp(prefix="visionmind_undo_")
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

    manual = w.manual

    def load_ready(index):
        w.load_image(index, force_reload=True)
        pump(4.0, until=lambda: manual.session.is_annotations_loaded
             and manual.session.is_committed)
        manual.wait_for_pending_writes(manual.current_image_path, 5000)

    def ann(label):
        return {"label": label, "bbox": [1, 1, 8, 8], "shape_type": "rect",
                "polygons": []}

    print("== 撤销 / 可见性一致性回归（D21）==")

    # ---------------- H1: 撤销帧必须携带可见性 ----------------
    load_ready(0)
    manual.clear_undo_history()
    manual.hidden_indices = {1}
    manual.set_current_annotations([ann("a"), ann("b"), ann("c")],
                                   for_path=manual.current_image_path)
    manual.save_undo_state(manual.current_annotations, manual.current_image_path)
    frame = manual._undo_stack[-1]
    has_field = "hidden_indices" in frame
    print("   H1 撤销帧字段=%s 可见性=%s"
          % (sorted(frame.keys()), sorted(frame.get("hidden_indices") or [])))
    check(has_field, "H1 撤销帧没有携带可见性快照（D21 根因）")
    check(set(frame.get("hidden_indices") or ()) == {1},
          "H1 撤销帧里的可见性与入栈时不一致: %r" % frame.get("hidden_indices"))
    # 入栈后原地修改集合不得污染已入栈的帧
    manual.hidden_indices.add(2)
    check(set(frame.get("hidden_indices") or ()) == {1},
          "H1 入栈后的原地修改污染了帧内快照（快照必须独立拷贝）")
    manual.hidden_indices = {1}

    # ---------------- H2: clear_annotations + 撤销必须恢复隐藏状态 ----------------
    load_ready(1)
    manual.clear_undo_history()
    manual.set_current_annotations([ann("x"), ann("y"), ann("z")],
                                   for_path=manual.current_image_path)
    manual.hidden_indices = {0, 2}
    before_hidden = set(manual.hidden_indices)
    before_labels = [a["label"] for a in manual.current_annotations]

    manual.clear_annotations()
    after_clear_hidden = set(manual.hidden_indices)
    after_clear_labels = [a["label"] for a in manual.current_annotations]

    manual.undo()
    restored_labels = [a["label"] for a in manual.current_annotations]
    restored_hidden = set(manual.hidden_indices)
    print("   H2 清除前 labels=%s hidden=%s" % (before_labels, sorted(before_hidden)))
    print("   H2 清除后 labels=%s hidden=%s" % (after_clear_labels, sorted(after_clear_hidden)))
    print("   H2 撤销后 labels=%s hidden=%s" % (restored_labels, sorted(restored_hidden)))
    check(restored_labels == before_labels,
          "H2 撤销没有恢复标注列表: %s" % restored_labels)
    check(restored_hidden == before_hidden,
          "H2 撤销没有恢复隐藏状态: 期望 %s 实得 %s（D21：隐藏状态永久丢失）"
          % (sorted(before_hidden), sorted(restored_hidden)))

    # ---------------- H3: 删除 + 撤销必须回滚下标位移 ----------------
    load_ready(2)
    manual.clear_undo_history()
    manual.set_current_annotations([ann("p0"), ann("p1"), ann("p2"), ann("p3")],
                                   for_path=manual.current_image_path)
    manual.hidden_indices = {3}          # 隐藏最后一个
    before_hidden = set(manual.hidden_indices)

    manual.delete_annotation(1)          # 删除中间那个 -> 下标 3 应位移为 2
    after_del_labels = [a["label"] for a in manual.current_annotations]
    after_del_hidden = set(manual.hidden_indices)

    manual.undo()
    undo_labels = [a["label"] for a in manual.current_annotations]
    undo_hidden = set(manual.hidden_indices)
    print("   H3 删除前 hidden=%s" % sorted(before_hidden))
    print("   H3 删除后 labels=%s hidden=%s（下标位移）" % (after_del_labels, sorted(after_del_hidden)))
    print("   H3 撤销后 labels=%s hidden=%s" % (undo_labels, sorted(undo_hidden)))
    check(len(after_del_labels) == 3, "H3 前置条件：删除未生效: %s" % after_del_labels)
    check(undo_labels == ["p0", "p1", "p2", "p3"],
          "H3 撤销没有恢复被删除的标注: %s" % undo_labels)
    check(undo_hidden == before_hidden,
          "H3 撤销后可见性下标与列表不自洽: 期望 %s 实得 %s"
          % (sorted(before_hidden), sorted(undo_hidden)))

    # ---------------- H4: 重做也要回滚可见性 ----------------
    manual.redo()
    redo_labels = [a["label"] for a in manual.current_annotations]
    redo_hidden = set(manual.hidden_indices)
    print("   H4 重做后 labels=%s hidden=%s" % (redo_labels, sorted(redo_hidden)))
    check(redo_labels == after_del_labels,
          "H4 重做没有回到删除后的标注: %s" % redo_labels)
    check(redo_hidden == after_del_hidden,
          "H4 重做没有回滚可见性: 期望 %s 实得 %s"
          % (sorted(after_del_hidden), sorted(redo_hidden)))

    # ---------------- H5: 越界与跨图保护仍然有效 ----------------
    manual.clear_undo_history()
    restored, err = manual.step_history(-1, manual.current_annotations,
                                        manual.current_image_path)
    print("   H5 空栈撤销 err=%r" % err)
    check(err is not None, "H5 空栈撤销本应报错")

    current_path = manual.current_image_path
    manual.save_undo_state([ann("stale")], paths[0])
    manual.hidden_indices = {0}
    restored, err = manual.step_history(-1, manual.current_annotations,
                                        current_path)
    print("   H5 跨图撤销 err=%r 可见性=%s" % (err, sorted(manual.hidden_indices)))
    if current_path != paths[0]:
        check(err is not None, "H5 跨图撤销未被拒绝（D3 保护失效）")
        check(manual.hidden_indices == {0},
              "H5 跨图撤销被拒时不应改动可见性: %s" % sorted(manual.hidden_indices))

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: H1~H5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
