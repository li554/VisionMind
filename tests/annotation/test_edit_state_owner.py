"""编辑态（撤销栈/可见性/副标注）必须并入唯一状态所有者 —— 收尾回归。

被测问题（"唯一状态所有者"的最后一处缺口）：`hidden_indices`、`_undo_stack`、
`_redo_stack`、`secondary_annotations`、`secondary_annotation_map` 原先各自存在于
`ManualAnnotationService`，与 `ImageSession` 并列成为**第二份状态**。它们全都是
**按图作用域**的：

  * 撤销栈 frame 记录某张图的标注 + 可见性，跨图存活就是 D3（把上一张图的标注
    写进当前图）；
  * `hidden_indices` 是**按下标寻址**的集合，与当前标注列表同生共死（D21：只回滚
    列表不回滚它会出现"列表=删除前、可见性=删除后"的不自洽）；
  * 副标注是当前图的对照标注，换图后必须整体失效。

断言：
  S1 service 上**不再**存在这些名字的实例属性（没有第二份真值）
  S2 写入 service 的属性会落到 session（读回一致）
  S3 对撤销栈的就地变更（append/clear）作用于 **session 的同一个列表对象**
  S4 界面层的 `_undo_stack`/`_redo_stack` 路由到同一个列表
  S5 换图（begin_load）必须清空全部编辑态
  S6 session.clear() 同样清空全部编辑态
  S7 撤销/重做跨图拒绝仍然有效（D3 保护未被削弱）

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_edit_state_owner.py
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

_EDIT_NAMES = ("hidden_indices", "_undo_stack", "_redo_stack",
               "secondary_annotations", "secondary_annotation_map")


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_owner_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    for i in range(3):
        img = QImage(48, 36, QImage.Format.Format_RGB32)
        img.fill(QColor(60 + 10 * i, 100, 150))
        img.save(os.path.join(d, "img%02d.png" % i))

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    m = w.manual
    w._non_project_format = "labelme"
    w.load_image(0, force_reload=True)

    def pump(seconds=4.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else False

    pump(until=lambda: m.session.is_annotations_loaded)

    print("== 编辑态唯一所有者回归（收尾）==")

    # ---------------- S1: 没有第二份真值 ----------------
    dups = [n for n in _EDIT_NAMES if n in vars(m)]
    print("   S1 service 上的独立副本=%s（应为空）" % dups)
    check(not dups, "S1 这些状态在 service 上仍有副本: %s" % dups)

    # ---------------- S2: 写入落到 session ----------------
    m.hidden_indices = {1, 3}
    ok_hidden = m.session.hidden_indices == {1, 3} and m.hidden_indices == {1, 3}
    m.secondary_annotations = [{"label": "ref"}]
    ok_sec = m.session.secondary_annotations == [{"label": "ref"}]
    m.secondary_annotation_map = {0: {"label": "ref"}}
    ok_map = m.session.secondary_annotation_map == {0: {"label": "ref"}}
    print("   S2 hidden=%s secondary=%s map=%s"
          % (ok_hidden, ok_sec, ok_map))
    check(ok_hidden, "S2 hidden_indices 写入未落到 session")
    check(ok_sec, "S2 secondary_annotations 写入未落到 session")
    check(ok_map, "S2 secondary_annotation_map 写入未落到 session")

    # ---------------- S3: 就地变更作用于同一对象 ----------------
    m.session.undo_stack.clear()
    before_obj = m._undo_stack
    m._undo_stack.append({"annotations": [{"label": "a"}], "image_path": m.current_image_path})
    same = m._undo_stack is m.session.undo_stack
    grew = len(m.session.undo_stack) == 1
    m._redo_stack.clear()
    m._redo_stack.append({"annotations": [], "image_path": m.current_image_path})
    grew_r = len(m.session.redo_stack) == 1
    print("   S3 同一列表对象=%s append 生效=%s redo append=%s"
          % (same, grew, grew_r))
    check(same, "S3 service 与 session 的撤销栈不是同一个列表（仍有两份）")
    check(grew, "S3 对撤销栈的 append 未反映到 session")
    check(grew_r, "S3 对重做栈的 append 未反映到 session")

    # ---------------- S4: 界面路由到同一列表 ----------------
    routed = (w._undo_stack is m.session.undo_stack
              and w._redo_stack is m.session.redo_stack)
    print("   S4 界面路由到同一列表=%s" % routed)
    check(routed, "S4 界面层的撤销/重做栈与 session 不是同一对象")

    # ---------------- S5: 换图清空编辑态 ----------------
    m.hidden_indices = {2}
    m.session.undo_stack.append({"annotations": [], "image_path": "img00.png"})
    m.secondary_annotations = [{"label": "x"}]
    m.session.begin_load(os.path.join(d, "img01.png"))
    cleared = (not m.session.hidden_indices and not m.session.undo_stack
               and not m.session.redo_stack and not m.session.secondary_annotations
               and not m.session.secondary_annotation_map)
    print("   S5 换图后 hidden=%s undo=%d redo=%d sec=%d map=%d"
          % (sorted(m.session.hidden_indices), len(m.session.undo_stack),
             len(m.session.redo_stack), len(m.session.secondary_annotations),
             len(m.session.secondary_annotation_map)))
    check(cleared, "S5 换图未清空编辑态（会跨图泄漏）")

    # ---------------- S6: clear() 清空 ----------------
    m.hidden_indices = {1}
    m.session.undo_stack.append({"annotations": []})
    m.session.redo_stack.append({"annotations": []})
    m.secondary_annotations = [{"label": "y"}]
    m.session.clear()
    cleared2 = (not m.session.hidden_indices and not m.session.undo_stack
                and not m.session.redo_stack and not m.session.secondary_annotations)
    print("   S6 clear() 后 hidden=%s undo=%d sec=%d"
          % (sorted(m.session.hidden_indices), len(m.session.undo_stack),
             len(m.session.secondary_annotations)))
    check(cleared2, "S6 session.clear() 未清空编辑态")

    # ---------------- S7: 跨图撤销拒绝仍有效（D3）----------------
    w.load_image(0, force_reload=True)
    pump(until=lambda: m.session.is_annotations_loaded)
    m.session.undo_stack.clear()
    m.session.undo_stack.append({"annotations": [{"label": "old", "bbox": [1, 2, 3, 4]}],
                                 "hidden_indices": set(),
                                 "image_path": os.path.join(d, "img02.png")})
    res = m.undo()
    err = res[1] if isinstance(res, tuple) and len(res) > 1 else None
    has_err = bool(err)
    print("   S7 跨图撤销 err=%s" % (str(err)[:60] if err else None))
    check(has_err, "S7 跨图撤销未被拒绝（D3 保护被削弱）")

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
