"""导航必须只有一条"可见图片"真值来源 —— 阶段 3e 回归。

被测缺陷（审计阶段 3e）：可见性原先有**两个来源**：
  * 界面 `switch_image` 从 `file_list` 隐藏状态推导（正确，会跳过被筛选隐藏的图）；
  * service `_navigate_ai` 硬编码 `set(range(len(image_files)))`（假设"没有筛选"）。

同一个"下一张"动作因此有两条真值：启用筛选后，**agent 走 next_image 会跳到被隐藏的
图片上**，而用户按方向键会正确跳过。这里把两端统一到 `visible_image_indices()`。

断言：
  N1 无筛选时：界面与 service 的可见集合一致（都等于全量）
  N2 启用筛选后：两端的可见集合仍然一致（都排除被隐藏项）
  N3 筛选后 service 导航**不再跳到隐藏图片**（这是原缺陷的直接表现）
  N4 界面按方向键与 service 导航结果相同（同一动作、同一答案）
  N5 边界行为不变：首/末张时返回 None（不越界、不循环）
  N6 静态：导航路径不得再出现"假设全可见"的硬编码集合

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_navigation_single_source.py
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

_N = 8


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

    root = tempfile.mkdtemp(prefix="visionmind_nav_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    for i in range(_N):
        img = QImage(48, 36, QImage.Format.Format_RGB32)
        img.fill(QColor(50 + 8 * i, 100, 150))
        img.save(os.path.join(d, "img%02d.png" % i))

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    def pump(seconds=1.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    manual = w.manual
    w._non_project_format = "labelme"
    w.load_image(0, force_reload=True)
    pump(4.0, until=lambda: manual.session.is_annotations_loaded)

    print("== 导航单一真值来源回归（阶段 3e）==")

    # ---------------- N1: 无筛选时两端一致 ----------------
    ui_vis = w.visible_image_indices()
    svc_vis = manual.visible_image_indices()
    print("   N1 无筛选: 界面可见=%d service 可见=%d（应都为 %d）"
          % (len(ui_vis), len(svc_vis), _N))
    check(ui_vis == svc_vis, "N1 两端可见集合不一致: %s vs %s"
          % (sorted(ui_vis), sorted(svc_vis)))
    check(len(svc_vis) == _N, "N1 无筛选时 service 可见集合应为全量")

    # ---------------- N2: 启用筛选后两端仍一致 ----------------
    # 隐藏奇数下标（模拟"按类别筛选掉一半图片"）
    hidden = {1, 3, 5, 7}
    for i in range(w.file_list.count()):
        w.file_list.item(i).setHidden(i in hidden)
    # service 侧的可见性来源：界面应用筛选时回写的 file_visibility
    manual.file_visibility = [(manual.image_files[i], i in hidden) for i in range(_N)]

    ui_vis2 = w.visible_image_indices()
    svc_vis2 = manual.visible_image_indices()
    print("   N2 筛选后: 界面可见=%s service 可见=%s（应一致）"
          % (sorted(ui_vis2), sorted(svc_vis2)))
    check(ui_vis2 == svc_vis2, "N2 筛选后两端可见集合不一致: %s vs %s"
          % (sorted(ui_vis2), sorted(svc_vis2)))
    check(svc_vis2 == {0, 2, 4, 6}, "N2 service 未排除被隐藏的图片: %s" % sorted(svc_vis2))

    # ---------------- N3: service 导航不得跳到隐藏图片 ----------------
    # 从 0 出发向后：必须到 2（而不是被隐藏的 1）
    nxt, err = manual.navigate_relative(0, 1, manual.visible_image_indices())
    print("   N3 从 0 向后导航 -> %s（应为 2，修复前会得到 1 即被隐藏的那张）" % nxt)
    check(nxt == 2, "N3 service 导航跳到了被筛选隐藏的图片: %s" % nxt)

    # 再往后：2 -> 4（跳过 3）
    nxt2, _ = manual.navigate_relative(2, 1, manual.visible_image_indices())
    check(nxt2 == 4, "N3 service 连续导航仍会落到隐藏项: %s" % nxt2)
    # 向前：0 之前没有可见项 -> None
    prv0, _ = manual.navigate_relative(0, -1, manual.visible_image_indices())
    check(prv0 is None, "N3 首张向前不该有目标: %s" % prv0)
    # 从 4 向前 -> 2
    prv, _ = manual.navigate_relative(4, -1, manual.visible_image_indices())
    check(prv == 2, "N3 向前导航结果不对: %s" % prv)

    # ---------------- N4: 界面与 service 结果一致 ----------------
    # 让界面停在 0，按"下一张"应落到 2
    w.file_list.setCurrentRow(0)
    w.load_image(0, force_reload=True)
    pump(4.0, until=lambda: manual.session.is_annotations_loaded)
    w.switch_image(1)                 # 界面入口
    pump(4.0, until=lambda: manual.session.is_annotations_loaded)
    ui_target = w.file_list.currentRow()
    svc_target, _ = manual.navigate_relative(0, 1, manual.visible_image_indices())
    print("   N4 同一个动作：界面 -> %s，service -> %s（应相同）" % (ui_target, svc_target))
    check(ui_target == svc_target,
          "N4 界面与 service 对同一个'下一张'给出了不同答案: %s vs %s"
          % (ui_target, svc_target))
    check(ui_target == 2, "N4 界面导航未跳过隐藏图片: %s" % ui_target)

    # ---------------- N5: 边界行为 ----------------
    last_visible = 6
    end_target, _ = manual.navigate_relative(last_visible, 1, manual.visible_image_indices())
    print("   N5 末张(%d)向后 -> %s（应为 None，不越界不循环）" % (last_visible, end_target))
    check(end_target is None, "N5 末张向后越界了: %s" % end_target)

    # ---------------- N6: 静态检查 ----------------
    import inspect
    src_ai = inspect.getsource(manual._navigate_ai)
    check("range(len(self.image_files))" not in src_ai,
          "N6 _navigate_ai 仍硬编码'假设全可见'的集合")
    check("visible_image_indices" in src_ai,
          "N6 _navigate_ai 没有使用唯一真值来源")
    src_ui = inspect.getsource(AnnotationInterface.switch_image)
    check("visible_image_indices" in src_ui,
          "N6 界面 switch_image 没有使用唯一真值来源")
    print("   N6 静态检查：两端都经 visible_image_indices()  OK")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: N1~N6 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
