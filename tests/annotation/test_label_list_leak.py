"""标注列表重建是否泄漏子控件 — 用真实 AnnotationInterface 测量。

被测假设（"快速切图一段时间后卡住"的候选元凶）：
  `refresh_label_list()`（annotation_interface.py:4166）用
      while count() > 0: takeItem(0)
  清空列表。Qt 语义下 `takeItem` 只解除 item 与列表的关系，**不会销毁**
  通过 `setItemWidget()` 挂上去的子控件 —— 那些控件仍是 viewport 的子对象。
  而每个 item 挂了 5 个控件（QWidget + CheckBox + QLabel + 2 个 TransparentToolButton）。
  `refresh_label_list()` 每次切图都会被调用一次
  （load_image -> load_image_annotations -> set_current_annotations -> refresh_label_list），
  于是"切 N 张图"= 泄漏 5 x 标注数 x N 个常驻控件 → 界面逐步变慢直到卡住。

断言（退出码即结论，与 test_switch_stress.py 同一约定）：
  L1 重建 30 次后 viewport 子控件数不得随重建次数增长
  L2 每次重建耗时不得随重建次数显著增长
      0 = 通过 / 1 = 断言失败 / 其他 = 异常退出

用法: python -u tests/annotation/test_label_list_leak.py [rebuilds] [annotations]
"""
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import build_fixture, isolate_state  # noqa: E402


def make_annotations(n):
    """n 个多边形标注，形状与 loader 产出的一致。"""
    anns = []
    for i in range(n):
        x = 20 + i * 15
        y = 20 + i * 15
        anns.append({
            "label": "defect_%d" % (i % 4),
            "bbox": [x, y, 40, 30],
            "polygons": [[[x, y], [x + 40, y], [x + 40, y + 30], [x, y + 30]]],
            "shape_type": "polygon",
        })
    return anns


def viewport_widget_count(interface):
    """label_list 的 viewport 下所有子控件数（泄漏规模指标）。"""
    from PySide6.QtWidgets import QWidget
    return len(interface.label_list.viewport().findChildren(QWidget))


def other_lists_widget_count(interface):
    """同类风险的其他列表（类别列表也走 takeItem 重建模式）。"""
    from PySide6.QtWidgets import QWidget
    return len(interface.category_list.viewport().findChildren(QWidget))


def main():
    rebuilds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_ann = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    w = AnnotationInterface(None, None)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.open_directory(build_fixture(state_dir))
    app.processEvents()

    # 标注装载是异步的（阶段 3a）：必须先等它落地再测量，否则基线会被随后到达的
    # 装载结果覆盖（没有标注文件 -> 清空列表），得到"基线 0 -> 之后 120"的假泄漏。
    _end = time.time() + 5
    while time.time() < _end:
        app.processEvents()
        if w.manual.session.is_annotations_loaded and not w._annotation_runner.busy():
            break
        time.sleep(0.005)

    anns = make_annotations(n_ann)

    # 第一次重建（基线）
    w.manual.set_current_annotations(anns)
    w.draw_area.update()
    t0 = time.perf_counter()
    w.refresh_label_list()
    app.processEvents()
    base_ms = (time.perf_counter() - t0) * 1000.0
    base_widgets = viewport_widget_count(w)
    base_cat = other_lists_widget_count(w)

    print("== 标注列表重建泄漏测量 ==")
    print("   标注数=%d  重建次数=%d" % (n_ann, rebuilds))
    print("   基线: viewport 子控件=%d  单次重建=%.1f ms  类别列表控件=%d"
          % (base_widgets, base_ms, base_cat))
    print("   理论期望: 每次重建后仍应约 %d 个控件（不随重建次数增长）" % base_widgets)

    samples = []
    worst_ms = base_ms
    for k in range(rebuilds):
        # 复刻每个切图的调用序列：装载标注 -> 重建列表
        w.manual.set_current_annotations(anns)
        w.draw_area.update()
        t = time.perf_counter()
        w.refresh_label_list()
        app.processEvents()
        ms = (time.perf_counter() - t) * 1000.0
        worst_ms = max(worst_ms, ms)
        samples.append((k + 1, viewport_widget_count(w), ms, other_lists_widget_count(w)))

    after_widgets = samples[-1][1]
    after_cat = samples[-1][3]

    print("\n   重建序号 | viewport 子控件 | 单次耗时ms | 类别列表控件")
    for k, wc, ms, cc in samples:
        if k in (1, 2, 5, 10, 20, 30) or k == rebuilds:
            print("   %8d | %15d | %10.1f | %12d" % (k, wc, ms, cc))

    failures = []
    growth = after_widgets - base_widgets
    print("\n== 结果 ==")
    print("   viewport 子控件: %d -> %d（增长 %+d）" % (base_widgets, after_widgets, growth))
    print("   单次重建耗时: 基线 %.1f ms -> 最差 %.1f ms" % (base_ms, worst_ms))
    print("   类别列表控件: %d -> %d" % (base_cat, after_cat))

    if growth > 0:
        per = growth / float(rebuilds)
        failures.append(
            "L1 泄漏: 每次 refresh_label_list 泄漏 %.1f 个控件（%d 次重建共泄漏 %d 个）；"
            "按每次切图调用一次计算，切 %d 张图就会多出 %d 个常驻控件"
            % (per, rebuilds, growth, 100, int(per * 100)))
    if worst_ms > base_ms * 3 and worst_ms - base_ms > 20:
        failures.append("L2 重建耗时劣化: %.1f ms -> %.1f ms" % (base_ms, worst_ms))

    if failures:
        print("   FAIL:")
        for f in failures:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: L1/L2 通过（无泄漏、耗时不劣化）")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
