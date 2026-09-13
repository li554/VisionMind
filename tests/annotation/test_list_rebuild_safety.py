"""列表重建不得发生在子控件的信号发射栈上 —— D16 回归。

被测缺陷（审计 D16）：

1. `load_directory_images()` 里 `search_box.clear()` **未 blockSignals** ——
   `clear()` 必然发射 `textChanged`，于是在"目录刚重建、currentRow 尚未确定"时
   启动一次搜索/去抖，状态被并发改写（实测：`handler('')` 与 `textChanged('')`
   都被触发）。
2. 列表重建（`refresh_label_list`，会销毁并重建列表行及其上的复选框/按钮）原先直接
   在信号槽里执行 —— 若该槽由**列表行上的子控件**发射触发，就等于在**发射栈上
   销毁发起控件**，Qt 下是未定义行为。

断言：
  T1 `load_directory_images` 期间 `search_box.clear()` 不得发射 textChanged
  T2 重建期间 file_list 的信号也被屏蔽（不得观察到 itemChanged 等）
  T3 非信号栈内调用 request_list_rebuild 同步执行（不引入延迟）
  T4 信号栈内调用 request_list_rebuild 必须**延后**（发射栈展开后才重建）
  T5 重建区内部（_list_rebuilding）再请求重建不嵌套延后，直接执行
  T6 静态：load_directory_images 的 clear() 处于 blockSignals 之间；
      删除/类别切换路径经 request_list_rebuild

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_list_rebuild_safety.py
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


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication, QPushButton

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_d16_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    for i in range(5):
        img = QImage(48, 36, QImage.Format.Format_RGB32)
        img.fill(QColor(50 + 10 * i, 100, 150))
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
        return until() if until is not None else False

    w._non_project_format = "labelme"
    w.load_image(0, force_reload=True)
    pump(4.0, until=lambda: w.manual.session.is_annotations_loaded)

    print("== 列表重建安全性回归（D16）==")

    # ---------------- T1: 目录加载期间 search_box 不得发射 ----------------
    seen = []
    w.search_box.textChanged.connect(lambda t: seen.append(t))
    w.search_box.setText("img0")          # 先放点内容进去
    app.processEvents()
    seen.clear()

    w.load_directory_images(d)
    app.processEvents()
    print("   T1 load_directory_images 期间 search_box.textChanged 次数=%d（应为 0）"
          % len(seen))
    check(not seen, "T1 清空搜索框时仍发射 textChanged: %r（会在重建中启动搜索）" % seen)
    check(w.search_box.text() == "", "T1 搜索框未被清空")

    # ---------------- T2: 重建期间 file_list 信号被屏蔽 ----------------
    # 注意范围：只统计**列表重建本身**产生的信号，不包含重建之后
    # `load_image -> file_list.setCurrentRow()` 引发的"选择已改变" ——
    # 后者是导航的正常结果（栈已确认为 setCurrentRow），不属于 D16。
    fl_signals = []
    w.file_list.itemSelectionChanged.connect(lambda: fl_signals.append("sel"))
    w.file_list.itemChanged.connect(lambda it: fl_signals.append("changed"))
    before = w.file_list.count()

    rebuild_signals = []
    real_refresh = w.refresh_label_list

    def traced_refresh():
        # 只在这个窗口内记账：列表重建区
        start = len(fl_signals)
        real_refresh()
        rebuild_signals.extend(fl_signals[start:])

    w.refresh_label_list = traced_refresh
    try:
        w.load_directory_images(d)
        app.processEvents()
    finally:
        w.refresh_label_list = real_refresh
    print("   T2 列表重建期间 file_list 信号次数=%d（应为 0，行数 %d -> %d）"
          % (len(rebuild_signals), before, w.file_list.count()))
    check(not rebuild_signals,
          "T2 重建期间发射了 file_list 信号: %r（会打到外部 -> 在发射栈上重建）"
          % rebuild_signals)

    # ---------------- T3: 非信号栈内同步执行 ----------------
    hits = []
    w.request_list_rebuild(lambda: hits.append("sync"))
    app.processEvents()
    print("   T3 非信号栈内同步执行: %s" % hits)
    check(hits == ["sync"], "T3 非信号栈内被延后了（会引入无谓延迟）: %s" % hits)

    # ---------------- T4: 信号栈内必须延后 ----------------
    # 用一个真实按钮：它的 clicked 槽里请求重建 —— 等价于"列表行上的子控件发射时重建"
    btn = QPushButton("probe", w)
    order = []

    def on_clicked():
        order.append("emitter_before_rebuild")
        # 该调用发生在按钮的发射栈内 -> 必须显式声明 from_signal=True 才会延后
        # （不能用 self.sender() 自动判定：lambda/嵌套函数里 sender() 返回 None）
        w.request_list_rebuild(lambda: order.append("rebuild"), from_signal=True)
        order.append("emitter_after_request")

    btn.clicked.connect(on_clicked)
    btn.click()
    # 只跑一轮事件循环：确保看到的是"延后一次"而不是"永远不执行"
    app.processEvents()
    print("   T4 信号栈内顺序=%s" % order)
    check(order[:2] == ["emitter_before_rebuild", "emitter_after_request"],
          "T4 重建在发射栈内被同步执行了（会在发射栈上销毁发起控件）: %s" % order)
    check("rebuild" in order, "T4 延后的重建从未执行: %s" % order)
    check(order.index("rebuild") > order.index("emitter_after_request"),
          "T4 重建执行时机不对: %s" % order)

    # ---------------- T5: 重建区内请求不嵌套延后 ----------------
    inner = []

    def outer():
        def nested():
            inner.append("nested")
        w.request_list_rebuild(nested)
        inner.append("outer_done")

    w.request_list_rebuild(outer)
    app.processEvents()
    print("   T5 重建区内嵌套请求顺序=%s" % inner)
    check(inner == ["nested", "outer_done"] or inner == ["outer_done", "nested"],
          "T5 重建区内的嵌套请求行为异常: %s" % inner)
    check("nested" in inner, "T5 重建区内的嵌套重建没有执行")

    # ---------------- T6: 静态检查 ----------------
    import inspect
    src = inspect.getsource(AnnotationInterface.load_directory_images)
    clear_at = src.find("self.search_box.clear()")
    block_before = src.rfind("blockSignals(True)", 0, clear_at)
    unblock_after = src.find("blockSignals(False)", clear_at)
    print("   T6 clear() 前后均有 blockSignals: %s"
          % (block_before != -1 and unblock_after != -1))
    check(block_before != -1 and unblock_after != -1,
          "T6 load_directory_images 的 search_box.clear() 未处于 blockSignals 之间")

    src_del = inspect.getsource(AnnotationInterface._on_delete_shortcut)
    print("   T6 删除路径经 request_list_rebuild: %s" % ("request_list_rebuild" in src_del))
    check("request_list_rebuild" in src_del,
          "T6 删除路径仍直接 refresh_label_list()（在发射栈上重建）")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: T1~T6 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
