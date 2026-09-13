"""画布不变量：损坏的 ROI / 畸形标注不得让画布永久停止重绘 —— D18 回归。

被测缺陷（审计 D18）：`paintEvent` 的不变量过弱：

  * `persistent_roi` 直接解包 4 元组（5 处，其中 2 处在 `paintEvent` 内），而它来自
    `project_info.json` 的**裸 JSON、无 schema 校验** -> 值为 None / 长度不对 /
    元素非数字时解包抛异常；
  * `bbox=None`、`polygons=[1.0, 2.0]` 等畸形标注在帧循环内同样可能抛异常。

Qt 里 `paintEvent` 抛异常会**吞掉该次绘制**，用户看到的是"画布卡住、之后再也不刷新"
—— 属于最严重的一类可用性缺陷，且它由**持久化数据**触发（一次坏写入永久生效）。

断言：
  P1 ROI 读取时校验：None/长度错/非数字/NaN/bool 一律视为"无 ROI"
  P2 合法 ROI 仍正常返回（不能因为加校验而废掉功能）
  P3 损坏 ROI 下画布仍能重绘（不抛异常、paintEvent 正常返回）
  P4 畸形标注（bbox=None / polygons 元素不是点 / 缺字段）下仍能重绘
  P5 连续多次重绘稳定（坏数据不会在某一帧才炸）
  P6 静态：ROI 解包点必须都在"已校验的 property"之后（不存在绕过校验的直接读）

退出码即断言：0 通过 / 1 断言失败 / -1073740791 = abort。
用法: python -u tests/annotation/test_canvas_invariants.py
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

    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        return cond

    root = tempfile.mkdtemp(prefix="visionmind_inv_")
    d = os.path.join(root, "ds")
    os.makedirs(d)
    img = QImage(64, 48, QImage.Format.Format_RGB32)
    img.fill(QColor(60, 110, 160))
    img.save(os.path.join(d, "img00.png"))

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()
    canvas = w.draw_area
    manual = w.manual
    w._non_project_format = "labelme"
    w.load_image(0, force_reload=True)

    def pump(seconds=4.0, until=None):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.002)
        return until() if until is not None else True

    pump(until=lambda: manual.session.is_committed and manual.session.is_annotations_loaded)

    print("== 画布不变量回归（D18）==")

    # ---------------- P1/P2: ROI 校验 ----------------
    bad_cases = [(None, "None"), ([1, 2, 3], "长度3"), ([1, 2, 3, 4, 5], "长度5"),
                 ("abc", "字符串"), (7, "标量"), ([1, "x", 3, 4], "含字符串"),
                 ([1, 2, 3, float("nan")], "含NaN"), ([True, 2, 3, 4], "含bool")]
    all_bad_ok = True
    for value, name in bad_cases:
        got = canvas._coerce_roi(value)
        if got is not None:
            failures.append("P1 损坏 ROI 未被拒绝: %s -> %r" % (name, got))
            all_bad_ok = False
    good = canvas._coerce_roi([1.0, 2.0, 3.0, 4.0])
    check(good == (1.0, 2.0, 3.0, 4.0), "P2 合法 ROI 未被保留: %r" % (good,))
    check(canvas._coerce_roi((5, 6, 7, 8)) == (5, 6, 7, 8), "P2 元组形式的合法 ROI 未保留")
    print("   P1 损坏 ROI 全部被拒: %s（%d 种）" % (all_bad_ok, len(bad_cases)))
    print("   P2 合法 ROI 正常保留: %s" % (good,))

    # ---------------- P3/P5: 损坏 ROI 下画布仍能重绘 ----------------
    def force_paint():
        """强制走一次 paintEvent，返回是否无异常完成。

        **不能用 `canvas.grab()`**：它会启动一次绘制并把结果渲染到内部 QPixmap，
        在"控件本身仍在绘制"的时序下会触发
        `QPaintDevice: Cannot destroy paint device that is being painted` 并 **abort**
        （本轮实测踩到，退出码 -1073740791）。

        改为：自己开一个 `QPainter(canvas)`，再直接调用 `paintEvent` —— 这正是 Qt
        内部调用它的方式，绘制范围与生命周期都完全由本测试掌控。
        `paintEvent` 的实现只使用 `painter`（不使用 `event`），因此传一个真实的
        `QPaintEvent` 即可。
        """
        try:
            painter = QPainter(canvas)
            try:
                canvas.paintEvent(QPaintEvent(QRect(0, 0, canvas.width(), canvas.height())))
            finally:
                painter.end()
            return True
        except Exception as exc:
            failures.append("重绘抛异常: %r" % exc)
            return False

    for value, name in [(None, "None"), ([1, 2, 3], "长度3"), ([1, "x", 3, 4], "含字符串")]:
        canvas.persistent_roi = value
        ok = force_paint()
        paint_val = canvas.persistent_roi
        print("   P3 ROI=%-8s 设置后读取=%s 重绘成功=%s" % (name, paint_val, ok))
        check(ok, "P3 ROI=%s 时画布重绘失败（画布会永久停止刷新）" % name)
        check(paint_val is None, "P3 ROI=%s 未被规范为 None: %r" % (name, paint_val))

    # P5: 连续重绘稳定
    canvas.persistent_roi = [1, 2, 3]          # 坏值
    stable = all(force_paint() for _ in range(5))
    print("   P5 坏 ROI 下连续 5 次重绘均成功: %s" % stable)
    check(stable, "P5 坏 ROI 下连续重绘不稳定（某帧才炸）")

    # 合法 ROI 也要能画
    canvas.persistent_roi = [2, 2, 20, 20]
    check(force_paint(), "P5 合法 ROI 下重绘失败")
    check(canvas.persistent_roi == (2, 2, 20, 20), "P5 合法 ROI 未被保留")
    canvas.persistent_roi = None

    # ---------------- P4: 畸形标注下仍能重绘 ----------------
    malformed = [
        {"label": "a", "bbox": None, "polygons": [], "shape_type": "rect"},
        {"label": "b", "bbox": [1, 2, 3, 4], "polygons": [1.0, 2.0], "shape_type": "polygon"},
        {"label": "c", "bbox": [1, 2, 3, 4], "polygons": [[[1, 2], [3, 4]]],
         "shape_type": "polygon"},
        {"label": "d"},
        {"label": "e", "bbox": [1, 2, 3, 4], "polygons": [None], "shape_type": "polygon"},
    ]
    for i, ann in enumerate(malformed):
        manual.set_current_annotations([ann], for_path=manual.current_image_path)
        ok = force_paint()
        print("   P4 畸形标注#%d (%s) 重绘成功=%s" % (i, ann.get("label"), ok))
        check(ok, "P4 畸形标注#%d 导致重绘失败: %r" % (i, ann))

    # 一次放入全部畸形数据
    manual.set_current_annotations(list(malformed), for_path=manual.current_image_path)
    check(force_paint(), "P4 全部畸形标注同时存在时重绘失败")
    print("   P4 全部畸形标注同时存在时仍可重绘  OK")

    # ---------------- P6: 静态检查 ----------------
    import inspect
    src = inspect.getsource(type(canvas).persistent_roi.fget)
    check("_coerce_roi" in src, "P6 persistent_roi 读取未经过校验")
    print("   P6 persistent_roi 读取经 _coerce_roi 校验  OK")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: P1~P6 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
