"""无 processEvents 重入 — 静态不变量 + 行为断言。

被测缺陷（审计 P1）：`annotation_interface.py` 有 7 处 `QApplication.processEvents()`，
根源都是"长任务在 GUI 线程同步执行"——不让出事件循环，进度条就不重绘、取消按钮就
点不动。而在**状态变更处理器内部**重入事件循环，会让用户在批量标注进行中继续触发
切图/保存/删除，界面状态与磁盘状态随之错乱。

修复方式：批量标注搬进 BackgroundTaskRunner（后台线程），进度走事件总线；
其余两处（不可取消的进度对话框）改为只 `repaint()` 该控件 —— 同步重绘但**不派发
任何输入/定时器/网络事件**，因此不存在重入。

断言：
  N1 静态：`plugins/annotation/` 下不得再出现 `processEvents(` 调用（注释不算）
  N2 行为：批量运行期间 GUI 事件循环必须照常运转（定时器能在 busy 期间触发）
          —— 若批量跑在 GUI 线程上，事件循环会被阻塞，此断言必然失败
  N3 取消：mid-batch 取消必须真的停下（processed < total，canceled=True）
  N4 并发：批量进行中第二次发起必须被拒绝
  N5 收尾：批量结束后进度对话框必须关闭，结果通过回调交付

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_no_processevents.py
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

_IMG = os.path.join(_ROOT, "resources", "img.png")
_PLUGIN_DIR = os.path.join(_ROOT, "plugins", "annotation")
_IMAGE_COUNT = 6
_PER_IMAGE_S = 0.025


def strip_comment(line):
    """去掉行内注释（足够用于本项静态检查）。"""
    idx = line.find("#")
    return line if idx < 0 else line[:idx]


def find_processevents_calls():
    """返回 [(relpath, lineno, text)] —— 排除注释与纯文字提及。"""
    hits = []
    for dirpath, _dirnames, filenames in os.walk(_PLUGIN_DIR):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, _ROOT)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    code = strip_comment(line)
                    if "processEvents(" in code:
                        hits.append((rel, i, line.strip()))
    return hits


def build_fixture(root):
    d = os.path.join(root, "ds")
    os.makedirs(d, exist_ok=True)
    names = []
    for i in range(_IMAGE_COUNT):
        name = "img%02d.png" % i
        shutil.copy(_IMG, os.path.join(d, name))
        names.append(os.path.join(d, name))
    return d, names


def main():
    state_dir = os.path.join(tempfile.gettempdir(), "visionmind_test_state")
    shutil.rmtree(state_dir, ignore_errors=True)
    isolate_state(state_dir)

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.interfaces.annotation_interface import AnnotationInterface

    failures = []

    # ---------------- N1: 静态不变量 ----------------
    hits = find_processevents_calls()
    print("== 无 processEvents 重入回归 ==")
    if hits:
        for rel, lineno, text in hits:
            failures.append("N1 %s:%d 仍有 processEvents 调用: %s" % (rel, lineno, text))
    else:
        print("   N1 静态检查: plugins/annotation 下 0 处 processEvents 调用  OK")

    w = AnnotationInterface(None, None)
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    d, images = build_fixture(tempfile.mkdtemp(prefix="visionmind_ppe_"))
    w.open_directory(d)
    app.processEvents()

    # 假标注器：每张图耗时 25ms，并记录被处理的图片
    calls = []

    def fake_annotate_by_current(**kwargs):
        time.sleep(_PER_IMAGE_S)
        calls.append(kwargs.get("image_path"))
        return {"status": "success", "new_categories": [], "annotations": []}

    w.auto.annotate_by_current = fake_annotate_by_current

    st = {"ticks": 0, "ticks_while_busy": 0, "cancel_at_call": None, "t0": None}

    def tick():
        st["ticks"] += 1
        if w._batch_runner.busy():
            st["ticks_while_busy"] += 1
        # 处理到第 2 张之后点"取消"
        if st["cancel_at_call"] is None and len(calls) >= 2:
            st["cancel_at_call"] = len(calls)
            w._on_batch_progress_canceled()

    ticker = QTimer()
    ticker.timeout.connect(tick)
    ticker.start(10)

    # ---------------- N2/N4: 立即返回 + 拒绝并发 ----------------
    st["t0"] = time.perf_counter()
    first = w.one_click_by_all(mode="text", text="probe")
    elapsed = (time.perf_counter() - st["t0"]) * 1000.0
    print("   N2 发起返回耗时 %.1f ms -> %r" % (elapsed, first))
    if not (isinstance(first, str) and first.startswith("批量标注已在后台开始")):
        failures.append("N2 发起未返回'已在后台开始': %r" % first)
    if elapsed > _PER_IMAGE_S * 1000:
        failures.append("N2 发起耗时 %.0fms 超过单图耗时，疑似仍在 GUI 线程同步执行" % elapsed)
    if not w._batch_runner.busy():
        failures.append("N2 发起后后台任务未处于忙碌状态")
    # 进度对话框应当已由 start 事件弹出
    if w._batch_progress_dialog is None:
        print("   注意: 进度对话框尚未建立（start 事件可能还在队列里）")

    second = w.one_click_by_all(mode="text", text="probe")
    print("   N4 并发第二次发起 -> %r" % second)
    if not (isinstance(second, str) and second.startswith("错误")):
        failures.append("N4 批量进行中第二次发起未被拒绝: %r" % second)

    # ---------------- 等后台任务收尾 ----------------
    deadline = time.time() + 15
    while w._batch_runner.busy() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.005)
    ticker.stop()

    result = w._batch_result or {}
    print("   N2 事件循环: 共 %d 次 tick，其中批量运行期间 %d 次"
          % (st["ticks"], st["ticks_while_busy"]))
    print("   N3 取消点=处理到第 %s 张，结果 processed=%s total=%s canceled=%s"
          % (st["cancel_at_call"], result.get("processed"), result.get("total"),
             result.get("canceled")))
    print("   N5 进度对话框已关闭=%s" % (w._batch_progress_dialog is None))

    # N2: 批量运行期间事件循环必须照常运转
    if st["ticks_while_busy"] < 1:
        failures.append("N2 批量运行期间事件循环一次都没运转 —— 说明批量仍占用 GUI 线程")
    # N3: 取消必须生效
    if not result:
        failures.append("N3 未收到批量结果（回调未交付）")
    else:
        if not result.get("canceled"):
            failures.append("N3 取消未生效：canceled=%r" % result.get("canceled"))
        if result.get("processed", 0) >= _IMAGE_COUNT:
            failures.append("N3 取消未生效：仍处理了全部 %s 张" % result.get("processed"))
        if len(calls) >= _IMAGE_COUNT:
            failures.append("N3 取消后标注器仍被调用 %d 次" % len(calls))
    # N5: 对话框收尾
    if w._batch_progress_dialog is not None:
        failures.append("N5 批量结束后进度对话框未关闭")

    if failures:
        print("   FAIL:")
        for f in failures:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: N1/N2/N3/N4/N5 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
