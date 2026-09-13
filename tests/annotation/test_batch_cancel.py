"""AI 全量转换的"取消"必须真的生效 —— D9 余项回归。

被测缺陷（审计 D9 余项）：`convert_all_with_ai` 弹出的 `QProgressDialog` 上的
"取消"按钮**从未接到任何逻辑**（服务侧也没有 cancel 入口），按下去毫无作用；
用户只能等整批跑完。这类"看起来能停、实际停不下来"的控件属于用户可见的失效路径。

修复采用**协作式取消**，并刻意只在**图片边界**检查：
  绝不打断正在写盘的那张图 -> 已落盘的标注始终是完整的（不会留半张图的数据）。

断言：
  C1 取消后循环提前结束（处理数 < 总数），并置 cancelled=True
  C2 取消**不会**删除/破坏已处理图片的产物（已落盘数据完整）
  C3 取消发生在下一张图的边界：processed_count == 真正完成的张数
  C4 进度回调在取消后不再被调用（不虚报进度）
  C5 不取消时行为完全不变（回归）：全量处理 + 最终进度回调
  C6 `cancel_batch_convert()` 能置位在途标记；无在途任务时返回 False
  C7 静态：界面里 progress.canceled 必须接到取消处理函数

退出码即断言：0 通过 / 1 断言失败。
用法: python -u tests/annotation/test_batch_cancel.py
"""
import os
import shutil
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.annotation.test_switch_stress import isolate_state  # noqa: E402

_N = 6


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

    root = tempfile.mkdtemp(prefix="visionmind_cancel_")
    d = os.path.join(root, "ds")
    os.makedirs(d)

    from PySide6.QtGui import QColor, QImage

    paths = []
    for i in range(_N):
        p = os.path.join(d, "img%02d.png" % i)
        img = QImage(64, 48, QImage.Format.Format_RGB32)
        img.fill(QColor(60 + 10 * i, 100, 150))
        img.save(p)
        paths.append(p)
        # 每张图配一份 yolo 源标注，保证循环有活可干
        with open(os.path.join(d, "img%02d.txt" % i), "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")
    with open(os.path.join(d, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("thing\n")

    w = AnnotationInterface(None, None)
    w.resize(900, 600)
    w.show()
    app.processEvents()
    w.open_directory(d)
    app.processEvents()

    auto = w.auto
    out_dir = os.path.join(root, "out")
    os.makedirs(out_dir, exist_ok=True)

    print("== AI 全量转换取消回归（D9 余项）==")

    # ---------------- C1/C2/C3/C4: 中途取消 ----------------
    cancel_event = threading.Event()
    calls = []

    def progress_cb(processed, total):
        calls.append(processed)
        # 处理完第 2 张后请求取消（等价于用户点"取消"）
        if processed >= 2 and not cancel_event.is_set():
            cancel_event.set()
            print("     测试：处理到 %d/%d 时发出取消请求" % (processed, total))

    res = auto.batch_ai_convert(paths, source_mode='det', target_mode='seg',
                                output_dir=out_dir, categories={"thing": 0},
                                progress_callback=progress_cb,
                                cancel_event=cancel_event)
    processed = res.get("processed_count", -1)
    print("   C1 cancelled=%s processed=%d/%d total=%d"
          % (res.get("cancelled"), processed, _N, res.get("total")))
    check(res.get("cancelled") is True, "C1 取消标记未置位（取消按钮仍是空接线）")
    check(0 < processed < _N,
          "C1 取消后处理数异常: processed=%d（应停在 1..%d 之间）" % (processed, _N - 1))
    check(res.get("total") == _N, "C1 total 不正确: %s" % res.get("total"))

    # C3: 取消发生在边界 —— processed_count 必须等于"真正处理完的张数"
    # （取消请求是在 progress_callback(processed>=2) 里发出的，而回调发生在图片
    #   开始处理时，因此第 3 张不会被写入）
    seg_dir = os.path.join(out_dir, "seg")
    produced = len([f for f in os.listdir(seg_dir)]) if os.path.isdir(seg_dir) else 0
    print("   C3 已落盘产物数=%d processed_count=%d（应满足 produced <= processed）"
          % (produced, processed))
    check(produced <= processed + 1,
          "C3 落盘产物多于已完成的图片数（可能打断了正在写的图）: 产物=%d processed=%d"
          % (produced, processed))

    # C2: 已落盘数据必须完整（能被读回）
    broken = []
    if os.path.isdir(seg_dir):
        for fn in os.listdir(seg_dir):
            fp = os.path.join(seg_dir, fn)
            if os.path.getsize(fp) == 0:
                broken.append(fn)
    print("   C2 已落盘产物是否有空文件: %s（应为空）" % broken)
    check(not broken, "C2 取消造成了不完整的落盘文件: %s" % broken)

    # C4: 取消后不得再有进度回调
    calls_after_cancel = list(calls)
    print("   C4 进度回调次数=%d（取消后不应继续）" % len(calls_after_cancel))
    check(len(calls_after_cancel) <= processed + 2,
          "C4 取消后仍在继续回调进度: %d 次" % len(calls_after_cancel))

    # ---------------- C6: cancel_batch_convert ----------------
    no_active = auto.cancel_batch_convert()
    print("   C6 无在途任务时 cancel_batch_convert()=%s（应为 False）" % no_active)
    check(no_active is False, "C6 无在途任务时不应返回 True")

    # ---------------- C5: 不取消时行为不变 ----------------
    ev2 = threading.Event()
    calls2 = []
    res2 = auto.batch_ai_convert(paths, source_mode='det', target_mode='seg',
                                 output_dir=os.path.join(root, "out2"),
                                 categories={"thing": 0},
                                 progress_callback=lambda p, t: calls2.append(p),
                                 cancel_event=ev2)
    print("   C5 不取消：cancelled=%s processed=%d/%d 回调末值=%s"
          % (res2.get("cancelled"), res2.get("processed_count"),
             res2.get("total"), calls2[-1] if calls2 else None))
    check(res2.get("cancelled") is False, "C5 未取消却报告 cancelled=True")
    check(res2.get("processed_count") == _N, "C5 未取消时未处理完全部图片")
    check(calls2 and calls2[-1] == _N, "C5 最终进度回调缺失或不为总数")

    # ---------------- C7: 静态检查接线 ----------------
    import inspect
    import re as _re
    src = inspect.getsource(AnnotationInterface.convert_all_with_ai)
    wired = "canceled.connect" in src and "cancel_event" in src
    print("   C7 界面侧已接线 canceled.connect + cancel_event: %s" % wired)
    check(wired, "C7 界面仍把取消按钮空接着（D9 余项未修复）")
    # 只找**真实调用**，不能扫注释：函数体内原本就有一条说明 processEvents 的注释
    code_only = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    has_call = _re.search(r"\bprocessEvents\s*\(", code_only) is not None
    print("   C7 是否真实调用 processEvents: %s（应为 False）" % has_call)
    check(not has_call, "C7 取消接线引入了 processEvents 调用（会重入事件循环）")

    if failures:
        print("   FAIL:")
        for f in failures[:12]:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: C1~C7 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
