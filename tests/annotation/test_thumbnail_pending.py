"""缩略图在途集合必须"必定收敛" — D19 回归。

被测缺陷：`_ThumbnailTask.run()` 原先在三个失败分支（文件不存在 / 解码为空 /
异常）**不发射任何信号就 return**，而 `_pending` 的清除只发生在接收端的
`_on_task_done` 里。于是这些路径会永久留在"在途"集合中，此后 `request()`
每次都因 `path in self._pending` 直接返回 -> **永不重试**；`clear_pending()`
全仓无调用者，也没有别的复位手段。

断言：
  P1 请求一个不存在的文件后，在途集合必须回到 0（失败也必须"完成"）
  P2 同一路径必须可以被**再次请求**（证明没有被永久占位）
  P3 正常文件也必须回到 0

退出码即断言：0 通过 / 1 断言失败 / 其他异常。
用法: python -u tests/annotation/test_thumbnail_pending.py
"""
import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def settle(app, manager, timeout_s=5.0):
    """驱动事件循环直到在途集合清空或超时。返回是否收敛。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if manager.pending_count() == 0:
            return True
        time.sleep(0.01)
    return manager.pending_count() == 0


def main():
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    from plugins.annotation.widgets.thumbnail_manager import ThumbnailManager

    failures = []
    mgr = ThumbnailManager(QSize(88, 88), parent=None)
    missing = os.path.join(tempfile.gettempdir(), "visionmind_no_such_image_%d.png" % os.getpid())
    real = os.path.join(_ROOT, "resources", "img.png")

    print("== 缩略图在途集合收敛性 ==")
    print("   缺失文件: %s" % missing)

    # --- P1: 缺失文件也必须收敛 ---
    mgr.request(missing)
    in_flight = mgr.pending_count()
    ok = settle(app, mgr)
    print("   P1 请求缺失文件: 在途(立即)=%d -> 收敛后=%d  收敛=%s"
          % (in_flight, mgr.pending_count(), ok))
    if in_flight == 0:
        failures.append("P1 请求未被登记（测试无效）")
    if not ok:
        failures.append("P1 缺失文件后 `_pending` 未收敛：%d 个路径永久滞留（永不重试）"
                        % mgr.pending_count())

    # --- P2: 同一缺失路径必须可再次请求 ---
    mgr.request(missing)
    again = mgr.pending_count()
    print("   P2 再次请求同一缺失文件: 在途=%d（>0 表示可重试）" % again)
    if again == 0:
        failures.append("P2 同一路径无法再次请求（被永久占位）")
    settle(app, mgr)

    # --- P3: 正常文件也要收敛 ---
    if os.path.exists(real):
        mgr.request(real)
        ok3 = settle(app, mgr)
        print("   P3 请求正常文件: 收敛=%s 在途=%d" % (ok3, mgr.pending_count()))
        if not ok3:
            failures.append("P3 正常文件后 `_pending` 未收敛")

    # --- P4: drain 后线程池必须为空 ---
    ok4 = mgr.drain(5000)
    print("   P4 drain(5000)=%s 活跃线程=%d" % (ok4, mgr.active_count()))
    if not ok4 or mgr.active_count() != 0:
        failures.append("P4 drain 后仍有 %d 个缩略图解码在跑" % mgr.active_count())

    if failures:
        print("   FAIL:")
        for f in failures:
            print("     - %s" % f)
        code = 1
    else:
        print("   PASS: P1/P2/P3/P4 全部通过")
        code = 0

    print("== 退出码 %d ==" % code)
    return code


if __name__ == "__main__":
    sys.exit(main())
