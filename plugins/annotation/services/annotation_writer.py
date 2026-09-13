"""AnnotationWriter — 标注写入的唯一出口（后台串行 · 同路径最新优先 · 可等待）。

为什么需要这个组件（审计阶段 3 的最后一块）：逐图标注的**写盘**原先在 GUI 线程同步
执行，而每次标注修改、每次切图的自动保存都会触发一次；`save_image_annotations` 内部
还要 `imread_unicode` 读回整张图只为取宽高，于是每次交互都有一次可感的同步 I/O。

但"把写入丢到后台"不能只搬线程，否则会引入两个新缺陷：

1. **乱序覆盖**：同一路径的两次写入若并发，旧快照可能后落地、把新内容盖掉。
   -> 本组件全局串行（同一时刻只有一个写入任务在跑），并且同一路径**最新优先**
      （排队期间又来一次保存，只保留最新快照，旧的直接丢弃）。
2. **读到写之前**：切走（排入后台保存）再切回同一张图，标注**读**可能跑在**写**之前，
   用户会看到"刚改的标注消失"。-> 提供 `wait_for_path()`，由标注装载任务（它本来就在
   工作线程上）在开始读之前调用；显式保存也先 `wait_for_path()` 以保证写入顺序。

线程用的是标准库 `threading.Thread`（而不是 QThreadPool）：这里需要的是"阻塞等待某个
路径写完"这种**队列语义 + 条件变量**，用普通线程表达最直接，且完全不涉及 Qt 对象的
所有权/析构问题 —— 生命周期由本组件显式管理（`drain()` 停止并 join）。
"""

import threading
import time
import traceback


class AnnotationWriter:
    """把标注写入排成一条串行队列，并提供"等到写完"的能力。"""

    def __init__(self, name="annotation.writer"):
        self._cv = threading.Condition(threading.Lock())
        self._pending = {}          # path -> (seq, write_fn)：同一路径只留最新
        self._inflight = None       # 正在写的 path
        self._seq = 0
        self._closed = False
        self._writes = 0            # 已完成的写入次数（回归测试用于观察合并效果）
        self._failures = 0
        self._failure_handler = None
        self._last_failure = None
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def set_failure_handler(self, handler):
        """设置失败回调（handler(path, message)）。

        **必须在后台线程之外自行 marshal**（回调是在写入线程上被调用的）。
        存在的理由：异步保存没有返回值，调用方拿不到 `{"status":"error"}`；
        若不上报，写盘失败就变成**静默丢数据**（例如格式/扩展名不匹配导致编码失败）。
        """
        with self._cv:
            self._failure_handler = handler

    def last_failure(self):
        with self._cv:
            return self._last_failure

    # ---- 生产者（GUI 线程）----

    def submit(self, path, write_fn):
        """排入一次写入；同一路径的旧快照被替换（最新优先）。返回是否受理。"""
        with self._cv:
            if self._closed:
                return False
            self._seq += 1
            self._pending[path] = (self._seq, write_fn)
            self._cv.notify_all()
            return True

    def pending_count(self):
        with self._cv:
            return len(self._pending) + (1 if self._inflight is not None else 0)

    def is_path_pending(self, path):
        with self._cv:
            return self._inflight == path or path in self._pending

    def writes_done(self):
        with self._cv:
            return self._writes

    def failures(self):
        with self._cv:
            return self._failures

    # ---- 等待（工作线程 / 显式保存）----

    def wait_for_path(self, path, timeout_ms=5000):
        """等到该路径没有在途/排队的写入。返回是否在超时内达成。

        供**标注装载**（工作线程）与显式保存调用：保证"读之前写完"。
        """
        if not path:
            return True
        deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
        with self._cv:
            while self._inflight == path or path in self._pending:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return False
                self._cv.wait(remain)
            return True

    def flush(self, timeout_ms=5000):
        """等到所有在途/排队的写入结束（关闭路径与显式保存用）。"""
        deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
        with self._cv:
            while self._pending or self._inflight is not None:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return False
                self._cv.wait(remain)
            return True

    def drain(self, timeout_ms=5000):
        """Drainable 契约：先排空队列，再停止工作线程并 join。"""
        ok = self.flush(timeout_ms)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(max(0.0, timeout_ms / 1000.0))
        alive = self._thread.is_alive()
        if alive:
            print("[AnnotationWriter] 写入线程未能在超时内结束")
        return ok and not alive

    # ---- 消费者（写入线程）----

    def _loop(self):
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if not self._pending:
                    if self._closed:
                        return
                    continue
                # 最新优先：同一路径只会有最新的一份快照
                path = None
                best_seq = -1
                for p, (seq, _fn) in self._pending.items():
                    if seq > best_seq:
                        path, best_seq = p, seq
                _seq, fn = self._pending.pop(path)
                self._inflight = path
            try:
                result = fn()
                if isinstance(result, dict) and result.get("status") == "error":
                    # 写盘函数自己报错（例如编码失败）。异步路径没有 UI 反馈，
                    # 必须显式上报，否则就是静默丢数据。
                    self._report_failure(path, str(result.get("message", "未知错误")))
            except Exception as exc:                      # 纯 I/O 失败：记录并上报，不抛出
                print("[AnnotationWriter] 写入失败 %s: %s" % (path, exc))
                traceback.print_exc()
                self._report_failure(path, "%s: %s" % (type(exc).__name__, exc))
            finally:
                with self._cv:
                    self._inflight = None
                    self._writes += 1
                    self._cv.notify_all()

    def _report_failure(self, path, message):
        with self._cv:
            self._failures += 1
            self._last_failure = (path, message)
            handler = self._failure_handler
        if handler is None:
            return
        # 刻意在锁外调用：回调可能 marshal 回 GUI 线程，不能持锁等待
        try:
            handler(path, message)
        except Exception:
            traceback.print_exc()
