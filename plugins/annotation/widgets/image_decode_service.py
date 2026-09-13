"""图像解码服务 — 有界并发、最新优先、生命周期自持的后台解码执行器。

替代原先 `ImageLoader(QThread)` + `self.loader = None` 的写法。原先那套写法的
致命组合是：

  1. 每切一张图就新建一个 `QThread`，并把旧的引用置 `None`；
  2. `requestInterruption()` 是空操作（`run()` 里从不检查 `isInterruptionRequested()`，
     而 `QImageReader.read()` 本身也不可中断），所以旧线程一定会跑完；
  3. 于是"运行中的 QThread 丢掉最后一个 Python 引用"→ shiboken 析构其 C++ 对象
     → `qFatal("QThread: Destroyed while thread '' is still running")` → `abort()`。
     这是进程级强杀，Python 层 try/except 拦不住。已在
     `scripts/_probe_interface_harness.py` 里通过真实 `switch_image()` 复现：
     第 2 次切换即 aborts（退出码 -1073740791）。

本模块的设计约束（每一条都对应上面某个根因，改动时不要破坏）：

  1. 不继承 QThread。线程由 `QThreadPool` 持有并由它回收，调用方永远不需要、
     也不允许"丢弃线程引用"，因此不可能出现运行中被析构的 QThread。
  2. 并发有界。最多 `max_workers` 张解码在飞，其余请求在排队阶段就被代次校验
     作废（启动即返回，不做任何 I/O），不会像旧实现那样堆积 N 个并发解码
     （实测 32 并发 = 780 MiB 瞬时分配）。
  3. 不遮蔽 Qt 内建信号名。对外只有 `ready` / `failed`，绝不覆盖
     `QThread.finished`（旧实现覆盖后，"线程真正结束"这个事件在 Python 侧再也拿不到）。
  4. 不调用 `terminate()`。解码不可中断，因此用"代次作废 + 结果丢弃"代替伪中断；
     强杀持有 Qt 图像插件内部锁的线程会导致后续解码死锁。
  5. 排空是显式的。`drain()` 供应用关闭/插件卸载时确定性收尾。
  6. **代次的唯一权威在调用方**。本服务不维护自己的 generation 计数器，只通过
     构造时注入的 `is_current` 只读查询。这避免了"widget 一个代次、service 一个
     代次"再次产生错位（原实现在 interface/service/widget 三层各存一份
     current_image_path 就是同一类错误的放大版）。

线程亲和性：`QImage` 在工作线程内解码产出，经排队信号回到 GUI 线程后，才由
调用方转换成 `QPixmap`（QPixmap 只能在 GUI 线程创建）。
"""
import traceback

from PySide6.QtCore import QObject, QRunnable, QSize, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader


class _DecodeSignals(QObject):
    """工作线程 → GUI 线程的信号载体。

    显式做成一个 QObject（而不是复用 QRunnable 自身）的原因与
    `thumbnail_manager.py` 相同：QRunnable 由 QThreadPool 拥有并在 run()
    结束后删除，不能安全地作为信号发送方长期存活。
    """

    # (generation, path, image, display_scale, original_size)
    ready = Signal(int, str, QImage, float, QSize)
    # (generation, path, message)
    failed = Signal(int, str, str)


class _DecodeTask(QRunnable):
    """单张图的解码任务（工作线程执行）。"""

    def __init__(self, generation, path, service, max_dim, signals):
        super().__init__()
        self._generation = generation
        self._path = path
        self._service = service
        self._max_dim = max_dim
        self._signals = signals
        self.setAutoDelete(True)

    def _is_current(self):
        """是否仍是当前代次。

        刻意不吞异常：历史上这里用 `except Exception: return False` 把一个
        AttributeError（服务端方法名写错）变成了"解码任务全部静默空转、一张图
        都不显示"。宁可抛到 run() 的日志里，也不要静默失效。
        """
        return bool(self._service.is_current(self._generation))

    def run(self):
        try:
            # 排队期间已被更新的请求作废：直接返回，不做任何 I/O。
            # 这是"最新优先"的实现方式，也让排队中的过期任务不占用解码槽。
            if not self._is_current():
                return

            reader = QImageReader(self._path)
            if not reader.canRead():
                self._fail("无法读取图片（格式不支持或文件缺失）")
                return

            orig_size = reader.size()
            width, height = orig_size.width(), orig_size.height()
            if width <= 0 or height <= 0:
                self._fail("图片尺寸无效")
                return

            # 超大图按比例降采样（Proxy），控制单帧内存
            scale = 1.0
            if width > self._max_dim or height > self._max_dim:
                scale = self._max_dim / float(max(width, height))
                reader.setScaledSize(QSize(max(1, int(width * scale)),
                                           max(1, int(height * scale))))

            image = reader.read()
            if image is None or image.isNull():
                self._fail("图片解码失败（数据损坏或内存不足）")
                return

            # 解码是长耗时步骤，期间用户很可能又切图了：丢弃结果，不再占用主线程
            if not self._is_current():
                return

            self._signals.ready.emit(self._generation, self._path, image,
                                     float(scale), orig_size)
        except Exception as exc:  # 工作线程里的异常必须自己吞掉并上报，否则丢失
            print(f"[ImageDecodeService] 解码异常 {self._path}: {exc}")
            traceback.print_exc()
            self._fail(str(exc))

    def _fail(self, message):
        """只在仍是当前代次时上报失败，避免快速切图时刷屏。"""
        if self._is_current():
            self._signals.failed.emit(self._generation, self._path, message)


class ImageDecodeService(QObject):
    """有界并发的最新优先解码执行器。

    用法（调用方持有代次权威）::

        self._generation = 0
        self._decoder = ImageDecodeService(lambda gen: gen == self._generation, parent=self)
        self._decoder.ready.connect(self._on_decoded)     # 自动走排队连接

        # 切图：
        self._generation += 1                 # 使所有在途解码立即作废
        self._decoder.request(self._generation, path)

        # 关闭：
        self._generation += 1
        self._decoder.drain(5000)
    """

    ready = Signal(int, str, QImage, float, QSize)
    failed = Signal(int, str, str)

    def __init__(self, is_current, parent=None, max_workers=3, max_dim=4096):
        """
        Args:
            is_current: `callable(generation) -> bool`，由调用方提供，用于判断某个
                代次是否仍然有效。会被工作线程调用，必须是无副作用、只读、
                线程安全的（读取一个 Python int 属性即可）。
            max_workers: 同时在跑的解码上限。取 3 的理由：解码耗时约 30-50ms
                （3200x1904），而按键自动重复约 30ms 一次，3 个槽位足以让最新请求
                几乎立刻开始，同时把单帧 ~24 MiB 的内存上限压在 ~75 MiB。
            max_dim: 超过该边长的图按比例降采样后再解码，控制单帧内存。
        """
        super().__init__(parent)
        if not callable(is_current):
            raise TypeError("is_current 必须是可调用对象（代次的唯一权威）")

        self._is_current = is_current
        self._max_dim = int(max_dim)

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_workers)))

        self._signals = _DecodeSignals(self)
        # 信号到信号：两侧都在 GUI 线程，直连；真正的跨线程排队发生在
        # self.ready 与调用方槽之间（发送方线程 != 接收方线程 → 队列连接）。
        self._signals.ready.connect(self.ready)
        self._signals.failed.connect(self.failed)

    def request(self, generation, path):
        """请求解码一张图。更早代次的在途任务会自行作废。"""
        if not path:
            return
        self._pool.start(_DecodeTask(int(generation), str(path), self,
                                     self._max_dim, self._signals))

    def is_current(self, generation):
        """某个代次是否仍然有效（工作线程会调用；代次的权威在调用方）。"""
        return bool(self._is_current(generation))

    def is_busy(self):
        """是否有解码在跑（供画布显示"正在加载"提示）。"""
        return self._pool.activeThreadCount() > 0

    def active_count(self):
        """当前活跃解码线程数（测试/诊断用）。"""
        return self._pool.activeThreadCount()

    def max_workers(self):
        """并发上限（测试用：断言并发有界）。"""
        return self._pool.maxThreadCount()

    def drain(self, timeout_ms=5000):
        """等待所有在途解码结束。返回是否在超时内排空。

        只能在 GUI 线程调用（这是关闭/卸载路径，允许阻塞）。调用前应先把
        generation 递增，让在途任务尽快自己作废返回。
        """
        return bool(self._pool.waitForDone(int(timeout_ms)))

    def wait_idle(self, timeout_ms=3000):
        """等待在途解码结束，但**不取消、不换代次、不动排队任务**。

        专供"需要释放该图片文件句柄"的操作调用（例如删除图片文件）：Windows 上
        解码进行中 `os.remove` 会 `WinError 32 另一个程序正在使用此文件`
        （已用 scripts/_probe_delete_lock.py 实测复现：open_directory 后
        缩略图池活跃 3、解码池活跃 1，此刻删除即失败；等约 200ms 后成功）。
        与 drain() 的区别是它没有任何副作用，因此可以在正常交互路径上使用。
        """
        return bool(self._pool.waitForDone(int(timeout_ms)))
