"""
异步缩略图加载器 — 图像列表缩略图的后台加载与缓存

设计要点：
- QThreadPool 工作线程中用 QImageReader 解码（读取时即按目标比例降采样，
  避免大图整张载入内存）；
- QPixmap 只能在主线程创建：工作线程产出 QImage，经自动队列信号回到
  主线程后再转 QPixmap；
- LRU 缓存（路径 + mtime 失效），列表重建/翻页时命中缓存即时显示，不重复加载。
"""
import os
from collections import OrderedDict

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot, Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QImageReader, QPainter

_CACHE_LIMIT = 800  # 最多缓存的缩略图条数


class _TaskSignals(QObject):
    done = Signal(str, QImage)  # (文件路径, 缩略图 QImage)


class _ThumbnailTask(QRunnable):
    """单个缩略图解码任务（工作线程执行）"""

    def __init__(self, path: str, size: QSize, on_done):
        super().__init__()
        self.path = path
        self.size = QSize(size)
        self.signals = _TaskSignals()
        # 发射方在工作线程，接收方（manager）在主线程 → Qt 自动走队列连接
        self.signals.done.connect(on_done)

    def run(self):
        canvas = None
        try:
            if os.path.isfile(self.path):
                reader = QImageReader(self.path)
                reader.setAutoTransform(True)
                src = reader.size()
                if src.isValid() and src.width() > 0 and src.height() > 0:
                    # 读取时先做快速降采样，控制解码内存（QSize.scaled 无质量参数）
                    reader.setScaledSize(
                        src.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio))
                image = reader.read()
                if not image.isNull():
                    # 等比缩放后居中填充到目标正方形，保证列表里每个缩略图
                    # 占用相同的像素宽，图像名因此能保持左对齐
                    image = image.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
                    canvas = QImage(self.size, QImage.Format_ARGB32)
                    canvas.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(canvas)
                    painter.drawImage((self.size.width() - image.width()) // 2,
                                      (self.size.height() - image.height()) // 2, image)
                    painter.end()
        except Exception:
            # 缩略图失败不影响主流程，条目保留占位图标
            canvas = None
        finally:
            # 必须**必定**发射 done：`_pending` 的清除只发生在接收端
            # `_on_task_done` 里（`_pending.discard`）。若失败路径直接 return，
            # 该路径会永久留在"在途"集合里，此后 `request()` 每次都会因
            # `path in self._pending` 直接返回 -> **永不重试**
            #（`clear_pending()` 全仓无调用者，也没有别的复位手段）。
            self.signals.done.emit(self.path, canvas if canvas is not None else QImage())



class ThumbnailManager(QObject):
    """缩略图加载管理器（主线程使用）"""

    loaded = Signal(str, QPixmap)  # (文件路径, 缩略图 QPixmap)，主线程发出

    def __init__(self, size: QSize = QSize(88, 88), max_workers: int = None, parent=None):
        super().__init__(parent)
        # 默认按 CPU 核数合理分配解码线程（2 个太慢，几千张图会排队很久）
        if max_workers is None:
            max_workers = max(4, min(8, (os.cpu_count() or 4) + 2))
        self._size = QSize(size)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_workers)
        self._cache = OrderedDict()  # path -> (mtime, QPixmap)
        self._pending = set()        # 正在加载的路径

    @Slot(str, QImage)
    def _on_task_done(self, path: str, image: QImage):
        self._pending.discard(path)
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        self._cache[path] = (mtime, pixmap)
        self._cache.move_to_end(path)
        while len(self._cache) > _CACHE_LIMIT:
            self._cache.popitem(last=False)
        self.loaded.emit(path, pixmap)

    def get_cached(self, path: str):
        """命中缓存返回 QPixmap，否则 None（mtime 不一致视为失效）"""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        entry = self._cache.get(path)
        if entry is not None and entry[0] == mtime:
            self._cache.move_to_end(path)
            return entry[1]
        return None

    def request(self, path: str):
        """请求一个缩略图；命中缓存不重复加载，完成后发 loaded 信号"""
        if self.get_cached(path) is not None or path in self._pending:
            return
        self._pending.add(path)
        self._pool.start(_ThumbnailTask(path, self._size, self._on_task_done))

    def clear_pending(self):
        """清空在途请求记录（列表重建后由调用方重新发起）"""
        self._pending.clear()

    def active_count(self):
        """当前活跃解码线程数（测试/诊断用）。"""
        return self._pool.activeThreadCount()

    def wait_idle(self, timeout_ms=3000):
        """等待在途缩略图解码结束（不取消、不清在途集合）。

        删除图片文件前必须调用：Windows 上解码进行中 `os.remove` 会
        `WinError 32`。打开一个目录会一次性为所有可见行排入缩略图任务，
        因此"刚打开目录就删图"原本必定失败（已实测复现）。
        """
        return bool(self._pool.waitForDone(int(timeout_ms)))

    def pending_count(self):
        """当前"在途"路径数（测试/诊断用：验证失败路径不再永久滞留）。"""
        return len(self._pending)

    def drain(self, timeout_ms=5000):
        """等待在途缩略图解码结束（关闭路径调用，实现 Drainable 契约）。

        返回是否在超时内排空。刻意不调用 `QThreadPool.clear()`：它会直接删除
        尚未开始的 QRunnable，而这里的 QRunnable 同时被 Python 侧持有，
        走"让它自己跑完（失败路径也已保证必定发信号、不会卡住 _pending）"更稳。
        排空前先清空在途集合，让本轮列表重建不再复用旧的在途记录。
        """
        self._pending.clear()
        return bool(self._pool.waitForDone(int(timeout_ms)))
