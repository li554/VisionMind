"""ImageSession — 标注界面的**唯一**图像/标注状态所有者。

为什么需要它（审计 W1/W2、D2/D3/D8 的共同根因）：同一件事原本有三份状态 ——

    interface.current_image_path      （interface 自己一份）
    service.current_image_path        （service 自己一份，save_current 用的是它）
    widget._current_image_path        （widget 自己一份，用于增强图）
    service.current_annotations       （widget 通过 property 读到的是它的**别名**）

于是"当前图片"与"当前标注"可以被三处代码分别改写，中间出现

    (图 N-1 的 pixels / original_image_size / zoom / pan)
  + (图 N 的 path / annotations)

这种既不自洽、又会被自动保存写进磁盘的状态。像素与标注是**两笔独立事务**，
`_load_generation` 只能保证像素不回退，对标注无话可说。

本模块把「当前路径 + 该图标注 + 已提交的像素与度量」绑成一个整体，**只允许整体
替换**，并给出唯一的权威读取点：

    begin_load(path)                     -> 换代次、改路径、清空"已提交"
    commit_pixels(gen, ...)              -> 只有代次匹配才落地
    set_annotations(anns, for_path=...)  -> 路径不匹配就拒绝（拦住"把 A 图标注
                                            装进 B 图的当前状态"这类静默污染）
    is_committed                         -> 像素属于当前图；交互/坐标换算的唯一闸门

`path` / `annotations` / `image` / `pixmap` / `original_image_size` / `display_scale`
都是**只读 property**：任何"我直接赋值一份自己的副本"的代码都会立刻
`AttributeError`，而不是留下第二份真值。
"""
import os

from PySide6.QtCore import QObject, QSize, Signal


class ImageSession(QObject):
    """(当前图片, 该图标注, 已提交像素与度量) 的唯一所有者。"""

    loaded_changed = Signal()        # 像素提交 / 清空
    annotations_changed = Signal()   # 标注集合被整体替换

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path = None
        self._generation = 0
        self._annotations = []
        self._image = None
        self._pixmap = None
        self._original_size = QSize(0, 0)
        self._display_scale = 1.0
        self._committed_generation = -1
        # 当前标注是否来自一次**成功**的装载（含"确实没有标注"）。装载失败
        # （文件损坏/编码错误）时为 False，见 is_save_allowed。
        self._annotations_loaded = True
        # ---- 与"当前图"同生命周期的编辑态（审计收尾：并入唯一所有者）----
        # 这三者原先分散在 ManualAnnotationService 上，但它们全部是**按图作用域**的：
        #   * 撤销/重做栈里的 frame 记录的是某张图的标注 + 可见性（换图必须清空，
        #     否则会把上一张图的标注写进当前图 —— 正是 D3 那条静默损坏）；
        #   * hidden_indices 是**按下标寻址**的可见性集合，与当前标注列表同生共死
        #     （D21：只回滚列表不回滚它就会出现"列表=删除前、可见性=删除后"的不自洽）；
        #   * 副标注是当前图的对照标注，换图后同样必须整体失效。
        # 放进 session 后，"换图要清哪些"不再靠 service 逐个记得，而是由
        # begin_load/clear 统一负责，少一个漏清的机会。
        self._undo_stack = []
        self._redo_stack = []
        self._hidden_indices = set()
        self._secondary_annotations = []
        self._secondary_annotation_map = {}

    # ------------------------------------------------------------------ 读

    @property
    def path(self):
        return self._path

    @property
    def generation(self):
        """加载代次：**唯一权威**。异步结果必须与它比对通过后才能提交。"""
        return self._generation

    @property
    def annotations(self):
        """当前图的标注列表（可变对象，就地编辑由上层负责发通知）。"""
        return self._annotations

    @property
    def image(self):
        return self._image

    @property
    def pixmap(self):
        return self._pixmap

    @property
    def original_image_size(self):
        return self._original_size

    @property
    def display_scale(self):
        return self._display_scale

    @property
    def is_committed(self):
        """像素与度量是否已属于**当前**代次。

        这是交互（鼠标/键盘/滚轮/命中测试/加 SAM 点）与 `get_state` 面向 agent
        暴露分辨率时的唯一闸门：未提交时 `original_image_size` / `zoom` /
        `pan_offset` 都不是当前图的，任何坐标换算都会算错。
        """
        return self._pixmap is not None and self._committed_generation == self._generation

    @property
    def committed_generation(self):
        return self._committed_generation

    # ---- 与当前图同生命周期的编辑态（唯一所有者，见 __init__ 的说明）----

    @property
    def undo_stack(self):
        return self._undo_stack

    @property
    def redo_stack(self):
        return self._redo_stack

    @property
    def hidden_indices(self):
        """按下标寻址的可见性集合 —— 与 `annotations` 必须始终自洽。"""
        return self._hidden_indices

    @hidden_indices.setter
    def hidden_indices(self, value):
        self._hidden_indices = set(value or ())

    @property
    def secondary_annotations(self):
        return self._secondary_annotations

    @secondary_annotations.setter
    def secondary_annotations(self, value):
        self._secondary_annotations = list(value or ())

    @property
    def secondary_annotation_map(self):
        return self._secondary_annotation_map

    @secondary_annotation_map.setter
    def secondary_annotation_map(self, value):
        self._secondary_annotation_map = dict(value or {})

    def clear_edit_state(self):
        """清空与"某张图"绑定的编辑态。

        换图 / 换项目 / 关闭目录时统一调用 —— 这些状态一旦跨图存活就是数据损坏源
        （D3 跨图撤销、D21 可见性与列表不自洽）。
        """
        self._undo_stack = []
        self._redo_stack = []
        self._hidden_indices = set()
        self._secondary_annotations = []
        self._secondary_annotation_map = {}

    @property
    def is_annotations_loaded(self):
        """当前图的标注是否已装载完成（成功装载，含"确实没有标注"）。"""
        return self._annotations_loaded

    @property
    def is_ready(self):
        """像素与标注**都**就绪 —— 交互（绘制/编辑）的唯一闸门。

        比 `is_committed` 更严格：标注装载到位之前不允许绘制，否则用户刚画的内容
        会被随后到达的"这张图既有标注"整体替换掉；同时 `is_save_allowed` 为 False，
        任何保存都被拒，不会把上一张图的标注写进这张图的文件。
        """
        return self.is_committed and self._annotations_loaded

    @property
    def is_save_allowed(self):
        """当前状态是否允许被保存到磁盘。

        装载失败（标注文件损坏 / 编码错误）时禁止保存：否则任何一次自动保存都会把
        空标注写进那张图的标注文件，把"可恢复的损坏"变成"确定的数据丢失"
        （审计 D8）。装载失败时界面保留内存里的旧标注只为不误覆盖，**但绝不能落盘**。
        """
        return self._path is not None and self._annotations_loaded

    def belongs_to_current(self, path):
        """给定路径是否是当前图（规范化比较，避免分隔符/大小写差异误判）。"""
        if not path or not self._path:
            return False
        try:
            return os.path.normcase(os.path.abspath(path)) == \
                   os.path.normcase(os.path.abspath(self._path))
        except Exception:
            return path == self._path

    # ------------------------------------------------------------- 写：唯一入口

    def begin_load(self, path):
        """开始加载新图：换代次 + 改路径 + **清空已提交的像素与度量**。

        清空是关键：这样"像素/度量属于哪张图"不再靠约定，而由 `is_committed`
        直接回答；提交之前任何交互都必须拒绝，因此不存在"新标注 + 旧度量"的窗口。
        返回新代次，调用方必须把它随解码请求一起传递。
        """
        self._generation += 1
        self._path = path
        self._discard_pixels()
        # 编辑态与图同生命周期：换图即清（撤销栈/可见性/副标注都不能跨图存活）。
        # 由这里统一负责，而不是指望每个调用点都记得清一遍（D3/D21 的根因）。
        self.clear_edit_state()
        # 新图的标注尚未装载（成功装载后 set_annotations 会置 True）；
        # 这段时间内 is_save_allowed 为 False，避免把空/旧标注写进新图的文件。
        self._annotations_loaded = False
        self.loaded_changed.emit()
        return self._generation

    def cancel_pending_load(self):
        """作废在途解码结果（非阻塞）：换代次并清空已提交状态。"""
        self._generation += 1
        self._discard_pixels()
        self.loaded_changed.emit()
        return self._generation

    def commit_pixels(self, generation, image, pixmap, original_size,
                      display_scale=1.0):
        """提交像素与度量。**只有代次匹配才落地**，过期结果被静默丢弃并返回 False。"""
        if generation != self._generation:
            return False
        if pixmap is None:
            return False
        self._image = image
        self._pixmap = pixmap
        self._original_size = original_size or QSize(0, 0)
        self._display_scale = float(display_scale or 1.0)
        self._committed_generation = generation
        self.loaded_changed.emit()
        return True

    def set_annotations(self, annotations, for_path=None, expected_generation=None,
                        loaded=True):
        """整体替换标注集合。任一校验不过则**拒绝**并返回 False。

        `for_path` 用来拦住"把 A 图的标注装进 B 图的当前状态"——这正是审计里
        D2/D3 那几条静默数据损坏的共同形态。
        `loaded=False` 表示这批标注**没有成功装载**（装载失败时上层保留旧标注），
        此时 `is_save_allowed` 为 False，任何保存都会被拒。
        """
        if expected_generation is not None and expected_generation != self._generation:
            return False
        if for_path is not None and not self.belongs_to_current(for_path):
            print(f"[ImageSession] 拒绝装载不属于当前图的标注: for_path={for_path} "
                  f"current={self._path}")
            return False
        self._annotations = annotations if annotations is not None else []
        self._annotations_loaded = bool(loaded)
        self.annotations_changed.emit()
        return True

    def mark_annotations_unloaded(self):
        """把"标注已装载"标记置回 False（装载失败路径）。

        装载失败时界面会保留内存里的旧标注只为不误覆盖，但必须同时禁止保存 ——
        否则一次自动保存就把空标注写进那张**可恢复的**坏文件（审计 D8）。
        """
        self._annotations_loaded = False

    def clear(self):
        """清空全部状态（换项目 / 关闭目录）。"""
        self._generation += 1
        self._path = None
        self._annotations = []
        self._annotations_loaded = True
        self.clear_edit_state()
        self._discard_pixels()
        self.loaded_changed.emit()
        self.annotations_changed.emit()

    def _discard_pixels(self):
        self._image = None
        self._pixmap = None
        self._original_size = QSize(0, 0)
        self._display_scale = 1.0
        self._committed_generation = -1

    def snapshot(self):
        """诊断用：把关键状态打成一个 dict（测试断言一致性时使用）。"""
        return {
            "path": self._path,
            "generation": self._generation,
            "committed_generation": self._committed_generation,
            "is_committed": self.is_committed,
            "annotation_count": len(self._annotations),
        }


__all__ = ["ImageSession"]
