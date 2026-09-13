import traceback
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QSize, QPoint, Slot
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor, QPolygonF, QTransform, QImageReader
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
import numpy as np
import cv2
from core.common.project_settings import project_settings
from core.common.settings import settings
from core.common.action_registry import action
from core.common.background_task import BackgroundTaskRunner
from .image_decode_service import ImageDecodeService

class DrawingMode:
    EDIT = 0
    RECT = 1
    POLYGON = 2
    SAM = 3  # AI 点工具 (智能分割点)
    OBB = 4
    ROI = 5
    AI_RECT = 6  # AI 矩形框工具

class DrawingWidget(QWidget):
    annotation_finished = Signal(list) # [x, y, w, h] or points
    delete_annotation = Signal(int)
    set_as_example = Signal(int)
    mode_changed = Signal(int)
    sam_point_added = Signal(list, int) # points, label
    context_menu_requested = Signal(int, QPoint) # index, global_pos
    annotation_double_clicked = Signal(int) # index
    annotation_selected = Signal(int) # index
    annotation_modified = Signal() # Emitted when annotation is moved/resized
    annotation_modify_started = Signal()  # Before drag / vertex edit — snapshot undo here
    batch_selection_changed = Signal(list) # list of selected annotation indices
    batch_points_selected = Signal(list, list) # list of (ann_idx, poly_idx, pt_idx), selection_rect
    roi_selected = Signal(list) # [x, y, w, h] in image coords
    roi_menu_requested = Signal(QPoint) # global_pos
    mouse_pos_changed = Signal(QPointF) # image coordinates
    image_loaded = Signal() # Emitted when image is loaded
    accept_secondary = Signal(int) # 主标注索引，用于接收副标注
    secondary_context_menu_requested = Signal(int, QPoint) # 副标注索引, global_pos
    enhance_toggled = Signal(bool) # 增强状态切换信号，参数为是否显示增强图

    def __init__(self, service, parent=None):
        super().__init__(parent)
        self._service = service
        self._is_clearing = False  # 标志：正在清理中，防止 paintEvent 访问无效数据
        # 注意：`image` / `pixmap` / `original_image_size` / `display_scale` /
        # `current_image_path` 以及加载代次，全部是**只读 property**，真值在
        # `service.session`（唯一状态所有者）。widget 刻意不再自己存一份 ——
        # 审计 W2 的根因正是 interface / service / widget 各存一份，使
        # "(图 N-1 的度量) + (图 N 的标注)" 这种算错坐标的状态可以出现。
        # 后台解码执行器：线程由 QThreadPool 持有，widget 既不创建也不丢弃 QThread，
        # 因此不可能出现"QThread: Destroyed while thread is still running"。
        self._decoder = ImageDecodeService(self._is_generation_current, parent=self)
        self._decoder.ready.connect(self._on_image_decoded,
                                    type=Qt.ConnectionType.QueuedConnection)
        self._decoder.failed.connect(self._on_image_decode_failed,
                                    type=Qt.ConnectionType.QueuedConnection)

        # 增强图生成器：LibLLIE + numpy 是 CPU 密集操作，原先在 GUI 线程同步执行，
        # 每次切图提交都会冻结界面（审计 D17）。放到单独的线程上，结果带代次校验。
        self._enhancer_runner = BackgroundTaskRunner(parent=self, max_threads=1,
                                                    name="canvas.enhance")
        self._pending_enhance_request = None

        # View state
        self.zoom = 1.0  # 缩放级别
        self.pan_offset = QPointF(0, 0)  # 平移偏移
        self.last_mouse_pos = QPointF(0, 0)

        # Drawing state
        self.mode = DrawingMode.SAM  # 当前绘制模式
        self.rect_start = None
        self.rect_end = None
        self.current_poly = []  # 当前多边形顶点
        self.obb_points = [] # For OBB drawing (3 points usually)
        self.sam_points = []  # SAM点击点列表

        # Display state
        self.show_bboxes = True  # 显示边界框
        self.show_labels = True  # 显示标签
        self.task_mode = 'seg'  # 任务模式(seg/det)
        self.ai_enabled = True  # AI辅助开关
        self.category_colors = {}  # 类别颜色映射

        # Data (annotations / secondary_annotations / hidden_indices / secondary_annotation_map 通过 property 从 service 读取)
        self.hover_index = -1  # 鼠标悬停标注索引
        self.selected_index = -1  # 选中标注索引
        self.is_near_first_point = False

        # 聚焦状态
        self.focused_index = -1  # 当前聚焦的标注索引
        self.focus_zoom_level = 4.0  # 聚焦时的放大倍数

        # Interaction state
        self.hover_point = None # (ann_idx, poly_idx, pt_idx)
        self.dragging_point = None # (ann_idx, poly_idx, pt_idx)
        self.dragging_annotation = -1 # index of annotation being moved
        self.drag_start_img_pos = None
        self.last_drawing_mode = DrawingMode.RECT # Remember last non-edit mode
        self.ignore_next_left_press = False
        self.context_vertex_info = None
        self.context_edge_info = None

        # Batch selection state (Ctrl+drag)
        self.batch_selecting = False
        self.batch_selection_start = None
        self.batch_selection_end = None
        self.batch_selected_indices = []
        self.batch_selected_points = []

        # Right-click context menu state
        self.right_press_pos = None  # QPointF: position where right button was pressed
        self.right_press_time = 0    # float: time when right button was pressed
        self.right_button_held = False  # bool: whether right button is currently held
        self._right_dragging = False  # bool: whether right button drag (pan) was triggered
        self.right_click_drag_threshold = 5  # pixels: max movement to consider as click vs drag
        self.right_click_time_threshold = 300  # ms: max time to consider as quick click

        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.status_text = None
        self.status_color = QColor(200, 200, 200)
        self.interactive = True # Whether to allow drawing/editing

        # 图像增强相关
        self._enhanced_pixmap = None  # 增强后的pixmap缓存
        self._show_enhanced = False  # 当前是否显示增强图
        self._enhance_requested = False  # 用户已请求增强，生成完成后自动切换显示
        self._enhancer_unavailable = False  # 增强器不可用（如缺少 libllie）时不再重试
        self._llie_enhancer = None  # LibLLIE增强器实例缓存

    @property
    def annotations(self):
        """从 service 读取当前标注列表（widget 不存储数据）"""
        return self._service.current_annotations if self._service else []

    @property
    def secondary_annotations(self):
        """从 service 读取副标注列表"""
        return getattr(self._service, 'secondary_annotations', []) if self._service else []

    @property
    def hidden_indices(self):
        """从 service 读取隐藏索引集合"""
        return self._service.hidden_indices if self._service else set()

    # ---- 以下全部是 session 的**只读投影**：widget 不持有第二份真值 ----

    @property
    def session(self):
        return getattr(self._service, "session", None) if self._service is not None else None

    @property
    def generation(self):
        """当前加载代次（权威在 session）。"""
        session = self.session
        return session.generation if session is not None else 0

    @property
    def is_committed(self):
        """像素与度量是否已属于**当前**图 —— 交互与坐标换算的唯一闸门。"""
        session = self.session
        return bool(session is not None and session.is_committed)

    @property
    def current_image_path(self):
        session = self.session
        return session.path if session is not None else None

    @property
    def image(self):
        session = self.session
        return session.image if session is not None else None

    @property
    def pixmap(self):
        session = self.session
        return session.pixmap if session is not None else None

    @property
    def original_image_size(self):
        session = self.session
        return session.original_image_size if session is not None else QSize(0, 0)

    @property
    def display_scale(self):
        session = self.session
        return session.display_scale if session is not None else 1.0

    def _interaction_ready(self):
        """交互闸门：**像素与标注都就绪**时才接受鼠标/键盘/滚轮与坐标换算。

        用 session.is_ready（而不只是 is_committed）：标注装载（后台 I/O）完成前
        不允许绘制，否则刚画的内容会被随后到达的"这张图既有标注"整体替换掉。
        这也是审计 W2 的根治点：未就绪时 `original_image_size` / `zoom` /
        `pan_offset` 都不属于当前图，任何 `to_image_coords` 都会算错，一落盘就是错标。
        """
        session = self.session
        return bool(session is not None and session.is_ready)

    @property
    def secondary_annotation_map(self):
        """从 service 读取主副标注映射"""
        return getattr(self._service, 'secondary_annotation_map', {}) if self._service else {}

    @property
    def persistent_roi(self):
        """持久化的 ROI 框 [x, y, w, h]，**读取时校验**（审计 D18）。

        这个值来自 `project_info.json` 的裸 JSON，没有任何 schema 校验。原先有 5 处
        直接 `px, py, pw, ph = self.persistent_roi`，其中两处在 `paintEvent` 里 ——
        一旦值是 None / 长度不对 / 元素不是数字，解包就抛异常，而 `paintEvent` 抛出异常
        会让**画布永久停止重绘**（用户看到的是"界面卡住了"）。

        因此把校验收敛到唯一入口：非法值一律当作"没有 ROI"（返回 None），
        这样所有使用点都天然安全（它们本来就都写成 `if self.persistent_roi:` 再解包）。
        """
        value = project_settings.get("roi")
        return self._coerce_roi(value)

    @staticmethod
    def _coerce_roi(value):
        """把任意来源的 ROI 值规范成 4 元组或 None。"""
        if value is None or isinstance(value, (str, bytes, bool)):
            return None
        try:
            items = list(value)
        except TypeError:
            return None
        if len(items) != 4:
            return None
        out = []
        for v in items:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            if v != v or v in (float("inf"), float("-inf")):   # NaN / inf
                return None
            out.append(v)
        return tuple(out)

    @persistent_roi.setter
    def persistent_roi(self, value):
        project_settings.set("roi", value)

    @staticmethod
    def _is_polygon_like(poly):
        """判断一个多边形元素是否**至少像**一条多边形（可 len、长度>=3）。

        审计 D18：`any(len(p) >= 3 for p in polys)` 对 `[1.0, 2.0]`、`[None]` 这类
        畸形元素会抛 TypeError，而它在 paintEvent 的帧循环里 —— 一次异常就让画布
        永久停止重绘。这里把"能不能 len"变成显式判断。
        """
        if poly is None or isinstance(poly, (str, bytes, bool)):
            return False
        try:
            return len(poly) >= 3
        except TypeError:
            return False

    @staticmethod
    def _safe_bbox(ann, default=(0.0, 0.0, 0.0, 0.0)):
        """取标注的 bbox 并保证返回 **4 个数值**（审计 D18）。

        `ann.get('bbox', default)` 只在键缺失时生效；键存在但值为 None / 长度不为 4 /
        元素不是数字时仍会返回坏值，随后 `bx, by, bw, bh = ...` 抛 TypeError。
        由于这段代码在 `paintEvent` 的帧循环里，一次异常就会让整张画布**永久停止重绘**。

        这里对每个元素做数值规范化（非数字 -> 用 default 的对应项），
        保证绘制路径永远不会因为数据畸形而中断。
        """
        if not isinstance(ann, dict):
            return default
        raw = ann.get("bbox")
        if raw is None or isinstance(raw, (str, bytes, bool)):
            return default
        try:
            items = list(raw)
        except TypeError:
            return default
        if len(items) != 4:
            return default
        out = []
        for idx, v in enumerate(items):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                out.append(default[idx])
            else:
                out.append(float(v))
        return tuple(out)

    def set_category_colors(self, colors):
        """设置类别颜色映射，统一归一化为 QColor（兼容 QColor/str/list 格式）"""
        normalized = {}
        for k, v in (colors or {}).items():
            if isinstance(v, QColor):
                normalized[k] = v
            elif isinstance(v, str):
                c = QColor(v)
                normalized[k] = c if c.isValid() else QColor(0, 255, 255)
            elif hasattr(v, '__len__') and len(v) >= 3:
                try:
                    normalized[k] = QColor(*[int(c) for c in v[:3]])
                except (TypeError, ValueError):
                    normalized[k] = QColor(0, 255, 255)
            else:
                normalized[k] = QColor(0, 255, 255)
        self.category_colors = normalized
        self.update()

    def set_display_controls(self, show_annotations, show_labels):
        self.show_bboxes = show_annotations
        self.show_labels = show_labels
        self.update()

    def focus_on_annotation(self, index):
        """聚焦到指定标注（放大4倍并居中）"""
        if index < 0 or index >= len(self.annotations):
            return

        # 如果已经聚焦到同一个标注，不做任何操作
        # （避免拖动标注点时触发 unfocus → reset_view 导致缩放重置）
        if self.focused_index == index:
            return

        # 设置新的聚焦索引
        self.focused_index = index

        # 获取标注的 bbox
        ann = self.annotations[index]
        bx, by, bw, bh = ann.get('bbox', [0, 0, 0, 0])

        # 如果 bbox 为空，从 polygons 计算
        if bw == 0 or bh == 0:
            all_points = []
            for poly in ann.get('polygons', []):
                for pt in poly:
                    try:
                        if hasattr(pt, '__len__') and len(pt) >= 2:
                            if hasattr(pt[0], '__len__'):
                                all_points.append((pt[0][0], pt[0][1]))
                            else:
                                all_points.append((pt[0], pt[1]))
                    except:
                        continue
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                bx = min(xs)
                by = min(ys)
                bw = max(xs) - bx
                bh = max(ys) - by

        if bw > 0 and bh > 0:
            # 计算目标中心
            center_x = bx + bw / 2
            center_y = by + bh / 2

            # 动态计算缩放：让标注占画布的60%，但不低于focus_zoom_level
            target_ratio = 0.6
            zoom_x = (self.width() * target_ratio) / bw if bw > 0 else self.focus_zoom_level
            zoom_y = (self.height() * target_ratio) / bh if bh > 0 else self.focus_zoom_level
            self.zoom = min(zoom_x, zoom_y, self.focus_zoom_level)

            # 计算平移偏移，使目标居中
            widget_center_x = self.width() / 2
            widget_center_y = self.height() / 2

            self.pan_offset = QPointF(
                widget_center_x - center_x * self.zoom,
                widget_center_y - center_y * self.zoom
            )

            self.update()

    def unfocus(self):
        """取消聚焦，恢复到正常视图状态（reset_view）"""
        self.focused_index = -1
        self.reset_view()

    def set_status(self, status_text):
        self.status_text = status_text
        if status_text == "NG":
            self.status_color = QColor(255, 80, 80)
        elif status_text == "OK":
            self.status_color = QColor(80, 220, 120)
        else:
            self.status_color = QColor(200, 200, 200)
        self.update()

    @action("interact.toggle_enhance", description="切换原图/增强图显示。\n- 增强图使用 LibLLIE 模型对低光照图像进行自动增强\n- 需要先加载图像；首次切换会在**后台**生成增强图（较慢），完成后自动切换显示\n- 再次调用可切回原图", category="画布", scope="agent")
    def toggle_enhance(self):
        """切换原图/增强图显示（增强图生成在后台）。"""
        if not self.image or not self.pixmap:
            return "错误: 没有加载图像，无法切换增强显示"

        if self._show_enhanced:
            self._show_enhanced = False
            self._enhance_requested = False
            self.enhance_toggled.emit(False)
            self.update()
            return False
        else:
            if self._enhanced_pixmap is not None:
                self._show_enhanced = True
                self.enhance_toggled.emit(True)
                self.update()
                return True
            # 需要生成：**后台**执行，完成后自动切到增强显示。
            # 原先这里同步调用 LibLLIE + numpy，每次切图提交都会冻结 GUI（审计 D17）。
            self._enhance_requested = True
            if not self._generate_enhanced_pixmap():
                self._enhance_requested = False
                return "错误: 增强图生成失败，请检查图像是否有效"
            self.status_text = "正在生成增强图..."
            self.update()
            return "增强图正在后台生成，完成后自动切换显示"

    # ---- 增强图：后台生成 + 结果校验 + 代次守卫 ----

    @staticmethod
    def _normalize_enhanced_array(arr):
        """把增强器返回值规范成 (h, w, 3) 的 uint8 连续数组。

        审计 D17 的第二处缺陷：原实现直接 `h, w, ch = enhanced_arr.shape` 并用
        `ch * w` 当 bytesPerLine 构造 QImage，对 dtype/通道数/连续性**零校验**
        —— 返回 float32/uint16 时行距与真实不符，会逐行越界读 numpy 缓冲（花屏或崩溃）。
        """
        out = np.asarray(arr)
        if out.ndim == 3 and out.shape[2] == 4:      # RGBA -> RGB
            out = out[:, :, :3]
        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError("增强结果形状非法: %r" % (getattr(out, "shape", None),))
        if out.shape[0] <= 0 or out.shape[1] <= 0:
            raise ValueError("增强结果尺寸非法: %r" % (out.shape,))
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)

    def _ensure_enhancer(self):
        """延迟创建增强器（只会在增强工作线程里被调用，单线程串行）。"""
        if self._llie_enhancer is None:
            from libllie.traditional.algorithms import LLIEnhancer
            self._llie_enhancer = LLIEnhancer.create_enhancer('clahe', output_type='numpy')
        return self._llie_enhancer

    def _enhance_image_blocking(self, image, generation, path):
        """在**工作线程**里执行增强。返回 (generation, path, enhanced_arr)。

        QImage 是可重入的值类型、这里只读，且提交只会**替换引用**而不会改写对象，
        因此跨线程读取安全（工作线程自己持有引用，对象不会被回收）。
        """
        img = image
        if img.format() != QImage.Format.Format_RGB888:
            img = img.convertToFormat(QImage.Format.Format_RGB888)
        width, height = img.width(), img.height()
        bpl = img.bytesPerLine()          # 行对齐后的字节数，可能大于 width*3
        ptr = img.constBits()
        if hasattr(ptr, 'setsize'):
            ptr.setsize(height * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, bpl))[:, :width * 3]
        arr = arr.reshape((height, width, 3))
        enhanced = self._ensure_enhancer()(arr)
        return generation, path, self._normalize_enhanced_array(enhanced)

    def _generate_enhanced_pixmap(self):
        """请求生成增强图（**后台**，CPU 密集）。返回是否已受理。"""
        session = self.session
        if session is None or not session.is_committed or session.image is None:
            self._enhanced_pixmap = None
            return False
        if getattr(self, "_enhancer_unavailable", False):
            return False
        request = {"generation": session.generation, "path": session.path}
        runner = self._enhancer_runner
        if runner.busy():
            self._pending_enhance_request = request      # 最新优先
            return True
        return self._start_enhance(request, session.image)

    def _start_enhance(self, request, image):
        started = self._enhancer_runner.start(
            lambda: self._enhance_image_blocking(image, request["generation"],
                                                 request["path"]),
            on_result=self._on_enhance_ready,
            on_error=self._on_enhance_failed,
        )
        if not started:
            self._pending_enhance_request = request
        return started

    def _on_enhance_ready(self, payload):
        try:
            generation, path, enhanced = payload
            session = self.session
            # 代次 + 路径双重守卫：过期结果绝不安装（否则会把 A 图的增强图贴到 B 图上）
            if session is None or generation != session.generation or path != session.path:
                print(f"[DrawingWidget] 丢弃过期增强结果: {path} (代次 {generation})")
            else:
                height, width = enhanced.shape[0], enhanced.shape[1]
                # 已经是连续 uint8 RGB，行距就是 3*width，不会越界
                qimg = QImage(enhanced.data, width, height, 3 * width,
                              QImage.Format.Format_RGB888)
                self._enhanced_pixmap = QPixmap.fromImage(qimg.copy())
                if self._enhance_requested:
                    self._show_enhanced = True
                    self.enhance_toggled.emit(True)
                self.status_text = None
                self.update()
        finally:
            self._drain_pending_enhance_request()

    def _on_enhance_failed(self, message):
        print(f"[DrawingWidget] 图像增强失败: {message}")
        # 增强器不可用（例如缺少 libllie）时不要每次切图都重试一遍
        if "libllie" in message or "ImportError" in message:
            self._enhancer_unavailable = True
            print("[DrawingWidget] 增强器不可用，本次会话内不再尝试")
        self._enhance_requested = False
        self.status_text = None
        self.update()
        self._drain_pending_enhance_request()

    def _drain_pending_enhance_request(self):
        pending = self._pending_enhance_request
        self._pending_enhance_request = None
        if pending is None:
            return
        session = self.session
        if session is None or pending["generation"] != session.generation \
                or pending["path"] != session.path or session.image is None:
            return                                   # 过期请求直接丢弃
        self._start_enhance(pending, session.image)

    def _is_generation_current(self, generation):
        """代次权威的唯一出口。

        会被解码工作线程只读调用：只读取一个 Python int（GIL 保证原子性），
        不触碰任何 Qt 对象，因此跨线程安全。权威在 session。
        """
        return generation == self.generation

    def _reset_transient_drag_state(self):
        """清掉全部"下标索引 / 绑定坐标"的瞬时交互状态（审计 D7 / D16）。

        这些状态在切图时若不清，就会与新图的标注和度量结合使用：
          * 按住左键切图后释放 -> 用旧图坐标向新图提交标注，而
            `annotation_modified` 会立即无条件落盘、撤销栈又已被清空 -> 静默数据损坏；
          * `batch_selected_points` 指向新图里另一个顶点 -> 按 Delete 删错顶点；
          * `hover_point` / `obb_points` / `is_near_first_point` 让下一次点击落在错对象上。
        因此任何"图像被替换"的时刻（set_image / clear / 提交释放被拒）都必须调用它。
        """
        self.rect_start = None
        self.rect_end = None
        self.current_poly = []
        self.obb_points = []
        self.sam_points = []
        self.dragging_annotation = None
        self.drag_start_img_pos = None
        self.dragging_point = None
        self.batch_selecting = False
        self.batch_selection_start = None
        self.batch_selection_end = None
        self.batch_selected_indices = []
        self.batch_selected_points = []
        self.right_button_held = False
        self.right_press_pos = None
        self._right_dragging = False
        self.right_press_time = 0
        self.hover_point = None
        self.is_near_first_point = False
        self.ignore_next_left_press = False
        self.hover_index = -1
        self.selected_index = -1
        self.focused_index = -1

    def cancel_pending_load(self):
        """作废所有在途解码结果（非阻塞）。

        线程由 QThreadPool 持有，因此这里既不需要也不允许 wait/terminate：
        递增代次后，在途任务在提交结果前会自行发现已过期并丢弃。
        """
        session = self.session
        if session is not None:
            session.cancel_pending_load()

    def drain(self, timeout_ms=5000):
        """确定性排空后台解码（关闭 / 插件卸载路径调用）。

        先作废在途结果让任务尽快返回，再等待线程池排空。返回是否在超时内排空。
        只能在 GUI 线程调用。实现 `Drainable` 契约，供应用级关闭协调器统一调用。
        """
        self.cancel_pending_load()
        ok = self._decoder.drain(timeout_ms)
        if not ok:
            print(f"[DrawingWidget] 解码线程池未在 {timeout_ms}ms 内排空")
        # 增强图线程池也必须在关闭前排空（它同样会持有 numpy/QImage 内存）
        if not self._enhancer_runner.drain(timeout_ms):
            print(f"[DrawingWidget] 增强线程未在 {timeout_ms}ms 内排空")
        return ok

    def wait_io_idle(self, timeout_ms=3000):
        """等待在途解码结束但**不作废代次、不取消任何请求**。

        专供"需要释放图片文件句柄"的正常交互路径（例如删除图片文件）调用：
        Windows 上解码进行中 `os.remove` 会 WinError 32。与 drain() 的区别是
        它没有副作用——不会让当前正在加载的图片被丢弃。
        """
        return self._decoder.wait_idle(timeout_ms)

    def set_image(self, image_path):
        """开始加载新图 —— **唯一提交入口**。

        `session.begin_load()` 换代次、改路径，并**清空"已提交"的像素与度量**。
        因此在新图提交之前 `is_committed` 为 False，交互与坐标换算全部被闸门挡住，
        不会出现"图 N 的标注 + 图 N-1 的度量"这种会算错坐标的状态（审计 W2）。
        这里不再有"丢弃运行中的 QThread"那一步 —— abort 的根因就在那里。
        """
        session = self.session
        if session is None:
            return
        generation = session.begin_load(image_path)

        # 清掉全部瞬时交互状态：切图后它们指向的是**上一张图**的标注下标与坐标
        self._reset_transient_drag_state()
        self.status_text = None

        # 切换图片时清除增强缓存（保留增强状态，新图加载后自动生成增强图）
        self._enhanced_pixmap = None

        try:
            self._decoder.request(generation, image_path)
        except Exception as e:
            print(f"[DrawingWidget] Error requesting image decode: {e}")
            traceback.print_exc()

    def _on_image_decode_failed(self, generation, path, message):
        # 过期失败不打扰用户（快速切图时会自然产生一批）
        if generation != self.generation:
            return
        print(f"[DrawingWidget] Image load error: {path}: {message}")
        self.update()

    def set_qimage(self, q_image):
        """用现成的 QImage 直接提交，取代当前图。

        走 session 的"换代次 + 原子提交"：`begin_load()` 先作废在途解码，因此
        此前同代次的解码结果不可能再覆盖它（审计指出原先 set_qimage 不换代次，
        在途解码到达时会静默覆盖刚设的图）。
        """
        if q_image.isNull():
            return
        session = self.session
        if session is None:
            return
        generation = session.begin_load(session.path)
        pixmap = QPixmap.fromImage(q_image)
        if not session.commit_pixels(generation, q_image, pixmap, q_image.size(), 1.0):
            return
        self.sam_points = []
        self.current_poly = []
        self.status_text = None
        # 增强了就保持增强：_enhance_requested 表示用户已请求，跨切图继续生成
        if self._show_enhanced or self._enhance_requested:
            self._generate_enhanced_pixmap()
        self.reset_view()
        self.update()

    def _on_image_decoded(self, generation, path, image, scale, original_size):
        """解码完成（GUI 线程）：代次与路径双重守卫通过后**原子提交**像素与度量。"""
        # 清理中：画布状态正在被重置，丢弃结果
        if self._is_clearing:
            return

        # 代次守卫：快速切图时必然有一批过期结果到这里，必须丢弃
        if generation != self.generation:
            return
        # 路径守卫（双保险）：代次与目标图必须同时一致，
        # 避免任何"把 A 的像素提交到 B"的可能
        if path != self.current_image_path:
            return

        session = self.session
        if session is None:
            return

        try:
            if image is None:
                print("[DrawingWidget] _on_image_decoded: image is None")
                return
            if hasattr(image, 'isNull') and image.isNull():
                print("[DrawingWidget] _on_image_decoded: image is null")
                return

            # 原子提交：像素 + 度量 + 已提交代次一次性落地；任一代次不符则整体丢弃
            if not session.commit_pixels(generation, image, QPixmap.fromImage(image),
                                         original_size, scale):
                return

            # 增强了就保持增强：_enhance_requested 表示用户已请求，跨切图继续生成
            if self._show_enhanced or self._enhance_requested:
                self._generate_enhanced_pixmap()

            self.reset_view()
            self.update()

            # 发送图片加载完成信号
            self.image_loaded.emit()
        except Exception as e:
            print(f"[DrawingWidget] Error in _on_image_decoded: {e}")
            traceback.print_exc()

    def clear(self):
        """Clears the canvas and all associated data.

        非阻塞：递增代次就让所有在途解码作废，这里既不等待也不强杀线程
        （原先的 wait(500) -> terminate() 会在半解码状态下杀死持有 Qt 图像
        插件内部锁的线程，后续解码可能死锁）。确定性排空由 drain() 在关闭路径负责。
        """
        self._is_clearing = True
        try:
            session = self.session
            if session is not None:
                session.clear()
            self._reset_transient_drag_state()
            self.status_text = None
            self._enhanced_pixmap = None
            if self._show_enhanced:
                self._show_enhanced = False
                self.enhance_toggled.emit(False)
            self.update()
        finally:
            self._is_clearing = False

    @action("interact.reset_view", description="重置画布视图，使图像完整居中显示。\n- 恢复默认缩放比例，图像居中适配画布大小\n- 需要先加载图像\n- 也可用于取消标注聚焦状态", category="画布", scope="agent")
    def reset_view(self):
        if not self.image or not self.pixmap:
            return "错误: 没有加载图像，无法重置视图"
        win_w, win_h = self.width(), self.height()
        img_w, img_h = self.original_image_size.width(), self.original_image_size.height()

        if img_w <= 0 or img_h <= 0:
            return "错误: 图像尺寸无效"

        if win_w <= 0 or win_h <= 0:
            self.zoom = 1.0
        else:
            self.zoom = max(0.01, min(win_w / img_w, win_h / img_h) * 0.9)

        self.pan_offset = QPointF(
            (win_w - img_w * self.zoom) / 2,
            (win_h - img_h * self.zoom) / 2
        )
        self.update()

    def get_transform(self):
        transform = QTransform()
        transform.translate(self.pan_offset.x(), self.pan_offset.y())
        transform.scale(self.zoom, self.zoom)
        return transform

    def to_image_coords(self, pos):
        # 未提交时**不做**坐标换算：此刻 original_image_size / zoom / pan_offset
        # 都不属于当前图，算出来的坐标会落成错标（审计 W2 的根治点）。
        # 调用方本来就有 None 分支；paintEvent 的 "pixmap is None 就返回" 与本闸门
        # 等价（commit_pixels 同时设置 pixmap 与 committed_generation）。
        if not self._interaction_ready():
            return None
        inv_transform, ok = self.get_transform().inverted()
        if not ok: return None
        p = inv_transform.map(QPointF(pos))
        return p

    def from_image_coords(self, p):
        return self.get_transform().map(QPointF(p))

    @action("interact.set_drawing_mode", description="切换画布绘制模式。\n- mode 可选值：\n  - rect: 手动绘制矩形框\n  - polygon: 手动绘制多边形（配合 add_polygon_vertex / close_polygon）\n  - obb: 旋转框（带角度矩形）\n  - roi: 设置感兴趣区域\n  - ai_rect: AI辅助矩形检测\n  - sam: 智能分割（配合 add_sam_point / confirm_sam_annotation）\n  - edit: 编辑模式（选中/修改已有标注）\n- 切换后立即生效\n- 大部分绘制操作要求先切换到对应模式", category="画布", params={"mode": "str"}, scope="agent")
    def set_mode(self, mode):
        if isinstance(mode, str):
            mode_map = {
                'edit': DrawingMode.EDIT, 'rect': DrawingMode.RECT,
                'polygon': DrawingMode.POLYGON, 'sam': DrawingMode.SAM,
                'obb': DrawingMode.OBB, 'roi': DrawingMode.ROI,
                'ai_rect': DrawingMode.AI_RECT,
            }
            mapped = mode_map.get(mode.lower())
            if mapped is None:
                return "错误: 不支持的模式 '%s'，可选值: %s" % (mode, ', '.join(mode_map.keys()))
            mode = mapped
        self.mode = mode
        if mode in [DrawingMode.RECT, DrawingMode.POLYGON, DrawingMode.OBB]:
            self.last_drawing_mode = mode
        if mode == DrawingMode.EDIT:
            self.setCursor(Qt.ArrowCursor)
        elif mode == DrawingMode.SAM:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)
        self.mode_changed.emit(mode)
        self.update()
        return {"status": "success", "message": "绘制模式已切换",
                "mode": mode.name.lower() if hasattr(mode, 'name') else str(mode)}

    def zoom_to_roi(self):
        """Zooms and pans the view to fit the persistent ROI."""
        if not self.image or not self.persistent_roi:
            return

        px, py, pw, ph = self.persistent_roi
        if pw <= 0 or ph <= 0:
            return

        win_w, win_h = self.width(), self.height()

        # Calculate required zoom to fit ROI with some padding
        padding = 1.1
        self.zoom = min(win_w / (pw * padding), win_h / (ph * padding))

        # Calculate pan to center ROI
        self.pan_offset = QPointF(
            win_w / 2 - (px + pw / 2) * self.zoom,
            win_h / 2 - (py + ph / 2) * self.zoom
        )
        self.update()

    def toggle_roi_zoom(self):
        # 未提交时 original_image_size 是 QSize(0,0)，下面按它做的比值会出现除零
        # 与奇异变换。虽然 to_image_coords 的闸门已经让这条路径短路，这里仍显式拒绝，
        # 避免将来有人改动顺序后重新引入 ZeroDivisionError。
        if not self._interaction_ready():
            return "错误: 图像尚未加载完成，无法切换 ROI 缩放"
        """Toggles between ROI zoom and full view."""
        if not self.persistent_roi:
            self.reset_view()
            return

        # Check if we are currently zoomed into ROI (roughly)
        px, py, pw, ph = self.persistent_roi
        center_x = px + pw / 2
        center_y = py + ph / 2

        current_center_in_img = self.to_image_coords(QPoint(self.width() // 2, self.height() // 2))

        is_zoomed = False
        if current_center_in_img:
            dist = np.sqrt((current_center_in_img.x() - center_x)**2 + (current_center_in_img.y() - center_y)**2)
            # If center is close and zoom is higher than reset zoom
            if dist < max(pw, ph) * 0.1 and self.zoom > 0.5 * min(self.width() / self.original_image_size.width(), self.height() / self.original_image_size.height()):
                is_zoomed = True

        if is_zoomed:
            self.reset_view()
        else:
            self.zoom_to_roi()

    def paintEvent(self, event):
        # 如果正在清理，直接返回，避免访问无效数据
        if self._is_clearing:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if not self.pixmap:
            # 空画布填充与界面/侧栏背景同色（#1a1a1a），保证无接缝色差
            painter.fillRect(self.rect(), QColor(26, 26, 26))
            if self._decoder.is_busy():
                painter.setPen(Qt.GlobalColor.white)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "正在加载超大图像...")
            return

        # Defensive zoom check to avoid ZeroDivisionError
        safe_zoom = max(0.0001, self.zoom)
        painter.setTransform(self.get_transform())

        # 1. Draw Image (scale proxy to original size)
        display_pixmap = self._enhanced_pixmap if self._show_enhanced and self._enhanced_pixmap else self.pixmap
        painter.drawPixmap(0, 0, self.original_image_size.width(), self.original_image_size.height(), display_pixmap)

        # 2. Draw Annotations
        if self.show_bboxes:
            for i, ann in enumerate(self.annotations):
                # 跳过非 dict 标注和隐藏的标注
                if not isinstance(ann, dict):
                    continue
                if i in self.hidden_indices:
                    continue

                is_hover = (i == self.hover_index)
                is_selected = (i == self.selected_index)
                label_name = ann.get('label', 'unknown')

                # Get color from category or default
                base_color = self.category_colors.get(label_name, QColor(0, 255, 255))

                # Highlight logic
                is_selected_or_hover = is_selected or is_hover

                if is_selected:
                    fill_alpha = 100 # Transparent fill when selected
                    edge_color = QColor(255, 200, 0, 255)
                    fill_color = QColor(255, 200, 0, fill_alpha)
                elif is_hover:
                    fill_alpha = 80
                    edge_color = QColor(255, 50, 50, 220)
                    fill_color = QColor(255, 50, 50, fill_alpha)
                else:
                    fill_alpha = 50
                    edge_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 200)
                    fill_color = QColor(base_color.red(), base_color.green(), base_color.blue(), fill_alpha)

                # Draw Polygons (Masks or OBB) - based on annotation content, not global task_mode
                # rectangle类型优先使用bbox渲染，即使有polygons数据也不画多边形
                # `any(len(p) >= 3 ...)` 对畸形元素（如 polygons=[1.0, 2.0] 或 [None]）会抛
                # TypeError —— 这里在 paintEvent 内，异常会让画布永久停止重绘（审计 D18）。
                # 因此显式判断元素是否"像一条多边形"（可 len 的可迭代对象）。
                raw_polys = ann.get('polygons')
                has_polygons = bool(raw_polys) and ann.get('shape_type') != 'rectangle' \
                    and any(self._is_polygon_like(p) for p in raw_polys)
                if has_polygons:
                    painter.setPen(QPen(edge_color, 2 / safe_zoom))
                    painter.setBrush(fill_color)
                    for poly_idx, poly in enumerate(ann.get('polygons', [])):
                        qpoly = QPolygonF()
                        for pt_idx, pt in enumerate(poly):
                            # Ensure pt is a point
                            try:
                                if hasattr(pt, '__len__') and len(pt) >= 2:
                                    if hasattr(pt[0], '__len__'):
                                        x, y = pt[0][0], pt[0][1]
                                    else:
                                        x, y = pt[0], pt[1]
                                    qp = QPointF(float(x), float(y))
                                    qpoly.append(qp)

                                    # Enhanced vertices display when selected/hovered or mouse is near
                                    is_point_hover = (self.hover_point == (i, poly_idx, pt_idx)) or (self.dragging_point == (i, poly_idx, pt_idx))
                                    if self.interactive and (is_selected_or_hover or self.mode == DrawingMode.EDIT or is_point_hover):
                                        painter.save()
                                        # Larger and brighter if hovering near this specific point
                                        if is_point_hover:
                                            painter.setBrush(QColor(255, 255, 0))
                                            r = 8 / safe_zoom  # Increased from 7
                                            painter.setPen(QPen(Qt.GlobalColor.white, 3 / safe_zoom)) # Increased from 2
                                        else:
                                            painter.setBrush(Qt.GlobalColor.white)
                                            r = 4 / safe_zoom
                                            painter.setPen(QPen(edge_color.darker(), 1 / safe_zoom))

                                        painter.drawEllipse(qp, r, r)
                                        painter.restore()
                            except:
                                if settings.DEBUG:
                                    raise
                                continue
                        painter.drawPolygon(qpoly)

                # Draw Bbox
                # 必须用**校验过的** bbox（审计 D18）：`ann.get('bbox', [0,0,0,0])` 只在
                # 键**缺失**时给默认值，键存在但为 None / 长度不对时依然返回坏值，
                # 解包即抛 TypeError。而 paintEvent 抛异常会让**画布永久停止重绘**。
                bx, by, bw, bh = self._safe_bbox(ann)
                painter.setPen(QPen(edge_color, 2 / safe_zoom))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if not has_polygons:
                    # Only fill bbox for pure rectangle annotations (no polygons)
                    if is_selected_or_hover:
                        painter.setBrush(fill_color)
                painter.drawRect(QRectF(bx, by, bw, bh))

                # Draw Bbox Corners if selected or hovered or mouse is near corner
                if self.interactive and (is_selected_or_hover or self.mode == DrawingMode.EDIT or (self.hover_point and self.hover_point[0] == i and self.hover_point[1] == -1) or (self.dragging_point and self.dragging_point[0] == i and self.dragging_point[1] == -1)):
                    corners = [(bx, by), (bx+bw, by), (bx+bw, by+bh), (bx, by+bh)]
                    for pt_idx, (px, py) in enumerate(corners):
                        is_point_hover = (self.hover_point == (i, -1, pt_idx)) or (self.dragging_point == (i, -1, pt_idx))
                        painter.save()
                        if is_point_hover:
                            painter.setBrush(QColor(255, 255, 0))
                            r = 8 / safe_zoom
                            painter.setPen(QPen(Qt.GlobalColor.white, 3 / safe_zoom))
                        else:
                            painter.setBrush(Qt.GlobalColor.white)
                            r = 4 / safe_zoom
                            painter.setPen(QPen(edge_color.darker(), 1 / safe_zoom))
                        painter.drawEllipse(QPointF(px, py), r, r)
                        painter.restore()

                # Draw Label Text - Positioned above the mask/bbox
                if self.show_labels:
                    bx, by, bw, bh = self._safe_bbox(ann)
                    painter.save()

                    # 判断是否有副标注匹配
                    has_secondary = i in self.secondary_annotation_map

                    # 构建标签文本，添加主/副标识
                    if self.secondary_annotations:
                        if has_secondary:
                            sec_idx = self.secondary_annotation_map[i]
                            sec_label = "unknown"
                            if 0 <= sec_idx < len(self.secondary_annotations):
                                sec_label = self.secondary_annotations[sec_idx].get('label', 'unknown')
                            if sec_label and sec_label != label_name:
                                display_label = f"[主+副] {label_name} + {sec_label}"
                            else:
                                display_label = f"[主+副] {label_name}"
                        else:
                            display_label = f"[主] {label_name}"
                    else:
                        display_label = label_name

                    # Calculate text size for better background sizing
                    font = painter.font()
                    font.setBold(True)
                    font.setPixelSize(14)
                    painter.setFont(font)

                    metrics = painter.fontMetrics()
                    text_w = metrics.horizontalAdvance(display_label) + 10
                    text_h = metrics.height() + 4

                    # Position label tightly above bbox
                    label_x = bx
                    label_y = by - text_h / safe_zoom

                    painter.translate(label_x, label_y)
                    painter.scale(1/safe_zoom, 1/safe_zoom)

                    # Label box width should be at least as wide as text
                    label_w = max(bw * safe_zoom, text_w)
                    label_rect = QRectF(0, 0, label_w, text_h)
                    # Use solid color for label background
                    painter.fillRect(label_rect, QColor(base_color.red(), base_color.green(), base_color.blue(), 220))
                    painter.setPen(Qt.GlobalColor.white)
                    # Left aligned text with some padding
                    text_rect = label_rect.adjusted(5, 0, 0, 0)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, display_label)
                    painter.restore()

        # 2.5 Draw unmatched secondary annotations (纯副标注)
        if self.secondary_annotations and self.show_bboxes:
            matched_secondary_indices = set(self.secondary_annotation_map.values())

            for j, s_ann in enumerate(self.secondary_annotations):
                if j in matched_secondary_indices:
                    continue

                label_name = s_ann.get('label', 'unknown')
                display_label = f"[副] {label_name}"

                # 副标注使用蓝色
                edge_color = QColor(100, 100, 255, 200)
                fill_color = QColor(100, 100, 255, 50)

                # Draw Polygons (Masks or OBB) - based on annotation content
                # rectangle类型优先使用bbox渲染，即使有polygons数据也不画多边形
                s_has_polygons = (s_ann.get('polygons') and any(len(p) >= 3 for p in s_ann.get('polygons', []))
                                  and s_ann.get('shape_type') != 'rectangle')
                if s_has_polygons:
                    painter.setPen(QPen(edge_color, 2 / safe_zoom))
                    painter.setBrush(fill_color)
                    for poly in s_ann.get('polygons', []):
                        qpoly = QPolygonF()
                        for pt in poly:
                            try:
                                if hasattr(pt, '__len__') and len(pt) >= 2:
                                    if hasattr(pt[0], '__len__'):
                                        x, y = pt[0][0], pt[0][1]
                                    else:
                                        x, y = pt[0], pt[1]
                                    qpoly.append(QPointF(float(x), float(y)))
                            except:
                                continue
                        if len(qpoly) > 0:
                            painter.drawPolygon(qpoly)

                # 获取 bbox，如果没有则从 polygons 计算
                bx, by, bw, bh = s_ann.get('bbox', [0, 0, 0, 0])
                if bw == 0 or bh == 0:
                    # 从 polygons 计算 bbox
                    all_points = []
                    for poly in s_ann.get('polygons', []):
                        for pt in poly:
                            try:
                                if hasattr(pt, '__len__') and len(pt) >= 2:
                                    if hasattr(pt[0], '__len__'):
                                        all_points.append((pt[0][0], pt[0][1]))
                                    else:
                                        all_points.append((pt[0], pt[1]))
                            except:
                                continue
                    if all_points:
                        xs = [p[0] for p in all_points]
                        ys = [p[1] for p in all_points]
                        bx = min(xs)
                        by = min(ys)
                        bw = max(xs) - bx
                        bh = max(ys) - by

                # Draw Bbox
                if bw > 0 and bh > 0:
                    painter.setPen(QPen(edge_color, 2 / safe_zoom))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(bx, by, bw, bh))

                # Draw Label
                if self.show_labels and bw > 0 and bh > 0:
                    painter.save()
                    font = painter.font()
                    font.setBold(True)
                    font.setPixelSize(14)
                    painter.setFont(font)

                    metrics = painter.fontMetrics()
                    text_w = metrics.horizontalAdvance(display_label) + 10
                    text_h = metrics.height() + 4

                    label_x = bx
                    label_y = by - text_h / safe_zoom

                    painter.translate(label_x, label_y)
                    painter.scale(1/safe_zoom, 1/safe_zoom)

                    label_w = max(bw * safe_zoom, text_w)
                    label_rect = QRectF(0, 0, label_w, text_h)
                    painter.fillRect(label_rect, QColor(100, 100, 255, 220))
                    painter.setPen(Qt.GlobalColor.white)
                    text_rect = label_rect.adjusted(5, 0, 0, 0)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, display_label)
                    painter.restore()

        # 3. Draw SAM points and temporary mask
        if self.mode == DrawingMode.SAM and self.sam_points:
            # We can draw something special for the active SAM session if needed
            # But currently they are already in self.annotations as a "preview"
            pass

        for x, y, label in self.sam_points:
            p_color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
            painter.setPen(QPen(Qt.GlobalColor.white, 1 / safe_zoom))
            painter.setBrush(p_color)
            radius = 5 / safe_zoom
            painter.drawEllipse(QPointF(x, y), radius, radius)

        # 4. Draw Persistent ROI (if exists)
        if self.persistent_roi:
            px, py, pw, ph = self.persistent_roi
            painter.setPen(QPen(QColor(255, 165, 0), 3 / safe_zoom, Qt.PenStyle.DashLine)) # Thick orange dashed line
            painter.setBrush(QColor(255, 165, 0, 30))
            painter.drawRect(QRectF(px, py, pw, ph))

            # Label for ROI
            painter.setBrush(QColor(255, 165, 0))
            painter.setPen(Qt.GlobalColor.white)
            font = painter.font()
            font.setPixelSize(int(14 / safe_zoom))
            painter.setFont(font)
            painter.drawText(QPointF(px, py - 5 / safe_zoom), "ACTIVE ROI")

        # 5. Draw Current Drawing (Rect or Poly or ROI)
        if self.mode == DrawingMode.ROI:
            painter.setPen(QPen(QColor(255, 165, 0), 2 / safe_zoom)) # Orange for ROI
            painter.setBrush(QColor(255, 165, 0, 50))
        else:
            painter.setPen(QPen(QColor(0, 255, 255), 2 / safe_zoom)) # Cyan for others
            painter.setBrush(QColor(0, 255, 255, 50))

        if self.mode in [DrawingMode.RECT, DrawingMode.AI_RECT, DrawingMode.ROI] and self.rect_start and self.rect_end:
            r = QRectF(self.rect_start, self.rect_end).normalized()
            # AI_RECT 模式使用不同的颜色
            if self.mode == DrawingMode.AI_RECT:
                painter.setPen(QPen(QColor(0, 200, 255), 2 / safe_zoom))  # 亮蓝色
                painter.setBrush(QColor(0, 200, 255, 30))
            painter.drawRect(r)
        elif self.mode == DrawingMode.POLYGON and self.current_poly:
            qpoly = QPolygonF()
            for pt in self.current_poly:
                qpoly.append(pt)
            # Draw lines
            painter.drawPolyline(qpoly)
            # Draw points
            for i, pt in enumerate(self.current_poly):
                if i == 0 and self.is_near_first_point:
                    # Highlight first point when mouse is near
                    painter.setBrush(QColor(255, 255, 0))
                    painter.setPen(QPen(Qt.GlobalColor.white, 2 / safe_zoom))
                    radius = 8 / safe_zoom
                    painter.drawEllipse(pt, radius, radius)
                else:
                    painter.setBrush(Qt.GlobalColor.white)
                    painter.setPen(QPen(QColor(0, 255, 255), 1 / safe_zoom))
                    radius = 3 / safe_zoom
                    painter.drawEllipse(pt, radius, radius)
        elif self.mode == DrawingMode.OBB and self.obb_points:
            # Draw OBB preview
            painter.setPen(QPen(QColor(0, 255, 255), 2 / safe_zoom))
            painter.setBrush(QColor(0, 255, 255, 50))

            pts = self.obb_points
            curr_pos = self.to_image_coords(self.mapFromGlobal(QCursor.pos()))

            if len(pts) == 1 and curr_pos:
                # Just the first line
                painter.drawLine(pts[0], curr_pos)
                painter.drawEllipse(pts[0], 3/safe_zoom, 3/safe_zoom)
            elif len(pts) == 2 and curr_pos:
                # The full rectangle
                p1, p2 = pts[0], pts[1]
                p3 = curr_pos

                # Vector p1->p2
                v12 = np.array([p2.x() - p1.x(), p2.y() - p1.y()])
                len12 = np.linalg.norm(v12)
                if len12 > 1e-6:
                    # Unit vector p1->p2
                    u12 = v12 / len12
                    # Unit vector perpendicular to p1->p2
                    u_perp = np.array([-u12[1], u12[0]])

                    # Vector p1->p3
                    v13 = np.array([p3.x() - p1.x(), p3.y() - p1.y()])
                    # Projection of v13 onto u_perp gives the "height"
                    dist = np.dot(v13, u_perp)

                    # The four corners
                    p4 = p1 + QPointF(u_perp[0] * dist, u_perp[1] * dist)
                    p3_rect = p2 + QPointF(u_perp[0] * dist, u_perp[1] * dist)

                    qpoly = QPolygonF([p1, p2, p3_rect, p4])
                    painter.drawPolygon(qpoly)
                else:
                    painter.drawLine(p1, p3)

        # 6. Draw batch selection rectangle and selected items
        # 显示选择框（正在选择时）或选中状态（选择完成后）
        if self.batch_selecting and self.batch_selection_start and self.batch_selection_end:
            r = QRectF(self.batch_selection_start, self.batch_selection_end).normalized()
            painter.setPen(QPen(QColor(255, 100, 255), 2 / safe_zoom, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 100, 255, 30))
            painter.drawRect(r)

        # 显示选中的标注和点（无论是否正在选择）
        if self.batch_selected_indices or self.batch_selected_points:
            # Highlight selected annotations
            for idx in self.batch_selected_indices:
                if idx < len(self.annotations):
                    ann = self.annotations[idx]
                    bx, by, bw, bh = ann.get('bbox', [0, 0, 0, 0])
                    painter.setPen(QPen(QColor(255, 100, 255), 3 / safe_zoom))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(bx, by, bw, bh))

            # Highlight selected points
            for ann_idx, poly_idx, pt_idx in self.batch_selected_points:
                if ann_idx < len(self.annotations):
                    ann = self.annotations[ann_idx]
                    polys = ann.get('polygons', [])
                    if poly_idx < len(polys):
                        poly = polys[poly_idx]
                        if pt_idx < len(poly):
                            pt = poly[pt_idx]
                            try:
                                if hasattr(pt, '__len__') and len(pt) >= 2:
                                    if hasattr(pt[0], '__len__'):
                                        x, y = pt[0][0], pt[0][1]
                                    else:
                                        x, y = pt[0], pt[1]
                                    painter.setBrush(QColor(255, 100, 255))
                                    painter.setPen(QPen(Qt.GlobalColor.white, 2 / safe_zoom))
                                    painter.drawEllipse(QPointF(x, y), 6 / safe_zoom, 6 / safe_zoom)
                            except:
                                pass

        # Reset transform for UI elements
        painter.setTransform(QTransform())

        if self.status_text:
            painter.save()
            font = painter.font()
            font.setBold(True)
            font.setPixelSize(26)
            painter.setFont(font)

            metrics = painter.fontMetrics()
            text = self.status_text
            text_w = metrics.horizontalAdvance(text)
            text_h = metrics.height()

            padding_x = 14
            padding_y = 8
            margin = 20

            box_w = text_w + padding_x * 2
            box_h = text_h + padding_y * 2

            x = self.width() - box_w - margin
            y = margin

            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x, y, box_w, box_h, 10, 10)

            painter.setPen(self.status_color)
            text_x = x + padding_x
            text_y = y + padding_y + metrics.ascent()
            painter.drawText(text_x, text_y, text)
            painter.restore()



        # 5. Draw Crosshair
        local_pos = self.mapFromGlobal(QCursor.pos())
        if self.interactive and self.rect().contains(local_pos):
            painter.setPen(QPen(QColor(0, 255, 255, 150), 1))
            painter.drawLine(0, local_pos.y(), self.width(), local_pos.y())
            painter.drawLine(local_pos.x(), 0, local_pos.x(), self.height())

    def wheelEvent(self, event):
        # 未提交（无图或在途解码）时不改 zoom/pan：否则会被随后的提交静默覆盖，
        # 用户表现为"滚轮没反应"但状态已被污染
        if not self._interaction_ready():
            return
        # Zoom at mouse position
        # Use position() for PySide6 compatibility (pos() is deprecated)
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        old_image_pos = self.to_image_coords(mouse_pos)

        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.zoom *= factor
        self.zoom = max(0.1, min(self.zoom, 50.0))

        # Adjust pan to keep mouse over same image spot
        new_image_pos = self.to_image_coords(mouse_pos)
        if old_image_pos and new_image_pos:
            delta = (new_image_pos - old_image_pos) * self.zoom
            self.pan_offset += delta

        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.hover_index != -1:
                self.annotation_double_clicked.emit(self.hover_index)
                return
        super().mouseDoubleClickEvent(event)

    def is_inside_roi(self, img_pos):
        """Checks if a point in image coordinates is inside the persistent ROI"""
        if not self.persistent_roi:
            return True # No ROI means everything is valid
        rx, ry, rw, rh = self.persistent_roi
        return rx <= img_pos.x() <= rx + rw and ry <= img_pos.y() <= ry + rh

    def _show_context_menu(self, global_pos, img_pos):
        """显示右键上下文菜单"""
        # 0. Check if clicking on the persistent ROI (no annotation hovered)
        if self.persistent_roi and self.hover_index == -1:
            rx, ry, rw, rh = self.persistent_roi
            if rx <= img_pos.x() <= rx + rw and ry <= img_pos.y() <= ry + rh:
                self.roi_menu_requested.emit(global_pos)
                return

        # 1. Highest Priority: SAM confirmation if points exist
        if self.mode == DrawingMode.SAM and self.sam_points:
            self.confirm_sam_annotation()  # 通过 @action 方法执行并录制
            return

        # 2. Second Priority: Target-specific context menu (vertex, edge, or annotation)
        target_idx = self.hover_index
        # Check polygon points for any annotation that has polygons
        if True:
            near_pt = self.find_near_point(img_pos, max_dist_screen=16.0)
            if near_pt:
                ann_idx, poly_idx, pt_idx = near_pt
                if poly_idx != -1: # Only for polygons, not bbox corners
                    self.context_vertex_info = (ann_idx, poly_idx, pt_idx)
                    target_idx = ann_idx # Ensure we show menu for this annotation

            near_edge = self.find_near_edge(img_pos)
            if near_edge:
                ann_idx, poly_idx, edge_idx = near_edge
                self.context_edge_info = (ann_idx, poly_idx, edge_idx, img_pos.x(), img_pos.y())
                if target_idx == -1:
                    target_idx = ann_idx

        # 3. Check if clicking on a pure secondary annotation (纯副标注)
        secondary_idx = self._find_secondary_at_position(img_pos)

        # 始终触发上下文菜单
        self.ignore_next_left_press = True
        if target_idx != -1:
            self.hover_index = target_idx

        # 如果点击的是纯副标注，触发副标注菜单
        if secondary_idx >= 0 and target_idx == -1:
            self.secondary_context_menu_requested.emit(secondary_idx, global_pos)
        else:
            self.context_menu_requested.emit(target_idx, global_pos)

    def _find_secondary_at_position(self, img_pos):
        """检测点击位置是否在纯副标注上，返回副标注索引"""
        if not self.secondary_annotations:
            return -1

        matched_secondary_indices = set(self.secondary_annotation_map.values())

        for j, s_ann in enumerate(self.secondary_annotations):
            if j in matched_secondary_indices:
                continue

            # 检查点是否在 bbox 内
            bx, by, bw, bh = s_ann.get('bbox', [0, 0, 0, 0])
            if bw == 0 or bh == 0:
                # 从 polygons 计算 bbox
                all_points = []
                for poly in s_ann.get('polygons', []):
                    for pt in poly:
                        try:
                            if hasattr(pt, '__len__') and len(pt) >= 2:
                                if hasattr(pt[0], '__len__'):
                                    all_points.append((pt[0][0], pt[0][1]))
                                else:
                                    all_points.append((pt[0], pt[1]))
                        except:
                            continue
                if all_points:
                    xs = [p[0] for p in all_points]
                    ys = [p[1] for p in all_points]
                    bx = min(xs)
                    by = min(ys)
                    bw = max(xs) - bx
                    bh = max(ys) - by

            if bw > 0 and bh > 0:
                if bx <= img_pos.x() <= bx + bw and by <= img_pos.y() <= by + bh:
                    return j

            # 检查点是否在 polygons 内
            for poly in s_ann.get('polygons', []):
                if self._point_in_polygon(img_pos, poly):
                    return j

        return -1

    def _point_in_polygon(self, point, polygon):
        """检测点是否在多边形内"""
        n = len(polygon)
        if n < 3:
            return False

        inside = False
        x, y = point.x(), point.y()

        j = n - 1
        for i in range(n):
            try:
                if hasattr(polygon[i], '__len__') and len(polygon[i]) >= 2:
                    if hasattr(polygon[i][0], '__len__'):
                        xi, yi = polygon[i][0][0], polygon[i][0][1]
                        xj, yj = polygon[j][0][0], polygon[j][0][1]
                    else:
                        xi, yi = polygon[i][0], polygon[i][1]
                        xj, yj = polygon[j][0], polygon[j][1]
                else:
                    continue

                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            except:
                continue

        return inside

    def mousePressEvent(self, event):
        # 交互闸门：像素未提交（无图 / 正在解码）时不接受任何绘制或编辑输入。
        # 这是审计 D7 与 W2 的共同修复点 —— 按住左键切图后，若继续用旧图的坐标系
        # 向新图提交标注，会即时落盘且不可撤销。
        if not self._interaction_ready():
            return
        # Use position() for PySide6 compatibility (pos() is deprecated)
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        self.last_mouse_pos = QPointF(mouse_pos)
        img_pos = self.to_image_coords(mouse_pos)

        # 如果点击在图片外部，直接忽略
        if img_pos is None:
            return

        if not self.interactive:
            # Still allow middle button panning even in non-interactive mode
            if event.button() == Qt.MouseButton.MiddleButton:
                self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton and self.ignore_next_left_press:
            self.ignore_next_left_press = False
            return

        if event.button() == Qt.MouseButton.RightButton:
            # 记录右键按下的位置和时间，用于区分点击和拖动
            from time import time
            self.right_press_pos = self.last_mouse_pos
            self.right_press_time = time() * 1000  # 转换为毫秒
            self.right_button_held = True
            self.context_vertex_info = None
            self.context_edge_info = None
            return

        elif event.button() == Qt.MouseButton.LeftButton:
            # 0. Handle Ctrl+drag for batch selection
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.batch_selecting = True
                self.batch_selection_start = img_pos
                self.batch_selection_end = img_pos
                self.batch_selected_indices = []
                self.batch_selected_points = []
                self.update()
                return

            # 1. Handle Point Interaction (Draggable vertices)
            # Check near point again to ensure it captures even if mouseMove didn't fire exactly
            near = self.find_near_point(img_pos)
            if near:
                self.annotation_modify_started.emit()
                self.dragging_point = near
                self.selected_index = near[0]
                self.annotation_selected.emit(near[0])
                # Don't jump to mouse position immediately to prevent accidental displacement
                self.update()
                return

            # 2. Handle Mode-specific creation
            if self.mode == DrawingMode.SAM:
                if img_pos:
                    # Filter points outside ROI
                    if not self.is_inside_roi(img_pos):
                        return

                    # Left click = positive point (1), Shift+Left = negative (0)
                    positive = not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    self.add_sam_point(img_pos.x(), img_pos.y(), positive)

            elif self.mode in [DrawingMode.RECT, DrawingMode.AI_RECT, DrawingMode.ROI]:
                if img_pos:
                    # Filter rectangle start outside ROI (only for RECT/AI_RECT mode, ROI mode is to define ROI)
                    if self.mode in [DrawingMode.RECT, DrawingMode.AI_RECT] and not self.is_inside_roi(img_pos):
                        return

                    self.rect_start = img_pos
                    self.rect_end = img_pos

            elif self.mode == DrawingMode.POLYGON:
                if img_pos:
                    # Filter polygon points outside ROI
                    if not self.is_inside_roi(img_pos):
                        return

                    # Check for closing polygon (click near first point)
                    if self.is_near_first_point:
                        self.close_polygon()
                        return

                    # Snapping to existing points
                    near = self.find_near_point(img_pos)
                    if near:
                        ann_idx, poly_idx, pt_idx = near
                        pt = self.annotations[ann_idx]['polygons'][poly_idx][pt_idx]
                        if hasattr(pt[0], '__len__'):
                            snap_pt = QPointF(pt[0][0], pt[0][1])
                        else:
                            snap_pt = QPointF(pt[0], pt[1])
                        self.add_polygon_vertex(snap_pt.x(), snap_pt.y())
                    else:
                        self.add_polygon_vertex(img_pos.x(), img_pos.y())

            elif self.mode == DrawingMode.OBB:
                if img_pos:
                    if len(self.obb_points) == 0:
                        self.obb_points.append(img_pos)
                    elif len(self.obb_points) == 1:
                        self.obb_points.append(img_pos)
                    elif len(self.obb_points) == 2:
                        # Finish OBB
                        p1, p2 = self.obb_points[0], self.obb_points[1]
                        p3 = img_pos

                        v12 = np.array([p2.x() - p1.x(), p2.y() - p1.y()])
                        len12 = np.linalg.norm(v12)
                        if len12 > 1e-6:
                            u12 = v12 / len12
                            u_perp = np.array([-u12[1], u12[0]])
                            v13 = np.array([p3.x() - p1.x(), p3.y() - p1.y()])
                            dist = np.dot(v13, u_perp)

                            p4 = p1 + QPointF(u_perp[0] * dist, u_perp[1] * dist)
                            p3_rect = p2 + QPointF(u_perp[0] * dist, u_perp[1] * dist)

                            # 保留浮点数精度
                            pts = [[p1.x(), p1.y()],
                                   [p2.x(), p2.y()],
                                   [p3_rect.x(), p3_rect.y()],
                                   [p4.x(), p4.y()]]
                            self.annotation_finished.emit(pts)
                            # Auto switch to EDIT mode
                            self.set_mode(DrawingMode.EDIT)

                        self.obb_points = []
                    self.update()

            elif self.mode == DrawingMode.EDIT:
                self.select_annotation_at(img_pos.x(), img_pos.y())
                hit_index = self.selected_index
                if hit_index != -1:
                    # Start dragging the whole annotation
                    self.annotation_modify_started.emit()
                    self.dragging_annotation = hit_index
                    self.drag_start_img_pos = img_pos
                else:
                    self.selected_index = -1
                    self.dragging_annotation = -1
                    self.drag_start_img_pos = None
                    # Emit selection cleared if needed
                    self.annotation_selected.emit(-1)

            else:
                # In other modes, clicking empty area should also deselect
                if self.hover_index == -1:
                    self.selected_index = -1
                    self.annotation_selected.emit(-1)

        self.update()

    def _update_batch_selection(self):
        """更新批量选择状态，检测选择区域内的标注和点

        选中逻辑：
        - 如果选择区域完全包含标注的bbox，则选中整个标注
        - 如果只有点在选择区域内，则选中这些点
        """
        if not self.batch_selection_start or not self.batch_selection_end:
            return

        r = QRectF(self.batch_selection_start, self.batch_selection_end).normalized()
        self.batch_selected_indices = []
        self.batch_selected_points = []

        for i, ann in enumerate(self.annotations):
            bx, by, bw, bh = ann.get('bbox', [0, 0, 0, 0])
            ann_rect = QRectF(bx, by, bw, bh)

            # 只有当选择区域完全包含标注的bbox时，才选中整个标注
            if r.contains(ann_rect):
                self.batch_selected_indices.append(i)
            else:
                # 否则检查是否有点在选择区域内
                for poly_idx, poly in enumerate(ann.get('polygons', [])):
                    for pt_idx, pt in enumerate(poly):
                        try:
                            if hasattr(pt, '__len__') and len(pt) >= 2:
                                if hasattr(pt[0], '__len__'):
                                    x, y = pt[0][0], pt[0][1]
                                else:
                                    x, y = pt[0], pt[1]
                                if r.contains(QPointF(x, y)):
                                    self.batch_selected_points.append((i, poly_idx, pt_idx))
                        except:
                            continue

    @action("edit.select_in_rect", description="框选标注，选择指定矩形区域内的所有标注。\n- x1, y1, x2, y2 为图像坐标（像素值）\n- 只有当矩形完全包含标注的 bbox 时，才选中整个标注\n- 返回选中标注的索引列表", category="画布", params={"x1": "int", "y1": "int", "x2": "int", "y2": "int"}, scope="agent")
    def select_annotations_in_rect(self, x1, y1, x2, y2):
        if any(v is None for v in [x1, y1, x2, y2]):
            return "错误: 坐标参数不能为空"
        try:
            self.batch_selection_start = QPointF(x1, y1)
            self.batch_selection_end = QPointF(x2, y2)

            if self.batch_selection_start is None or self.batch_selection_end is None:
                return []

            r = QRectF(self.batch_selection_start, self.batch_selection_end).normalized()
            self.batch_selected_indices = []
            self.batch_selected_points = []

            for i, ann in enumerate(self.annotations):
                bx, by, bw, bh = ann.get('bbox', [0, 0, 0, 0])
                ann_rect = QRectF(bx, by, bw, bh)

                if r.contains(ann_rect):
                    self.batch_selected_indices.append(i)
                else:
                    for poly_idx, poly in enumerate(ann.get('polygons', [])):
                        for pt_idx, pt in enumerate(poly):
                            try:
                                if hasattr(pt, '__len__') and len(pt) >= 2:
                                    if hasattr(pt[0], '__len__'):
                                        x, y = pt[0][0], pt[0][1]
                                    else:
                                        x, y = pt[0], pt[1]
                                    if r.contains(QPointF(x, y)):
                                        self.batch_selected_points.append((i, poly_idx, pt_idx))
                            except:
                                continue

            self.update()
            return self.batch_selected_indices
        except Exception as e:
            return f"错误: 框选失败: {e}"

    def find_near_point(self, img_pos, max_dist_screen=12.0):
        """Finds the nearest vertex across all annotations."""
        if img_pos is None: return None

        found = None
        # Use a constant screen distance (e.g., pixels on screen)
        # Convert it to image coordinates by dividing by zoom
        safe_zoom = max(0.0001, self.zoom)
        best_dist = float(max_dist_screen) / safe_zoom

        # Priority: Check currently selected annotation first
        check_order = list(range(len(self.annotations)))
        if self.selected_index is not None and 0 <= self.selected_index < len(self.annotations):
            check_order.remove(self.selected_index)
            check_order.insert(0, self.selected_index)

        for i in check_order:
            ann = self.annotations[i]
            if not isinstance(ann, dict):
                continue
            # rectangle类型即使有polygons也使用bbox角点交互
            is_rectangle = ann.get('shape_type') == 'rectangle'
            # 1. Check polygons (for any annotation that has polygons, except rectangle type)
            polys = ann.get('polygons', [])
            if polys and not is_rectangle:
                for poly_idx, poly in enumerate(polys):
                    for pt_idx, pt in enumerate(poly):
                        if hasattr(pt, '__len__') and len(pt) >= 1 and hasattr(pt[0], '__len__'): # Handle [[x,y]]
                            px, py = pt[0][0], pt[0][1]
                        elif hasattr(pt, '__len__') and len(pt) >= 2:
                            px, py = pt[0], pt[1]
                        else:
                            continue

                        dist = ((img_pos.x() - px)**2 + (img_pos.y() - py)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            found = (i, poly_idx, pt_idx)

            # 2. Check bbox corners (for annotations without polygons, or rectangle type)
            if 'bbox' in ann and ann['bbox'] and (not ann.get('polygons') or is_rectangle):
                bx, by, bw, bh = ann['bbox']
                corners = [(bx, by), (bx+bw, by), (bx+bw, by+bh), (bx, by+bh)]
                for pt_idx, (px, py) in enumerate(corners):
                    dist = ((img_pos.x() - px)**2 + (img_pos.y() - py)**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        # Use poly_idx = -1 to indicate bbox corner dragging
                        found = (i, -1, pt_idx)
        return found

    def move_point(self, point_info, img_pos):
        """Moves a specific vertex or bbox corner to a new position."""
        if not point_info or not img_pos: return

        ann_idx, poly_idx, pt_idx = point_info
        if ann_idx >= len(self.annotations): return

        ann = self.annotations[ann_idx]

        if poly_idx == -1:
            # Dragging bbox corner - 保留浮点数精度
            if 'bbox' not in ann or not ann['bbox']: return
            bx, by, bw, bh = ann['bbox']
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh

            # 根据当前角点索引，确定对角锚点
            # 对角锚点在拖动过程中保持不动
            if pt_idx == 0: # Top-left → 对角锚点是 Bottom-right
                ax, ay = x2, y2
            elif pt_idx == 1: # Top-right → 对角锚点是 Bottom-left
                ax, ay = x1, y2
            elif pt_idx == 2: # Bottom-right → 对角锚点是 Top-left
                ax, ay = x1, y1
            elif pt_idx == 3: # Bottom-left → 对角锚点是 Top-right
                ax, ay = x2, y1
            else:
                return

            mx, my = img_pos.x(), img_pos.y()
            # 计算新的 bbox（允许坐标交换，实时根据位置确定左右/上下边界）
            new_x1, new_y1 = min(ax, mx), min(ay, my)
            new_x2, new_y2 = max(ax, mx), max(ay, my)
            ann['bbox'] = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]

            # 根据锚点和鼠标位置的关系，动态更新角点角色
            # 确保后续拖动帧中 dragging_point 对应正确的角点
            if mx < ax:
                # 鼠标在锚点左侧
                if my < ay:
                    new_pt_idx = 0  # Top-left
                else:
                    new_pt_idx = 3  # Bottom-left
            else:
                # 鼠标在锚点右侧
                if my < ay:
                    new_pt_idx = 1  # Top-right
                else:
                    new_pt_idx = 2  # Bottom-right

            if new_pt_idx != pt_idx:
                self.dragging_point = (point_info[0], -1, new_pt_idx)

            # Pure rectangle: sync polygons if present
            if not ann.get('polygons'):
                pass  # No polygons to sync for pure bbox
            elif ann.get('shape_type') == 'rectangle' and 'polygons' in ann:
                bx, by, bw, bh = ann['bbox']
                new_poly = [[bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]]
                ann['polygons'] = [new_poly]
        else:
            # Dragging polygon point
            if 'polygons' not in ann or poly_idx >= len(ann['polygons']): return
            poly = ann['polygons'][poly_idx]
            if pt_idx >= len(poly): return

            # rotation类型：拖动角点时保持旋转矩形形状
            if ann.get('shape_type') == 'rotation' and len(poly) == 4:
                anchor_idx = (pt_idx + 2) % 4
                adj1_idx = (pt_idx + 1) % 4
                adj2_idx = (pt_idx + 3) % 4

                # 获取锚点和邻接点的坐标
                def _get_pt(p):
                    if hasattr(p, '__len__') and len(p) >= 2:
                        if hasattr(p[0], '__len__'):
                            return float(p[0][0]), float(p[0][1])
                        return float(p[0]), float(p[1])
                    return 0.0, 0.0

                anchor = _get_pt(poly[anchor_idx])
                adj1 = _get_pt(poly[adj1_idx])
                adj2 = _get_pt(poly[adj2_idx])

                # 计算从锚点到两个邻接点的方向向量
                dir1 = (adj1[0] - anchor[0], adj1[1] - anchor[1])
                dir2 = (adj2[0] - anchor[0], adj2[1] - anchor[1])
                len1 = (dir1[0]**2 + dir1[1]**2) ** 0.5
                len2 = (dir2[0]**2 + dir2[1]**2) ** 0.5

                if len1 > 0.01 and len2 > 0.01:
                    # 单位方向向量
                    u1 = (dir1[0]/len1, dir1[1]/len1)
                    u2 = (dir2[0]/len2, dir2[1]/len2)

                    # 新点相对锚点的投影
                    dx = img_pos.x() - anchor[0]
                    dy = img_pos.y() - anchor[1]
                    proj1 = dx * u1[0] + dy * u1[1]
                    proj2 = dx * u2[0] + dy * u2[1]

                    # 限制最小尺寸
                    proj1 = max(proj1, 5.0)
                    proj2 = max(proj2, 5.0)

                    # 重新计算4个角点
                    new_pts = [[0.0, 0.0]] * 4
                    new_pts[anchor_idx] = [anchor[0], anchor[1]]
                    new_pts[pt_idx] = [anchor[0] + proj1*u1[0] + proj2*u2[0],
                                        anchor[1] + proj1*u1[1] + proj2*u2[1]]
                    new_pts[adj1_idx] = [anchor[0] + proj1*u1[0],
                                          anchor[1] + proj1*u1[1]]
                    new_pts[adj2_idx] = [anchor[0] + proj2*u2[0],
                                          anchor[1] + proj2*u2[1]]

                    ann['polygons'][poly_idx] = new_pts
                else:
                    # 退化情况：按普通多边形点处理
                    poly[pt_idx] = [img_pos.x(), img_pos.y()]
            else:
                # 普通多边形：直接移动单个点
                if hasattr(poly[pt_idx], '__len__') and len(poly[pt_idx]) >= 1 and hasattr(poly[pt_idx][0], '__len__'):
                    poly[pt_idx][0][0] = img_pos.x()
                    poly[pt_idx][0][1] = img_pos.y()
                elif hasattr(poly[pt_idx], '__len__') and len(poly[pt_idx]) >= 2:
                    poly[pt_idx] = [img_pos.x(), img_pos.y()]

            # Update bbox for this annotation
            self.update_bbox(ann)

    @action("edit.delete_polygon_vertex", description="删除多边形的一个顶点。\n- ann_idx: 标注索引\n- poly_idx: 多边形索引（标注可能包含多个多边形）\n- pt_idx: 要删除的顶点索引\n- 多边形至少需要保留3个顶点（三角形）\n- 删除后自动更新 bbox", category="画布",
            params={"ann_idx": "int", "poly_idx": "int", "pt_idx": "int"}, scope="agent")
    def delete_poly_point(self, ann_idx, poly_idx, pt_idx):
        if ann_idx >= len(self.annotations):
            return "错误: 标注索引 %d 超出范围（共 %d 个标注）" % (ann_idx, len(self.annotations))
        ann = self.annotations[ann_idx]
        if poly_idx >= len(ann.get('polygons', [])):
            return "错误: 多边形索引 %d 超出范围" % poly_idx

        poly = ann['polygons'][poly_idx]
        if not isinstance(poly, list):
            poly = poly.tolist()
            ann['polygons'][poly_idx] = poly

        if len(poly) <= 3:
            return "错误: 多边形至少需要3个顶点，无法继续删除"
        if pt_idx < 0 or pt_idx >= len(poly):
            return "错误: 顶点索引 %d 超出范围（共 %d 个顶点）" % (pt_idx, len(poly))

        self.annotation_modify_started.emit()
        poly.pop(pt_idx)
        self.update_bbox(ann)
        self.annotation_modified.emit()
        self.update()

    @action("edit.add_polygon_vertex_on_edge", description="在多边形的一条边上插入一个新顶点。\n- ann_idx: 标注索引\n- poly_idx: 多边形索引\n- edge_idx: 目标边的起始顶点索引（在 p[edge_idx]→p[edge_idx+1] 之间插入）\n- x, y: 新顶点的图像坐标（像素值）\n- 配合 find_near_edge 确定 edge_idx\n- 用于精细化调整多边形轮廓", category="画布",
            params={"ann_idx": "int", "poly_idx": "int", "edge_idx": "int", "x": "float", "y": "float"}, scope="agent")
    def add_poly_point(self, ann_idx, poly_idx, edge_idx, x: float = 0, y: float = 0, img_pos=None):
        if img_pos is not None:
            pos = img_pos
        elif x != 0 or y != 0:
            pos = QPointF(x, y)
        else:
            return "错误: 未提供坐标参数"

        if ann_idx >= len(self.annotations):
            return "错误: 标注索引 %d 超出范围（共 %d 个标注）" % (ann_idx, len(self.annotations))
        ann = self.annotations[ann_idx]
        if poly_idx >= len(ann.get('polygons', [])):
            return "错误: 多边形索引 %d 超出范围" % poly_idx

        poly = ann['polygons'][poly_idx]
        if not isinstance(poly, list):
            poly = poly.tolist()
            ann['polygons'][poly_idx] = poly

        if edge_idx < 0 or edge_idx >= len(poly):
            return "错误: 边索引 %d 超出范围（共 %d 个顶点）" % (edge_idx, len(poly))
        if len(poly) < 3:
            return "错误: 多边形顶点数不足，无法添加顶点"

        sample_pt = poly[0]
        if hasattr(sample_pt, '__len__') and len(sample_pt) >= 1 and hasattr(sample_pt[0], '__len__'):
            new_pt = [[pos.x(), pos.y()]]
        else:
            new_pt = [pos.x(), pos.y()]

        self.annotation_modify_started.emit()
        poly.insert(edge_idx + 1, new_pt)
        self.update_bbox(ann)
        self.annotation_modified.emit()
        self.update()

    def find_near_edge(self, img_pos):
        """Finds the nearest edge of a polygon for point insertion."""
        if img_pos is None:
            return None

        safe_zoom = max(0.0001, self.zoom)
        best_dist = 10.0 / safe_zoom
        found = None # (ann_idx, poly_idx, edge_start_idx)

        # Priority: Selected annotation
        check_order = list(range(len(self.annotations)))
        if self.selected_index is not None and 0 <= self.selected_index < len(self.annotations):
            check_order.remove(self.selected_index)
            check_order.insert(0, self.selected_index)

        for i in check_order:
            ann = self.annotations[i]
            # rectangle类型不查找polygon边（应使用bbox角点交互）
            if ann.get('shape_type') == 'rectangle':
                continue
            polys = ann.get('polygons', [])
            for poly_idx, poly in enumerate(polys):
                for j in range(len(poly)):
                    p1 = poly[j]
                    p2 = poly[(j + 1) % len(poly)]

                    # Normalize point format
                    if hasattr(p1, '__len__') and len(p1) >= 1 and hasattr(p1[0], '__len__'):
                        x1, y1 = p1[0][0], p1[0][1]
                        x2, y2 = p2[0][0], p2[0][1]
                    else:
                        x1, y1 = p1[0], p1[1]
                        x2, y2 = p2[0], p2[1]

                    # Calculate distance from point to segment
                    px, py = img_pos.x(), img_pos.y()
                    dx, dy = x2 - x1, y2 - y1
                    if dx == 0 and dy == 0: continue

                    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
                    if 0 <= t <= 1:
                        # Nearest point is on segment
                        closest_x = x1 + t * dx
                        closest_y = y1 + t * dy
                        dist = ((px - closest_x)**2 + (py - closest_y)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            found = (i, poly_idx, j)
        return found

    def update_bbox(self, ann):
        all_pts = []
        for p in ann.get('polygons', []):
            for pt in p:
                if hasattr(pt[0], '__len__'):
                    all_pts.append([pt[0][0], pt[0][1]])
                else:
                    all_pts.append([pt[0], pt[1]])
        if all_pts:
            # 使用浮点精度的 min/max 计算 bbox，避免 cv2.boundingRect 的整数截断
            xs = [pt[0] for pt in all_pts]
            ys = [pt[1] for pt in all_pts]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            ann['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def mouseMoveEvent(self, event):
        if not self._interaction_ready():
            return
        # Use position() for PySide6 compatibility (pos() is deprecated)
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        curr_pos = QPointF(mouse_pos)
        img_pos = self.to_image_coords(mouse_pos)

        # 发送鼠标位置信号
        self.mouse_pos_changed.emit(img_pos)

        # Handle batch selection (Ctrl+drag)
        if self.batch_selecting and img_pos:
            self.batch_selection_end = img_pos
            self._update_batch_selection()
            self.update()
            self.last_mouse_pos = curr_pos
            return

        # Handle Panning (Right button or Middle button)
        if event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            # 右键拖动：检查是否超过阈值，如果是则触发拖动（平移）
            if event.buttons() & Qt.MouseButton.RightButton and self.right_button_held and self.right_press_pos:
                move_distance = (curr_pos - self.right_press_pos).manhattanLength()
                if move_distance > self.right_click_drag_threshold:
                    # 超过阈值，转为拖动模式（清除按下位置，避免释放时触发菜单）
                    self.right_press_pos = None
                    self._right_dragging = True

            # 右键/中键拖动始终允许画布平移
            delta = curr_pos - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = curr_pos
            self.update()
            return

        if not self.interactive:
            # Update hover state for viewing
            self.hover_index = -1
            if img_pos:
                for i in range(len(self.annotations)-1, -1, -1):
                    ann = self.annotations[i]
                    if 'bbox' in ann and ann['bbox']:
                        bx, by, bw, bh = ann['bbox']
                        if bx <= img_pos.x() <= bx + bw and by <= img_pos.y() <= by + bh:
                            self.hover_index = i
                            break
            self.last_mouse_pos = curr_pos
            self.update()
            return

        # 1. Handle Point or Annotation Dragging
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.dragging_point and img_pos:
                self.move_point(self.dragging_point, img_pos)
                self.update()
                self.last_mouse_pos = curr_pos
                return
            elif self.dragging_point:
                # 拖动点时鼠标在图像外，仅更新位置不处理拖动
                self.last_mouse_pos = curr_pos
                return
            elif self.dragging_annotation != -1 and img_pos and self.drag_start_img_pos:
                # Drag the entire annotation - 保留浮点数精度
                dx = img_pos.x() - self.drag_start_img_pos.x()
                dy = img_pos.y() - self.drag_start_img_pos.y()

                if abs(dx) > 0.5 or abs(dy) > 0.5:  # 添加阈值避免微小抖动
                    ann = self.annotations[self.dragging_annotation]

                    # Move all polygons
                    for poly in ann.get('polygons', []):
                        for pt in poly:
                            if hasattr(pt[0], '__len__'):
                                pt[0][0] += dx
                                pt[0][1] += dy
                            else:
                                pt[0] += dx
                                pt[1] += dy

                    # Move bbox
                    if 'bbox' in ann and ann['bbox']:
                        ann['bbox'][0] += dx
                        ann['bbox'][1] += dy

                    self.drag_start_img_pos = img_pos
                    self.update()

                self.last_mouse_pos = curr_pos
                return

        # 2. Update Hover State
        self.hover_index = -1
        self.hover_point = None

        if img_pos:
            # Check for point hover first (higher priority)
            self.hover_point = self.find_near_point(img_pos)

            if self.hover_point:
                self.hover_index = self.hover_point[0]
                self.setCursor(Qt.PointingHandCursor)
            else:
                # Check for bbox hover
                for i in range(len(self.annotations)-1, -1, -1):
                    ann = self.annotations[i]
                    if 'bbox' not in ann or not ann['bbox']:
                        continue
                    bx, by, bw, bh = ann['bbox']
                    if bx <= img_pos.x() <= bx + bw and by <= img_pos.y() <= by + bh:
                        self.hover_index = i
                        break

                if self.mode == DrawingMode.EDIT or self.hover_point:
                    self.setCursor(Qt.ArrowCursor if self.hover_index == -1 else Qt.PointingHandCursor)

        # 3. Handle Polygon Snapping and Cursor
        self.is_near_first_point = False
        if self.mode == DrawingMode.POLYGON and not self.hover_point:
            if len(self.current_poly) > 2:
                p1 = self.from_image_coords(self.current_poly[0])
                dist = (curr_pos - p1).manhattanLength()
                if dist < 15:
                    self.is_near_first_point = True
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.CrossCursor)
            else:
                # Snapping to existing points when drawing
                near = self.find_near_point(img_pos)
                if near:
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.CrossCursor)

        if self.mode == DrawingMode.OBB and not self.hover_point:
            self.setCursor(Qt.CrossCursor)
            self.update()

        # 4. Handle Panning
        if event.buttons() & Qt.MouseButton.RightButton:
            if not ((self.mode == DrawingMode.EDIT or self.mode == DrawingMode.SAM) and self.hover_index != -1):
                delta = curr_pos - self.last_mouse_pos
                self.pan_offset += delta

        # 5. Handle Rect/ROI Drawing
        elif event.buttons() & Qt.MouseButton.LeftButton:
            if self.mode in [DrawingMode.RECT, DrawingMode.AI_RECT, DrawingMode.ROI] and self.rect_start:
                self.rect_end = img_pos

        self.last_mouse_pos = curr_pos
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._interaction_ready():
            # 按下时已提交、释放时图像已换：必须丢弃这次释放，并清掉可能残留的
            # 拖拽/框选瞬时状态，否则下一次事件会拿旧坐标对新图动手（D7）。
            self._reset_transient_drag_state()
            return
        if not self.interactive:
            if event.button() == Qt.MouseButton.MiddleButton:
                self.setCursor(Qt.CrossCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.setCursor(Qt.CrossCursor)
            self.right_button_held = False

            # 判断是否触发右键菜单（快速点击且移动距离小，且未发生拖动平移）
            if self.right_press_pos is not None and not self._right_dragging:
                from time import time
                release_time = time() * 1000
                press_duration = release_time - self.right_press_time

                # 获取当前鼠标位置的图像坐标
                mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
                img_pos = self.to_image_coords(mouse_pos)

                if img_pos is not None and press_duration < self.right_click_time_threshold:
                    # 触发右键菜单
                    self._show_context_menu(event.globalPos(), img_pos)

            self.right_press_pos = None
            self._right_dragging = False

        elif event.button() == Qt.MouseButton.LeftButton:
            # Handle batch selection end
            if self.batch_selecting:
                self.batch_selecting = False
                if self.batch_selection_start and self.batch_selection_end:
                    r = QRectF(self.batch_selection_start, self.batch_selection_end).normalized()
                    if r.width() > 5 and r.height() > 5:
                        if self.batch_selected_indices:
                            self.batch_selection_changed.emit(self.batch_selected_indices)
                        if self.batch_selected_points:
                            selection_rect = [r.x(), r.y(), r.width(), r.height()]
                            self.batch_points_selected.emit(self.batch_selected_points, selection_rect)
                self.batch_selection_start = None
                self.batch_selection_end = None
                self.update()
                return

            if self.dragging_point:
                # Point was dragged, notify about update if needed
                self.dragging_point = None
                self.annotation_modified.emit()
            elif self.dragging_annotation != -1:
                # Annotation was dragged
                self.dragging_annotation = -1
                self.annotation_modified.emit()
            elif self.mode in [DrawingMode.RECT, DrawingMode.AI_RECT, DrawingMode.ROI] and self.rect_start and self.rect_end:
                r = QRectF(self.rect_start, self.rect_end).normalized()
                # 最小矩形框尺寸阈值为1像素
                if r.width() > 1 and r.height() > 1:
                    # 保留浮点数精度，实现亚像素级别的标注
                    bbox = [r.x(), r.y(), r.width(), r.height()]
                    if self.mode == DrawingMode.ROI:
                        self.persistent_roi = [round(v) for v in bbox]
                        self.roi_selected.emit(self.persistent_roi)
                    else:
                        # AI_RECT 模式会触发 AI 辅助，RECT 模式不会
                        self.annotation_finished.emit(bbox)

                    # 录制：拖拽完成时记录步骤（拖拽不是 @action，需手动记录）
                    from core.common.action_recorder import ActionRecorder
                    recorder = ActionRecorder.instance()
                    if recorder.is_recording:
                        if self.mode == DrawingMode.ROI:
                            recorder.record_raw_step({
                                "action": "input.mouse",
                                "op": "drag",
                                "x1": int(self.rect_start.x()), "y1": int(self.rect_start.y()),
                                "x2": int(self.rect_end.x()), "y2": int(self.rect_end.y()),
                                "description": "拖拽画ROI区域"
                            })
                        else:
                            recorder.record_raw_step({
                                "action": "draw_rect",
                                "x": int(r.x()), "y": int(r.y()),
                                "w": int(r.width()), "h": int(r.height()),
                                "description": "拖拽画矩形"
                            })

                    # Auto switch to EDIT mode
                    self.set_mode(DrawingMode.EDIT)
                self.rect_start = None
                self.rect_end = None

        self.update()

    def keyPressEvent(self, event):
        # 交互闸门：未提交时不处理快捷键。否则 Backspace 会拿 stale 的 hover_point
        # 去删另一张图/另一个标注的顶点（审计 D7），下标还可能越界。
        if not self._interaction_ready():
            return
        if event.key() == Qt.Key.Key_Escape:
            self.current_poly = []
            self.sam_points = []
            # Reset SAM active state in parent if exists
            if hasattr(self.parent(), '_sam_active'):
                self.parent()._sam_active = False
            self.update()
        elif event.key() == Qt.Key.Key_Backspace:
            # Delete vertex if hovering over a polygon vertex
            if self.hover_point and self.hover_point[1] != -1:
                ann_idx, poly_idx, pt_idx = self.hover_point
                if ann_idx < len(self.annotations) and poly_idx < len(self.annotations[ann_idx].get('polygons', [])):
                    poly = self.annotations[ann_idx]['polygons'][poly_idx]
                    if len(poly) > 3:  # Keep at least a triangle
                        self.delete_poly_point(ann_idx, poly_idx, pt_idx)
                        self.hover_point = None
                        return
            # Original backspace behavior for polygon drawing
            if self.mode == DrawingMode.POLYGON and self.current_poly:
                self.current_poly.pop()
                self.update()
            elif self.mode == DrawingMode.SAM and self.sam_points:
                self.sam_points.pop()
                self.sam_point_added.emit(self.sam_points, -1) # -1 indicates removal
                self.update()
        elif event.key() == Qt.Key.Key_W:
            # Enable drawing mode
            if self.mode == DrawingMode.EDIT:
                self.set_mode(self.last_drawing_mode)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.mode == DrawingMode.SAM:
                # Commit SAM session
                if hasattr(self.parent(), '_sam_active'):
                    self.parent()._sam_active = False
                self.sam_points = []
                self.update()
                # Auto switch to EDIT mode
                self.set_mode(DrawingMode.EDIT)
                return

            if self.mode == DrawingMode.POLYGON and len(self.current_poly) > 2:
                # Convert poly to annotation - 保留浮点数精度
                pts = [[p.x(), p.y()] for p in self.current_poly]
                self.annotation_finished.emit(pts)
                # Auto switch to EDIT mode
                self.set_mode(DrawingMode.EDIT)
                self.current_poly = []
                # Auto switch to EDIT mode
                self.set_mode(DrawingMode.EDIT)
        super().keyPressEvent(event)

    @action("interact.zoom", description="放大/缩小画布。\n- direction：1=放大，-1=缩小\n- 以画布中心为基准放大/缩小1.2倍\n- 缩放范围限制在 0.1x ~ 50x\n- 需要先加载图像\n- 配合 reset_view 使用", category="画布", params={"direction": "int"}, scope="agent")
    def action_zoom(self, direction: int = 1):
        if not self.image or not self.pixmap:
            return "错误: 没有加载图像"
        center = QPoint(self.width() // 2, self.height() // 2)
        old_img_pos = self.to_image_coords(center)
        if direction >= 0:
            self.zoom *= 1.2
        else:
            self.zoom /= 1.2
        self.zoom = max(0.1, min(self.zoom, 50.0))
        new_img_pos = self.to_image_coords(center)
        if old_img_pos and new_img_pos:
            delta = (new_img_pos - old_img_pos) * self.zoom
            self.pan_offset += delta
        self.update()
        return {"status": "success", "message": "画布已缩放", "zoom": round(self.zoom, 4)}

    @action("interact.add_sam_point", description="添加 SAM 正/负样本点进行交互式分割。\n- x, y 为图像坐标（像素值），不受画布缩放影响\n- 必须先通过 set_drawing_mode('sam') 切换到 SAM 模式\n- positive=true 为正样本（属于目标区域）\n- positive=false 为负样本（不属于目标区域）\n- 添加后观察实时分割预览，可连续添加多个点优化\n- 完成后调用 confirm_sam_annotation() 确认分割结果\n\n坐标系统说明：本软件有两套坐标系统。标注操作（如 SAM 打点、绘制多边形）使用图像坐标，原点在图像左上角，单位像素，不受画布缩放平移影响。不要用鼠标模拟操作（input.mouse 的 click/move）传入图像坐标进行标注，标注请直接调用本方法。鼠标模拟操作使用控件坐标，原点在画布控件左上角，受缩放和平移影响。", category="画布", params={"x": "float", "y": "float", "positive": "bool"}, scope="agent")
    def add_sam_point(self, x: float, y: float, positive: bool = True):
        if not self._interaction_ready():
            return "错误: 图像尚未加载完成（像素未提交），此时按旧图分辨率校验坐标会落成错标"
        if self.mode != DrawingMode.SAM:
            return "错误: 当前不是 SAM 模式，请先调用 set_drawing_mode('sam') 切换模式"
        image_w = self.original_image_size.width()
        image_h = self.original_image_size.height()
        if not (0 <= x <= image_w and 0 <= y <= image_h):
            return f"错误: 坐标 ({x:.1f}, {y:.1f}) 超出图像范围 ({image_w}x{image_h})"
        label = 1 if positive else 0
        self.sam_points.append((x, y, label))
        self.sam_point_added.emit(self.sam_points, label)
        self.update()

    @action("interact.add_polygon_vertex", description="添加多边形顶点。\n- x, y 为图像坐标（像素值）\n- 必须先通过 set_drawing_mode('polygon') 切换到多边形模式\n- 每次调用添加一个顶点，连续调用可勾勒多边形轮廓\n- 添加至少3个顶点后调用 close_polygon() 闭合", category="画布", params={"x": "float", "y": "float"}, scope="agent")
    def add_polygon_vertex(self, x: float, y: float):
        if not self._interaction_ready():
            return "错误: 图像尚未加载完成（像素未提交），此时按旧图分辨率校验坐标会落成错标"
        if self.mode != DrawingMode.POLYGON:
            return "错误: 当前不是多边形模式，请先调用 set_drawing_mode('polygon') 切换模式"
        image_w = self.original_image_size.width()
        image_h = self.original_image_size.height()
        if not (0 <= x <= image_w and 0 <= y <= image_h):
            return f"错误: 坐标 ({x:.1f}, {y:.1f}) 超出图像范围 ({image_w}x{image_h})"
        self.current_poly.append(QPointF(x, y))
        self.update()

    @action("interact.close_polygon", description="闭合当前正在绘制的多边形并提交为标注。\n- 需要有至少3个已添加的顶点\n- 在调用 add_polygon_vertex() 添加顶点后调用\n- 闭合后自动切换到编辑模式", category="画布", scope="agent")
    def close_polygon(self):
        if len(self.current_poly) < 3:
            return "错误: 多边形至少需要3个顶点，当前只有 %d 个" % len(self.current_poly)
        pts = [[p.x(), p.y()] for p in self.current_poly]
        self.annotation_finished.emit(pts)
        self.current_poly = []
        self.is_near_first_point = False
        self.set_mode(DrawingMode.EDIT)
        self.update()

    @action("interact.confirm_sam_annotation", description="确认当前 SAM 分割预览结果，生成最终标注。\n- 调用前必须已通过 add_sam_point() 添加了至少一个样本点\n- 调用后不可再添加样本点\n- 确认后可用 refine_single_annotation() 精修边缘\n- 确认后调用 save_current() 保存", category="画布", scope="agent")
    def confirm_sam_annotation(self):
        if self.mode != DrawingMode.SAM:
            return "错误: 当前不是 SAM 模式"
        if not self.sam_points:
            return "错误: 没有 SAM 样本点，请先调用 add_sam_point() 添加"
        # 查找父级 AnnotationInterface 的 on_sam_commit 方法
        parent = self.parent()
        while parent:
            if hasattr(parent, 'on_sam_commit'):
                parent.on_sam_commit()
                return
            parent = parent.parent()
        # 没有找到父级处理器，自行清理
        self.sam_points = []
        self.update()

    @action("edit.select_annotation_at", description="选中指定图像坐标位置的标注。\n- x, y 为图像坐标（像素值）\n- 按 bbox 命中检测，选中首个匹配的标注（从上层到下层）\n- 未命中任何标注时取消当前选中\n- 选中后可用 edit.remove_annotation 删除\n- 坐标可通过 add_sam_point / add_polygon_vertex 等确认", category="画布", params={"x": "float", "y": "float"}, scope="agent")
    def select_annotation_at(self, x: float, y: float):
        if x is None or y is None:
            return "错误: 坐标参数不能为空"
        if not self.annotations:
            return "错误: 当前没有标注"
        hit_index = -1
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            if 'bbox' in ann and ann['bbox']:
                bx, by, bw, bh = ann['bbox']
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    hit_index = i
                    break
        if hit_index != -1:
            self.selected_index = hit_index
            self.annotation_selected.emit(hit_index)
        else:
            self.selected_index = -1
            self.annotation_selected.emit(-1)
        self.update()
