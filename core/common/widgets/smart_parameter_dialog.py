import os
from typing import List, Dict, Any, Optional
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QCheckBox, QSpinBox,
                             QDoubleSpinBox, QSlider, QFrame,
                             QWidget, QMessageBox)

from qfluentwidgets import (CaptionLabel, StrongBodyLabel, PrimaryPushButton,
                            PushButton, SpinBox, DoubleSpinBox, Slider,
                            CheckBox, LineEdit, InfoBar, ScrollArea)
from core.common.image_utils import imread_unicode, imwrite_unicode
from core.common.settings import settings
from core.common.widgets.base import BaseDialog


class SmartParameterDialog(BaseDialog):
    """智能调参窗口 - 一次性推理，实时参数调整"""

    parameter_changed = Signal(dict)  # 参数变更信号

    def __init__(self, parent=None, image_path: str = None,
                 instances: Dict[str, Any] = None, support_info: Dict[str, Any] = None):
        super().__init__(parent)
        self.image_path = image_path
        # 鲁棒处理 instances 字典，避免 KeyError
        if instances is None:
            instances = {}

        self.original_instances = instances.get('annotations', []) or []

        # 优先使用显式传入的 support_info，否则尝试从 instances 中获取
        if support_info is not None:
            self.support_info = support_info
        else:
            self.support_info = instances.get('support_info', {}) or {}

        self.current_instances = []
        self.current_rules = {}

        self.setWindowTitle("智能调参 - 实时预览")
        self.setMinimumSize(1000, 700)
        self.setModal(True)

        self.init_ui()
        self.load_image()
        self.apply_current_rules()

    def init_ui(self):
        """初始化界面"""
        main_layout = QHBoxLayout(self)

        # 左侧：参数控制面板
        self.param_panel = self.create_parameter_panel()

        # 右侧：预览面板
        self.preview_panel = self.create_preview_panel()

        # 添加到主布局
        main_layout.addWidget(self.param_panel, 1)  # 参数面板占1份
        main_layout.addWidget(self.preview_panel, 2)  # 预览面板占2份


    def create_parameter_panel(self):
        """创建参数控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # 标题
        title = StrongBodyLabel("参数调节")
        layout.addWidget(title)

        # 滚动区域
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 1. 置信度
        conf_group = self.create_confidence_group()
        scroll_layout.addWidget(conf_group)

        # 2. 面积范围
        area_group = self.create_area_group()
        scroll_layout.addWidget(area_group)

        # 3. 宽度范围
        width_group = self.create_width_group()
        scroll_layout.addWidget(width_group)

        # 4. 高度范围
        height_group = self.create_height_group()
        scroll_layout.addWidget(height_group)

        # 5. 宽高比范围
        aspect_group = self.create_aspect_group()
        scroll_layout.addWidget(aspect_group)

        # 6. 中心点偏差
        center_group = self.create_center_group()
        scroll_layout.addWidget(center_group)

        # 7. 灰度范围
        gray_group = self.create_gray_group()
        scroll_layout.addWidget(gray_group)

        # 8. 最大实例数
        max_inst_group = self.create_max_instances_group()
        scroll_layout.addWidget(max_inst_group)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # 底部按钮
        btn_layout = QHBoxLayout()

        self.btn_reset = PushButton("重置")
        self.btn_reset.clicked.connect(self.reset_parameters)

        self.btn_apply = PrimaryPushButton("应用")
        self.btn_apply.clicked.connect(self.apply_and_close)

        self.btn_cancel = PushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)

        layout.addLayout(btn_layout)

        return panel

    def create_confidence_group(self):
        """创建置信度控制组"""
        group = QGroupBox("置信度阈值")
        layout = QVBoxLayout(group)

        self.cb_conf = CheckBox("启用置信度过滤")
        self.cb_conf.setChecked(True)
        self.cb_conf.stateChanged.connect(self.on_parameter_changed)

        self.slider_conf = Slider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self.on_parameter_changed)

        self.label_conf = CaptionLabel("阈值: 0.50")
        self.slider_conf.valueChanged.connect(
            lambda v: self.label_conf.setText(f"阈值: {v/100:.2f}")
        )

        layout.addWidget(self.cb_conf)
        layout.addWidget(self.slider_conf)
        layout.addWidget(self.label_conf)

        return group

    def create_area_group(self):
        """创建面积范围控制组"""
        group = QGroupBox("面积范围 (相对平均面积波动)")
        layout = QVBoxLayout(group)

        self.cb_area = CheckBox("启用面积过滤")
        self.cb_area.setChecked(True)
        self.cb_area.stateChanged.connect(self.on_parameter_changed)

        # 最小值 (负向波动)
        min_layout = QHBoxLayout()
        min_layout.addWidget(CaptionLabel("允许负向波动:"))
        self.spin_area_min = DoubleSpinBox()
        self.spin_area_min.setRange(-1.0, -0.01)
        self.spin_area_min.setSingleStep(0.05)
        self.spin_area_min.setValue(-0.5)
        self.spin_area_min.valueChanged.connect(self.on_parameter_changed)
        min_layout.addWidget(self.spin_area_min)

        # 最大值 (正向波动)
        max_layout = QHBoxLayout()
        max_layout.addWidget(CaptionLabel("允许正向波动:"))
        self.spin_area_max = DoubleSpinBox()
        self.spin_area_max.setRange(0.01, 10.0)
        self.spin_area_max.setSingleStep(0.1)
        self.spin_area_max.setValue(1.5)
        self.spin_area_max.valueChanged.connect(self.on_parameter_changed)
        max_layout.addWidget(self.spin_area_max)

        layout.addWidget(self.cb_area)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)

        return group

    def create_width_group(self):
        """创建宽度范围控制组"""
        group = QGroupBox("宽度范围 (相对平均宽度波动)")
        layout = QVBoxLayout(group)

        self.cb_width = CheckBox("启用宽度过滤")
        self.cb_width.setChecked(False)
        self.cb_width.stateChanged.connect(self.on_parameter_changed)

        # 最小值 (负向波动)
        min_layout = QHBoxLayout()
        min_layout.addWidget(CaptionLabel("允许负向波动:"))
        self.spin_width_min = DoubleSpinBox()
        self.spin_width_min.setRange(-1.0, -0.01)
        self.spin_width_min.setSingleStep(0.05)
        self.spin_width_min.setValue(-0.5)
        self.spin_width_min.valueChanged.connect(self.on_parameter_changed)
        min_layout.addWidget(self.spin_width_min)

        # 最大值 (正向波动)
        max_layout = QHBoxLayout()
        max_layout.addWidget(CaptionLabel("允许正向波动:"))
        self.spin_width_max = DoubleSpinBox()
        self.spin_width_max.setRange(0.01, 10.0)
        self.spin_width_max.setSingleStep(0.1)
        self.spin_width_max.setValue(1.5)
        self.spin_width_max.valueChanged.connect(self.on_parameter_changed)
        max_layout.addWidget(self.spin_width_max)

        layout.addWidget(self.cb_width)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)

        return group

    def create_height_group(self):
        """创建高度范围控制组"""
        group = QGroupBox("高度范围 (相对平均高度波动)")
        layout = QVBoxLayout(group)

        self.cb_height = CheckBox("启用高度过滤")
        self.cb_height.setChecked(False)
        self.cb_height.stateChanged.connect(self.on_parameter_changed)

        # 最小值 (负向波动)
        min_layout = QHBoxLayout()
        min_layout.addWidget(CaptionLabel("允许负向波动:"))
        self.spin_height_min = DoubleSpinBox()
        self.spin_height_min.setRange(-1.0, -0.01)
        self.spin_height_min.setSingleStep(0.05)
        self.spin_height_min.setValue(-0.5)
        self.spin_height_min.valueChanged.connect(self.on_parameter_changed)
        min_layout.addWidget(self.spin_height_min)

        # 最大值 (正向波动)
        max_layout = QHBoxLayout()
        max_layout.addWidget(CaptionLabel("允许正向波动:"))
        self.spin_height_max = DoubleSpinBox()
        self.spin_height_max.setRange(0.01, 10.0)
        self.spin_height_max.setSingleStep(0.1)
        self.spin_height_max.setValue(1.5)
        self.spin_height_max.valueChanged.connect(self.on_parameter_changed)
        max_layout.addWidget(self.spin_height_max)

        layout.addWidget(self.cb_height)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)

        return group

    def create_aspect_group(self):
        """创建宽高比范围控制组"""
        group = QGroupBox("宽高比范围 (相对平均比例波动)")
        layout = QVBoxLayout(group)

        self.cb_aspect = CheckBox("启用宽高比过滤")
        self.cb_aspect.setChecked(True)
        self.cb_aspect.stateChanged.connect(self.on_parameter_changed)

        # 最小值 (负向波动)
        min_layout = QHBoxLayout()
        min_layout.addWidget(CaptionLabel("允许负向波动:"))
        self.spin_aspect_min = DoubleSpinBox()
        self.spin_aspect_min.setRange(-1.0, -0.01)
        self.spin_aspect_min.setSingleStep(0.05)
        self.spin_aspect_min.setValue(-0.5)
        self.spin_aspect_min.valueChanged.connect(self.on_parameter_changed)
        min_layout.addWidget(self.spin_aspect_min)

        # 最大值 (正向波动)
        max_layout = QHBoxLayout()
        max_layout.addWidget(CaptionLabel("允许正向波动:"))
        self.spin_aspect_max = DoubleSpinBox()
        self.spin_aspect_max.setRange(0.01, 10.0)
        self.spin_aspect_max.setSingleStep(0.1)
        self.spin_aspect_max.setValue(2.0)
        self.spin_aspect_max.valueChanged.connect(self.on_parameter_changed)
        max_layout.addWidget(self.spin_aspect_max)

        layout.addWidget(self.cb_aspect)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)

        return group

    def create_center_group(self):
        """创建中心点偏差控制组"""
        group = QGroupBox("中心点偏差范围 (px)")
        layout = QVBoxLayout(group)

        self.cb_center = CheckBox("启用中心点偏差过滤")
        self.cb_center.setChecked(True)
        self.cb_center.stateChanged.connect(self.on_parameter_changed)

        # X轴偏差
        x_layout = QHBoxLayout()
        x_layout.addWidget(CaptionLabel("X轴最大偏差:"))
        self.slider_center_x = Slider(Qt.Orientation.Horizontal)
        self.slider_center_x.setRange(0, 500)
        self.slider_center_x.setValue(100)
        self.slider_center_x.valueChanged.connect(self.on_parameter_changed)
        self.label_center_x = CaptionLabel("100 px")
        self.slider_center_x.valueChanged.connect(
            lambda v: self.label_center_x.setText(f"{v} px")
        )
        x_layout.addWidget(self.slider_center_x)
        x_layout.addWidget(self.label_center_x)

        # Y轴偏差
        y_layout = QHBoxLayout()
        y_layout.addWidget(CaptionLabel("Y轴最大偏差:"))
        self.slider_center_y = Slider(Qt.Orientation.Horizontal)
        self.slider_center_y.setRange(0, 500)
        self.slider_center_y.setValue(100)
        self.slider_center_y.valueChanged.connect(self.on_parameter_changed)
        self.label_center_y = CaptionLabel("100 px")
        self.slider_center_y.valueChanged.connect(
            lambda v: self.label_center_y.setText(f"{v} px")
        )
        y_layout.addWidget(self.slider_center_y)
        y_layout.addWidget(self.label_center_y)

        layout.addWidget(self.cb_center)
        layout.addLayout(x_layout)
        layout.addLayout(y_layout)

        return group

    def create_gray_group(self):
        """创建灰度范围控制组"""
        group = QGroupBox("灰度范围 (区域平均灰度 0~255)")
        layout = QVBoxLayout(group)

        self.cb_gray = CheckBox("启用灰度过滤")
        self.cb_gray.setChecked(False)
        self.cb_gray.stateChanged.connect(self.on_parameter_changed)

        # 最小灰度
        min_layout = QHBoxLayout()
        min_layout.addWidget(CaptionLabel("最小灰度:"))
        self.slider_gray_min = Slider(Qt.Orientation.Horizontal)
        self.slider_gray_min.setRange(0, 255)
        self.slider_gray_min.setValue(0)
        self.slider_gray_min.valueChanged.connect(self.on_parameter_changed)
        self.label_gray_min = CaptionLabel("0")
        self.slider_gray_min.valueChanged.connect(
            lambda v: self.label_gray_min.setText(str(v))
        )
        min_layout.addWidget(self.slider_gray_min)
        min_layout.addWidget(self.label_gray_min)

        # 最大灰度
        max_layout = QHBoxLayout()
        max_layout.addWidget(CaptionLabel("最大灰度:"))
        self.slider_gray_max = Slider(Qt.Orientation.Horizontal)
        self.slider_gray_max.setRange(0, 255)
        self.slider_gray_max.setValue(255)
        self.slider_gray_max.valueChanged.connect(self.on_parameter_changed)
        self.label_gray_max = CaptionLabel("255")
        self.slider_gray_max.valueChanged.connect(
            lambda v: self.label_gray_max.setText(str(v))
        )
        max_layout.addWidget(self.slider_gray_max)
        max_layout.addWidget(self.label_gray_max)

        layout.addWidget(self.cb_gray)
        layout.addLayout(min_layout)
        layout.addLayout(max_layout)

        return group

    def create_max_instances_group(self):
        """创建最大实例数控制组"""
        group = QGroupBox("最大检测数量")
        layout = QVBoxLayout(group)

        self.cb_max_inst = CheckBox("启用数量限制")
        self.cb_max_inst.setChecked(False)
        self.cb_max_inst.stateChanged.connect(self.on_parameter_changed)

        self.spin_max_inst = SpinBox()
        self.spin_max_inst.setRange(1, 1000)
        self.spin_max_inst.setValue(100)
        self.spin_max_inst.valueChanged.connect(self.on_parameter_changed)

        layout.addWidget(self.cb_max_inst)
        layout.addWidget(self.spin_max_inst)

        return group

    def create_preview_panel(self):
        """创建预览面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. 标题与基础统计
        header_layout = QHBoxLayout()
        title = StrongBodyLabel("实时预览")
        self.stats_label = CaptionLabel("原始: 0 | 过滤后: 0")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.stats_label)
        layout.addLayout(header_layout)

        # 2. 详细指标面板
        self.info_group = QGroupBox("数据指标与参考值 (波动值 = |当前-参考|/参考)")
        info_layout = QVBoxLayout(self.info_group)

        # 使用表格或网格布局显示详细信息
        stats_grid = QHBoxLayout()

        # 面积列
        self.area_stats_label = CaptionLabel("面积: -\n波动: -")
        # 宽高比列
        self.aspect_stats_label = CaptionLabel("宽高比: -\n波动: -")
        # 中心点列
        self.center_stats_label = CaptionLabel("中心点: -\n偏移: -")

        stats_grid.addWidget(self.area_stats_label)
        stats_grid.addSpacing(20)
        stats_grid.addWidget(self.aspect_stats_label)
        stats_grid.addSpacing(20)
        stats_grid.addWidget(self.center_stats_label)
        stats_grid.addStretch(1)

        # 参考值显示
        self.reference_label = CaptionLabel("参考值 (Support Info): 加载中...")
        self.reference_label.setStyleSheet("color: #aaaaaa;")

        info_layout.addLayout(stats_grid)
        info_layout.addWidget(self.reference_label)

        layout.addWidget(self.info_group)

        # 3. 图像预览
        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; }")
        self.image_label.setScaledContents(True)

        scroll = ScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(400, 300)
        scroll.setFrameShape(QFrame.NoFrame)

        layout.addWidget(scroll)

        return panel

    def load_image(self):
        """加载图像"""
        if not self.image_path or not os.path.exists(self.image_path):
            return

        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            # 缩放以适应窗口
            scaled_pixmap = pixmap.scaled(
                600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def on_parameter_changed(self):
        """参数变更时的处理"""
        self.apply_current_rules()

    def get_current_rules(self):
        """获取当前规则"""
        rules = {}

        # 置信度
        if self.cb_conf.isChecked():
            rules['conf_threshold'] = self.slider_conf.value() / 100.0

        # 面积范围
        if self.cb_area.isChecked():
            rules['area_range'] = [
                abs(self.spin_area_min.value()), # 转为正数供后端逻辑使用
                self.spin_area_max.value()
            ]

        # 宽度范围
        if self.cb_width.isChecked():
            rules['width_range'] = [
                abs(self.spin_width_min.value()),
                self.spin_width_max.value()
            ]

        # 高度范围
        if self.cb_height.isChecked():
            rules['height_range'] = [
                abs(self.spin_height_min.value()),
                self.spin_height_max.value()
            ]

        # 宽高比范围
        if self.cb_aspect.isChecked():
            rules['aspect_ratio_range'] = [
                abs(self.spin_aspect_min.value()), # 转为正数供后端逻辑使用
                self.spin_aspect_max.value()
            ]

        # 中心点偏差
        if self.cb_center.isChecked():
            rules['center_range'] = [
                float(self.slider_center_x.value()), # 直接传像素值
                float(self.slider_center_y.value())
            ]

        # 灰度范围
        if self.cb_gray.isChecked():
            rules['gray_range'] = [
                float(self.slider_gray_min.value()),
                float(self.slider_gray_max.value())
            ]

        # 最大实例数
        if self.cb_max_inst.isChecked():
            rules['max_instances'] = self.spin_max_inst.value()

        return rules

    def apply_current_rules(self):
        """应用当前规则并更新预览"""
        try:
            # 获取当前规则
            self.current_rules = self.get_current_rules()

            # 导入规则应用函数
            from core.backend.utils import apply_rules

            # 加载图像（灰度过滤需要）
            image = None
            if 'gray_range' in self.current_rules and self.image_path:
                image = imread_unicode(self.image_path)

            # 应用规则
            self.current_instances = apply_rules(
                self.original_instances.copy(),
                self.current_rules,
                self.support_info,
                image
            )

            # 更新基础统计信息
            original_count = len(self.original_instances)
            filtered_count = len(self.current_instances)
            self.stats_label.setText(f"原始: {original_count} | 过滤后: {filtered_count}")

            # 计算并更新详细统计指标
            self._update_detailed_statistics()

            # 更新图像预览
            self.update_preview()

        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"[SmartParameterDialog] 应用规则时出错: {e}")

    def _update_detailed_statistics(self):
        """计算并更新详细统计指标和波动值"""
        # 1. 显示参考值
        ref_area = self.support_info.get('avg_area', 0)
        ref_width = self.support_info.get('avg_width', 0)
        ref_height = self.support_info.get('avg_height', 0)
        ref_aspect = self.support_info.get('avg_aspect_ratio', 0)
        ref_cx = self.support_info.get('avg_center_x', 0)
        ref_cy = self.support_info.get('avg_center_y', 0)

        self.reference_label.setText(
            f"参考值: 面积={ref_area:.1f}, 宽={ref_width:.1f}, 高={ref_height:.1f}, 宽高比={ref_aspect:.2f}, 中心=({ref_cx:.1f}, {ref_cy:.1f})"
        )

        if not self.current_instances:
            self.area_stats_label.setText("面积: -\n波动: -")
            self.aspect_stats_label.setText("宽高比: -\n波动: -")
            self.center_stats_label.setText("中心点: -\n偏移: -")
            return

        # 2. 计算当前指标
        areas = []
        aspects = []
        centers_x = []
        centers_y = []

        for inst in self.current_instances:
            # 面积 (优先使用之前计算出的 _calc_area，如果已经被清理则重新计算)
            if 'mask' in inst and inst['mask'] is not None:
                areas.append(np.sum(inst['mask'] > 0))
            elif 'polygons' in inst and inst['polygons']:
                import cv2
                poly_area = sum(cv2.contourArea(np.array(p, dtype=np.int32).reshape((-1, 1, 2))) for p in inst['polygons'])
                areas.append(poly_area)

            # 宽高比
            if 'bbox' in inst:
                x, y, w, h = inst['bbox']
                if h > 0:
                    aspects.append(w / h)
                    centers_x.append(x + w / 2)
                    centers_y.append(y + h / 2)

        # 3. 更新 UI 显示
        # 面积统计
        if areas:
            cur_avg_area = np.mean(areas)
            # 计算带正负号的波动值
            area_fluct = (cur_avg_area - ref_area) / ref_area if ref_area > 0 else 0
            # 格式化显示，如果是正数显示 + 号
            fluct_str = f"{area_fluct*100:+.1f}%"
            self.area_stats_label.setText(f"平均面积: {cur_avg_area:.1f}\n波动: {fluct_str}")

            if abs(area_fluct) > 0.2: # 绝对值波动超过20%标红
                self.area_stats_label.setStyleSheet("color: #ff4d4f;")
            else:
                self.area_stats_label.setStyleSheet("")

        # 宽高比统计
        if aspects:
            cur_avg_aspect = np.mean(aspects)
            # 计算带正负号的波动值
            aspect_fluct = (cur_avg_aspect - ref_aspect) / ref_aspect if ref_aspect > 0 else 0
            fluct_str = f"{aspect_fluct*100:+.1f}%"
            self.aspect_stats_label.setText(f"平均宽高比: {cur_avg_aspect:.2f}\n波动: {fluct_str}")

            if abs(aspect_fluct) > 0.2:
                self.aspect_stats_label.setStyleSheet("color: #ff4d4f;")
            else:
                self.aspect_stats_label.setStyleSheet("")

        # 中心点统计
        if centers_x:
            cur_cx = np.mean(centers_x)
            cur_cy = np.mean(centers_y)
            # 计算 X/Y 轴分别的偏移
            dx = cur_cx - ref_cx
            dy = cur_cy - ref_cy
            # 计算欧氏距离总偏移
            offset = np.sqrt(dx**2 + dy**2)

            self.center_stats_label.setText(
                f"平均中心: ({cur_cx:.1f}, {cur_cy:.1f})\n"
                f"偏移: {offset:.1f} px (ΔX:{dx:+.1f}, ΔY:{dy:+.1f})"
            )

            if offset > 50: # 偏移超过50像素标黄
                self.center_stats_label.setStyleSheet("color: #faad14;")
            else:
                self.center_stats_label.setStyleSheet("")

    def update_preview(self):
        """更新预览图像"""
        try:
            if not self.image_path or not os.path.exists(self.image_path):
                return

            import cv2
            from core.backend.utils import create_visualization

            # 加载原图
            image = imread_unicode(self.image_path)
            if image is None:
                return

            # 创建可视化
            preview_image = create_visualization(image, self.current_instances, alpha=0.6)

            # 转换为QPixmap
            height, width, channel = preview_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(preview_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)

            # 缩放显示
            scaled_pixmap = pixmap.scaled(
                600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"[SmartParameterDialog] 更新预览时出错: {e}")

    def reset_parameters(self):
        """重置参数到默认值"""
        # 置信度
        self.slider_conf.setValue(50)
        self.cb_conf.setChecked(True)

        # 面积范围
        self.spin_area_min.setValue(-0.5)
        self.spin_area_max.setValue(1.5)
        self.cb_area.setChecked(True)

        # 宽度范围
        self.spin_width_min.setValue(-0.5)
        self.spin_width_max.setValue(1.5)
        self.cb_width.setChecked(False)

        # 高度范围
        self.spin_height_min.setValue(-0.5)
        self.spin_height_max.setValue(1.5)
        self.cb_height.setChecked(False)

        # 宽高比范围
        self.spin_aspect_min.setValue(-0.5)
        self.spin_aspect_max.setValue(2.0)
        self.cb_aspect.setChecked(True)

        # 中心点偏差
        self.slider_center_x.setValue(20)
        self.slider_center_y.setValue(20)
        self.cb_center.setChecked(True)

        # 灰度范围
        self.slider_gray_min.setValue(0)
        self.slider_gray_max.setValue(255)
        self.cb_gray.setChecked(False)

        # 最大实例数
        self.spin_max_inst.setValue(100)
        self.cb_max_inst.setChecked(False)

    def apply_and_close(self):
        """应用参数并关闭窗口"""
        self.parameter_changed.emit(self.current_rules)
        self.accept()

    def get_rules(self):
        """获取当前规则"""
        return self.current_rules
