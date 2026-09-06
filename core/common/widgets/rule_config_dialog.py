import traceback

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
from qfluentwidgets import (PrimaryPushButton, PushButton, CaptionLabel,
                            LineEdit, SpinBox, DoubleSpinBox, ComboBox, CheckBox,
                            InfoBar)

from core.common.settings import settings
from core.common.widgets.base import BaseDialog


class RuleConfigDialog(BaseDialog):
    """规则配置对话框

    Args:
        parent: 父窗口
        initial_rules: 初始规则字典
        select_examples: 选中的示例列表
        smart_tuning_fn: 智能调参回调函数，签名为
            smart_tuning_fn(image_path, model_type, rules) -> dict
            当提供时显示智能调参按钮，为 None 时隐藏
        image_path: 当前图像路径（传给 smart_tuning_fn）
        model_type: 模型类型（传给 smart_tuning_fn）
    """
    def __init__(self, parent=None, initial_rules=None, select_examples=None,
                 smart_tuning_fn=None, image_path=None, model_type=None):
        super().__init__(parent)
        self.setWindowTitle("规则指定 - 过滤自动标注结果")
        self.setMinimumWidth(450)
        self.rules = initial_rules or {}
        self.select_examples = select_examples or []
        self._has_examples = bool(self.select_examples)
        self.smart_tuning_fn = smart_tuning_fn
        self.image_path = image_path
        self.model_type = model_type
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Text Prompt
        text_layout = QVBoxLayout()
        text_layout.addWidget(CaptionLabel("文本提示 (增强检测):"))
        self.text_prompt_edit = LineEdit()
        self.text_prompt_edit.setPlaceholderText("例如: defect, scratch, crack")
        self.text_prompt_edit.setText(self.rules.get('text', ''))
        text_layout.addWidget(self.text_prompt_edit)
        layout.addLayout(text_layout)

        # Max Instances
        max_inst_layout = QHBoxLayout()
        self.max_instances_cb = CheckBox("启用最大检测数量限制")
        self.max_instances_spin = SpinBox()
        self.max_instances_spin.setRange(1, 1000)
        self.max_instances_spin.setValue(self.rules.get('max_instances', 100))
        self.max_instances_spin.setEnabled(False)
        self.max_instances_cb.setChecked(False)
        self.max_instances_cb.stateChanged.connect(lambda s: self.max_instances_spin.setEnabled(s == Qt.CheckState.Checked.value))

        max_inst_layout.addWidget(self.max_instances_cb)
        max_inst_layout.addStretch(1)
        max_inst_layout.addWidget(self.max_instances_spin)
        layout.addLayout(max_inst_layout)

        # Area Range
        area_group_layout = QVBoxLayout()
        if self._has_examples:
            self.area_cb = CheckBox("启用面积过滤 (相对于示例, 1.0表示原始大小)")
        else:
            self.area_cb = CheckBox("启用面积过滤 (绝对值, 单位: 像素)")
        area_group_layout.addWidget(self.area_cb)

        area_layout = QHBoxLayout()
        self.area_min_spin = DoubleSpinBox()
        if self._has_examples:
            # 相对模式：负值表示允许缩小的比例
            self.area_min_spin.setRange(-1.0, 0.0)
            area_val = self.rules.get('area_range', [0.5, 1.5])[0]
            self.area_min_spin.setValue(-abs(area_val) if area_val > 0 else area_val)
        else:
            # 绝对值模式：直接用像素值
            self.area_min_spin.setRange(0, 10000000)
            area_val = self.rules.get('area_range', [100, 50000])[0]
            self.area_min_spin.setValue(abs(area_val))
        self.area_max_spin = DoubleSpinBox()
        if self._has_examples:
            self.area_max_spin.setRange(0.0, 100.0)
            self.area_max_spin.setValue(self.rules.get('area_range', [0.5, 1.5])[1])
        else:
            self.area_max_spin.setRange(0, 10000000)
            self.area_max_spin.setValue(self.rules.get('area_range', [100, 50000])[1])

        self.area_min_spin.setEnabled(False)
        self.area_max_spin.setEnabled(False)
        self.area_cb.setChecked(False)
        self.area_cb.stateChanged.connect(lambda s: [self.area_min_spin.setEnabled(s == Qt.CheckState.Checked.value), self.area_max_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        if self._has_examples:
            area_layout.addWidget(CaptionLabel("允许负向波动:"))
            area_layout.addWidget(self.area_min_spin)
            area_layout.addWidget(CaptionLabel("允许正向波动:"))
        else:
            area_layout.addWidget(CaptionLabel("最小面积:"))
            area_layout.addWidget(self.area_min_spin)
            area_layout.addWidget(CaptionLabel("最大面积:"))
        area_layout.addWidget(self.area_max_spin)
        area_group_layout.addLayout(area_layout)
        layout.addLayout(area_group_layout)

        # Width Range
        width_group_layout = QVBoxLayout()
        if self._has_examples:
            self.width_cb = CheckBox("启用宽度过滤 (相对于示例)")
        else:
            self.width_cb = CheckBox("启用宽度过滤 (绝对值, 单位: 像素)")
        width_group_layout.addWidget(self.width_cb)

        width_layout = QHBoxLayout()
        self.width_min_spin = DoubleSpinBox()
        if self._has_examples:
            self.width_min_spin.setRange(-1.0, 0.0)
            width_val = self.rules.get('width_range', [0.5, 1.5])[0]
            self.width_min_spin.setValue(-abs(width_val) if width_val > 0 else width_val)
        else:
            self.width_min_spin.setRange(0, 100000)
            width_val = self.rules.get('width_range', [10, 2000])[0]
            self.width_min_spin.setValue(abs(width_val))
        self.width_max_spin = DoubleSpinBox()
        if self._has_examples:
            self.width_max_spin.setRange(0.0, 100.0)
            self.width_max_spin.setValue(self.rules.get('width_range', [0.5, 1.5])[1])
        else:
            self.width_max_spin.setRange(0, 100000)
            self.width_max_spin.setValue(self.rules.get('width_range', [10, 2000])[1])

        self.width_min_spin.setEnabled(False)
        self.width_max_spin.setEnabled(False)
        self.width_cb.setChecked(False)
        self.width_cb.stateChanged.connect(lambda s: [self.width_min_spin.setEnabled(s == Qt.CheckState.Checked.value), self.width_max_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        if self._has_examples:
            width_layout.addWidget(CaptionLabel("允许负向波动:"))
            width_layout.addWidget(self.width_min_spin)
            width_layout.addWidget(CaptionLabel("允许正向波动:"))
        else:
            width_layout.addWidget(CaptionLabel("最小宽度:"))
            width_layout.addWidget(self.width_min_spin)
            width_layout.addWidget(CaptionLabel("最大宽度:"))
        width_layout.addWidget(self.width_max_spin)
        width_group_layout.addLayout(width_layout)
        layout.addLayout(width_group_layout)

        # Height Range
        height_group_layout = QVBoxLayout()
        if self._has_examples:
            self.height_cb = CheckBox("启用高度过滤 (相对于示例)")
        else:
            self.height_cb = CheckBox("启用高度过滤 (绝对值, 单位: 像素)")
        height_group_layout.addWidget(self.height_cb)

        height_layout = QHBoxLayout()
        self.height_min_spin = DoubleSpinBox()
        if self._has_examples:
            self.height_min_spin.setRange(-1.0, 0.0)
            height_val = self.rules.get('height_range', [0.5, 1.5])[0]
            self.height_min_spin.setValue(-abs(height_val) if height_val > 0 else height_val)
        else:
            self.height_min_spin.setRange(0, 100000)
            height_val = self.rules.get('height_range', [10, 2000])[0]
            self.height_min_spin.setValue(abs(height_val))
        self.height_max_spin = DoubleSpinBox()
        if self._has_examples:
            self.height_max_spin.setRange(0.0, 100.0)
            self.height_max_spin.setValue(self.rules.get('height_range', [0.5, 1.5])[1])
        else:
            self.height_max_spin.setRange(0, 100000)
            self.height_max_spin.setValue(self.rules.get('height_range', [10, 2000])[1])

        self.height_min_spin.setEnabled(False)
        self.height_max_spin.setEnabled(False)
        self.height_cb.setChecked(False)
        self.height_cb.stateChanged.connect(lambda s: [self.height_min_spin.setEnabled(s == Qt.CheckState.Checked.value), self.height_max_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        if self._has_examples:
            height_layout.addWidget(CaptionLabel("允许负向波动:"))
            height_layout.addWidget(self.height_min_spin)
            height_layout.addWidget(CaptionLabel("允许正向波动:"))
        else:
            height_layout.addWidget(CaptionLabel("最小高度:"))
            height_layout.addWidget(self.height_min_spin)
            height_layout.addWidget(CaptionLabel("最大高度:"))
        height_layout.addWidget(self.height_max_spin)
        height_group_layout.addLayout(height_layout)
        layout.addLayout(height_group_layout)

        # Aspect Ratio Range
        aspect_group_layout = QVBoxLayout()
        self.aspect_cb = CheckBox("启用宽高比过滤 (W/H)")
        aspect_group_layout.addWidget(self.aspect_cb)

        aspect_layout = QHBoxLayout()
        self.aspect_min_spin = DoubleSpinBox()
        if self._has_examples:
            self.aspect_min_spin.setRange(-1.0, 0.0)
            aspect_val = self.rules.get('aspect_ratio_range', [0.1, 10.0])[0]
            self.aspect_min_spin.setValue(-abs(aspect_val) if aspect_val > 0 else aspect_val)
        else:
            self.aspect_min_spin.setRange(0.0, 1000.0)
            aspect_val = self.rules.get('aspect_ratio_range', [0.1, 10.0])[0]
            self.aspect_min_spin.setValue(abs(aspect_val))
        self.aspect_max_spin = DoubleSpinBox()
        self.aspect_max_spin.setRange(0.0, 1000.0)
        self.aspect_max_spin.setValue(self.rules.get('aspect_ratio_range', [0.1, 10.0])[1])

        self.aspect_min_spin.setEnabled(False)
        self.aspect_max_spin.setEnabled(False)
        self.aspect_cb.setChecked(False)
        self.aspect_cb.stateChanged.connect(lambda s: [self.aspect_min_spin.setEnabled(s == Qt.CheckState.Checked.value), self.aspect_max_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        if self._has_examples:
            aspect_layout.addWidget(CaptionLabel("允许负向波动:"))
            aspect_layout.addWidget(self.aspect_min_spin)
            aspect_layout.addWidget(CaptionLabel("允许正向波动:"))
        else:
            aspect_layout.addWidget(CaptionLabel("最小宽高比:"))
            aspect_layout.addWidget(self.aspect_min_spin)
            aspect_layout.addWidget(CaptionLabel("最大宽高比:"))
        aspect_layout.addWidget(self.aspect_max_spin)
        aspect_group_layout.addLayout(aspect_layout)
        layout.addLayout(aspect_group_layout)

        # Center Range (偏差范围)
        center_group_layout = QVBoxLayout()
        self.center_cb = CheckBox("启用中心点偏差过滤 (单位: 像素)")
        center_group_layout.addWidget(self.center_cb)

        center_layout = QHBoxLayout()
        self.center_x_spin = DoubleSpinBox()
        self.center_x_spin.setRange(0.0, 500.0)
        self.center_x_spin.setValue(self.rules.get('center_range', [50, 50])[0])
        self.center_y_spin = DoubleSpinBox()
        self.center_y_spin.setRange(0.0, 500.0)
        self.center_y_spin.setValue(self.rules.get('center_range', [50, 50])[1])

        self.center_x_spin.setEnabled(False)
        self.center_y_spin.setEnabled(False)
        self.center_cb.setChecked(False)
        self.center_cb.stateChanged.connect(lambda s: [self.center_x_spin.setEnabled(s == Qt.CheckState.Checked.value), self.center_y_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        center_layout.addWidget(CaptionLabel("X轴偏差 (px):"))
        center_layout.addWidget(self.center_x_spin)
        center_layout.addWidget(CaptionLabel("Y轴偏差 (px):"))
        center_layout.addWidget(self.center_y_spin)
        center_group_layout.addLayout(center_layout)
        layout.addLayout(center_group_layout)

        # Gray Range (灰度过滤)
        gray_group_layout = QVBoxLayout()
        self.gray_cb = CheckBox("启用灰度过滤 (区域平均灰度, 0~255)")
        gray_group_layout.addWidget(self.gray_cb)

        gray_layout = QHBoxLayout()
        self.gray_min_spin = DoubleSpinBox()
        self.gray_min_spin.setRange(0.0, 255.0)
        self.gray_min_spin.setValue(self.rules.get('gray_range', [0, 255])[0])
        self.gray_min_spin.setSingleStep(1.0)
        self.gray_max_spin = DoubleSpinBox()
        self.gray_max_spin.setRange(0.0, 255.0)
        self.gray_max_spin.setValue(self.rules.get('gray_range', [0, 255])[1])
        self.gray_max_spin.setSingleStep(1.0)

        self.gray_min_spin.setEnabled(False)
        self.gray_max_spin.setEnabled(False)
        self.gray_cb.setChecked(False)
        self.gray_cb.stateChanged.connect(lambda s: [self.gray_min_spin.setEnabled(s == Qt.CheckState.Checked.value), self.gray_max_spin.setEnabled(s == Qt.CheckState.Checked.value)])

        gray_layout.addWidget(CaptionLabel("最小灰度:"))
        gray_layout.addWidget(self.gray_min_spin)
        gray_layout.addWidget(CaptionLabel("最大灰度:"))
        gray_layout.addWidget(self.gray_max_spin)
        gray_group_layout.addLayout(gray_layout)
        layout.addLayout(gray_group_layout)

        # Confidence Threshold
        conf_layout = QHBoxLayout()
        self.conf_cb = CheckBox("启用置信度阈值")
        self.conf_spin = DoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(self.rules.get('conf_threshold', 0.3))
        self.conf_spin.setSingleStep(0.05)

        self.conf_cb.setChecked('conf_threshold' in self.rules if self.rules else True)
        self.conf_spin.setEnabled('conf_threshold' in self.rules if self.rules else True)
        self.conf_cb.stateChanged.connect(lambda s: self.conf_spin.setEnabled(s == Qt.CheckState.Checked.value))

        conf_layout.addWidget(self.conf_cb)
        conf_layout.addStretch(1)
        conf_layout.addWidget(self.conf_spin)
        layout.addLayout(conf_layout)

        # Mode Selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(CaptionLabel("拼接模式:"))
        self.mode_combo = ComboBox()
        self.mode_combo.addItems(["特征复用 (Reuse)", "网格拼接 (Grid)", "复制粘贴 (Copy-Paste)"])

        # Load mode from rules
        mode_map = {"reuse": 0, "grid": 1, "copy_paste": 2}
        current_mode = self.rules.get('mode', 'reuse')
        self.mode_combo.setCurrentIndex(mode_map.get(current_mode, 0))

        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_smart_tuning = PushButton("智能调参")
        self.btn_smart_tuning.clicked.connect(self.on_smart_tuning_clicked)
        self.btn_ok = PrimaryPushButton("确定")
        self.btn_ok.clicked.connect(self.on_accept)

        # 仅在提供了 smart_tuning_fn 时显示智能调参按钮
        if self.smart_tuning_fn is None:
            self.btn_smart_tuning.hide()

        btn_layout.addWidget(self.btn_smart_tuning)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    @Slot()
    def on_accept(self):
        self.rules = self.get_current_rules_from_ui()
        self.accept()

    @Slot()
    def on_smart_tuning_clicked(self):
        """智能调参按钮点击事件

        通过 smart_tuning_fn 回调执行推理，而非直接访问父窗口。
        """
        if self.smart_tuning_fn is None:
            return

        try:
            if not self.image_path:
                InfoBar.warning(title='提示', content='请先选择一张图像', parent=self)
                return

            # 获取当前规则
            current_rules = self.get_current_rules_from_ui()

            InfoBar.info(title='正在推理', content='正在执行模型推理，请稍候...', parent=self)

            # 通过回调执行推理
            instances = self.smart_tuning_fn(self.image_path, self.model_type, current_rules)

            if not instances or instances.get('status') == 'error':
                error_msg = instances.get('message', '未知错误') if instances else '模型推理未产生任何结果'
                InfoBar.warning(title='推理失败', content=error_msg, parent=self)
                return

            # 显示智能调参对话框
            from core.common.widgets.smart_parameter_dialog import SmartParameterDialog
            dialog = SmartParameterDialog(
                parent=self,
                image_path=self.image_path,
                instances=instances
            )

            if dialog.exec_() == QDialog.Accepted:
                # 用户点击了应用按钮，更新当前规则
                new_rules = dialog.get_rules()
                self.set_rules_to_ui(new_rules)
                InfoBar.success(title='参数更新', content='规则参数已更新', parent=self)

        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"[RuleConfigDialog] 智能调参失败: {e}")
            traceback.print_exc()
            InfoBar.error(title='智能调参失败', content=str(e), parent=self)

    def get_rules(self):
        return self.rules

    def get_current_rules_from_ui(self):
        """从UI控件获取当前规则"""
        rules = {}

        # Text Prompt
        text = self.text_prompt_edit.text().strip()
        if text:
            rules['text'] = text

        if self.max_instances_cb.isChecked():
            rules['max_instances'] = self.max_instances_spin.value()

        if self.area_cb.isChecked():
            if self._has_examples:
                # 相对模式：后台期望正数比例，取绝对值
                rules['area_range'] = [abs(self.area_min_spin.value()), self.area_max_spin.value()]
            else:
                # 绝对值模式：直接存像素值
                rules['area_range'] = [self.area_min_spin.value(), self.area_max_spin.value()]

        if self.width_cb.isChecked():
            if self._has_examples:
                rules['width_range'] = [abs(self.width_min_spin.value()), self.width_max_spin.value()]
            else:
                rules['width_range'] = [self.width_min_spin.value(), self.width_max_spin.value()]

        if self.height_cb.isChecked():
            if self._has_examples:
                rules['height_range'] = [abs(self.height_min_spin.value()), self.height_max_spin.value()]
            else:
                rules['height_range'] = [self.height_min_spin.value(), self.height_max_spin.value()]

        if self.aspect_cb.isChecked():
            if self._has_examples:
                rules['aspect_ratio_range'] = [abs(self.aspect_min_spin.value()), self.aspect_max_spin.value()]
            else:
                rules['aspect_ratio_range'] = [self.aspect_min_spin.value(), self.aspect_max_spin.value()]

        if self.center_cb.isChecked():
            rules['center_range'] = [self.center_x_spin.value(), self.center_y_spin.value()]

        if self.gray_cb.isChecked():
            rules['gray_range'] = [self.gray_min_spin.value(), self.gray_max_spin.value()]

        if self.conf_cb.isChecked():
            rules['conf_threshold'] = self.conf_spin.value()

        if self.mode_combo.currentIndex() == 0:
            rules['mode'] = "reuse"
        elif self.mode_combo.currentIndex() == 1:
            rules['mode'] = "grid"
        else:
            rules['mode'] = "copy_paste"

        return rules

    def set_rules_to_ui(self, rules):
        """将规则设置到UI控件"""
        if not rules:
            return

        # Text Prompt
        if 'text' in rules:
            self.text_prompt_edit.setText(rules['text'])
        else:
            self.text_prompt_edit.clear()

        # Max Instances
        if 'max_instances' in rules:
            self.max_instances_cb.setChecked(True)
            self.max_instances_spin.setValue(rules['max_instances'])
        else:
            self.max_instances_cb.setChecked(False)

        # Area Range
        if 'area_range' in rules and len(rules['area_range']) == 2:
            self.area_cb.setChecked(True)
            val = rules['area_range'][0]
            if self._has_examples:
                self.area_min_spin.setValue(-abs(val) if val > 0 else val)
            else:
                self.area_min_spin.setValue(abs(val))
            self.area_max_spin.setValue(rules['area_range'][1])
        else:
            self.area_cb.setChecked(False)

        # Width Range
        if 'width_range' in rules and len(rules['width_range']) == 2:
            self.width_cb.setChecked(True)
            val = rules['width_range'][0]
            if self._has_examples:
                self.width_min_spin.setValue(-abs(val) if val > 0 else val)
            else:
                self.width_min_spin.setValue(abs(val))
            self.width_max_spin.setValue(rules['width_range'][1])
        else:
            self.width_cb.setChecked(False)

        # Height Range
        if 'height_range' in rules and len(rules['height_range']) == 2:
            self.height_cb.setChecked(True)
            val = rules['height_range'][0]
            if self._has_examples:
                self.height_min_spin.setValue(-abs(val) if val > 0 else val)
            else:
                self.height_min_spin.setValue(abs(val))
            self.height_max_spin.setValue(rules['height_range'][1])
        else:
            self.height_cb.setChecked(False)

        # Aspect Ratio Range
        if 'aspect_ratio_range' in rules and len(rules['aspect_ratio_range']) == 2:
            self.aspect_cb.setChecked(True)
            val = rules['aspect_ratio_range'][0]
            if self._has_examples:
                self.aspect_min_spin.setValue(-abs(val) if val > 0 else val)
            else:
                self.aspect_min_spin.setValue(abs(val))
            self.aspect_max_spin.setValue(rules['aspect_ratio_range'][1])
        else:
            self.aspect_cb.setChecked(False)

        # Center Range
        if 'center_range' in rules and len(rules['center_range']) == 2:
            self.center_cb.setChecked(True)
            self.center_x_spin.setValue(rules['center_range'][0])
            self.center_y_spin.setValue(rules['center_range'][1])
        else:
            self.center_cb.setChecked(False)

        # Gray Range
        if 'gray_range' in rules and len(rules['gray_range']) == 2:
            self.gray_cb.setChecked(True)
            self.gray_min_spin.setValue(rules['gray_range'][0])
            self.gray_max_spin.setValue(rules['gray_range'][1])
        else:
            self.gray_cb.setChecked(False)

        # Confidence Threshold
        if 'conf_threshold' in rules:
            self.conf_cb.setChecked(True)
            self.conf_spin.setValue(rules['conf_threshold'])
        else:
            self.conf_cb.setChecked(False)

        # Mode
        mode_map = {"reuse": 0, "grid": 1, "copy_paste": 2}
        if 'mode' in rules and rules['mode'] in mode_map:
            self.mode_combo.setCurrentIndex(mode_map[rules['mode']])
        else:
            self.mode_combo.setCurrentIndex(0)
