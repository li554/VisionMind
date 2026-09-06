from qfluentwidgets import (StrongBodyLabel, CaptionLabel,
                            ComboBox as FluentComboBox, SpinBox, CheckBox)
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QGridLayout, QWidget
from core.common.custom_card import SimpleSolidCardWidget as SimpleCardWidget
from core.common.settings import settings


class AnnotationSettingsCard(SimpleCardWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(StrongBodyLabel('标注设置'))

        # 交互模型
        model_group = SimpleCardWidget()
        model_layout = QVBoxLayout(model_group)
        model_layout.setContentsMargins(12, 12, 12, 12)
        model_layout.addWidget(CaptionLabel('交互模型'))

        model_grid = QGridLayout()
        model_grid.setSpacing(8)

        model_grid.addWidget(CaptionLabel('模型类型'), 0, 0)
        self.model_combo = FluentComboBox()
        self.model_combo.addItems(['mobilesam_onnx', 'sam_onnx_b', 'sam3'])
        model_grid.addWidget(self.model_combo, 0, 1)

        model_grid.addWidget(CaptionLabel('推理尺寸'), 1, 0)
        self.img_size_spin = SpinBox()
        self.img_size_spin.setRange(256, 2048)
        self.img_size_spin.setSingleStep(64)
        model_grid.addWidget(self.img_size_spin, 1, 1)

        model_layout.addLayout(model_grid)
        layout.addWidget(model_group)

        # 功能开关
        option_group = SimpleCardWidget()
        option_layout = QVBoxLayout(option_group)
        option_layout.setContentsMargins(12, 12, 12, 12)
        option_layout.addWidget(CaptionLabel('功能开关'))

        self.auto_save_cb = CheckBox('自动保存')
        self.ai_assistance_cb = CheckBox('AI辅助')
        option_layout.addWidget(self.auto_save_cb)
        option_layout.addWidget(self.ai_assistance_cb)

        layout.addWidget(option_group)

    def load_settings(self):
        self.model_combo.setCurrentText(settings.get("default_interactive_model", "mobilesam_onnx"))
        self.img_size_spin.setValue(settings.get("default_interactive_img_size", 1024))
        self.auto_save_cb.setChecked(settings.get("auto_save_enabled", "打开") == "打开")
        self.ai_assistance_cb.setChecked(settings.get("ai_assistance_enabled", "打开") == "打开")

    def save_settings(self):
        settings.set("default_interactive_model", self.model_combo.currentText())
        settings.set("default_interactive_img_size", self.img_size_spin.value())
        settings.set("auto_save_enabled", "打开" if self.auto_save_cb.isChecked() else "关闭")
        settings.set("ai_assistance_enabled", "打开" if self.ai_assistance_cb.isChecked() else "关闭")
        settings.save()
        self.settings_changed.emit()
