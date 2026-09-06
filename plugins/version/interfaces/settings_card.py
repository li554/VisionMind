from qfluentwidgets import (StrongBodyLabel, CaptionLabel,
                            ComboBox as FluentComboBox, SpinBox)
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QGridLayout
from core.common.custom_card import SimpleSolidCardWidget as SimpleCardWidget
from core.common.plugin_config import PluginConfig
import os

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
version_config = PluginConfig(PLUGIN_DIR)


class VersionSettingsCard(SimpleCardWidget):
    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = version_config
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(StrongBodyLabel('版本设置'))

        # 默认导出格式
        grid = QGridLayout()
        grid.setSpacing(8)

        grid.addWidget(CaptionLabel('默认导出格式'), 0, 0)
        self.default_export_format_combo = FluentComboBox()
        self.default_export_format_combo.addItems(['YOLO', 'COCO', 'VOC'])
        grid.addWidget(self.default_export_format_combo, 0, 1)

        layout.addLayout(grid)

    def load_settings(self):
        self.default_export_format_combo.setCurrentText(self.config.get("default_export_format", "YOLO"))

    def save_settings(self):
        self.config.set("default_export_format", self.default_export_format_combo.currentText())
        self.config.save()
        self.settings_changed.emit()
