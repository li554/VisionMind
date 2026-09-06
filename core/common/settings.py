import os
import json
import sys
import warnings
import multiprocessing
from PySide6.QtCore import QObject, Signal
from .config import Config

def is_main_process():
    """Checks if the current process is the main GUI process."""
    return multiprocessing.current_process().name == 'MainProcess'

if not is_main_process():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


class CoreSettingsManager(QObject):
    """
    核心配置管理器，从 core/core.json 加载配置。
    支持新的 schema 格式（数组）和旧的 key-value 格式。
    """
    settings_changed = Signal(str, object)

    def __init__(self, filename="core/core.json"):
        super().__init__()
        root_dir = Config.ROOT_DIR
        self.filepath = os.path.join(root_dir, filename)
        self._schema_data: list = []
        self._data: dict = {}
        self.load()

    def load(self):
        """从 core.json 加载配置（支持新旧格式）"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    # 新格式：schema 数组
                    self._schema_data = loaded
                    self._data = {}
                    for item in loaded:
                        key = item.get('key')
                        if key:
                            self._data[key] = item.get('value', item.get('default'))
                else:
                    # 旧格式：简单 key-value
                    self._schema_data = self._convert_old_to_schema(loaded)
                    self._data = dict(loaded)
                if is_main_process():
                    print(f"[CoreSettingsManager] Loaded settings from {self.filepath}")
            except Exception as e:
                if self.DEBUG:
                    raise
                if is_main_process():
                    print(f"[CoreSettingsManager] Error loading settings: {e}")
        else:
            if is_main_process():
                print(f"[CoreSettingsManager] No settings file found at {self.filepath}, using defaults")

    def _convert_old_to_schema(self, old_data: dict) -> list:
        """将旧格式转换为新 schema 格式（兼容用）"""
        schema = []
        for key, value in old_data.items():
            if isinstance(value, (dict, list)):
                continue  # 跳过复杂对象
            item = {"key": key, "title": key, "type": "string", "default": value, "value": value}
            schema.append(item)
        return schema

    def save(self):
        """保存配置到 core.json"""
        for item in self._schema_data:
            key = item.get('key')
            if key and key in self._data:
                item['value'] = self._data[key]
        # Also persist dynamic keys not in schema
        schema_keys = {item.get('key') for item in self._schema_data}
        for key, value in self._data.items():
            if key not in schema_keys:
                self._schema_data.append({"key": key, "value": value})
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._schema_data, f, indent=2, ensure_ascii=False)
            if is_main_process():
                print(f"[CoreSettingsManager] Saved settings to {self.filepath}")
        except Exception as e:
            if self.DEBUG:
                raise
            if is_main_process():
                print(f"[CoreSettingsManager] Error saving settings: {e}")

    def get(self, key, default=None):
        """获取配置值，自动转换 '打开'/'关闭' 为布尔值"""
        value = self._data.get(key, default)
        # 兼容：将 '打开'/'关闭' 转换为 True/False
        if value == '打开':
            return True
        elif value == '关闭':
            return False
        return value

    def set(self, key, value):
        """设置配置值并自动保存，自动转换布尔值为 '打开'/'关闭'"""
        # 兼容：将 True/False 转换为 '打开'/'关闭'
        if isinstance(value, bool):
            value = '打开' if value else '关闭'
        if self._data.get(key) != value:
            self._data[key] = value
            self.save()
            self.settings_changed.emit(key, value)

    def get_schema(self) -> list:
        """获取配置 schema（用于 UI 生成）"""
        return self._schema_data

    def set_schema_item(self, key, field, value):
        """修改 schema 中某个字段（如 description）"""
        for item in self._schema_data:
            if item.get('key') == key:
                item[field] = value
                break

    @property
    def DEBUG(self):
        debug_setting = self._data.get('debug', None)
        if debug_setting is not None:
            return bool(debug_setting)
        return Config.DEBUG

    def set_debug(self, enabled: bool):
        self.set('debug', enabled)


SettingsManager = CoreSettingsManager
settings = CoreSettingsManager()
