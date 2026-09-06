import json
import os
from typing import Any, Dict, Optional


class PluginConfig:
    """
    插件配置基类，从 config.json 加载配置（包含schema定义和值）

    config.json 格式:
    [
        {"key": "setting_key", "title": "设置名称", "type": "combo|int|bool|string",
         "options": [...], "min": 0, "max": 100, "default": "默认值", "value": "当前值"}
    ]

    用法:
        config = PluginConfig("/path/to/plugins/annotation")
        config.get("interactive_model", "mobilesam_onnx")
        config.set("interactive_model", "sam3")
        config.save()
    """

    def __init__(self, plugin_dir: str):
        """
        Args:
            plugin_dir: 插件根目录路径
        """
        self.plugin_dir = plugin_dir
        self.config_path = os.path.join(plugin_dir, "config.json")
        self._data: Dict[str, Any] = {}
        self._schema_data: list = []
        self.load()

    def load(self):
        """从 config.json 加载配置（包含schema和值）"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._schema_data = json.load(f)
                # 从 schema 中提取值
                self._data = {}
                for item in self._schema_data:
                    key = item.get('key')
                    if key:
                        self._data[key] = item.get('value', item.get('default'))
            except (json.JSONDecodeError, IOError) as e:
                print(f"[PluginConfig] Failed to load {self.config_path}: {e}")
                self._data = {}
                self._schema_data = []
        else:
            self._data = {}
            self._schema_data = []

    def save(self):
        """保存配置到 config.json（更新value字段）"""
        # 更新 schema 中的值
        for item in self._schema_data:
            key = item.get('key')
            if key and key in self._data:
                item['value'] = self._data[key]
        # 保存到 config.json
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._schema_data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """设置配置值（需调用 save() 持久化）"""
        self._data[key] = value

    def update(self, data: Dict[str, Any]):
        """批量更新配置"""
        self._data.update(data)

    def to_dict(self) -> Dict[str, Any]:
        """返回配置的完整字典"""
        return dict(self._data)

    def get_schema(self) -> list:
        """获取配置schema（用于UI生成）"""
        return self._schema_data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
