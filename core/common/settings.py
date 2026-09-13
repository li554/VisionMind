import os
import json
import sys
import threading
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


# 会话/运行时状态前缀：这些键**不落盘**。
#
# 原因（审计 D30）：`core/core.json` 是**受版本控制**的配置文件，而 `set()` 会整文件
# 重写它。把"上次看到第几张图""上次打开的目录"这类纯运行时状态写进去，有两个后果：
#   1. 每次切图都触发一次整文件落盘（切图是热路径）；
#   2. 该文件永久 dirty —— 会话状态混进受版本控制的配置，git diff 全是噪声，还可能被误提交。
# `last_trained_model_path` 是跨会话有意保留的**用户选择**，因此不在前缀列表内。
SESSION_KEY_PREFIXES = ("last_idx_", "last_opened_dir")

# 配置文件保存的串行化锁：`settings.set()` 可能从多个线程调用，而保存是
# "读改写整文件" —— 没有这把锁时并发保存会互相覆盖（审计 D30 第二层）。
_save_lock = threading.Lock()


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

    @staticmethod
    def is_session_key(key) -> bool:
        """该键是否是"只在本次运行内有效"的运行时状态（不落盘，审计 D30）。"""
        return isinstance(key, str) and key.startswith(SESSION_KEY_PREFIXES)

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
                    # 诊断：schema 元数据大面积丢失（只剩纯值条目）时给出明确提示。
                    # 这种情况下设置界面只能降级显示（曾导致启动时 KeyError: 'title'），
                    # 文件可用 `git checkout core/core.json` 恢复元数据。
                    titled = sum(1 for it in self._schema_data
                                 if isinstance(it, dict) and it.get('title'))
                    untitled = sum(1 for it in self._schema_data if isinstance(it, dict)) - titled
                    if untitled > 0 and untitled > titled:
                        print("[CoreSettingsManager] !! core.json 的 schema 元数据疑似丢失"
                              "（%d/%d 条目没有 title）：设置界面将降级显示，"
                              "建议用 git checkout core/core.json 恢复"
                              % (untitled, len(self._schema_data)))
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
        """保存配置到 core.json

        **会话态键一律跳过**（审计 D30）：它们只存在于内存里的 `_data`，因此"上次看到
        第几张图"这类状态仍然可用，但不会污染受版本控制的配置文件，也不会让每次切图都
        落一次盘。若这些键此前已经被写进文件，这里会顺带把它们清理掉。
        """
        for item in self._schema_data:
            key = item.get('key')
            if key and key in self._data:
                item['value'] = self._data[key]
        # 动态键（不在 schema 里的）也持久化，但排除会话态键。
        # 必须补上 title/type：设置界面按 `item['title']` 渲染分组项，纯 {"key","value"}
        # 条目会让 ConfigInterface 启动即 KeyError: 'title'（core.json 一旦丢了 schema
        # 元数据，每次启动都会崩在设置界面上）。
        schema_keys = {item.get('key') for item in self._schema_data}
        for key, value in self._data.items():
            if key not in schema_keys and not self.is_session_key(key):
                self._schema_data.append({
                    "key": key,
                    "title": key,
                    # dict/list 值不是可编辑控件，标记 hidden 避免渲染出错误控件
                    "type": "hidden" if isinstance(value, (dict, list)) else "string",
                    "value": value,
                })
        # 清理历史遗留：旧版本可能已把会话态键写进文件
        before = len(self._schema_data)
        self._schema_data = [item for item in self._schema_data
                             if not self.is_session_key(item.get('key'))]
        if len(self._schema_data) != before and is_main_process():
            print("[CoreSettingsManager] 已从配置文件中移除 %d 个会话态键（D30）"
                  % (before - len(self._schema_data)))
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            # 原子写 + 串行化：`core.json` 是**跨线程共享**的配置文件
            # （工作线程里也有 settings.set 调用，例如模型训练路径），
            # 原先直接 open('w') 既不原子也不互斥 —— 并发保存会互相截断/交错。
            from core.common.atomic_io import atomic_write_text
            with _save_lock:
                atomic_write_text(self.filepath, json.dumps(
                    self._schema_data, indent=2, ensure_ascii=False))
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
        """设置配置值并自动保存，自动转换布尔值为 '打开'/'关闭'

        会话态键（见 `SESSION_KEY_PREFIXES`）只更新内存，**不落盘**；想显式表达
        "这是本次运行内的状态"请直接用 `set_session()`。
        """
        # 兼容：将 True/False 转换为 '打开'/'关闭'
        if isinstance(value, bool):
            value = '打开' if value else '关闭'
        if self._data.get(key) != value:
            self._data[key] = value
            if self.is_session_key(key):
                # 会话态：不落盘（切图热路径每次都会调用到这里）
                self.settings_changed.emit(key, value)
                return
            self.save()
            self.settings_changed.emit(key, value)

    def set_session(self, key, value):
        """写入**只在本次运行内有效**的状态（不落盘）。

        用于"上次看到第几张图/上次打开的目录"这类每次切图都会变的状态：既保留
        跨切图的行为，又不改写受版本控制的 `core/core.json`（审计 D30）。
        """
        if isinstance(value, bool):
            value = '打开' if value else '关闭'
        if self._data.get(key) != value:
            self._data[key] = value
            self.settings_changed.emit(key, value)

    def get_session(self, key, default=None):
        """读取会话态（内存优先，兼容旧版本已落盘的值）。"""
        return self.get(key, default)

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
