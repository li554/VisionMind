"""
快捷键管理器
- 维护所有快捷键的默认绑定和自定义绑定
- 自定义绑定存储在全局配置 core.json 的 shortcuts 条目中（经 settings 读写）
- 提供查询、修改、重置接口
"""
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QKeySequence, QShortcut


@dataclass
class ShortcutInfo:
    """快捷键信息"""
    action_id: str       # 动作ID，如 "tool_sam"
    description: str     # 描述，如 "AI 点工具"
    category: str        # 分类，如 "工具切换"、"图片操作"
    default_key: str     # 默认快捷键，如 "Q"
    custom_key: str = "" # 自定义快捷键（空表示使用默认）

    @property
    def key(self) -> str:
        """实际生效的快捷键"""
        return self.custom_key if self.custom_key else self.default_key


class ShortcutManager(QObject):
    """全局快捷键管理器"""
    shortcut_changed = Signal(str, str)  # (action_id, new_key)

    _instance: Optional['ShortcutManager'] = None

    # 所有默认快捷键定义：(action_id, description, category, default_key)
    DEFAULTS: List[Tuple[str, str, str, str]] = [
        # 工具切换
        ("tool_sam",        "AI 点工具",        "工具切换", "Q"),
        ("tool_ai_rect",    "AI 矩形框工具",    "工具切换", "W"),
        ("tool_rect",       "矩形标注",         "工具切换", "R"),
        ("tool_poly",       "多边形标注",       "工具切换", "P"),
        ("tool_obb",        "旋转检测",         "工具切换", "X"),
        ("tool_edit",       "编辑模式",         "工具切换", "E"),
        ("tool_roi",        "ROI 区域选择",     "工具切换", "S"),

        # 数字键切换
        ("num_sam",         "AI 点工具(数字)",  "工具切换", "1"),
        ("num_ai_rect",     "AI 矩形框(数字)",  "工具切换", "2"),
        ("num_rect",        "矩形标注(数字)",   "工具切换", "3"),
        ("num_poly",        "多边形标注(数字)",  "工具切换", "4"),
        ("num_obb",         "旋转检测(数字)",   "工具切换", "5"),
        ("num_edit",        "编辑模式(数字)",   "工具切换", "6"),
        ("num_roi",         "ROI选择(数字)",    "工具切换", "7"),

        # 图片操作
        ("prev_image",      "上一张图片",       "图片操作", "A"),
        ("next_image",      "下一张图片",       "图片操作", "D"),
        ("toggle_view",     "切换视图",         "图片操作", "V"),
        ("toggle_enhance",  "图像增强",         "图片操作", "F"),
        ("reset_view",      "重置视图",         "图片操作", "Ctrl+0"),

        # AI操作
        ("example_annotate", "示例标注",        "AI操作",   "G"),
        ("text_annotate",    "文本标注",        "AI操作",   "T"),
        ("refine",           "边缘细化",        "AI操作",   "C"),

        # 标注操作
        ("select_prev",     "选中上一标注",     "标注操作", "Up"),
        ("select_next",     "选中下一标注",     "标注操作", "Down"),
        ("switch_category", "切换类别",         "标注操作", "Tab"),
        ("toggle_visibility", "切换标注可见性",  "标注操作", "Space"),
        ("toggle_all_visibility", "切换全部可见性", "标注操作", "Ctrl+Space"),
        ("delete_annotation", "删除标注",        "标注操作", "Delete"),
        ("undo",            "撤销",             "标注操作", "Ctrl+Z"),
        ("redo",            "重做",             "标注操作", "Ctrl+Y"),

        # 通用
        ("escape",          "取消/退出",        "通用",     "Escape"),
        ("confirm",         "确认/提交",        "通用",     "Return"),
        ("backspace",       "退格删除",         "通用",     "Backspace"),
    ]

    def __init__(self):
        super().__init__()
        self._shortcuts: Dict[str, ShortcutInfo] = {}
        self._load_defaults()
        self._load_custom()

    @classmethod
    def instance(cls) -> 'ShortcutManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def _load_defaults(self):
        """加载默认快捷键"""
        for action_id, desc, category, key in self.DEFAULTS:
            self._shortcuts[action_id] = ShortcutInfo(
                action_id=action_id,
                description=desc,
                category=category,
                default_key=key,
            )

    def _load_custom(self):
        """从全局配置读取自定义快捷键（走 settings，兼容测试状态隔离）"""
        try:
            from core.common.settings import settings as core_settings
            custom_map = core_settings.get("shortcuts")
            if isinstance(custom_map, dict):
                for action_id, key in custom_map.items():
                    if action_id in self._shortcuts and key:
                        self._shortcuts[action_id].custom_key = key
        except Exception as e:
            print(f"[ShortcutManager] 加载自定义快捷键失败: {e}")

    def _save_custom(self):
        """保存自定义快捷键到全局配置

        必须经 `settings.set()`：它按 schema 更新 `shortcuts` 条目的 value（原子 +
        加锁 + 保留 core.json 的 schema 元数据）。早期版本在这里直接
        `open(core.json,'w')` 整文件重写，既不原子也不加锁，且在 schema 已缺失时会把
        纯值条目写回文件 —— core.json 元数据一旦丢失，设置界面启动即崩。
        """
        custom_map = {action_id: info.custom_key
                      for action_id, info in self._shortcuts.items() if info.custom_key}
        try:
            from core.common.settings import settings as core_settings
            core_settings.set("shortcuts", custom_map)
        except Exception as e:
            print(f"[ShortcutManager] 保存自定义快捷键失败: {e}")

    def get_key(self, action_id: str) -> str:
        """获取快捷键（优先自定义，其次默认）"""
        info = self._shortcuts.get(action_id)
        return info.key if info else ""

    def set_key(self, action_id: str, key: str):
        """设置自定义快捷键"""
        if action_id not in self._shortcuts:
            return
        info = self._shortcuts[action_id]
        # 如果和默认值相同，清除自定义
        if key == info.default_key:
            info.custom_key = ""
        else:
            info.custom_key = key
        self._save_custom()
        self.shortcut_changed.emit(action_id, key)

    def reset_key(self, action_id: str):
        """重置为默认快捷键"""
        if action_id in self._shortcuts:
            self._shortcuts[action_id].custom_key = ""
            self._save_custom()
            self.shortcut_changed.emit(action_id, self._shortcuts[action_id].default_key)

    def reset_all(self):
        """重置所有快捷键为默认"""
        for info in self._shortcuts.values():
            info.custom_key = ""
        self._save_custom()

    def get_all(self) -> Dict[str, ShortcutInfo]:
        """获取所有快捷键信息"""
        return dict(self._shortcuts)

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        cats = set()
        for info in self._shortcuts.values():
            cats.add(info.category)
        return sorted(cats)

    def get_by_category(self, category: str) -> List[ShortcutInfo]:
        """按分类获取快捷键"""
        return [info for info in self._shortcuts.values() if info.category == category]

    def is_conflict(self, key: str, exclude_action: str = "") -> Optional[str]:
        """检查快捷键是否冲突，返回冲突的 action_id"""
        for action_id, info in self._shortcuts.items():
            if action_id == exclude_action:
                continue
            if info.key == key:
                return action_id
        return None

    def apply_shortcuts(self, target_widget, action_slots: Dict[str, Callable]) -> List[QShortcut]:
        """
        在目标控件上应用快捷键绑定

        Args:
            target_widget: 目标控件（如 annotation_interface）
            action_slots: {action_id: callable} 映射

        Returns:
            已创建的 QShortcut 列表
        """
        created = []
        for action_id, slot in action_slots.items():
            key = self.get_key(action_id)
            if not key:
                continue
            sc = QShortcut(QKeySequence(key), target_widget)
            sc.activated.connect(slot)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            created.append(sc)
        return created
