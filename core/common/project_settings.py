import os
import json
import copy
from PySide6.QtCore import QObject, Signal
from .config import Config
from .settings import settings


# ====== 自动标注规则 schema ======
# 写入 project_info.json 的 rules_schema 字段：
#   1 = 旧版「相对规则」：area/width/height/aspect 是相对参考示例平均值的偏差比例，
#       center_range 是相对示例中心的像素偏差
#   2 = 当前「绝对规则」：最大数量/面积/宽度/高度/宽高比/灰度/置信度 全部为绝对阈值
#       （center_range 已删除）
RULES_SCHEMA_VERSION = 2

# 语义发生不兼容变化的规则字段：旧值（相对比例/示例偏差）在绝对语义下会误过滤，
# 例如 area_range=[0.5, 1.5] 会被当成 0.5~1.5 像素²，把目标全部滤掉，因此加载旧
# 项目时一次性移除（视为未启用），由用户重新按绝对值设置。
LEGACY_RELATIVE_RULE_KEYS = ("area_range", "width_range", "height_range",
                             "aspect_ratio_range", "center_range")


def migrate_legacy_rules(rules, schema_version=None):
    """把旧版（相对语义）规则升级为当前「绝对规则」版本。

    返回 (rules, migrated)：migrated=True 表示确有旧字段被移除，调用方应把结果
    与 RULES_SCHEMA_VERSION 一起写回项目配置，使文件、内存缓存、规则配置对话框
    与预测使用的规则保持一致。

    语义未变的字段（conf_threshold / max_instances / gray_range / text / mode）
    原样保留。
    """
    new_rules = dict(rules or {})
    try:
        version = int(schema_version or 1)
    except (TypeError, ValueError):
        version = 1
    if version >= RULES_SCHEMA_VERSION:
        return new_rules, False

    migrated = False
    for key in LEGACY_RELATIVE_RULE_KEYS:
        if key in new_rules:
            new_rules.pop(key, None)
            migrated = True
    return new_rules, migrated


class ProjectSettingsManager(QObject):
    """
    Manages project-specific settings (from project_info.json).
    
    预定义参数：
    - task_mode: 任务模式 ('seg', 'det', 'obb')
    - export_format: 导出格式 ('yoloseg', 'yolodet', 'yoloobb', 'coco', 'voc', 'labelme', 'mask', 'sa1b')
    - current_category: 当前选中的类别
    - roi: ROI 区域 [x, y, w, h]
    - rules: 自动标注规则配置（全部为**绝对值**规则）
        - 「启用」由字段是否存在表示（apply_rules 只处理存在的字段），
          默认只启用 conf_threshold；面积/宽高比等过滤规则默认禁用（不写入
          默认值，避免 load_project 的深合并把它们补回内存、再被 save() 写回
          project_info.json，造成对话框/文件/预测三处规则不一致）
        - text: 文本规则
        - max_instances: 最大保留数量
        - conf_threshold: 置信度阈值 (0~1)
        - area_range: 面积范围 [min, max]，单位：像素²
        - width_range: 宽度范围 [min, max]，单位：像素
        - height_range: 高度范围 [min, max]，单位：像素
        - aspect_ratio_range: 宽高比 W/H 范围 [min, max]
        - gray_range: 区域平均灰度范围 [min, max] (0~255)
        - mode: 模式 ('reuse', 'grid', 'copy_paste')
    - categories: 类别字典 {id: name}
    - defect_categories: 缺陷类别列表
    - rules_schema: 规则 schema 版本（见 RULES_SCHEMA_VERSION）
    """
    settings_changed = Signal(str, object)
    project_loaded = Signal(str)

    # 预定义的默认设置
    DEFAULT_SETTINGS = {
        "task_mode": "seg",
        "export_format": "yoloseg",
        "current_category": "defect",
        "roi": None,
        "rules": {
            "text": "",
            "conf_threshold": 0.5,
            "mode": "reuse"
        },
        "categories": {},
        "defect_categories": []
    }

    def __init__(self):
        super().__init__()
        self._project_name = None
        self._project_path = None
        self._settings = {}
        
        # Try to load current project from global settings
        project_name = settings.get("current_project_name")
        if project_name:
            self.load_project(project_name)

    def load_project(self, project_name):
        """Loads project settings from project_info.json."""
        if not project_name:
            self._project_name = None
            self._project_path = None
            self._settings = {}
            return

        project_path = os.path.join(Config.PROJECTS_DIR, project_name)
        info_path = os.path.join(project_path, "project_info.json")
        
        # 深拷贝默认设置：DEFAULT_SETTINGS 含嵌套 dict（rules 等），浅拷贝会让
        # _deep_update 就地污染类级默认值，导致上一项目的 rules/roi 泄漏到下一项目
        self._settings = copy.deepcopy(self.DEFAULT_SETTINGS)
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # 合并加载的设置到默认设置
                    self._deep_update(self._settings, loaded_settings)
                self._project_name = project_name
                self._project_path = project_path
                self.project_loaded.emit(project_name)
            except Exception as e:
                print(f"[ProjectSettings] Error loading project info: {e}")
                self._settings = copy.deepcopy(self.DEFAULT_SETTINGS)
        else:
            self._project_name = project_name
            self._project_path = project_path
            # 新项目，使用默认设置

    def _deep_update(self, base_dict, update_dict):
        """递归更新字典，保留未覆盖的默认值"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def save(self):
        """Saves current project settings to project_info.json."""
        if not self._project_name or not self._project_path:
            return

        info_path = os.path.join(self._project_path, "project_info.json")
        try:
            # 先读取现有文件，避免覆盖其他进程更新的数据
            existing_info = {}
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8-sig') as f:
                        existing_info = json.load(f)
                except:
                    pass
            
            # 合并当前设置到现有信息（保留其他字段）
            # 注意：categories 应该由 project_service.update_project_categories 管理
            # 这里不覆盖 categories，除非明确设置了 _updating_categories 标志
            saved_categories = existing_info.get('categories', {})
            
            for key, value in self._settings.items():
                if key != 'categories' or getattr(self, '_updating_categories', False):
                    existing_info[key] = value
            
            # 保留文件中的 categories（如果当前设置中没有明确更新）
            if 'categories' not in existing_info:
                existing_info['categories'] = saved_categories
            
            # 写入合并后的数据
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(existing_info, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ProjectSettings] Error saving project info: {e}")

    def get(self, key, default=None):
        """Gets a project setting value."""
        return self._settings.get(key, default)

    def set(self, key, value):
        """Sets a project setting value and saves to disk."""
        if self._settings.get(key) != value:
            self._settings[key] = value
            self.save()
            self.settings_changed.emit(key, value)

    @property
    def name(self):
        return self._project_name

    @property
    def path(self):
        return self._project_path

# Global instance
project_settings = ProjectSettingsManager()
