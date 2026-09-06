import os
import json
import copy
from PySide6.QtCore import QObject, Signal
from .config import Config
from .settings import settings

class ProjectSettingsManager(QObject):
    """
    Manages project-specific settings (from project_info.json).
    
    预定义参数：
    - task_mode: 任务模式 ('seg', 'det', 'obb')
    - export_format: 导出格式 ('yoloseg', 'yolodet', 'yoloobb', 'coco', 'voc', 'labelme', 'mask', 'sa1b')
    - current_category: 当前选中的类别
    - roi: ROI 区域 [x, y, w, h]
    - rules: 自动标注规则配置
        - text: 文本规则
        - max_instances: 最大实例数
        - conf_threshold: 置信度阈值
        - area_range: 面积范围 [min, max]
        - aspect_ratio_range: 宽高比范围 [min, max]
        - center_range: 中心位置范围 [min, max]
        - mode: 模式 ('reuse', 'grid', 'copy_paste')
    - categories: 类别字典 {id: name}
    - defect_categories: 缺陷类别列表
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
            "max_instances": 100,
            "conf_threshold": 0.6,
            "area_range": [0.5, 1.5],
            "aspect_ratio_range": [0.1, 10.0],
            "center_range": [50.0, 50.0],
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
