"""ProjectContext — 手动/自动服务共享的项目状态容器

手动（ManualAnnotationService）与自动（AutoAnnotationService）领域服务
通过共享的 ProjectContext 实例访问与项目相关的状态变量。
"""


class ProjectContext:
    """手动/自动服务共享的项目状态"""

    def __init__(self):
        # ====== 状态变量: 类别 & 当前工具 ======
        self.categories = {}
        self._recent_categories = []
        self.current_category = ""
        self.task_mode = "seg"
        self.project_has_own_categories = False
        self.auto_save_enabled = True
        self.export_format = "labelme"

        # ====== 状态变量: 项目信息 ======
        self.output_dir = ""
        self.outputs_root = ""
        self.current_project_name = None
        self.current_project_rules = {}
        self.prompt_library = []
        self.hard_samples = {}

        # ====== 状态变量: 非项目模式 ======
        self._is_non_project_mode = False
        self._non_project_format = None
        self._non_project_save_dir = None
