"""
DashboardService — 项目大厅业务逻辑层

封装项目管理和数据集导入的纯数据操作，不依赖任何 Qt UI 组件。
Interface 层调用本服务的方法后自行处理 UI 更新。

说明：项目管理的 scope="agent" action（project.*）已注册在 core.service.ProjectService，
本类不再重复注册，仅保留 dashboard 直接调用的便捷转发方法（非 action）。
"""

import os
import shutil
from core.service.project_service import ProjectService
from core.service.dataset_service import DatasetService
from core.common.action_registry import action


class DashboardService:
    def __init__(self, project_service=None, dataset_service=None):
        self.service = project_service or ProjectService()
        self.dataset_service = dataset_service or DatasetService()

    def get_all_projects(self):
        return self.service.get_all_projects()

    # 以下为便捷转发方法（供 dashboard_interface 直接调用），非 @action；
    # project.* 的 scope="agent" action 统一注册在 core.service.ProjectService。
    def delete_project(self, project_name):
        return self.service.delete_project(project_name)

    def create_project(self, name, task_type):
        return self.service.create_project(name, task_type)

    def load_project(self, project_name):
        return self.service.load_project(project_name)

    def get_image_count(self, project_name=None):
        return self.service._get_image_count(project_name)

    def import_and_normalize(self, source_dir, project_name, output_root, force=True, progress_callback=None):
        return self.dataset_service.import_and_normalize(
            source_dir,
            project_name=project_name,
            output_root=output_root,
            force=force,
            progress_callback=progress_callback
        )

    def update_project_info(self, project_name, data):
        self.service._update_project_info(project_name, data)

    # ---- Agent 可调用的数据集导入 / 图片导入（纯数据操作，scope=agent） ----

    @action("dashboard.import_images", background=True,
            description="从指定目录导入图片到已存在的项目。\n- project_name：目标项目名称（必须已存在）\n- source_dir：图片源目录路径\n- 将目录内 jpg/jpeg/png/bmp 图片复制到项目 images 目录，并更新图片计数\n- 返回 {status, message, image_count}",
            category="项目", params={"project_name": "str", "source_dir": "str"}, scope="agent")
    def import_images(self, project_name="", source_dir=""):
        """从目录复制图片到已存在项目的 images 目录（纯数据）。"""
        if not project_name or not source_dir or not os.path.isdir(source_dir):
            return {"status": "error", "message": "project_name 与有效 source_dir 必填"}
        info = self.service.load_project(project_name)
        if not info:
            return {"status": "error", "message": f"项目不存在: {project_name}"}
        images_dir = os.path.join(info.get("path", ""), "images")
        if not os.path.isdir(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        count = 0
        for fname in os.listdir(source_dir):
            if not fname.lower().endswith(exts):
                continue
            src = os.path.join(source_dir, fname)
            dst = os.path.join(images_dir, fname)
            if not os.path.exists(dst) and os.path.isfile(src):
                shutil.copy2(src, dst)
                count += 1
        self.service._update_project_info(project_name, {
            "image_count": self.service._get_image_count(project_name)
        })
        return {"status": "success", "message": f"导入 {count} 张图片到项目 {project_name}", "image_count": count}

    @action("dashboard.import_dataset", background=True,
            description="导入外部数据集(自动识别 COCO/YOLO/VOC/LabelMe 标注格式)到项目，并规范化落盘。\n- project_name：目标项目名称（不存在则自动创建）\n- source_dir：数据源目录（含图片与标注）\n- 图片复制到项目 images/，标注统一转 labelme 存 annotations/，合并类别，写 dataset_split\n- 返回 {status, message, project_name, output_dir}",
            category="项目", params={"project_name": "str", "source_dir": "str"}, scope="agent")
    def import_dataset(self, project_name="", source_dir=""):
        """导入外部数据集并规范化到项目目录（纯数据）。"""
        if not project_name or not source_dir or not os.path.isdir(source_dir):
            return {"status": "error", "message": "project_name 与有效 source_dir 必填"}
        existing = {p.get("name") for p in self.service.get_all_projects()}
        if project_name not in existing:
            created = self.service.create_project(name=project_name, task_type="seg")
            if not created or created.get("status") != "success":
                return {"status": "error", "message": f"项目创建失败: {project_name}"}
        from core.common.config import Config
        return self.dataset_service.import_and_normalize(
            source_dir,
            project_name=project_name,
            output_root=Config.PROJECTS_DIR,
            force=True
        )
