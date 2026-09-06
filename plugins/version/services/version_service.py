"""
VersionService — 版本管理业务逻辑层

封装版本管理的纯数据操作，不依赖任何 Qt UI 组件。
Interface 层调用本服务的方法后自行处理 UI 更新。
"""

import json
import os
import time
from typing import List, Dict, Any
from core.common.action_registry import action
from core.service.project_service import ProjectService
from core.service.common_service import ExampleSelectionManager


class VersionService:
    def __init__(self, project_service=None):
        self.service = project_service or ProjectService()
        self.example_selector = ExampleSelectionManager()

    @property
    def base_dir(self):
        return self.service.base_dir

    @property
    def current_project(self):
        return self.service.current_project

    def get_all_projects(self):
        return self.service.get_all_projects()

    def load_project(self, project_name):
        return self.service.load_project(project_name)

    @action("version.list_versions", read_only=True,
             description="列出指定项目的所有已导出版本。\n- project_name：项目名称\n- 返回值：版本列表，包含名称、日期和路径",
             category="版本管理", params={"project_name": "str"}, scope="agent")
    def scan_versions(self, project_name):
        project_path = os.path.join(self.service.base_dir, project_name)
        versions_dir = os.path.join(project_path, "versions")
        versions = []
        if os.path.exists(versions_dir):
            for v in sorted(os.listdir(versions_dir), reverse=True):
                v_path = os.path.join(versions_dir, v)
                if not os.path.isdir(v_path):
                    continue
                try:
                    mtime = os.path.getmtime(v_path)
                    date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                except:
                    date_str = "未知时间"
                versions.append({"name": v, "date": date_str, "path": v_path})
        return versions

    def get_version_path(self, project_name, version_name):
        return os.path.join(self.service.base_dir, project_name, "versions", version_name)

    @action("version.delete_version",
             description="删除指定项目的指定版本文件夹。\n- project_name：项目名称\n- version_name：版本名称（如 v1、v2）\n- 操作不可恢复",
             category="版本管理", params={"project_name": "str", "version_name": "str"}, scope="agent")
    def delete_version_folder(self, project_name, version_name):
        import shutil
        path = self.get_version_path(project_name, version_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

    def get_all_projects_action(self):
        """获取所有项目列表（通过 project_service 实现）"""
        return self.get_all_projects()

    def export_version_v2(self, config, progress_callback=None):
        return self.service.export_version_v2(config, progress_callback=progress_callback)

    @action("version.export", background=True,
            description="将指定项目的数据集导出为指定格式并生成版本。\n"
                        "- project_name：目标项目名称（必须已存在且有标注）\n"
                        "- export_format：导出格式，可选 yolo / yoloseg / yoloobb / coco / voc / labelme\n"
                        "- export_task_type：导出任务类型，可选 det（检测）/ seg（分割）/ obb（旋转）；留空则由导出格式自动推断\n"
                        "- 导出结果写入 <项目>/versions/vN，返回版本名与路径。默认导出包含未标注图片（export_unannotated=True）\n"
                        "- 导出后用 version.get_export_content 校验产物格式是否符合配置，用 version.delete_version 清理临时版本",
            category="版本管理", params={"project_name": "str", "export_format": "str", "export_task_type": "str"}, scope="agent")
    def export_version(self, project_name="", export_format="yolo", export_task_type=""):
        if not project_name:
            return {"status": "error", "message": "缺少 project_name"}
        valid = {"yolo", "yoloseg", "yoloobb", "coco", "voc", "labelme"}
        if export_format not in valid:
            return {"status": "error", "message": f"不支持的导出格式: {export_format}，可选 {sorted(valid)}"}
        config = {
            "project_name": project_name,
            "use_manual_split": False,
            "ratios": [0.7, 0.2, 0.1],
            "preprocessing": {},
            "augmentations": {},
            "export_format": export_format,
            "export_task_type": export_task_type,
            "export_unannotated": True,
            "roi_bbox": None,
        }
        res = self.export_version_v2(config)
        if res.get("status") != "success":
            return res
        res["export_format"] = export_format
        res["project_name"] = project_name
        return res

    @action("version.get_export_content", read_only=True,
            description="读取导出版本的产物结构，用于校验导出格式是否符合配置。\n"
                        "- project_name：项目名\n- version_name：版本名（如 v1，用 version.list_versions 或 version.export 获取）\n"
                        "- 返回每个划分（train/val/test）下的图片清单，以及标注/配置文件：相对路径、扩展名、内容首行样本\n"
                        "- 用途核对：yolo→labels/*.txt 每行形如『cls cx cy w h』；coco→coco_annotations/*.json 含 images/annotations/categories；voc→voc_annotations/*.xml 含 annotation/object/bndbox；labelme→annotations/*.json 含 shapes",
            category="版本管理", params={"project_name": "str", "version_name": "str"}, scope="agent")
    def get_export_content(self, project_name, version_name):
        vpath = self.get_version_path(project_name, version_name)
        if not os.path.isdir(vpath):
            return {"status": "error", "message": f"版本 {version_name} 不存在"}
        splits = {}
        for split in ["train", "val", "test"]:
            sp = os.path.join(vpath, split)
            if not os.path.isdir(sp):
                continue
            images = []
            img_dir = os.path.join(sp, "images")
            if os.path.isdir(img_dir):
                images = sorted(f for f in os.listdir(img_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
            files = []
            images_rel = (os.path.join(split, "images") + os.sep)
            for root, _dirs, fnames in os.walk(sp):
                rel_root = os.path.relpath(root, vpath)
                if rel_root == "images" or rel_root.startswith(images_rel):
                    continue  # 跳过 images 目录下的图片文件
                for fn in sorted(fnames):
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, vpath).replace("\\", "/")
                    sample = ""
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                            sample = f.readline().strip()[:200]
                    except Exception:
                        pass
                    files.append({"path": rel, "ext": os.path.splitext(fn)[1].lstrip("."), "sample": sample})
            splits[split] = {"images": images, "files": files}
        return {"status": "success", "project_name": project_name, "version": version_name, "splits": splits}

    def version_exists(self, project_name, version_name):
        return os.path.exists(self.get_version_path(project_name, version_name))

    def get_support_sets_dir(self, project_name=None):
        return self.service.get_project_support_sets_dir(project_name)

    def random_split_dataset(self, project_name, split_ratios, progress_callback=None):
        return self.service.random_split_dataset(project_name, split_ratios, progress_callback=progress_callback)

    @action("resource.list_example_sets", read_only=True,
             description="列出当前项目可用的示例数据集。\n- project_name：项目名称（可选，默认当前项目）\n- 返回示例数据集列表，包含名称、类别等信息\n- AI 可根据此结果决定选择哪些示例用于版本生成",
             category="资源管理", params={"project_name": "str"}, scope="agent")
    def list_example_sets(self, project_name=None) -> List[Dict[str, Any]]:
        project_name = project_name or (self.current_project.get('name') if self.current_project else None)
        support_sets_dir = self.get_support_sets_dir(project_name)
        if not support_sets_dir and project_name:
            support_sets_dir = os.path.join(self.service.base_dir, project_name, "support_sets")

        if not support_sets_dir or not os.path.exists(support_sets_dir):
            return []

        results = []
        for entry in sorted(os.listdir(support_sets_dir)):
            entry_path = os.path.join(support_sets_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            info = {"name": entry, "path": entry_path}
            info_path = os.path.join(entry_path, "info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8-sig') as f:
                        meta = json.load(f)
                        info["category"] = meta.get('category', '')
                        info["description"] = meta.get('description', '')
                except Exception:
                    pass
            results.append(info)
        return results

    @action("resource.set_example_sets",
             description="选择用于版本生成的示例数据集。\n- example_names: 要使用的示例名称列表（可用 list_example_sets 获取）\n- 选择后会影响后续版本生成时的 Copy-Paste 示例来源",
             category="资源管理", params={"example_names": "list"}, scope="agent")
    def set_example_sets(self, example_names: list = None):
        """选择示例数据集"""
        if not example_names:
            self.example_selector.clear_selected_examples()
            return {"status": "success", "message": "已清空示例选择"}
        example_sets = self.list_example_sets()
        available_names = ExampleSelectionManager.get_available_names(example_sets)
        result = self.example_selector.set_selected_examples(example_names, available_names)
        return result

    @action("resource.get_selected_example_sets", read_only=True,
             description="获取当前已选择的示例数据集名称列表。\n- 无参数\n- 返回值：当前选中的示例名称列表",
             category="资源管理", scope="agent")
    def get_selected_example_sets(self):
        """获取当前选中的示例数据集名称"""
        return self.example_selector.get_selected_examples()

    @action("resource.clear_example_sets_selection",
             description="清空当前示例数据集选择。\n- 取消所有已选示例\n- 之后版本生成时将不使用 Copy-Paste 示例",
             category="资源管理", scope="agent")
    def clear_example_sets_selection(self):
        """清空示例数据集选择"""
        self.example_selector.clear_selected_examples()
        return {"status": "success", "message": "已清空示例选择"}


