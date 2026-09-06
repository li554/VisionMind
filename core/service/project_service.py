import os
import json
import shutil
import time
from datetime import datetime
from core.common.settings import settings
from core.common.config import Config
from core.common.project_settings import project_settings
from core.common.image_utils import imread_unicode, imwrite_unicode
from core.backend.formats import load_annotations as formats_load_annotations
from core.common.action_registry import action

# 任务类型：内部统一使用短名 det/seg/obb（存盘字段 task_type 与内存 task_mode 同值）。
# 展示名用于配置界面等场景的中文显示。
TASK_TYPE_DISPLAY = {
    "det": "目标检测",
    "seg": "目标分割",
    "obb": "旋转目标检测",
}

# 每个任务类型对应的默认导出格式
TASK_TYPE_DEFAULT_EXPORT = {
    "det": "yolodet",
    "seg": "yoloseg",
    "obb": "yoloobb",
}


def normalize_task_type(task_type):
    """统一任务类型为内部短名 det/seg/obb（兼容旧长名 detection/segmentation）。"""
    t = (task_type or "").strip().lower()
    return {"detection": "det", "segmentation": "seg"}.get(
        t, t if t in ("det", "seg", "obb") else "seg")


class ProjectService:
    def __init__(self):
        self.base_dir = Config.PROJECTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self.current_project = None
        self._event_bus = None  # 事件总线，由 core/main.py 注入；用于发布项目数据变更事件

    def set_event_bus(self, bus):
        """注入事件总线，供发布 dashboard:projects_changed 等事件。"""
        self._event_bus = bus

    def _publish_projects_changed(self, project_action, project_name):
        """发布项目列表变更事件，驱动各界面自动刷新。"""
        if self._event_bus is not None:
            from core.event_bus import StateType
            self._event_bus.publish_state(StateType.PROJECTS_CHANGED, {
                "action": project_action,
                "project_name": project_name,
            })

    def get_all_projects(self):
        """获取所有项目列表"""
        projects = []
        if not os.path.exists(self.base_dir):
            return projects
            
        for name in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, name)
            if not os.path.isdir(path):
                continue
                
            info_path = os.path.join(path, "project_info.json")
            if not os.path.exists(info_path):
                # Auto-create if it's likely a project folder (contains annotations/ or images/)
                if any(os.path.exists(os.path.join(path, d)) for d in ['annotations', 'images']):
                    # Use a non-state-changing method to create info
                    self._reconstruct_project_info(name)
            
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8-sig') as f:
                        info = json.load(f)
                        info['name'] = name
                        projects.append(info)
                except:
                    if settings.DEBUG:
                        raise
                    pass
        return projects

    def get_default_rules(self):
        """获取默认的自动标注规则"""
        return {
            "max_instances": 100,
            "conf_threshold": 0.5,
            "area_range": [0.5, 1.0],          # 面积偏差范围 (下限，上限)
            "aspect_ratio_range": [0.5, 0.5],   # 宽高比偏差范围 (下限，上限)
            "center_range": [50.0, 50.0],       # 中心点偏差范围 (x, y)，单位像素
            "text": "",
            "mode": "reuse"                     # 拼接模式：reuse, grid, copy_paste
        }

    def get_project_support_sets_dir(self, project_name=None):
        """获取项目的 support_sets 目录路径（仅在目录已存在时返回，不自动创建）"""
        if project_name is None:
            project_name = settings.get("current_project_name")
        if not project_name:
            return None
        project_path = os.path.join(self.base_dir, project_name)
        support_sets_dir = os.path.join(project_path, "support_sets")
        # 仅在目录已存在时返回，不自动创建
        if os.path.exists(support_sets_dir):
            return support_sets_dir
        return None

    def _reconstruct_project_info(self, name):
        """Reconstruct project_info.json without changing self.current_project"""
        project_path = os.path.join(self.base_dir, name)
        # 统一使用annotations/目录，不再区分det/seg/obb
        task_type = "det"
        ann_dir = os.path.join(project_path, "annotations")
        if os.path.exists(ann_dir):
            # 根据标注内容判断任务类型
            try:
                from core.backend.formats import load_annotations
                for f in os.listdir(ann_dir):
                    if f.endswith('.json'):
                        result = load_annotations(os.path.join(ann_dir, f))
                        for ann in result.get("annotations", []):
                            if ann.get("polygons") and ann.get("shape_type") == "rotation":
                                task_type = "obb"
                                break
                            elif ann.get("polygons") and ann.get("shape_type") == "polygon":
                                task_type = "seg"
                                break
                        if task_type != "det":
                            break
            except Exception:
                pass
        
        categories = {}
        # Try multiple possible locations for classes.txt
        possible_classes = [
            os.path.join(project_path, "annotations", "classes.txt"),
            os.path.join(project_path, "classes.txt")
        ]
        
        for classes_txt in possible_classes:
            if os.path.exists(classes_txt):
                with open(classes_txt, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        name_str = line.strip()
                        if name_str:
                            categories[str(i)] = name_str
                if categories: break

        # 根据 task_type 确定 task_mode 和默认导出格式（统一短名）
        task_type = normalize_task_type(task_type)
        task_mode = task_type
        default_export_format = TASK_TYPE_DEFAULT_EXPORT.get(task_type, "yoloseg")

        info = {
            "task_type": task_type,
            "task_mode": task_mode,
            "export_format": default_export_format,
            "current_category": "defect",
            "categories": categories,
            "defect_categories": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": self._get_image_count(name),
            "version_count": 0,
            "roi": None,
            "rules": self.get_default_rules()
        }

        info_path = os.path.join(project_path, "project_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        return info

    @action("project.create_project_data",
             description="创建一个新的标注项目。\n- name：项目名称\n- task_type：任务类型，可选 seg(分割)/det(检测)/obb(旋转框)\n- 创建后项目立即可用",
             category="项目", params={"name": "str", "task_type": "str"}, scope="agent")
    def create_project(self, name, task_type="det", categories=None):
        """创建一个新项目"""
        project_path = os.path.join(self.base_dir, name)
        if os.path.exists(project_path):
            return {"status": "error", "message": "项目已存在"}

        try:
            # 创建目录结构 (统一annotations/目录存储labelme格式)
            os.makedirs(os.path.join(project_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "annotations"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "versions"), exist_ok=True)
            # support_sets 在添加示例时才创建

            # 统一任务类型为内部短名 det/seg/obb
            task_type = normalize_task_type(task_type)
            task_mode = task_type
            default_export_format = TASK_TYPE_DEFAULT_EXPORT.get(task_type, "yoloseg")

            # 创建项目信息文件 - 包含所有预定义参数
            info = {
                "task_type": task_type,
                "task_mode": task_mode,
                "export_format": default_export_format,
                "current_category": "defect",
                "categories": categories or {},
                "defect_categories": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_count": 0,
                "version_count": 0,
                "roi": None,
                "rules": self.get_default_rules()
            }

            with open(os.path.join(project_path, "project_info.json"), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)

            self._publish_projects_changed("created", name)
            return {"status": "success", "project": info}
        except Exception as e:
            if settings.DEBUG:
                raise
            return {"status": "error", "message": str(e)}

    @action("project.load_project_data",
             description="加载指定项目的配置信息。\n- project_name：项目名称\n- 返回项目信息字典",
             category="项目", params={"project_name": "str"}, scope="agent")
    def load_project(self, project_name):
        """加载项目并设为当前活跃项目，同时刷新统计信息"""
        name = project_name
        project_path = os.path.join(self.base_dir, name)
        info_path = os.path.join(project_path, "project_info.json")
        
        if not os.path.exists(info_path):
            self._reconstruct_project_info(name)
            
        img_count = self._get_image_count(name)
        
        categories = {}
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8-sig') as f:
                try:
                    existing_info = json.load(f)
                    categories = existing_info.get('categories', {})
                    print(f"[DEBUG] load_project: read categories from file = {categories}")
                except:
                    if settings.DEBUG:
                        raise
                    pass
        
        v_count = 0
        versions_dir = os.path.join(project_path, "versions")
        if os.path.exists(versions_dir):
            v_count = len([d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))])
            
        self._update_project_info(name, {
            "image_count": img_count,
            "version_count": v_count
        })
        
        try:
            with open(info_path, 'r', encoding='utf-8-sig') as f:
                info = json.load(f)
                info['name'] = name
                info['path'] = project_path
                self.current_project = info
                
                settings.set("current_project_name", name)
                settings.set("output_dir", project_path) 
                project_settings.load_project(name)

                # 同步操作记录器到当前项目目录（持久化操作历史）
                try:
                    from core.agent.operation_logger import OperationLogger
                    OperationLogger.instance().set_project_dir(project_settings.path or project_path)
                except Exception as e:
                    print(f"[DEBUG] OperationLogger 关联项目失败: {e}")

                print(f"[DEBUG] load_project: returning info with categories = {info.get('categories')}")
                return info
        except:
            if settings.DEBUG:
                raise
            return self._reconstruct_project_info(name)


    def create_project_info(self, name, task_type="det", categories=None):
        """Helper to create project_info.json if missing"""
        project_path = os.path.join(self.base_dir, name)
        os.makedirs(project_path, exist_ok=True)
        info = {
            "task_type": normalize_task_type(task_type),
            "categories": categories or {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": 0,
            "version_count": 0,
            "rules": self.get_default_rules()
        }
        with open(os.path.join(project_path, "project_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        return info

    def import_images(self, project_name, image_paths):
        """导入图片到项目的 images 文件夹"""
        project_path = os.path.join(self.base_dir, project_name)
        images_dir = os.path.join(project_path, "images")
        
        count = 0
        for path in image_paths:
            if os.path.exists(path):
                ext = os.path.splitext(path)[1]
                target_path = os.path.join(images_dir, f"img_{datetime.now().timestamp()}_{count}{ext}")
                shutil.copy2(path, target_path)
                count += 1
        
        # 更新图片数量
        self._update_project_info(project_name, {"image_count": self._get_image_count(project_name)})
        return count

    @action("project.get_image_count", read_only=True,
             description="获取指定项目的图片数量统计。\n- project_name：项目名称（可选，不传时使用当前项目）\n- 返回值：图片总数",
             category="项目", params={"project_name": "str(可选，默认当前项目)"}, scope="agent")
    def _get_image_count(self, project_name=None):
        if project_name is None:
            project_name = settings.get("current_project_name")
        if not project_name:
            return 0
        project_path = os.path.join(self.base_dir, project_name)
        count = 0
        
        # Check images folder
        images_dir = os.path.join(project_path, "images")
        if os.path.exists(images_dir):
            count += len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            
        # Check normalized folders (det and seg)
        for task in ['det', 'seg']:
            for split in ['train', 'val', 'test']:
                img_dir = os.path.join(project_path, task, split, "images")
                if os.path.exists(img_dir):
                    # We only count once even if image exists in both det and seg
                    # But for simplicity, if we have det/ and seg/, we might be double counting
                    # Let's just count unique basenames across all tasks/splits
                    pass
        
        # Better approach: count unique image filenames in the project
        image_files = set()
        for root, dirs, files in os.walk(project_path):
            # Skip versions folder to avoid counting exported images
            if "versions" in root:
                continue
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.add(f)
        
        return len(image_files)

    @action("project.delete_project_data",
             description="删除指定的项目及其所有数据。\n- project_name：要删除的项目名称\n- 谨慎使用，删除不可恢复",
             category="项目", params={"project_name": "str"}, scope="agent")
    def delete_project(self, project_name):
        """删除项目及其所有文件"""
        name = project_name
        project_path = os.path.join(self.base_dir, name)
        if os.path.exists(project_path):
            try:
                shutil.rmtree(project_path)
                # 如果删除的是当前项目，清除全局设置
                if settings.get("current_project_name") == name:
                    settings.set("current_project_name", "")
                    settings.set("output_dir", "")
                self._publish_projects_changed("deleted", name)
                return True
            except:
                if settings.DEBUG:
                    raise
                return False
        return False

    def _write_json_safely(self, file_path, data):
        """Safely write JSON to a file with retries and atomic replacement."""
        temp_path = file_path + ".tmp"
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # 1. Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 2. Write to temp file
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                # 3. Atomic replacement
                if os.path.exists(file_path):
                    try:
                        os.replace(temp_path, file_path)
                    except OSError:
                        if settings.DEBUG:
                            raise
                        # Fallback for some Windows edge cases
                        os.remove(file_path)
                        os.rename(temp_path, file_path)
                else:
                    os.rename(temp_path, file_path)
                return True
            except Exception as e:
                if settings.DEBUG:
                    raise
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error saving JSON to {file_path} after {max_attempts} attempts: {e}")
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
                return False

    def update_project_rules(self, project_name, rules):
        """更新项目的自动标注规则"""
        return self._update_project_info(project_name, {"rules": rules})

    def update_project_field(self, project_name, field, value):
        """更新项目配置中的指定字段"""
        return self._update_project_info(project_name, {field: value})

    @action("project.update_project_info",
             description="更新项目的元数据信息。\n- project_name：项目名称\n- data：要更新的字段字典，支持 rules、categories、image_count、version_count 等\n- 不会创建新项目，只更新已有项目的配置",
             category="项目", params={"project_name": "str", "data": "dict"}, scope="ui")
    def _update_project_info(self, project_name, data):
        updates = data
        project_path = os.path.join(self.base_dir, project_name)
        info_path = os.path.join(project_path, "project_info.json")
        info = {}
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8-sig') as f:
                    info = json.load(f)
            except:
                if settings.DEBUG:
                    raise
                print(f"Warning: Reconstructing corrupted project_info.json for {project_name}")
                info = {
                    "task_type": "det",
                    "categories": {},
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_count": 0,
                    "version_count": 0
                }
        
        print(f"[DEBUG] _update_project_info: before update, info.get('categories') = {info.get('categories')}")
        print(f"[DEBUG] _update_project_info: updates = {updates}")
        info.update(updates)
        print(f"[DEBUG] _update_project_info: after update, info.get('categories') = {info.get('categories')}")
        self._write_json_safely(info_path, info)
        
        if project_settings.name == project_name:
            for k, v in updates.items():
                project_settings._settings[k] = v
            project_settings.settings_changed.emit("all", project_settings._settings)

    def update_project_categories(self, project_name, categories):
        """
        Updates categories in project_info.json.
        Categories can be {name: color_hex}, {name: QColor}, or {id: name}.
        We store {id: name} in project_info.json for consistency with classes.txt.
        """
        print(f"[DEBUG] update_project_categories: project_name={project_name}, categories={list(categories.keys()) if categories else 'None'}")
        
        if not categories:
            print(f"[DEBUG] update_project_categories: SKIPPED - categories is empty")
            return

        if all(str(k).isdigit() for k in categories.keys()):
            id_to_name = {str(k): v for k, v in categories.items()}
        else:
            id_to_name = {str(i): name for i, name in enumerate(categories.keys())}

        print(f"[DEBUG] update_project_categories: saving id_to_name = {id_to_name}")
        self._update_project_info(project_name, {"categories": id_to_name})

    def _load_image_annotations(self, image_path, output_dir=None, fmt="mask"):
        """
        直接使用 core.backend.formats 加载标注数据，
        不依赖 AnnotationService。

        Args:
            image_path: 图像文件路径
            output_dir: 标注所在目录
            fmt: 格式类型 ("labelme", "coco", "yolo", "yoloseg", "yoloobb", "voc", "mask", "all")

        Returns:
            dict: {
                'status': 'success' | 'error',
                'annotations': List[Dict],
                'image_info': Dict,
                'categories': List[Dict],
                'format': str
            }
        """
        try:
            if not output_dir:
                output_dir = settings.get("output_dir")

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            fmt_config = settings.get("annotation_formats", {})

            formats_to_try = ["labelme", "coco", "yolo", "voc", "mask"] if fmt == "all" else [fmt]

            for try_fmt in formats_to_try:
                config = fmt_config.get(try_fmt)
                if not config:
                    continue
                subdir_path = os.path.join(output_dir, config["subdir"])
                filename = config.get("filename", f"{base_name}{config['ext']}")

                file_path = None
                if os.path.exists(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    if not os.path.exists(file_path):
                        file_path = None

                if file_path is None:
                    file_path = os.path.join(output_dir, filename)
                    if not os.path.exists(file_path):
                        continue

                load_kwargs = {"image_path": image_path} if try_fmt in ["coco", "yolo", "yoloseg", "yoloobb"] else {}
                result = formats_load_annotations(file_path, **load_kwargs)
                if result and result.get("annotations"):
                    return {
                        'status': 'success',
                        'annotations': result["annotations"],
                        'image_info': result.get("image_info", {}),
                        'categories': result.get("categories", []),
                        'format': try_fmt
                    }

            return {'status': 'success', 'annotations': [], 'image_info': {}, 'categories': [], 'format': 'unknown'}
        except Exception as e:
            if settings.DEBUG:
                raise
            return {'status': 'error', 'message': str(e), 'annotations': [], 'image_info': {}, 'categories': []}

    def export_version_v2(self, config, progress_callback=None):
        """
        v2 导出功能：支持手动/随机划分、预处理和数据增强参数设置
        """
        import cv2
        import numpy as np
        import random
        
        project_name = config['project_name']
        project_path = os.path.join(self.base_dir, project_name)
        
        if progress_callback: progress_callback(5) # Start progress
        
        # 1. 确定版本号
        versions_dir = os.path.join(project_path, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        existing_versions = [d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))]
        v_idx = len(existing_versions) + 1
        version_name = f"v{v_idx}"
        version_path = os.path.join(versions_dir, version_name)
        os.makedirs(version_path, exist_ok=True)
        
        # 2. 划分逻辑
        splits = {"train": [], "val": [], "test": []}
        
        # 获取图片源目录
        img_src_root = os.path.join(project_path, "images")
        
        if not os.path.exists(img_src_root):
            return {"status": "error", "message": "未找到图片目录"}
            
        all_images = [f for f in os.listdir(img_src_root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if config['use_manual_split']:
            # 从 dataset_split.json 加载划分
            split_json_path = os.path.join(project_path, "dataset_split.json")
            manual_splits = {}
            if os.path.exists(split_json_path):
                try:
                    with open(split_json_path, 'r', encoding='utf-8-sig') as f:
                        manual_splits = json.load(f)
                except: pass
            
            # 建立文件名到划分的映射 (更高效)
            name_to_split = {os.path.basename(k): v for k, v in manual_splits.items()}
            
            for img_name in all_images:
                found_split = name_to_split.get(img_name, "train")
                splits[found_split].append(img_name)
        else:
            # 随机划分
            random.shuffle(all_images)
            total = len(all_images)
            ratios = config['ratios']
            tr_end = int(total * ratios[0])
            val_end = tr_end + int(total * ratios[1])
            splits["train"] = all_images[:tr_end]
            splits["val"] = all_images[tr_end:val_end]
            splits["test"] = all_images[val_end:]

        # 3. 处理每个划分并导出
        preprocessing = config['preprocessing']
        augmentations = config['augmentations']

        # 获取项目任务类型和类别
        # 优先使用用户显式选择的导出任务类型
        export_format = config.get('export_format', '')
        export_task_type = config.get('export_task_type', '')  # det/seg/obb
        if export_task_type in ('det', 'seg', 'obb'):
            task_type = export_task_type
        else:
            # 兼容：如果未指定export_task_type，从导出格式推断
            fmt_to_task = {
                'yolo': 'det', 'yoloseg': 'seg', 'yoloobb': 'obb',
                'coco': 'seg', 'voc': 'det', 'labelme': None
            }
            task_type = fmt_to_task.get(export_format)

        info_path = os.path.join(project_path, "project_info.json")
        class_names = []
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8-sig') as f:
                    info = json.load(f)
                    # 如果导出格式未指定task_type（如labelme/coco），从项目信息读取
                    if task_type is None:
                        raw_task_type = info.get('task_type', 'detection')
                        if raw_task_type in ['segmentation', 'seg']:
                            task_type = 'seg'
                        elif raw_task_type == 'obb':
                            task_type = 'obb'
                        else:
                            task_type = 'det'

                    categories = info.get('categories', {})
                    if isinstance(categories, list):
                        class_names = categories
                    elif isinstance(categories, dict):
                        try:
                            ids = sorted([int(k) for k in categories.keys()])
                            class_names = [categories[str(i)] for i in ids]
                        except:
                            if settings.DEBUG:
                                raise
                            class_names = list(categories.keys())
            except Exception as e:
                if settings.DEBUG:
                    raise
                print(f"[Export] Error reading project_info: {e}")
                if task_type is None:
                    task_type = 'det'

        if not class_names: class_names = ["target"]
        
        # 获取 ROI 设置
        roi_bbox = config.get('roi_bbox')  # [x, y, w, h]
        use_roi = roi_bbox is not None and len(roi_bbox) == 4 and roi_bbox[2] > 0 and roi_bbox[3] > 0

        # --- 2.5 提取 Copy-Paste 对象池 (如果启用) ---
        object_pool = []
        example_pool = []
        
        # A. 提取本地项目对象池
        if augmentations.get('copypaste'):
            max_pool_size = 50 # 限制大小以节省内存
            # 随机挑选一些图片来提取对象
            pool_images = random.sample(all_images, min(len(all_images), 30))
            for p_img_name in pool_images:
                if len(object_pool) >= max_pool_size: break
                
                p_img_path = os.path.join(img_src_root, p_img_name)
                p_img = imread_unicode(p_img_path)
                if p_img is None: continue
                
                # 统一从annotations/目录读取labelme格式标注
                ann_dir = os.path.join(project_path, "annotations")
                p_ann_res = self._load_image_annotations(p_img_path, output_dir=ann_dir, fmt="labelme")

                if p_ann_res['status'] == 'success' and p_ann_res['annotations']:
                    for ann in p_ann_res['annotations']:
                        # 优先使用多边形（分割/旋转标注）
                        if 'polygons' in ann and ann['polygons']:
                            poly = np.array(ann['polygons'][0], dtype=np.int32)
                            x, y, w, h = cv2.boundingRect(poly)
                            if w < 10 or h < 10: continue

                            crop = p_img[y:y+h, x:x+w].copy()
                            mask = np.zeros((h, w), dtype=np.uint8)
                            rel_poly = poly - [x, y]
                            cv2.fillPoly(mask, [rel_poly], 255)

                            object_pool.append({
                                'crop': crop,
                                'mask': mask,
                                'cls': ann.get('class_id', 0),
                                'poly': rel_poly.tolist()
                            })
                        elif 'bbox' in ann and ann['bbox']:
                            # 检测标注：用bbox裁剪
                            x, y, w, h = ann['bbox']
                            if w < 10 or h < 10: continue
                            crop = p_img[y:y+h, x:x+w].copy()
                            mask = np.ones((h, w), dtype=np.uint8) * 255
                            rel_poly = [[0, 0], [w, 0], [w, h], [0, h]]
                            object_pool.append({
                                'crop': crop,
                                'mask': mask,
                                'cls': ann.get('class_id', 0),
                                'poly': rel_poly
                            })
            
        # B. 提取示例库对象池 (Example Library)
        if augmentations.get('example_copypaste'):
            cp_config = augmentations['example_copypaste']
            selected_examples = cp_config.get('Selected', []) # 手动指定的示例文件夹列表
            
            support_sets_dir = self.get_project_support_sets_dir(project_name)
            if not support_sets_dir:
                support_sets_dir = settings.get("support_sets_dir") or os.path.join(self.base_dir, "support_sets")
            if os.path.exists(support_sets_dir):
                # 如果指定了 Selected，则只从选中的文件夹加载；否则加载全部
                folders_to_scan = selected_examples if selected_examples else os.listdir(support_sets_dir)
                
                for folder_name in folders_to_scan:
                    folder_path = os.path.join(support_sets_dir, folder_name)
                    if not os.path.isdir(folder_path): continue
                    
                    info_path = os.path.join(folder_path, "info.json")
                    if not os.path.exists(info_path): continue
                    
                    try:
                        with open(info_path, 'r', encoding='utf-8-sig') as f:
                            info = json.load(f)
                        
                        ann = info.get('annotation')
                        if not ann: continue
                        
                        img_path = info.get('image_path')
                        if not img_path or not os.path.exists(img_path):
                            # 尝试在文件夹中找图片
                            for f in os.listdir(folder_path):
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(folder_path, f)
                                    break
                        
                        if not img_path: continue
                        
                        p_img = imread_unicode(img_path)
                        if p_img is None: continue
                        
                        # 类别映射：尝试根据名称匹配当前项目的类别 ID
                        cat_name = info.get('category') or ann.get('label')
                        cls_id = 0
                        if cat_name in class_names:
                            cls_id = class_names.index(cat_name)
                        
                        # 处理分割标注
                        if 'polygons' in ann and ann['polygons']:
                            poly = np.array(ann['polygons'][0], dtype=np.int32)
                            x, y, w, h = cv2.boundingRect(poly)
                            if w < 10 or h < 10: continue
                            
                            crop = p_img[y:y+h, x:x+w].copy()
                            mask = np.zeros((h, w), dtype=np.uint8)
                            rel_poly = poly - [x, y]
                            cv2.fillPoly(mask, [rel_poly], 255)
                            
                            example_pool.append({
                                'crop': crop,
                                'mask': mask,
                                'cls': cls_id,
                                'poly': rel_poly.tolist()
                            })
                        # 处理检测标注
                        elif 'bbox' in ann:
                            x, y, w, h = [int(v) for v in ann['bbox']]
                            if w < 10 or h < 10: continue
                            crop = p_img[y:y+h, x:x+w].copy()
                            mask = np.ones((h, w), dtype=np.uint8) * 255
                            example_pool.append({
                                'crop': crop,
                                'mask': mask,
                                'cls': cls_id,
                                'poly': [[0,0], [w,0], [w,h], [0,h]]
                            })
                    except Exception as e:
                        if settings.DEBUG:
                            raise
                        print(f"[Export] Failed to load example from {folder_path}: {e}")

        # --- 3. 循环导出数据 ---
        total_images = sum(len(imgs) for imgs in splits.values())
        processed_count = 0
        
        for split_name, split_images in splits.items():
            if not split_images: continue
            
            # 版本内的结构: {split}/images, {split}/{annotation_dir}
            v_split_root = os.path.join(version_path, split_name)
            v_img_dir = os.path.join(v_split_root, "images")
            os.makedirs(v_img_dir, exist_ok=True)
            
            # 根据导出格式创建对应的标注目录
            format_to_ann_dir = {
                "yolo": "labels", "yoloseg": "labels", "yoloobb": "labels",
                "coco": "coco_annotations",
                "voc": "voc_annotations",
                "labelme": "annotations",
            }
            ann_dir_name = format_to_ann_dir.get(export_format, "labels")
            v_label_dir = os.path.join(v_split_root, ann_dir_name)
            os.makedirs(v_label_dir, exist_ok=True)
            
            for img_name in split_images:
                img_src_path = os.path.join(img_src_root, img_name)
                if not os.path.exists(img_src_path): 
                    processed_count += 1
                    continue
                
                img = imread_unicode(img_src_path)
                if img is None:
                    processed_count += 1
                    continue
                
                # --- ROI 裁剪 ---
                rx, ry, rw, rh = 0, 0, img.shape[1], img.shape[0]
                if use_roi:
                    rx, ry, rw, rh = roi_bbox
                    # 边界检查
                    rx = max(0, min(rx, img.shape[1] - 1))
                    ry = max(0, min(ry, img.shape[0] - 1))
                    rw = max(1, min(rw, img.shape[1] - rx))
                    rh = max(1, min(rh, img.shape[0] - ry))
                    img = img[ry:ry+rh, rx:rx+rw]

                # Update progress (map 20-100 range for actual processing)
                processed_count += 1
                if progress_callback:
                    prog = 20 + int((processed_count / total_images) * 80)
                    progress_callback(min(prog, 100))
                
                base_name = os.path.splitext(img_name)[0]
                
                # 初始化本图片的标注列表
                det_annotations = []
                seg_annotations = []
                obb_annotations = []
                
                # --- 加载标注 ---
                # 从 annotations/ 目录读取 labelme 格式标注
                ann_dir = os.path.join(project_path, "annotations")
                ann_file = os.path.join(ann_dir, f"{base_name}.json")

                all_annotations = {'status': 'error', 'annotations': []}
                if os.path.exists(ann_file):
                    try:
                        from core.backend.formats import load_annotations
                        result = load_annotations(ann_file)
                        if result and result.get('annotations'):
                            all_annotations = {'status': 'success', 'annotations': result['annotations']}
                    except Exception as e:
                        if settings.DEBUG:
                            raise
                        print(f"[Export] Error loading annotation {ann_file}: {e}")

                # 未导出未标注图片时，跳过没有标注的图片（no status=success 即无标注）
                if not config.get('export_unannotated', True) and all_annotations['status'] != 'success':
                    continue
                
                if all_annotations['status'] == 'success':
                    loaded_anns = all_annotations.get('annotations', [])

                    for ann in loaded_anns:
                        cid = ann.get('class_id', 0)
                        shape_type = ann.get('shape_type', '')
                        has_polygons = 'polygons' in ann and ann['polygons']
                        has_bbox = 'bbox' in ann and ann['bbox']

                        # 根据导出任务类型转换标注
                        if task_type == 'det':
                            # 导出检测：所有标注转为水平矩形 (bbox)
                            if has_polygons:
                                # polygon/rotation → 最小外接矩形
                                poly = np.array(ann['polygons'][0], dtype=np.int32)
                                bx, by, bw, bh = cv2.boundingRect(poly)
                                nx1, ny1 = max(rx, bx), max(ry, by)
                                nx2, ny2 = min(rx + rw, bx + bw), min(ry + rh, by + bh)
                                if nx2 > nx1 and ny2 > ny1:
                                    rel_x, rel_y = nx1 - rx, ny1 - ry
                                    rel_w, rel_h = nx2 - nx1, ny2 - ny1
                                    det_annotations.append([cid, (rel_x + rel_w/2)/rw, (rel_y + rel_h/2)/rh, rel_w/rw, rel_h/rh])
                            elif has_bbox:
                                x, y, w, h = ann['bbox']
                                nx1, ny1 = max(rx, x), max(ry, y)
                                nx2, ny2 = min(rx + rw, x + w), min(ry + rh, y + h)
                                if nx2 > nx1 and ny2 > ny1:
                                    rel_x, rel_y = nx1 - rx, ny1 - ry
                                    rel_w, rel_h = nx2 - nx1, ny2 - ny1
                                    det_annotations.append([cid, (rel_x + rel_w/2)/rw, (rel_y + rel_h/2)/rh, rel_w/rw, rel_h/rh])

                        elif task_type == 'seg':
                            # 导出分割：所有标注转为多边形 (polygon)
                            if has_polygons:
                                # polygon/rotation → 直接用多边形点
                                poly = ann['polygons'][0]
                                rel_poly = []
                                is_inside = False
                                for pt in poly:
                                    px, py = pt[0], pt[1]
                                    rpx, rpy = px - rx, py - ry
                                    rel_poly.append([rpx / rw, rpy / rh])
                                    if rx <= px <= rx + rw and ry <= py <= ry + rh:
                                        is_inside = True
                                if is_inside:
                                    row = [cid]
                                    for pt in rel_poly:
                                        row.extend([max(0.0, min(1.0, pt[0])), max(0.0, min(1.0, pt[1]))])
                                    seg_annotations.append(row)
                            elif has_bbox:
                                # rectangle → 4角点转polygon
                                x, y, w, h = ann['bbox']
                                nx1, ny1 = max(rx, x), max(ry, y)
                                nx2, ny2 = min(rx + rw, x + w), min(ry + rh, y + h)
                                if nx2 > nx1 and ny2 > ny1:
                                    rel_x, rel_y = nx1 - rx, ny1 - ry
                                    rel_w, rel_h = nx2 - nx1, ny2 - ny1
                                    row = [cid, rel_x/rw, rel_y/rh, (rel_x+rel_w)/rw, rel_y/rh, (rel_x+rel_w)/rw, (rel_y+rel_h)/rh, rel_x/rw, (rel_y+rel_h)/rh]
                                    seg_annotations.append(row)

                        elif task_type == 'obb':
                            # 导出旋转：所有标注转为旋转矩形 (rotation, 4点)
                            if has_polygons:
                                poly_pts = ann['polygons'][0]
                                if shape_type == 'rotation' and len(poly_pts) == 4:
                                    # rotation → 直接用4个角点
                                    rel_poly = []
                                    is_inside = False
                                    for pt in poly_pts:
                                        px, py = pt[0], pt[1]
                                        rpx, rpy = px - rx, py - ry
                                        rel_poly.append([rpx / rw, rpy / rh])
                                        if rx <= px <= rx + rw and ry <= py <= ry + rh:
                                            is_inside = True
                                    if is_inside:
                                        row = [cid]
                                        for pt in rel_poly:
                                            row.extend([max(0.0, min(1.0, pt[0])), max(0.0, min(1.0, pt[1]))])
                                        obb_annotations.append(row)
                                else:
                                    # polygon → 最小外接旋转矩形 (minAreaRect)
                                    poly_np = np.array(poly_pts, dtype=np.float32)
                                    rect = cv2.minAreaRect(poly_np)
                                    box = cv2.boxPoints(rect)
                                    box = np.int0(box)
                                    # 检查是否在ROI内
                                    is_inside = False
                                    rel_poly = []
                                    for pt in box:
                                        px, py = float(pt[0]), float(pt[1])
                                        rpx, rpy = px - rx, py - ry
                                        rel_poly.append([rpx / rw, rpy / rh])
                                        if rx <= px <= rx + rw and ry <= py <= ry + rh:
                                            is_inside = True
                                    if is_inside:
                                        row = [cid]
                                        for pt in rel_poly:
                                            row.extend([max(0.0, min(1.0, pt[0])), max(0.0, min(1.0, pt[1]))])
                                        obb_annotations.append(row)
                            elif has_bbox:
                                # rectangle → 4角点转rotation
                                x, y, w, h = ann['bbox']
                                nx1, ny1 = max(rx, x), max(ry, y)
                                nx2, ny2 = min(rx + rw, x + w), min(ry + rh, y + h)
                                if nx2 > nx1 and ny2 > ny1:
                                    rel_x, rel_y = nx1 - rx, ny1 - ry
                                    rel_w, rel_h = nx2 - nx1, ny2 - ny1
                                    row = [cid, rel_x/rw, rel_y/rh, (rel_x+rel_w)/rw, rel_y/rh, (rel_x+rel_w)/rw, (rel_y+rel_h)/rh, rel_x/rw, (rel_y+rel_h)/rh]
                                    obb_annotations.append(row)
                else:
                    pass
                    # print(f"DEBUG: load_annotations failed for {img_name}: {all_annotations.get('message')}")

                # --- A. 预处理 ---
                if preprocessing.get('resize'):
                    w_target = int(preprocessing['resize']['Width'])
                    h_target = int(preprocessing['resize']['Height'])
                    img = cv2.resize(img, (w_target, h_target))
                
                # --- B. 数据增强 (仅对训练集生效) ---
                if split_name == 'train':
                    if augmentations.get('flip') and random.random() > 0.5:
                        img = cv2.flip(img, 1)
                        for ann in det_annotations: ann[1] = 1.0 - ann[1]
                        for ann in seg_annotations:
                            # YOLO seg: cls x1 y1 x2 y2 ...
                            for i in range(1, len(ann), 2):
                                ann[i] = 1.0 - ann[i]
                        for ann in obb_annotations:
                            # YOLO obb: cls x1 y1 x2 y2 x3 y3 x4 y4
                            for i in range(1, len(ann), 2):
                                ann[i] = 1.0 - ann[i]
                    
                    if augmentations.get('hsv'):
                        h_gain, s_gain, v_gain = augmentations['hsv']['H'], augmentations['hsv']['S'], augmentations['hsv']['V']
                        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
                        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                        dtype = img.dtype
                        x = np.arange(0, 256, dtype=np.int16)
                        lut_hue = ((x * r[0]) % 180).astype(dtype)
                        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
                        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
                        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
                        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

                    if augmentations.get('blur'):
                        sigma = augmentations['blur']['Sigma']
                        if sigma > 0: img = cv2.GaussianBlur(img, (0, 0), sigma)

                    # --- B.4 Copy-Paste ---
                    if augmentations.get('copypaste') and object_pool:
                        cp_config = augmentations['copypaste']
                        count = int(cp_config.get('Count', 3))
                        h, w = img.shape[:2]
                        
                        for _ in range(count):
                            obj = random.choice(object_pool)
                            ocrop, omask, ocls, opoly = obj['crop'], obj['mask'], obj['cls'], obj['poly']
                            oh, ow = ocrop.shape[:2]
                            
                            # 随机缩放对象 (0.5x ~ 1.2x)
                            scale = random.uniform(0.5, 1.2)
                            # 如果对象太大，强制缩小
                            if oh * scale > h * 0.8 or ow * scale > w * 0.8:
                                scale = min(h * 0.5 / oh, w * 0.5 / ow)
                                
                            curr_crop = ocrop
                            curr_mask = omask
                            curr_poly = opoly
                            
                            if scale != 1.0:
                                curr_crop = cv2.resize(ocrop, (0, 0), fx=scale, fy=scale)
                                curr_mask = cv2.resize(omask, (0, 0), fx=scale, fy=scale)
                                curr_poly = [[p[0]*scale, p[1]*scale] for p in opoly]
                            
                            oh, ow = curr_crop.shape[:2]
                            
                            # 随机位置
                            if w - ow <= 0 or h - oh <= 0: continue
                            x1 = random.randint(0, w - ow)
                            y1 = random.randint(0, h - oh)
                            
                            # 粘贴
                            mask_3d = cv2.merge([curr_mask, curr_mask, curr_mask]) / 255.0
                            roi = img[y1:y1+oh, x1:x1+ow]
                            img[y1:y1+oh, x1:x1+ow] = (roi * (1 - mask_3d) + curr_crop * mask_3d).astype(np.uint8)
                            
                            # 更新标注
                            # Det: [cls, cx, cy, nw, nh]
                            cx_new, cy_new = (x1 + ow/2)/w, (y1 + oh/2)/h
                            nw_new, nh_new = ow/w, oh/h
                            det_annotations.append([ocls, cx_new, cy_new, nw_new, nh_new])
                            
                            # Seg: [cls, x1, y1, x2, y2...]
                            new_seg = [ocls]
                            for pt in curr_poly:
                                new_seg.extend([(pt[0] + x1)/w, (pt[1] + y1)/h])
                            seg_annotations.append(new_seg)

                            # OBB: [cls, x1, y1, x2, y2, x3, y3, x4, y4]
                            if len(curr_poly) >= 4:
                                new_obb = [ocls]
                                for i in range(4):
                                    new_obb.extend([(curr_poly[i][0] + x1)/w, (curr_poly[i][1] + y1)/h])
                                obb_annotations.append(new_obb)
                            else:
                                new_obb = [ocls, x1/w, y1/h, (x1+ow)/w, y1/h, (x1+ow)/w, (y1+oh)/h, x1/w, (y1+oh)/h]
                                obb_annotations.append(new_obb)

                    # --- B.5 Example-Copy-Paste ---
                    if augmentations.get('example_copypaste') and example_pool:
                        ex_config = augmentations['example_copypaste']
                        count = int(ex_config.get('Count', 3))
                        h, w = img.shape[:2]
                        
                        for _ in range(count):
                            obj = random.choice(example_pool)
                            ocrop, omask, ocls, opoly = obj['crop'], obj['mask'], obj['cls'], obj['poly']
                            oh, ow = ocrop.shape[:2]
                            
                            # 随机缩放对象 (0.5x ~ 1.2x)
                            scale = random.uniform(0.5, 1.2)
                            if oh * scale > h * 0.8 or ow * scale > w * 0.8:
                                scale = min(h * 0.5 / oh, w * 0.5 / ow)
                                
                            curr_crop = ocrop
                            curr_mask = omask
                            curr_poly = opoly
                            
                            if scale != 1.0:
                                curr_crop = cv2.resize(ocrop, (0, 0), fx=scale, fy=scale)
                                curr_mask = cv2.resize(omask, (0, 0), fx=scale, fy=scale)
                                curr_poly = [[p[0]*scale, p[1]*scale] for p in opoly]
                            
                            oh, ow = curr_crop.shape[:2]
                            
                            # 随机位置
                            if w - ow <= 0 or h - oh <= 0: continue
                            x1 = random.randint(0, w - ow)
                            y1 = random.randint(0, h - oh)
                            
                            # 粘贴
                            mask_3d = cv2.merge([curr_mask, curr_mask, curr_mask]) / 255.0
                            roi = img[y1:y1+oh, x1:x1+ow]
                            img[y1:y1+oh, x1:x1+ow] = (roi * (1 - mask_3d) + curr_crop * mask_3d).astype(np.uint8)
                            
                            # 更新标注
                            # Det: [cls, cx, cy, nw, nh]
                            cx_new, cy_new = (x1 + ow/2)/w, (y1 + oh/2)/h
                            nw_new, nh_new = ow/w, oh/h
                            det_annotations.append([ocls, cx_new, cy_new, nw_new, nh_new])
                            
                            # Seg: [cls, x1, y1, x2, y2...]
                            new_seg = [ocls]
                            for pt in curr_poly:
                                new_seg.extend([(pt[0] + x1)/w, (pt[1] + y1)/h])
                            seg_annotations.append(new_seg)

                            # OBB: [cls, x1, y1, x2, y2, x3, y3, x4, y4]
                            if len(curr_poly) >= 4:
                                new_obb = [ocls]
                                for i in range(4):
                                    new_obb.extend([(curr_poly[i][0] + x1)/w, (curr_poly[i][1] + y1)/h])
                                obb_annotations.append(new_obb)
                            else:
                                new_obb = [ocls, x1/w, y1/h, (x1+ow)/w, y1/h, (x1+ow)/w, (y1+oh)/h, x1/w, (y1+oh)/h]
                                obb_annotations.append(new_obb)

                # --- 保存结果 ---
                imwrite_unicode(os.path.join(v_img_dir, img_name), img)
                
                if export_format == "labelme":
                    # labelme格式：根据task_type转换shape_type后保存
                    v_ann_dir = os.path.join(v_split_root, "annotations")
                    os.makedirs(v_ann_dir, exist_ok=True)
                    if os.path.exists(ann_file):
                        if task_type and task_type in ('det', 'seg', 'obb'):
                            with open(ann_file, 'r', encoding='utf-8') as f:
                                labelme_data = json.load(f)
                            for shape in labelme_data.get('shapes', []):
                                orig_type = shape.get('shape_type', '')
                                points = shape.get('points', [])
                                if task_type == 'det' and orig_type != 'rectangle':
                                    # 转为rectangle：从多边形计算最小外接矩形
                                    if points:
                                        xs = [p[0] for p in points]
                                        ys = [p[1] for p in points]
                                        x1, y1 = min(xs), min(ys)
                                        x2, y2 = max(xs), max(ys)
                                        shape['points'] = [[x1, y1], [x2, y2]]
                                    shape['shape_type'] = 'rectangle'
                                elif task_type == 'seg' and orig_type != 'polygon':
                                    # 转为polygon：从矩形生成4角点
                                    if orig_type == 'rectangle' and len(points) == 2:
                                        x1, y1 = points[0]
                                        x2, y2 = points[1]
                                        shape['points'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                    shape['shape_type'] = 'polygon'
                                elif task_type == 'obb' and orig_type != 'rotation':
                                    # 转为rotation：从多边形计算最小外接旋转矩形
                                    if points and len(points) >= 2:
                                        poly_np = np.array(points, dtype=np.float32)
                                        if orig_type == 'rectangle' and len(points) == 2:
                                            x1, y1 = points[0]
                                            x2, y2 = points[1]
                                            poly_np = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                                        rect = cv2.minAreaRect(poly_np)
                                        box = cv2.boxPoints(rect)
                                        shape['points'] = [[float(p[0]), float(p[1])] for p in box]
                                    shape['shape_type'] = 'rotation'
                            with open(os.path.join(v_ann_dir, f"{base_name}.json"), 'w', encoding='utf-8') as f:
                                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
                        else:
                            shutil.copy2(ann_file, os.path.join(v_ann_dir, f"{base_name}.json"))
                elif export_format == "coco":
                    # COCO格式：在split结束后生成单个JSON文件，此处先缓存标注数据
                    if not hasattr(self, '_coco_annotations_cache'):
                        self._coco_annotations_cache = {}
                    if split_name not in self._coco_annotations_cache:
                        self._coco_annotations_cache[split_name] = []
                    if task_type == 'seg':
                        target_annotations = seg_annotations
                    elif task_type == 'obb':
                        target_annotations = obb_annotations
                    else:
                        target_annotations = det_annotations
                    if target_annotations:
                        img_path = os.path.join(v_img_dir, img_name)
                        img_shape = img.shape if img is not None else None
                        img_w = img_shape[1] if img_shape is not None else 640
                        img_h = img_shape[0] if img_shape is not None else 480
                        self._coco_annotations_cache[split_name].append({
                            'file_name': img_name,
                            'width': img_w,
                            'height': img_h,
                            'annotations': target_annotations,
                            'task_type': task_type
                        })
                elif export_format == "voc":
                    # VOC格式：生成XML文件（仅支持检测任务）
                    if task_type == 'det':
                        target_annotations = det_annotations
                    else:
                        # VOC不支持分割/旋转，退化为检测
                        target_annotations = det_annotations

                    if target_annotations:
                        import xml.etree.ElementTree as ET
                        annotation = ET.Element('annotation')
                        ET.SubElement(annotation, 'folder').text = 'images'
                        ET.SubElement(annotation, 'filename').text = img_name
                        size = ET.SubElement(annotation, 'size')
                        ET.SubElement(size, 'width').text = str(img.shape[1] if img is not None else 640)
                        ET.SubElement(size, 'height').text = str(img.shape[0] if img is not None else 480)
                        ET.SubElement(size, 'depth').text = '3'
                        for raw_ann in target_annotations:
                            cls_id = int(raw_ann[0])
                            label_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                            cx, cy, w, h = raw_ann[1], raw_ann[2], raw_ann[3], raw_ann[4]
                            img_w = img.shape[1] if img is not None else 640
                            img_h = img.shape[0] if img is not None else 480
                            x1 = int((cx - w / 2) * img_w)
                            y1 = int((cy - h / 2) * img_h)
                            x2 = int((cx + w / 2) * img_w)
                            y2 = int((cy + h / 2) * img_h)
                            obj = ET.SubElement(annotation, 'object')
                            ET.SubElement(obj, 'name').text = label_name
                            ET.SubElement(obj, 'pose').text = 'Unspecified'
                            ET.SubElement(obj, 'truncated').text = '0'
                            ET.SubElement(obj, 'difficult').text = '0'
                            bndbox = ET.SubElement(obj, 'bndbox')
                            ET.SubElement(bndbox, 'xmin').text = str(x1)
                            ET.SubElement(bndbox, 'ymin').text = str(y1)
                            ET.SubElement(bndbox, 'xmax').text = str(x2)
                            ET.SubElement(bndbox, 'ymax').text = str(y2)
                        tree = ET.ElementTree(annotation)
                        tree.write(os.path.join(v_label_dir, f"{base_name}.xml"), encoding='utf-8', xml_declaration=True)
                else:
                    # YOLO等格式：根据task_type保存对应标注
                    if task_type == 'seg':
                        target_annotations = seg_annotations
                    elif task_type == 'obb':
                        target_annotations = obb_annotations
                    else:
                        target_annotations = det_annotations
                    
                    if target_annotations:
                        with open(os.path.join(v_label_dir, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                            for ann in target_annotations:
                                f.write(f"{int(ann[0])} {' '.join([f'{x:.6f}' for x in ann[1:]])}\n")

        # 3.5 生成COCO格式JSON文件
        if export_format == "coco" and hasattr(self, '_coco_annotations_cache'):
            coco_data_base = {
                'info': {'description': 'COCO Dataset', 'version': '1.0', 'year': 2024},
                'images': [], 'annotations': [], 'categories': []
            }
            label_to_cat_id = {}
            next_cat_id = 1
            next_img_id = 1
            next_ann_id = 1

            for split_name in ["train", "val", "test"]:
                coco_split_data = json.loads(json.dumps(coco_data_base))
                coco_split_data['images'] = []
                coco_split_data['annotations'] = []
                label_to_cat_id_local = {}
                next_cat_id_local = 1
                next_img_id_local = 1
                next_ann_id_local = 1

                cached_items = self._coco_annotations_cache.get(split_name, [])
                for item in cached_items:
                    img_id = next_img_id_local
                    coco_split_data['images'].append({
                        'id': img_id,
                        'file_name': item['file_name'],
                        'width': item['width'],
                        'height': item['height']
                    })
                    for raw_ann in item['annotations']:
                        cls_id = int(raw_ann[0])
                        label_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                        if label_name not in label_to_cat_id_local:
                            label_to_cat_id_local[label_name] = next_cat_id_local
                            coco_split_data['categories'].append({
                                'id': next_cat_id_local, 'name': label_name
                            })
                            next_cat_id_local += 1
                        cat_id = label_to_cat_id_local[label_name]
                        ann_id = next_ann_id_local
                        next_ann_id_local += 1

                        coco_ann = {'id': ann_id, 'image_id': img_id, 'category_id': cat_id, 'iscrowd': 0}
                        task_t = item.get('task_type', 'det')
                        if task_t == 'det':
                            cx, cy, w, h = raw_ann[1], raw_ann[2], raw_ann[3], raw_ann[4]
                            x, y = (cx - w / 2) * item['width'], (cy - h / 2) * item['height']
                            bw, bh = w * item['width'], h * item['height']
                            coco_ann['bbox'] = [x, y, bw, bh]
                            coco_ann['area'] = bw * bh
                        elif task_t in ('seg', 'obb'):
                            pts = raw_ann[1:]
                            seg = []
                            for i in range(0, len(pts), 2):
                                seg.extend([pts[i] * item['width'], pts[i + 1] * item['height']])
                            coco_ann['segmentation'] = [seg]
                            all_x = [pts[i] * item['width'] for i in range(0, len(pts), 2)]
                            all_y = [pts[i + 1] * item['height'] for i in range(1, len(pts), 2)]
                            x_min, y_min = min(all_x), min(all_y)
                            w_box, h_box = max(all_x) - x_min, max(all_y) - y_min
                            coco_ann['bbox'] = [x_min, y_min, w_box, h_box]
                            coco_ann['area'] = w_box * h_box

                        coco_split_data['annotations'].append(coco_ann)

                # 保存COCO JSON到对应的split目录
                # 注意：主循环会跳过空划分（如小数据集随机划分后 train/val 可能为 0 张），
                # 因此这里必须确保 coco_annotations 目录存在，否则空划分会触发 FileNotFoundError
                v_split_root = os.path.join(version_path, split_name)
                coco_ann_dir = os.path.join(v_split_root, "coco_annotations")
                os.makedirs(coco_ann_dir, exist_ok=True)
                coco_path = os.path.join(coco_ann_dir, "_annotations.coco.json")
                with open(coco_path, 'w', encoding='utf-8') as f:
                    json.dump(coco_split_data, f, indent=2, ensure_ascii=False)

            # 清理缓存
            del self._coco_annotations_cache

        # 4. 保存 classes.txt 和 dataset.yaml
        for split_name in ["train", "val", "test"]:
            v_split_root = os.path.join(version_path, split_name)
            if os.path.exists(v_split_root):
                with open(os.path.join(v_split_root, "classes.txt"), 'w', encoding='utf-8') as f:
                    for name in class_names:
                        f.write(f"{name}\n")
        
        yaml_content = {
            "path": os.path.abspath(version_path).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names
        }
        
        import yaml
        with open(os.path.join(version_path, "dataset.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)

        # 5. 更新项目信息
        self._update_project_info(project_name, {"version_count": v_idx})
        
        # 5. 保存导出配置快照
        with open(os.path.join(version_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
            
        return {"status": "success", "version": version_name, "path": version_path}

    def stop_recording(self, project_name: str = "") -> dict:
        """停止录制并返回录制结果，纯数据方法

        Args:
            project_name: 项目名称（可选）
        Returns:
            dict: 包含录制步骤数和录制数据
        """
        from core.common.action_recorder import ActionRecorder
        recorder = ActionRecorder.instance()
        if not recorder.is_recording:
            return {"success": False, "step_count": 0, "scenario": None, "error": "当前未在录制"}
        scenario = recorder.stop()
        step_count = len(scenario.get("steps", []))
        return {
            "success": True,
            "step_count": step_count,
            "scenario": scenario
        }
