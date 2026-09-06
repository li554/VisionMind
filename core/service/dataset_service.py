import os
import shutil
import cv2
import numpy as np
import json
from typing import List, Dict, Any, Optional
from core.common.config import Config
from core.common.settings import settings
from core.backend.utils import find_label_file
from core.backend.formats import (
    load_annotations, save_annotations
)

class DatasetService:
    def __init__(self, output_root=None):
        if output_root is None:
            try:
                output_root = settings.get("output_dir", Config.OUTPUTS_DIR)
            except:
                if settings.DEBUG:
                    raise
                output_root = Config.OUTPUTS_DIR
        self.output_root = output_root

    def import_and_normalize(self, source_dir: str, project_name: str = None, output_root: str = None, force=False, progress_callback=None):
        """
        Imports a dataset from source_dir, copies it to outputs/<project_name>,
        and ensures a normalized structure with det/ and seg/ subfolders.
        """
        if not project_name:
            project_name = os.path.basename(source_dir)

        # 1. Define base output paths
        root = output_root or self.output_root
        project_output_dir = os.path.join(root, project_name)

        # Check if project already exists
        if os.path.exists(project_output_dir) and not force:
            # If it's a valid project, we might still want to proceed with importing more data
            pass

        # 2. Scan for images
        image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')
        image_files = []
        for root_walk, dirs, files in os.walk(source_dir):
            # # Skip output directory if it's inside source_dir
            # if root in os.path.abspath(root_walk):
            #     continue
            for f in files:
                if f.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root_walk, f))

        if not image_files:
            return {"status": "error", "message": "未在选择的目录中找到图片。"}

        # 3. Get class names from source AND merge with existing project categories
        class_names = self._load_class_names(source_dir)
        print(f"[DatasetImport] Loaded class names from source: {class_names}")

        # 3.1 Merge with existing project categories if project already exists
        existing_categories = self._load_existing_project_categories(project_output_dir)
        if existing_categories:
            print(f"[DatasetImport] Found existing project categories: {existing_categories}")
            # Merge: existing categories first, then add new ones from source
            merged_names = list(existing_categories)
            for name in class_names:
                if name not in merged_names:
                    merged_names.append(name)
            class_names = merged_names
            print(f"[DatasetImport] Merged categories: {class_names}")

        # 4. Process each image and track splits
        processed_count = 0
        total_count = len(image_files)
        image_splits = {}
        all_found_ids = set()

        for i, img_path in enumerate(image_files):
            # Determine split (train/val/test)
            split = self._determine_split(img_path, source_dir)

            # Find corresponding label
            label_path = find_label_file(img_path)
            if label_path:
                print(f"[DatasetImport] Found label for {os.path.basename(img_path)}: {label_path}")
            else:
                print(f"[DatasetImport] No label found for {os.path.basename(img_path)}")

            # Copy and convert (Now saves to flat structure)
            # Returns the final destination path of the image and found class IDs
            dest_img_path, found_ids = self._process_single_image(img_path, label_path, project_output_dir, class_names, split)

            if found_ids:
                all_found_ids.update(found_ids)

            # Record split in memory
            if dest_img_path:
                image_splits[dest_img_path] = split
                processed_count += 1

            # Update progress
            if progress_callback:
                progress_callback(int((i + 1) / total_count * 100))

        # If no class names were found initially, generate them from found IDs
        if not class_names and all_found_ids:
            max_id = max(all_found_ids)
            class_names = [f"class_{i}" for i in range(max_id + 1)]

        # 5. Save dataset_split.json
        split_path = os.path.join(project_output_dir, "dataset_split.json")
        try:
            # Merge with existing splits if file already exists
            if os.path.exists(split_path):
                with open(split_path, 'r', encoding='utf-8-sig') as f:
                    existing_splits = json.load(f)
                    existing_splits.update(image_splits)
                    image_splits = existing_splits

            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(image_splits, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"Error saving dataset_split.json: {e}")

        # 6. Generate configuration files for both tasks
        self._generate_configs(project_output_dir, class_names)

        return {
            "status": "success",
            "message": f"成功导入 {processed_count} 张图片至 {project_output_dir}",
            "output_dir": project_output_dir,
            "project_name": project_name
        }

    def _load_class_names(self, source_dir: str) -> List[str]:
        # Search for classes.txt in various possible locations
        search_paths = [
            os.path.join(source_dir, "classes.txt"),
            os.path.join(source_dir, "labels", "classes.txt"),
            os.path.join(source_dir, "train", "classes.txt"),
            os.path.join(source_dir, "train", "labels", "classes.txt"),
            os.path.join(source_dir, "data.yaml"), # YOLOv5/v8 sometimes has names in yaml
        ]

        for path in search_paths:
            if not os.path.exists(path): continue

            if path.endswith(".yaml"):
                try:
                    import yaml
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            if isinstance(data['names'], list):
                                return data['names']
                            elif isinstance(data['names'], dict):
                                return [data['names'][i] for i in sorted(data['names'].keys())]
                except: pass
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]

        # Try to extract class names from annotation files (all formats)
        class_names = self._extract_class_names_from_annotations(source_dir)
        if class_names:
            return class_names

        return []

    def _load_existing_project_categories(self, project_dir: str) -> List[str]:
        """加载项目已存在的类别名称（用于合并导入）"""
        # 尝试从det或seg目录下的classes.txt加载
        for task in ['det', 'seg']:
            classes_path = os.path.join(project_dir, task, "classes.txt")
            if os.path.exists(classes_path):
                try:
                    with open(classes_path, 'r', encoding='utf-8') as f:
                        return [line.strip() for line in f if line.strip()]
                except:
                    pass

        # 尝试从project_info.json加载
        info_path = os.path.join(project_dir, "project_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8-sig') as f:
                    info = json.load(f)
                    categories = info.get('categories', {})
                    if categories:
                        # 处理 {id: name} 格式
                        if all(isinstance(v, str) for v in categories.values()):
                            sorted_ids = sorted([int(k) for k in categories.keys() if str(k).isdigit()])
                            return [categories[str(i)] for i in sorted_ids]
            except:
                pass

        return []

    def _extract_class_names_from_annotations(self, source_dir: str) -> List[str]:
        """从标注文件中提取类别名称列表（支持所有格式）"""

        class_set = set()

        print(f"[DatasetImport] Scanning for annotation files in: {source_dir}")

        coco_paths = []
        for root_walk, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith('.json'):
                    json_path = os.path.join(root_walk, f)
                    if self._is_coco_format(json_path):
                        coco_paths.append(json_path)

        if coco_paths:
            print(f"[DatasetImport] Found {len(coco_paths)} COCO format files")
            for coco_path in coco_paths:
                try:
                    with open(coco_path, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    categories = data.get('categories', [])
                    for cat in categories:
                        name = cat.get('name')
                        if name:
                            class_set.add(name)
                except Exception as e:
                    print(f"[DatasetImport] Error reading COCO file {coco_path}: {e}")

            if class_set:
                print(f"[DatasetImport] Extracted {len(class_set)} classes from COCO: {sorted(list(class_set))}")
                return sorted(list(class_set))

        ann_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        ann_count = 0
        processed_ann_paths = set()

        for root_walk, dirs, files in os.walk(source_dir):
            for f in files:
                if f.lower().endswith(ann_extensions):
                    ann_count += 1
                    img_path = os.path.join(root_walk, f)
                    ann_path = find_label_file(img_path)

                    if ann_path and ann_path not in processed_ann_paths:
                        processed_ann_paths.add(ann_path)
                        try:
                            result = load_annotations(ann_path)
                            instances = result.get("annotations", [])
                            for inst in instances:
                                label = inst.get('label')
                                if label:
                                    class_set.add(label)
                        except Exception as e:
                            if settings.DEBUG:
                                raise
                            print(f"[DatasetImport] Error parsing {ann_path}: {e}")
                            continue

        print(f"[DatasetImport] Scanned {ann_count} images, {len(processed_ann_paths)} unique annotation files, extracted {len(class_set)} classes: {sorted(list(class_set))}")

        if class_set:
            return sorted(list(class_set))
        return []

    def _is_coco_format(self, json_path: str) -> bool:
        """检查JSON文件是否为COCO格式"""
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return 'images' in data and 'annotations' in data and 'categories' in data
        except:
            return False

    def _determine_split(self, img_path: str, source_root: str) -> str:
        rel_path = os.path.relpath(img_path, source_root).lower()
        if 'train' in rel_path: return 'train'
        if 'val' in rel_path: return 'val'
        if 'test' in rel_path: return 'test'
        return 'train'

    def _process_single_image(self, img_path, label_path, project_dir, class_names, split="train"):
        img_name = os.path.basename(img_path)
        found_ids = []

        # 1. Save Image to project_dir/images (common for both det and seg)
        img_dest_dir = os.path.join(project_dir, "images")
        os.makedirs(img_dest_dir, exist_ok=True)

        dest_path = os.path.join(img_dest_dir, img_name)

        # Handle filename collisions (e.g., same name in different split folders)
        if os.path.exists(dest_path):
            # If it's the exact same file (same path), we don't need to do anything
            # But usually it's a different file with the same name.
            # We prefix with split name to distinguish.
            img_name = f"{split}_{img_name}"
            dest_path = os.path.join(img_dest_dir, img_name)

            # If it still exists, add a counter as a last resort
            counter = 1
            base, ext = os.path.splitext(img_name)
            while os.path.exists(dest_path):
                img_name = f"{base}_{counter}{ext}"
                dest_path = os.path.join(img_dest_dir, img_name)
                counter += 1

        if not os.path.exists(dest_path):
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                if settings.DEBUG:
                    raise
                print(f"Error copying image {img_path} to {dest_path}: {e}")
                return None, []

        base_name = os.path.splitext(img_name)[0]

        if not label_path:
            return dest_path, []

        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None: return dest_path, []
            h, w = img.shape[:2]
        except Exception as e:
            if settings.DEBUG:
                raise
            print(f"Error reading image {img_path}: {e}")
            return dest_path, []

        result = load_annotations(label_path)
        instances = result.get("annotations", [])
        if not instances:
            return dest_path, []

        for inst in instances:
            if 'label' not in inst and class_names and inst.get('class_id', 0) < len(class_names):
                inst['label'] = class_names[inst['class_id']]
            elif 'label' not in inst:
                inst['label'] = f"class_{inst.get('class_id', 0)}"
            if "class_id" not in inst:
                inst["class_id"] = class_names.index(inst["label"])

        found_ids = [inst.get('class_id', 0) for inst in instances]

        categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
        save_data = {
            "annotations": instances,
            "image_info": {"filename": img_name, "width": w, "height": h},
            "categories": categories
        }

        # 统一保存为labelme格式到annotations/目录
        ann_dir = os.path.join(project_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        save_annotations(save_data, os.path.join(ann_dir, f"{base_name}.json"), format_type="labelme")

        return dest_path, found_ids




    def _generate_configs(self, project_dir, class_names):
        if not class_names:
            class_names = ["class_0"] # Default to class_0 if none found

        # 统一在annotations/目录下保存classes.txt
        ann_dir = os.path.join(project_dir, "annotations")
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir, exist_ok=True)

        # Save classes.txt in annotations root
        with open(os.path.join(ann_dir, "classes.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_names))

        # Also save classes.txt at project root for compatibility
        with open(os.path.join(project_dir, "classes.txt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_names))

        # Also update project_info.json if it exists
        from core.service.project_service import ProjectService
        project_name = os.path.basename(project_dir)
        service = ProjectService()

        # Determine categories
        categories = {str(i): name for i, name in enumerate(class_names)}

        # Use project service to update info safely
        service._update_project_info(project_name, {"categories": categories})
