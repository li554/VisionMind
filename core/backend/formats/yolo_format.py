import os
import cv2
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo
from ..utils import imread_unicode, imwrite_unicode


class YOLOFormat(AnnotationFormat):
    """YOLO 格式处理器 (支持 det/seg/obb)"""

    EXTENSIONS = ['.txt']

    @classmethod
    def can_load(cls, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() == '.txt'

    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描YOLO目录中所有标注文件，提取类别信息"""
        categories = []
        label_set = set()
        
        labels_dir = os.path.join(dir_path, "labels")
        if not os.path.exists(labels_dir):
            labels_dir = dir_path
        
        try:
            for filename in os.listdir(labels_dir):
                if not filename.endswith('.txt') or filename == 'classes.txt':
                    continue
                
                filepath = os.path.join(labels_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(float(parts[0]))
                                label_set.add(class_id)
                except:
                    continue
            
            max_id = max(label_set) if label_set else -1
            categories = [{"id": i, "name": f"class_{i}"} for i in range(max_id + 1)]
        except Exception as e:
            print(f"[YOLOFormat] Failed to scan categories: {e}")
        
        return categories

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从YOLO格式文件加载标注

        Args:
            file_path: YOLO格式TXT文件路径
            **kwargs: 额外参数
                - image_path: 图像文件路径（用于获取图像尺寸）

        Returns:
            AnnotationData: 统一格式的标注数据
        """
        annotations: List[Dict[str, Any]] = []
        image_info: Dict[str, Any] = {}
        categories: List[Dict[str, Any]] = []

        if not os.path.exists(file_path):
            return {"annotations": annotations, "image_info": image_info, "categories": categories}

        image_path = kwargs.get('image_path')
        img_w, img_h = 1920, 1080

        try:
            if image_path and os.path.exists(image_path):
                img = imread_unicode(image_path)
                if img is not None:
                    img_h, img_w = img.shape[:2]
                image_info = {
                    "filename": os.path.basename(image_path),
                    "width": img_w,
                    "height": img_h
                }

            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            categories = cls.get_or_create_categories(parent_dir)
            
            id_to_name = {cat["id"]: cat["name"] for cat in categories}

            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                label = id_to_name.get(class_id, f"class_{class_id}")

                if len(parts) == 5:
                    x_center, y_center, box_w, box_h = map(float, parts[1:5])
                    x = (x_center - box_w / 2) * img_w
                    y = (y_center - box_h / 2) * img_h
                    bbox = [x, y, box_w * img_w, box_h * img_h]

                    annotations.append({
                        'label': label,
                        'class_id': class_id,
                        'bbox': bbox,
                        'polygons': [[[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]],
                                      [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]]
                    })
                elif len(parts) == 9:
                    pts = []
                    for i in range(1, 9, 2):
                        pts.append([float(parts[i]) * img_w, float(parts[i + 1]) * img_h])

                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                        annotations.append({
                            'label': label,
                            'class_id': class_id,
                            'bbox': [float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))],
                            'polygons': [pts]
                        })
                elif len(parts) > 9 and len(parts) % 2 == 1:
                    pts = []
                    for i in range(5, len(parts), 2):
                        if i + 1 < len(parts):
                            pts.append([float(parts[i]) * img_w, float(parts[i + 1]) * img_h])

                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                        annotations.append({
                            'label': label,
                            'class_id': class_id,
                            'bbox': [float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))],
                            'polygons': [pts]
                        })

        except Exception as e:
            print(f"[YOLOFormat] Failed to load: {e}")

        return {"annotations": annotations, "image_info": image_info, "categories": categories}

    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为YOLO格式

        Args:
            data: 统一格式的保存数据
            output_path: 输出TXT文件路径
            **kwargs: 额外参数
                - mode: 任务模式 ('det', 'seg', 'obb')
        """
        import cv2
        import numpy as np

        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [])

        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        mode = kwargs.get('mode', 'det')

        label_to_id = {cat["name"]: cat["id"] for cat in categories} if categories else {}

        with open(output_path, 'w', encoding='utf-8') as f:
            for inst in annotations:
                label = inst.get('label', '')
                class_id = label_to_id.get(label, inst.get('class_id', 0))

                if mode == "seg":
                    polys = inst.get('polygons') or ([inst.get('polygon')] if inst.get('polygon') else [])

                    if polys:
                        for poly in polys:
                            if len(poly) < 3:
                                continue

                            pts = np.array(poly, dtype=np.float32)
                            x, y, w, h = cv2.boundingRect(pts)

                            cx = (x + w / 2) / img_w
                            cy = (y + h / 2) / img_h
                            nw = w / img_w
                            nh = h / img_h

                            norm_pts = [f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in poly]

                            line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {' '.join(norm_pts)}\n"
                            f.write(line)
                    elif 'bbox' in inst:
                        x, y, bw, bh = inst['bbox']
                        cx = (x + bw / 2) / img_w
                        cy = (y + bh / 2) / img_h
                        nw = bw / img_w
                        nh = bh / img_h
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                elif mode == "obb":
                    polys = inst.get('polygons')

                    if not polys and 'bbox' in inst:
                        bx, by, bw, bh = inst['bbox']
                        polys = [[[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]]

                    if polys:
                        all_pts = []
                        for p in polys:
                            all_pts.extend(p)

                        if all_pts:
                            pts = np.array(all_pts, dtype=np.float32)
                            rect = cv2.minAreaRect(pts)
                            box = cv2.boxPoints(rect).tolist()

                            norm_pts = []
                            for i in range(4):
                                norm_pts.append(f"{box[i][0] / img_w:.6f}")
                                norm_pts.append(f"{box[i][1] / img_h:.6f}")
                            f.write(f"{class_id} {' '.join(norm_pts)}\n")

                else:
                    if 'bbox' in inst:
                        x, y, w, h = inst['bbox']
                        cx = (x + w / 2) / img_w
                        cy = (y + h / 2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if categories:
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, categories)
            cls._save_dataset_yaml(output_path, [cat["name"] for cat in sorted(categories, key=lambda x: x["id"])])

    @classmethod
    def _save_dataset_yaml(cls, output_path: str, class_names: List[str]) -> None:
        import yaml
        label_dir = os.path.dirname(output_path)
        dataset_root = os.path.dirname(label_dir)
        yaml_path = os.path.join(dataset_root, "dataset.yaml")

        content = [
            f"path: {dataset_root}/images",
            f"train: {dataset_root}/images/train",
            f"val: {dataset_root}/images/val",
            "",
            "names:"
        ]
        for i, name in enumerate(class_names):
            content.append(f"  {i}: {name}")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """更新YOLO目录中的类别名称（更新classes.txt）
        
        Args:
            dir_path: 包含 .txt 标注文件的目录（classes.txt 可以在当前目录或父目录）
        """
        try:
            classes_path = os.path.join(dir_path, "classes.txt")
            parent_classes_path = os.path.join(os.path.dirname(dir_path), "classes.txt")
            
            if os.path.exists(parent_classes_path) and not os.path.exists(classes_path):
                classes_path = parent_classes_path
            
            if not os.path.exists(classes_path):
                return 0

            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]

            if old_label not in class_names:
                return 0

            old_class_id = class_names.index(old_label)
            
            if new_label in class_names:
                new_class_id = class_names.index(new_label)
                class_names.pop(old_class_id)
                
                for filename in os.listdir(dir_path):
                    if filename.endswith('.txt') and filename != 'classes.txt':
                        filepath = os.path.join(dir_path, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            
                            new_lines = []
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                class_id = int(float(parts[0]))
                                if class_id == old_class_id:
                                    parts[0] = str(new_class_id)
                                    new_lines.append(' '.join(parts) + '\n')
                                elif class_id > old_class_id:
                                    parts[0] = str(class_id - 1)
                                    new_lines.append(' '.join(parts) + '\n')
                                else:
                                    new_lines.append(line)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(new_lines)
                        except Exception as e:
                            print(f"[YOLOFormat] Failed to update {filepath}: {e}")
            else:
                class_names[old_class_id] = new_label

            with open(classes_path, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")

            return 1
        except Exception as e:
            print(f"[YOLOFormat] Failed to update category: {e}")
            return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """从YOLO目录中删除指定类别的所有标注
        
        Args:
            dir_path: 包含 .txt 标注文件的目录（classes.txt 可以在当前目录或父目录）
        """
        try:
            classes_path = os.path.join(dir_path, "classes.txt")
            parent_classes_path = os.path.join(os.path.dirname(dir_path), "classes.txt")
            
            if os.path.exists(parent_classes_path) and not os.path.exists(classes_path):
                classes_path = parent_classes_path
            
            if not os.path.exists(classes_path):
                return 0

            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]

            if category_name not in class_names:
                return 0

            target_class_id = class_names.index(category_name)
            total_processed = 0

            for filename in os.listdir(dir_path):
                if filename.endswith('.txt') and filename != 'classes.txt':
                    filepath = os.path.join(dir_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        new_lines = []
                        removed_count = 0
                        id_remapped = False

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue

                            class_id = int(float(parts[0]))

                            if class_id == target_class_id:
                                removed_count += 1
                                continue

                            new_class_id = class_id
                            if class_id > target_class_id:
                                new_class_id = class_id - 1
                                id_remapped = True

                            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                            new_lines.append(new_line)

                        if removed_count > 0 or id_remapped:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.writelines(new_lines)
                            total_processed += 1
                    except Exception as e:
                        print(f"[YOLOFormat] Error processing {filepath}: {e}")

            class_names.remove(category_name)
            with open(classes_path, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")

            return total_processed
        except Exception as e:
            print(f"[YOLOFormat] Failed to remove category: {e}")
            return 0
