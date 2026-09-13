import json
import os
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo
from core.common.atomic_io import atomic_open
from core.common.atomic_io import atomic_open


class LabelMeFormat(AnnotationFormat):
    """LabelMe 格式处理器"""

    EXTENSIONS = ['.json']

    @classmethod
    def can_load(cls, file_path: str) -> bool:
        """检查是否为LabelMe格式文件"""
        if os.path.splitext(file_path)[1].lower() != '.json':
            return False

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return 'shapes' in data and 'annotations' not in data
        except:
            return False

    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描LabelMe目录中所有标注文件，提取类别信息"""
        categories = []
        label_set = set()
        
        ann_dir = os.path.join(dir_path, "annotations")
        if not os.path.exists(ann_dir):
            ann_dir = dir_path
        
        try:
            for filename in os.listdir(ann_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(ann_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                    
                    if 'annotations' in data and 'categories' in data:
                        continue
                    
                    for shape in data.get('shapes', []):
                        label = shape.get('label')
                        if label:
                            label_set.add(label)
                except:
                    continue
            
            categories = [{"id": i, "name": label} for i, label in enumerate(sorted(label_set))]
        except Exception as e:
            print(f"[LabelMeFormat] Failed to scan categories: {e}")
        
        return categories

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从LabelMe格式文件加载标注

        Args:
            file_path: LabelMe格式JSON文件路径
            **kwargs: 额外参数

        Returns:
            AnnotationData: 统一格式的标注数据
        """
        annotations: List[Dict[str, Any]] = []
        image_info: Dict[str, Any] = {}
        categories: List[Dict[str, Any]] = []

        if not os.path.exists(file_path):
            return {"annotations": annotations, "image_info": image_info, "categories": categories}

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            image_info = {
                "filename": data.get("imagePath", os.path.basename(file_path).replace('.json', '.jpg')),
                "width": data.get("imageWidth", 0),
                "height": data.get("imageHeight", 0)
            }

            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            categories = cls.get_or_create_categories(parent_dir)
            
            name_to_id = {cat["name"]: cat["id"] for cat in categories}

            for shape in data.get('shapes', []):
                label = shape.get('label')
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', 'polygon')

                if shape_type == 'rectangle':
                    # rectangle类型：2个点 [[x1,y1],[x2,y2]]
                    if len(points) == 2:
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                    elif len(points) == 4:
                        # 兼容4个点的rectangle
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        continue
                    annotations.append({
                        'label': label,
                        'class_id': name_to_id.get(label, 0),
                        'bbox': bbox,
                        'polygons': [],
                        'shape_type': 'rectangle'
                    })
                elif shape_type == 'polygon':
                    # polygon类型：多边形点列表（包括旋转矩形4个点）
                    poly = [[p[0], p[1]] for p in points]
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    ann_dict = {
                        'label': label,
                        'class_id': name_to_id.get(label, 0),
                        'bbox': bbox,
                        'polygons': [poly],
                        'shape_type': 'polygon'
                    }
                    # 旋转矩形：4个点且不是轴对齐的，标记为rotation
                    if len(poly) == 4:
                        # 检查是否为轴对齐矩形（兼容旧数据中polygon存储的普通矩形）
                        xs_sorted = sorted(xs)
                        ys_sorted = sorted(ys)
                        is_axis_aligned = (abs(xs[1] - xs[0]) < 1 or abs(ys[1] - ys[0]) < 1 or
                                           (abs(poly[0][0] - poly[1][0]) < 1 and abs(poly[1][1] - poly[2][1]) < 1) or
                                           (abs(poly[0][1] - poly[1][1]) < 1 and abs(poly[1][0] - poly[2][0]) < 1))
                        if not is_axis_aligned:
                            ann_dict['shape_type'] = 'rotation'
                    annotations.append(ann_dict)

        except Exception as e:
            print(f"[LabelMeFormat] Failed to load: {e}")

        return {"annotations": annotations, "image_info": image_info, "categories": categories}

    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为LabelMe格式

        Args:
            data: 统一格式的保存数据
            output_path: 输出JSON文件路径
            **kwargs: 额外参数
        """
        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [])

        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        image_filename = image_info.get("filename", os.path.basename(output_path).replace('.json', '.jpg'))

        label_to_id = {cat["name"]: cat["id"] for cat in categories} if categories else {}

        shapes = []
        for inst in annotations:
            label = inst.get('label') or f"class_{inst.get('class_id', 0)}"
            polys = inst.get('polygons') or ([inst.get('polygon')] if inst.get('polygon') else [])
            inst_shape_type = inst.get('shape_type', '')

            # rectangle类型即使有polygons也使用bbox保存为rectangle格式（2个点）
            if polys and inst_shape_type == 'rectangle' and 'bbox' in inst:
                x, y, w, h = inst['bbox']
                shapes.append({
                    "label": label,
                    "points": [[float(x), float(y)], [float(x + w), float(y + h)]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })
            elif polys:
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    points = [[float(p[0]), float(p[1])] for p in poly]
                    # 根据shape_type决定保存类型
                    # rotation → polygon（4个旋转点）
                    # polygon → polygon
                    # 其他（无polygons的bbox走elif分支）
                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
            elif 'bbox' in inst:
                x, y, w, h = inst['bbox']
                # 有shape_type='rotation'标记且无polygons时，从task_mode推导
                if inst_shape_type == 'rotation' and 'task_mode' in kwargs and kwargs['task_mode'] == 'obb':
                    # OBB模式下bbox是轴对齐的，不保存为rotation
                    # 这种情况理论上不应出现（obb应有polygons）
                    points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
                else:
                    # 普通矩形框 → rectangle类型，2个点
                    shapes.append({
                        "label": label,
                        "points": [[float(x), float(y)], [float(x + w), float(y + h)]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })

        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with atomic_open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        if categories:
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, categories)

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """更新LabelMe目录中的所有标注文件的类别名称
        
        Args:
            dir_path: 标注文件所在目录（直接包含JSON文件的目录）
        """
        try:
            if not os.path.exists(dir_path):
                return 0
            
            total_processed = 0
            for filename in os.listdir(dir_path):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)

                    if 'annotations' in data:
                        continue

                    changed = False
                    for shape in data.get('shapes', []):
                        if shape.get('label') == old_label:
                            shape['label'] = new_label
                            changed = True

                    if changed:
                        with atomic_open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        total_processed += 1
                except Exception as e:
                    pass

            return total_processed
        except Exception as e:
            print(f"[LabelMeFormat] Failed to update category: {e}")
            return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """从LabelMe目录中删除指定类别的所有标注
        
        Args:
            dir_path: 标注文件所在目录（直接包含JSON文件的目录）
        """
        try:
            if not os.path.exists(dir_path):
                return 0
            
            total_processed = 0
            for filename in os.listdir(dir_path):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(dir_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)

                    if 'annotations' in data and 'categories' in data:
                        continue

                    original_count = len(data.get('shapes', []))
                    data['shapes'] = [s for s in data.get('shapes', []) if s.get('label') != category_name]
                    new_count = len(data['shapes'])

                    if new_count < original_count:
                        with atomic_open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        total_processed += 1
                except Exception as e:
                    pass

            return total_processed
        except Exception as e:
            print(f"[LabelMeFormat] Failed to remove category: {e}")
            return 0
