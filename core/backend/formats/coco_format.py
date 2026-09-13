import json
import os
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo
from core.common.atomic_io import atomic_open
from core.common.atomic_io import atomic_open


class COCOFormat(AnnotationFormat):
    """COCO 格式处理器"""

    EXTENSIONS = ['.json']

    @classmethod
    def can_load(cls, file_path: str) -> bool:
        """检查是否为COCO格式文件"""
        if os.path.splitext(file_path)[1].lower() != '.json':
            return False

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return 'images' in data and 'annotations' in data and 'categories' in data
        except:
            return False

    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描COCO目录中的标注文件，提取类别信息"""
        categories = []
        
        coco_path = os.path.join(dir_path, "coco_annotations", "_annotations.coco.json")
        if not os.path.exists(coco_path):
            coco_path = os.path.join(dir_path, "_annotations.coco.json")
        
        if not os.path.exists(coco_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.json'):
                    coco_path = os.path.join(dir_path, filename)
                    break
        
        if not os.path.exists(coco_path):
            return categories
        
        try:
            with open(coco_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            categories = data.get('categories', [])
        except Exception as e:
            print(f"[COCOFormat] Failed to scan categories: {e}")
        
        return categories

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从COCO格式文件加载标注

        Args:
            file_path: COCO格式JSON文件路径
            **kwargs: 额外参数
                - image_path: 图像文件路径（用于匹配）

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

            file_categories = data.get('categories', [])
            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            
            if file_categories:
                categories = file_categories
                cat_map = {cat['id']: cat.get('name', f"class_{cat['id']}") for cat in categories}
            else:
                categories = cls.get_or_create_categories(parent_dir)
                cat_map = {cat['id']: cat.get('name', f"class_{cat['id']}") for cat in categories}

            image_id_to_info = {}
            if 'images' in data:
                for img in data['images']:
                    image_id_to_info[img['id']] = {
                        'filename': img.get('file_name', ''),
                        'width': img.get('width', 0),
                        'height': img.get('height', 0),
                        'id': img['id']
                    }

            image_path = kwargs.get('image_path')
            target_image_filename = os.path.basename(image_path) if image_path else None
            target_image_id = None

            if target_image_filename:
                for img_id, info in image_id_to_info.items():
                    if info['filename'] == target_image_filename:
                        target_image_id = img_id
                        image_info = info
                        break

                if target_image_id is None:
                    return {"annotations": annotations, "image_info": image_info, "categories": categories}

            for ann in data.get('annotations', []):
                if target_image_id is not None and ann.get('image_id') != target_image_id:
                    continue

                cid = ann.get('category_id', 0)
                label = cat_map.get(cid, f"class_{cid}")

                instance = {
                    'label': label,
                    'class_id': cid,
                    'image_id': ann.get('image_id'),
                    'annotation_id': ann.get('id'),
                }

                if 'bbox' in ann:
                    bbox = ann['bbox']
                    if len(bbox) == 4:
                        x, y, w, h = [float(v) for v in bbox]
                        instance['bbox'] = [x, y, w, h]
                        instance['polygons'] = [[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]]

                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    polygons = []
                    if isinstance(seg, list):
                        if len(seg) > 0 and isinstance(seg[0], list):
                            for poly in seg:
                                if len(poly) >= 6:
                                    points = []
                                    for i in range(0, len(poly), 2):
                                        points.append([float(poly[i]), float(poly[i+1])])
                                    polygons.append(points)
                        elif len(seg) >= 6:
                            points = []
                            for i in range(0, len(seg), 2):
                                points.append([float(seg[i]), float(seg[i+1])])
                            polygons.append(points)

                    if polygons:
                        instance['polygons'] = polygons
                        if 'bbox' not in instance:
                            all_x = [p[0] for poly in polygons for p in poly]
                            all_y = [p[1] for poly in polygons for p in poly]
                            x_min, x_max = min(all_x), max(all_x)
                            y_min, y_max = min(all_y), max(all_y)
                            instance['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]

                if 'area' in ann:
                    instance['area'] = ann['area']

                if 'iscrowd' in ann:
                    instance['iscrowd'] = ann['iscrowd']

                annotations.append(instance)

        except Exception as e:
            print(f"[COCOFormat] Failed to load: {e}")

        return {"annotations": annotations, "image_info": image_info, "categories": categories}

    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为COCO格式（所有标注在一个文件中）
        
        如果图像ID已存在，则更新该图像的标注；否则添加新图像和标注

        Args:
            data: 统一格式的保存数据
            output_path: 输出JSON文件路径
            **kwargs: 额外参数
                - coco_cache: COCO数据缓存字典
                - coco_cache_path: 缓存对应的文件路径
        """
        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [])

        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        image_filename = image_info.get("filename", os.path.basename(output_path).replace('.json', '.jpg'))

        unique_labels = []
        for inst in annotations:
            label = inst.get('label', 'unknown')
            if label not in unique_labels:
                unique_labels.append(label)

        coco_cache = kwargs.get('coco_cache')
        coco_cache_path = kwargs.get('coco_cache_path')
        
        use_cache = (coco_cache is not None and 
                    coco_cache_path is not None and 
                    coco_cache_path == output_path and 
                    'data' in coco_cache)
        
        if use_cache:
            coco_data = coco_cache['data']
            existing_label_to_id = {cat['name']: cat['id'] for cat in coco_data.get('categories', [])}
            max_cat_id = max([cat['id'] for cat in coco_data.get('categories', [])] + [0])
            
            label_to_cat_id = {}
            new_categories = []
            for label in unique_labels:
                if label in existing_label_to_id:
                    label_to_cat_id[label] = existing_label_to_id[label]
                else:
                    max_cat_id += 1
                    label_to_cat_id[label] = max_cat_id
                    new_categories.append({'id': max_cat_id, 'name': label})
            
            coco_data['categories'].extend(new_categories)

            existing_img = None
            image_id = None
            for img in coco_data.get('images', []):
                if img.get('file_name') == image_filename:
                    existing_img = img
                    image_id = img['id']
                    break
            
            if existing_img:
                existing_img['width'] = img_w
                existing_img['height'] = img_h
                coco_data['annotations'] = [
                    ann for ann in coco_data.get('annotations', [])
                    if ann.get('image_id') != image_id
                ]
            else:
                max_img_id = max([img['id'] for img in coco_data.get('images', [])] + [0])
                image_id = max_img_id + 1
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': image_filename,
                    'width': img_w,
                    'height': img_h
                })

            max_ann_id = 0
            for ann in coco_data.get('annotations', []):
                max_ann_id = max(max_ann_id, ann.get('id', 0))
        elif os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8-sig') as f:
                raw = f.read().strip()
                if not raw:
                    # 空文件，当作不存在处理
                    coco_data = {
                        'images': [], 'annotations': [], 'categories': []
                    }
                else:
                    try:
                        coco_data = json.loads(raw)
                    except json.JSONDecodeError:
                        # 损坏的 JSON 文件，重新初始化
                        coco_data = {
                            'images': [], 'annotations': [], 'categories': []
                        }
            
            existing_label_to_id = {cat['name']: cat['id'] for cat in coco_data.get('categories', [])}
            max_cat_id = max([cat['id'] for cat in coco_data.get('categories', [])] + [0])
            
            label_to_cat_id = {}
            new_categories = []
            for label in unique_labels:
                if label in existing_label_to_id:
                    label_to_cat_id[label] = existing_label_to_id[label]
                else:
                    max_cat_id += 1
                    label_to_cat_id[label] = max_cat_id
                    new_categories.append({'id': max_cat_id, 'name': label})
            
            coco_data['categories'].extend(new_categories)

            existing_img = None
            image_id = None
            for img in coco_data.get('images', []):
                if img.get('file_name') == image_filename:
                    existing_img = img
                    image_id = img['id']
                    break
            
            if existing_img:
                existing_img['width'] = img_w
                existing_img['height'] = img_h
                coco_data['annotations'] = [
                    ann for ann in coco_data.get('annotations', [])
                    if ann.get('image_id') != image_id
                ]
            else:
                max_img_id = max([img['id'] for img in coco_data.get('images', [])] + [0])
                image_id = max_img_id + 1
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': image_filename,
                    'width': img_w,
                    'height': img_h
                })

            max_ann_id = 0
            for ann in coco_data.get('annotations', []):
                max_ann_id = max(max_ann_id, ann.get('id', 0))
        else:
            label_to_cat_id = {label: i + 1 for i, label in enumerate(unique_labels)}
            new_categories = [{'id': i + 1, 'name': label} for i, label in enumerate(unique_labels)]
            
            image_id = 1
            
            coco_data = {
                'info': {
                    'description': 'COCO Dataset',
                    'version': '1.0',
                    'year': 2024
                },
                'images': [{
                    'id': image_id,
                    'file_name': image_filename,
                    'width': img_w,
                    'height': img_h
                }],
                'annotations': [],
                'categories': new_categories
            }
            max_ann_id = 0

        for i, inst in enumerate(annotations):
            ann_id = max_ann_id + i + 1
            label = inst.get('label', 'unknown')
            cid = label_to_cat_id.get(label, 1)

            ann = {
                'id': ann_id,
                'image_id': image_id,
                'category_id': cid,
            }

            if 'bbox' in inst:
                bbox = inst['bbox']
                ann['bbox'] = [float(v) for v in bbox]
                ann['area'] = float(bbox[2] * bbox[3])

            if 'polygons' in inst and inst['polygons']:
                polygons = inst['polygons']
                segmentations = []
                for poly in polygons:
                    if len(poly) >= 3:
                        seg = []
                        for p in poly:
                            seg.extend([float(p[0]), float(p[1])])
                        segmentations.append(seg)

                if segmentations:
                    ann['segmentation'] = segmentations
                    if 'bbox' not in ann:
                        all_x = [p[0] for poly in polygons for p in poly]
                        all_y = [p[1] for poly in polygons for p in poly]
                        x_min, x_max = min(all_x), max(all_x)
                        y_min, y_max = min(all_y), max(all_y)
                        w, h = x_max - x_min, y_max - y_min
                        ann['bbox'] = [x_min, y_min, w, h]
                        ann['area'] = w * h

            ann['iscrowd'] = 0
            coco_data['annotations'].append(ann)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with atomic_open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        if coco_cache is not None:
            coco_cache['data'] = coco_data

        if coco_data.get('categories'):
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, coco_data['categories'])

    @classmethod
    def save_dataset(cls, annotations_by_image: Dict[str, List[Dict[str, Any]]],
                     output_path: str,
                     categories: List[Dict[str, Any]],
                     image_info: Dict[str, Dict[str, Any]],
                     info: Optional[Dict[str, Any]] = None) -> None:
        """
        保存整个数据集为COCO格式

        Args:
            annotations_by_image: 按图像分组的标注 {image_filename: [instances, ...]}
            output_path: 输出JSON文件路径
            categories: 类别列表 [{'id': int, 'name': str}, ...]
            image_info: 图像信息 {image_filename: {'width': int, 'height': int, 'id': int}}
            info: 数据集信息 {'description': str, 'version': str, 'year': int}
        """
        if info is None:
            info = {
                'description': 'COCO Dataset',
                'version': '1.0',
                'year': 2024
            }
        
        coco_data = {
            'info': info,
            'images': [],
            'annotations': [],
            'categories': categories
        }

        ann_id = 1
        for image_filename, instances in annotations_by_image.items():
            img_info = image_info.get(image_filename, {})
            img_id = img_info.get('id', len(coco_data['images']) + 1)
            img_w = img_info.get('width', 1920)
            img_h = img_info.get('height', 1080)

            coco_data['images'].append({
                'id': img_id,
                'file_name': image_filename,
                'width': img_w,
                'height': img_h
            })

            for inst in instances:
                label = inst.get('label', '')
                cid = 0
                for cat in categories:
                    if cat.get('name') == label:
                        cid = cat.get('id', 0)
                        break
                
                ann = {
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cid,
                    'iscrowd': 0
                }

                if 'bbox' in inst:
                    bbox = inst['bbox']
                    ann['bbox'] = [float(v) for v in bbox]
                    ann['area'] = float(bbox[2] * bbox[3])

                if 'polygons' in inst and inst['polygons']:
                    polygons = inst['polygons']
                    segmentations = []
                    for poly in polygons:
                        if len(poly) >= 3:
                            seg = []
                            for p in poly:
                                seg.extend([float(p[0]), float(p[1])])
                            segmentations.append(seg)
                    if segmentations:
                        ann['segmentation'] = segmentations

                coco_data['annotations'].append(ann)
                ann_id += 1

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with atomic_open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """更新COCO格式文件中的类别名称
        
        Args:
            dir_path: 包含 _annotations.coco.json 文件的目录
        """
        try:
            coco_path = os.path.join(dir_path, "_annotations.coco.json")
            if not os.path.exists(coco_path):
                return 0
            
            with open(coco_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            if 'categories' not in data:
                return 0

            old_cat_id = None
            new_cat_id = None
            for cat in data.get('categories', []):
                if cat.get('name') == old_label:
                    old_cat_id = cat.get('id')
                if cat.get('name') == new_label:
                    new_cat_id = cat.get('id')

            if old_cat_id is None:
                return 0

            changed = False
            
            if new_cat_id is not None:
                data['categories'] = [cat for cat in data.get('categories', []) 
                                      if cat.get('name') != old_label]
                
                for ann in data.get('annotations', []):
                    if ann.get('category_id') == old_cat_id:
                        ann['category_id'] = new_cat_id
                        changed = True
            else:
                for cat in data.get('categories', []):
                    if cat.get('name') == old_label:
                        cat['name'] = new_label
                        changed = True

            if changed:
                with atomic_open(coco_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return 1

            return 0
        except Exception as e:
            print(f"[COCOFormat] Failed to update category: {e}")
            return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """从COCO格式文件中删除指定类别
        
        Args:
            dir_path: 包含 _annotations.coco.json 文件的目录
        """
        try:
            coco_path = os.path.join(dir_path, "_annotations.coco.json")
            if not os.path.exists(coco_path):
                return 0
            
            with open(coco_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            if 'categories' not in data or 'annotations' not in data:
                return 0

            cat_ids_to_remove = []
            new_categories = []
            for cat in data.get('categories', []):
                if cat.get('name') == category_name:
                    cat_ids_to_remove.append(cat['id'])
                else:
                    new_categories.append(cat)

            if not cat_ids_to_remove:
                return 0

            data['categories'] = new_categories

            original_count = len(data.get('annotations', []))
            data['annotations'] = [
                ann for ann in data['annotations']
                if ann.get('category_id') not in cat_ids_to_remove
            ]
            new_count = len(data['annotations'])

            if new_count < original_count:
                with atomic_open(coco_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return 1

            return 0
        except Exception as e:
            print(f"[COCOFormat] Failed to remove category: {e}")
            return 0
