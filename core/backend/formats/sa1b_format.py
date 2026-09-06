import json
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo


class SA1BFormat(AnnotationFormat):
    """SA1B 格式处理器"""
    
    EXTENSIONS = ['.json']
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        if not file_path.endswith('.json'):
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return 'annotations' in data and 'image' in data
        except:
            return False
    
    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描SA1B目录中所有标注文件，提取类别信息"""
        return [{"id": 0, "name": "object"}]

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从SA1B格式文件加载标注

        Args:
            file_path: SA1B格式JSON文件路径
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
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_data = data.get('image', {})
            image_info = {
                "filename": img_data.get('file_name', os.path.basename(file_path).replace('.json', '.jpg')),
                "width": img_data.get('width', 1920),
                "height": img_data.get('height', 1080)
            }
            
            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            categories = cls.get_or_create_categories(parent_dir)
            id_to_name = {cat["id"]: cat["name"] for cat in categories}

            for ann in data.get('annotations', []):
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    x, y, w, h = [int(v) for v in bbox]
                    polygons = []
                    for seg in ann.get('segmentation', []):
                        if len(seg) >= 6:
                            poly = []
                            for i in range(0, len(seg), 2):
                                poly.append([seg[i], seg[i+1]])
                            polygons.append(poly)
                    
                    annotations.append({
                        'label': id_to_name.get(0, "object"),
                        'class_id': 0,
                        'bbox': [x, y, w, h],
                        'polygons': polygons,
                        'area': ann.get('area', 0),
                        'conf': ann.get('predicted_iou', 1.0)
                    })

        except Exception as e:
            print(f"[SA1BFormat] Failed to load: {e}")
        
        return {"annotations": annotations, "image_info": image_info, "categories": categories}
    
    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为SA1B格式

        Args:
            data: 统一格式的保存数据
            output_path: 输出JSON文件路径
            **kwargs: 额外参数
        """
        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [{"id": 0, "name": "object"}])

        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        image_filename = image_info.get("filename", os.path.basename(output_path).replace('.json', '.jpg'))
        
        sa1b_annotations = []
        for i, inst in enumerate(annotations):
            polys = inst.get('polygons') or []
            segmentation = []
            area = 0
            for poly in polys:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                segmentation.append(pts.reshape(-1).tolist())
                area += cv2.contourArea(pts)
            
            sa1b_annotations.append({
                "id": i, 
                "bbox": inst.get('bbox', [0, 0, 0, 0]), 
                "area": int(area),
                "segmentation": segmentation,
                "predicted_iou": inst.get('conf', 1.0)
            })
        
        sa1b_data = {
            "image": {
                "file_name": image_filename,
                "width": img_w,
                "height": img_h
            },
            "annotations": sa1b_annotations
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sa1b_data, f, indent=2, ensure_ascii=False)
        
        if categories:
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, categories)
    
    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """SA1B 格式不支持按类别删除"""
        return 0

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """SA1B 格式不支持类别名称更新"""
        return 0
