import os
import numpy as np
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo
from ..utils import imread_unicode, imwrite_unicode
from core.common.atomic_io import atomic_open, atomic_writer_path


class MaskFormat(AnnotationFormat):
    """Mask 图像格式处理器"""
    
    EXTENSIONS = ['.png', '.bmp', '.tiff', '.tif']
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in cls.EXTENSIONS
    
    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描Mask目录中所有标注文件，提取类别信息"""
        categories = []
        label_set = set()
        
        mask_subdirs = ['masks']
        
        for mask_subdir in mask_subdirs:
            masks_dir = os.path.join(dir_path, mask_subdir)
            if not os.path.exists(masks_dir):
                continue
            
            try:
                for filename in os.listdir(masks_dir):
                    if not filename.endswith(('.png', '.bmp', '.tiff', '.tif')):
                        continue
                    
                    filepath = os.path.join(masks_dir, filename)
                    try:
                        import cv2
                        mask = imread_unicode(filepath, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            continue
                        
                        unique_values = np.unique(mask)
                        unique_values = unique_values[unique_values > 0]
                        for class_id in unique_values:
                            label_set.add(int(class_id))
                    except:
                        continue
            except:
                continue
        
        max_id = max(label_set) if label_set else -1
        categories = [{"id": i, "name": f"class_{i}"} for i in range(max_id + 1)]
        return categories

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从Mask格式文件加载标注

        Args:
            file_path: Mask格式图像文件路径
            **kwargs: 额外参数
                - min_area: 最小区域面积

        Returns:
            AnnotationData: 统一格式的标注数据
        """
        annotations: List[Dict[str, Any]] = []
        image_info: Dict[str, Any] = {}
        categories: List[Dict[str, Any]] = []

        if not os.path.exists(file_path):
            return {"annotations": annotations, "image_info": image_info, "categories": categories}
        
        try:
            import cv2
            mask = imread_unicode(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return {"annotations": annotations, "image_info": image_info, "categories": categories}

            img_h, img_w = mask.shape[:2]
            image_info = {
                "filename": os.path.basename(file_path),
                "width": img_w,
                "height": img_h
            }
            
            min_area = kwargs.get('min_area', 100)
            
            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            categories = cls.get_or_create_categories(parent_dir)
            
            id_to_name = {cat["id"]: cat["name"] for cat in categories}

            unique_values = np.unique(mask)
            unique_values = unique_values[unique_values > 0]

            for class_id in unique_values:
                binary_mask = (mask == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) < 3:
                        continue
                    
                    pts = approx.reshape(-1, 2).tolist()
                    
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    
                    label = id_to_name.get(int(class_id), f"class_{int(class_id)}")
                    
                    annotations.append({
                        'label': label,
                        'class_id': int(class_id),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'polygons': [pts]
                    })

        except Exception as e:
            print(f"[MaskFormat] Failed to load: {e}")
        
        return {"annotations": annotations, "image_info": image_info, "categories": categories}
    
    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为彩色Mask格式

        Args:
            data: 统一格式的保存数据
            output_path: 输出图像文件路径
            **kwargs: 额外参数
        """
        import cv2
        from ..utils import generate_distinct_colors

        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [])

        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        num_classes = len(categories) if categories else 80

        label_to_id = {cat["name"]: cat["id"] for cat in categories} if categories else {}

        colored_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        class_colors = generate_distinct_colors(num_classes)

        for inst in annotations:
            label = inst.get('label', '')
            class_id = label_to_id.get(label, inst.get('class_id', 0))
            color = class_colors[class_id % num_classes]
            polys = inst.get('polygons') or ([inst.get('polygon')] if inst.get('polygon') else [])

            if polys:
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(colored_mask, [pts], color)
            elif 'bbox' in inst:
                x, y, w, h = [int(v) for v in inst['bbox']]
                cv2.rectangle(colored_mask, (x, y), (x + w, y + h), color, -1)

        with atomic_writer_path(output_path) as _tmp:
            if not imwrite_unicode(
                    _tmp, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR),
                    ext=os.path.splitext(output_path)[1]):
                raise IOError(f"写入掩码失败: {output_path}")

        if categories:
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, categories)

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """更新Mask目录中的类别名称（更新classes.txt）
        
        Args:
            dir_path: 包含 classes.txt 和 mask 图片文件的目录
        """
        try:
            import cv2
            import numpy as np
            
            classes_path = os.path.join(dir_path, "classes.txt")
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
                    if not filename.endswith(('.png', '.jpg', '.bmp')):
                        continue
                    filepath = os.path.join(dir_path, filename)
                    try:
                        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            continue
                        
                        mask[mask == old_class_id] = new_class_id
                        mask[mask > old_class_id] -= 1
                        
                        with atomic_writer_path(filepath) as _tmp:
                            if not imwrite_unicode(
                                    _tmp, mask,
                                    ext=os.path.splitext(filepath)[1]):
                                raise IOError(f"写入掩码失败: {filepath}")
                    except Exception as e:
                        print(f"[MaskFormat] Failed to update {filepath}: {e}")
            else:
                class_names[old_class_id] = new_label

            with atomic_open(classes_path, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")

            return 1
        except Exception as e:
            print(f"[MaskFormat] Failed to update category: {e}")
            return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """从Mask目录中删除指定类别的所有标注
        
        Args:
            dir_path: 包含 classes.txt 和 mask 图片文件的目录
        """
        try:
            import cv2
            import numpy as np

            classes_path = os.path.join(dir_path, "classes.txt")
            if not os.path.exists(classes_path):
                return 0

            with open(classes_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]

            if category_name not in class_names:
                return 0

            target_class_id = class_names.index(category_name)
            total_processed = 0

            for filename in os.listdir(dir_path):
                if not filename.endswith('.png'):
                    continue

                filepath = os.path.join(dir_path, filename)
                try:
                    mask = imread_unicode(filepath, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue

                    needs_update = np.any(mask == target_class_id)
                    if not needs_update:
                        for old_id in range(target_class_id + 1, len(class_names)):
                            if np.any(mask == old_id):
                                needs_update = True
                                break

                    if needs_update:
                        mask[mask == target_class_id] = 0
                        for old_id in range(target_class_id + 1, len(class_names)):
                            mask[mask == old_id] = old_id - 1
                        with atomic_writer_path(filepath) as _tmp:
                            if not imwrite_unicode(
                                    _tmp, mask,
                                    ext=os.path.splitext(filepath)[1]):
                                raise IOError(f"写入掩码失败: {filepath}")
                        total_processed += 1
                except Exception as e:
                    print(f"[MaskFormat] Error processing {filepath}: {e}")

            class_names.remove(category_name)
            with atomic_open(classes_path, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")

            return total_processed
        except Exception as e:
            print(f"[MaskFormat] Failed to remove category: {e}")
            return 0
