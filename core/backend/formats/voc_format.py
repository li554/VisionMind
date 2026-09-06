import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from .base import AnnotationFormat, AnnotationData, SaveData, CategoryInfo


class VOCFormat(AnnotationFormat):
    """VOC XML 格式处理器"""
    
    EXTENSIONS = ['.xml']
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() == '.xml'
    
    @classmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """扫描VOC目录中所有标注文件，提取类别信息"""
        categories = []
        label_set = set()
        
        voc_dirs = [
            os.path.join(dir_path, "voc_annotations"),
            os.path.join(dir_path, "annotations")
        ]
        
        for voc_dir in voc_dirs:
            if not os.path.exists(voc_dir):
                continue
            
            try:
                for filename in os.listdir(voc_dir):
                    if not filename.endswith('.xml'):
                        continue
                    
                    filepath = os.path.join(voc_dir, filename)
                    try:
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        
                        for obj in root.findall('object'):
                            name_elem = obj.find('name')
                            if name_elem is not None and name_elem.text:
                                label_set.add(name_elem.text)
                    except:
                        continue
            except:
                continue
        
        categories = [{"id": i, "name": label} for i, label in enumerate(sorted(label_set))]
        return categories

    @classmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从VOC格式文件加载标注

        Args:
            file_path: VOC格式XML文件路径
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
            tree = ET.parse(file_path)
            root = tree.getroot()

            filename_elem = root.find('filename')
            image_filename = filename_elem.text if filename_elem is not None else os.path.basename(file_path).replace('.xml', '.jpg')

            size_elem = root.find('size')
            img_w, img_h = 1920, 1080
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                if width_elem is not None:
                    img_w = int(width_elem.text)
                if height_elem is not None:
                    img_h = int(height_elem.text)

            image_info = {
                "filename": image_filename,
                "width": img_w,
                "height": img_h
            }

            dir_path = os.path.dirname(file_path)
            parent_dir = os.path.dirname(dir_path)
            categories = cls.get_or_create_categories(parent_dir)
            
            name_to_id = {cat["name"]: cat["id"] for cat in categories}

            for obj in root.findall('object'):
                name_elem = obj.find('name')
                label = name_elem.text if name_elem is not None else "unknown"
                
                bbox_elem = obj.find('bndbox')
                if bbox_elem is None:
                    continue
                
                xmin = bbox_elem.find('xmin')
                ymin = bbox_elem.find('ymin')
                xmax = bbox_elem.find('xmax')
                ymax = bbox_elem.find('ymax')
                
                if all(elem is not None for elem in [xmin, ymin, xmax, ymax]):
                    x1 = float(xmin.text)
                    y1 = float(ymin.text)
                    x2 = float(xmax.text)
                    y2 = float(ymax.text)
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    
                    annotations.append({
                        'label': label,
                        'class_id': name_to_id.get(label, 0),
                        'bbox': bbox,
                        'polygons': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
                    })

        except Exception as e:
            print(f"[VOCFormat] Failed to load: {e}")
        
        return {"annotations": annotations, "image_info": image_info, "categories": categories}
    
    @classmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注为VOC格式

        Args:
            data: 统一格式的保存数据
            output_path: 输出XML文件路径
            **kwargs: 额外参数
        """
        annotations = data.get("annotations", [])
        image_info = data.get("image_info", {})
        categories = data.get("categories", [])

        image_filename = image_info.get("filename", os.path.basename(output_path).replace('.xml', '.jpg'))
        img_w = image_info.get("width", 1920)
        img_h = image_info.get("height", 1080)
        
        annotation = ET.Element('annotation')
        
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'images'
        
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_filename
        
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_w)
        height = ET.SubElement(size, 'height')
        height.text = str(img_h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        for inst in annotations:
            label = inst.get('label') or f"class_{inst.get('class_id', 0)}"
            bbox = inst.get('bbox')
            
            if bbox:
                x, y, w, h = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                obj = ET.SubElement(annotation, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = label
                
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(x1)
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(y1)
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(x2)
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(y2)
        
        tree = ET.ElementTree(annotation)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

        if categories:
            dir_path = os.path.dirname(output_path)
            parent_dir = os.path.dirname(dir_path)
            cls.save_classes_txt(parent_dir, categories)

    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """更新VOC目录中的所有标注文件的类别名称
        
        Args:
            dir_path: 包含 .xml 标注文件的目录
        """
        try:
            if not os.path.exists(dir_path):
                return 0

            total_processed = 0
            for filename in os.listdir(dir_path):
                if not filename.endswith('.xml'):
                    continue

                filepath = os.path.join(dir_path, filename)
                tree = ET.parse(filepath)
                root = tree.getroot()

                changed = False
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text == old_label:
                        name_elem.text = new_label
                        changed = True

                if changed:
                    tree.write(filepath, encoding='utf-8', xml_declaration=True)
                    total_processed += 1

            return total_processed
        except Exception as e:
            print(f"[VOCFormat] Failed to update category: {e}")
            return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """从VOC目录中删除指定类别的所有标注
        
        Args:
            dir_path: 包含 .xml 标注文件的目录
        """
        try:
            if not os.path.exists(dir_path):
                return 0

            total_processed = 0
            for filename in os.listdir(dir_path):
                if not filename.endswith('.xml'):
                    continue

                filepath = os.path.join(dir_path, filename)
                tree = ET.parse(filepath)
                root = tree.getroot()

                objects_to_remove = []
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text == category_name:
                        objects_to_remove.append(obj)

                removed_count = len(objects_to_remove)
                for obj in objects_to_remove:
                    root.remove(obj)

                if removed_count > 0:
                    tree.write(filepath, encoding='utf-8', xml_declaration=True)
                    total_processed += 1

            return total_processed
        except Exception as e:
            print(f"[VOCFormat] Failed to remove category: {e}")
            return 0
