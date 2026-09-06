from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypedDict
import os


class ImageInfo(TypedDict, total=False):
    """图像信息"""
    filename: str
    width: int
    height: int
    id: int


class CategoryInfo(TypedDict, total=False):
    """类别信息"""
    id: int
    name: str


class AnnotationData(TypedDict, total=False):
    """标注数据统一格式"""
    annotations: List[Dict[str, Any]]
    image_info: ImageInfo
    categories: List[CategoryInfo]


class SaveData(TypedDict, total=False):
    """保存数据统一格式"""
    annotations: List[Dict[str, Any]]
    image_info: ImageInfo
    categories: List[CategoryInfo]


class AnnotationFormat(ABC):
    """标注格式基类，定义统一的 load 和 save 接口"""
    
    EXTENSIONS: List[str] = []
    CLASSES_FILE: str = "classes.txt"
    
    @classmethod
    @abstractmethod
    def load(cls, file_path: str, **kwargs) -> AnnotationData:
        """
        从文件加载标注
        
        Args:
            file_path: 标注文件路径
            **kwargs: 额外参数
                - image_path: 图像文件路径（用于获取图像尺寸和匹配）
        
        Returns:
            AnnotationData: 统一格式的标注数据
            {
                "annotations": [
                    {
                        "label": str,           # 类别名称
                        "class_id": int,        # 类别ID（可选）
                        "bbox": [x, y, w, h],   # 边界框（可选）
                        "polygons": [[x,y],...],# 多边形点列表（可选）
                        "mask": np.ndarray,     # 掩码（可选）
                        "conf": float,          # 置信度（可选）
                    },
                    ...
                ],
                "image_info": {
                    "filename": str,
                    "width": int,
                    "height": int,
                    "id": int  # 可选，用于COCO格式
                },
                "categories": [  # 可选
                    {"id": int, "name": str},
                    ...
                ]
            }
        """
        pass
    
    @classmethod
    @abstractmethod
    def save(cls, data: SaveData, output_path: str, **kwargs) -> None:
        """
        保存标注到文件
        
        Args:
            data: 统一格式的保存数据
            {
                "annotations": [...],
                "image_info": {
                    "filename": str,
                    "width": int,
                    "height": int
                },
                "categories": [
                    {"id": int, "name": str},
                    ...
                ]
            }
            output_path: 输出文件路径
            **kwargs: 额外参数
                - mode: 任务模式 ('det', 'seg', 'obb')
        """
        pass
    
    @classmethod
    @abstractmethod
    def scan_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """
        扫描目录中所有标注文件，提取类别信息
        
        Args:
            dir_path: 目录路径
            
        Returns:
            类别列表 [{"id": int, "name": str}, ...]
        """
        pass
    
    @classmethod
    def can_load(cls, file_path: str) -> bool:
        """检查是否可以加载该文件格式"""
        return False
    
    @classmethod
    def get_extensions(cls) -> List[str]:
        """获取支持的文件扩展名列表"""
        return cls.EXTENSIONS
    
    @classmethod
    def load_classes_txt(cls, dir_path: str) -> List[CategoryInfo]:
        """
        从目录中加载 classes.txt 文件
        
        Args:
            dir_path: 目录路径
            
        Returns:
            类别列表 [{"id": int, "name": str}, ...]，如果文件不存在则返回空列表
        """
        classes_path = os.path.join(dir_path, cls.CLASSES_FILE)
        if not os.path.exists(classes_path):
            return []
        
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            categories = []
            for i, line in enumerate(lines):
                name = line.strip()
                if name:
                    categories.append({"id": i, "name": name})
            return categories
        except Exception as e:
            print(f"[{cls.__name__}] Failed to load classes.txt: {e}")
            return []
    
    @classmethod
    def save_classes_txt(cls, dir_path: str, categories: List[CategoryInfo]) -> None:
        """
        保存类别到 classes.txt 文件
        
        Args:
            dir_path: 目录路径
            categories: 类别列表 [{"id": int, "name": str}, ...]
        """
        if not categories:
            return
        
        classes_path = os.path.join(dir_path, cls.CLASSES_FILE)
        try:
            os.makedirs(dir_path, exist_ok=True)
            sorted_categories = sorted(categories, key=lambda x: x.get("id", 0))
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cat in sorted_categories:
                    f.write(f"{cat.get('name', '')}\n")
        except Exception as e:
            print(f"[{cls.__name__}] Failed to save classes.txt: {e}")
    
    @classmethod
    def get_or_create_categories(cls, dir_path: str) -> List[CategoryInfo]:
        """
        获取类别列表：优先从 classes.txt 加载，如果不存在则扫描标注文件并生成
        
        Args:
            dir_path: 目录路径
            
        Returns:
            类别列表 [{"id": int, "name": str}, ...]
        """
        categories = cls.load_classes_txt(dir_path)
        if categories:
            return categories
        
        categories = cls.scan_categories(dir_path)
        if categories:
            cls.save_classes_txt(dir_path, categories)
        return categories
    
    @classmethod
    def update_category(cls, dir_path: str, old_label: str, new_label: str) -> int:
        """
        更新目录中所有标注文件的类别名称

        Args:
            dir_path: 目录路径
            old_label: 旧类别名称
            new_label: 新类别名称

        Returns:
            处理的文件数量
        """
        return 0

    @classmethod
    def remove_category(cls, dir_path: str, category_name: str) -> int:
        """
        从目录中删除指定类别的所有标注

        Args:
            dir_path: 目录路径
            category_name: 要删除的类别名称

        Returns:
            处理的文件数量
        """
        return 0
