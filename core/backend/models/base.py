import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Optional

class ModelInterface(ABC):
    """统一的模型接口"""
    model_type: str = ""
    type: str = "example"
    img_size: int = 640

    @abstractmethod
    def load(self, weight_path: str) -> bool:
        pass

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=640, **kwargs) -> List[Any]:
        pass

    @abstractmethod
    def release(self):
        """释放模型资源"""
        pass

    @abstractmethod
    def reset_cache(self):
        """重置模型推理缓存（如图像编码特征）"""
        pass

class MockBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32) if not isinstance(xyxy, torch.Tensor) else xyxy
        self.conf = torch.tensor(conf, dtype=torch.float32) if not isinstance(conf, torch.Tensor) else conf
        self.cls = torch.tensor(cls, dtype=torch.float32) if not isinstance(cls, torch.Tensor) else cls
        
        # Calculate xywh
        if len(self.xyxy) > 0:
            x1, y1, x2, y2 = self.xyxy.unbind(-1)
            self.xywh = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
        else:
            self.xywh = torch.empty((0, 4))
    
    def __len__(self):
        return len(self.xyxy)
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.xyxy.numpy()

class MockMasks:
    def __init__(self, data, scores=None):
        if data is None:
            self.data = torch.empty((0, 0, 0))
        else:
            self.data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        
        self.scores = torch.from_numpy(scores) if scores is not None else (
            torch.ones(len(self.data)) if len(self.data) > 0 else torch.empty(0)
        )
    
    def __len__(self):
        return len(self.data)

class MockResults:
    def __init__(self, boxes, masks, orig_img, names=None, polygons=None):
        self.boxes = boxes
        self.masks = masks
        self.orig_img = orig_img
        self.names = names or {0: 'object'}
        self.polygons = polygons  # 可选的多边形数据

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0
