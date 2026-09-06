import numpy as np
import torch
import gc
import cv2
from ultralytics.models.sam import Predictor as SAMPredictor
from .base import ModelInterface
from .registry import register_model
from ..utils import _cleanup_runs

@register_model("sam_b", img_size=640, weight_path_key="sam_b_path", category="sam", role="interactive")
@register_model("sam_l", img_size=640, weight_path_key="sam_l_path", category="sam", role="interactive")
@register_model("sam_h", img_size=640, weight_path_key="sam_vit_h_path", category="sam", role="interactive")
class SAMWrapper(ModelInterface):
    """Ultralytics SAM 模型包装器"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.predictor = None
        self.model = None # Keep for compatibility if needed, but mainly use predictor
        self._last_image = None # Cache for last processed image

    def reset_cache(self):
        """重置模型缓存"""
        self._last_image = None
        if self.predictor:
            # Clear internal features if accessible
            if hasattr(self.predictor, 'features'):
                self.predictor.features = None
            if hasattr(self.predictor, 'is_image_set'):
                self.predictor.is_image_set = False
                
    def _set_image_if_needed(self, image: np.ndarray):
        """Internal helper to set image only if changed"""
        if self.predictor is None:
             return
             
        # Check if image is the same object or content
        if self._last_image is not None:
            # Fast check: object identity
            if image is self._last_image:
                return
            # Content check (slower but necessary if new array passed)
            if image.shape == self._last_image.shape and np.array_equal(image, self._last_image):
                return
                
        # If different, reset old cache first
        self.reset_cache()
        
        # Set new image
        self.predictor.set_image(image)
        self._last_image = image

    def load(self, weight_path: str) -> bool:
        try:
            # Create SAMPredictor with overrides
            overrides = dict(
                conf=0.25, 
                task="segment", 
                mode="predict", 
                imgsz=640, 
                model=weight_path
            )
            self.predictor = SAMPredictor(overrides=overrides)
            return True
        except Exception as e:
            print(f"[SAMWrapper] 加载 {self.model_type} 失败: {e}")
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
            
        # Set image to predictor (computes features)
        self._set_image_if_needed(image)
        
        # Return features
        features = getattr(self.predictor, 'features', None)
        if features is None:
             raise RuntimeError("Features not found in predictor after set_image")
             
        # Handle dict features (e.g. from some Ultralytics versions or specific models)
        if isinstance(features, dict):
            # Prioritize 'features' key if exists
            if 'features' in features:
                features = features['features']
            # Or assume the first value is the feature map
            elif len(features) > 0:
                features = list(features.values())[0]
                
        # Handle list/tuple features (e.g. multi-scale)
        if isinstance(features, (list, tuple)):
            if len(features) > 0:
                features = features[0]
                
        if not isinstance(features, torch.Tensor):
             # Try to find a tensor in the structure if still not found
             if isinstance(features, dict):
                 for v in features.values():
                     if isinstance(v, torch.Tensor):
                         features = v
                         break
                         
        if not isinstance(features, torch.Tensor):
             raise RuntimeError(f"Could not extract tensor features from predictor. Got type: {type(features)}")
             
        return features

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=640, **kwargs):
        if self.predictor is None:
            return []
        
        if img_size != getattr(self, '_last_img_size', 640):
            self._last_img_size = img_size
            if hasattr(self.predictor, 'args'):
                self.predictor.args.imgsz = img_size
        
        self._set_image_if_needed(image)

        bboxes_xyxy = None
        if bboxes:
            bboxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        
        # 必须传入 source=image，否则 stream_inference 会用
        # self.args.source（空字符串）重建数据集，导致 postprocess 用错误的
        # orig_imgs 缩放 mask，结果偏移
        results = self.predictor(
            source=image,
            bboxes=bboxes_xyxy,
            points=points,
            labels=labels,
        )
        
        _cleanup_runs()
        return results

    def release(self):
        if self.predictor is not None:
            print(f"[SAMWrapper] 彻底销毁模型: {self.model_type}")
            self.predictor = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
