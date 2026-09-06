import numpy as np
import torch
import gc
import cv2
from ultralytics.models.sam import SAM2Predictor
from .base import ModelInterface
from .registry import register_model
from ..utils import _cleanup_runs

@register_model("sam2_b", img_size=640, weight_path_key="sam2_b_path", category="sam2", role="interactive")
@register_model("sam2_l", img_size=640, weight_path_key="sam2_l_path", category="sam2", role="interactive")
class SAM2Wrapper(ModelInterface):
    """Ultralytics SAM2 模型包装器"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.predictor = None
        self.model = None
        self._last_image = None # Cache for last processed image to avoid re-encoding
        
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
        # print(f"[SAM2Wrapper] Setting new image for encoding...")
        self.predictor.set_image(image)
        self._last_image = image # Store reference (or copy if needed, but reference is usually fine for read-only)

    def load(self, weight_path: str) -> bool:
        try:
            print(f"[SAM2Wrapper] 正在加载 {self.model_type} 从 {weight_path}...")
            # Create SAMPredictor with overrides (Same as SAM)
            overrides = dict(
                conf=0.25, 
                task="segment", 
                mode="predict", 
                imgsz=640, 
                model=weight_path
            )
            self.predictor = SAM2Predictor(overrides=overrides)
            print(f"[SAM2Wrapper] 加载完成。")
            return True
        except Exception as e:
            print(f"[SAM2Wrapper] 加载 {self.model_type} 失败: {e}")
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
            
        # Set image to predictor
        self._set_image_if_needed(image)
        
        # Return features
        features = getattr(self.predictor, 'features', None)
        if features is None:
             raise RuntimeError("Features not found in predictor after set_image")
             
        # --- Multi-scale Feature Fusion (User Request) ---
        # User mentioned 'high_res_features' in SAM2 features dict.
        # We fuse them with 'image_embed' to handle scale variance.
        if isinstance(features, dict) and 'high_res_features' in features:
            try:
                high_res = features['high_res_features']
                image_embed = features.get('image_embed', None)
                
                # Collect all available feature maps
                feature_maps = []
                
                # 1. Add High-Res Features (List of tensors)
                if isinstance(high_res, (list, tuple)):
                    feature_maps.extend(high_res)
                elif isinstance(high_res, torch.Tensor):
                    feature_maps.append(high_res)
                    
                # 2. Add Base Image Embedding
                if image_embed is not None:
                    feature_maps.append(image_embed)
                
                if feature_maps:
                    # Determine target resolution (Max H, Max W)
                    max_h, max_w = 0, 0
                    for f in feature_maps:
                        if isinstance(f, torch.Tensor):
                            _, _, h, w = f.shape
                            if h > max_h:
                                max_h = h
                                max_w = w
                    
                    if max_h > 0 and max_w > 0:
                        # Upsample all to max resolution and concatenate
                        processed_maps = []
                        for f in feature_maps:
                            if isinstance(f, torch.Tensor):
                                if f.shape[-2:] != (max_h, max_w):
                                    f = torch.nn.functional.interpolate(
                                        f, size=(max_h, max_w), mode='bilinear', align_corners=False
                                    )
                                processed_maps.append(f)
                        
                        if processed_maps:
                            # Concatenate along channel dimension (dim=1)
                            fused_features = torch.cat(processed_maps, dim=1)
                            return fused_features
            except Exception as e:
                print(f"[SAM2Wrapper] Multi-scale fusion failed: {e}. Falling back to standard extraction.")
        # -------------------------------------------------

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
        
        if img_size != getattr(self, '_last_img_size', 1024):
            self._last_img_size = img_size
            if hasattr(self.predictor, 'args'):
                self.predictor.args.imgsz = img_size
        
        self._set_image_if_needed(image)
        
        bboxes_xyxy = None
        if bboxes:
            bboxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
            
        # Run inference — 必须传入 source=image，否则 stream_inference 会用
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
            print(f"[SAM2Wrapper] 彻底销毁模型: {self.model_type}")
            self.predictor = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
