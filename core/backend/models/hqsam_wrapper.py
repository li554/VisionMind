import numpy as np
import torch
import gc
import cv2
import warnings

# Suppress tiny_vit registration warnings from segment_anything_hq
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Overwriting tiny_vit_.* in registry")
    from segment_anything_hq import sam_model_registry, SamPredictor

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..utils import _cleanup_runs

@register_model("hqsam_b", img_size=1024, weight_path_key="sam_hq_b_path", category="hqsam", role="interactive")
@register_model("hqsam_l", img_size=1024, weight_path_key="sam_hq_l_path", category="hqsam", role="interactive")
@register_model("hqsam_tiny", img_size=1024, weight_path_key="sam_hq_tiny_path", category="hqsam", role="interactive")
class HQSAMWrapper(ModelInterface):
    """HQ-SAM 模型包装器 (High Quality Segment Anything Model)"""
    def __init__(self, model_type: str = "vit_b"):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.model = None
        self._last_image = None

    def _set_image_if_needed(self, image: np.ndarray):
        """Internal helper to set image only if changed"""
        if self.predictor is None:
             return
             
        # Check if image is the same object or content
        if self._last_image is not None:
            # Fast check: object identity
            if image is self._last_image:
                return
            # Content check
            if image.shape == self._last_image.shape and np.array_equal(image, self._last_image):
                return
                
        # If different, reset old cache first
        self.reset_cache()
        
        # Set new image
        self.predictor.set_image(image)
        self._last_image = image

    def load(self, weight_path: str) -> bool:
        try:
            print(f"[HQSAMWrapper] 正在加载 {self.model_type} 从 {weight_path}...")
            # model_type mapping: vit_b, vit_l, vit_h, vit_tiny
            # HQ-SAM usually uses these types
            hq_type = self.model_type
            if hq_type.startswith("sam_hq_"):
                hq_type = hq_type.replace("sam_hq_", "")
            
            # Map standard types if needed
            type_map = {
                "b": "vit_b",
                "l": "vit_l",
                "h": "vit_h",
                "tiny": "vit_tiny"
            }
            hq_type = type_map.get(hq_type, hq_type)
            
            self.model = sam_model_registry[hq_type](checkpoint=weight_path)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
            print(f"[HQSAMWrapper] 加载完成。")
            return True
        except Exception as e:
            print(f"[HQSAMWrapper] 加载 {self.model_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        # HQSAM SamPredictor handles normalization and resizing internally via set_image
        self._set_image_if_needed(image)
        return self.predictor.get_image_embedding()

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=1024, **kwargs):
        if self.predictor is None:
            print("[HQSAMWrapper] 错误: 模型未加载")
            return []

        self._last_img_size = img_size

        # 1. 设置图像
        self._set_image_if_needed(image)

        final_results = []
        
        # 2. 处理 Prompt
        # HQ-SAM predictor.predict 支持 multishape prompts
        # 但为了简单和兼容性，我们按实例处理或合并处理
        
        if bboxes:
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                input_box = np.array([x, y, x + w, y + h])
                
                # HQ-SAM predict call
                masks, scores, logits = self.predictor.predict(
                    box=input_box,
                    multimask_output=True,
                    hq_token_only=True # 使用 HQ-SAM 的高精度输出
                )
                
                # Convert to MockResults format
                res = self._format_results(masks, scores, image)
                final_results.append(res)
                
        elif points:
            # 确保 points 是 (N, 2) 的 float32 数组
            pts = np.array(points, dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, 2)
            elif pts.ndim == 3 and pts.shape[0] == 1:
                pts = pts.reshape(-1, 2)
            
            # 确保 labels 是 (N,) 的 int 数组
            if labels is not None:
                lbls = np.array(labels, dtype=np.int32).reshape(-1)
            else:
                lbls = np.ones(len(pts), dtype=np.int32)

            # 检查点和标签数量是否匹配
            if len(pts) != len(lbls):
                print(f"[HQSAMWrapper] 警告: 点数 ({len(pts)}) 与标签数 ({len(lbls)}) 不匹配")
                lbls = np.ones(len(pts), dtype=np.int32)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=True,
                hq_token_only=True
            )
            
            res = self._format_results(masks, scores, image)
            final_results.append(res)
        
        _cleanup_runs()
        return final_results

    def _format_results(self, masks, scores, image):
        """将 HQ-SAM 结果转换为 MockResults"""
        # masks: [1, H, W] or [3, H, W]
        # scores: [1] or [3]
        
        # Find best score index
        best_idx = np.argmax(scores)
        mask = masks[best_idx] # [H, W] bool
        score = scores[best_idx]
        
        # Calculate bbox from mask
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0:
            x1, y1, x2, y2 = x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()
            bbox = [x1, y1, x2, y2]
        else:
            bbox = [0, 0, 0, 0]
            
        boxes = MockBoxes([bbox], [score], [0])
        masks_mock = MockMasks(mask[None, ...].astype(np.float32))
        
        return MockResults(boxes, masks_mock, image)

    def release(self):
        if self.model is not None:
            print(f"[HQSAMWrapper] 释放资源: {self.model_type}")
            self.model = None
            self.predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def reset_cache(self):
        """重置模型缓存"""
        self._last_image = None
        if self.predictor:
            # HQSAM SamPredictor logic for cache reset
            self.predictor.is_image_set = False
            self.predictor.features = None
            self.predictor.orig_h = None
            self.predictor.orig_w = None
            self.predictor.input_h = None
            self.predictor.input_w = None
