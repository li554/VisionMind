import numpy as np
import torch
import gc
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from .base import ModelInterface
from .registry import register_model
from ..utils import _cleanup_runs

@register_model("yoloe-26x", img_size=640, weight_path_key="yoloe_26x_path", category="yoloe26", role="example")
@register_model("yoloe-26n", img_size=640, weight_path_key="yoloe_26n_path", category="yoloe26", role="example")
@register_model("yoloe-26s", img_size=640, weight_path_key="yoloe_26s_path", category="yoloe26", role="example")
class YOLOEWrapper(ModelInterface):
    """Ultralytics YOLO-EVP 模型包装器"""
    def __init__(self):
        self.model = None

    def load(self, weight_path: str) -> bool:
        try:
            self.model = YOLO(weight_path)
            return True
        except Exception as e:
            print(f"[YOLOEWrapper] 加载失败: {e}")
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        # YOLOE typically doesn't support the same embedding interface as SAM
        # If needed, we might extract backbone features, but for now, raise Error
        raise NotImplementedError("YOLOEWrapper does not support 'get_features'")

    def reset_cache(self):
        """重置模型缓存"""
        # YOLO模型通常每次 predict 都是独立的
        pass

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=640, **kwargs):
        if texts:
            names = [texts] if isinstance(texts, str) else list(texts)
            try:
                if hasattr(self.model, 'set_classes'):
                    self.model.set_classes(names, self.model.get_text_pe(names))
            except Exception as e:
                print(f"[YOLOEWrapper] set_classes 失败: {e}")

        prompts_dict = {}
        if bboxes:
            bboxes_xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in bboxes])
            prompts_dict = dict(bboxes=bboxes_xyxy, cls=np.zeros(len(bboxes_xyxy), dtype=int))
            results = self.model.predict(
                source=image,
                save=False,
                verbose=False,
                imgsz=img_size,
                visual_prompts=prompts_dict,
                predictor=YOLOEVPSegPredictor
            )
        else:
            results = self.model.predict(
                source=image,
                save=False,
                verbose=False,
                imgsz=img_size
            )
        _cleanup_runs()
        return results

    def release(self):
        if self.model is not None:
            print("[YOLOEWrapper] 彻底销毁模型")
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Ensure external predictor refs are also cleared if any exist
        try:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            # Predictors often hold references to the model; 
            # if we have a way to find them, we should clear them.
        except: pass
