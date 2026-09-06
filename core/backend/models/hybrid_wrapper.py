import cv2
import numpy as np
from typing import List, Any

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..foreground_check import SMALL_BOX_THRESHOLD, traditional_mask, mask_to_polygon


@register_model("hybrid", img_size=1024, category="refine", role="refine")
class HybridWrapper(ModelInterface):
    """混合分割模型：小目标经典CV前景提取，中大目标交互式分割（SAM2）。"""

    def __init__(self, model_type="hybrid"):
        self.model_type = model_type
        self.sam2_model_type = "sam2_l"
        self._loaded = False

    def load(self, weight_path=None) -> bool:
        self._loaded = True
        return True

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None,
                conf=0.5, img_size=640, **kwargs) -> List[Any]:
        if not self._loaded or bboxes is None:
            return []
        sam2_model_type = kwargs.get("sam2_model_type", self.sam2_model_type)
        masks = []
        polygons = []
        methods = []
        for box in bboxes:
            vals = [float(v) for v in box[:4]]
            x, y, w, h = [int(round(v)) for v in vals]
            if w <= 0 or h <= 0:
                continue
            if max(w, h) < SMALL_BOX_THRESHOLD:
                mask = traditional_mask(image, (x, y, w, h))
                method = "CV"
            else:
                mask = self._interactive_mask(image, (x, y, w, h), sam2_model_type)
                method = "SAM2"
            if mask is None or int(np.count_nonzero(mask)) <= 0:
                continue
            mask = (mask > 0).astype(np.uint8)
            masks.append(mask.astype(np.float32) / 255.0)
            polygons.append(mask_to_polygon(mask))
            methods.append(method)
        if not masks:
            return []
        result = MockResults(
            boxes=MockBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            masks=MockMasks(np.stack(masks)),
            orig_img=image,
            polygons=polygons,
        )
        result.methods = methods
        return [result]

    def _interactive_mask(self, image, box, model_type="sam2_l"):
        try:
            from ..core import interactive_predict
            from ..defaults import resolve_interactive_params
            params = resolve_interactive_params(model_type, refine=False, img_size=1024)
            results = interactive_predict(image, bboxes=[list(box)], **params)
            if not results or results[0].masks is None:
                return None
            masks = results[0].masks.data.cpu().numpy()
            if masks.ndim != 3 or len(masks) == 0:
                return None
            m = masks[0]
            H, W = image.shape[:2]
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            return (m > 0.5).astype(np.uint8)
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    def release(self):
        self._loaded = False

    def reset_cache(self):
        pass
