import os
import cv2
import gc
import torch
import numpy as np
from typing import List, Any

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model


# ── CascadePSP / ViTMatte implementations (moved from refinement.py) ──

class CascadePSPRerefiner:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.refiner = None

    def _load_model(self):
        import segmentation_refinement as refine
        from ..path_resolver import get_dir
        if self.refiner is None:
            print(f"[CascadePSP] 正在加载模型到 {self.device}...")
            self.refiner = refine.Refiner(device=self.device, model_folder=get_dir("refine_dir"))
            print("[CascadePSP] 模型加载完成。")

    def refine(self, image: np.ndarray, mask: np.ndarray, fast: bool = False, L: int = 900, use_roi: bool = True, margin: float = 0.15) -> np.ndarray:
        self._load_model()

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        if not use_roi:
            return self.refiner.refine(image, mask, fast=fast, L=L)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return mask

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        h, w = mask.shape
        roi_h, roi_w = y_max - y_min, x_max - x_min

        pad_h = int(roi_h * margin)
        pad_w = int(roi_w * margin)

        y1 = max(0, y_min - pad_h)
        y2 = min(h, y_max + pad_h)
        x1 = max(0, x_min - pad_w)
        x2 = min(w, x_max + pad_w)

        roi_image = image[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]

        refined_roi = self.refiner.refine(roi_image, roi_mask, fast=fast, L=L)

        full_refined_mask = np.zeros_like(mask)
        full_refined_mask[y1:y2, x1:x2] = refined_roi

        return full_refined_mask

    def release(self):
        if self.refiner is not None:
            print("[CascadePSP] 正在释放资源...")
            self.refiner = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ViTMatteRefiner:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', use_onnx: bool = False):
        self.device = device
        self.use_onnx = use_onnx
        self.processor = None
        self.model = None
        self.onnx_session = None

    def _load_model(self):
        import onnxruntime as ort
        from PIL import Image
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        from ..path_resolver import get_dir, get_path

        vitmatte_dir = get_dir("vitmatte_dir")
        if not vitmatte_dir or not os.path.isdir(vitmatte_dir):
            raise FileNotFoundError(f"VitMatte模型目录不存在: {vitmatte_dir}")

        if self.processor is None:
            self.processor = VitMatteImageProcessor.from_pretrained(vitmatte_dir)

        if self.use_onnx:
            if self.onnx_session is None:
                onnx_path = get_path("vitmatte_onnx_path")
                if not onnx_path or not os.path.exists(onnx_path):
                    raise FileNotFoundError(f"VitMatte ONNX模型文件不存在: {onnx_path}")
                print(f"[ViTMatte] 正在加载 ONNX 模型到 {self.device}...")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                print("[ViTMatte] ONNX 模型加载完成。")
        else:
            if self.model is None:
                print(f"[ViTMatte] 正在加载 PyTorch 模型到 {self.device}...")
                self.model = VitMatteForImageMatting.from_pretrained(vitmatte_dir).to(self.device)
                self.model.eval()
                print("[ViTMatte] PyTorch 模型加载完成。")

    def refine(self, image: np.ndarray, mask: np.ndarray, L: int = 896, trimap_thickness: int = 20, use_roi: bool = False, margin: float = 0.15, **kwargs) -> np.ndarray:
        import onnxruntime as ort
        from PIL import Image

        self._load_model()

        h, w = mask.shape[:2]

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if use_roi:
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()

                roi_h, roi_w = y_max - y_min, x_max - x_min
                pad_h = int(roi_h * margin)
                pad_w = int(roi_w * margin)

                y1 = max(0, y_min - pad_h)
                y2 = min(h, y_max + pad_h)
                x1 = max(0, x_min - pad_w)
                x2 = min(w, x_max + pad_w)

                roi_image = image[y1:y2, x1:x2]
                roi_mask = mask[y1:y2, x1:x2]

                refined_roi = self.refine(roi_image, roi_mask, L=L, trimap_thickness=trimap_thickness, use_roi=False)

                full_refined_mask = np.zeros_like(mask)
                full_refined_mask[y1:y2, x1:x2] = refined_roi
                return full_refined_mask

        if max(h, w) > L:
            if w > h:
                new_w = L
                new_h = int(h * L / w)
            else:
                new_h = L
                new_w = int(w * L / h)
            img_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_small = image
            mask_small = mask
            new_w, new_h = w, h

        trimap = mask_small.copy()
        contours, _ = cv2.findContours(trimap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(trimap, [contour], -1, 128, trimap_thickness)

        pil_image = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        pil_trimap = Image.fromarray(trimap).convert("L")

        inputs = self.processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")

        if self.use_onnx:
            pixel_values = inputs["pixel_values"].numpy()
            onnx_inputs = {"pixel_values": pixel_values.astype(np.float32)}
            outputs = self.onnx_session.run(None, onnx_inputs)
            alphas = outputs[0]
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas)
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                alphas = outputs.alphas

        alphas = alphas.squeeze(0).squeeze(0).cpu().numpy()

        if alphas.shape[0] > new_h or alphas.shape[1] > new_w:
            alphas = alphas[:new_h, :new_w]

        refined_mask_small = (alphas > 0.5).astype(np.uint8) * 255

        if (new_w, new_h) != (w, h):
            refined_mask = cv2.resize(refined_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            refined_mask = refined_mask_small

        return refined_mask

    def release(self):
        if self.model is not None:
            print("[ViTMatte] 正在释放 PyTorch 资源...")
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.onnx_session is not None:
            print("[ViTMatte] 正在释放 ONNX 资源...")
            self.onnx_session = None
            self.processor = None
            gc.collect()


# ── RefineWrapper (ModelInterface) ──

@register_model("cascadepsp", img_size=640, category="refine", role="refine")
@register_model("vitmatte", img_size=640, category="refine", role="refine")
class RefineWrapper(ModelInterface):
    """Wraps CascadePSP/ViTMatte as a ModelInterface-compatible model."""

    def __init__(self, model_type: str = "vitmatte"):
        self.model_type = model_type
        self._refiner = None

    def load(self, weight_path: str = None) -> bool:
        try:
            if self.model_type == "cascadepsp":
                self._refiner = CascadePSPRerefiner()
            else:
                self._refiner = ViTMatteRefiner(use_onnx=True)
            return True
        except Exception as e:
            print(f"[RefineWrapper] Load failed: {e}")
            return False

    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, texts=None,
                conf=0.5, img_size=640, **kwargs) -> List[Any]:
        """Refine existing masks. Expects 'masks' in kwargs (list of uint8 0/255 numpy arrays)."""
        masks = kwargs.get('masks')
        if masks is None:
            return []

        refined_masks = []
        for mask in masks:
            mask_u8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            refined = self._refiner.refine(
                image, mask_u8,
                fast=kwargs.get('fast', False),
                L=kwargs.get('L', 900 if self.model_type == 'vitmatte' else 320),
                use_roi=kwargs.get('use_roi', True),
                margin=kwargs.get('margin', 0.15),
                trimap_thickness=kwargs.get('trimap_thickness', 20),
            )
            refined_masks.append(refined.astype(np.float32) / 255.0)

        if not refined_masks:
            return []

        return [MockResults(
            boxes=MockBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)),
            masks=MockMasks(np.stack(refined_masks)),
            orig_img=image,
        )]

    def release(self):
        if self._refiner is not None:
            self._refiner.release()
            self._refiner = None

    def reset_cache(self):
        pass
