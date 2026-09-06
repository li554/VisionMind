from typing import Any, List, Tuple, Union
import gc

import cv2
import numpy as np
import onnxruntime as ort
import torch

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..utils import _cleanup_runs


class SAM2ImageEncoder:
    def __init__(self, path: str, device: str = "cpu") -> None:
        # Initialize model
        providers = ["CPUExecutionProvider"]
        if device.lower() == "gpu":
            providers = ["CUDAExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            path, providers=providers, sess_options=sess_options
        )

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        outputs = self.forward_encoder(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.resize(image, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def forward_encoder(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        return outputs

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]


class SAM2ImageDecoder:
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        encoder_input_size: Tuple[int, int] = (1024, 1024),
        orig_im_size: Tuple[int, int] = None,
        mask_threshold: float = 0.0,
    ) -> None:
        # Initialize model
        providers = ["CPUExecutionProvider"]
        if device.lower() == "gpu":
            providers = ["CUDAExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            path, providers=providers, sess_options=sess_options
        )

        self.orig_im_size = (
            orig_im_size if orig_im_size is not None else encoder_input_size
        )
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        return self.predict(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

    def predict(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

        outputs = self.forward_decoder(inputs)

        return self.process_output(outputs)

    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ):
        input_point_coords, input_point_labels = self.prepare_points(
            point_coords, point_labels
        )

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            input_point_coords,
            input_point_labels,
            mask_input,
            has_mask_input,
        )

    def prepare_points(
        self,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max([coords.shape[0] for coords in point_coords])
            # We need to make sure that all inputs have the same number of points
            # Add invalid points to pad the input (0, 0) with -1 value for labels
            input_point_coords = np.zeros(
                (len(point_coords), max_num_points, 2), dtype=np.float32
            )
            input_point_labels = (
                np.ones((len(point_coords), max_num_points), dtype=np.float32)
                * -1
            )

            for i, (coords, labels) in enumerate(
                zip(point_coords, point_labels)
            ):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (
            input_point_coords[..., 0]
            / self.orig_im_size[1]
            * self.encoder_input_size[1]
        )  # Normalize x
        input_point_coords[..., 1] = (
            input_point_coords[..., 1]
            / self.orig_im_size[0]
            * self.encoder_input_size[0]
        )  # Normalize y

        return input_point_coords.astype(
            np.float32
        ), input_point_labels.astype(np.float32)

    def forward_decoder(self, inputs) -> List[np.ndarray]:
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[i]: inputs[i]
                for i in range(len(self.input_names))
            },
        )
        return outputs

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[List[Union[np.ndarray, Any]], np.ndarray]:
        scores = outputs[1].squeeze()
        masks = outputs[0][0]

        # Select the best masks based on the scores
        best_mask = masks[np.argmax(scores)]
        best_mask = cv2.resize(
            best_mask, (self.orig_im_size[1], self.orig_im_size[0])
        )
        return (
            np.array([[best_mask]]),
            scores,
        )

    def set_image_size(self, orig_im_size: Tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]


class SegmentAnything2ONNX:
    """Segmentation model using Segment Anything 2 (SAM2)"""

    def __init__(self, encoder_model_path, decoder_model_path, device="cpu") -> None:
        self.encoder = SAM2ImageEncoder(encoder_model_path, device)
        self.decoder = SAM2ImageDecoder(
            decoder_model_path, device, self.encoder.input_shape[2:]
        )

    def encode(self, cv_image: np.ndarray) -> List[np.ndarray]:
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(
            cv_image
        )
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(self, embedding, prompt) -> List[np.ndarray]:
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)
        print(f"[SegmentAnything2ONNX] predict_masks: points shape={points.shape}, labels shape={labels.shape}")

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)
        masks, scores = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            points,
            labels,
        )

        return masks, scores


@register_model("sam2_onnx_b", img_size=1024, weight_path_key="sam2_onnx_b_encoder_path,sam2_onnx_b_decoder_path", category="sam2_onnx", role="interactive")
@register_model("sam2_onnx_l", img_size=1024, weight_path_key="sam2_onnx_l_encoder_path,sam2_onnx_l_decoder_path", category="sam2_onnx", role="interactive")
class SAM2ONNXWrapper(ModelInterface):
    """SAM2 ONNX 模型包装器"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.sam_model = None
        self._last_embedding = None
        self._last_image_hash = None
        self.device = "cpu"

    def load(self, weight_path: str) -> bool:
        try:
            encoder_path, decoder_path = weight_path.split(",")
            encoder_path = encoder_path.strip()
            decoder_path = decoder_path.strip()

            self.sam_model = SegmentAnything2ONNX(encoder_path, decoder_path, self.device)
            return True
        except Exception as e:
            print(f"[SAM2ONNXWrapper] 加载 {self.model_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_image_hash(self, image: np.ndarray) -> int:
        return hash(image.tobytes())

    def reset_cache(self):
        self._last_embedding = None
        self._last_image_hash = None

    def get_features(self, image: np.ndarray):
        img_hash = self.get_image_hash(image)
        if self._last_image_hash == img_hash and self._last_embedding is not None:
            return self._last_embedding["image_embedding"]
        
        self._last_embedding = self.sam_model.encode(image)
        self._last_image_hash = img_hash
        return self._last_embedding["image_embedding"]

    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=640, **kwargs):
        try:
            print(f"[SAM2ONNXWrapper] predict called with bboxes={bboxes}, points={points is not None}, labels={labels is not None}")
            img_hash = self.get_image_hash(image)
            
            if self._last_image_hash != img_hash:
                print(f"[SAM2ONNXWrapper] Encoding new image...")
                self._last_embedding = self.sam_model.encode(image)
                self._last_image_hash = img_hash

            if self._last_embedding is None:
                print(f"[SAM2ONNXWrapper] No embedding available")
                return []

            prompt = []
            
            if bboxes:
                print(f"[SAM2ONNXWrapper] Processing {len(bboxes)} bboxes")
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    prompt.append({"type": "rectangle", "data": [x1, y1, x2, y2]})
            
            if points and labels:
                print(f"[SAM2ONNXWrapper] Processing {len(points)} point groups, {len(labels)} label groups")
                for point_group, label_group in zip(points, labels):
                    print(f"[SAM2ONNXWrapper] point_group shape: {np.array(point_group).shape}, label_group shape: {np.array(label_group).shape}")
                    for point, label in zip(point_group, label_group):
                        prompt.append({"type": "point", "data": point, "label": label})

            print(f"[SAM2ONNXWrapper] Total prompts: {len(prompt)}")
            if not prompt:
                print(f"[SAM2ONNXWrapper] No prompts generated, returning empty")
                return []

            masks, scores = self.sam_model.predict_masks(self._last_embedding, prompt)
            print(f"[SAM2ONNXWrapper] predict_masks returned masks shape={masks.shape}, scores shape={scores.shape}")

            if len(masks) == 0:
                print(f"[SAM2ONNXWrapper] Empty masks returned")
                return []

            # 二值化
            masks[masks > 0] = 1
            masks[masks <= 0] = 0

            # masks 形状是 (1, 1, H, W)，取第一个 batch 和第一个 mask
            masks = masks[0, 0]  # (H, W)
            
            # 转换为 (1, H, W) 格式
            masks_array = np.array([masks])

            masks_tensor = torch.from_numpy(masks_array).float()
            
            # 使用实际的 scores
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            scores_tensor = torch.from_numpy(np.array([best_score])).float()
            print(f"[SAM2ONNXWrapper] best_score={best_score}")

            # 从 mask 生成 bbox
            mask_np = masks_array[0]  # (H, W)
            coords = np.argwhere(mask_np > 0)
            if len(coords) > 0:
                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0)
                bbox = [[float(x1), float(y1), float(x2), float(y2)]]
                bbox_conf = [float(best_score)]
                bbox_cls = [0]
            else:
                bbox = []
                bbox_conf = []
                bbox_cls = []
            
            boxes = MockBoxes(bbox, bbox_conf, bbox_cls)
            mock_masks = MockMasks(masks_tensor.cpu().numpy(), scores_tensor.cpu().numpy())
            result = MockResults(boxes, mock_masks, image, {0: 'object'})

            print(f"[SAM2ONNXWrapper] Returning result with masks")
            _cleanup_runs()
            return [result]
            
        except Exception as e:
            print(f"[SAM2ONNXWrapper] 预测失败: {e}")
            import traceback
            traceback.print_exc()
            _cleanup_runs()
            return []

    def release(self):
        if self.sam_model is not None:
            self.sam_model.encoder.session = None
            self.sam_model.decoder.session = None
            self.sam_model = None
        self._last_embedding = None
        self._last_image_hash = None
        gc.collect()
