import gc
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime
import torch

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import register_model
from ..utils import _cleanup_runs


class SegmentAnythingHQONNX:
    """Segmentation model using SegmentAnythingHQ"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.target_size = 1024
        self.input_size = (684, 1024)

        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path

        # Load models
        self.providers = onnxruntime.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        self.providers = [
            p for p in self.providers if p != "TensorrtExecutionProvider"
        ]

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=self.providers, sess_options=sess_options
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=self.providers, sess_options=sess_options
        )

    def get_input_points(self, prompt):
        """Get input points"""
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
        return points, labels

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(
        self, image_embeddings, interm_embeddings, original_size, transform_matrix, prompt
    ):
        """Run decoder"""
        input_points, input_labels = self.get_input_points(prompt)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size
        ).astype(np.float32)

        # Apply the transformation matrix to the coordinates.
        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        masks, scores, _ = self.decoder_session.run(None, decoder_inputs)

        # Transform the masks back to the original image size.
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix
        )

        return transformed_masks, scores

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        features = self.encoder_session.run(None, encoder_inputs)
        image_embeddings = features[0]
        interm_embeddings = np.stack(features[1:])

        return {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks, scores = self.run_decoder(
            embedding["image_embeddings"],
            embedding["interm_embeddings"],
            embedding["original_size"],
            embedding["transform_matrix"],
            prompt,
        )

        return masks, scores


@register_model("hqsam_onnx_b", img_size=1024, weight_path_key="hqsam_onnx_b_encoder_path,hqsam_onnx_b_decoder_path", category="hqsam_onnx", role="interactive")
@register_model("hqsam_onnx_l", img_size=1024, weight_path_key="hqsam_onnx_l_encoder_path,hqsam_onnx_l_decoder_path", category="hqsam_onnx", role="interactive")
class HQSAMONNXWrapper(ModelInterface):
    """HQSAM ONNX 模型包装器"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.sam_model = None
        self._last_embedding = None
        self._last_image_hash = None

    def load(self, weight_path: str) -> bool:
        try:
            encoder_path, decoder_path = weight_path.split(",")
            encoder_path = encoder_path.strip()
            decoder_path = decoder_path.strip()

            self.sam_model = SegmentAnythingHQONNX(encoder_path, decoder_path)
            return True
        except Exception as e:
            print(f"[HQSAMONNXWrapper] 加载 {self.model_type} 失败: {e}")
            return False

    def get_image_hash(self, image: np.ndarray) -> int:
        return hash(image.tobytes())

    def reset_cache(self):
        self._last_embedding = None
        self._last_image_hash = None

    def get_features(self, image: np.ndarray):
        img_hash = self.get_image_hash(image)
        if self._last_image_hash == img_hash and self._last_embedding is not None:
            return self._last_embedding["image_embeddings"]
        
        self._last_embedding = self.sam_model.encode(image)
        self._last_image_hash = img_hash
        return self._last_embedding["image_embeddings"]

    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=1024, **kwargs):
        try:
            img_hash = self.get_image_hash(image)
            
            if self._last_image_hash != img_hash:
                self._last_embedding = self.sam_model.encode(image)
                self._last_image_hash = img_hash

            if self._last_embedding is None:
                return []

            prompt = []
            
            if bboxes:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                    prompt.append({"type": "rectangle", "data": [x1, y1, x2, y2]})
            
            if points and labels:
                for point_group, label_group in zip(points, labels):
                    for point, label in zip(point_group, label_group):
                        prompt.append({"type": "point", "data": point, "label": label})

            if not prompt:
                return []

            masks, scores = self.sam_model.predict_masks(self._last_embedding, prompt)

            if len(masks) == 0:
                return []

            # 二值化
            masks[masks > 0] = 1
            masks[masks <= 0] = 0

            # masks 形状是 (1, 3, H, W)，取第一个 batch
            masks = masks[0]  # (3, H, W)
            
            # 转换为 (N, H, W) 格式
            masks_list = []
            for i in range(masks.shape[0]):
                masks_list.append(masks[i])
            masks_array = np.array(masks_list)

            masks_tensor = torch.from_numpy(masks_array).float()

            # 使用实际的 scores
            scores_tensor = torch.from_numpy(scores).float()

            # 从 mask 生成 bbox
            bboxes = []
            bbox_conf = []
            bbox_cls = []
            for i, mask in enumerate(masks_list):
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    y1, x1 = coords.min(axis=0)
                    y2, x2 = coords.max(axis=0)
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    bbox_conf.append(float(scores[i]))
                    bbox_cls.append(0)

            boxes = MockBoxes(bboxes, bbox_conf, bbox_cls)
            mock_masks = MockMasks(masks_tensor.cpu().numpy(), scores_tensor.cpu().numpy())
            result = MockResults(boxes, mock_masks, image, {0: 'object'})

            _cleanup_runs()
            return [result]
            
        except Exception as e:
            print(f"[HQSAMONNXWrapper] 预测失败: {e}")
            import traceback
            traceback.print_exc()
            _cleanup_runs()
            return []

    def release(self):
        if self.sam_model is not None:
            self.sam_model.encoder_session = None
            self.sam_model.decoder_session = None
            self.sam_model = None
        self._last_embedding = None
        self._last_image_hash = None
        gc.collect()
