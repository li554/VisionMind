"""
YOLO 已训练模型包装器
用于加载用户训练好的 YOLO 模型权重，提供统一的推理接口
"""
import numpy as np
import torch
import gc
import cv2
from pathlib import Path
from typing import List, Any, Optional
from ultralytics import YOLO

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from ..utils import _cleanup_runs


class YOLOTrainedWrapper(ModelInterface):
    """
    用户训练好的 YOLO 模型包装器
    支持检测、分割、旋转检测(OBB)任务
    """

    def __init__(self):
        self.model = None
        self.model_path = None
        self.task = None  # 'detect', 'segment', 'obb'
        self.names = {}   # 类别名称映射

    def load(self, weight_path: str) -> bool:
        """
        加载训练好的 YOLO 模型权重

        Args:
            weight_path: 模型权重文件路径 (.pt)

        Returns:
            是否加载成功
        """
        try:
            self.model = YOLO(weight_path)
            self.model_path = weight_path
            self.names = self.model.names if hasattr(self.model, 'names') else {}

            # 自动检测任务类型
            if hasattr(self.model, 'task') and self.model.task:
                self.task = self.model.task
            else:
                # 从模型文件名推断任务类型
                weight_name = Path(weight_path).name.lower()
                if '-seg' in weight_name or 'segment' in weight_name:
                    self.task = 'segment'
                elif '-obb' in weight_name or 'obb' in weight_name:
                    self.task = 'obb'
                else:
                    self.task = 'detect'

            print(f"[YOLOTrainedWrapper] 模型加载成功: {weight_path}, 任务类型: {self.task}")
            return True
        except Exception as e:
            print(f"[YOLOTrainedWrapper] 模型加载失败: {e}")
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征 - YOLO 模型通常不直接支持此接口"""
        raise NotImplementedError("YOLOTrainedWrapper does not support 'get_features'")

    def reset_cache(self):
        """重置模型缓存 - YOLO 每次 predict 都是独立的"""
        pass

    def predict(self, image: np.ndarray, bboxes=None, points=None, labels=None, texts=None,
                conf: float = 0.25, iou: float = 0.45, img_size: int = 640, **kwargs) -> List[MockResults]:
        """
        对图像进行推理

        Args:
            image: 输入图像 (numpy array)
            bboxes: 可选的边界框提示（YOLO 不使用）
            points: 可选的点提示（YOLO 不使用）
            labels: 可选的标签提示（YOLO 不使用）
            texts: 可选的文本提示（YOLO 不使用）
            conf: 置信度阈值
            iou: NMS IoU 阈值
            img_size: 输入图像尺寸
            **kwargs: 其他参数

        Returns:
            MockResults 列表
        """
        if self.model is None:
            print("[YOLOTrainedWrapper] 模型未加载")
            return []

        try:
            # 调试信息
            print(f"[YOLOTrainedWrapper] 开始推理:")
            print(f"  - 图像形状: {image.shape}, dtype: {image.dtype}")
            print(f"  - 图像范围: [{image.min()}, {image.max()}]")
            print(f"  - conf: {conf}, iou: {iou}, img_size: {img_size}")
            print(f"  - 模型任务: {self.task}")
            print(f"  - 模型类别: {self.names}")

            # 执行推理
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=img_size,
                verbose=False,
                save=False
            )

            print(f"[YOLOTrainedWrapper] 原始结果数量: {len(results)}")

            # 转换为 MockResults 格式
            mock_results = []
            for result in results:
                # 调试原始结果
                num_boxes = len(result.boxes) if result.boxes is not None else 0
                num_masks = len(result.masks) if result.masks is not None else 0
                print(f"[YOLOTrainedWrapper] 处理结果 - boxes: {num_boxes}, masks: {num_masks}")

                # 提取边界框
                boxes = None
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = MockBoxes(
                        xyxy=result.boxes.xyxy,
                        conf=result.boxes.conf,
                        cls=result.boxes.cls
                    )
                    print(f"[YOLOTrainedWrapper] 检测到 {len(result.boxes)} 个边界框")
                    for i, (cls, conf) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
                        cls_id = int(cls.item())
                        cls_name = self.names.get(cls_id, f"class_{cls_id}")
                        print(f"  - 目标 {i+1}: {cls_name}, conf={conf.item():.3f}")

                # 提取分割掩码
                masks = None
                if result.masks is not None and len(result.masks) > 0:
                    masks = MockMasks(
                        data=result.masks.data.cpu().numpy(),
                        scores=result.boxes.conf.cpu().numpy() if result.boxes is not None else None
                    )

                # 处理 OBB 结果
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                    # OBB 结果也使用 boxes 存储，但格式不同
                    boxes = MockBoxes(
                        xyxy=result.obb.xyxy,  # 转换为水平框用于兼容性
                        conf=result.obb.conf,
                        cls=result.obb.cls
                    )
                    # 存储 OBB 原始信息在 masks 中以便后续处理
                    if masks is None:
                        masks = MockMasks(data=None)
                    masks.obb_xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()

                # 从掩码或边界框提取多边形（用于兼容 process_mask_results）
                polygons = None
                if masks is not None and hasattr(masks, 'data') and len(masks.data) > 0:
                    # 从分割掩码提取多边形
                    polygons = []
                    for i in range(len(masks.data)):
                        mask = masks.data[i]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        # 二值化并提取轮廓
                        mask_binary = (mask > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # 取最大的轮廓
                            largest_contour = max(contours, key=cv2.contourArea)
                            # 简化轮廓
                            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                            pts = approx.reshape(-1, 2)
                            if len(pts) >= 3:
                                polygons.append([pts])
                            else:
                                polygons.append([])
                        else:
                            polygons.append([])
                elif boxes is not None and hasattr(boxes, 'xyxy') and len(boxes.xyxy) > 0:
                    # 检测模型：从边界框生成矩形多边形
                    print(f"[YOLOTrainedWrapper] 从边界框生成多边形")
                    polygons = []
                    for i in range(len(boxes.xyxy)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        # 创建矩形多边形（顺时针）
                        rect_polygon = np.array([
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ], dtype=np.float32)
                        polygons.append([rect_polygon])
                        print(f"  - 目标 {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

                mock_result = MockResults(
                    boxes=boxes,
                    masks=masks,
                    orig_img=result.orig_img,
                    names=self.names,
                    polygons=polygons
                )
                mock_results.append(mock_result)

            print(f"[YOLOTrainedWrapper] 返回 {len(mock_results)} 个 MockResults")
            _cleanup_runs()
            return mock_results

        except Exception as e:
            print(f"[YOLOTrainedWrapper] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def release(self):
        """释放模型资源"""
        if self.model is not None:
            print("[YOLOTrainedWrapper] 释放模型资源")
            self.model = None
            self.model_path = None
            self.names = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def infer(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45,
              img_size: int = 640) -> dict:
        """
        简化版推理接口，直接返回标注格式结果

        Args:
            image: 输入图像
            conf: 置信度阈值
            iou: NMS IoU 阈值
            img_size: 输入图像尺寸

        Returns:
            包含 annotations 的字典，格式与 training_service.run_inference 兼容
        """
        results = self.predict(image, conf=conf, iou=iou, img_size=img_size)

        if not results:
            return {"status": "success", "annotations": []}

        result = results[0]
        annotations = []

        # 处理检测框
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                label = result.names.get(cls_id, f"class_{cls_id}")
                conf_val = float(result.boxes.conf[i])
                coords = result.boxes.xyxy[i].tolist()

                annotations.append({
                    'type': 'rect',
                    'label': label,
                    'confidence': conf_val,
                    'points': [[coords[0], coords[1]], [coords[2], coords[3]]]
                })

        # 处理分割掩码
        if result.masks is not None and hasattr(result.masks, 'data') and result.masks.data is not None:
            # 获取对应的 boxes 信息
            if result.boxes is not None:
                for i in range(len(result.masks.data)):
                    if i < len(result.boxes):
                        cls_id = int(result.boxes.cls[i])
                        label = result.names.get(cls_id, f"class_{cls_id}")
                        conf_val = float(result.boxes.conf[i])

                        # 从掩码提取多边形
                        mask = result.masks.data[i]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()

                        # 二值化并提取轮廓
                        mask_binary = (mask > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            # 取最大的轮廓
                            largest_contour = max(contours, key=cv2.contourArea)
                            # 简化轮廓
                            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                            pts = approx.reshape(-1, 2).tolist()

                            if len(pts) >= 3:
                                annotations.append({
                                    'type': 'polygon',
                                    'label': label,
                                    'confidence': conf_val,
                                    'points': pts
                                })

        # 处理 OBB 结果
        if result.masks is not None and hasattr(result.masks, 'obb_xyxyxyxy'):
            obb_points = result.masks.obb_xyxyxyxy
            if result.boxes is not None:
                for i in range(len(obb_points)):
                    if i < len(result.boxes):
                        cls_id = int(result.boxes.cls[i])
                        label = result.names.get(cls_id, f"class_{cls_id}")
                        conf_val = float(result.boxes.conf[i])

                        pts = obb_points[i].tolist()
                        annotations.append({
                            'type': 'obb',
                            'label': label,
                            'confidence': conf_val,
                            'points': pts
                        })

        return {"status": "success", "annotations": annotations}
