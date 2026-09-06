import numpy as np
import os
import cv2
import gc
import torch
import sys
from typing import List, Dict, Any, Optional
import time
from .path_resolver import get_path, get_dir, get_url, get_all_download_urls, get_weights_dir, _PROJECT_ROOT
from .utils import merge_multi_contours

from .models import (
    ModelInterface, YOLOTrainedWrapper
)
from .models.registry import get_registry

# Global instances for lazy loading
_interactive_predictor = None
_example_predictor = None
_models = {}  # Store all loaded models here: {model_type: model_object}


class ModelFactory:
    """模型工厂：基于 decorator 注册表自动发现模型"""
    _custom_weight_paths: Dict[str, str] = {}
    _custom_roles: Dict[str, str] = {}
    _creators: Dict[str, Any] = {}  # Only custom models stored here

    TRAINING_MODELS = {
        "detection": [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolo26n.pt", "yolo26s.pt"
        ],
        "segmentation": [
            "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
            "yolo26n-seg.pt", "yolo26s-seg.pt"
        ],
        "obb": [
            "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",
            "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt",
            "yolo26n-obb.pt", "yolo26s-obb.pt"
        ]
    }

    @classmethod
    def get_interactive_models(cls) -> List[str]:
        """获取所有可用的交互式模型名称列表（含自定义模型）"""
        registry = get_registry()
        builtin = [mt for mt, info in registry.items()
                   if info["role"] == "interactive" and mt not in cls._custom_weight_paths]
        custom = [mt for mt, role in cls._custom_roles.items() if role == "interactive"]
        return builtin + custom

    @classmethod
    def get_auto_models(cls) -> List[str]:
        """获取所有可用的自动标注模型名称列表（含自定义模型）"""
        registry = get_registry()
        builtin = [mt for mt, info in registry.items()
                   if info["role"] == "example" and mt not in cls._custom_weight_paths]
        custom = [mt for mt, role in cls._custom_roles.items() if role == "auto"]
        return builtin + custom

    @classmethod
    def get_interactive_categories(cls) -> List[str]:
        """获取交互式模型可用类别"""
        registry = get_registry()
        return sorted({info["category"] for info in registry.values() if info["role"] == "interactive"})

    @classmethod
    def get_auto_categories(cls) -> List[str]:
        """获取自动标注模型可用类别"""
        registry = get_registry()
        return sorted({info["category"] for info in registry.values() if info["role"] == "example"})

    @classmethod
    def get_refine_models(cls) -> List[str]:
        """获取所有可用的细化模型名称列表（含自定义模型）"""
        registry = get_registry()
        builtin = [mt for mt, info in registry.items()
                   if info["role"] == "refine" and mt not in cls._custom_weight_paths]
        custom = [mt for mt, role in cls._custom_roles.items() if role == "refine"]
        return builtin + custom

    @classmethod
    def get_refine_categories(cls) -> List[str]:
        """获取细化模型可用类别"""
        registry = get_registry()
        return sorted({info["category"] for info in registry.values() if info["role"] == "refine"})

    PLAIN_BUILTIN_CATEGORIES = ["yolo_trained"]

    @classmethod
    def get_plain_models(cls) -> List[str]:
        """获取所有可用的普通模型名称列表（含自定义模型）"""
        registry = get_registry()
        builtin = [mt for mt, info in registry.items()
                   if info["role"] == "plain" and mt not in cls._custom_weight_paths]
        custom = [mt for mt, role in cls._custom_roles.items() if role == "plain"]
        return builtin + custom

    @classmethod
    def get_plain_categories(cls) -> List[str]:
        """获取普通模型可用类别（内置类别，如 yolo_trained，后续可扩展 rtdetr 等）"""
        registry = get_registry()
        registered = sorted({info["category"] for info in registry.values() if info["role"] == "plain"})
        for cat in cls.PLAIN_BUILTIN_CATEGORIES:
            if cat not in registered:
                registered.append(cat)
        return registered

    @classmethod
    def is_plain_model(cls, model_type: str) -> bool:
        """判断模型名称是否为普通模型（内置 role=plain 或自定义 role=plain）"""
        if model_type in cls._custom_roles:
            return cls._custom_roles[model_type] == "plain"
        info = get_registry().get(model_type)
        return bool(info and info["role"] == "plain")

    @classmethod
    def register_custom_model(cls, model_type: str, category: str, weight_path: str, role: str = "auto"):
        """注册自定义模型到工厂"""
        # Find wrapper class from registry by category
        wrapper_cls = None
        for info in get_registry().values():
            if info["category"] == category:
                wrapper_cls = info["wrapper_class"]
                break
        if not wrapper_cls:
            # Fallback: also check yolo_trained category
            if category == "yolo_trained":
                wrapper_cls = YOLOTrainedWrapper
            else:
                print(f"[ModelFactory] 未知模型类别: {category}")
                return False

        # Auto-detect needs_model_type from registry
        needs_mt = False
        for info in get_registry().values():
            if info["wrapper_class"] is wrapper_cls:
                needs_mt = info.get("needs_model_type", False)
                break

        if needs_mt:
            cls._creators[model_type] = lambda mt=model_type, wc=wrapper_cls: wc(mt)
        else:
            cls._creators[model_type] = wrapper_cls

        cls._custom_weight_paths[model_type] = weight_path
        cls._custom_roles[model_type] = role

        print(f"[ModelFactory] 已注册自定义模型: {model_type} (类别={category}, 角色={role})")
        return True

    @classmethod
    def unregister_custom_model(cls, model_type: str, role: str = "auto"):
        """注销自定义模型"""
        cls._creators.pop(model_type, None)
        cls._custom_weight_paths.pop(model_type, None)
        cls._custom_roles.pop(model_type, None)
        print(f"[ModelFactory] 已注销自定义模型: {model_type}")

    @classmethod
    def get_custom_model_info(cls, model_type: str) -> Optional[Dict]:
        """获取自定义模型信息"""
        if model_type in cls._custom_weight_paths:
            return {
                "name": model_type,
                "weight_path": cls._custom_weight_paths[model_type],
            }
        return None

    @classmethod
    def get_training_models(cls, task_type: str = "detection") -> List[str]:
        """获取指定任务类型的可训练模型列表"""
        return cls.TRAINING_MODELS.get(task_type, [])

    @classmethod
    def get_default_img_size(cls, model_type: str) -> int:
        """获取模型默认图像大小"""
        info = get_registry().get(model_type)
        return info["img_size"] if info else 1024

    @classmethod
    def _get_weight_path(cls, model_type: str) -> str:
        """获取模型权重路径（从注册表的 weight_path_key 解析）"""
        info = get_registry().get(model_type)
        if not info or not info.get("weight_path_key"):
            return ""
        key = info["weight_path_key"]
        if "," in key:
            keys = [k.strip() for k in key.split(",")]
            paths = [get_path(k) for k in keys]
            return ",".join(p for p in paths if p)
        return get_path(key)

    @classmethod
    def create(cls, model_type: str, weight_path: str = None) -> Optional[ModelInterface]:
        """创建并加载模型"""
        from .utils import download_file

        # 1. Check custom creators
        creator = cls._creators.get(model_type)
        if creator:
            wrapper = creator() if callable(creator) else creator
        else:
            # 2. Look up from registry
            info = get_registry().get(model_type)
            if not info:
                print(f"[ModelFactory] 未知模型类型: {model_type}")
                return None
            wrapper_cls = info["wrapper_class"]
            wrapper = wrapper_cls(model_type) if info.get("needs_model_type") else wrapper_cls()
            # Set metadata on instance
            wrapper.model_type = model_type
            wrapper.type = info["role"]
            wrapper.img_size = info["img_size"]

        # 3. Resolve weight path
        if model_type in cls._custom_weight_paths:
            weight_path = cls._custom_weight_paths[model_type]
        elif weight_path is None:
            weight_path = cls._get_weight_path(model_type)

        if wrapper.load(weight_path):
            return wrapper

        # 4. Auto-download
        print(f"[ModelFactory] 模型加载失败，尝试自动下载: {model_type}")

        if "," in weight_path:
            encoder_path, decoder_path = weight_path.split(",")
            encoder_path = encoder_path.strip()
            decoder_path = decoder_path.strip()

            if encoder_path:
                os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            if decoder_path:
                os.makedirs(os.path.dirname(decoder_path), exist_ok=True)

            encoder_filename = os.path.basename(encoder_path)
            decoder_filename = os.path.basename(decoder_path)

            encoder_url = get_all_download_urls().get(encoder_filename)
            decoder_url = get_all_download_urls().get(decoder_filename)

            if encoder_url and not os.path.exists(encoder_path):
                if not download_file(encoder_url, encoder_path):
                    print(f"[ModelFactory] Encoder下载失败: {encoder_filename}")
                    return None

            if decoder_url and not os.path.exists(decoder_path):
                if not download_file(decoder_url, decoder_path):
                    print(f"[ModelFactory] Decoder下载失败: {decoder_filename}")
                    return None
        else:
            if weight_path:
                filename = os.path.basename(weight_path)
                url = get_all_download_urls().get(filename)
                if url:
                    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                    if not download_file(url, weight_path):
                        print(f"[ModelFactory] 模型下载失败: {filename}")
                        return None

        if wrapper.load(weight_path):
            return wrapper

        print(f"[ModelFactory] 模型加载失败: {model_type} -> {weight_path}")
        return None


class BaseSegmentor:
    """通用分割容器基类"""

    def __init__(self, category: str):
        self.category = category
        self.models: Dict[str, ModelInterface] = {}
        self._external_models: Dict[str, ModelInterface] = {}

    def register_model(self, model_type: str, wrapper: ModelInterface):
        """支持外部手动注册模型实例"""
        self._external_models[model_type] = wrapper
        print(f"[{self.category}] 成功注册外部模型: {model_type}")

    def get_model(self, model_type: str, weight_path: str = None) -> Optional[ModelInterface]:
        """
        获取模型实例

        Args:
            model_type: 模型类型名称
            weight_path: 可选的权重路径，用于加载用户训练的模型

        Returns:
            ModelInterface 实例或 None
        """
        print(f"[{self.category}] get_model 被调用: model_type={model_type}, weight_path={weight_path}")
        # 1. 优先从外部注册的模型中获取
        if model_type in self._external_models:
            print(f"[{self.category}] 从外部注册模型中获取: {model_type}")
            return self._external_models[model_type]

        # 2. 如果请求的模型已经在缓存中，直接返回
        if model_type in self.models:
            print(f"[{self.category}] 从缓存中获取模型: {model_type}")
            return self.models[model_type]

        # 3. 如果是交互式分割器，切换模型时释放之前的模型以节省显存
        if self.category == "InteractivePredictor":
            print(f"[{self.category}] 切换模型到 {model_type}，正在释放旧模型...")
            self.release_models()

        # Double check cache after release (should be empty, but just in case logic changes)
        if model_type in self.models:
            return self.models[model_type]

        # 4. 如果提供了 weight_path，创建 YOLOTrainedWrapper
        if weight_path and os.path.exists(weight_path):
            print(f"[{self.category}] 创建 YOLOTrainedWrapper, weight_path={weight_path}")
            wrapper = YOLOTrainedWrapper()
            if wrapper.load(weight_path):
                self.models[model_type] = wrapper
                print(f"[{self.category}] YOLOTrainedWrapper 加载成功")
                return wrapper
            else:
                print(f"[{self.category}] 无法加载模型权重: {weight_path}")

        # 5. 尝试通过工厂动态创建
        print(f"[{self.category}] 尝试通过工厂创建模型: {model_type}")
        wrapper = ModelFactory.create(model_type)
        if wrapper:
            self.models[model_type] = wrapper
            print(f"[{self.category}] 工厂创建模型成功: {model_type}")
            return wrapper

        print(f"[{self.category}] 无法获取模型: {model_type}")
        return None

    def _smooth_mask_contours(self, mask: np.ndarray, epsilon: float = 0.001) -> List[np.ndarray]:
        """
        使用道格拉斯 - 普克算法将掩码轮廓转换为多边形
        
        参数:
            mask: 二值掩码 (H, W), 值域 0-255
            epsilon: 轮廓简化系数 (0.0001-0.01), 越小保留细节越多，越大边缘越平滑
        
        返回:
            简化后的多边形列表，每个多边形形状为 (N, 2)
        """
        # 确保是 uint8 类型
        mask = mask.astype(np.uint8)

        # 提取轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # 使用道格拉斯 - 普克算法简化轮廓
        approx_contours = []
        for contour in contours:
            # epsilon = 系数 * 轮廓周长
            epsilon_len = epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_len, True)
            approx_contours.append(approx)

        # 过滤异常大的轮廓 (>90% 图像面积)
        if len(approx_contours) > 1:
            image_size = mask.shape[0] * mask.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered = [
                contour for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]
            if filtered:
                approx_contours = filtered

        # 过滤异常小的轮廓 (< 平均面积的 20%)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)
            filtered = [
                contour for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            if filtered:
                approx_contours = filtered

        # 使用 merge_multi_contours 合并多个轮廓为一个多边形
        merged_poly = merge_multi_contours(approx_contours)

        # 返回单个多边形数组
        return merged_poly

    def predict(self, image, model_type: str, refine: bool = True, smooth: bool = False, smooth_epsilon: float = 0.001,
                img_size: int = 640, **kwargs):
        """执行预测"""
        # 从 kwargs 中提取 weight_path，如果存在则传递给 get_model
        weight_path = kwargs.pop('weight_path', None)
        print(f"[{self.category}] predict 被调用: model_type={model_type}, weight_path={weight_path}, smooth={smooth}")
        wrapper = self.get_model(model_type, weight_path=weight_path)
        if not wrapper:
            print(f"[{self.category}] 无法加载模型: {model_type}")
            return []

        print(f"[{self.category}] 调用 wrapper.predict, 图像形状: {image.shape}")
        results = wrapper.predict(image, img_size=img_size, **kwargs)
        print(f"[{self.category}] wrapper.predict 返回 {len(results)} 个结果")

        if refine and results:
            refine_method = kwargs.get('refine_method', 'vitmatte')
            if refine_method != "hybrid":
                refine_model_type = "vitmatte" if refine_method == "vitmatte" else "cascadepsp"
                refiner = ModelFactory.create(refine_model_type)
                if refiner:
                    for result in results:
                        if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
                            mask_data = result.masks.data.cpu().numpy()
                            mask_list = [(mask_data[i] * 255).astype(np.uint8) for i in range(mask_data.shape[0])]
                            refined_results = refiner.predict(
                                image, masks=mask_list,
                                fast=kwargs.get('fast', False),
                                L=kwargs.get('L', 900 if refine_method == 'vitmatte' else 320),
                                use_roi=kwargs.get('use_roi', True),
                                margin=kwargs.get('margin', 0.15),
                                trimap_thickness=kwargs.get('trimap_thickness', 20),
                            )
                            if refined_results and hasattr(refined_results[0], 'masks'):
                                result.masks.data = refined_results[0].masks.data
                    # Cache refiner for reuse
                    self.models[f"_refiner_{refine_method}"] = refiner
        
        # 将掩码转换为多边形 (无论是否平滑都执行此操作)
        if results:
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:

                    # 获取掩码数据 (N, H, W)
                    mask_data = result.masks.data.cpu().numpy()
                    img_region = kwargs.get('region', None)
                    region_offset_x, region_offset_y = 0, 0
                    if img_region is not None and len(img_region) > 0:
                        rx1, ry1, rw, rh = img_region
                        mask_data = mask_data[:, ry1:ry1+rh, rx1:rx1+rw]
                        region_offset_x, region_offset_y = rx1, ry1

                    # 根据新的掩码更新 boxes，并过滤掉全零的 mask
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        orig_box_data = None
                        if hasattr(result.boxes, 'data') and result.boxes.data is not None:
                            orig_box_data = result.boxes.data.clone()

                        valid_indices = []
                        new_boxes = []
                        for i in range(mask_data.shape[0]):
                            mask = mask_data[i]
                            # 找到掩码的非零像素
                            y_indices, x_indices = np.where(mask > 0)
                            if len(y_indices) > 0 and len(x_indices) > 0:
                                mx1, my1 = x_indices.min(), y_indices.min()
                                mx2, my2 = x_indices.max(), y_indices.max()
                                # 加回 region 偏移，转换回原图坐标
                                mx1 += region_offset_x
                                my1 += region_offset_y
                                mx2 += region_offset_x
                                my2 += region_offset_y
                                # 存为 xyxy 格式，以兼容 Ultralytics Boxes.xywh 属性
                                new_boxes.append([mx1, my1, mx2, my2])
                                valid_indices.append(i)
                            # 如果掩码为空（全零），跳过，不保留

                        # 只保留有效的 mask 和 boxes
                        if valid_indices:
                            mask_data = mask_data[valid_indices]
                            result.masks.data = torch.from_numpy(mask_data)
                            new_box_tensor = torch.tensor(new_boxes, dtype=torch.float32)
                            if orig_box_data is not None and orig_box_data.shape[0] >= len(valid_indices):
                                orig_valid = orig_box_data[valid_indices]
                                if orig_valid.shape[1] >= 6:
                                    # 保留原始 conf/cls，避免多文本标注时类别全部变成第一个提示词
                                    new_box_tensor = torch.cat([new_box_tensor, orig_valid[:, 4:6].cpu()], dim=1)
                            result.boxes.data = new_box_tensor
                        else:
                            # 如果没有有效的 mask，清空 masks 和 boxes
                            result.masks = None
                            result.boxes = None

                    all_polygons = []

                    for i in range(mask_data.shape[0]):
                        # 转换为 0-255 uint8
                        mask = (mask_data[i] * 255).astype(np.uint8)
                        # 提取多边形
                        if smooth:
                            # 使用平滑处理
                            polygon = self._smooth_mask_contours(mask, epsilon=smooth_epsilon)
                        else:
                            # 不使用平滑，直接提取轮廓并合并
                            mask_uint8 = mask.astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            polygon = merge_multi_contours(contours)

                        all_polygons.append([polygon] if len(polygon) > 0 else [])

                    # 将多边形数据附加到结果对象上
                    result.polygons = all_polygons

        return results

    def release_models(self):
        """释放所有加载的模型"""
        # Store keys to avoid runtime error during iteration if dictionary changes
        model_types = list(self.models.keys())
        for model_type in model_types:
            try:
                self.models[model_type].release()
            except Exception as e:
                print(f"[{self.category}] Error releasing model {model_type}: {e}")
        self.models.clear()

        # Don't release external models here as they might be managed elsewhere or needed later?
        # Actually, for InteractivePredictor, we want to clear everything to save VRAM.
        # But let's be careful with external references.
        # The user intent is usually to switch models within the same session.
        # If we registered it externally, we should probably keep it? 
        # But 'release_models' implies cleaning up resources.
        # Let's release external ones too, as they consume VRAM.
        ext_types = list(self._external_models.keys())
        for model_type in ext_types:
            try:
                self._external_models[model_type].release()
            except Exception as e:
                print(f"[{self.category}] Error releasing external model {model_type}: {e}")
        self._external_models.clear()

        # 强制回收并清理缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[{self.category}] 所有模型已释放。")


class InteractivePredictor(BaseSegmentor):
    def __init__(self):
        super().__init__("InteractivePredictor")

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, model_type="mobilesam_onnx",
                refine=True, smooth=False, smooth_epsilon=0.001, **kwargs):
        return super().predict(image, model_type, refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon,
                               bboxes=bboxes, points=points, labels=labels, texts=texts, conf=conf, **kwargs)


class ExamplePredictor(BaseSegmentor):
    def __init__(self):
        super().__init__("ExamplePredictor")

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, model_type="trtsam3",
                geom_label=None, refine=True, smooth=False, smooth_epsilon=0.001, img_size=640, **kwargs):
        """执行预测"""
        return super().predict(image, model_type, refine=refine, smooth=smooth, smooth_epsilon=smooth_epsilon,
                               bboxes=bboxes, points=points, labels=labels, texts=texts, conf=conf,
                               geom_label=geom_label, img_size=img_size, **kwargs)


# Global Wrapper Functions
def get_interactive_predictor():
    global _interactive_predictor
    if _interactive_predictor is None:
        _interactive_predictor = InteractivePredictor()
    return _interactive_predictor


def get_example_predictor():
    global _example_predictor
    if _example_predictor is None:
        _example_predictor = ExamplePredictor()
    return _example_predictor


def load_models():
    """兼容性函数：预加载默认模型，并加载自定义模型"""
    # 先加载自定义模型到工厂
    _load_custom_models_from_config()

    print("[core] 正在预加载默认模型 (MobileSAM & TRTSAM3)...")
    # 显著拉大分步加载间隔，给 UI 交互留出充足的响应窗口
    get_interactive_predictor().get_model("mobilesam_onnx")
    time.sleep(1.0)
    get_example_predictor().get_model("trtsam3")
    print("[core] 默认模型 (MobileSAM & TRTSAM3) 加载完成。")


def _load_custom_models_from_config():
    """从 core.json 加载自定义模型配置并注册到 ModelFactory"""
    from .path_resolver import get_custom_models
    for role in ["interactive", "auto", "plain"]:
        models = get_custom_models(role)
        for model_info in models:
            name = model_info.get("name", "")
            category = model_info.get("category", "")
            weight_path = model_info.get("weight_path", "")
            if name and category and weight_path:
                ModelFactory.register_custom_model(name, category, weight_path, role=role)


def release_all_models():
    """释放所有全局模型实例及其资源"""
    print("[core] 正在彻底销毁所有模型资源...")
    if _interactive_predictor is not None:
        _interactive_predictor.release_models()
    if _example_predictor is not None:
        _example_predictor.release_models()

    # 强制进行垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[core] 所有模型资源彻底销毁完成。")


def interactive_predict(image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, model_type="mobilesam_onnx", refine=True,
            img_size=640, **kwargs):
    """使用 InteractivePredictor 进行预测"""
    return get_interactive_predictor().predict(image, bboxes, points, labels, texts, conf, model_type, refine=refine,
                                               img_size=img_size, **kwargs)


def example_predict(image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, model_type="trtsam3",
                           geom_label=None, refine=True, img_size=640, **kwargs):
    """使用 ExamplePredictor 进行预测"""
    return get_example_predictor().predict(image, bboxes, points, labels, texts, conf, model_type, geom_label,
                                        refine=refine, img_size=img_size, **kwargs)
