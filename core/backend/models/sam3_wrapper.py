import numpy as np
import torch
import gc
import os
# 注意：根据用户提供的代码，需要从特殊的 Predictor 类导入
from ultralytics.models.sam import SAM3SemanticPredictor
from .base import ModelInterface
from .registry import register_model
from ..utils import _cleanup_runs


def _ensure_tokenizer_callable():
    """兼容 ultralytics 8.4.x 的 SAM3 与标准 clip 包。

    ultralytics 的 SAM3 文本编码器按「可调用版 CLIP(fork)」编写，内部以
    `self.tokenizer(text, context_length=...)` 调用分词器并期望返回
    [b, ctx_len] 的 token 张量（0 为 padding）；而环境安装的标准
    `clip.simple_tokenizer.SimpleTokenizer` 只有 encode() 且不可调用，
    会抛 `'SimpleTokenizer' object is not callable`。
    此处为 SimpleTokenizer 补上 __call__，产出与 fork 一致的
    [<start>] + encode(text) + [<end>] 序列（截断/补零到 context_length）。
    """
    try:
        from clip.simple_tokenizer import SimpleTokenizer
    except Exception:
        return
    # 注意：不能用 getattr(SimpleTokenizer, "__call__") 判断——那会命中元类
    # type.__call__ 恒非 None；须检查类自身 __dict__ 是否定义了可调用实现
    if "__call__" in SimpleTokenizer.__dict__:
        return  # 已是可调用实现（fork 版），无需补丁

    def _call(self, texts, context_length):
        if isinstance(texts, str):
            texts = [texts]
        sot = self.encoder["<|startoftext|>"]
        eot = self.encoder["<|endoftext|>"]
        batch = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, t in enumerate(texts):
            tokens = [sot] + list(self.encode(t)) + [eot]
            tokens = tokens[:context_length]
            if tokens:
                batch[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return batch

    SimpleTokenizer.__call__ = _call


@register_model("sam3", img_size=1008, weight_path_key="sam3_path", category="sam3", role="example")
class SAM3Wrapper(ModelInterface):
    """Ultralytics SAM3 语义模型包装器 (专门用于多模态示例增强)"""
    def __init__(self):
        self.predictor = None
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
        from ..path_resolver import get_path
        try:
            # 先补 SimpleTokenizer 可调用接口，避免 ultralytics 8.4.x SAM3
            # 文本编码器 `self.tokenizer(text, ...)` 抛 `not callable`
            _ensure_tokenizer_callable()
            if not weight_path or not os.path.exists(weight_path):
                weight_path = get_path("sam3_path")
            
            if not os.path.exists(weight_path):
                print(f"[SAM3Wrapper] 错误: 权重文件不存在 {weight_path}")
                return False
            try:
                # 参考用户代码：使用 SAM3SemanticPredictor 并设置 overrides
                self.predictor = SAM3SemanticPredictor(overrides=dict(
                    conf=0.5, 
                    task="segment", 
                    mode="predict", 
                    model=weight_path, 
                    half=True, # 改为 False 尝试解决部分环境下的属性错误
                    save=False, 
                    imgsz=(644, 644), 
                    verbose=False
                ),bpe_path=get_path("sam3_bpe_path"))
            except Exception as e:
                print(f"[SAM3Wrapper] only ultralytics 8.0237 needs bpe_path, {e}, try without bpe_path")
                # 参考用户代码：使用 SAM3SemanticPredictor 并设置 overrides
                self.predictor = SAM3SemanticPredictor(overrides=dict(
                    conf=0.5, 
                    task="segment", 
                    mode="predict", 
                    model=weight_path, 
                    half=True, # 改为 False 尝试解决部分环境下的属性错误
                    save=False, 
                    imgsz=(644, 644), 
                    verbose=False
                ))
            return True
        except Exception as e:
            print(f"[SAM3Wrapper] 加载失败: {e}")
            return False

    def get_features(self, image: np.ndarray) -> torch.Tensor:
        """提取图像特征"""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
            
        # Set image to predictor
        # 某些版本的 Ultralytics SAM3 在 set_image 后调用会触发 '_prepare_backbone_features' 属性错误
        # 但为了获取特征，必须调用 set_image
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

    def reset_cache(self):
        """重置模型缓存"""
        self._last_image = None
        if self.predictor:
            # 清理特征缓存
            if hasattr(self.predictor, 'features'):
                self.predictor.features = None
            if hasattr(self.predictor, 'is_image_set'):
                self.predictor.is_image_set = False

    def predict(self, image, bboxes=None, points=None, labels=None, texts=None, conf=0.5, img_size=644, **kwargs):
        if self.predictor is None:
            return []
        
        if img_size != getattr(self, '_last_img_size', 644):
            self._last_img_size = img_size
            if hasattr(self.predictor, 'args'):
                self.predictor.args.imgsz = (img_size, img_size)
        
        # 准备提示词
        bboxes_xyxy = None
        if bboxes:
            bboxes_xyxy = [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in bboxes]

        try:
            # 某些版本的 Ultralytics SAM3 在 set_image 后调用会触发 '_prepare_backbone_features' 属性错误
            self._set_image_if_needed(image)
            results = self.predictor(
                bboxes=bboxes_xyxy, 
                text=texts, 
                save=False
            )
            _cleanup_runs()
            return results
        except Exception as e:
            print(f"[SAM3Wrapper] 推理失败: {e}")
            return []

    def release(self):
        if self.predictor is not None:
            print("[SAM3Wrapper] 释放语义预测器资源")
            self.predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
