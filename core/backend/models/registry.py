from enum import Enum
from typing import Dict, List, Optional
import inspect


class ModelType(str, Enum):
    INTERACTIVE = "interactive"
    EXAMPLE = "example"
    YOLO_TRAINED = "yolo_trained"
    REFINE = "refine"
    PLAIN = "plain"


# Global registry: populated by @register_model decorators at import time
_registry: Dict[str, dict] = {}


def register_model(
    *model_types: str,
    img_size: int = 640,
    weight_path_key: Optional[str] = None,
    category: Optional[str] = None,
    role: str = "example",
):
    """
    Decorator that registers a wrapper class for one or more model type names.

    Args:
        *model_types: One or more model type names (e.g. "sam_onnx_b", "mobilesam_onnx").
        img_size: Default image size for this model type.
        weight_path_key: Key(s) passed to path_resolver.get_path() to resolve the weight
            file path. For multi-path models (ONNX encoder+decoder, TRT), use comma-separated
            key names. If None, weight_path must be supplied at creation time.
        category: Category string for custom model registration mapping.
            If None, uses the first model_type as the category key.
        role: "interactive", "example" or "plain" — determines which model list this belongs to.
    """
    def decorator(cls):
        cat = category or model_types[0]
        needs_mt = _check_needs_model_type(cls)
        for mt in model_types:
            _registry[mt] = {
                "wrapper_class": cls,
                "model_type": mt,
                "img_size": img_size,
                "weight_path_key": weight_path_key,
                "category": cat,
                "role": role,
                "needs_model_type": needs_mt,
            }
        return cls
    return decorator


def _check_needs_model_type(cls) -> bool:
    """Check if the wrapper class __init__ requires a model_type parameter."""
    sig = inspect.signature(cls.__init__)
    params = [p for p in sig.parameters if p != 'self']
    return len(params) > 0 and params[0] == 'model_type'


def get_registry() -> Dict[str, dict]:
    """Return the full registry dict."""
    return dict(_registry)
