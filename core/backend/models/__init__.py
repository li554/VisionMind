import os
import importlib
import inspect
from pathlib import Path

from .base import ModelInterface, MockBoxes, MockMasks, MockResults
from .registry import get_registry, ModelType, register_model

# 自动扫描并导入所有 Wrapper 类
_wrapper_classes = {}
_models_dir = Path(__file__).parent

for _f in _models_dir.glob("*_wrapper.py"):
    _module_name = _f.stem
    try:
        _module = importlib.import_module(f".{_module_name}", package=__package__)
        for _name, _obj in inspect.getmembers(_module, inspect.isclass):
            if _name.endswith("Wrapper") and issubclass(_obj, ModelInterface) and _obj is not ModelInterface:
                globals()[_name] = _obj
                _wrapper_classes[_name] = _obj
    except Exception as e:
        print(f"[models] 加载 {_module_name} 失败: {e}")

# 便捷别名
SAMWrapper = _wrapper_classes.get("SAMWrapper")
SAM2Wrapper = _wrapper_classes.get("SAM2Wrapper")
HQSAMWrapper = _wrapper_classes.get("HQSAMWrapper")
TRTSAMWrapper = _wrapper_classes.get("TRTSAMWrapper")
YOLOEWrapper = _wrapper_classes.get("YOLOEWrapper")
SAM3Wrapper = _wrapper_classes.get("SAM3Wrapper")
FSWrapper = _wrapper_classes.get("FSWrapper")
SAMONNXWrapper = _wrapper_classes.get("SAMONNXWrapper")
HQSAMONNXWrapper = _wrapper_classes.get("HQSAMONNXWrapper")
SAM2ONNXWrapper = _wrapper_classes.get("SAM2ONNXWrapper")
YOLOTrainedWrapper = _wrapper_classes.get("YOLOTrainedWrapper")


def get_wrapper_class(name: str):
    """按名称获取 Wrapper 类"""
    return _wrapper_classes.get(name)


def get_all_wrapper_classes():
    """获取所有已发现的 Wrapper 类 {name: cls}"""
    return dict(_wrapper_classes)


__all__ = [
    'ModelInterface', 'MockBoxes', 'MockMasks', 'MockResults',
    'ModelType', 'register_model', 'get_registry',
    'get_wrapper_class', 'get_all_wrapper_classes',
] + list(_wrapper_classes.keys())
