"""模型路径解析器 - 从 core.json 读取模型路径配置，替代 config.py"""
import json
import os
from typing import Optional, Dict, Any

# 项目根目录：core/backend/ -> 项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "core", "core.json")

# 缓存
_data: Dict[str, Any] = {}
_loaded = False


def _load_config():
    """加载 core.json 中的 model_paths 配置"""
    global _data, _loaded
    if _loaded:
        return
    _loaded = True
    if not os.path.exists(_CONFIG_PATH):
        print(f"[path_resolver] 配置文件不存在: {_CONFIG_PATH}")
        return
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8-sig') as f:
            full_data = json.load(f)
        # core.json 支持两种格式：
        # 1. 扁平数组 [{key, value}, ...]（model_paths 段落）
        # 2. 嵌套对象 {model_paths: {files: {...}}}
        if isinstance(full_data, list):
            # 扁平数组格式：转为 {files: {key: value, ...}}
            files = {}
            for item in full_data:
                if isinstance(item, dict) and "key" in item:
                    files[item["key"]] = item.get("value", item.get("default", ""))
            _data = {"files": files}
        else:
            _data = full_data.get("model_paths", {})
    except Exception as e:
        print(f"[path_resolver] 加载配置失败: {e}")


def _resolve_path(relative_or_absolute: str) -> str:
    """将路径解析为绝对路径。相对路径基于项目根目录。"""
    if not relative_or_absolute:
        return ""
    if os.path.isabs(relative_or_absolute):
        return relative_or_absolute
    return os.path.join(_PROJECT_ROOT, relative_or_absolute)


def _ensure_dir(path: str):
    """确保目录存在"""
    if path:
        dir_path = path if os.path.isdir(path) or path.endswith('/') or path.endswith('\\') else os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


def get_path(key: str) -> str:
    """获取模型文件路径（绝对路径）"""
    _load_config()
    files = _data.get("files", {})
    relative = files.get(key, "")
    return _resolve_path(relative)


def get_dir(key: str) -> str:
    """获取模型目录路径（绝对路径），并确保目录存在"""
    _load_config()
    # 优先从 dirs 子对象取，其次从 files 取（扁平数组格式兼容）
    dirs = _data.get("dirs", {})
    relative = dirs.get(key, "")
    if not relative:
        files = _data.get("files", {})
        relative = files.get(key, "")
    abs_path = _resolve_path(relative)
    _ensure_dir(abs_path)
    return abs_path


def get_url(filename: str) -> Optional[str]:
    """获取模型文件下载 URL"""
    _load_config()
    urls = _data.get("download_urls", {})
    if not urls:
        # 扁平数组格式兼容：download_urls 可能存在 files 中
        files = _data.get("files", {})
        dl = files.get("download_urls", {})
        if isinstance(dl, dict):
            return dl.get(filename)
    return urls.get(filename)


def get_weights_dir() -> str:
    """获取权重根目录（绝对路径）"""
    _load_config()
    # 优先从顶层取，其次从 files 取
    relative = _data.get("weights_dir", "")
    if not relative:
        files = _data.get("files", {})
        relative = files.get("weights_dir", "weights")
    abs_path = _resolve_path(relative)
    _ensure_dir(abs_path)
    return abs_path


def get_custom_models(role: str = "interactive") -> list:
    """获取自定义模型列表"""
    _load_config()
    custom = _data.get("custom_models", {})
    if not custom:
        # 扁平数组格式兼容
        files = _data.get("files", {})
        cm = files.get("custom_models", {})
        if isinstance(cm, dict):
            return cm.get(role, [])
    return custom.get(role, [])


def save_custom_models(role, models):
    """保存自定义模型列表到 core.json，保持原文件格式不变（不破坏 schema 元数据）。"""
    _load_config()
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8-sig') as f:
            full_data = json.load(f)
    except Exception:
        full_data = {}

    if isinstance(full_data, list):
        # schema 数组格式：不转 dict、不丢元数据，仅更新 key='custom_models' 的条目
        item = None
        for it in full_data:
            if isinstance(it, dict) and it.get("key") == "custom_models":
                item = it
                break
        if item is None:
            item = {"key": "custom_models", "title": "自定义模型", "type": "object",
                    "value": {"interactive": [], "auto": [], "plain": [], "refine": []}}
            full_data.append(item)
        cm = item.setdefault("value", {})
        if not isinstance(cm, dict):
            cm = {"interactive": [], "auto": [], "plain": [], "refine": []}
            item["value"] = cm
        for r in ("interactive", "auto", "plain", "refine"):
            cm.setdefault(r, [])
        cm[role] = models
        # 同步内存缓存（files 路径供 get_custom_models 读取）
        files = _data.setdefault("files", {})
        files["custom_models"] = cm
        _data["custom_models"] = cm
    else:
        # dict（旧格式）：保持 model_paths 结构
        if "model_paths" not in full_data:
            full_data["model_paths"] = {}
        if "custom_models" not in full_data["model_paths"]:
            full_data["model_paths"]["custom_models"] = {"interactive": [], "auto": [], "plain": [], "refine": []}
        full_data["model_paths"]["custom_models"][role] = models
        _data["custom_models"] = full_data["model_paths"]["custom_models"]

    with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)


def get_all_download_urls() -> Dict[str, str]:
    """获取所有下载 URL 映射"""
    _load_config()
    urls = _data.get("download_urls", {})
    if not urls:
        # 扁平数组格式兼容
        files = _data.get("files", {})
        dl = files.get("download_urls", {})
        if isinstance(dl, dict):
            return dl
    return urls



