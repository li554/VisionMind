# SPDX-License-Identifier: AGPL-3.0-or-later
"""AI 提供商与模型注册表（按提供商分组的多模型配置）。

以 object 型配置键 `ai_providers` 存储：

    {
      "providers": [
        {
          "id": "p123",
          "name": "Agnes",
          "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
          "api_key": "sk-...",
          "models": [
            {
              "name": "agnes-2.0-flash",
              "max_input_tokens": 16384,
              "max_output_tokens": 4096,
              "supports_image": true
            }
          ],
          "default_model": "agnes-2.0-flash"
        }
      ],
      "active_provider": "p123"
    }

每个模型是独立的配置对象（名称 / 输入上限 / 输出上限 / 是否支持图像），
而非普通字符串。兼容旧版两个来源自动规范化：
- 旧版 ai_api_key / ai_base_url / ai_model 单模型配置：首次访问迁移。
- 旧版字符串模型列表（"models": ["a"]）：自动归一化为模型对象。
"""
import copy
import hashlib
import os
import time

from core.common.config import Config
from core.common.settings import settings

PROVIDERS_KEY = "ai_providers"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_INPUT = 16384
DEFAULT_MAX_OUTPUT = 4096

# API Key 通过环境变量管理，明文不落 core.json：
# - 每个提供商一个变量：VISIONMIND_AI_KEY_<提供商ID>（ID 大写，非字母数字转下划线）
# - 全局兜底变量：VISIONMIND_AI_API_KEY（提供商专属变量缺失时使用）
# - 持久化文件：项目根目录 .env（纳入 .gitignore，不进版本库）
ENV_KEY_PREFIX = "VISIONMIND_AI_KEY_"
ENV_KEY_GLOBAL = "VISIONMIND_AI_API_KEY"
ENV_FILE = os.path.join(Config.ROOT_DIR, ".env")


def _get_data():
    """读取原始 provider 数据的深拷贝；无有效数据时返回 None。

    返回副本而非内部引用，确保后续原地修改后 settings.set 能正确触发保存与信号。
    """
    d = settings.get(PROVIDERS_KEY, None)
    if isinstance(d, dict) and isinstance(d.get("providers"), list):
        return copy.deepcopy(d)
    return None


def _save(data):
    settings.set(PROVIDERS_KEY, data)


def _load_env_file():
    """把项目根目录 .env 加载进当前进程环境（不覆盖已有环境变量）。"""
    if not os.path.exists(ENV_FILE):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        return
    except ImportError:
        pass
    # python-dotenv 缺失时的最小回退解析：逐行 KEY=VALUE
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def env_var_name(pid):
    """提供商 id 映射为环境变量名。"""
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(pid or "")).upper()
    return ENV_KEY_PREFIX + (safe or "DEFAULT")


def _write_env_var(name, value):
    """在 .env 写入 / 更新一个变量（value 为空则删除），保留其余内容。"""
    lines = []
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            lines = []
    prefix = name + "="
    out = []
    replaced = False
    for line in lines:
        if line.strip().startswith(prefix):
            replaced = True
            if value:
                out.append(f"{name}={value}")
            continue
        out.append(line)
    if value and not replaced:
        out.append(f"{name}={value}")
    text = "\n".join(out).strip("\n")
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write((text + "\n") if text else "")
    except OSError:
        pass


def get_provider_api_key(pid):
    """读取提供商密钥：优先专属环境变量，其次全局兜底变量。"""
    key = (os.environ.get(env_var_name(pid), "") or "").strip()
    if key:
        return key
    return (os.environ.get(ENV_KEY_GLOBAL, "") or "").strip()


def set_provider_api_key(pid, value):
    """设置提供商密钥：更新当前进程环境变量并写入 .env（空值则清除）。"""
    value = (value or "").strip()
    name = env_var_name(pid)
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    _write_env_var(name, value)


def _resolved(p):
    """返回把 api_key 换成环境变量值的提供商副本（仅供读取消费，不写回配置）。"""
    if not isinstance(p, dict):
        return p
    env_key = get_provider_api_key(p.get("id", ""))
    if env_key and env_key != p.get("api_key"):
        p = dict(p)
        p["api_key"] = env_key
    return p


def _migrate_plaintext_keys(providers):
    """把 core.json 中的明文 api_key 迁移到环境变量 / .env 并清空明文。"""
    changed = False
    for p in providers:
        stored = (p.get("api_key") or "").strip()
        if not stored:
            continue
        name = env_var_name(p.get("id", ""))
        if not (os.environ.get(name) or "").strip():
            os.environ[name] = stored
            _write_env_var(name, stored)
        p["api_key"] = ""
        changed = True
    return changed


def _maybe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_model(m):
    """把模型条目规范化为模型对象 dict（兼容字符串与 dict 两种旧格式）。"""
    if isinstance(m, dict):
        return {
            "name": str(m.get("name", "")).strip(),
            "max_input_tokens": _maybe_int(m.get("max_input_tokens", m.get("max_input", DEFAULT_MAX_INPUT)),
                                           DEFAULT_MAX_INPUT),
            "max_output_tokens": _maybe_int(m.get("max_output_tokens", m.get("max_tokens", DEFAULT_MAX_OUTPUT)),
                                            DEFAULT_MAX_OUTPUT),
            "supports_image": bool(m.get("supports_image", m.get("supportsImages", False))),
        }
    name = str(m).strip()
    return {
        "name": name,
        "max_input_tokens": DEFAULT_MAX_INPUT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT,
        "supports_image": False,
    }


def normalize_models(models):
    """把任意模型列表规范化为模型对象列表，并过滤空名。"""
    out = []
    for m in (models or []):
        if not isinstance(m, (dict, str)):
            continue
        nm = normalize_model(m)
        if nm["name"]:
            out.append(nm)
    return out


def _make_provider(pid, name, base_url, api_key, models, default_model=""):
    models = normalize_models(models)
    names = [m["name"] for m in models]
    return {
        "id": pid,
        "name": name or pid,
        "base_url": (base_url or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "api_key": (api_key or "").strip(),
        "models": models,
        "default_model": default_model or (names[0] if names else ""),
    }


def _gen_id():
    return "p" + hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]


def _host_name(base_url):
    """从 Base URL 推断一个可读的提供商名（取主机名）。"""
    url = (base_url or "").strip()
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    host = url.split("/")[0].split(":")[0]
    return host or "默认提供商"


def ensure():
    """读取 provider 数据；无提供商时从旧版单模型配置迁移，并规范化模型结构。"""
    data = _get_data()
    if data is None:
        data = {"providers": [], "active_provider": ""}
    changed = False
    providers = list(data.get("providers") or [])

    # 尚无任何提供商时，从旧版单模型配置迁移（保留 agnes 等已有配置）
    if not providers:
        api_key = settings.get("ai_api_key", "")
        base_url = settings.get("ai_base_url", DEFAULT_BASE_URL)
        model = settings.get("ai_model", DEFAULT_MODEL)
        if api_key or model:
            providers.append(_make_provider(
                "default", _host_name(base_url), base_url, api_key, [model] if model else [],
                default_model=model,
            ))
            changed = True

    # 明文密钥迁移到环境变量 / .env，配置内不再保留明文
    if _migrate_plaintext_keys(providers):
        changed = True

    # 规范化模型结构与提供商字段
    for p in providers:
        norm = normalize_models(p.get("models", []))
        if norm != p.get("models", []):
            p["models"] = norm
            changed = True
        names = [m["name"] for m in norm]
        if not p.get("default_model") or p.get("default_model") not in names:
            p["default_model"] = names[0] if names else ""
            changed = True
        if not str(p.get("base_url", "")).strip():
            p["base_url"] = DEFAULT_BASE_URL
            changed = True

    data["providers"] = providers
    if providers:
        ids = {p.get("id") for p in providers}
        if data.get("active_provider") not in ids:
            data["active_provider"] = providers[0]["id"]
            changed = True
    else:
        if data.get("active_provider"):
            data["active_provider"] = ""
            changed = True

    if changed:
        _save(data)
    return data


def get_providers():
    return [_resolved(p) for p in ensure().get("providers", [])]


def model_names(models):
    """提取模型对象列表中的模型名称列表。"""
    return [m["name"] for m in (models or []) if isinstance(m, dict)]


def get_model(pid, model_name):
    """取某提供商下的模型对象；不存在返回 None。"""
    p = get_provider(pid)
    if p is None:
        return None
    for m in p.get("models", []):
        if isinstance(m, dict) and m.get("name") == model_name:
            return m
    return None


def get_provider(pid):
    for p in get_providers():
        if p.get("id") == pid:
            return p
    return None


def get_active_provider():
    data = ensure()
    pid = data.get("active_provider") or ""
    p = get_provider(pid)
    if p is None:
        providers = get_providers()
        if providers:
            p = providers[0]
            data["active_provider"] = p["id"]
            _save(data)
    return p or {}


def get_active_credentials():
    """返回 (api_key, base_url, model) —— 当前激活提供商的证书与默认模型。"""
    p = get_active_provider()
    names = model_names(p.get("models", []))
    model = p.get("default_model") or (names[0] if names else DEFAULT_MODEL)
    return (
        p.get("api_key", ""),
        p.get("base_url", DEFAULT_BASE_URL),
        model,
    )


def set_active_provider(pid):
    if get_provider(pid) is None:
        return False
    data = ensure()
    if data.get("active_provider") != pid:
        data["active_provider"] = pid
        _save(data)
    return True


def add_provider(name, base_url, api_key, models, default_model=""):
    pid = _gen_id()
    p = _make_provider(pid, name, base_url, "", models, default_model)
    data = ensure()
    data["providers"].append(p)
    if not data.get("active_provider") or get_provider(data["active_provider"]) is None:
        data["active_provider"] = pid
    set_provider_api_key(pid, api_key)
    _save(data)
    return _resolved(p)


def update_provider(pid, name=None, base_url=None, api_key=None,
                    models=None, default_model=None):
    data = ensure()
    p = None
    for item in data["providers"]:
        if item.get("id") == pid:
            p = item
            break
    if p is None:
        return None
    if name is not None:
        p["name"] = name
    if base_url is not None:
        p["base_url"] = base_url.strip() or DEFAULT_BASE_URL
    if api_key is not None:
        set_provider_api_key(pid, api_key)
        p["api_key"] = ""
    if models is not None:
        p["models"] = normalize_models(models)
        names = model_names(p["models"])
        if not p.get("default_model") or p.get("default_model") not in names:
            p["default_model"] = names[0] if names else ""
    if default_model is not None:
        if default_model in model_names(p.get("models", [])):
            p["default_model"] = default_model
    _save(data)
    return p


def delete_provider(pid):
    data = ensure()
    before = len(data["providers"])
    data["providers"] = [p for p in data["providers"] if p.get("id") != pid]
    if len(data["providers"]) != before:
        if data.get("active_provider") == pid:
            data["active_provider"] = data["providers"][0]["id"] if data["providers"] else ""
        set_provider_api_key(pid, "")
        _save(data)
        return True
    return False


def grouped_models():
    """返回按提供商分组的模型名称列表：[(provider_id, provider_name, [model_names])]。"""
    return [(p["id"], p.get("name", p["id"]), model_names(p.get("models", [])))
            for p in get_providers()]


def all_models():
    """返回 {model: provider_id} 映射，用于定位某个模型所属提供商。"""
    out = {}
    for pid, _, names in grouped_models():
        for name in names:
            out[name] = pid
    return out


# 模块导入时即加载 .env，保证首次读取凭据前环境变量已就绪
_load_env_file()