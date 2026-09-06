"""
基于装饰器的 Action 自动注册框架

使用方式：
    @action("select_image", description="选择图片", category="标注", params={"image_index": "int"})
    def select_image(self, image_index: int = 0):
        ...

装饰器在类定义时只做标记，运行时通过 ActionRegistry.register_instance() 扫描实例并注册。
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable


# 嵌套调用深度守卫：action 内再调用 action 时，只记录最外层 action，
# 避免 UI action 委托 service action 时录制重复步骤导致回放重复执行。
_nested_call_depth = threading.local()


@dataclass
class ActionMeta:
    """Action 元信息"""
    name: str
    description: str = ""
    category: str = "通用"
    scope: str = "ui"  # "agent"=AI可调用  "ui"=仅供工程回放
    params: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = False  # True=纯查询/分析(计划模式可用) False=会修改数据/文件/配置
    background: bool = False  # True=重计算,agent 调用时在后台线程执行(不阻塞主线程)
    # verify 已移除（原用于执行后状态验证，因缺少 before-state 快照导致验证无效）

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "scope": self.scope,
            "params": self.params,
            "read_only": self.read_only,
            "background": self.background,
        }


def action(name: str, description: str = "", category: str = "通用", scope: str = "ui", params: Dict = None, read_only: bool = False, background: bool = False):
    """
    标记一个方法为可调用的 action。

    装饰器在类定义时执行，将 ActionMeta 附加到函数对象上，
    并包装函数使其执行后自动通知录制器。
    运行时由 ActionRegistry.register_instance() 扫描并注册。

    Args:
        name: action 名称（全局唯一）
        description: 功能描述
        category: 分类（如 "标注"、"输入模拟"、"断言"）
        scope: "agent"=AI可调用  "ui"=仅供工程回放（默认）
        params: 参数描述 {"param_name": "type_or_description"}
        read_only: True=纯查询/分析类（计划模式下仍可见可调用）
        background: True=重计算类（推理/批量/导出等），agent 调用时在后台线程执行，不冻结界面
    """
    meta = ActionMeta(name=name, description=description, category=category,
                      scope=scope, params=params or {}, read_only=read_only,
                      background=background)

    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            depth = getattr(_nested_call_depth, 'depth', 0)
            _nested_call_depth.depth = depth + 1
            try:
                # 仅在非嵌套（最外层）调用时录制，避免 UI action 委托
                # service action 时内外两层都被录制、回放重复执行。
                if depth == 0:
                    # 提取调用参数（排除 self/cls）
                    call_params = {}
                    import inspect
                    try:
                        sig = inspect.signature(func)
                        param_names = list(sig.parameters.keys())
                        start = 1 if param_names and param_names[0] in ('self', 'cls') else 0
                        effective_names = param_names[start:]
                        for i, val in enumerate(args[start:]):
                            if i < len(effective_names):
                                call_params[effective_names[i]] = val
                        call_params.update(kwargs)
                    except Exception:
                        call_params = dict(kwargs)

                    # 在调用 func 之前录制，确保对于内部调用 dialog.exec_() 的 action，
                    # action 本身先于对话框内部操作被录制（录制顺序与用户操作顺序一致）。
                    from core.common.action_recorder import ActionRecorder
                    recorder = ActionRecorder.instance()
                    if recorder.is_recording:
                        # 过滤不可序列化的值
                        record_params = {}
                        for k, v in call_params.items():
                            try:
                                import json
                                json.dumps(v)
                                record_params[k] = v
                            except (TypeError, ValueError):
                                record_params[k] = str(v)
                        recorder.record(meta.name, record_params)

                    # OperationLogger 始终后台记录（供意图分析 Agent 使用）
                    before_state = {}
                    try:
                        from core.agent.operation_logger import OperationLogger, capture_state_snapshot
                        op_logger = OperationLogger.instance()
                        before_state = capture_state_snapshot()
                    except Exception:
                        op_logger = None

                    result = func(*args, **kwargs)

                    if op_logger is not None:
                        try:
                            after_state = capture_state_snapshot()
                            op_logger.record(meta.name, dict(call_params), {}, before_state, after_state)
                        except Exception:
                            pass
                    return result
                return func(*args, **kwargs)
            finally:
                _nested_call_depth.depth = depth

        wrapper._action_meta = meta
        return wrapper

    return decorator


class ActionRegistry:
    """
    运行时 action 注册表

    管理所有已注册的 action，支持：
    - 通过 @action 装饰器自动发现
    - 别名机制（兼容旧名称）
    - AI 助手工具 schema 导出
    """

    _instance: Optional['ActionRegistry'] = None

    def __init__(self):
        # name -> (instance, bound_method, meta)
        self._actions: Dict[str, tuple] = {}
        # alias -> canonical_name
        self._aliases: Dict[str, str] = {}

    @classmethod
    def instance(cls) -> 'ActionRegistry':
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例（仅用于测试）"""
        cls._instance = None

    def register_instance(self, instance: object, prefix: str = ""):
        """
        扫描实例的类 MRO，注册所有带 @action 装饰器的方法。

        Args:
            instance: 要扫描的对象实例
            prefix: 名称前缀（如 "annotation"），最终 action 名为 "prefix.name"。
                     如果 meta.name 已以 "prefix." 开头，则不重复添加 prefix。
        """
        registered = []
        seen_names = set()  # 避免子类覆盖父类方法时重复注册

        for cls in type(instance).__mro__:
            if cls is object:
                continue
            for attr_name in vars(cls):
                if attr_name in seen_names:
                    continue
                attr = vars(cls).get(attr_name)
                if attr is None:
                    continue
                # 检查是否是函数且有 _action_meta
                if callable(attr) and hasattr(attr, '_action_meta'):
                    meta = attr._action_meta
                    # 如果 meta.name 已以 "prefix." 开头，则不再重复添加
                    if prefix and meta.name.startswith(f"{prefix}."):
                        full_name = meta.name
                    else:
                        full_name = f"{prefix}.{meta.name}" if prefix else meta.name
                    if full_name not in self._actions:
                        bound = getattr(instance, attr_name)
                        self._actions[full_name] = (instance, bound, meta)
                        registered.append(full_name)
                    seen_names.add(attr_name)

        if registered:
            print(f"[ActionRegistry] 注册 {len(registered)} 个 action 来自 {type(instance).__name__}: {registered}")

    def register_alias(self, alias: str, canonical_name: str):
        """
        注册别名，使旧名称仍可使用。

        Args:
            alias: 别名（如 "select_image"）
            canonical_name: 规范名称（如 "annotation.select_image"）
        """
        self._aliases[alias] = canonical_name

    def call(self, name: str, params: Dict = None, **kwargs) -> Any:
        """
        调用已注册的 action。

        Args:
            name: action 名称（支持别名）
            params: 参数字典（优先级高于 kwargs）
            **kwargs: 传递给 action 方法的参数

        Returns:
            action 方法的返回值

        Raises:
            KeyError: action 未注册
        """
        resolved = self._resolve_name(name)
        if resolved not in self._actions:
            raise KeyError(f"Action '{name}' 未注册 (已解析为 '{resolved}')")
        _, bound, meta = self._actions[resolved]
        # 合并参数
        all_kwargs = {}
        if params:
            all_kwargs.update(params)
        all_kwargs.update(kwargs)
        # 过滤掉不属于目标方法参数的 kwargs
        all_kwargs = self._filter_kwargs(bound, all_kwargs)
        result = bound(**all_kwargs)

        return result

    def has_action(self, name: str) -> bool:
        """检查 action 是否存在"""
        resolved = self._resolve_name(name)
        return resolved in self._actions

    def get_action(self, name: str) -> Optional[Callable]:
        """获取 action 的可调用方法"""
        resolved = self._resolve_name(name)
        if resolved in self._actions:
            return self._actions[resolved][1]
        return None

    def get_instance(self, name: str):
        """获取 action 所属实例"""
        resolved = self._resolve_name(name)
        if resolved in self._actions:
            return self._actions[resolved][0]
        return None

    def get_meta(self, name: str) -> Optional[ActionMeta]:
        """获取 action 的元信息"""
        resolved = self._resolve_name(name)
        if resolved in self._actions:
            return self._actions[resolved][2]
        return None

    def list_actions(self, category: str = None) -> List[ActionMeta]:
        """
        列出所有已注册的 action。

        Args:
            category: 按分类过滤（可选）
        """
        metas = [meta for _, _, meta in self._actions.values()]
        if category:
            metas = [m for m in metas if m.category == category]
        return sorted(metas, key=lambda m: (m.category, m.name))

    def list_categories(self) -> List[str]:
        """列出所有分类"""
        categories = set()
        for _, _, meta in self._actions.values():
            categories.add(meta.category)
        return sorted(categories)

    def get_tool_schema(self) -> List[Dict]:
        """
        生成 AI 助手可用的工具调用 schema（OpenAI Function Calling 格式）。

        支持 params 的三种格式：
        - 格式1（简单字符串）: {"name": "str"} → {"type": "string"}
        - 格式2（带描述）: {"name": {"type": "str", "description": "xxx"}}
        - 格式3（带enum）: {"name": {"type": "str", "enum": [...], "description": "xxx"}}
        - 格式4（动态enum）: {"name": {"type": "str", "dynamic_enum": "xxx"}}

        Returns:
            工具描述列表
        """
        tools = []
        for name, (_, _, meta) in sorted(self._actions.items()):
            properties = {}
            required = []
            for param_name, param_def in meta.params.items():
                prop = self._parse_param_schema(param_def)
                properties[param_name] = prop
                if self._param_required(param_def):
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    @staticmethod
    def _parse_param_schema(param_def) -> Dict:
        """
        解析单个参数的 schema，支持字符串和字典两种格式。

        Args:
            param_def: 参数定义，可以是：
                - "str" / "int" 等简单类型字符串
                - {"type": "str", "description": "xxx", "enum": [...], "dynamic_enum": "xxx"}
        """
        if isinstance(param_def, str):
            # 格式1: 简单类型字符串
            type_str = "string"
            ptype = param_def.lower().strip()
            if ptype in ("int", "integer"):
                type_str = "integer"
            elif ptype in ("float", "number", "double"):
                type_str = "number"
            elif ptype in ("bool", "boolean"):
                type_str = "boolean"
            elif ptype in ("list", "array"):
                type_str = "array"
            elif ptype in ("dict", "object"):
                type_str = "object"
            return {"type": type_str}

        if isinstance(param_def, dict):
            # 格式2/3/4: 字典格式
            result = {}

            # 类型
            raw_type = param_def.get("type", "str").lower().strip()
            type_map = {
                "str": "string", "string": "string",
                "int": "integer", "integer": "integer",
                "float": "number", "number": "number", "double": "number",
                "bool": "boolean", "boolean": "boolean",
                "list": "array", "array": "array",
                "dict": "object", "object": "object",
            }
            result["type"] = type_map.get(raw_type, "string")

            # 描述
            if "description" in param_def:
                result["description"] = param_def["description"]

            # 静态 enum
            if "enum" in param_def:
                result["enum"] = param_def["enum"]

            # 动态 enum 标记（运行时由 DynamicToolManager 解析）
            if "dynamic_enum" in param_def:
                result["dynamic_enum"] = param_def["dynamic_enum"]

            return result

        # fallback
        return {"type": "string"}

    @staticmethod
    def _param_required(param_def) -> bool:
        """判断参数是否为必填。

        dict 格式的 params 支持 "required": true 标记，用于生成 OpenAI
        Function Calling schema 的 required 数组，让 LLM 知道必须提供该参数。
        """
        return isinstance(param_def, dict) and param_def.get("required") in (True, "true", "True")

    def _resolve_name(self, name: str) -> str:
        """解析名称到规范名称：精确匹配 → 别名匹配 → 后缀匹配

        后缀匹配存在多候选（如 UI 的 category.add_category 与 service 的
        manual.add_category 都匹配 "add_category"）时，优先解析为 scope="ui"
        的 action，保证测试配置里的短名行为与纯 UI 架构一致；agent 始终使用
        完整注册名（如 annotation.manual.add_category），走精确匹配不受影响。
        """
        # 1. 精确匹配
        if name in self._actions:
            return name
        # 2. 别名匹配
        if name in self._aliases:
            return self._aliases[name]
        # 3. 后缀匹配：如 "toggle_tool" 匹配 "annotation.toggle_tool"
        suffix = f".{name}"
        matches = [canonical for canonical in self._actions if canonical.endswith(suffix)]
        if not matches:
            return name
        if len(matches) > 1:
            for canonical in matches:
                if self._actions[canonical][2].scope == "ui":
                    return canonical
        return matches[0]

    @staticmethod
    def _filter_kwargs(bound: Callable, kwargs: Dict) -> Dict:
        """
        过滤 kwargs，只保留目标方法接受的参数。

        避免因 step dict 中的多余字段导致 TypeError。
        """
        import inspect
        try:
            sig = inspect.signature(bound)
            params = sig.parameters
            # 如果方法有 **kwargs，直接返回所有参数
            for p in params.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return kwargs
            # 只保留方法接受的参数
            return {k: v for k, v in kwargs.items() if k in params}
        except (ValueError, TypeError):
            return kwargs

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, name: str) -> bool:
        return self.has_action(name)
