"""
ActionTool — 将单个 action 包装为 CoreCoder Tool

每个 action 对应一个独立的 ActionTool 实例，带有完整的参数 schema，
让 LLM 能直接看到每个 action 的名称、描述和参数定义。
所有 action 执行通过主线程调度器，避免跨线程操作 Qt 控件。
"""

import json
from core.corecoder.tools import Tool


class ActionTool(Tool):
    """将 VisionMind 的单个 action 包装为 CoreCoder Tool"""

    def __init__(self, action_name, display_name, description, parameters_schema):
        """
        Args:
            action_name: ActionRegistry 中的完整 action 名称（如 "annotation.next_image"）
            display_name: 暴露给 LLM 的 tool 名称（如 "next_image"）
            description: action 描述
            parameters_schema: JSON Schema 格式的参数定义
        """
        self.name = display_name
        self.description = description
        self.parameters = parameters_schema
        self._action_name = action_name
        # 只读/执行分级(计划模式过滤 + 只读工具并行分组用),取自注册表 meta
        # background=True 的重计算动作(推理/批量/导出)直接在调用线程执行,
        # 不经主线程调度器——否则推理期间整个界面冻结
        self.read_only = False
        self.background = False
        try:
            from core.common.action_registry import ActionRegistry
            meta = ActionRegistry.instance().get_meta(action_name)
            if meta:
                self.read_only = bool(meta.read_only)
                self.background = bool(meta.background)
        except Exception:
            pass

    def execute(self, **kwargs) -> str:
        """执行 action 并返回结果。

        read_only(查询 UI 状态)与未标记动作在主线程执行,避免跨线程操作控件;
        background 重计算动作在调用线程执行,UI 刷新事件由 EventBus 自动
        marshal 回主线程。
        """
        try:
            from core.common.action_registry import ActionRegistry

            def _do_call():
                registry = ActionRegistry.instance()
                return registry.call(self._action_name, params=kwargs)

            if self.background:
                print(f"[ActionTool] {self.name} 后台线程执行(不阻塞UI)")
                result = _do_call()
            else:
                from core.common.main_thread_dispatcher import run_on_main
                result = run_on_main(_do_call)

            # 元组返回约定：(data..., error)，末位为错误字符串或含 error 键的 dict 时视为失败
            tuple_error = self._extract_tuple_error(result)
            if tuple_error:
                return f"❌ {self.name} 失败: {tuple_error}"

            if result is None:
                return f"✅ {self.name} 执行成功"
            elif isinstance(result, str):
                # action 返回错误消息（以"错误"开头的字符串）
                if result.startswith("错误"):
                    return f"❌ {self.name} 失败: {result}"
                return f"✅ {self.name} 执行成功\n返回: {result}"
            elif isinstance(result, bool):
                return f"✅ {self.name} 执行成功" if result else f"❌ {self.name} 执行失败"
            elif isinstance(result, (dict, list, tuple)):
                return f"✅ {self.name} 执行成功\n返回: {self._serialize(result)}"
            else:
                return f"✅ {self.name} 执行成功\n返回: {result}"
        except KeyError:
            return f"❌ 错误: action '{self._action_name}' 未注册"
        except Exception as e:
            return f"❌ 错误: {self.name} 执行失败: {type(e).__name__}: {e}"

    @staticmethod
    def _extract_tuple_error(result):
        """从元组返回中提取错误信息。返回错误字符串，无错误返回 None。"""
        if not isinstance(result, tuple) or not result:
            return None
        last = result[-1]
        if isinstance(last, str) and last.startswith("错误"):
            return last
        if isinstance(last, dict):
            err = last.get("error")
            if isinstance(err, str) and err:
                return err
        return None

    def _serialize(self, data):
        """将结果序列化为 LLM 可读的 JSON 文本，处理 numpy/Qt 等不可序列化对象。"""
        try:
            return json.dumps(self._to_jsonable(data), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(data)

    @classmethod
    def _to_jsonable(cls, obj):
        """递归将对象转换为 JSON 可序列化结构。"""
        if isinstance(obj, dict):
            return {str(k): cls._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_jsonable(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            return [cls._to_jsonable(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except Exception:
            pass
        get_rgb = getattr(obj, 'getRgb', None)
        if callable(get_rgb):
            try:
                return list(get_rgb()[:3])
            except Exception:
                pass
        name = getattr(obj, 'name', None)
        if callable(name):
            try:
                return name()
            except Exception:
                pass
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)
