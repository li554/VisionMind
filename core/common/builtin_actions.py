"""
内置通用 Action

包含基础控制、断言、循环、变量等与特定界面无关的 action。
"""

import json
import os
import time
from os.path import normpath
from typing import Any, Dict, List

from core.common.action_registry import action


class BuiltinActions:
    """内置通用 action 集合"""

    def __init__(self, engine=None):
        """
        Args:
            engine: TestEngine 实例，用于访问 variables、timers 等状态
        """
        self._engine = engine

    def bind_engine(self, engine):
        """绑定 TestEngine 实例"""
        self._engine = engine

    # ==================== 对话框交互 ====================
    # 回放时由 ReplayDialogInterceptor 执行对应操作。
    # 统一入口 dialog.op 的 action 名称与 AutoInteractionRecorder 录制输出
    # 保持一致（录制名即注册名），确保录制和回放使用同一套 action 名。

    @action("dialog.op", description="统一对话框交互入口（op 参数驱动）。\n- op=accept: 点击当前活动对话框的确认/确定按钮（dialog_type 标识类型，用于填入录制值）\n- op=reject: 点击当前活动对话框的取消/关闭按钮\n- op=delete_item: 在对话框列表中删除指定项（item 标识，如示例名称、提示词文本）\n- op=clear_all: 清空对话框列表中的所有项\n- op=add_item: 在对话框列表中添加新项（会弹出输入对话框）", category="对话框交互",
            params={"op": "str", "dialog_type": "str", "item": "str"})
    def dialog_op(self, op: str = "accept", dialog_type: str = "", item: str = ""):
        """统一对话框交互入口 — 按 op 分发到不同对话框操作逻辑。

        录制时由 AutoInteractionRecorder 拦截对话框信号自动录制，
        回放时由 ReplayDialogInterceptor 执行对应操作。

        Args:
            op: 操作类型（accept/reject/delete_item/clear_all/add_item）
            dialog_type: 对话框类型标识（如 "set_as_example", "add_category" 等）
            item: 要删除的项标识（如示例名称、提示词文本），用于定位删除按钮
        """
        if not self._engine:
            return True
        interceptor = getattr(self._engine, '_dialog_interceptor', None)
        if not interceptor or not interceptor.is_active:
            return True

        def _accept():
            interceptor.accept_current_dialog(dialog_type)
            return True

        def _reject():
            interceptor.reject_current_dialog(dialog_type)
            return True

        def _delete_item():
            interceptor.click_dialog_button("delete", item)
            return True

        def _clear_all():
            interceptor.click_dialog_button("clear_all")
            return True

        def _add_item():
            interceptor.click_dialog_button("add")
            return True

        handlers = {
            "accept": _accept,
            "reject": _reject,
            "delete_item": _delete_item,
            "clear_all": _clear_all,
            "add_item": _add_item,
        }
        handler = handlers.get(op)
        if handler is None:
            return f"错误: 未知对话框操作 '{op}'，可选: accept/reject/delete_item/clear_all/add_item"
        return handler()

    # ==================== 基础控制 ====================

    @action("control.wait", description="等待指定时间后再继续执行后续步骤。\n- duration: 等待时长（毫秒）\n- 用于等待界面响应或异步操作完成", category="基础控制", params={"duration": "int"})
    def wait(self, duration: int = 1000):
        """等待指定毫秒数

        在 TestEngine 模式下，通过设置 _wait_duration 让 TestEngine 的 QTimer 处理等待（不阻塞UI）。
        在非 TestEngine 模式下（如直接调用），使用 time.sleep。
        """
        if self._engine and hasattr(self._engine, 'main_window'):
            # TestEngine 模式：设置等待时长，由 TestEngine 的 QTimer 处理
            self._engine._wait_duration = duration
        else:
            import time
            time.sleep(duration / 1000.0)
        return True

    @action("control.print", description="在控制台输出一条消息。\n- message: 要输出的文本内容", category="基础控制", params={"message": "str"})
    def print_message(self, message: str = ""):
        """打印消息"""
        print(f"    [Print] {message}")
        return True

    @action("control.print_result", description="将上一步的执行结果输出到控制台。\n- message: 附加的说明文字", category="基础控制", params={"message": "str"})
    def print_result(self, message: str = ""):
        """打印结果"""
        print(f"    [Result] {message}")
        return True

    @action("control.set_variable", description="设置一个可在后续步骤中引用的变量。\n- name: 变量名\n- value: 变量值", category="基础控制", params={"name": "str", "value": "str"})
    def set_variable(self, name: str = "", value: str = ""):
        """设置变量"""
        if self._engine:
            self._engine._variables[name] = value
        print(f"    [Variable] ${name} = {value}")
        return True

    @action("system.close_app", description="安全关闭整个应用程序。\n- 无参数\n- 会触发保存未保存的数据", category="基础控制", params={})
    def close_app(self):
        """关闭应用程序"""
        print(f"    [Close App] 正在关闭程序...")
        from PySide6.QtWidgets import QApplication
        import sys
        QApplication.quit()
        sys.exit(0)
        return True

    # ==================== 循环控制 ====================

    @action("control.loop", description="循环执行一组步骤指定的次数。\n- count: 循环次数\n- steps: 要循环执行的步骤列表", category="循环控制", params={"count": "int", "steps": "list"})
    def loop(self, count: int = 1, steps: list = None):
        """循环执行步骤"""
        engine = self._engine
        steps = steps or []
        if not steps:
            return True

        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()

        def run_iteration(iteration):
            if iteration >= count:
                return True

            for s in steps:
                resolved = engine._resolve_variables(s)
                action = resolved.get('action', '')
                if registry.has_action(action):
                    params = {k: v for k, v in resolved.items()
                              if k not in ('action', 'description', 'wait_after')}
                    try:
                        registry.call(action, params=params)
                    except Exception as e:
                        print(f"      [Error] 循环步骤执行异常: {e}")
                        return False
                else:
                    print(f"      [Error] 循环内未知动作: {action}")
                    return False

            return run_iteration(iteration + 1)

        return run_iteration(0)

    # ==================== 计时器 ====================

    @action("timer.start", description="开始一个指定名称的计时器，用于性能测量。\n- timer_name: 计时器名称", category="计时器", params={"timer_name": "str"})
    def start_timer(self, timer_name: str = "default"):
        """开始计时"""
        if self._engine:
            self._engine._timers[timer_name] = time.time()
        print(f"    开始计时: {timer_name}")
        return True

    @action("timer.stop", description="停止指定名称的计时器并记录耗时。\n- timer_name: 计时器名称", category="计时器", params={"timer_name": "str"})
    def stop_timer(self, timer_name: str = "default"):
        """停止计时"""
        if self._engine and timer_name in self._engine._timers:
            elapsed = (time.time() - self._engine._timers[timer_name]) * 1000
            self._engine._timers[f"{timer_name}_elapsed"] = elapsed
            print(f"    停止计时: {timer_name}, 耗时: {elapsed:.2f}ms")
            return True
        print(f"    [Error] 计时器 '{timer_name}' 不存在")
        return False

    @action("timer.assert_less_than", description="断言指定计时器的耗时小于阈值。\n- timer_name: 计时器名称\n- max_time: 最大允许耗时（毫秒）\n- 超过阈值时测试失败", category="计时器",
            params={"timer_name": "str", "max_ms": "int", "message": "str"})
    def assert_timer_less_than(self, timer_name: str = "default", max_ms: int = 5000, message: str = ""):
        """断言计时时间小于指定值"""
        if self._engine:
            elapsed_key = f"{timer_name}_elapsed"
            if elapsed_key in self._engine._timers:
                elapsed = self._engine._timers[elapsed_key]
                if elapsed < max_ms:
                    print(f"    [Assert OK] 计时 '{timer_name}' 耗时 {elapsed:.2f}ms < {max_ms}ms")
                    return True
                else:
                    print(f"    [Assert FAIL] {message}")
                    print(f"    [ERROR] 计时 '{timer_name}' 耗时 {elapsed:.2f}ms >= {max_ms}ms")
                    return False
        print(f"    [Error] 计时器 '{timer_name}' 未停止或不存在")
        return False

    # ==================== 通用断言（lambda 字符串） ====================

    @action("assert.generic", description="使用 lambda 表达式进行通用条件断言。\n- expression: lambda 表达式字符串\n- message: 断言失败时的提示信息\n- 表达式为 False 时测试失败", category="断言",
            params={"check": "str", "message": "str"})
    def assert_check(self, check: str = "", message: str = ""):
        """执行 lambda 断言

        check 格式: "lambda e: <表达式>"
        e 为 TestEngine 实例，可访问 e.annotation_interface, e.dashboard_interface 等
        """
        if not check:
            return False
        try:
            # 提供受限的eval环境，允许常见内置函数但禁止危险操作
            safe_builtins = {
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
                'normpath': normpath,
                'True': True, 'False': False, 'None': None,
            }
            fn = eval(check, {"__builtins__": safe_builtins}, {})
            result = fn(self._engine)
            if result:
                print(f"    [Assert OK] {message or check}")
            else:
                print(f"    [Assert FAIL] {message or '断言失败'}")
            return bool(result)
        except Exception as e:
            print(f"    [Assert Error] {e}")
            return False

    # ==================== 条件判断 ====================

    @action("control.if", description="根据条件表达式的值执行不同的分支步骤。\n- expression: lambda 表达式字符串\n- then_steps: 条件为 True 时执行的步骤\n- else_steps: 条件为 False 时执行的步骤（可选）", category="流程控制",
            params={"condition": "str", "then": "list", "else": "list"})
    def if_condition(self, **kwargs):
        """条件判断"""
        engine = self._engine
        condition = kwargs.get('condition', '')
        then_steps = kwargs.get('then', [])
        else_steps = kwargs.get('else', [])

        interface = engine.annotation_interface if engine else None
        result = False
        if condition == 'has_categories':
            result = interface is not None and len(interface.categories) > 0
        elif condition == 'has_annotations':
            result = interface is not None and len(interface.draw_area.annotations) > 0
        elif condition == 'has_images':
            result = interface is not None and interface.file_list.count() > 0
        elif condition.startswith('${'):
            var_name = condition[2:-1]
            result = bool(engine._variables.get(var_name, False)) if engine else False
        else:
            # 支持lambda表达式条件判断
            try:
                safe_builtins = {
                    'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                    'abs': abs, 'min': min, 'max': max, 'sum': sum,
                    'range': range, 'enumerate': enumerate, 'zip': zip,
                    'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
                    'True': True, 'False': False, 'None': None,
                }
                fn = eval(condition, {"__builtins__": safe_builtins}, {})
                result = bool(fn(engine))
            except Exception:
                pass

        print(f"    条件判断: {condition} = {result}")
        steps_to_run = then_steps if result else else_steps

        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        for s in steps_to_run:
            resolved = engine._resolve_variables(s)
            act_name = resolved.get('action', '')
            if registry.has_action(act_name):
                params = {k: v for k, v in resolved.items()
                          if k not in ('action', 'description', 'wait_after')}
                registry.call(act_name, params=params)
            else:
                print(f"    [Error] 条件分支内未知动作: {act_name}")
        return True

    # ==================== UI 响应断言 ====================

    @action("assert.ui_responsive", description="断言 UI 在指定时间内保持响应状态。\n- message: 断言说明文字\n- UI 无响应时测试失败", category="断言", params={"message": "str"})
    def assert_ui_responsive(self, message: str = "验证UI保持响应"):
        """断言UI保持响应"""
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            print(f"    [Assert OK] {message}")
            return True
        except Exception as e:
            print(f"    [Assert FAIL] UI无响应: {e}")
            return False

    # ==================== 异步等待 ====================

    @action("control.wait_until", description="轮询等待直到指定条件满足或超时。\n- 循环检查条件表达式\n- 超时后返回 False", category="基础控制",
            params={"check": "str", "timeout": "int", "interval": "int", "message": "str"})
    def wait_until(self, check: str = "", timeout: int = 30000, interval: int = 500, message: str = ""):
        """轮询等待条件满足（在Qt事件循环中）

        Args:
            check: lambda表达式，如 "lambda e: e.annotation_interface.draw_area.annotations"
            timeout: 超时毫秒数，默认30000(30秒)
            interval: 轮询间隔毫秒数，默认500
            message: 描述信息
        """
        if not check:
            return False
        try:
            safe_builtins = {
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
                'normpath': normpath,
                'True': True, 'False': False, 'None': None,
            }
            fn = eval(check, {"__builtins__": safe_builtins}, {})
        except Exception as e:
            print(f"    [wait_until Error] 条件解析失败: {e}")
            return False

        from PySide6.QtWidgets import QApplication
        import time as _time
        start = _time.time()
        while True:
            QApplication.processEvents()
            try:
                if fn(self._engine):
                    elapsed = (_time.time() - start) * 1000
                    print(f"    [wait_until OK] {message or check} ({elapsed:.0f}ms)")
                    return True
            except Exception:
                pass
            elapsed_ms = (_time.time() - start) * 1000
            if elapsed_ms >= timeout:
                print(f"    [wait_until TIMEOUT] {message or check} (>{timeout}ms)")
                return False
            _time.sleep(interval / 1000.0)

    # ==================== AI Agent 测试 ====================

    @action("agent.initialize", description="初始化 AI Agent（供自动化测试调用）。\n- 从当前激活的 AI 提供商配置读取 API Key 初始化内嵌 Agent\n- 已初始化时幂等跳过\n- 未配置 API Key 时返回 False", category="基础控制",
            params={"api_key": "str", "base_url": "str", "model": "str", "exclude_core_tools": "list"})
    def agent_initialize(self, api_key: str = "", base_url: str = "", model: str = "", exclude_core_tools: list = None):
        """初始化 AI Agent，参数缺省时从当前激活提供商读取"""
        if not self._engine or not getattr(self._engine, 'main_window', None):
            return False
        mw = self._engine.main_window
        agent = getattr(mw, '_ai_agent', None)
        if agent is None:
            print("    [Agent] 主窗口未创建 AI Agent")
            return False
        if agent.is_ready():
            print("    [Agent] Agent 已初始化，跳过")
            return True
        if not api_key:
            from core.common.ai_providers import get_active_credentials
            api_key, base_url, model = get_active_credentials()
        if not api_key:
            print("    [Agent] 未配置 API Key，请先在系统配置中设置")
            return False
        agent.initialize(api_key, base_url, model, main_window=mw, exclude_core_tools=exclude_core_tools)
        ok = agent.is_ready()
        print(f"    [Agent] 初始化完成: {model} → {ok}")
        return ok

    @action("agent.chat", description="向 AI Agent 发送一条自然语言指令（异步执行）。\n"
            "- text: 要发送的指令文本\n"
            "- reset: 发送前是否重置会话（默认 true，隔离各场景上下文，避免历史累积干扰）\n"
            "- max_wait: 若 Agent 正忙于上一任务，请求停止并等待空闲的最大毫秒数（默认 600000）\n"
            "- 等待超时或 Agent 未初始化时返回 False\n"
            "- 执行完成后通过 wait_until 轮询 e.main_window._ai_agent.is_busy() 判断",
            category="基础控制", params={"text": "str", "reset": "bool", "max_wait": "int"})
    def agent_chat(self, text: str = "", reset: bool = True, max_wait: int = 600000):
        """发送自然语言指令给 AI Agent（异步）。

        若 Agent 正忙（上一场景任务遗留），先请求停止并泵事件等待其退出，
        再（按需）重置会话，保证每个场景在干净的会话上独立执行。
        """
        if not self._engine or not getattr(self._engine, 'main_window', None):
            return False
        if not text:
            print("    [Agent] 指令文本为空")
            return False
        mw = self._engine.main_window
        agent = getattr(mw, '_ai_agent', None)
        if agent is None or not agent.is_ready():
            print("    [Agent] Agent 未初始化，请先执行 agent.initialize")
            return False
        if agent.is_busy():
            print("    [Agent] Agent 正在执行上一任务，请求停止并等待空闲...")
            if not agent.stop_and_wait(max_wait):
                print(f"    [Agent] 等待 Agent 空闲超时（>{max_wait}ms）")
                return False
        if reset:
            print("    [Agent] 重置会话（场景隔离）")
            agent.reset()
        agent._last_response = ""
        print(f"    [Agent] 发送指令: {text}")
        agent.chat(text)
        return True

    @action("agent.result", description="打印 AI Agent 最近一次完整回复，便于人工核对 LLM 输出。\n- message: 附加说明文字", category="基础控制",
            params={"message": "str"})
    def agent_result(self, message: str = ""):
        """打印 Agent 最近一次回复"""
        if not self._engine or not getattr(self._engine, 'main_window', None):
            return False
        mw = self._engine.main_window
        agent = getattr(mw, '_ai_agent', None)
        if agent is None:
            print("    [Agent] 主窗口未创建 AI Agent")
            return False
        resp = getattr(agent, 'last_response', '') or ''
        prefix = f"{message}: " if message else ""
        print(f"    [Agent 回复] {prefix}{resp}")
        return True

    @action("agent.review", description="审查最近一次 Agent 会话记录，验证执行质量。\n"
            "- 从会话记录中解析最近一次指令后的工具调用序列（名称/参数/结果）\n"
            "- 检查项：①无工具报错 ②按计划执行（首条带工具调用的助手消息前须输出「## 执行计划」）③任务完成（有最终文本回复）④期望工具全部被调用 ⑤未调用禁止工具\n"
            "- expected_tools: 期望被调用的工具名列表（两段式名称，如 manual.navigate/manual.get_state，兼容短名）\n"
            "- forbidden_tools: 禁止调用的工具名列表（出现即判定为多余操作）\n"
            "- require_plan: 是否强制「按计划执行」检查，默认 True；单步简单任务可设为 False 豁免（报告仍如实显示计划状态）\n"
            "- match_instruction: 指定按包含该文本的用户指令定位审查段；留空则使用最近一条用户指令（推荐填写，避免 Agent 追问补答后只审查到后半段）\n"
            "- 返回 True=全部通过，False=存在失败项；报告同时打印到控制台并落盘", category="基础控制",
            params={"expected_tools": "list", "forbidden_tools": "list", "require_plan": "bool", "save_report": "str", "match_instruction": "str"})
    def agent_review(self, expected_tools=None, forbidden_tools=None, require_plan=True, save_report="", match_instruction=""):
        """审查 Agent 最近一次会话记录"""
        if not self._engine or not getattr(self._engine, 'main_window', None):
            return False
        mw = self._engine.main_window
        agent = getattr(mw, '_ai_agent', None)
        if agent is None or not agent.is_ready():
            print("    [Agent] Agent 未初始化，无法审查")
            return False
        core_agent = getattr(agent, '_agent', None)
        messages = getattr(core_agent, 'messages', None) if core_agent else None
        if not messages:
            print("    [Agent] 无会话记录可审查")
            return False

        # 定位用户指令：match_instruction 存在时按包含文本定位（可跨补答审查完整段），否则用最近一条 user
        last_user = -1
        if match_instruction:
            for i, m in enumerate(messages):
                if m.get("role") == "user" and match_instruction in (m.get("content") or ""):
                    last_user = i
        else:
            for i, m in enumerate(messages):
                if m.get("role") == "user":
                    last_user = i
        if last_user < 0:
            print("    [Agent] 未找到用户指令消息")
            return False
        instruction = messages[last_user].get("content", "")
        segment = messages[last_user + 1:]

        tool_calls, assistant_texts, plan_found = self._parse_session(segment)
        called_names = [tc["name"] for tc in tool_calls]

        # 工具名匹配支持两种形式：两段式全名（如 manual.get_state）与短名（如 get_state）。
        # Agent 可能调用 action 工具（manual.get_state）或等价 core 工具（get_state）。
        called_variants = set()
        for n in called_names:
            called_variants.add(n)
            if "." in n:
                called_variants.add(n.split(".", 1)[1])

        def _hit(tool):
            if tool in called_names:
                return True
            if "." in tool and tool.split(".", 1)[1] in called_variants:
                return True
            return False

        # ① 无工具报错
        errors = [tc for tc in tool_calls if tc.get("ok") is False]
        no_error_ok = not errors

        # ② 按计划执行
        plan_ok = True
        if plan_found is False:
            plan_ok = False

        # ③ 任务完成：最后一条 assistant 消息为纯文本（无工具调用）
        final_assistant = next((m.get("content") for m in reversed(segment)
                                if m.get("role") == "assistant" and not m.get("tool_calls")
                                and (m.get("content") or "").strip()), "")
        task_ok = bool(final_assistant.strip())
        # LLM 可能不输出最终总结文本（工具调用全部成功也算任务完成）
        if not task_ok and tool_calls and all(tc.get("ok") is True for tc in tool_calls):
            task_ok = True

        # ④ 期望工具全部被调用（支持两段式名/短名）
        expected = list(expected_tools or [])
        missing = [t for t in expected if not _hit(t)]
        expected_ok = not missing

        # ⑤ 未调用禁止工具（支持两段式名/短名）
        forbidden = list(forbidden_tools or [])
        hit = [t for t in forbidden if _hit(t)]
        forbidden_ok = not hit

        all_ok = no_error_ok and (plan_ok if require_plan else True) and task_ok and expected_ok and forbidden_ok

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instruction": instruction,
            "require_plan": bool(require_plan),
            "passed": all_ok,
            "checks": {
                "no_error": no_error_ok,
                "plan_followed": plan_ok,
                "task_completed": task_ok,
                "expected_tools_ok": expected_ok,
                "forbidden_tools_ok": forbidden_ok,
            },
            "details": {
                "errors": [{"name": e["name"], "result": e["result"][:200]} for e in errors],
                "missing_tools": missing,
                "forbidden_hit": hit,
                "final_reply": final_assistant[:500],
            },
            "tool_calls": [{"name": tc["name"], "arguments": tc["arguments"], "ok": tc["ok"]} for tc in tool_calls],
        }
        self._print_review_report(report, tool_calls)
        self._save_review_report(report, save_report)
        return all_ok

    @action("agent.test_action", description="通用工具调用自测：模拟对话测试大模型能否正确找到并调用指定 action/工具。\n"
            "- 根据目标 action 的描述/参数自动生成自然语言指令（也可用 instruction 传入自定义指令），模拟真实的对话场景\n"
            "- 发送指令后自动等待 Agent 空闲，并从会话记录中判定「目标工具是否被调用 + 调用是否无报错 + 任务是否完成」\n"
            "- 适用于新插件开发：验证新注册的 scope=agent action 能被 Agent 正确发现并调用\n"
            "- name: 目标 action 完整注册名（如 annotation.manual.get_state）；expect_tool 不填时取该 action 的显示名\n"
            "- hint: true 时指令中点名工具名（只测调用链路）；false（默认）纯描述测「能否被找到」\n"
            "- require_plan: 是否强制「按计划执行」检查，默认 False（单步/查询类工具常直接调用，不做流程编排审查，报告仍如实显示该状态）\n"
            "- self_stop: true 时按「agent 自主完整执行到自停」判定（不强制命中某个目标工具），适合多步真实对话指令；false（默认）仍强制期望工具命中（单步精确测试）\n"
            "- 返回 True=测试通过，False=存在失败项；报告打印到控制台并落盘", category="基础控制",
            params={"name": "str", "instruction": "str", "hint": "bool", "expect_tool": "str", "require_plan": "bool", "self_stop": "bool", "save_report": "str", "max_wait": "int", "strategy": "str"})
    def agent_test_action(self, name: str = "", instruction: str = "", hint: bool = False,
                          expect_tool: str = "", require_plan: bool = False, self_stop: bool = False,
                          save_report: str = "", max_wait: int = 600000,
                          strategy: str = ""):
        """通用工具调用自测：让 LLM 找并调用一个指定 action，审查会话验证调用正确性。"""
        if not self._engine or not getattr(self._engine, 'main_window', None):
            print("    [Agent] 无主窗口，无法测试")
            return False
        if not name:
            print("    [Agent] 未指定目标 action 名称")
            return False

        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()

        # 解析目标 action：支持完整名与短名（经别名/后缀解析）
        resolved = registry._resolve_name(name)
        meta = registry.get_meta(resolved)
        if meta is None:
            print(f"    [Agent] action '{name}' 未注册")
            return False
        if getattr(meta, "scope", None) != "agent":
            print(f"    [Agent] action '{resolved}' 的 scope={meta.scope}，非 agent 可调用，Skip")
            return False

        # 期望被调用的工具：默认取该 action 的显示名（去掉前缀后的一段，如 get_state）
        display_name = resolved.split(".", 1)[1] if "." in resolved else resolved
        expected = expect_tool or display_name

        # 生成指令（默认从 meta 语义生成，不泄露工具名；hint=True 时点名工具）
        prompt = instruction
        if not prompt:
            prompt = self._build_action_prompt(resolved, meta, display_name, hint)

        # 生成并执行会话，同时做判定
        mw = self._engine.main_window
        agent = getattr(mw, '_ai_agent', None)
        if agent is None:
            print("    [Agent] 主窗口未创建 AI Agent")
            return False

        # ① 初始化（复用 agent.initialize，幂等；未配置 API Key 时返回 False）
        if not self.agent_initialize():
            print("    [Agent] Agent 初始化失败（请确认已配置 API Key）")
            return False

        # ② 发送指令（等待前一任务退出后重置会话）
        if agent.is_busy():
            if not agent.stop_and_wait(max_wait):
                print(f"    [Agent] 等待 Agent 空闲超时（>{max_wait}ms）")
                return False
        # ---- 基准测试：策略注入与指标采集 ----
        # 工具调度策略（基准测试用）：不指定则维持当前策略不变
        prev_strategy = agent.tool_strategy
        if strategy:
            print(f"    [Agent] 切换工具调度策略: {strategy}")
            agent.set_tool_strategy(strategy)
        from core.agent.token_budget import estimate_tokens, estimate_tools_tokens
        _llm = getattr(agent, "_message_compressor_llm", None)
        _tok_before_prompt = getattr(_llm, "total_prompt_tokens", 0) or 0
        _tok_before_comp = getattr(_llm, "total_completion_tokens", 0) or 0
        core_agent = getattr(agent, '_agent', None)

        metrics = {}
        try:
            agent.reset()
            agent._last_response = ""
            print(f"    [Agent] 目标工具: {expected}  →  指令: {prompt}")
            # 打开 AI 侧栏并走真实 UI 发送通道，渲染用户气泡 + 流式助手回复 + 工具调用块
            try:
                if hasattr(mw, "_ai_sidebar") and not mw._ai_sidebar.isVisible():
                    mw.toggle_ai_sidebar()
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
            except Exception:
                pass
            if hasattr(mw, "_ai_sidebar") and hasattr(mw._ai_sidebar, "_on_input_send"):
                mw._ai_sidebar._on_input_send(prompt)
            else:
                agent.chat(prompt)

            # chat() 内 _update_tools 已同步执行，此刻 agent.tools 即为暴露给 LLM 的工具集。
            # 采集任务开始时的上下文大小（系统提示词 + 工具 schema token）与暴露工具名。
            tool_list = list(core_agent.tools) if core_agent else []
            exposed_names = sorted(t.name for t in tool_list)
            metrics["strategy"] = strategy or prev_strategy
            metrics["exposed_tool_count"] = len(exposed_names)
            metrics["exposed_tools"] = exposed_names
            try:
                metrics["context_system_tokens"] = estimate_tokens(core_agent._system)
            except Exception:
                metrics["context_system_tokens"] = 0
            try:
                metrics["context_tools_tokens"] = estimate_tools_tokens(tool_list)
            except Exception:
                metrics["context_tools_tokens"] = 0
            metrics["context_total_tokens"] = (
                metrics.get("context_system_tokens", 0) + metrics.get("context_tools_tokens", 0)
            )
            # lru 窗口统计（窗口大小 / 实际窗口 / 保底数 / 裁剪数），便于对比窗口前后差异
            try:
                tm = getattr(agent, '_tool_manager', None)
                if tm is not None and hasattr(tm, 'get_window_stats'):
                    metrics["lru_window"] = tm.get_lru_window_size()
                    stats = tm.get_window_stats()
                    if stats:
                        metrics["lru_window_stats"] = {
                            k: stats.get(k) for k in
                            ("window_size", "window_target", "max_window_size", "mandatory",
                             "protected", "protect_min_score", "relevant_candidates",
                             "activated_mandatory", "relevance_budget_trimmed",
                             "candidates", "exposed_actions", "dropped", "auto_expanded",
                             "keyword_domains", "active_domains",
                             # 保底名单：基准报告需要能证明「保底集合里有没有任务期望工具」
                             # （analyze 时不必再扒日志）；scorer 提供打分器统计
                             "scorer", "protected_names", "dropped_names",
                             "relevance_terms")
                        }
            except Exception as e:
                print(f"    [Agent] 窗口统计采集失败（忽略）: {e}")

            # ③ 泵事件等待 Agent 空闲（LLM 调用 + 工具执行期间主线程仍响应）
            import time as _time
            start = _time.time()
            from PySide6.QtWidgets import QApplication
            while agent.is_busy():
                QApplication.processEvents()
                if _time.time() - start > max_wait / 1000.0:
                    print(f"    [Agent] 等待测试完成超时（>{max_wait}ms）")
                    return False
                _time.sleep(0.2)

        finally:
            # 任务结束恢复原策略，避免影响后续非基准运行
            if strategy:
                try:
                    agent.set_tool_strategy(prev_strategy)
                except Exception as e:
                    print(f"    [Agent] 恢复策略失败（忽略）: {e}")

        # ---- 消耗 token ----
        try:
            metrics["tokens_prompt"] = (getattr(_llm, "total_prompt_tokens", 0) or 0) - _tok_before_prompt
            metrics["tokens_completion"] = (getattr(_llm, "total_completion_tokens", 0) or 0) - _tok_before_comp
            metrics["tokens_consumed"] = metrics["tokens_prompt"] + metrics["tokens_completion"]
        except Exception:
            metrics["tokens_prompt"] = metrics["tokens_completion"] = metrics["tokens_consumed"] = 0

        # ④ 从会话记录判定
        messages = getattr(core_agent, 'messages', None) if core_agent else None
        if not messages:
            print("    [Agent] 无会话记录可判定")
            return False

        tool_calls, assistant_texts, plan_found = self._parse_session(messages)
        called_names = [tc["name"] for tc in tool_calls]
        # 兼容：工具调用可能是 display_name 或完整名，做短名/包含归一
        called_variants = set()
        for n in called_names:
            called_variants.add(n)
            if "." in n:
                called_variants.add(n.split(".", 1)[1])
        expected_hit = expected in called_names or expected in called_variants

        errors = [tc for tc in tool_calls if tc.get("ok") is False]
        no_error_ok = not errors
        task_ok = any((t.strip()) for t in assistant_texts) or (tool_calls and all(tc.get("ok") is True for tc in tool_calls))
        plan_ok = plan_found is not False

        # 自停完整性判定（多步真实对话指令）：不强制命中某个目标工具，
        # 只看 agent 是否自主、正确地完整执行到自然停止 —— 无报错 + 有最终回复 + 未跑飞。
        # 业务错误不计入"访问/调用错"（命中率单独统计），故此处 no_error_ok 指调用层无崩溃。
        self_stop_runner_ok = bool(tool_calls) and no_error_ok and task_ok

        # 防跑飞：工具调用过多，或反复调用同一工具（含失败重试）视为异常冗余，任务即使"完成"也不予通过
        _MAX_CALLS = 50
        _MAX_SAME = 4
        runaway_ok = True
        if len(tool_calls) > _MAX_CALLS:
            runaway_ok = False
        else:
            seq = 1
            for i in range(1, len(tool_calls)):
                if tool_calls[i]["name"] == tool_calls[i-1]["name"]:
                    seq += 1
                    if seq > _MAX_SAME:
                        runaway_ok = False
                        break
                else:
                    seq = 1

        if self_stop:
            # agent 自停完整性：多轮完整执行 + 无调用错 + 有最终回复 + 未跑飞。
            all_ok = self_stop_runner_ok and runaway_ok and (plan_ok if require_plan else True)
        else:
            all_ok = expected_hit and no_error_ok and (plan_ok if require_plan else True) and task_ok and runaway_ok

        # ---- 基准测试指标：工具命中率 / 轮次 ----
        # 命中定义（用户口径）=「正确访问」+「正确调用」：
        #   - 访问错(unknown tool/未注册/not found) 或 调用错(bad arguments/Error executing) → 未命中
        #   - 业务逻辑返回错误（如"已达最后一张图"）不算调用错，仍计为命中
        try:
            from core.agent.tool_strategy import classify_tool_hit
            hit_calls = []
            for tc in tool_calls:
                hit, reason = classify_tool_hit(tc.get("result") or "", tc.get("ok") is not False)
                hit_calls.append({
                    "name": tc["name"],
                    "hit": hit,
                    "reason": reason,
                    "ok": tc.get("ok"),
                    "result": (tc.get("result") or "")[:200],
                })
            n_hit = sum(1 for c in hit_calls if c["hit"])
            n_call = len(hit_calls)
            metrics["tool_calls_total"] = n_call
            metrics["tool_calls_hit"] = n_hit
            metrics["tool_calls_miss"] = n_call - n_hit
            metrics["hit_rate"] = round(n_hit / n_call, 4) if n_call else 1.0
            metrics["missed_calls"] = [c for c in hit_calls if not c["hit"]]
            # 轮次 = 本任务内带工具调用的 assistant 消息条数（LLM 工具调用循环次数）
            metrics["rounds"] = sum(
                1 for m in messages
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
        except Exception as e:
            print(f"    [Agent] 命中率/轮次统计失败: {e}")
            metrics.setdefault("hit_rate", 1.0)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": "action_test",
            "target_action": resolved,
            "expected_tool": expected,
            "hint": bool(hint),
            "instruction": prompt,
            "passed": all_ok,
            "mode": "self_stop" if self_stop else "expected_tool",
            "checks": {
                "expected_tool_called": expected_hit,
                "no_error": no_error_ok,
                "plan_followed": plan_ok,
                "task_completed": task_ok,
                "no_runaway": runaway_ok,
            },
            "details": {
                "not_called": False if expected_hit else expected,
                "errors": [{"name": e["name"], "result": (e.get("result") or "")[:200]} for e in errors],
                "final_reply": (assistant_texts[-1] if assistant_texts else "")[:300],
            },
            "tool_calls": [{"name": tc["name"], "arguments": tc["arguments"], "ok": tc["ok"]} for tc in tool_calls],
            "metrics": metrics,
        }
        self._print_action_test_report(report, tool_calls)
        self._save_action_test_report(report, save_report)
        return all_ok

    @action("agent.verify_saved_session", read_only=True, description="验证最近保存的会话文件不包含 base64 图片与全路径映射。\n- save: 是否先调用 save_session 保存当前会话（默认 true）\n- 读取 ~/.corecoder/sessions 目录下最新的会话 JSON 文件，检查：\n  - 不包含 data:image（无 base64）\n  - 不包含 image_splits（无全项目路径划分映射）\n  - 不包含硬编码 hard_samples 全路径映射\n- 任一违规返回 False，同时打印检查详情", category="基础控制", params={"save": "bool"})
    def agent_verify_saved_session(self, save: bool = True) -> bool:
        """校验最近保存的会话文件是否干净（无 base64 / 无全路径映射）。"""
        if not self._engine or not getattr(self._engine, 'main_window', None):
            print("    [Verify] 无主窗口")
            return False
        agent = getattr(self._engine.main_window, '_ai_agent', None)
        if agent is None:
            print("    [Verify] 无 AI Agent")
            return False
        if save and agent.is_ready():
            try:
                agent.save_session()
            except Exception as e:
                print(f"    [Verify] save_session 异常: {e}")
                return False
        import os
        sess_dir = os.path.expanduser("~/.corecoder/sessions")
        if not os.path.isdir(sess_dir):
            print(f"    [Verify] 会话目录不存在: {sess_dir}")
            return False
        files = [f for f in os.listdir(sess_dir) if f.endswith(".json")]
        if not files:
            print("    [Verify] 无会话文件")
            return False
        files.sort(key=lambda f: os.path.getmtime(os.path.join(sess_dir, f)), reverse=True)
        path = os.path.join(sess_dir, files[0])
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except Exception as e:
            print(f"    [Verify] 读取会话文件失败: {e}")
            return False
        has_image = "data:image" in raw
        has_splits = "image_splits" in raw
        has_hard = "hard_samples" in raw
        ok = not has_image and not has_splits and not has_hard
        print(f"    [Verify] 会话文件: {path}")
        print(f"    [Verify] 含 base64(data:image): {has_image} | 含 image_splits: {has_splits} | 含 hard_samples: {has_hard} → {'通过' if ok else '失败'}")
        return ok

    @staticmethod
    def _build_action_prompt(resolved, meta, display_name, hint):
        """根据 action 元信息生成自然语言指令。

        默认从 description 提取任务语义，不点名工具名（测「能否被找到」）；
        hint=True 时点名结果含义（display_name），聚焦测「调用链路正确性」。
        """
        desc = (meta.description or "").replace("\n", "，").strip()
        if not desc:
            desc = f"调用 {display_name} 功能"
        if hint:
            return (f"请使用工具 {display_name} 完成以下任务：{desc}。"
                    f"如果工具需要参数，请根据实际情况合理提供。完成操作后简要说明结果。")
        # 纯描述：把参数枚举信息转成人话，让 LLM 自行 search 并选择工具
        param_hint = ""
        if meta.params:
            names = ", ".join(list(meta.params.keys())[:6])
            param_hint = f"（工具所需参数可能包含：{names}）"
        category = meta.category or "通用"
        return (f"请执行「{category}」类的以下任务：{desc}{param_hint}。"
                f"请先使用搜索/激活工具找到能完成该操作的工具再调用它，不要使用不相关的工具。"
                f"执行完成后简要说明结果。")

    @staticmethod
    def _print_action_test_report(report, tool_calls):
        """打印工具自测报告到控制台"""
        print("========== [工具调用自测] ==========")
        print(f"  目标工具: {report['expected_tool']}")
        print(f"  指令: {report['instruction'][:120]}")
        print(f"  工具调用序列 ({len(tool_calls)} 次):")
        for i, tc in enumerate(tool_calls, 1):
            md = "✅" if tc["ok"] is not False else "❌"
            args = tc["arguments"]
            if len(args) > 120:
                args = args[:120] + "..."
            print(f"    [{i}] {md} {tc['name']}({args})")
        c = report["checks"]
        print("  检查结果:")
        print(f"    {'✅' if c['expected_tool_called'] else '❌'} 目标工具被调用")
        print(f"    {'✅' if c['no_error'] else '❌'} 调用无报错")
        print(f"    {'✅' if c['plan_followed'] else '❌'} 按计划执行")
        print(f"    {'✅' if c['task_completed'] else '❌'} 任务完成（有最终回复）")
        d = report["details"]
        if d["not_called"]:
            print(f"    [未调用] {d['not_called']}")
        if d["errors"]:
            print(f"    [工具错误] {d['errors']}")
        print(f"  结论: {'✅ 通过' if report['passed'] else '❌ 失败'}")
        print("======================================")

    @staticmethod
    def _save_action_test_report(report, save_report=""):
        """保存工具自测报告到 outputs/ 目录"""
        try:
            from core.common.config import Config
            path = save_report or os.path.join(Config.OUTPUTS_DIR,
                                               f"action_test_{time.strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"    [Agent] 自测报告已保存: {path}")
        except Exception as e:
            print(f"    [Agent] 自测报告保存失败: {e}")

    # ==================== 意图分析（操作/状态实时分析） ====================

    @action("intent.get_recent", read_only=True, description="查询最近用户操作历史记录（供 Agent 感知用户行为）。\n- count: 返回条数，默认 20\n- 每条记录含 action/params/before_state/after_state/时间戳", category="意图分析",
            params={"count": "int"}, scope="agent")
    def intent_get_recent(self, count: int = 20):
        """返回最近用户操作历史"""
        try:
            from core.agent.operation_logger import OperationLogger
            history = OperationLogger.instance().get_history(count=count)
            import json
            return json.dumps({"status": "ok", "total": len(history), "records": history}, ensure_ascii=False)
        except Exception as e:
            return f"错误: {type(e).__name__}: {e}"

    @action("intent.analyze", read_only=True, description="通过 LLM 分析用户操作历史与程序状态，判断当前用户意图并给出可执行建议。\n- 返回 intent(意图标签)/intent_description(意图描述)/suggestions(建议列表，含可执行 action 名)\n- 注意：LLM 分析是异步的，首次调用可能返回上一次缓存结果或触发新的异步分析", category="意图分析",
            scope="agent")
    def intent_analyze(self):
        """LLM 分析当前用户意图与建议（异步触发，返回缓存结果）"""
        try:
            from core.agent.intent_agent import IntentAgent
            agent = IntentAgent.instance()
            if agent.has_llm():
                agent.analyze_async()
            result = agent.analyze()
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"错误: {type(e).__name__}: {e}"

    @action("intent.set_enabled", description="开启或关闭意图分析功能。\n- enabled: true 开启 / false 关闭\n- 关闭后不再实时分析用户操作、不生成执行建议，悬浮状态框显示已关闭", category="意图分析",
            params={"enabled": "bool"}, scope="agent")
    def intent_set_enabled(self, enabled: bool = True):
        """开启/关闭意图分析"""
        try:
            from core.agent.intent_agent import IntentAgent
            IntentAgent.instance().set_enabled(bool(enabled))
            return "意图分析已开启" if enabled else "意图分析已关闭"
        except Exception as e:
            return f"错误: {type(e).__name__}: {e}"

    @staticmethod
    def _parse_session(segment):
        """解析会话段：提取工具调用序列、纯文本助手消息、计划标记"""
        tool_calls = []
        assistant_texts = []
        pending = {}
        plan_text = ""
        plan_found = None
        for m in segment:
            role = m.get("role")
            if role == "assistant":
                content = m.get("content") or ""
                tcs = m.get("tool_calls")
                if tcs:
                    for tc in tcs:
                        fn = tc.get("function", {})
                        idx = len(tool_calls)
                        tool_calls.append({
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "") or "",
                            "result": "",
                            "ok": None,
                        })
                        pending[tc.get("id")] = idx
                    if plan_found is None:
                        plan_found = ("## 执行计划" in plan_text + content) or ("执行计划" in plan_text + content)
                else:
                    assistant_texts.append(content)
                    if plan_found is None:
                        plan_text += "\n" + content
            elif role == "tool":
                tid = m.get("tool_call_id")
                if tid in pending:
                    idx = pending[tid]
                    result = m.get("content") or ""
                    tool_calls[idx]["result"] = result
                    # 失败判定：既支持 ❌/Error 前缀，也支持 JSON 结构错误（如 {"status": "error"}）
                    stripped = result.lstrip()
                    json_error = '"status": "error"' in stripped[:600] or '"status":"error"' in stripped[:600]
                    tool_calls[idx]["ok"] = not (stripped.startswith("❌") or stripped.startswith("Error") or json_error)
        return tool_calls, assistant_texts, plan_found

    @staticmethod
    def _print_review_report(report, tool_calls):
        """打印审查报告到控制台"""
        print("========== [Agent 会话审查] ==========")
        print(f"  指令: {report['instruction'][:120]}")
        print(f"  工具调用序列 ({len(tool_calls)} 次):")
        for i, tc in enumerate(tool_calls, 1):
            mark = "✅" if tc["ok"] is not False else "❌"
            args = tc["arguments"]
            if len(args) > 120:
                args = args[:120] + "..."
            print(f"    [{i}] {mark} {tc['name']}({args})")
        c = report["checks"]
        print("  检查结果:")
        print(f"    {'✅' if c['no_error'] else '❌'} 无工具报错")
        print(f"    {'✅' if c['plan_followed'] else '❌'} 按计划执行（## 执行计划）")
        print(f"    {'✅' if c['task_completed'] else '❌'} 任务完成（有最终文本回复）")
        print(f"    {'✅' if c['expected_tools_ok'] else '❌'} 期望工具全部调用")
        print(f"    {'✅' if c['forbidden_tools_ok'] else '❌'} 无禁止工具被调用")
        d = report["details"]
        if d["missing_tools"]:
            print(f"    [缺失工具] {d['missing_tools']}")
        if d["forbidden_hit"]:
            print(f"    [多余操作] {d['forbidden_hit']}")
        if d["errors"]:
            print(f"    [工具错误] {d['errors']}")
        if d["final_reply"]:
            print(f"  最终回复: {d['final_reply'][:300]}")
        print(f"  审查结论: {'✅ 通过' if report['passed'] else '❌ 失败'}")
        print("======================================")

    @staticmethod
    def _save_review_report(report, save_report=""):
        """保存审查报告到 outputs/ 目录"""
        try:
            from core.common.config import Config
            path = save_report or os.path.join(Config.OUTPUTS_DIR,
                                               f"agent_review_{time.strftime('%Y%m%d_%H%M%S')}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"    [Agent] 审查报告已保存: {path}")
        except Exception as e:
            print(f"    [Agent] 审查报告保存失败: {e}")
