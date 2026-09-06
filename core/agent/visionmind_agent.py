"""
VisionMind Agent 封装 — 将 CoreCoder 集成到 VisionMind

提供 VisionMindAgent 类，封装 CoreCoder 的 Agent，
添加 VisionMind 专属工具和系统提示词。

支持动态 tool 管理：根据当前界面自动暴露对应的 action 工具。
"""

from PySide6.QtCore import QObject, Signal, QThread


class LLMWorker(QThread):
    """在后台线程执行 Agent 调用，避免阻塞 UI"""

    finished = Signal(str)  # Agent 回复
    error = Signal(str)  # 错误信息
    tool_called = Signal(str, dict)  # tool_name, arguments
    tool_result = Signal(str, str)  # tool_name, result_string
    token_received = Signal(str)  # 流式 token (逐 chunk)
    stopped = Signal()  # 用户主动停止

    def __init__(self, agent, user_input: str | list, tool_manager=None):
        super().__init__()
        self.agent = agent
        self.user_input = user_input
        self.tool_manager = tool_manager
        self._stop_requested = False  # 停止标志

    def request_stop(self):
        """请求停止：设置标志位，让正在运行的循环在下一个工具调用前退出"""
        self._stop_requested = True

    def run(self):
        from core.agent.tools.action_tool import ActionTool
        wrapped_tools = []
        for tool in self.agent.tools:
            if isinstance(tool, ActionTool):
                original_execute = tool.execute
                tool_name = tool.name

                def make_wrapper(name, orig):
                    def wrapped(**kwargs):
                        # 工具执行前检查停止标志
                        if self._stop_requested:
                            raise KeyboardInterrupt("用户已停止执行")
                        result = orig(**kwargs)
                        self.tool_result.emit(name, result)
                        return result
                    return wrapped

                tool.execute = make_wrapper(tool_name, original_execute)
                wrapped_tools.append((tool, original_execute))
            else:
                # 非 ActionTool（如 GetStateTool）也有 result，需要拦截
                original_execute = getattr(tool, 'execute', None)
                if original_execute:
                    tool_name = tool.name

                    def make_wrapper(name, orig):
                        def wrapped(*args, **kwargs):
                            if self._stop_requested:
                                raise KeyboardInterrupt("用户已停止执行")
                            result = orig(*args, **kwargs)
                            self.tool_result.emit(name, result)
                            return result
                        return wrapped

                    setattr(tool, 'execute', make_wrapper(tool_name, original_execute))
                    wrapped_tools.append((tool, original_execute))

        try:
            def on_token(token):
                self.token_received.emit(token)

            def on_tool(name, args):
                print(f"[LLMWorker] 工具调用: {name}({args})")
                if self.tool_manager:
                    self.tool_manager.touch_tool(name)
                self.tool_called.emit(name, args)

            print(f"[LLMWorker] 发送消息给 LLM: {self.user_input[:100]}...")

            # 停止检查：若用户在 LLM 调用前已请求停止，直接退出
            if self._stop_requested:
                print("[LLMWorker] 已在 LLM 调用前停止")
                self.stopped.emit()
                return

            result = self.agent.chat(self.user_input, on_token=on_token, on_tool=on_tool)
            print(f"[LLMWorker] LLM 回复: {result[:200]}...")

            # 如果在 agent.chat 执行期间被停止，发出 stopped 信号而非 finished
            if self._stop_requested:
                self.stopped.emit()
            else:
                self.finished.emit(result)
        except KeyboardInterrupt:
            # 用户主动停止 — KeyboardInterrupt 是 BaseException，
            # 绕过 CoreCoder _exec_tool 的 except Exception，由 agent.chat()
            # 的 except KeyboardInterrupt 回填 tool 回复后 re-raise 到此处
            print("[LLMWorker] 执行被用户停止")
            self.stopped.emit()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[LLMWorker] 错误: {error_msg}")
            self.error.emit(error_msg)
        finally:
            # 恢复原始 execute 方法
            for tool, orig in wrapped_tools:
                tool.execute = orig


class VisionMindAgent(QObject):
    """VisionMind 内嵌 Agent"""

    response_ready = Signal(str)  # Agent 回复
    tool_called = Signal(str, dict)  # 工具调用通知
    tool_result = Signal(str, str)  # 工具执行结果 (tool_name, result)
    error_occurred = Signal(str)  # 错误
    token_received = Signal(str)  # 流式 token
    stopped = Signal()  # 用户主动停止执行
    bash_approval_requested = Signal(str, object)  # bash 解锁审批 (reason, respond 回调)
    ask_user_requested = Signal(str, object)  # 向用户提问 (questionsJson, respond 回调)
    context_stats = Signal(str)  # 上下文用量摘要 (get_summary JSON + max)
    compression_report = Signal(str)  # 上下文压缩完成报告（手动 /compact 或阈值自动触发）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._agent = None
        self._worker = None
        self._tool_manager = None
        self._main_window = None
        self._core_tools = []
        # Task 9: 动态提示词注入系统相关状态
        self._token_budget = None  # TokenBudgetManager 实例
        self._message_compressor_llm = None  # 用于消息压缩的 LLM 引用
        self._conversation_turn = 0  # 对话轮次计数器
        self._recent_tool_categories = []  # 最近使用的工具分类列表
        self._compression_history = []  # 压缩历史记录
        self._last_response = ""  # 最近一次 Agent 完整回复（供自动化测试断言）
        self._plan_mode = False  # 计划模式:仅开放只读工具,产出计划待用户审批

    def initialize(self, api_key: str, base_url: str, model: str, main_window=None):
        """
        初始化 Agent（需要 LLM 配置）

        Args:
            api_key: LLM API Key
            base_url: LLM API Base URL
            model: 模型名称
            main_window: MainWindow 实例（用于动态 tool 管理）
        """
        from core.corecoder import Agent, LLM
        from core.corecoder.tools import ALL_TOOLS
        from core.agent.dynamic_tool_manager import DynamicToolManager
        from core.agent.tools.get_state import GetStateTool
        from core.agent.token_budget import TokenBudgetManager

        llm = LLM(model=model, api_key=api_key, base_url=base_url)

        # 机制 B: 包装 llm.chat — 发送前扫描工具结果中的 [IMG_VIEW] 标记,
        # 把图片注入为 user 消息(真实持久于 agent.messages 历史)。
        from core.agent.multimodal import build_image_user_message
        agent_holder = {"agent": None}

        def multimodal_chat(messages, tools=None, on_token=None):
            agent = agent_holder["agent"]
            if agent is not None:
                pending = []
                for m in agent.messages:
                    if m.get("role") == "tool":
                        content = m.get("content", "")
                        if isinstance(content, str) and "[IMG_VIEW:" in content:
                            img_msg = build_image_user_message(content)
                            if img_msg:
                                # 供 LLM 阅读的 tool 结果保留 [已加载图片: path],
                                # 收敛时可据此匹配并回填结构化描述(不泄露内部标记)
                                m["content"] = f"[已加载图片: {content[len('[IMG_VIEW:'):].rstrip(']').strip()}]"
                                pending.append(img_msg)
                if pending:
                    agent.messages.extend(pending)
                    messages = agent._full_messages()
            result = original_chat(messages, tools=tools, on_token=on_token)
            return result

        original_chat = llm.chat
        llm.chat = multimodal_chat

        self._main_window = main_window

        # Task 9: 保存 LLM 引用供消息压缩使用
        self._message_compressor_llm = llm

        # Task 9: 初始化 TokenBudgetManager。
        # max_context 与 corecoder Agent 默认 128_000 对齐,避免按 32_000 过早
        # 触发消息压缩(压缩会额外调用 LLM 摘要引发卡顿,且曾注入中间 system)。
        self._token_budget = TokenBudgetManager(max_context_tokens=128_000)
        # 注: 统计推送延后到 __init__ 末尾(系统提示词/工具就绪后)，避免初始全 0

        # CoreCoder 基础工具 + 提示词管理工具 + 工具发现/激活
        from core.agent.tools.prompt_tools import (
            ListPromptSectionsTool,
            InjectPromptTool,
            GetActivePromptsTool,
            ToolSearchAgent,
            ActivateToolsTool,
        )
        # 文档生成工具 + python 脚本分析沙箱（深度图像分析）
        from core.agent.tools.document_tools import (
            GenerateDocTool,
            ReadDocTool,
            ListDocsTool,
            OpenDocTool,
        )
        from core.agent.tools.python_tools import RunScriptTool
        from core.agent.tools.view_image_tool import ViewImageTool
        # bash 门控:默认锁定,agent 需经 request_bash(reason) 向用户申请,
        # 审批卡批准后才解锁(仅当前任务内有效,chat() 每条用户消息重置)
        from core.agent.tools.bash_gate import BashGate, LockedBashTool, RequestBashTool
        self._bash_gate = BashGate()
        self._bash_gate.approval_handler = self._request_bash_approval
        self._bash_gate.stop_checker = lambda: bool(
            self._worker is not None and getattr(self._worker, "_stop_requested", False)
        )
        corecoder_tools = [
            LockedBashTool(t, self._bash_gate) if t.name == "bash" else t
            for t in ALL_TOOLS
        ]
        corecoder_tools.append(RequestBashTool(self._bash_gate))
        corecoder_tools.append(self._make_ask_user_tool())
        self._core_tools = corecoder_tools + [
            GetStateTool(),
            ListPromptSectionsTool(),
            InjectPromptTool(),
            GetActivePromptsTool(),
            ToolSearchAgent(),
            ActivateToolsTool(),
            GenerateDocTool(),
            ReadDocTool(),
            ListDocsTool(),
            OpenDocTool(),
            RunScriptTool(),
            ViewImageTool(),
        ]

        # 启动意图分析子 Agent（订阅操作记录事件，LLM 通过 QApplication 反查获取）
        try:
            from core.agent.intent_agent import IntentAgent
            IntentAgent.instance().start()
        except Exception as e:
            print(f"[VisionMindAgent] 启动意图分析子 Agent 失败: {e}")

        # 初始化动态 tool 管理器
        self._tool_manager = DynamicToolManager(main_window)

        # 获取初始 tool 列表
        tools = self._tool_manager.get_tools_for_context(core_tools=self._core_tools)

        # 构建自定义系统提示词
        from core.agent.system_prompt import build_system_prompt
        system = build_system_prompt(tools)

        self._agent = Agent(llm=llm, tools=tools)
        agent_holder["agent"] = self._agent
        self._agent._system = system  # 替换默认系统提示词
        # 每轮处理后把历史中的多模态 base64 图片替换为结构化文本。
        # 此时代理本轮 assistant 已 append 到 self.messages,图片之后的解读可用。
        from core.agent.multimodal import replace_image_urls_with_captions

        def _round_end():
            try:
                self._agent.messages = replace_image_urls_with_captions(self._agent.messages)
            except Exception:
                pass

        self._agent._after_round = _round_end

        # 打开侧栏即显示真实用量(系统提示词 + 工具已就绪)，避免初始全 0
        try:
            self._token_budget.update(system, tools, [])
        except Exception:
            pass
        self._push_context_stats()

        print(f"[VisionMindAgent] 初始化完成，模型: {model}，工具数: {len(tools)}")

    def is_ready(self) -> bool:
        """检查 Agent 是否已初始化"""
        return self._agent is not None

    def set_history(self, messages: list):
        """用加载的会话历史替换 Agent 内存消息(已清理 base64)。

        加载会话后调用,使后续对话带着完整历史继续。直接替换 reference,
        并同步 token 预算。
        """
        if not self._agent:
            return
        cleaned = list(messages or [])
        from core.agent.multimodal import replace_image_urls_with_captions
        cleaned = replace_image_urls_with_captions(cleaned)
        self._agent.messages = [m for m in cleaned if isinstance(m, dict)]
        try:
            self._token_budget.update(
                self._agent._system, self._agent.tools, self._agent.messages
            )
        except Exception:
            pass
        self._push_context_stats()

    def is_busy(self) -> bool:
        """检查 Agent 是否正在执行（worker 线程运行中）"""
        return self._worker is not None and self._worker.isRunning()

    @property
    def last_response(self) -> str:
        """最近一次 Agent 完整回复文本"""
        return self._last_response

    def _update_tools(self, user_input: str = "", recent_categories: list = None):
        """根据上下文动态更新 tool 列表

        Task 9: 上下文感知过滤 — 不再在此处重建 system_prompt，
        由 chat() 流程统一管理系统提示词生命周期。
        """
        if not self._agent or not self._tool_manager:
            return
        current_interface = self._tool_manager.get_current_interface()
        new_tools = self._tool_manager.get_tools_for_context(
            core_tools=self._core_tools,
            user_input=user_input,
            current_interface=current_interface,
            recent_categories=recent_categories or [],
        )
        # 计划模式:执行类工具整体不可见不可调用(含补充索引)
        if self._plan_mode:
            new_tools = [t for t in new_tools if self._is_read_only_tool(t)]
        self._agent.tools = new_tools
        # _tool_by_name 先包含活跃工具（与 agent.tools 一致）
        self._agent._tool_by_name = {t.name: t for t in new_tools}
        # 补充所有 scope="agent" 的工具到索引（不加入 agent.tools，避免 token 膨胀）。
        # 这样 LLM 从 search_tools 结果中看到工具后直接调用也不会 unknown tool，
        # 即使该工具的分类不在当前活跃分类中。
        for tool in self._tool_manager.get_all_ai_action_tools():
            if tool.name not in self._agent._tool_by_name:
                if self._plan_mode and not self._is_read_only_tool(tool):
                    continue
                self._agent._tool_by_name[tool.name] = tool

    @staticmethod
    def _is_read_only_tool(tool) -> bool:
        """ActionTool 查注册表 meta,其余工具查 read_only 类属性(默认执行类)"""
        try:
            from core.agent.tools.action_tool import ActionTool
            if isinstance(tool, ActionTool):
                from core.common.action_registry import ActionRegistry
                meta = ActionRegistry.instance().get_meta(tool._action_name)
                return bool(meta.read_only) if meta else False
        except Exception:
            pass
        return bool(getattr(tool, "read_only", False))

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    def set_plan_mode(self, on: bool):
        """切换计划模式;退出时恢复完整工具集"""
        if self._plan_mode == on:
            return
        self._plan_mode = on
        print(f"[VisionMindAgent] 计划模式: {'开' if on else '关'}")
        if self._agent:
            self._update_tools(user_input="", recent_categories=self._recent_tool_categories)
            from core.agent.system_prompt import mark_dirty
            mark_dirty()

    def _push_context_stats(self):
        """向侧栏推送上下文用量摘要(百分比 + 分项),供悬浮卡展示。"""
        if not self._token_budget:
            return
        try:
            summary = self._token_budget.get_summary()
            summary["max"] = self._token_budget.get_max()
            self.context_stats.emit(
                __import__("json").dumps(summary, ensure_ascii=False)
            )
        except Exception as e:
            print(f"[VisionMindAgent] 上下文统计推送失败: {e}")

    PLAN_PREAMBLE = (
        "\n\n## 计划模式（当前生效）\n\n"
        "你当前处于**计划模式**：\n"
        "- 只开放了只读工具（查询状态/读取知识段/搜索等），用于完成调研\n"
        "- 任何会修改数据、文件、配置的工具都不可见，也严禁设法绕过\n"
        "- 本轮任务：基于用户需求做必要调研，然后输出完整的「## 执行计划」\n"
        "  （目标、分步骤、每步调用的工具与预期效果），交由用户审批\n"
        "- 计划中的执行步骤可包含只读调研中确认存在的执行类工具\n"
        "- 计划输出后停止，等待用户选择「执行」或「继续修改」\n"
    )

    def _build_system(self, tools) -> str:
        from core.agent.system_prompt import build_system_prompt
        system = build_system_prompt(tools)
        if self._plan_mode:
            system += self.PLAN_PREAMBLE
        return system

    def compact_now(self) -> str:
        """手动触发上下文压缩(/compact):保留最近 10 条,旧消息 LLM 摘要。"""
        if not self._agent:
            return "Agent 未初始化。"
        if self.is_busy():
            return "Agent 正在执行任务，请稍后再压缩。"
        from core.agent.message_compressor import compress_messages
        from core.agent.token_budget import estimate_tokens

        before = len(self._agent.messages)
        before_tok = estimate_tokens(self._agent._system) + sum(
            estimate_tokens(str(m.get("content", ""))) for m in self._agent.messages
        )
        self._agent.messages = compress_messages(
            self._agent.messages, keep_recent_n=10,
            llm=self._message_compressor_llm,
        )
        after = len(self._agent.messages)
        after_tok = estimate_tokens(self._agent._system) + sum(
            estimate_tokens(str(m.get("content", ""))) for m in self._agent.messages
        )
        self._compression_history.append({
            "turn": self._conversation_turn, "level": "manual",
            "before": before, "after": after,
        })
        try:
            self._token_budget.update(
                self._agent._system, self._agent.tools, self._agent.messages
            )
        except Exception:
            pass
        self._push_context_stats()
        report = f"上下文已压缩：{before} 条消息 → {after} 条（约 {before_tok} → {after_tok} tokens）。"
        self.compression_report.emit(report)
        return report

    def chat(self, user_input: str | list):
        """异步发送消息给 Agent

        Task 9: 动态提示词注入系统主流程 — 8 步预处理：
          1) 上下文分析
          2) 知识段 LRU 续期
          3) 工具集上下文感知过滤
          4) 知识段注入（按 token 上限 LRU 淘汰）
          5) Token 预算检查
          6) LRU 淘汰（知识段 + 激活工具）
          7) 消息压缩（critical/emergency 时触发）
          8) 兜底重建系统提示词
        """
        # 机制 A: 字符串输入在这里统一解析(UI 路径与 agent.chat action 路径共用)。
        # _on_input_send 已把 content 解析为 list 时直接透传,不二次解析。
        content = user_input
        if isinstance(user_input, str):
            try:
                from core.agent.multimodal import ReferenceResolver
                content = ReferenceResolver().resolve_message(user_input)
            except Exception:
                content = user_input
        # chat() 后续 8 步预处理全部按纯文本匹配(keyword/LRU/知识段注入),
        # 从多模态 content 中提取 text 部件拼接,避免 list 传入 lower()/in 崩溃
        text = content if isinstance(content, str) else " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
        if not self._agent:
            self.error_occurred.emit("Agent 未初始化，请先配置 API Key")
            return

        if self._worker and self._worker.isRunning():
            self.error_occurred.emit("Agent 正在处理中，请稍候")
            return

        self._conversation_turn += 1
        # bash 门控随新任务重置:解锁仅在上一个任务内有效,新任务需重新申请
        if getattr(self, "_bash_gate", None) is not None:
            self._bash_gate.reset()

        from core.agent.prompt_manager import PromptManager
        from core.agent.system_prompt import build_system_prompt, mark_dirty, is_dirty, _get_agents_md
        from core.agent.token_budget import estimate_tokens

        pm = PromptManager.instance()
        tm = self._tool_manager

        # 步骤 1: 上下文分析
        current_interface = tm.get_current_interface()

        # 步骤 2: 知识段续期 — 用户输入匹配的知识段移动到 LRU 最新位置
        pm.touch_by_user_input(text)

        # 步骤 3: 工具集更新 — 上下文感知过滤
        self._update_tools(user_input=text, recent_categories=self._recent_tool_categories)
        new_tools = self._agent.tools

        # 步骤 4: 知识段注入 — 检查系统提示词 token 上限，超限按 LRU 淘汰
        try:
            base_prompt = _get_agents_md()
            base_tokens = estimate_tokens(base_prompt)
            system_prompt_limit = self._token_budget.get_system_prompt_limit()
            injected = pm.auto_inject(
                text, max_count=3,
                system_prompt_limit=system_prompt_limit,
                base_tokens=base_tokens,
            )
            if injected:
                print(f"[VisionMindAgent] 自动注入知识段: {injected}")
                mark_dirty()
        except Exception as e:
            print(f"[VisionMindAgent] 知识段注入失败: {e}")
            injected = []

        # 重建系统提示词（如果有新注入或 dirty）
        need_rebuild = bool(injected) or is_dirty()
        if need_rebuild:
            self._agent._system = build_system_prompt(new_tools)

        # 步骤 5: Token 预算检查
        try:
            self._token_budget.update(self._agent._system, new_tools, self._agent.messages)
            budget_level = self._token_budget.check_budget()
            print(f"[VisionMindAgent] token 预算: {self._token_budget.get_summary()}")
        except Exception as e:
            print(f"[VisionMindAgent] token 预算检查失败: {e}")
            budget_level = "safe"

        # 步骤 6: LRU 淘汰 — 若达 warning 或系统提示词超限
        try:
            if self._token_budget.should_evict():
                # 先淘汰知识段使系统提示词降至上限内
                evicted_sections = pm.evict_until_fit(system_prompt_limit, base_tokens)
                if evicted_sections:
                    print(f"[VisionMindAgent] LRU 淘汰知识段(超上限): {evicted_sections}")
                    mark_dirty()
                # 再按总预算淘汰知识段
                evicted_sections2 = pm.evict_until_budget(self._token_budget, min_keep=1)
                if evicted_sections2:
                    print(f"[VisionMindAgent] LRU 淘汰知识段(预算): {evicted_sections2}")
                    mark_dirty()
                # 淘汰激活工具
                evicted_tools = tm.evict_lru_tools(token_budget=self._token_budget.get_total())
                if evicted_tools:
                    print(f"[VisionMindAgent] LRU 淘汰工具: {evicted_tools}")
                    # 工具集变更，需重新获取
                    self._update_tools(user_input=text, recent_categories=self._recent_tool_categories)
                    new_tools = self._agent.tools
                # 重新估算
                self._token_budget.update(self._agent._system, new_tools, self._agent.messages)
                budget_level = self._token_budget.check_budget()
        except Exception as e:
            print(f"[VisionMindAgent] LRU 淘汰失败: {e}")

        # 步骤 7: 消息压缩 — 按阈值级别触发不同压缩策略
        # 软阈值(warning, 50%)：历史消息超滑动窗口(10)时压缩
        # 硬阈值(critical, 65%)：截断工具结果 + 缩小滑动窗口至 5 条
        # 临界阈值(emergency, 75%)：紧急截断 + 清除非核心知识段和激活工具
        if budget_level in ("warning", "critical", "emergency"):
            try:
                from core.agent.message_compressor import (
                    compress_messages, truncate_tool_results, emergency_truncate,
                )
                original_count = len(self._agent.messages)
                if budget_level == "warning":
                    # 软阈值：仅当历史消息超过滑动窗口时压缩
                    if original_count > 10:
                        self._agent.messages = compress_messages(
                            self._agent.messages, keep_recent_n=10,
                            llm=self._message_compressor_llm,
                        )
                elif budget_level == "critical":
                    # 硬阈值：截断工具结果 + 缩小滑动窗口至 5 条
                    self._agent.messages = truncate_tool_results(
                        self._agent.messages, max_chars=200, keep_recent_n=5
                    )
                    self._agent.messages = compress_messages(
                        self._agent.messages, keep_recent_n=5,
                        llm=self._message_compressor_llm,
                    )
                else:  # emergency
                    # 临界阈值：紧急截断
                    self._agent.messages = emergency_truncate(
                        self._agent.messages, keep_recent_n=5
                    )
                    # 清除所有非核心已注入知识段和激活工具
                    pm.reset_injected()
                    tm.reset_activated()
                    mark_dirty()
                new_count = len(self._agent.messages)
                if new_count != original_count:
                    self._compression_history.append({
                        "turn": self._conversation_turn,
                        "level": budget_level,
                        "before": original_count,
                        "after": new_count,
                    })
                    print(f"[VisionMindAgent] 消息压缩: {original_count} → {new_count} (level={budget_level})")
                    # 阈值触发的自动压缩同样在界面以可展开工具卡片展示
                    level_names = {"warning": "软阈值", "critical": "硬阈值", "emergency": "临界"}
                    self.compression_report.emit(
                        f"上下文已压缩：{original_count} 条消息 → {new_count} 条"
                        f"（触发级别：{level_names.get(budget_level, budget_level)}）"
                    )
                # 重建系统提示词
                if is_dirty():
                    self._agent._system = self._build_system(new_tools)
                    self._token_budget.update(self._agent._system, new_tools, self._agent.messages)
            except Exception as e:
                print(f"[VisionMindAgent] 消息压缩失败: {e}")

        # 步骤 8: 若有变更则重建系统提示词（兜底）
        if is_dirty():
            self._agent._system = self._build_system(new_tools)

        self._push_context_stats()

        # 启动 LLMWorker
        self._worker = LLMWorker(self._agent, content, self._tool_manager)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self.error_occurred.emit)
        self._worker.tool_called.connect(self._on_tool_called)
        self._worker.tool_result.connect(self._on_tool_result)
        self._worker.token_received.connect(self.token_received.emit)
        self._worker.stopped.connect(self.stopped.emit)
        self._worker.start()

    def _on_worker_finished(self, result: str):
        """worker 正常完成：记录最近回复并转发响应信号"""
        self._last_response = result
        self._push_context_stats()
        self.response_ready.emit(result)

    def _on_tool_result(self, tool_name: str, result: str):
        """工具结果:消息历史增长,推送上下文统计后转发"""
        self._push_context_stats()
        self.tool_result.emit(tool_name, result)

    def _on_tool_called(self, tool_name: str, args: dict):
        """工具调用回调：更新 LRU 顺序 + 记录最近分类 + 转发信号

        Task 9: 此方法连接到 LLMWorker.tool_called 信号，在工具调用时
        - 更新工具在 DynamicToolManager 中的 LRU 位置
        - 记录最近使用的工具分类（保留最近 5 个）
        - 转发 tool_called 信号给上层 UI
        """
        # 更新工具 LRU 顺序
        if self._tool_manager:
            self._tool_manager.touch_tool(tool_name)
        # 记录最近使用的工具分类
        try:
            from core.common.action_registry import ActionRegistry
            registry = ActionRegistry.instance()
            meta = registry.get_meta(tool_name)
            if meta and meta.category:
                if meta.category not in self._recent_tool_categories:
                    self._recent_tool_categories.insert(0, meta.category)
                    # 只保留最近 5 个分类
                    self._recent_tool_categories = self._recent_tool_categories[:5]
        except Exception as e:
            print(f"[VisionMindAgent] 记录工具分类失败: {e}")
        # 转发信号
        self.tool_called.emit(tool_name, args)

    def _request_bash_approval(self, reason: str) -> bool:
        """工作线程调用：向 UI 发出 bash 解锁审批请求并阻塞等待用户决定。

        经 bash_approval_requested 信号跨线程投递到主线程（AgentDialog
        弹审批卡），用户点击允许/拒绝后经 respond 回调唤醒。超时 120s
        或用户点「停止」均视为拒绝。
        """
        import threading

        ev = threading.Event()
        holder = {"ok": False}

        def respond(ok: bool):
            holder["ok"] = bool(ok)
            ev.set()

        self.bash_approval_requested.emit(reason, respond)
        waited = 0.0
        while not ev.wait(0.25):
            waited += 0.25
            if waited >= 120.0:
                break
            worker = self._worker
            if worker is not None and getattr(worker, "_stop_requested", False):
                break
        return holder["ok"]

    def _make_ask_user_tool(self):
        """ask_user 工具:工作线程阻塞等待用户在侧栏作答(仅「停止」可打断)。"""
        import json as _json
        import threading

        from core.agent.tools.ask_user_tool import AskUserTool

        def requester(qlist):
            ev = threading.Event()
            holder = {"answers": None}

            def respond(answers):
                holder["answers"] = answers
                ev.set()

            self.ask_user_requested.emit(
                _json.dumps(qlist, ensure_ascii=False), respond
            )
            while not ev.wait(0.25):
                worker = self._worker
                if worker is not None and getattr(worker, "_stop_requested", False):
                    return None
            return holder["answers"]

        return AskUserTool(requester)

    def stop(self):
        """用户请求停止当前执行。

        设置 worker 的停止标志，使其在下一个工具调用前抛出 RuntimeError 退出。
        若 agent.chat() 正在进行 LLM 请求（无法中断），则在该请求返回后的
        下一次工具调用前退出。此方法不阻塞，立即返回。
        """
        if self._worker and self._worker.isRunning():
            print("[VisionMindAgent] 用户请求停止执行")
            self._worker.request_stop()
        else:
            print("[VisionMindAgent] 无正在运行的 worker，无需停止")

    def stop_and_wait(self, max_ms: int = 15000) -> bool:
        """请求停止当前 worker 并等待其退出（最多 max_ms 毫秒）。

        若 worker 正阻塞在 LLM 网络请求中，停止标志将在该请求返回后于
        下一个工具调用边界生效，因此等待时间取决于当前 LLM 调用剩余时长。
        使用 QThread.wait 阻塞等待，不泵事件循环，避免干扰外部事件调度。
        返回 True=已空闲，False=超时仍忙。
        """
        if not (self._worker and self._worker.isRunning()):
            return True
        self._worker.request_stop()
        finished = self._worker.wait(max_ms)
        if finished:
            print("[VisionMindAgent] 已停止上一任务")
        else:
            print(f"[VisionMindAgent] 等待 worker 退出超时（>{max_ms}ms）")
        return finished

    def reset(self):
        """重置对话历史（重置前自动保存当前会话）

        Task 9: 清除所有动态提示词注入系统的生命周期状态：
        - 工具激活 LRU 队列
        - 知识段 LRU 注入状态
        - Token 预算估算
        - 对话轮次/分类/压缩历史
        - 系统提示词脏标记
        """
        try:
            self.save_session()
        except Exception as e:
            print(f"[VisionMindAgent] 重置时保存会话失败（忽略）: {e}")
        if self._agent:
            self._agent.reset()
        # 清除工具激活 LRU 队列
        if self._tool_manager:
            self._tool_manager.reset_activated()
        # 清除知识段 LRU
        from core.agent.prompt_manager import PromptManager
        PromptManager.instance().reset_injected()
        # 重置 token 预算状态
        if self._token_budget:
            self._token_budget.update("", [], [])
        # 重置计数器和历史
        self._conversation_turn = 0
        self._recent_tool_categories = []
        self._compression_history = []
        self._last_response = ""
        # 标记系统提示词需重建
        from core.agent.system_prompt import mark_dirty
        mark_dirty()

    @staticmethod
    def _redact_messages(messages):
        """对会话消息中的敏感凭据做掩码，防止 API Key 明文落盘到会话日志。

        扫描每条消息的 content 字段，将形如 sk-xxxx... 的凭据替换为掩码形式。
        支持字符串 content 与多模态 list content（仅掩码 text 部件）。
        """
        import re

        # 匹配常见 API Key 前缀模式（sk-、sk-ant-、sk-proj- 等）+ 足够长的字符
        key_pattern = re.compile(
            r'(sk-[a-zA-Z0-9\-_]{20,})',
            re.IGNORECASE
        )

        def _redact_text(text: str) -> str:
            if key_pattern.search(text):
                return key_pattern.sub(
                    lambda m: m.group(1)[:6] + "***" + m.group(1)[-4:],
                    text
                )
            return text

        redacted = []
        for msg in messages:
            if not isinstance(msg, dict):
                redacted.append(msg)
                continue
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, str):
                new_msg["content"] = _redact_text(content)
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        p = dict(part)
                        p["text"] = _redact_text(part.get("text", ""))
                        new_parts.append(p)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts
            redacted.append(new_msg)
        return redacted

    def save_session(self):
        """保存当前对话到磁盘（含调试元信息）

        保存位置：~/.corecoder/sessions/<id>.json（CoreCoder 标准格式）
        额外保存：~/.corecoder/sessions/_visionmind_<id>.json（调试用完整快照）

        保存前会对消息中的 API Key 等敏感凭据做掩码，避免明文落盘。
        """
        if not self._agent:
            return
        if not self._agent.messages:
            return

        model = getattr(self._agent.llm, 'model', 'unknown')

        # 0. 落盘前先收敛多模态 base64 图片:删除图片 user 消息、解读写回 tool,
        #    保证保存的会话不含 base64(即使对话已结束于图片中间态,也能兜底清理)。
        try:
            from core.agent.multimodal import replace_image_urls_with_captions
            cleaned = replace_image_urls_with_captions(self._agent.messages)
        except Exception:
            cleaned = self._agent.messages

        # 2. 掩码敏感凭据后再保存，防止 API Key 明文写入会话文件
        safe_messages = self._redact_messages(cleaned)

        # 3. 保存标准 CoreCoder 会话（纯 messages + model）
        from core.corecoder.session import save_session
        session_id = save_session(safe_messages, model)

        # 2. 保存调试扩展信息
        import json, os
        from core.agent.prompt_manager import PromptManager
        from core.corecoder.session import SESSIONS_DIR

        # 读取已注入的知识段
        pm = PromptManager.instance()
        active = pm.get_active_sections()
        injected_info = []
        for s in active:
            injected_info.append({
                "id": s.section_id,
                "title": s.title,
                "category": s.category,
                "scenes": s.scenes,
            })

        # 当前工具列表（仅名称+分类）
        tools_info = []
        for t in self._agent.tools:
            cat = ""
            from core.agent.tools.action_tool import ActionTool
            if isinstance(t, ActionTool):
                from core.common.action_registry import ActionRegistry
                meta = ActionRegistry.instance().get_meta(t._action_name)
                cat = meta.category if meta else ""
            tools_info.append({
                "name": t.name,
                "category": cat,
                "description": t.description[:100] if t.description else "",
            })

        debug_data = {
            "session_id": session_id,
            "model": model,
            "saved_at": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": getattr(self._agent, '_system', '')[:2000],
            "total_messages": len(self._agent.messages),
            "message_roles": [m.get("role") for m in self._agent.messages],
            "injected_sections": injected_info,
            "tool_count": len(tools_info),
            "tools": tools_info,
            # Task 9: 动态提示词注入系统状态快照
            "token_budget": self._token_budget.get_summary() if self._token_budget else None,
            "compression_history": list(self._compression_history),
            "conversation_turn": self._conversation_turn,
            "recent_tool_categories": list(self._recent_tool_categories),
            "activated_tools": self._tool_manager.get_activated_tool_names() if self._tool_manager else [],
            "injected_lru_order": PromptManager.instance().get_lru_order(),
        }

        debug_path = os.path.join(
            str(SESSIONS_DIR),
            f"_visionmind_{session_id}.json"
        )
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[VisionMindAgent] 保存调试信息失败: {e}")

        print(f"[VisionMindAgent] 会话已保存: {session_id} ({len(self._agent.messages)} 条消息)")
        return session_id
