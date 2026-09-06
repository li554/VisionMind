"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import inspect
from .llm import LLM
from .tools import ALL_TOOLS
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        # 系统提示词必须位于消息列表最前(部分 API 强制 "system message must be
        # at the beginning")。历史中可能混入压缩摘要等 role=="system" 的消息,
        # 合并到主系统提示词之后、统一置于最前,避免出现"中间 system"。
        system_blocks = [self._system] if self._system else []
        rest: list[dict] = []
        for m in self.messages:
            if m.get("role") == "system" and isinstance(m.get("content"), str) and m["content"].strip():
                system_blocks.append(m["content"])
            else:
                rest.append(m)
        combined = "\n\n".join(b for b in system_blocks if b)
        return [{"role": "system", "content": combined}] + rest

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        for _ in range(self.max_rounds):
            resp = self.llm.chat(
                messages=self._full_messages(),
                tools=self._tool_schemas(),
                on_token=on_token,
            )

            # no tool calls -> LLM is done, return text
            if not resp.tool_calls:
                self.messages.append(resp.message)
                # return 前也同步历史状态(如收敛最后注入的多模态图片 base64),
                # 否则最后一条注入的图片 user 消息会因提前 return 而残留
                self._after_round()
                return resp.content

            # tool calls -> execute (parallel when multiple, like Claude Code's
            # StreamingToolExecutor which runs independent tools concurrently)
            self.messages.append(resp.message)

            try:
                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # compress if tool outputs are big
            self.context.maybe_compress(self.messages, self.llm)

            # 每轮已处理(assistant 已 append)后,交给子类/宿主同步历史状态。
            # 用于按需替换历史中的多模态 base64 图片(见 multimodal 模块)。
            self._after_round()

        return "(reached maximum tool-call rounds)"

    def _after_round(self):
        """每轮 LLM/tool 处理完成后回调,供宿主覆写。默认空实现。"""

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Execute tool calls, returning results in call order.

        分组策略:连续的只读工具并行执行(实际 action 实体经
        main_thread_dispatcher 仍在主线程串行,并行只是让等待重叠),
        执行类工具按调用顺序串行。结果按调用顺序回填,LLM 历史顺序不变。
        """
        results: list = [None] * len(tool_calls)
        i = 0
        while i < len(tool_calls):
            tc = tool_calls[i]
            if on_tool:
                on_tool(tc.name, tc.arguments)
            if not self._is_read_only_call(tc):
                results[i] = self._exec_tool(tc)
                i += 1
                continue
            # 收集连续只读段
            j = i
            batch = []
            while j < len(tool_calls) and self._is_read_only_call(tool_calls[j]):
                if j != i and on_tool:
                    on_tool(tool_calls[j].name, tool_calls[j].arguments)
                batch.append(tool_calls[j])
                j += 1
            if len(batch) == 1:
                results[i] = self._exec_tool(batch[0])
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(8, len(batch))
                ) as pool:
                    outs = list(pool.map(self._exec_tool, batch))
                for k, out in enumerate(outs):
                    results[i + k] = out
            i = j
        return results

    def _is_read_only_call(self, tc) -> bool:
        tool = getattr(self, "_tool_by_name", {}).get(tc.name)
        return bool(getattr(tool, "read_only", False)) if tool else False

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
