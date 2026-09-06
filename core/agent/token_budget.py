"""
Token 预算管理 — 估算上下文 token 占用并按阈值分级告警

负责：
- 估算文本 / 工具 schema / OpenAI 消息列表的 token 数
- 跟踪系统提示词（base + 已注入知识段）、工具、消息三部分占用
- 按占 max_context 的百分比划分 safe / warning / critical / emergency 四级
- 提供 LRU 淘汰触发判断与可供知识段使用的剩余预算
"""

from __future__ import annotations

import json
from typing import Any, Dict

# 已注入插件知识段的分隔标记（与 PromptManager.get_active_formatted 保持一致）
_SECTIONS_MARKER = "# 已注入的插件知识段"


def estimate_tokens(text: str) -> int:
    """估算文本 token 数。优先 tiktoken，回退字符估算。"""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        # 混合中英文估算：约 3 字符 / token
        return max(1, len(text) // 3)


def estimate_tools_tokens(tools: list) -> int:
    """估算工具列表的 schema 总 token 数。

    对每个 tool，序列化其 name、description、parameters（JSON）为字符串后估算。
    兼容对象（带 name/description/parameters 属性）与 dict 两种形式。
    """
    if not tools:
        return 0
    total = 0
    for tool in tools:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        parameters = getattr(tool, "parameters", None)
        if isinstance(tool, dict):
            name = tool.get("name", name)
            description = tool.get("description", description)
            parameters = tool.get("parameters", parameters)

        parts = []
        if name is not None:
            parts.append(str(name))
        if description is not None:
            parts.append(str(description))
        if parameters is not None:
            try:
                parts.append(json.dumps(parameters, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(parameters))
        total += estimate_tokens(" ".join(parts))
    return total


def estimate_messages_tokens(messages: list) -> int:
    """估算 OpenAI 格式消息列表的总 token 数。

    对每条 message，序列化 role、content（若有 tool_calls 也序列化）为字符串后估算。
    """
    if not messages:
        return 0
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            total += estimate_tokens(str(msg))
            continue

        parts = []
        role = msg.get("role")
        if role is not None:
            parts.append(str(role))
        content = msg.get("content")
        if content is not None:
            # 多模态列表(含 base64 图片)不应整体编码估算——满分把它
            # str() 后喂给 tiktoken,既慢又把整张图 base64 计成海量 token,
            # 会虚高触发压缩。text 部分正常编码,图片部件按固定小额计。
            if isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "text":
                        parts.append(str(p.get("text", "")))
                    elif ptype == "image_url":
                        parts.append("[image]")
            else:
                parts.append(str(content))
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            try:
                parts.append(json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(tool_calls))
        total += estimate_tokens(" ".join(parts))
    return total


class TokenBudgetManager:
    """上下文 token 预算管理器

    将上下文占用拆分为三部分：
    - system_prompt：系统提示词（base + 已注入知识段）
    - tools：工具 schema
    - messages：对话消息列表

    并与 max_context_tokens 对比，划分四级告警：
    - safe      ：<50%
    - warning   ：50%-65%
    - critical  ：65%-75%
    - emergency ：>75%
    """

    def __init__(self, max_context_tokens: int = 32000, system_prompt_token_limit: int = 2000):
        self._max_context = max_context_tokens
        self._system_prompt_limit = system_prompt_token_limit
        self._system_prompt_tokens = 0
        self._tools_tokens = 0
        self._messages_tokens = 0
        self._base_prompt_tokens = 0   # Prompt.md 基础提示词
        self._sections_tokens = 0      # 已注入知识段

    def update(self, system_prompt: str, tools: list, messages: list) -> None:
        """更新所有估算值。

        分别估算 system_prompt、tools、messages 的 token 数；
        同时按 "# 已注入的插件知识段" 标记将 system_prompt 拆分为 base 和 sections。
        """
        # 工具与消息
        self._tools_tokens = estimate_tools_tokens(tools)
        self._messages_tokens = estimate_messages_tokens(messages)

        # 系统提示词拆分 base / sections
        if system_prompt:
            idx = system_prompt.find(_SECTIONS_MARKER)
            if idx >= 0:
                base_text = system_prompt[:idx]
                sections_text = system_prompt[idx:]
            else:
                base_text = system_prompt
                sections_text = ""
            self._base_prompt_tokens = estimate_tokens(base_text)
            self._sections_tokens = estimate_tokens(sections_text)
        else:
            self._base_prompt_tokens = 0
            self._sections_tokens = 0
        self._system_prompt_tokens = self._base_prompt_tokens + self._sections_tokens

    def check_budget(self) -> str:
        """返回预算等级：safe / warning / critical / emergency。"""
        total = self.get_total()
        if self._max_context <= 0:
            return "emergency"
        percentage = total / self._max_context * 100
        if percentage < 50:
            return "safe"
        elif percentage < 65:
            return "warning"
        elif percentage <= 75:
            return "critical"
        else:
            return "emergency"

    def check_system_prompt(self) -> bool:
        """返回系统提示词总 token（base + sections）是否超过上限。"""
        return self._system_prompt_tokens > self._system_prompt_limit

    def should_evict(self) -> bool:
        """返回是否应触发 LRU 淘汰（warning 及以上，或系统提示词超限）。"""
        if self.check_budget() != "safe":
            return True
        if self.check_system_prompt():
            return True
        return False

    def get_total(self) -> int:
        """返回总 token 数。"""
        return self._system_prompt_tokens + self._tools_tokens + self._messages_tokens

    def get_max(self) -> int:
        """返回上下文窗口上限。"""
        return self._max_context

    def get_summary(self) -> Dict[str, Any]:
        """返回预算摘要字典。"""
        total = self.get_total()
        level = self.check_budget()
        percentage = (total / self._max_context * 100) if self._max_context > 0 else 100.0
        return {
            "system_prompt": self._system_prompt_tokens,
            "tools": self._tools_tokens,
            "messages": self._messages_tokens,
            "total": total,
            "percentage": round(percentage, 2),
            "level": level,
            "system_prompt_over_limit": self.check_system_prompt(),
            "base_prompt": self._base_prompt_tokens,
            "sections": self._sections_tokens,
        }

    def get_system_prompt_limit(self) -> int:
        """返回系统提示词上限。"""
        return self._system_prompt_limit

    def get_available_for_sections(self) -> int:
        """返回可供知识段使用的 token 数 = system_prompt_token_limit - base_prompt_tokens。"""
        return max(0, self._system_prompt_limit - self._base_prompt_tokens)
