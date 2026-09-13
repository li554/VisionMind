"""高危工具审批门 —— edit_file / write_file / run_script 默认锁定，经用户审批后解锁

问题：这些工具可直接修改磁盘文件（写代码/数据），agent 未经确认就能改动
平台源码或业务数据；而完全禁用又让少量确需写文件的场景无从下手。

方案（硬约束，不依赖提示词自觉，复用 bash 门控模式）：
- LockedTool 包装原工具：未解锁时调用直接拒绝，不产生副作用。
- RequestApprovalTool（request_approval）：把解锁申请以审批卡形式展示给
  用户，用户点「允许」才解锁；拒绝/超时/停止均不解锁。审批通道缺失时
  （无头环境/测试）退化为记录理由后放行。
- 解锁仅在当前任务内有效：VisionMindAgent.chat() 开头重置。
- 同轮并行调用：request_approval 等待审批期间被并行调用的高危工具会等待
  审批结果，避免"同轮申请+执行"时抢跑拿到锁定错误。
"""

import threading
import time

from core.corecoder.tools.base import Tool

# 纳入审批门的高危工具名
HIGH_RISK_TOOLS = ("edit_file", "write_file", "run_script")

_LOCKED_MSG_TMPL = (
    "[{name} 已锁定] 本次调用被拒绝，文件/数据未被改动。\n"
    "该工具可直接修改磁盘文件，使用前必须先调用 "
    "request_approval(tool=\"{name}\", reason=...) 向用户申请解锁，"
    "批准后才能使用。解锁仅当前任务内有效。"
)

_DENIED_MSG_TMPL = (
    "[{name} 解锁被拒] 用户未批准本次解锁申请（拒绝/超时/已停止），文件/数据未被改动。\n"
    "不得重复申请解锁。请改用平台内置工具完成任务；"
    "确实无法完成时向用户说明原因并停止。"
)

# request_approval 等待用户审批时,并行高危调用的最长等待(秒)
_WAIT_APPROVAL_TIMEOUT = 180.0


class ApprovalGate:
    """高危工具解锁状态（每条用户消息重置）"""

    def __init__(self):
        # tool_name -> 解锁理由；非空即视为已解锁
        self._approved: dict = {}
        # 审批通道: callable(tool_name, reason) -> bool，阻塞直到用户决定。
        # None = 无 UI 通道，RequestApprovalTool 记录理由后自动放行。
        self.approval_handler = None
        # 停止检查: callable() -> bool，用户点了「停止」时等待循环提前退出
        self.stop_checker = None
        # request_approval 正在等待用户审批的工具名（并行的同工具调用据此等待）
        self.approval_pending = set()
        # 最近一次解锁申请被拒的工具名（后续调用据此返回"不得重复申请"）
        self.denied = set()

    def is_unlocked(self, tool_name: str) -> bool:
        return tool_name in self._approved

    def unlock(self, tool_name: str, reason: str):
        self._approved[tool_name] = (reason or "").strip()
        self.denied.discard(tool_name)

    def mark_denied(self, tool_name: str):
        self.denied.add(tool_name)

    def reset(self):
        self._approved.clear()
        self.approval_pending.clear()
        self.denied.clear()

    def is_stopped(self) -> bool:
        try:
            return bool(self.stop_checker()) if self.stop_checker else False
        except Exception:
            return False


class LockedTool(Tool):
    read_only = False
    """默认拒绝执行的高危工具；用户批准解锁后透传给原始工具。

    description 覆写为门控语义，LLM 从工具列表就能看到调用前置条件。
    """

    def __init__(self, inner: Tool, gate: ApprovalGate):
        self._inner = inner
        self._gate = gate
        self.name = inner.name
        self.description = (
            f"默认锁定：直接调用会被拒绝，不会产生任何改动。"
            f"使用前必须先调用 request_approval(tool=\"{inner.name}\", reason=...) "
            f"向用户申请解锁，批准后才能使用。解锁仅当前任务内有效。"
            f"与 request_approval 同一条消息调用时，本工具会等待用户审批结果后再执行。"
        )
        self.parameters = inner.parameters

    def execute(self, **kwargs) -> str:
        if not self._gate.is_unlocked(self.name):
            # 与 request_approval 同轮并行时,等待用户审批结果再判定
            waited_for_approval = False
            if self.name in self._gate.approval_pending:
                waited_for_approval = True
                waited = 0.0
                while self.name in self._gate.approval_pending and waited < _WAIT_APPROVAL_TIMEOUT:
                    if self._gate.is_stopped():
                        raise KeyboardInterrupt("用户已停止执行")
                    time.sleep(0.25)
                    waited += 0.25
            if not self._gate.is_unlocked(self.name):
                # 等过审批或近期有被拒申请 → 提示不得重复申请
                denied = waited_for_approval or self.name in self._gate.denied
                if denied:
                    return _DENIED_MSG_TMPL.format(name=self.name)
                return _LOCKED_MSG_TMPL.format(name=self.name)
        return self._inner.execute(**kwargs)


class RequestApprovalTool(Tool):
    read_only = False
    """向用户申请解锁高危工具（审批卡展示理由，批准后仅当前任务内有效）"""

    name = "request_approval"
    description = (
        "向用户申请解锁高危工具（edit_file / write_file / run_script）。调用后会在"
        "对话中弹出审批卡展示你的理由，用户点「允许」后解锁（仅当前任务内有效，"
        "下一条用户消息后重新锁定），拒绝或超时则不解锁且不得重复申请。"
        "仅当确认平台内置工具无法完成目标操作时才可调用；reason 需写明要做什么、"
        "为什么内置工具做不到。解锁后仍严禁用这些工具实现平台已有功能"
        "（数据查询/标注/配置/模型注册等）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": list(HIGH_RISK_TOOLS),
                "description": "要解锁的高危工具名",
            },
            "reason": {
                "type": "string",
                "description": "解锁理由：要执行什么操作、为什么内置工具无法完成",
            },
        },
        "required": ["tool", "reason"],
    }

    def __init__(self, gate: ApprovalGate):
        self._gate = gate

    def execute(self, tool: str = "", reason: str = "") -> str:
        tool = (tool or "").strip()
        reason = (reason or "").strip()
        if tool not in HIGH_RISK_TOOLS:
            return f"错误: tool 必须是 {list(HIGH_RISK_TOOLS)} 之一"
        if not reason:
            return "错误: 必须提供 reason 说明解锁理由"
        handler = self._gate.approval_handler
        if handler is None:
            # 无 UI 审批通道(无头环境/测试):记录理由后放行
            self._gate.unlock(tool, reason)
            print(f"[ApprovalGate] {tool} 已解锁(无审批通道,自动放行): {reason}")
            return (
                f"{tool} 已解锁（仅本次任务有效）。理由已记录: {reason}\n"
                "注意: 平台已有内置工具能完成的操作仍禁止用该工具实现。"
            )
        self._gate.approval_pending.add(tool)
        try:
            approved = bool(handler(tool, reason))
        finally:
            self._gate.approval_pending.discard(tool)
        if not approved:
            self._gate.mark_denied(tool)
            print(f"[ApprovalGate] {tool} 解锁被拒: {reason}")
            return (
                f"用户未批准本次 {tool} 解锁申请（拒绝/超时/已停止）。\n"
                "不得再次申请解锁。请改用平台内置工具完成任务；"
                "确实无法完成时向用户说明原因并停止。"
            )
        self._gate.unlock(tool, reason)
        print(f"[ApprovalGate] {tool} 已解锁(用户批准): {reason}")
        return (
            f"用户已批准解锁 {tool}（仅本次任务有效）。申请理由: {reason}\n"
            "注意: 平台已有内置工具能完成的操作仍禁止用该工具实现。"
        )
