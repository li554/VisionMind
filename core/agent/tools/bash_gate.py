"""bash 工具门控 —— 默认锁定，经用户审批后解锁，且仅当前任务内有效

问题：bash 长期可用时 agent 容易绕过内置 action 直接 shell 乱操作
（扫盘注册模型、读磁盘 JSON、sleep 等待等）；而完全禁用又让少量
确无内置工具的操作（运行外部命令行程序等）无从下手。

方案（硬约束，不依赖提示词自觉）：
- LockedBashTool 包装原 BashTool：未解锁时调用直接拒绝，命令不执行。
- RequestBashTool（request_bash）：把解锁申请以审批卡形式展示给用户，
  用户点「允许」才解锁；拒绝/超时/停止均不解锁。审批通道缺失时
  （无头环境/测试）退化为记录理由后放行。
- 解锁仅在当前任务内有效：VisionMindAgent.chat() 开头重置。
- 同轮并行调用（corecoder _exec_tools_parallel）：request_bash 等待
  审批期间被并行调用的 bash 会等待审批结果，避免"同轮申请+执行"
  时 bash 抢跑拿到锁定错误。
"""

import threading
import time

from core.corecoder.tools.base import Tool

_BASH_LOCKED_MSG = (
    "[bash 已锁定] 本次调用被拒绝，命令未执行。\n"
    "平台绝大多数操作都有内置 action 工具（先 search_tools 确认），"
    "深度分析用 run_script。\n"
    "仅当确认平台没有任何内置工具、run_script 也无法完成该操作时"
    "（如需要运行外部命令行程序），才调用 request_bash(reason=...) "
    "向用户申请解锁，批准后才能使用。"
)

_BASH_DENIED_MSG = (
    "[bash 解锁被拒] 用户未批准本次解锁申请（拒绝/超时/已停止），命令未执行。\n"
    "不得重复申请解锁。请改用平台内置工具或 run_script 完成任务；"
    "确实无法完成时向用户说明原因并停止。"
)

# request_bash 等待用户审批时,并行 bash 调用的最长等待(秒)
_WAIT_APPROVAL_TIMEOUT = 180.0


class BashGate:
    """bash 解锁状态（每条用户消息重置）"""

    def __init__(self):
        self.reason = ""
        # 审批通道: callable(reason) -> bool，阻塞直到用户决定。
        # None = 无 UI 通道，RequestBashTool 记录理由后自动放行。
        self.approval_handler = None
        # 停止检查: callable() -> bool，用户点了「停止」时等待循环提前退出
        self.stop_checker = None
        # request_bash 正在等待用户审批（并行的 bash 调用据此等待）
        self.approval_pending = False
        # 最近一次解锁申请被拒（bash 调用据此返回"不得重复申请"）
        self.denied = False

    @property
    def unlocked(self) -> bool:
        return bool(self.reason)

    def unlock(self, reason: str):
        self.reason = (reason or "").strip()
        self.denied = False

    def reset(self):
        self.reason = ""
        self.approval_pending = False
        self.denied = False

    def is_stopped(self) -> bool:
        try:
            return bool(self.stop_checker()) if self.stop_checker else False
        except Exception:
            return False


class LockedBashTool(Tool):
    read_only = False
    """默认拒绝执行的 bash；用户批准解锁后透传给原始 BashTool。

    description 覆写为门控语义，LLM 从工具列表就能看到调用前置条件。
    """

    def __init__(self, inner, gate: BashGate):
        self._inner = inner
        self._gate = gate
        self.name = inner.name
        self.description = (
            "Execute a shell command。默认锁定：直接调用会被拒绝，命令不会执行。"
            "仅当平台内置工具（search_tools 确认后）与 run_script 都无法完成操作时，"
            "先调用 request_bash(reason=...) 向用户申请解锁，批准后才能使用。"
            "解锁仅当前任务内有效。与 request_bash 同一条消息调用时，"
            "本工具会等待用户审批结果后再执行。"
        )
        self.parameters = inner.parameters

    def execute(self, **kwargs) -> str:
        if not self._gate.unlocked:
            # 与 request_bash 同轮并行时,等待用户审批结果再判定
            waited_for_approval = False
            if self._gate.approval_pending:
                waited_for_approval = True
                waited = 0.0
                while self._gate.approval_pending and waited < _WAIT_APPROVAL_TIMEOUT:
                    if self._gate.is_stopped():
                        raise KeyboardInterrupt("用户已停止执行")
                    time.sleep(0.25)
                    waited += 0.25
            if not self._gate.unlocked:
                # 等过审批或近期有被拒申请 → 提示不得重复申请
                denied = waited_for_approval or self._gate.denied
                return _BASH_DENIED_MSG if denied else _BASH_LOCKED_MSG
        return self._inner.execute(**kwargs)


class RequestBashTool(Tool):
    read_only = False
    """向用户申请解锁 bash（审批卡展示理由，批准后仅当前任务内有效）"""

    name = "request_bash"
    description = (
        "向用户申请解锁 bash 工具。调用后会在对话中弹出审批卡展示你的理由，"
        "用户点「允许」后解锁（仅当前任务内有效，下一条用户消息后重新锁定），"
        "拒绝或超时则不解锁且不得重复申请。"
        "必须已通过 search_tools 确认平台没有任何内置工具、run_script 也无法完成"
        "目标操作后才可调用；reason 需写明要做什么、为什么内置工具做不到。"
        "解锁后仍严禁用 bash 实现平台已有功能（数据查询/标注/配置/模型注册等）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "解锁理由：要执行什么操作、为什么内置工具无法完成",
            },
        },
        "required": ["reason"],
    }

    def __init__(self, gate: BashGate):
        self._gate = gate

    def execute(self, reason: str = "") -> str:
        reason = (reason or "").strip()
        if not reason:
            return "错误: 必须提供 reason 说明解锁理由"
        handler = self._gate.approval_handler
        if handler is None:
            # 无 UI 审批通道(无头环境/测试):记录理由后放行
            self._gate.unlock(reason)
            print(f"[BashGate] bash 已解锁(无审批通道,自动放行): {reason}")
            return (
                f"bash 已解锁（仅本次任务有效）。理由已记录: {reason}\n"
                "注意: 平台已有内置工具能完成的操作仍禁止用 bash 实现。"
            )
        self._gate.approval_pending = True
        try:
            approved = bool(handler(reason))
        finally:
            self._gate.approval_pending = False
        if not approved:
            self._gate.denied = True
            print(f"[BashGate] bash 解锁被拒: {reason}")
            return (
                "用户未批准本次 bash 解锁申请（拒绝/超时/已停止）。\n"
                "不得再次申请解锁。请改用平台内置工具或 run_script 完成任务；"
                "确实无法完成时向用户说明原因并停止。"
            )
        self._gate.unlock(reason)
        print(f"[BashGate] bash 已解锁(用户批准): {reason}")
        return (
            f"用户已批准解锁 bash（仅本次任务有效）。申请理由: {reason}\n"
            "注意: 平台已有内置工具能完成的操作仍禁止用 bash 实现。"
        )
