"""
IntentAgent — 意图分析子 Agent（LLM 驱动）

与对话 Agent 分工：
- 对话 Agent：处理用户主动发出的指令
- IntentAgent：实时监测用户操作记录（OperationLogger）+ 当前程序状态，
  通过 LLM 分析用户当前意图并主动提出可执行建议。

与 ToolSearchAgent 同属子 Agent 模式：
- 不创建独立 LLM 实例，通过 QApplication 反查主 Agent 的 LLM 引用
- LLM 调用复用 message_compressor._call_llm
- 独立上下文，不继承主 Agent 对话历史

使用方式：
    agent = IntentAgent.instance()
    agent.start()          # 订阅 OperationLogger.operation_recorded
    agent.analyze_async()  # 异步触发 LLM 分析，结果通过 analysis_updated 信号返回

流程：
    用户操作 → OperationLogger.record → _on_operation（节流）→
    IntentAgentWorker（QThread）→ LLM 分析 → analysis_updated 信号 → 悬浮框刷新
"""

import json
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal


class IntentAgentWorker(QThread):
    """后台线程：调用 LLM 分析操作意图（避免阻塞 UI）"""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, llm: Any, prompt: str, parent=None):
        super().__init__(parent)
        self._llm = llm
        self._prompt = prompt

    def run(self):
        try:
            from core.agent.message_compressor import _call_llm
            text = _call_llm(self._llm, self._prompt)
            if not text:
                self.error.emit("LLM 返回空响应")
                return
            result = self._parse_response(text)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    @staticmethod
    def _parse_response(text: str) -> dict:
        """从 LLM 文本响应中解析 JSON 结果"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start) if "```" in text[start:] else -1
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first:last + 1])
            except json.JSONDecodeError:
                pass
        return {
            "intent": "解析失败",
            "intent_description": "LLM 响应无法解析为 JSON",
            "recent_actions": [],
            "suggestions": [],
            "raw_response": text[:500],
        }


class IntentAgent(QObject):
    """意图分析子 Agent（全局单例，LLM 驱动）"""

    analysis_updated = Signal(dict)  # 新分析结果
    enabled_changed = Signal(bool)   # 意图分析开关变化

    _instance: Optional['IntentAgent'] = None

    def __init__(self):
        super().__init__()
        self._connected = False
        self._last_analysis: Dict = {}
        self._last_suggestion_time = 0.0
        self._suggestion_cooldown = self._load_cooldown()  # LLM 调用最小间隔（秒）
        self._worker: Optional[IntentAgentWorker] = None
        try:
            from core.common.settings import settings
            self._enabled = bool(settings.get("intent_analysis_enabled", True))
        except Exception:
            self._enabled = True

    def _load_cooldown(self) -> float:
        """从配置读取意图分析最小间隔（秒），默认 30 秒"""
        try:
            from core.common.settings import settings
            return float(settings.get("intent_analysis_cooldown", 30.0))
        except Exception:
            return 30.0

    def _on_settings_changed(self, key: str, value):
        """配置变更时动态更新冷却时间"""
        if key == "intent_analysis_cooldown":
            try:
                self._suggestion_cooldown = float(value)
            except (TypeError, ValueError):
                self._suggestion_cooldown = 30.0

    @classmethod
    def instance(cls) -> 'IntentAgent':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_enabled(self) -> bool:
        """意图分析是否开启"""
        return self._enabled

    def set_enabled(self, enabled: bool):
        """开启/关闭意图分析（持久化到配置）"""
        enabled = bool(enabled)
        if self._enabled == enabled:
            return
        self._enabled = enabled
        try:
            from core.common.settings import settings
            settings.set("intent_analysis_enabled", enabled)
        except Exception as e:
            print(f"[IntentAgent] 保存开关状态失败: {e}")
        self.enabled_changed.emit(enabled)
        if enabled:
            self.analyze_async()

    def _get_llm(self):
        """通过 QApplication 找到主窗口的 VisionMindAgent，获取 LLM 引用。"""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            return None
        for widget in app.topLevelWidgets():
            if hasattr(widget, '_ai_agent'):
                agent = widget._ai_agent
                return getattr(agent, '_message_compressor_llm', None)
        return None

    def has_llm(self) -> bool:
        """LLM 是否可用"""
        return self._get_llm() is not None

    def start(self):
        """订阅操作记录事件"""
        if self._connected:
            return
        try:
            from core.agent.operation_logger import OperationLogger
            logger = OperationLogger.instance()
            logger.operation_recorded.connect(self._on_operation)
            from core.common.settings import settings
            settings.settings_changed.connect(self._on_settings_changed)
            self._connected = True
        except Exception as e:
            print(f"[IntentAgent] start 失败: {e}")

    def _on_operation(self, record: dict):
        """新操作到达时触发异步 LLM 分析（内部节流）"""
        if not self.is_enabled():
            return
        if self._is_agent_busy():
            return
        now = time.time()
        if now - self._last_suggestion_time < self._suggestion_cooldown:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._last_suggestion_time = now
        self.analyze_async()

    def _is_agent_busy(self) -> bool:
        """对话 Agent 是否正在执行任务"""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            return False
        for widget in app.topLevelWidgets():
            if hasattr(widget, '_ai_agent'):
                agent = getattr(widget, '_ai_agent', None)
                is_busy = getattr(agent, 'is_busy', None)
                if callable(is_busy):
                    return bool(is_busy())
                return False
        return False

    def analyze_async(self):
        """异步触发 LLM 分析（后台线程，不阻塞 UI）"""
        llm = self._get_llm()
        if llm is None:
            self.analysis_updated.emit({
                "intent": "未配置",
                "intent_description": "LLM 未初始化，请先打开 Agent 对话栏配置 API Key",
                "recent_actions": [],
                "suggestions": [],
            })
            return

        prompt = self._build_prompt()
        if prompt is None:
            return

        self._worker = IntentAgentWorker(llm, prompt, self)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def analyze(self, force: bool = False) -> dict:
        """同步分析（兼容旧接口，返回上次分析结果或触发新分析）

        注意：LLM 调用是异步的，此方法仅返回缓存结果或触发异步分析。
        如需立即获取结果，请连接 analysis_updated 信号。
        """
        if force and self.has_llm():
            self.analyze_async()
        return self._last_analysis or {
            "intent": "等待分析",
            "intent_description": "操作后将自动分析意图",
            "recent_actions": [],
            "suggestions": [],
        }

    def _on_analysis_finished(self, result: dict):
        """LLM 分析完成（关闭状态忽略过期结果，避免覆盖"已关闭"UI）"""
        if not self.is_enabled():
            return
        self._last_analysis = result
        self.analysis_updated.emit(result)

    def _on_analysis_error(self, error: str):
        """LLM 分析失败（关闭状态忽略过期错误）"""
        if not self.is_enabled():
            return
        print(f"[IntentAgent] LLM 分析失败: {error}")
        result = {
            "intent": "分析失败",
            "intent_description": f"LLM 调用出错: {error}",
            "recent_actions": [],
            "suggestions": [],
        }
        self._last_analysis = result
        self.analysis_updated.emit(result)

    def _build_prompt(self) -> Optional[str]:
        """构建 LLM 分析提示词（独立上下文，不继承主 agent 历史）"""
        try:
            from core.agent.operation_logger import OperationLogger
            logger = OperationLogger.instance()
        except Exception:
            return None

        history = logger.get_history(count=20)
        if not history:
            self.analysis_updated.emit({
                "intent": "等待操作",
                "intent_description": "暂无操作记录，操作后将自动分析",
                "recent_actions": [],
                "suggestions": [],
            })
            return None

        # 精简操作历史：action + params + 状态变化
        ops_lines = []
        for h in history[-20:]:
            action = h.get("action", "")
            params = h.get("params", {})
            after = h.get("after_state", {})
            state_str = ""
            if after:
                state_parts = []
                for k in ("interface", "image_index", "total_images", "annotation_count"):
                    if k in after:
                        state_parts.append(f"{k}={after[k]}")
                if state_parts:
                    state_str = f" [{', '.join(state_parts)}]"
            params_str = json.dumps(params, ensure_ascii=False) if params else ""
            ops_lines.append(f"  - {action}({params_str}){state_str}")

        ops_text = "\n".join(ops_lines)
        available_actions = self._get_available_actions()

        prompt = f"""你是一个图像标注平台的意图分析助手。根据用户的最近操作历史和程序状态，分析用户当前的意图，并给出可执行建议。

## 最近操作历史（共 {len(history)} 条）
{ops_text}

## 可执行的 action 列表
{available_actions}

## 输出要求
请以 JSON 格式返回分析结果，不要包含其他文字。格式如下：
```json
{{
  "intent": "意图标签（如：手动标注中/自动标注中/浏览中/保存中/导出中/筛选中等）",
  "intent_description": "对用户当前意图的简短描述（一句话）",
  "suggestions": [
    {{
      "title": "建议标题（简短）",
      "message": "建议的详细说明",
      "action": "建议执行的 action 名称（必须从上面的可用 action 列表中选取，如无合适则留空字符串）",
      "params": {{}}
    }}
  ]
}}
```

## 分析要点
1. 根据操作序列判断用户正在做什么（标注、浏览、导出等）
2. 如果发现用户可能需要帮助（如长时间未保存、连续浏览未标注、频繁出错等），给出 1-3 条建议
3. 建议的 action 必须是上面列表中存在的，params 留空 {{}} 即可
4. 如果用户操作正常无需建议，suggestions 返回空数组 []
5. 只返回 JSON，不要有任何其他文字"""

        return prompt

    @staticmethod
    def _get_available_actions() -> str:
        """获取可用 action 列表（精简版，供 LLM 参考）"""
        try:
            from core.common.action_registry import ActionRegistry
            registry = ActionRegistry.instance()
            lines = []
            for meta in registry.list_actions():
                if meta.scope == "agent":
                    lines.append(f"  - {meta.name}: {meta.description.split(chr(10))[0][:60]}")
            return "\n".join(lines[-50:])
        except Exception:
            return "  (无法获取 action 列表)"
