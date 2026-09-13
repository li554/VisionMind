"""
Web Agent 侧栏桥 — DSH 视觉复刻前端 + QWebChannel 桥接 corecoder Agent

架构:
- WebAgentSidebar(QFrame): QWebEngineView 加载本地 webui/index.html,
  与经典 AgentDialog 同构 API(main_window 无差别调用)。
- _BridgeObject(QObject): 注册进 QWebChannel。
  JS→Python: sendMessage / stopClicked / approvalRespond / requestInit
  Python→JS: userAdded / assistantToken / reasoning* / toolAdded /
             toolUpdated / execStatus / approvalRequested ... 信号
- 对话/思考折叠/工具折叠/markdown/审批卡全部由前端实现;
  设计 token 逐条取自 deepseek-harness web 前端源码(见 .dsh_design/)。
"""

import html
import json
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Qt, Signal, Slot, QTimer
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import QFrame, QHBoxLayout

from pathlib import Path

_WEBUI_DIR = Path(__file__).parent


def web_sidebar_available() -> bool:
    try:
        import PySide6.QtWebEngineWidgets  # noqa: F401
        import PySide6.QtWebChannel  # noqa: F401
        return (_WEBUI_DIR / "index.html").exists()
    except Exception:
        return False


def _user_html(content) -> str:
    """用户消息 → 安全 HTML(纯文本气泡;多模态图片渲染缩略图)。"""
    parts: list[str] = []
    images: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                if url:
                    images.append(url)
    text = html.escape("\n".join(x for x in parts if x))
    text = text.replace("\n", "<br/>")
    imgs_html = "".join(
        f'<img class="msg-img" src="{html.escape(u)}"/>' for u in images
    )
    out = text
    if imgs_html:
        out += f'<div class="msg-imgs">{imgs_html}</div>'
    elif images:
        out += '<span class="ref-chip">🖼️ 图片 ×%d</span>' % len(images)
    return out


def _tool_payload(input_data: str = "", output_data: str = "",
                  steps=None, duration: str = "", status: str = "running") -> str:
    return json.dumps(
        {
            "input": input_data or "",
            "output": output_data or "",
            "steps": [
                {"label": s.get("label", ""), "status": s.get("status", "pending")}
                for s in (steps or [])
            ],
            "duration": duration or "",
            "status": status or "running",
        },
        ensure_ascii=False,
    )


class _BridgeObject(QObject):
    """QWebChannel 桥对象:信号推事件给前端,Slot 收前端动作。"""

    userAdded = Signal(str)
    systemAdded = Signal(str)
    assistantStarted = Signal()
    assistantToken = Signal(str)
    assistantFinished = Signal()
    reasoningStarted = Signal()
    reasoningToken = Signal(str)
    reasoningFinished = Signal()
    toolAdded = Signal(str, str, str, str)  # id, name, summary, payloadJson
    toolUpdated = Signal(str, str, str)     # id, status, output
    toolRemoved = Signal(str)               # id
    execStatus = Signal(str)                # 空串=隐藏
    cleared = Signal()
    approvalRequested = Signal(str, str, str)    # id, tool, reason
    askUserRequested = Signal(str, str)     # id, questionsJson
    referenceAdded = Signal(str)            # 「添加到对话」引用文本
    planBannerChanged = Signal(bool)        # 计划模式横幅开关
    planApprovalRequested = Signal()        # 计划审批卡(执行/继续修改)
    contextStats = Signal(str)              # 上下文用量摘要 JSON
    turnFinished = Signal()                 # 轮次结束(清扫 running 行)
    historyCleared = Signal(str)            # 历史会话加载/新建会话(清空对话区) mode: load/new
    statePayload = Signal(str)              # requestInit 应答

    @Slot(str)
    def sendMessage(self, text: str):
        text = (text or "").strip()
        if text and self._sidebar is not None:
            # 本地回显用户消息(与经典侧栏 _on_input_send 行为一致,
            # main_window 不负责补用户气泡)
            self._sidebar.add_message("user", text)
            self._sidebar.message_sent.emit(text)

    @Slot(str)
    def sendMessagePayload(self, payload_json: str):
        """带附件的发送:{text, images:[dataURL...]} → 多模态 content list。"""
        if self._sidebar is None:
            return
        try:
            payload = json.loads(payload_json or "{}")
        except Exception:
            payload = {}
        text = (payload.get("text") or "").strip()
        images = [u for u in (payload.get("images") or [])
                  if isinstance(u, str) and u.startswith("data:image/")]
        if not text and not images:
            return
        content = text
        try:
            from core.agent.multimodal import ReferenceResolver
            content = ReferenceResolver().resolve_message(text)
        except Exception:
            content = text
        if images:
            if isinstance(content, str):
                content = [{"type": "text", "text": content}] if content else []
            for url in images:
                content.append({"type": "image_url", "image_url": {"url": url}})
        self._sidebar.add_message("user", content)
        self._sidebar.message_sent.emit(content)

    @Slot(result=str)
    def getModels(self) -> str:
        """模型菜单数据:[{pid, name, models, active, active_model}]"""
        try:
            from core.common.ai_providers import get_active_provider, grouped_models
            active = get_active_provider()
            data = [
                {
                    "pid": pid,
                    "name": pname,
                    "models": models,
                    "active": pid == active.get("id", ""),
                    "active_model": active.get("default_model", ""),
                }
                for pid, pname, models in grouped_models()
            ]
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return "[]"

    @Slot(str, str)
    def selectModel(self, pid: str, model: str):
        if self._sidebar is not None:
            self._sidebar.model_selected.emit(pid, model)

    @Slot()
    def stopClicked(self):
        if self._sidebar is not None:
            self._sidebar.stop_requested.emit()

    @Slot(str, bool)
    def approvalRespond(self, req_id: str, ok: bool):
        if self._sidebar is not None:
            self._sidebar._resolve_approval(req_id, ok)

    @Slot(str, str)
    def answerQuestion(self, req_id: str, answers_json: str):
        if self._sidebar is not None:
            self._sidebar._resolve_ask_user(req_id, answers_json)

    @Slot(result=str)
    def getSessions(self) -> str:
        """历史会话列表(最新在前,最多 20 条)"""
        try:
            from corecoder.session import list_sessions
            out = []
            for s in list_sessions():
                saved = str(s.get("saved_at", ""))
                out.append({
                    "id": s.get("id", ""),
                    "time": saved[5:16] if len(saved) >= 16 else saved,
                    "preview": str(s.get("preview", ""))[:80],
                    "model": s.get("model", ""),
                })
            return json.dumps(out, ensure_ascii=False)
        except Exception:
            return "[]"

    @Slot(str)
    def openSession(self, session_id: str):
        """加载历史会话:清空对话区并回放消息,同时写入 Agent 历史"""
        if self._sidebar is None:
            return
        try:
            from corecoder.session import load_session
            result = load_session(session_id)
            if result is None:
                return
            messages, model = result
            try:
                from core.agent.multimodal import replace_image_urls_with_captions
                messages = replace_image_urls_with_captions(messages)
            except Exception:
                pass
        except Exception as e:
            print(f"[WebAgentSidebar] 会话加载失败: {e}")
            return
        self._sidebar.history_loaded.emit(messages)
        self._sidebar._replay_history(messages)

    @Slot()
    def newSession(self):
        """新建会话:清空对话区并重置 Agent 历史"""
        if self._sidebar is not None:
            self._sidebar.history_loaded.emit([])
            self._sidebar.mark_new_session()

    @Slot(str)
    def deleteSession(self, session_id: str):
        try:
            from core.corecoder.session import SESSIONS_DIR
            path = SESSIONS_DIR / f"{session_id}.json"
            if path.exists():
                path.unlink()
        except Exception as e:
            print(f"[WebAgentSidebar] 会话删除失败: {e}")

    @Slot(str, result=str)
    def searchFiles(self, prefix: str) -> str:
        """@ 引用的文件名搜索(与经典侧栏 _search_image_paths 同源)。"""
        if self._sidebar is None:
            return "[]"
        try:
            from core.common.action_registry import ActionRegistry
            result = ActionRegistry.instance().call(
                "annotation.manual.search_images", keyword=prefix
            )
            files = (result.get("files") or []) if isinstance(result, dict) else []
            out = []
            for entry in files:
                p = entry[0] if isinstance(entry, (list, tuple)) else entry
                if isinstance(p, str) and p.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    out.append(p)
                if len(out) >= 20:
                    break
            return json.dumps(out, ensure_ascii=False)
        except Exception:
            return "[]"

    @Slot(bool)
    def planApprove(self, execute: bool):
        cb = self._sidebar._pending_plan_cb if self._sidebar else None
        if self._sidebar:
            self._sidebar._pending_plan_cb = None
        if cb is not None:
            try:
                cb(execute)
            except Exception as e:
                print(f"[WebAgentSidebar] 计划审批回调失败: {e}")

    @Slot()
    def requestInit(self):
        if self._sidebar is not None:
            self.statePayload.emit(
                json.dumps(
                    {
                        "model": self._sidebar.model_display,
                        "theme": "light" if self._sidebar.is_light else "dark",
                    },
                    ensure_ascii=False,
                )
            )

    @Slot()
    def frontendReady(self):
        """JS 已连完全部信号:置就绪并重放就绪前排队的事件。"""
        if self._sidebar is not None:
            self._sidebar._on_frontend_ready()

    def __init__(self, sidebar=None):
        super().__init__()
        self._sidebar = sidebar


class WebAgentSidebar(QFrame):
    """DSH 视觉复刻的 Web 对话侧栏(与经典 AgentDialog 同构 API)"""

    # 与经典侧栏一致的对外信号
    message_sent = Signal(object)
    stop_requested = Signal()
    model_selected = Signal(str, str)
    history_loaded = Signal(object)
    debug_requested = Signal()
    suggestion_clicked = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DshWebSidebar")
        self.setMinimumWidth(300)
        self.setMaximumWidth(720)

        self._tool_names: dict[str, str] = {}   # id -> tool_name
        self._tool_seq = 0
        self._pending_approvals: dict[str, callable] = {}
        self._approval_seq = 0
        self._pending_questions: dict[str, callable] = {}
        self._question_seq = 0
        self._pending_plan_cb = None
        self._assistant_open = False
        self._event_bus = None
        # 经典侧栏兼容属性(main_window._on_ai_tool_called 会探测)
        self._current_assistant_bubble = None
        # JS 就绪前缓存事件(页面加载期间到达的消息流不丢失)
        self._js_ready = False
        self._event_queue: list[tuple[Signal, tuple]] = []

        try:
            from core.common.theme_manager import theme_manager
            self.is_light = theme_manager.is_light_theme()
        except Exception:
            self.is_light = False
        self.model_display = ""

        self._bridge = _BridgeObject(self)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)

        from PySide6.QtWebEngineWidgets import QWebEngineView

        self._view = QWebEngineView(self)
        self._view.setPage  # noqa: B018 — 确保页面已创建
        self._view.page().setWebChannel(self._channel)
        index = (_WEBUI_DIR / "index.html").resolve()
        self._view.load(QUrl.fromLocalFile(str(index)))
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        # 左缘拖拽把手(复用经典侧栏组件): [把手][webview]
        # 拖拽期间不实时重排——半透明窗口重排空隙会透出桌面,WebEngine
        # 重合成慢会留残影;改为显示指示线预览宽度,松手一次性应用
        from core.agent.agent_dialog import ResizeHandle

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._handle = ResizeHandle(self)
        layout.addWidget(self._handle)
        layout.addWidget(self._view, 1)
        self._drag_line = QFrame(self)
        self._drag_line.setStyleSheet("background-color: #2563eb; border: none;")
        self._drag_line.setFixedWidth(2)
        self._drag_line.hide()
        self._drag_width = None
        self._handle.width_changed.connect(self._on_handle_width)
        self._handle.resize_finished.connect(self._on_resize_finished)

        try:
            from core.common.theme_manager import theme_manager
            theme_manager.theme_changed.connect(self._on_theme_changed)
        except Exception:
            pass

    # ---------------- 事件分发(就绪前排队) ----------------

    def _emit(self, signal, *args):
        if self._js_ready:
            signal.emit(*args)
        else:
            self._event_queue.append((signal, args))

    def _on_frontend_ready(self):
        self._js_ready = True
        queue, self._event_queue = self._event_queue, []
        for signal, args in queue:
            signal.emit(*args)

    def _state_json(self) -> str:
        return json.dumps(
            {
                "model": self.model_display,
                "theme": "light" if self.is_light else "dark",
            },
            ensure_ascii=False,
        )

    # ---------------- 主题/模型 ----------------

    def _on_theme_changed(self, *_):
        try:
            from core.common.theme_manager import theme_manager
            self.is_light = theme_manager.is_light_theme()
        except Exception:
            self.is_light = False
        self._emit(self._bridge.statePayload, self._state_json())

    def set_model(self, model_name: str):
        if model_name:
            self.model_display = model_name
            self._emit(self._bridge.statePayload, self._state_json())

    def get_model(self) -> str:
        return self.model_display

    def set_event_bus(self, bus):
        """订阅「添加到对话」引用事件(与经典侧栏同语义,幂等)。"""
        if bus is None or getattr(self, "_event_bus", None) is bus:
            return
        self._event_bus = bus
        from core.event_bus import StateRouter, StateType

        self._ref_router = StateRouter(bus).register(
            StateType.REFERENCE_ADDED, self._on_reference_added
        )

    def _on_reference_added(self, ref_text):
        if ref_text:
            self._emit(self._bridge.referenceAdded, str(ref_text))

    # ---------------- 消息流 API(与经典侧栏同构) ----------------

    def add_message(self, role: str, content):
        if role == "user":
            self._emit(self._bridge.userAdded, _user_html(content))
        elif role == "system":
            text = content if isinstance(content, str) else str(content)
            self._emit(self._bridge.systemAdded, html.escape(text))
        else:
            text = content if isinstance(content, str) else ""
            self._emit(self._bridge.assistantStarted)
            if text.strip():
                self._emit(self._bridge.assistantToken, text)
            self._emit(self._bridge.assistantFinished)

    def start_assistant_message(self, initial_text: str = ""):
        self._assistant_open = True
        self._emit(self._bridge.assistantStarted)
        if initial_text.strip():
            self._emit(self._bridge.assistantToken, initial_text)

    def append_to_assistant(self, text: str):
        if text:
            self._emit(self._bridge.assistantToken, text)

    def finalize_assistant(self):
        self._assistant_open = False
        self._emit(self._bridge.assistantFinished)

    def add_tool_call(self, tool_name: str, status: str, duration: str = "",
                      input_data: str = "", output_data: str = "", steps=None):
        self._tool_seq += 1
        tool_id = str(self._tool_seq)
        self._tool_names[tool_id] = tool_name
        summary = (input_data or "").strip().splitlines()
        summary = summary[0][:120] if summary else tool_name
        self._emit(
            self._bridge.toolAdded, tool_id, tool_name, summary,
            _tool_payload(input_data, output_data, steps, duration, status),
        )
        return tool_id

    def update_tool_status(self, tool_name_or_block, status: str, output: str = ""):
        tool_id = ""
        if isinstance(tool_name_or_block, str) and tool_name_or_block in self._tool_names:
            tool_id = tool_name_or_block
        else:
            for tid, name in self._tool_names.items():
                if name == tool_name_or_block:
                    tool_id = tid
                    break
        if tool_id:
            self._emit(self._bridge.toolUpdated, tool_id, status, output)

    def show_exec_status(self, text: str = ""):
        self._emit(self._bridge.execStatus, text or "正在执行...")

    def update_exec_status(self, text: str):
        self._emit(self._bridge.execStatus, text or "正在执行...")

    def hide_exec_status(self):
        self._emit(self._bridge.execStatus, "")

    def show_thinking(self, thinking: bool):
        pass  # 思考以 reasoning 流呈现;占位保持 API 兼容

    def clear_messages(self):
        self._tool_names.clear()
        self._emit(self._bridge.cleared)

    def add_session(self, *args, **kwargs):
        return 0

    def new_session(self):
        self.clear_messages()

    # ---------------- 工具解锁审批 ----------------

    def request_bash_approval(self, reason: str, respond) -> None:
        self._request_tool_approval("bash", reason, respond)

    def request_tool_approval(self, tool: str, reason: str, respond) -> None:
        self._approval_seq += 1
        req_id = f"approval-{self._approval_seq}"
        self._pending_approvals[req_id] = respond
        self._emit(self._bridge.approvalRequested, req_id, tool, reason)

    def _resolve_approval(self, req_id: str, ok: bool):
        respond = self._pending_approvals.pop(req_id, None)
        if respond is not None:
            try:
                respond(ok)
            except Exception as e:
                print(f"[WebAgentSidebar] 审批回调失败: {e}")

    # ---------------- ask_user 提问 ----------------

    def request_ask_user(self, qlist, respond) -> None:
        self._question_seq += 1
        req_id = f"ask-{self._question_seq}"
        self._pending_questions[req_id] = respond
        self._emit(
            self._bridge.askUserRequested,
            req_id,
            json.dumps(qlist or [], ensure_ascii=False),
        )

    def _resolve_ask_user(self, req_id: str, answers_json: str):
        respond = self._pending_questions.pop(req_id, None)
        if respond is None:
            return
        try:
            data = json.loads(answers_json)
            answers = data.get("answers") if isinstance(data, dict) else data
            respond(answers or [])
        except Exception as e:
            print(f"[WebAgentSidebar] 提问回答解析失败: {e}")
            try:
                respond(None)
            except Exception:
                pass

    # ---------------- 计划模式 ----------------

    def set_plan_banner(self, on: bool):
        """横幅已按需求移除;保留接口兼容 main_window 调用。"""

    def show_plan_approval(self, on_decide) -> None:
        self._pending_plan_cb = on_decide
        self._emit(self._bridge.planApprovalRequested)

    def _replay_history(self, messages):
        """把历史消息回放为前端渲染事件(assistant 的 tool_calls 与
        tool 结果消息按 id 配对成工具行)。"""
        self._emit(self._bridge.historyCleared, "load")
        tool_results = {}
        for m in messages:
            if m.get("role") == "tool":
                tool_results[m.get("tool_call_id", "")] = m.get("content", "")

        n = 0
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                self._emit(self._bridge.userAdded, _user_html(content))
            elif role == "system":
                self._emit(self._bridge.systemAdded,
                           html.escape(str(content or "")))
            elif role == "assistant":
                if isinstance(content, str) and content.strip():
                    self._emit(self._bridge.assistantStarted)
                    self._emit(self._bridge.assistantToken, content)
                    self._emit(self._bridge.assistantFinished)
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    n += 1
                    args = fn.get("arguments", "")
                    summary = args.splitlines()[0][:120] if args else fn.get("name", "")
                    res = tool_results.get(tc.get("id", ""), "")
                    self._emit(self._bridge.toolAdded,
                               f"hist-{n}", fn.get("name", ""), summary,
                               _tool_payload(args, res, None, "", "success"))
            # tool 角色已按 id 配对进工具行,不再单独渲染

    # ---------------- 上下文用量 ----------------

    def push_context_stats(self, stats_json: str):
        self._emit(self._bridge.contextStats, stats_json)

    def mark_new_session(self):
        self._emit(self._bridge.historyCleared, "new")

    def mark_turn_finished(self):
        self._emit(self._bridge.turnFinished)

    # ---------------- 拖拽调宽(指示线预览,松手应用) ----------------

    def _on_handle_width(self, w: int):
        # 记录拖拽目标宽度(handle 松手时上报的是侧栏当前宽度,已过时)
        self._drag_width = w
        # 指示线挂在主窗口上(侧栏子控件会被裁剪,向外拖时显示不出):
        # 侧栏右缘固定于窗口右缘,线的窗口坐标 x = 窗口宽 - 目标宽
        win = self.window()
        if self._drag_line.parent() is not win:
            self._drag_line.setParent(win)
        self._drag_line.setGeometry(
            max(0, win.width() - w - 1), 0, 2, win.height())
        self._drag_line.show()
        self._drag_line.raise_()

    def _on_resize_finished(self, width: int):
        self._drag_line.hide()
        self.setFixedWidth(self._drag_width or width)

    # ---------------- 生命周期 ----------------

    def shutdown(self):
        pass  # 无子进程资源;保留接口对称

    def deleteLater(self):
        if self._channel is not None:
            self._channel.deleteLater()
        super().deleteLater()
