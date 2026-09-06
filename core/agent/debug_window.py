"""
系统提示词调试窗口 — 实时查看当前 Agent 的系统提示词

显示：
- 当前完整系统提示词（只读）+ 活跃工具集
- 当前已注入的知识段（含 LRU 信息）
- 所有注册的知识段
- Token 预算摘要
- 工具分类管理
- 刷新按钮重新加载
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QWidget, QTabWidget
)
from qfluentwidgets import (
    PushButton, PrimaryPushButton, StrongBodyLabel,
    CaptionLabel, InfoBar
)


class SystemPromptDebugDialog(QDialog):
    """系统提示词调试对话框"""

    _EDIT_STYLE = """
        QPlainTextEdit {
            font-family: "Consolas", "Courier New", monospace;
            font-size: 12px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 8px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统提示词调试")
        self.setMinimumSize(700, 500)
        self.setObjectName("SystemPromptDebugDialog")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = StrongBodyLabel("系统提示词调试")
        layout.addWidget(title)

        desc = CaptionLabel("当前发送给 LLM 的完整系统提示词（含工具列表和已注入知识段）。用于调试 Agent 行为。")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Tab 切换
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)

        # 系统提示词 Tab
        self._prompt_edit = self._make_edit()
        self._tabs.addTab(self._prompt_edit, "系统提示词")

        # 已注入知识段 Tab
        self._injected_edit = self._make_edit()
        self._tabs.addTab(self._injected_edit, "已注入知识段")

        # 所有注册知识段 Tab
        self._all_sections_edit = self._make_edit()
        self._tabs.addTab(self._all_sections_edit, "所有知识段")

        # Token 预算 Tab
        self._token_budget_edit = self._make_edit()
        self._tabs.addTab(self._token_budget_edit, "Token 预算")

        # 工具管理 Tab
        self._tool_manager_edit = self._make_edit()
        self._tabs.addTab(self._tool_manager_edit, "工具管理")

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        refresh_btn = PrimaryPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        btn_layout.addWidget(refresh_btn)

        close_btn = PushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        # 初始加载
        self.refresh()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _make_edit(self) -> QPlainTextEdit:
        """创建统一样式的只读 QPlainTextEdit"""
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setStyleSheet(self._EDIT_STYLE)
        return edit

    def _find_agent(self):
        """查找当前 VisionMindAgent 实例"""
        from core.agent.visionmind_agent import VisionMindAgent  # noqa: F401
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app:
            return None
        for widget in app.topLevelWidgets():
            if hasattr(widget, '_ai_agent'):
                return widget._ai_agent
        return None

    # ------------------------------------------------------------------
    # 刷新入口
    # ------------------------------------------------------------------

    def refresh(self):
        """重新加载所有 Tab 内容"""
        try:
            agent = self._find_agent()
            self._refresh_system_prompt(agent)
            self._refresh_injected(agent)
            self._refresh_all_sections(agent)
            self._refresh_token_budget(agent)
            self._refresh_tool_manager(agent)
        except Exception as e:
            error_text = f"刷新失败: {type(e).__name__}: {e}"
            self._prompt_edit.setPlainText(error_text)
            InfoBar.error("刷新失败", str(e), parent=self)

    # ------------------------------------------------------------------
    # 系统提示词 Tab（含活跃工具集）
    # ------------------------------------------------------------------

    def _refresh_system_prompt(self, agent):
        if agent and agent._agent:
            system = agent._agent._system or "(无)"
            self._prompt_edit.setPlainText(system + "\n\n" + self._build_active_tools_block(agent))
        else:
            self._prompt_edit.setPlainText("Agent 未初始化")

    def _build_active_tools_block(self, agent) -> str:
        """构建活跃工具集信息块"""
        tm = getattr(agent, '_tool_manager', None)
        if tm is None:
            return "(工具管理器未就绪)"

        try:
            current_interface = tm.get_current_interface()
            active_domains = tm.get_active_domains(
                "", current_interface, getattr(agent, '_recent_tool_categories', [])
            )
            activated_tools = tm.get_activated_tool_names()

            # 核心工具名集合
            try:
                from core.agent.dynamic_tool_manager import CORE_TOOL_NAMES
                core_tools = sorted(CORE_TOOL_NAMES)
            except Exception:
                core_tools = []

            lines = ["--- 活跃工具集 ---"]
            lines.append(f"核心工具: {', '.join(core_tools) if core_tools else '(无)'}")
            lines.append(f"当前界面: {current_interface or '(未检测)'}")
            lines.append(f"活跃分类: {', '.join(active_domains) if active_domains else '(无)'}")
            lines.append(f"显式激活工具: {', '.join(activated_tools) if activated_tools else '(无)'}")

            recent = getattr(agent, '_recent_tool_categories', [])
            lines.append(f"最近使用分类: {', '.join(recent) if recent else '(无)'}")
            return "\n".join(lines)
        except Exception as e:
            return f"(活跃工具集获取失败: {type(e).__name__}: {e})"

    # ------------------------------------------------------------------
    # 已注入知识段 Tab（含 LRU 信息）
    # ------------------------------------------------------------------

    def _refresh_injected(self, agent):
        from core.agent.prompt_manager import PromptManager
        pm = PromptManager.instance()
        active = pm.get_active_sections()
        if not active:
            self._injected_edit.setPlainText("当前未注入任何知识段")
            return

        lru_order = pm.get_lru_order()  # 从最旧到最新
        total_lru = len(lru_order)

        lines = [f"已注入 {len(active)} 个知识段：\n"]
        for s in active:
            lines.append(f"## {s.section_id} [{s.category}]")
            lines.append(f"标题: {s.title}")
            lines.append(f"场景: {', '.join(s.scenes)}" if s.scenes else "场景: (无)")
            lines.append(f"优先级: {s.priority}")

            # LRU 位置
            if s.section_id in lru_order:
                idx = lru_order.index(s.section_id)
                position_str = f"{idx + 1}/{total_lru}"
                if total_lru > 0:
                    if idx == total_lru - 1:
                        position_str += " (最新使用)"
                    elif idx == 0:
                        position_str += " (最久未使用)"
                lines.append(f"LRU 位置: {position_str}")
            else:
                lines.append("LRU 位置: (未在队列中)")

            token_count = pm.get_section_token_count(s.section_id)
            lines.append(f"Token 占用: {token_count}")
            lines.append("")
            lines.append(s.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        self._injected_edit.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # 所有知识段 Tab
    # ------------------------------------------------------------------

    def _refresh_all_sections(self, agent):
        from core.agent.prompt_manager import PromptManager
        pm = PromptManager.instance()
        all_sections = pm.list_sections()
        if not all_sections:
            self._all_sections_edit.setPlainText("未注册任何知识段")
            return

        lines = [f"共注册 {len(all_sections)} 个知识段：\n"]
        for s in all_sections:
            status = "✓" if s.is_injected else " "
            scenes_str = f" [{', '.join(s.scenes)}]" if s.scenes else ""
            lines.append(f"[{status}] {s.section_id} [{s.category}]{scenes_str} - {s.title}")
        self._all_sections_edit.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Token 预算 Tab
    # ------------------------------------------------------------------

    def _refresh_token_budget(self, agent):
        if agent is None:
            self._token_budget_edit.setPlainText("Agent 未初始化")
            return

        tb = getattr(agent, '_token_budget', None)
        if tb is None:
            self._token_budget_edit.setPlainText("Token 预算管理器未就绪")
            return

        try:
            summary = tb.get_summary()
            system_limit = tb.get_system_prompt_limit()
            max_context = getattr(tb, '_max_context', 32000)

            conversation_turn = getattr(agent, '_conversation_turn', 0)
            compression_history = getattr(agent, '_compression_history', [])

            over_limit = summary.get("system_prompt_over_limit", False)
            over_str = "是" if over_limit else "否"

            level = summary.get("level", "unknown")
            percentage = summary.get("percentage", 0.0)

            lines = [
                "== Token 预算摘要 ==",
                f"级别: {level} ({percentage:.2f}%)",
                f"系统提示词: {summary.get('system_prompt', 0)} tokens "
                f"(基础: {summary.get('base_prompt', 0)}, 知识段: {summary.get('sections', 0)})",
                f"工具 schema: {summary.get('tools', 0)} tokens",
                f"消息历史: {summary.get('messages', 0)} tokens",
                f"总计: {summary.get('total', 0)} tokens ({percentage:.2f}% of {max_context})",
                "",
                f"系统提示词上限: {system_limit} tokens",
                f"是否超限: {over_str}",
                "",
                f"对话轮次: {conversation_turn}",
                f"压缩历史: {len(compression_history)} 次",
            ]

            # 压缩历史详情（若有）
            if compression_history:
                lines.append("")
                lines.append("== 压缩历史详情 ==")
                for rec in compression_history:
                    if isinstance(rec, dict):
                        turn = rec.get("turn", "?")
                        lines.append(f"  轮次 {turn}: {rec}")
                    else:
                        lines.append(f"  {rec}")

            self._token_budget_edit.setPlainText("\n".join(lines))
        except Exception as e:
            self._token_budget_edit.setPlainText(
                f"Token 预算摘要获取失败: {type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------------
    # 工具管理 Tab
    # ------------------------------------------------------------------

    def _refresh_tool_manager(self, agent):
        from core.common.action_registry import ActionRegistry
        from core.agent.dynamic_tool_manager import ALL_CATEGORIES  # noqa: F401

        try:
            registry = ActionRegistry.instance()
            all_metas = registry.list_actions()
        except Exception as e:
            self._tool_manager_edit.setPlainText(
                f"ActionRegistry 加载失败: {type(e).__name__}: {e}"
            )
            return

        # 按 category 分组（仅 scope=agent）
        categories = {}
        for meta in all_metas:
            if getattr(meta, 'scope', None) != "agent":
                continue
            cat = meta.category or "其他"
            categories.setdefault(cat, []).append(meta)

        # agent / tool_manager 上下文
        tm = getattr(agent, '_tool_manager', None) if agent else None
        if tm is not None:
            try:
                current_interface = tm.get_current_interface()
            except Exception:
                current_interface = ""
            try:
                active_domains = tm.get_active_domains(
                    "", current_interface, getattr(agent, '_recent_tool_categories', [])
                )
            except Exception:
                active_domains = []
            try:
                activated_tools = tm.get_activated_tool_names()
            except Exception:
                activated_tools = []
        else:
            current_interface = ""
            active_domains = []
            activated_tools = []

        lines = ["== 工具分类管理 =="]
        lines.append(f"当前界面: {current_interface or '(未检测)'}")
        lines.append(f"活跃分类: {', '.join(active_domains) if active_domains else '(无)'}")
        lines.append(f"激活工具(LRU): {', '.join(activated_tools) if activated_tools else '(无)'}")
        lines.append("")

        # 激活工具集合（用于判断 [激活] 标记）
        activated_set = set(activated_tools)

        lines.append("== 分类详情 ==")
        # 排序：活跃分类在前，非活跃在后；按 category 名字母序
        active_set = set(active_domains)
        sorted_cats = sorted(
            categories.keys(),
            key=lambda c: (0 if c in active_set else 1, c)
        )

        for cat in sorted_cats:
            metas = categories[cat]
            is_active = cat in active_set
            tag = "[活跃]" if is_active else "[非活跃]"
            lines.append(f"{tag} {cat} ({len(metas)} 个工具)")
            for meta in metas:
                # 激活工具追加 LRU 位置
                if meta.name in activated_set and activated_tools:
                    idx = activated_tools.index(meta.name)
                    total = len(activated_tools)
                    pos_str = f" (LRU 位置: {idx + 1}/{total}"
                    if idx == total - 1:
                        pos_str += ", 最新使用"
                    pos_str += ")"
                    lines.append(f"  - {meta.name}{pos_str} - {meta.description}")
                else:
                    lines.append(f"  - {meta.name} - {meta.description}")
            lines.append("")

        self._tool_manager_edit.setPlainText("\n".join(lines))
