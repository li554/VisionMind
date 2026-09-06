"""
提示词管理工具 — 让 Agent 自主发现和注入知识段

提供三个工具：
- list_prompt_sections: 搜索可用的知识段
- inject_prompt: 注入一个知识段到当前对话
- get_active_prompts: 查看当前已注入了哪些知识段
"""

from core.corecoder.tools import Tool
from core.agent.prompt_manager import PromptManager


class ListPromptSectionsTool(Tool):
    read_only = True
    """搜索可用的知识段"""

    name = "list_prompt_sections"
    description = (
        "搜索所有插件注册的知识段。知识段包含功能说明、操作指南等提示词信息。"
        "当你需要了解某个功能的详细信息时，先调用此工具搜索可用的知识段，"
        "然后使用 inject_prompt 注入到对话中。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "搜索关键词（可选），如 'SAM', '标注', '项目' 等。留空返回所有可用知识段。",
            },
            "category": {
                "type": "string",
                "description": "按知识类型筛选（可选）：concept(概念)、guide(操作指南)、example(代码示例)、reference(参考信息)、rule(规则约束)",
            },
            "scene": {
                "type": "string",
                "description": "按使用场景筛选（可选）：sam(SAM交互分割)、manual_draw(手动绘制)、batch(批量标注)、quality_check(质量检查)、export(数据导出)、navigation(导航)",
            },
        },
        "required": [],
    }

    def execute(self, keywords: str = "", category: str = "", scene: str = "") -> str:
        pm = PromptManager.instance()
        sections = pm.list_sections(keywords=keywords, category=category, scene=scene)
        if not sections:
            return "未找到匹配的知识段"

        lines = [f"找到 {len(sections)} 个知识段：\n"]
        for s in sections:
            status = "✓已注入" if s.is_injected else "未注入"
            scenes_str = f" | 场景: {', '.join(s.scenes)}" if s.scenes else ""
            lines.append(f"【{s.section_id}】{s.title} [{s.category}]{scenes_str}")
            lines.append(f"  说明: {s.description}")
            lines.append(f"  关键词: {', '.join(s.keywords)} | 状态: {status}")
            lines.append("")
        return "\n".join(lines)


class InjectPromptTool(Tool):
    read_only = True
    """注入一个知识段到当前对话"""

    name = "inject_prompt"
    description = (
        "将一个知识段注入到当前对话。注入后你将立即获得该知识段的完整内容，"
        "并且后续对话也会保留该知识。先用 list_prompt_sections 搜索可用的知识段。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": "知识段ID，如 'annotation.interact.sam'。从 list_prompt_sections 获取。",
            },
        },
        "required": ["section_id"],
    }

    def execute(self, section_id: str) -> str:
        pm = PromptManager.instance()
        section = pm.inject(section_id)
        if not section:
            return (
                f"错误: 未找到知识段 '{section_id}'。"
                f"请先用 list_prompt_sections 搜索可用的知识段ID。"
            )
        return (
            f"✅ 已注入知识段「{section.title}」\n\n"
            f"--- {section.title} ---\n\n"
            f"{section.content}\n\n"
            f"--- 结束 ---"
        )


class GetActivePromptsTool(Tool):
    read_only = True
    """查看当前已注入了哪些知识段"""

    name = "get_active_prompts"
    description = "查看当前对话中已注入了哪些知识段。"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> str:
        pm = PromptManager.instance()
        active = pm.get_active_sections()
        if not active:
            return "当前未注入任何知识段。你可以使用 list_prompt_sections 搜索并用 inject_prompt 注入。"
        lines = [f"当前已注入 {len(active)} 个知识段：\n"]
        for s in active:
            lines.append(f"- 【{s.section_id}】{s.title}")
        return "\n".join(lines)


class ToolSearchAgent(Tool):
    read_only = True
    """工具搜索子 Agent — 语义搜索可用的 action（LLM 语义匹配，关键词回退）

    与 IntentAgent 同属子 Agent 模式：
    - 不创建独立 LLM 实例，通过 QApplication 反查主 Agent 的 LLM 引用
    - LLM 调用复用 message_compressor._call_llm
    - 独立上下文，不继承主 Agent 对话历史
    """

    name = "search_tools"
    description = (
        "语义搜索所有可调用的工具（action）。当你需要的工具不在当前可用列表中时，"
        "用自然语言描述你想完成的操作，本工具通过 LLM 语义匹配找到最合适的工具。"
        "找到后调用 activate_tools 激活所需工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "自然语言描述你想完成的操作，如 '隐藏当前图片的所有标注'、'给所有图片批量自动标注' 等",
            },
            "category": {
                "type": "string",
                "description": "按分类筛选（可选）：标注、画布、自动标注、训练、推理、版本管理、项目、配置等",
            },
        },
        "required": ["query"],
    }

    def execute(self, query: str = "", category: str = "", keywords: str = "") -> str:
        # 兼容旧调用方式（keywords 参数）
        search_text = query or keywords
        from core.common.action_registry import ActionRegistry

        registry = ActionRegistry.instance()
        all_metas = registry.list_actions()

        # 仅暴露 scope="agent" 的 action
        matched = [m for m in all_metas if m.scope == "agent"]

        # 分类过滤
        if category:
            matched = [m for m in matched if m.category == category]

        if not matched:
            return "未找到匹配工具。"

        # 优先 LLM 语义检索（子 Agent 单次调用，不继承主 Agent 上下文）
        if search_text:
            llm_result = self._llm_semantic_search(search_text, matched, registry)
            if llm_result is not None:
                return llm_result

        # 回退：关键词匹配（LLM 不可用或查询为空时）
        return self._keyword_search(search_text, matched, registry)

    # ====== LLM 语义检索 ======

    def _llm_semantic_search(self, query: str, matched, registry) -> str | None:
        """通过 LLM 语义匹配查找工具。返回格式化结果字符串，失败返回 None。"""
        llm = self._get_llm()
        if llm is None:
            return None

        from core.agent.message_compressor import _call_llm
        catalog = self._build_tool_catalog(matched, registry)
        prompt = self._build_search_prompt(query, catalog)
        response = _call_llm(llm, prompt)
        if not response:
            return None

        tool_names = self._parse_tool_names(response)
        if not tool_names:
            return None

        matched_metas = self._match_tool_names(tool_names, matched, registry)
        if not matched_metas:
            return None

        return self._format_results(matched_metas, registry, query, semantic=True)

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

    @staticmethod
    def _get_display_name(full_name: str) -> str:
        """计算暴露给 LLM 的工具名（去掉前缀，与 dynamic_tool_manager 一致）。

        如 "annotation.manual.select_category" → "manual.select_category"。
        必须与 DynamicToolManager.get_tools_for_context 中的 display_name 逻辑保持一致，
        否则 LLM 拿到的工具名无法被 Agent 识别（unknown tool）。
        """
        return full_name.split(".", 1)[1] if "." in full_name else full_name

    def _build_tool_catalog(self, matched, registry) -> str:
        """构建工具清单文本：序号. 可调用名(参数) — 描述。

        注意：这里必须用 display_name（去掉前缀），与 LLM 实际可调用的工具名一致，
        否则 LLM 照抄完整注册名调用会报 unknown tool。
        """
        lines = []
        for i, meta in enumerate(matched, 1):
            full_name = self._find_full_name(registry, meta.name)
            display_name = self._get_display_name(full_name)
            param_parts = []
            for pname, pdef in meta.params.items():
                prop = registry._parse_param_schema(pdef)
                ptype = prop.get("type", "string")
                param_parts.append(f"{pname}: {ptype}")
            params_str = ", ".join(param_parts)
            desc = meta.description or ""
            lines.append(f"{i}. {display_name}({params_str}) — {desc}")
        return "\n".join(lines)

    def _build_search_prompt(self, query: str, catalog: str) -> str:
        """构建子 Agent 搜索提示词（独立上下文，不继承主 Agent 历史）。"""
        return (
            "你是一个工具搜索助手。用户想完成某个任务，请从以下工具列表中找出最匹配的工具。\n\n"
            f"用户需求：{query}\n\n"
            f"可用工具列表：\n{catalog}\n\n"
            "请返回最匹配的工具（最多 5 个），按相关度从高到低排序。\n"
            "只返回工具名称（即列表中括号前的名称），每行一个，不要序号、不要解释、不要其他文字。\n"
            '如果没有匹配的工具，回复"无"。'
        )

    def _parse_tool_names(self, response: str) -> list:
        """从 LLM 响应中解析工具名列表（每行一个，容忍序号/符号前缀）。"""
        names = []
        for line in response.strip().splitlines():
            line = line.strip().strip('-').strip('*').strip('•').strip()
            # 去除序号前缀（如 "1." "2."）
            if '.' in line:
                prefix, rest = line.split('.', 1)
                if prefix.strip().isdigit():
                    line = rest.strip()
            if not line or line in ("无", "none", "None", "无匹配"):
                continue
            names.append(line)
        return names

    def _match_tool_names(self, tool_names: list, matched, registry) -> list:
        """将 LLM 返回的工具名匹配到实际 meta。

        同时支持三种名称：完整注册名（annotation.manual.select_category）、
        display_name（manual.select_category）、meta.name（manual.select_category）。
        """
        name_to_meta = {}
        for meta in matched:
            full_name = self._find_full_name(registry, meta.name)
            display_name = self._get_display_name(full_name)
            name_to_meta[full_name] = meta      # 完整名
            name_to_meta[display_name] = meta    # display_name
            name_to_meta[meta.name] = meta       # meta.name（通常等同 display_name）

        result = []
        seen = set()
        for name in tool_names:
            meta = name_to_meta.get(name)
            if meta and meta.name not in seen:
                result.append(meta)
                seen.add(meta.name)
        return result

    # ====== 关键词回退 ======

    def _keyword_search(self, keywords: str, matched, registry) -> str:
        """关键词匹配（LLM 不可用时的回退策略）。"""
        if keywords:
            kw_list = [k.lower() for k in keywords.split() if k.strip()]
            if kw_list:
                filtered = []
                for m in matched:
                    name_lower = m.name.lower()
                    desc_lower = (m.description or "").lower()
                    if any(kw in name_lower or kw in desc_lower for kw in kw_list):
                        filtered.append(m)
                matched = filtered

        if not matched:
            return (
                "未找到匹配工具。建议使用更宽泛的关键词，"
                "或调用 search_tools() 不带参数查看所有工具。"
            )
        return self._format_results(matched, registry, keywords, semantic=False)

    # ====== 通用工具方法 ======

    def _format_results(self, matched_metas, registry, query: str, semantic: bool = False) -> str:
        """格式化搜索结果输出。

        输出的工具名必须是 LLM 可直接调用的 display_name（去掉前缀），
        与 DynamicToolManager 暴露给 LLM 的工具名一致。
        """
        method = "LLM 语义匹配" if semantic else "关键词匹配"
        lines = [f"找到 {len(matched_metas)} 个匹配的工具（{method}）：\n"]
        for i, meta in enumerate(matched_metas, 1):
            full_name = self._find_full_name(registry, meta.name)
            display_name = self._get_display_name(full_name)
            param_parts = []
            for pname, pdef in meta.params.items():
                prop = registry._parse_param_schema(pdef)
                ptype = prop.get("type", "string")
                param_parts.append(f"{pname}: {ptype}")
            params_str = ", ".join(param_parts)
            lines.append(f"{i}. {display_name}({params_str}) - {meta.description}")
        lines.append("")
        lines.append(
            "提示：调用 activate_tools(tool_names=[\"工具名1\", \"工具名2\"]) 激活所需工具，下一轮即可使用。"
            "激活后请用上面列出的工具名（括号前的名称）直接调用。"
        )
        return "\n".join(lines)

    @staticmethod
    def _find_full_name(registry, meta_name: str) -> str:
        """根据 meta.name 找到完整的注册名（带前缀）"""
        for full_name in registry._actions:
            if full_name == meta_name or full_name.endswith(f".{meta_name}"):
                return full_name
        return meta_name


class ActivateToolsTool(Tool):
    read_only = True
    """显式激活工具，使其在下一轮对话中可用"""

    name = "activate_tools"
    description = (
        "激活指定的工具，使其在下一轮对话中可用。"
        "当你需要使用当前不在活跃集中的工具时，先激活它。"
        "使用 search_tools 查找工具名称。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要激活的工具名称列表，如 [\"batch.annotate\", \"canvas.draw_rectangle\"]",
            },
        },
        "required": ["tool_names"],
    }

    def execute(self, tool_names: list) -> str:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if not app:
            return "错误：无法获取应用实例"

        # 通过主窗口查找 VisionMindAgent 实例
        agent = None
        for widget in app.topLevelWidgets():
            if hasattr(widget, '_ai_agent'):
                agent = widget._ai_agent
                break

        if not agent or not getattr(agent, '_tool_manager', None):
            return "错误：无法找到 AI Agent 或工具管理器未初始化"

        activated, not_found, ui_only = agent._tool_manager.activate_tools(tool_names or [])

        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()

        # 复用 ToolSearchAgent 的 display_name 计算，保证输出名与 LLM 可调用名一致
        display_name_fn = ToolSearchAgent._get_display_name

        lines = []
        if activated:
            lines.append(f"成功激活 {len(activated)} 个工具：")
            for name in activated:
                meta = registry.get_meta(name)
                desc = meta.description if meta else ""
                callable_name = display_name_fn(name)
                lines.append(f"- {callable_name}: {desc}")
            lines.append(
                "这些工具将在下一轮对话中可用。"
                "下一轮请直接用上方列出的工具名（'-' 后、冒号前的名称）调用，"
                "不要带插件前缀，也不要在本轮调用。")

        if ui_only:
            lines.append(
                f"\n以下 {len(ui_only)} 个工具为界面内部工具（scope=ui），"
                f"仅供 UI 操作与录制回放使用，不对 AI 开放，无法被激活调用：")
            for name in ui_only:
                meta = registry.get_meta(name)
                desc = meta.description if meta else ""
                lines.append(f"- {display_name_fn(name)}: {desc}")
            lines.append(
                "请改用 search_tools 搜索功能等价的 AI 可用工具（scope=agent）。")

        if not_found:
            lines.append(f"\n未找到 {len(not_found)} 个工具：")
            for name in not_found:
                lines.append(f"- {name}")
            lines.append(
                "请使用 search_tools 搜索正确的工具名称，"
                "并使用搜索结果中列出的工具名（括号前的名称）来激活。")

        if not activated and not not_found and not ui_only:
            return "未激活任何工具（tool_names 为空）。"

        return "\n".join(lines)
