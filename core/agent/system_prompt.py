"""
系统提示词构建 — 动态生成 Agent 的系统提示词

Prompt.md 已重写为轻量版，作为基础提示词直接读取（不再含 {tool_list} 占位符）。
插件专属提示词通过 PromptManager 注册和注入。
模块维护 _is_dirty 标记，供外部模块（PromptManager、DynamicToolManager）
在工具/提示词变更时通知缓存失效。
"""

from pathlib import Path
from typing import List


_is_dirty = True  # 初始为 True，首次构建后置 False


def mark_dirty():
    """标记系统提示词缓存失效，下次构建需重新生成"""
    global _is_dirty
    _is_dirty = True


def is_dirty() -> bool:
    """返回当前是否为脏状态（需重新构建）"""
    return _is_dirty


def _get_prompt_md() -> str:
    """读取 Prompt.md 模板内容"""
    prompt_path = Path(__file__).parent.parent / "Prompt.md"
    return prompt_path.read_text(encoding="utf-8")


def _get_agents_md() -> str:
    """读取 Agent 基础提示词模板（core/Prompt.md）。"""
    return _get_prompt_md()


def build_system_prompt(tools) -> str:
    """
    构建完整的系统提示词。

    Prompt.md 已为轻量版，直接作为基础提示词（不再替换 {tool_list}）；
    插件知识段通过 PromptManager 注入。构建后清除 _is_dirty 标记。

    Args:
        tools: 当前可用的 tool 列表（含 ActionTool 和 CoreCoder 基础工具）
    """
    global _is_dirty
    base_prompt = _get_prompt_md()

    from core.agent.prompt_manager import PromptManager
    plugin_prompt = PromptManager.instance().get_active_formatted()

    if plugin_prompt:
        result = base_prompt + "\n\n" + plugin_prompt
    else:
        result = base_prompt

    _is_dirty = False
    return result


def _build_category_overview(tools) -> str:
    """
    生成精简的工具分类概览（仅分类名 + 工具数量，非逐条工具描述）。

    遍历 tools，区分 ActionTool 和非 ActionTool：
      - ActionTool 通过 ActionRegistry 获取 meta.category 按分类分组计数；
      - 非 ActionTool 归入"基础工具"计数。
    返回格式示例：
        ## 工具分类概览（共 N 个工具）
        - 标注: 5 个工具
        - 画布: 3 个工具
        - 导航: 4 个工具
        ...
        使用 search_tools 查询具体工具名称和参数。

    本函数暂不被 build_system_prompt 调用（Prompt.md 中已有静态分类概览），
    保留供调试使用。

    Args:
        tools: 当前可用的 tool 列表

    Returns:
        分类概览文本
    """
    from core.agent.tools.action_tool import ActionTool
    from core.common.action_registry import ActionRegistry

    action_tools = []
    other_count = 0
    for t in tools:
        if isinstance(t, ActionTool):
            action_tools.append(t)
        else:
            other_count += 1

    registry = ActionRegistry.instance()
    category_counts = {}
    for t in action_tools:
        meta = registry.get_meta(t._action_name)
        cat = meta.category if meta else "其他"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    total = len(action_tools) + other_count
    lines = [f"## 工具分类概览（共 {total} 个工具）"]
    for cat in sorted(category_counts.keys()):
        lines.append(f"- {cat}: {category_counts[cat]} 个工具")
    if other_count:
        lines.append(f"- 基础工具: {other_count} 个工具")
    lines.append("使用 search_tools 查询具体工具名称和参数。")

    return "\n".join(lines)
