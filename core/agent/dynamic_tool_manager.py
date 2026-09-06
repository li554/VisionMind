"""
DynamicToolManager — 上下文感知的动态工具管理器

根据当前界面、用户输入关键词、最近使用分类以及显式激活的工具（LRU 队列）
动态过滤需要暴露给 Agent 的 action 工具，避免一次性暴露所有分类导致
工具 schema 过大。

过滤策略：
1. 始终包含 CORE_TOOL_NAMES 对应的核心工具（由调用方通过 core_tools 传入）
2. 始终包含 NAVIGATION_CATEGORIES（导航分类）
3. 根据 current_interface 从 INTERFACE_CATEGORIES 取该界面的默认分类
4. 根据 user_input 中的关键词匹配 _DOMAIN_KEYWORDS，加入对应分类
5. 合并 recent_categories（最近使用的工具分类）
6. 附加显式激活的工具（_activated_tools LRU 队列），即使其分类不在
   active_categories 中也保留，支持跨界面多轮对话
"""

from collections import OrderedDict
from typing import List, Optional
from core.corecoder.tools import Tool


# 界面 objectName → 应额外暴露的 category
# 用于记录各界面依赖哪些分类，以便统一收集所有分类
INTERFACE_CATEGORIES = {
    "AnnotationInterface": ["标注", "画布", "自动标注", "录制", "视图"],
    "DashboardInterface": ["项目"],
    "VersionInterface": ["版本管理"],
    "ConfigInterface": ["配置"],
    "FlowInterface": ["流程配置"],
    "TrainingInterface": ["训练"],
    "InferenceInterface": ["推理"],
    "KachiInterface": ["尺寸测量"],
    "UnsupervisedTrainingInterface": ["无监督训练"],
    "TestInterface": ["测试"],
}

# 所有界面都暴露的全局 category
GLOBAL_CATEGORIES = ["导航", "基础控制", "循环控制", "流程控制", "断言", "计时器", "对话框交互", "输入模拟", "意图分析"]

# 所有分类的并集（始终全部暴露，不按界面过滤）
ALL_CATEGORIES = sorted(set(GLOBAL_CATEGORIES + sum(INTERFACE_CATEGORIES.values(), []) + ["通用"]))

# switch_to 快捷名映射
INTERFACE_SHORT_NAMES = {
    "DashboardInterface": "dashboard",
    "AnnotationInterface": "annotation",
    "GenerationInterface": "generation",
    "TrainingInterface": "training",
    "InferenceInterface": "inference",
    "ConfigInterface": "config",
    "FlowInterface": "flow",
    "KachiInterface": "kachi",
    "VersionInterface": "version",
    "UnsupervisedTrainingInterface": "unsupervised",
    "TestInterface": "test_interface",
}

# 核心工具名称（始终可用，不受过滤影响）
CORE_TOOL_NAMES = {"ask_user", "bash", "write_file", "get_state", "search_tools",
                   "activate_tools", "list_prompt_sections", "inject_prompt", "get_active_prompts",
                   "generate_doc", "read_doc", "list_docs", "open_doc", "run_script"}

# 导航工具分类（始终纳入）
NAVIGATION_CATEGORIES = ["导航"]

# 用户输入关键词 → 工具分类映射
_DOMAIN_KEYWORDS = {
    "标注": ["标注", "画框", "画矩形", "分割", "polygon", "rectangle", "标注框", "labelme", "打点"],
    "画布": ["画布", "缩放", "平移", "截图", "canvas", "zoom", "pan"],
    "自动标注": ["自动标注", "一键", "批量标注", "auto", "batch"],
    "训练": ["训练", "train", "模型训练", "epoch", "loss"],
    "推理": ["推理", "inference", "预测", "predict", "检测"],
    "版本管理": ["版本", "version", "导出", "export", "coco", "yolo"],
    "项目": ["项目", "project", "数据集", "dataset", "导入"],
    "配置": ["配置", "config", "设置", "setting"],
    "尺寸测量": ["尺寸", "测量", "kachi", "measure"],
    "无监督训练": ["无监督", "unsupervised"],
}


class DynamicToolManager:
    """根据当前界面动态管理 tool 列表"""

    def __init__(self, main_window=None):
        """
        Args:
            main_window: MainWindow 实例，用于获取当前界面和插件信息
        """
        self._main_window = main_window
        # LRU 工具激活队列：tool_name → token_count
        self._activated_tools: "OrderedDict[str, int]" = OrderedDict()

    def set_main_window(self, main_window):
        """设置 MainWindow 引用"""
        self._main_window = main_window

    def get_current_interface(self) -> str:
        """获取当前界面的 objectName"""
        if not self._main_window:
            return ""
        try:
            current = self._main_window.stackedWidget.currentWidget()
            return current.objectName() if current else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # LRU 工具激活管理
    # ------------------------------------------------------------------

    def activate_tools(self, tool_names: list) -> tuple:
        """
        将工具加入 LRU 激活队列。已存在的工具会被移动到队尾（最新位置）。

        scope="ui" 的工具仅供界面操作/录制回放使用，不对 AI 暴露
        （get_tools_for_context 会过滤掉），因此不会被加入激活队列。

        Args:
            tool_names: 待激活的工具名列表（短名或完整名均可）

        Returns:
            (成功激活的工具全名列表, 未找到的工具名列表, 仅 UI 使用的工具全名列表)
        """
        from core.common.action_registry import ActionRegistry

        registry = ActionRegistry.instance()

        activated = []
        not_found = []
        ui_only = []

        for name in tool_names or []:
            if not name:
                continue
            # 通过 registry 判断工具是否存在
            if not registry.has_action(name):
                not_found.append(name)
                continue

            full_name = registry._resolve_name(name)
            meta = registry.get_meta(name)
            # scope="ui" 的工具不对 AI 暴露，禁止激活
            if getattr(meta, "scope", None) != "agent":
                ui_only.append(full_name)
                continue

            token_count = self._estimate_tool_tokens(meta)
            # 写入（或更新）token_count，并移动到队尾表示最新使用
            self._activated_tools[full_name] = token_count
            self._activated_tools.move_to_end(full_name)
            activated.append(full_name)

        if activated or ui_only:
            print(f"[DynamicToolManager] activate_tools: "
                  f"activated={activated}, ui_only={ui_only}, not_found={not_found}, "
                  f"queue_size={len(self._activated_tools)}")

        return activated, not_found, ui_only

    def touch_tool(self, tool_name: str):
        """
        工具被调用时移动到 LRU 最新位置。

        Args:
            tool_name: 工具名（短名或完整名）
        """
        if not tool_name:
            return

        from core.common.action_registry import ActionRegistry

        registry = ActionRegistry.instance()
        # 解析为完整名再匹配队列
        full_name = registry._resolve_name(tool_name) if registry.has_action(tool_name) else tool_name

        if full_name in self._activated_tools:
            self._activated_tools.move_to_end(full_name)
        elif tool_name in self._activated_tools:
            self._activated_tools.move_to_end(tool_name)

    def evict_lru_tools(self, token_budget=None) -> list:
        """
        按 LRU 年龄 × token 贡献量排序，淘汰优先级最高的工具。

        优先级计算：score = token_count * lru_position_weight
        其中 lru_position_weight 从 1.0（最旧）线性递减到 0.1（最新）。
        score 越高越优先被淘汰。

        Args:
            token_budget: 可选，目标 token 预算。若提供则淘汰至总 token <= 预算；
                          若为 None 则清空整个队列。

        Returns:
            被淘汰的工具名列表
        """
        if not self._activated_tools:
            return []

        # 不提供预算：清空队列
        if token_budget is None:
            evicted = list(self._activated_tools.keys())
            self._activated_tools.clear()
            print(f"[DynamicToolManager] evict_lru_tools: cleared {len(evicted)} tools")
            return evicted

        total_tokens = sum(self._activated_tools.values())
        if total_tokens <= token_budget:
            return []

        n = len(self._activated_tools)
        # 计算每个工具的淘汰分数
        scored_tools = []
        for i, (tool_name, token_count) in enumerate(self._activated_tools.items()):
            # i=0 是最旧（队首），i=n-1 是最新（队尾）
            # weight: 1.0 (最旧) → 0.1 (最新)
            if n > 1:
                weight = 1.0 - 0.9 * (i / (n - 1))
            else:
                weight = 1.0
            score = token_count * weight
            scored_tools.append((tool_name, score, token_count))

        # 按分数降序：分数高的优先淘汰
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        evicted = []
        for tool_name, _score, token_count in scored_tools:
            if total_tokens <= token_budget:
                break
            self._activated_tools.pop(tool_name, None)
            evicted.append(tool_name)
            total_tokens -= token_count

        if evicted:
            print(f"[DynamicToolManager] evict_lru_tools: evicted={evicted}, "
                  f"remaining={len(self._activated_tools)}, total_tokens={total_tokens}")

        return evicted

    def get_activated_tool_names(self) -> list:
        """返回当前 LRU 队列中的工具名列表（按 LRU 顺序，最旧在前）"""
        return list(self._activated_tools.keys())

    def reset_activated(self):
        """清空 _activated_tools 队列"""
        size = len(self._activated_tools)
        self._activated_tools.clear()
        if size:
            print(f"[DynamicToolManager] reset_activated: cleared {size} tools")

    # ------------------------------------------------------------------
    # 上下文感知过滤
    # ------------------------------------------------------------------

    def get_active_domains(self, user_input: str, current_interface: str,
                           recent_categories: list) -> list:
        """
        根据上下文信息计算当前活跃的工具分类列表。

        合并来源：
        1. user_input 中的关键词匹配 _DOMAIN_KEYWORDS
        2. current_interface 从 INTERFACE_CATEGORIES 获取的默认分类
        3. recent_categories 最近使用的工具分类
        4. NAVIGATION_CATEGORIES（始终包含）

        Args:
            user_input: 用户输入文本
            current_interface: 当前界面 objectName
            recent_categories: 最近使用的工具分类列表

        Returns:
            去重后的活跃分类列表
        """
        active = []
        seen = set()

        def _add_category(cat):
            if cat and cat not in seen:
                seen.add(cat)
                active.append(cat)

        # 1. 关键词匹配
        if user_input:
            user_lower = user_input.lower()
            for domain, keywords in _DOMAIN_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in user_lower:
                        _add_category(domain)
                        break

        # 2. 当前界面的默认分类
        if current_interface and current_interface in INTERFACE_CATEGORIES:
            for cat in INTERFACE_CATEGORIES[current_interface]:
                _add_category(cat)

        # 3. 最近使用的分类
        if recent_categories:
            for cat in recent_categories:
                _add_category(cat)

        # 4. 始终包含导航分类
        for cat in NAVIGATION_CATEGORIES:
            _add_category(cat)

        # 5. 始终包含全局分类（所有界面都暴露，如 意图分析/基础控制 等）
        for cat in GLOBAL_CATEGORIES:
            _add_category(cat)

        return active

    def get_tools_for_context(self, core_tools: Optional[List[Tool]] = None,
                              user_input: str = "",
                              current_interface: str = "",
                              recent_categories: Optional[list] = None) -> List[Tool]:
        """
        生成上下文感知的工具列表。

        只收集 active_categories 中的 action（按当前界面/关键词/最近分类过滤），
        并附加显式激活的工具（LRU 队列）。最后合并 core_tools。

        Args:
            core_tools: CoreCoder 基础工具列表（read_file, write_file 等）
            user_input: 用户输入文本，用于关键词匹配
            current_interface: 当前界面 objectName；为空时自动获取
            recent_categories: 最近使用的工具分类列表

        Returns:
            合并后的 tool 列表
        """
        from core.agent.tools.action_tool import ActionTool
        from core.common.action_registry import ActionRegistry

        registry = ActionRegistry.instance()

        # 1. 获取活跃分类
        if not current_interface:
            current_interface = self.get_current_interface()
        active_domains = self.get_active_domains(
            user_input, current_interface, recent_categories or []
        )

        # 2. 收集 active_categories 中的 action
        all_metas = registry.list_actions()
        matched_metas = [
            m for m in all_metas
            if m.category in active_domains and m.scope == "agent"
        ]

        # 3. 为每个 action 生成 ActionTool
        action_tools = []
        seen_names = set()
        for meta in matched_metas:
            # 获取完整 action name（带前缀）
            full_name = self._find_full_name(registry, meta.name)
            if not full_name:
                continue

            # 生成 display_name（去掉前缀，如 "annotation.next_image" → "next_image"）
            display_name = meta.name
            if "." in full_name:
                display_name = full_name.split(".", 1)[1]

            # 避免重复（如 drawing_widget 和 annotation_interface 都有 toggle_enhance）
            if display_name in seen_names:
                continue
            seen_names.add(display_name)

            # 构建参数 schema
            params_schema = self._build_params_schema(meta, registry)

            tool = ActionTool(
                action_name=full_name,
                display_name=display_name,
                description=meta.description,
                parameters_schema=params_schema,
            )
            action_tools.append(tool)

        # 4. 附加显式激活的工具（LRU 队列中，即使其分类不在 active_domains 中）
        activated_names = self.get_activated_tool_names()
        for act_name in activated_names:
            meta = registry.get_meta(act_name)
            if not meta:
                continue
            # 仅暴露 scope="agent" 的工具给 Agent
            if getattr(meta, "scope", None) != "agent":
                continue

            display_name = act_name.split(".", 1)[1] if "." in act_name else act_name
            if display_name in seen_names:
                continue
            seen_names.add(display_name)

            params_schema = self._build_params_schema(meta, registry)
            tool = ActionTool(
                action_name=act_name,
                display_name=display_name,
                description=meta.description,
                parameters_schema=params_schema,
            )
            action_tools.append(tool)

        # 5. 合并 CoreCoder 基础工具 + 动态 action tools
        result = list(core_tools or [])
        result.extend(action_tools)

        print(f"[DynamicToolManager] ",
              f"active_domains={active_domains}, "
              f"activated={len(activated_names)}, "
              f"actions={len(action_tools)}, total_tools={len(result)}")

        return result

    def get_all_ai_action_tools(self) -> list:
        """返回所有 scope="agent" 的 ActionTool 列表（不经过分类过滤）。

        用于补充 _tool_by_name 索引：LLM 可能在 search_tools 后调用不在活跃列表中的工具，
        将所有 AI 工具加入索引可避免 unknown tool 错误。
        这些工具不加入 agent.tools（不占用 LLM token 预算），仅用于工具查找。
        """
        from core.agent.tools.action_tool import ActionTool
        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        all_metas = registry.list_actions()
        tools = []
        seen = set()
        for meta in all_metas:
            if getattr(meta, "scope", None) != "agent":
                continue
            full_name = self._find_full_name(registry, meta.name)
            if not full_name:
                continue
            display_name = full_name.split(".", 1)[1] if "." in full_name else full_name
            if display_name in seen:
                continue
            seen.add(display_name)
            params_schema = self._build_params_schema(meta, registry)
            tools.append(ActionTool(
                action_name=full_name,
                display_name=display_name,
                description=meta.description,
                parameters_schema=params_schema,
            ))
        return tools

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _estimate_tool_tokens(self, meta) -> int:
        """估算工具的 token 消耗量（基于参数数量，简单启发式）"""
        if not meta:
            return 100
        params = getattr(meta, "params", None)
        param_count = len(params) if params else 0
        # 基础 50 token + 每个参数约 50 token（含 schema 描述）
        return 50 + param_count * 50

    def _find_full_name(self, registry, meta_name: str) -> Optional[str]:
        """根据 meta.name 找到完整的注册名（带前缀）"""
        for full_name in registry._actions:
            if full_name == meta_name or full_name.endswith(f".{meta_name}"):
                return full_name
        return meta_name

    def _build_params_schema(self, meta, registry) -> dict:
        """为 action 构建 JSON Schema 格式的参数定义"""
        properties = {}
        required = []

        for param_name, param_def in meta.params.items():
            prop = registry._parse_param_schema(param_def)

            # 处理动态 enum
            if "dynamic_enum" in prop:
                dynamic_values = self._resolve_dynamic_enum(prop.pop("dynamic_enum"))
                if dynamic_values:
                    prop["enum"] = dynamic_values

            properties[param_name] = prop

            if registry._param_required(param_def):
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _resolve_dynamic_enum(self, enum_type: str) -> list:
        """解析动态枚举值"""
        if enum_type == "switch_to_targets":
            return self._get_interface_targets()
        return []

    def _get_interface_targets(self) -> list:
        """从 _plugin_interfaces 动态获取所有可用界面名称。

        只返回 switchTo 实现支持的 target：objectName（如 "AnnotationInterface"）
        和快捷名（如 "annotation"）。显示名不被 switchTo 识别，不放入 enum。
        """
        if not self._main_window:
            return []

        targets = []
        try:
            for widget, (icon, _text, order, route_key) in self._main_window._plugin_interfaces.items():
                # 添加 objectName（如 "AnnotationInterface"）
                if route_key not in targets:
                    targets.append(route_key)
                # 添加快捷名（如 "annotation"）
                short = INTERFACE_SHORT_NAMES.get(route_key)
                if short and short not in targets:
                    targets.append(short)
        except Exception as e:
            print(f"[DynamicToolManager] 获取界面列表失败: {e}")

        return targets
