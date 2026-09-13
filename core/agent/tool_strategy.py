"""
通用工具调度策略接口 — 决定每轮暴露给 Agent 的工具列表

策略职责：
- build_tools() 返回本轮应暴露给 Agent 的工具列表（含 core_tools）
- on_round_end() 可选：每轮会话结束后回调（供「调用失败后补全」等动态策略二次调整工具集）

扩展方式：子类化 ToolStrategy 并实现 build_tools()（可选覆盖 on_round_end/on_tool_result），
用 @register_strategy 登记，即被核心调度识别；核心调度只面向本接口分发，无需改动。

当前内置策略（唯一）：**lru** ——「界面/关键词/最近分类 + LRU 激活队列」上下文感知过滤，
可选「窗口大小」（window_size）把暴露的 action 工具数裁到窗口内：
- 保底 = LRU 激活队列 + 命中分类内「与指令文本强相关（规则分 >= protect_min_score）」的工具；
  保底永不裁剪，窗口按需自动扩张（上限 max_window_size）
- 相关性打分只有一种实现：RuleScorer（名称命中 2 分 / 描述命中 1 分），零依赖、确定性
- 历史策略 full / compact / full_sorted / search_only / rag 已按用户要求移除（2026-09-14）；
  需要新增策略时按上面的扩展方式实现并登记即可

策略规格（spec）：
策略名可带 `@` 参数，用于基准实验直接对比同一策略的不同配置，例如
`lru@20`（窗口 20）、`lru@window=20,max=48,protect=3`。未带参数时等价于纯策略名。
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# 策略注册表
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(cls: type) -> type:
    """装饰器：登记一个 ToolStrategy 子类到注册表（按 cls.name）。"""
    _STRATEGY_REGISTRY[cls.name] = cls
    return cls


def lookup_strategy(name: str) -> Optional[type]:
    """按名称查策略类；未注册返回 None。"""
    return _STRATEGY_REGISTRY.get(name)


def list_strategies() -> List[str]:
    """返回所有已注册策略名（稳定排序）。"""
    return sorted(_STRATEGY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 结果判定工具（与 builtin_actions._parse_session 的失败口径对齐）
# ---------------------------------------------------------------------------

_JSON_STATUS_ERROR_1 = '"status": "error"'
_JSON_STATUS_ERROR_2 = '"status":"error"'


def is_result_error(content: str) -> bool:
    """判断工具结果文本是否为「错误/失败」。

    与 builtin_actions._parse_session 的 ok 判定保持一致：
    - ❌/Error 前缀
    - JSON 结构错误（{"status": "error"}）
    用于 compact 策略「失败后补齐完整描述」的触发判定。
    """
    if not content:
        return False
    s = content.lstrip()
    if s.startswith("❌") or s.startswith("Error"):
        return True
    head = s[:600]
    return _JSON_STATUS_ERROR_1 in head or _JSON_STATUS_ERROR_2 in head


# 访问/调用层面的失败标记，用于「工具命中率」判定（正确访问 + 正确调用才算命中）
# 注意：ActionTool 的执行期异常是中文文案 `❌ 错误: <工具> 执行失败: <异常>`，
# 只匹配英文 "Error executing" 会把真实崩溃算成命中（2026-09-11 窗口基准实测发现），
# 故补上中文标记 "执行失败"。
_ACCESS_OR_CALL_FAILURES = (
    ("unknown_tool", "unknown tool"),
    ("not_registered", "未注册"),
    ("bad_arguments", "bad arguments"),
    ("execution_error", "Error executing"),
    ("execution_error", "执行失败"),
    ("not_found", "not found"),
)


def classify_tool_hit(result: str, fallback_ok: bool = True):
    """逐次判定一次工具调用的命中情况。

    命中定义（用户口径）：工具被「正确访问」且被「正确调用」→ 命中；
    否则（访问错/调用错）→ 未命中。

    - 正确访问：工具存在/已注册（结果里未出现 unknown tool / 未注册 / not found）
    - 正确调用：参数有效、工具真正执行（结果里未出现 bad arguments / Error executing）
    - 业务逻辑返回的错误（如「已达最后一张图」）不算调用错误，仍视为命中。

    Returns:
        (hit: bool, reason: str)
        hit=True  时 reason 为 "ok"
        hit=False 时 reason 为具体失败类别（unknown_tool / not_registered /
                   bad_arguments / execution_error / not_found）
    """
    if not result:
        # 无结果文本（罕见）：fallback_ok 由调用方根据其它线索决定（默认当作命中）
        return bool(fallback_ok), "ok" if fallback_ok else "no_result"
    for reason, marker in _ACCESS_OR_CALL_FAILURES:
        if marker.lower() in result.lower():
            return False, reason
    return True, "ok"


# ---------------------------------------------------------------------------
# 策略抽象接口
# ---------------------------------------------------------------------------


class ToolStrategy(ABC):
    """工具调度策略抽象接口。

    子类需实现 build_tools()；可按需覆盖 on_round_end()。
    构造时传入 DynamicToolManager，可复用其 _build_params_schema / _find_full_name
    等工具构造辅助方法（不 import 该模块，避免循环依赖）。
    """

    #: 策略标识（注册名）
    name: str = ""

    def __init__(self, manager: Any):
        self.manager = manager

    @property
    def label(self) -> str:
        """策略展示标签（默认 = 注册名；带参数的策略可覆盖，如 lru@20）。

        供基准报告 / 调试窗口显示「同名策略 + 不同配置」的区分标识。
        """
        return self.name

    @abstractmethod
    def build_tools(self, core_tools: List[Any], user_input: str = "",
                    current_interface: str = "", recent_categories: Optional[list] = None) -> List[Any]:
        """返回本轮应暴露给 Agent 的工具列表（含 core_tools）。"""

    def on_round_end(self, messages: List[dict]) -> bool:
        """每轮会话结束后回调（在 Agent 工作线程内、LLM chat 循环轮次之间执行）。

        若返回 True，表示暴露的工具集需要重建（由宿主负责调用 build_tools 并刷新
        agent.tools / 系统提示词）。默认返回 False。
        """
        return False


# ---------------------------------------------------------------------------
# 内置策略：lru（现状）
# ---------------------------------------------------------------------------


# 工具在窗口内的相关性分层（越小越优先）
TIER_ACTIVATED = 0     # LRU 激活队列（最近使用）—— 保底
TIER_RELEVANT = 1      # 与指令强相关（相关性分 >= protect_min_score）—— 保底
TIER_SCORED = 2        # 与指令弱相关（相关性分 > 0）
TIER_KEYWORD = 3       # 命中分类内、但工具本身与指令无文本相关的工具
TIER_INTERFACE = 4     # 当前界面的默认分类
TIER_RECENT = 5        # 最近使用过的分类
TIER_REST = 6          # 其余活跃分类（导航/全局等）

#: 保底分层：这两层永不裁剪（超出窗口时窗口自动扩张）
MANDATORY_TIERS = (TIER_ACTIVATED, TIER_RELEVANT)

#: 判定「强相关」的默认阈值：名称命中权重 2、描述命中权重 1，
#: 故 2 分 = 名称命中 或 描述命中 2 个语义词（见 tool_relevance）
DEFAULT_PROTECT_MIN_SCORE = 2


def relevance_terms(user_input: str, keyword_domains: Set[str]) -> Set[str]:
    """指令中与「命中分类」相关的语义词（用于挑出分类内真正相关的工具）。

    来源：
    1. 命中分类的关键词表（如 版本管理 → 版本/version/导出/export/coco/yolo）
    2. 指令里的英文标识符（如 yolo、coco、项目名），长度 >= 3
    """
    terms: Set[str] = set()
    if not user_input:
        return terms
    try:
        from core.agent.dynamic_tool_manager import _DOMAIN_KEYWORDS
        for domain in keyword_domains:
            for kw in _DOMAIN_KEYWORDS.get(domain, []):
                if kw:
                    terms.add(kw.lower())
    except Exception:
        pass
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", user_input):
        terms.add(token.lower())
    return terms


def tool_relevance(display_name: str, description: str, terms: Set[str]) -> int:
    """工具与指令的相关分：名称命中权重 2，描述命中权重 1。"""
    if not terms:
        return 0
    name = (display_name or "").lower()
    desc = (description or "").lower()
    score = 0
    for term in terms:
        if term in name:
            score += 2
        elif term in desc:
            score += 1
    return score


class RuleScorer:
    """相关性打分器（唯一实现）：名称命中 2 分 / 描述命中 1 分。

    已知局限（离线实测）：依赖人工词表（"提示词库/示例库/自定义模型"这类词表外说法会漏）；
    长描述堆词会虚高；分类泛词（"标注"）几乎命中半个插件 → 区分度低。
    窗口预算 + 保底层是这些局限的兜底，也是 baseline 的表现来源。
    """

    backend = "rule"

    def __init__(self, protect_min_score: int = DEFAULT_PROTECT_MIN_SCORE):
        self._protect_min_score = int(protect_min_score)
        self._scores: Dict[str, float] = {}
        self.terms: Set[str] = set()
        self.fallback_reason = ""

    def prepare(self, user_input: str, descriptions: Dict[str, str],
                keyword_domains: Set[str]) -> None:
        self.terms = relevance_terms(user_input, keyword_domains)
        self._scores = {
            name: float(tool_relevance(name, desc or "", self.terms))
            for name, desc in descriptions.items()
        }

    def score(self, name: str) -> float:
        return float(self._scores.get(name, 0.0))

    def is_strong(self, name: str) -> bool:
        return self.score(name) >= self._protect_min_score

    def is_weak(self, name: str) -> bool:
        return self.score(name) > 0

    def order_key(self, name: str) -> tuple:
        """窗口内排序主键（越小越前）：规则分降序。"""
        return (-self.score(name),)

    @property
    def available(self) -> bool:
        return True

    def stats(self) -> Dict[str, Any]:
        scores = list(self._scores.values())
        return {
            "backend": self.backend,
            "available": True,
            "fallback_reason": self.fallback_reason,
            "protect_min_score": self._protect_min_score,
            "strong": sum(1 for s in scores if s >= self._protect_min_score),
            "weak": sum(1 for s in scores if s > 0),
            "terms": sorted(self.terms),
        }


def select_in_window(ranked: List[str], mandatory: int, window_size: int,
                     max_window_size: int = 0) -> tuple:
    """把已排序的候选工具按窗口裁剪（纯函数，便于离线验证）。

    Args:
        ranked: 已按相关性排序的工具名列表；前 ``mandatory`` 个是保底项
        mandatory: 保底项数量（永不裁剪；超过窗口时窗口自动扩张）
        window_size: 配置的窗口大小；``<=0`` 表示不裁剪（保持历史行为）
        max_window_size: 自动扩张的上限；``<=0`` 表示不额外限制。
            保底项本身不受该上限约束（宁可超窗口，也不丢关键工具）

    Returns:
        (kept, dropped, target)：保留列表、被裁掉列表、实际窗口大小。
        不裁剪时 target = len(ranked)。
    """
    ranked = list(ranked)
    total = len(ranked)
    mandatory = max(0, min(int(mandatory), total))
    if window_size <= 0:
        return ranked, [], total
    window_size = int(window_size)
    # 目标窗口：至少覆盖保底项，至多不超过扩张上限（上限不裁剪保底项）
    cap = int(max_window_size) if max_window_size and max_window_size > 0 else window_size
    target = min(max(window_size, mandatory), max(cap, mandatory))
    target = min(target, total)
    return ranked[:target], ranked[target:], target


@register_strategy
class LRUStrategy(ToolStrategy):
    """现状策略：界面/关键词/最近分类 + LRU 激活队列 的上下文感知过滤（可选窗口）。

    完整调度逻辑在本策略内实现（自包含，不再依赖 manager.get_tools_for_context）：
    1. 始终合并 core_tools（含 search_tools / get_state 等核心工具，不被过滤）
    2. 由 get_active_domains 计算活跃分类（关键词 + 当前界面 + 最近分类 + 导航/全局）
    3. 收集活跃分类下的 scope="agent" action（完整 schema）
    4. 附加 LRU 激活队列中的工具（跨界面多轮记忆），即使其分类不在活跃集
    5. 「窗口」：以 window_size 为上限裁剪 action 工具数量（0 = 不裁剪；默认 16）

    窗口口径（避免重演 lru 因漏掉 version 分类而整条任务失败的旧问题）：
    - 保底项：
      1) LRU 激活队列（最近使用）——**不受窗口约束**，超出窗口时窗口自动扩张到覆盖它
         （上限 max_window_size，扩张本身也不裁剪），因为那是跨轮记忆里真正用过的工具
      2) **与指令强相关的工具**（名称/描述命中指令语义词，`protect_min_score` 默认 2 分，
         名称命中权重 2、描述命中权重 1）——按工具判定、不按分类归属；
         但**最多占满窗口预算**（`window_size - 激活数`），否则长指令动辄命中几十个词、
         窗口会被保底顶穿而失效（实测 mandatory=40 > window=8）
    - 剩余槽位按相关性分层填充：弱相关工具（分>0）→ 命中分类的其余工具 →
      当前界面分类 → 最近使用分类 → 其余活跃分类
    - 被裁掉的工具仍在 agent 的 `_tool_by_name` 兜底索引中；只要被显式
      activate_tools 激活，下一轮即进入保底层重新可见

    窗口大小配置优先级：构造参数/set_window_size 覆盖 > core.json 的
    `lru_window_size` > 环境变量 `VISIONLEE_LRU_WINDOW_SIZE` > 16（默认）。
    0 = 不裁剪（想回到全量暴露就显式设 0 / `lru@0`）。
    """

    name = "lru"

    #: core.json 中持久化窗口大小的键（0 = 不裁剪）
    SETTING_KEY = "lru_window_size"
    #: 环境变量覆盖（便于实验不改配置直接对比）
    ENV_KEY = "VISIONLEE_LRU_WINDOW_SIZE"
    #: 自动扩张上限：非保底槽位的上限（保底项不受此限）
    DEFAULT_MAX_WINDOW_SIZE = 64
    #: 默认窗口大小：核心配置缺失时使用（实测 16 是"上下文/成功率"的拐点）
    DEFAULT_WINDOW_SIZE = 16

    def __init__(self, manager, window_size=None, max_window_size=None,
                 protect_min_score=None):
        super().__init__(manager)
        #: 显式窗口覆盖（构造参数 / spec，如 lru@20）；None = 走配置
        self._window_override = None if window_size is None else max(0, int(window_size))
        self._max_window_size = (int(max_window_size) if max_window_size
                                 else self.DEFAULT_MAX_WINDOW_SIZE)
        #: 保底阈值：规则分 >= 该值算「强相关」，永不裁剪
        self._protect_min_score = (max(1, int(protect_min_score))
                                   if protect_min_score else DEFAULT_PROTECT_MIN_SCORE)
        #: 最近一次 build_tools 的窗口统计（供调试窗口 / 基准报告读取）
        self.last_window_stats: Dict[str, Any] = {}
        #: 设置读取失败只提示一次，避免每轮刷屏
        self._setting_warned = False

    # ---- 窗口配置 --------------------------------------------------------

    def configured_window_size(self) -> int:
        """当前生效的窗口大小（0 = 不裁剪；无任何配置时用 DEFAULT_WINDOW_SIZE）。"""
        if self._window_override is not None:
            return max(0, int(self._window_override))
        val = self._read_setting()
        if val is not None:
            return max(0, int(val))
        env = (os.environ.get(self.ENV_KEY) or "").strip()
        if env.lstrip("+-").isdigit():
            return max(0, int(env))
        return self.DEFAULT_WINDOW_SIZE

    def _read_setting(self):
        """读 core.json 的窗口设置；缺失/读取失败返回 None。"""
        try:
            from core.common.settings import settings
            val = settings.get(self.SETTING_KEY, None)
            if val is None or val == "":
                return None
            return int(val)
        except Exception as e:
            if not self._setting_warned:
                self._setting_warned = True
                print(f"[ToolStrategy] lru 读取窗口设置失败（按默认 {self.DEFAULT_WINDOW_SIZE} 处理）: {e}")
            return None

    def set_window_size(self, size, persist=True) -> int:
        """设置窗口大小；persist=True 时写入 core.json（设置界面 / 调试窗口用）。

        Returns:
            设置后实际生效的窗口大小。
        """
        size = max(0, int(size))
        self._window_override = None   # 让 core.json 成为唯一事实来源（改动即时生效）
        if persist:
            try:
                from core.common.settings import settings
                settings.set(self.SETTING_KEY, size)
            except Exception as e:
                print(f"[ToolStrategy] lru 窗口设置持久化失败，改为本次会话内生效: {e}")
        if self.configured_window_size() != size:
            self._window_override = size   # 持久化不可用（如无 core.json）时兜底
        return self.configured_window_size()

    @property
    def label(self) -> str:
        """展示标签：窗口开启时为 ``lru@<窗口大小>``。"""
        window = self.configured_window_size()
        return f"{self.name}@{window}" if window > 0 else self.name

    # ---- 相关性打分 ------------------------------------------------------

    @staticmethod
    def _relevance_terms(user_input: str, keyword_domains: Set[str]) -> Set[str]:
        """（兼容包装）见模块级 :func:`relevance_terms`。"""
        return relevance_terms(user_input, keyword_domains)

    @staticmethod
    def _tool_relevance(display_name: str, description: str, terms: Set[str]) -> int:
        """（兼容包装）见模块级 :func:`tool_relevance`。"""
        return tool_relevance(display_name, description, terms)

    @staticmethod
    def _category_tier(category, keyword_domains: Set[str], interface_domains: Set[str],
                       recent_categories: Set[str]) -> int:
        """分类来源决定的兜底分层（与指令无文本相关的工具用）。"""
        if category in keyword_domains:
            return TIER_KEYWORD
        if category in interface_domains:
            return TIER_INTERFACE
        if category in recent_categories:
            return TIER_RECENT
        return TIER_REST

    # ---- 调度主流程 ------------------------------------------------------

    def build_tools(self, core_tools, user_input="", current_interface="",
                    recent_categories=None):
        from core.agent.tools.action_tool import ActionTool
        from core.common.action_registry import ActionRegistry

        m = self.manager
        registry = ActionRegistry.instance()
        if not current_interface:
            current_interface = m.get_current_interface()
        recent_categories = list(recent_categories or [])
        active_domains = m.get_active_domains(user_input, current_interface, recent_categories)

        # 分层依据：只取关键词命中 / 界面默认分类（get_active_domains 的两个来源），
        # 用于判定保底项与窗口内优先级
        _kw = getattr(m, "get_keyword_domains", None)
        keyword_domains = set(_kw(user_input) if callable(_kw) else [])
        _if = getattr(m, "get_interface_domains", None)
        interface_domains = set(_if(current_interface) if callable(_if) else [])
        recent_set = set(recent_categories)
        activated_names = list(m.get_activated_tool_names())   # 最旧 → 最新
        terms = self._relevance_terms(user_input, keyword_domains)
        scorer = RuleScorer(self._protect_min_score)

        # 2. 收集活跃分类中的 agent action（完整 schema）
        candidates: Dict[str, Dict[str, Any]] = {}
        legacy_order: List[str] = []      # 不裁剪时的历史顺序：活跃分类 → 激活补充
        for meta in registry.list_actions():
            if meta.category not in active_domains or getattr(meta, "scope", None) != "agent":
                continue
            full_name = m._find_full_name(registry, meta.name)
            if not full_name:
                continue
            display_name = full_name.split(".", 1)[1] if "." in full_name else full_name
            if display_name in candidates:
                continue
            candidates[display_name] = {
                "full_name": full_name,
                "meta": meta,
                "category": getattr(meta, "category", None),
                "score": 0.0,
                "index": len(legacy_order),
                "tier": None,
                "activated_rank": None,
            }
            legacy_order.append(display_name)

        # 3. 附加 LRU 激活队列中的工具（分类不在活跃集也保留）→ 保底层
        for recency, act_name in enumerate(reversed(activated_names)):
            meta = registry.get_meta(act_name)
            if not meta or getattr(meta, "scope", None) != "agent":
                continue
            display_name = act_name.split(".", 1)[1] if "." in act_name else act_name
            entry = candidates.get(display_name)
            if entry is None:
                entry = {
                    "full_name": act_name,
                    "meta": meta,
                    "category": getattr(meta, "category", None),
                    "score": 0.0,
                    "index": len(legacy_order),
                    "tier": None,
                    "activated_rank": recency,
                }
                candidates[display_name] = entry
                legacy_order.append(display_name)
            else:
                entry["activated_rank"] = recency

        # 3.3 相关性打分（规则分：名称命中 2 / 描述命中 1）
        scorer.prepare(user_input,
                       {n: (e["meta"].description or "") for n, e in candidates.items()},
                       keyword_domains)
        for display_name, entry in candidates.items():
            entry["score"] = scorer.score(display_name)
        scorer_stats = scorer.stats()

        # 3.4 分层：相关性优先，分类只作兜底
        #     —— 按「工具」而非「分类」判定保底：像 manual.select_project 这类
        #        关键工具的真实分类是「导航」，只按命中分类保底会把它漏掉
        for display_name, entry in candidates.items():
            if entry["activated_rank"] is not None:
                entry["tier"] = TIER_ACTIVATED
            elif scorer.is_strong(display_name):
                entry["tier"] = TIER_RELEVANT
            elif scorer.is_weak(display_name):
                entry["tier"] = TIER_SCORED
            else:
                entry["tier"] = self._category_tier(
                    entry["category"], keyword_domains, interface_domains, recent_set)

        def _rank_key(name: str):
            entry = candidates[name]
            ok = scorer.order_key(name)
            ok = (tuple(ok) + (0.0, 0.0))[:2]        # 各后端统一补齐为 2 元，避免排序类型错配
            act = entry["activated_rank"]
            return (entry["tier"], act if act is not None else 0, ok[0], ok[1], entry["index"])

        ranked = sorted(candidates.keys(), key=_rank_key)

        # 3.5 保底集合：LRU 激活工具 + 强相关工具
        #     - 强相关部分**最多占满窗口预算**：长指令可能命中几十个语义词，
        #       若让强相关无限保底（自动扩张），窗口就永远裁不动（实测 mandatory=40 顶穿 window=8）
        #     - 只有 agent 自己激活过的工具（LRU 队列）才有「突破窗口」的扩张权：
        #       那是跨轮记忆里真正用过的工具，宁可超窗口也不丢
        window_size = self.configured_window_size()
        activated_kept = [n for n in ranked if candidates[n]["tier"] == TIER_ACTIVATED]
        relevant_all = [n for n in ranked if candidates[n]["tier"] == TIER_RELEVANT]
        if window_size > 0:
            budget = max(0, window_size - len(activated_kept))
            relevant_kept = relevant_all[:budget]
        else:
            relevant_kept = list(relevant_all)
        mandatory_names = activated_kept + relevant_kept
        protected_names = relevant_kept

        # 4. 应用窗口（window_size <= 0 = 不裁剪，输出保持历史顺序）
        kept, dropped, target = select_in_window(
            ranked, len(mandatory_names), window_size, self._max_window_size)
        ordered = legacy_order if window_size <= 0 else kept
        if window_size > 0 and len(activated_kept) > self._max_window_size:
            print(f"[ToolStrategy] lru 警告: 激活工具 {len(activated_kept)} 个 > 扩张上限 "
                  f"{self._max_window_size}，本轮到 {target} 个（激活工具不裁剪）")

        action_tools = [ActionTool(
            action_name=candidates[n]["full_name"],
            display_name=n,
            description=candidates[n]["meta"].description,
            parameters_schema=m._build_params_schema(candidates[n]["meta"], registry),
        ) for n in ordered]

        self.last_window_stats = {
            "window_size": window_size,
            "window_target": target,
            "max_window_size": self._max_window_size,
            "protect_min_score": self._protect_min_score,
            "scorer": scorer_stats,
            "mandatory": len(mandatory_names),
            "protected": len(protected_names),
            "protected_names": protected_names[:12],
            "relevant_candidates": len(relevant_all),
            "activated_mandatory": len(activated_kept),
            "relevance_budget_trimmed": len(relevant_all) > len(relevant_kept),
            "candidates": len(candidates),
            "exposed_actions": len(action_tools),
            "dropped": len(dropped),
            "dropped_names": list(dropped),
            "auto_expanded": bool(window_size > 0 and target > window_size),
            "keyword_domains": sorted(keyword_domains),
            "relevance_terms": sorted(terms),
            "active_domains": list(active_domains),
            "current_interface": current_interface,
        }

        # 1. 合并 core_tools（含 search_tools 等核心工具） + action tools
        result = list(core_tools or [])
        result.extend(action_tools)

        window_str = window_size if window_size > 0 else "∞"
        print(f"[ToolStrategy] lru: window={window_str} target={target} "
              f"mandatory={len(mandatory_names)}(激活{len(activated_names)}"
              f"+强相关{len(protected_names)}) "
              f"actions={len(candidates)}→{len(action_tools)} dropped={len(dropped)} "
              f"keyword_domains={sorted(keyword_domains)}, "
              f"activated={len(activated_names)}, total={len(result)}")
        if protected_names:
            print(f"[ToolStrategy] lru 强相关保底 {len(protected_names)}/{len(relevant_all)}: "
                  f"{protected_names[:12]}")
        if dropped:
            print(f"[ToolStrategy] lru 窗口外工具（可由 search_tools/activate_tools 唤回）: "
                  f"{dropped}")
        return result

    def window_status_text(self) -> str:
        """窗口状态多行文本（调试窗口显示用）。"""
        s = self.last_window_stats
        if not s:
            return ("== LRU 工具窗口 ==\n"
                    f"配置窗口: {self.configured_window_size() or '∞ (不裁剪)'}\n"
                    "(尚未构建过工具集，暂无效统计数据)")
        window = s.get("window_size") or 0
        lines = [
            "== LRU 工具窗口 ==",
            f"相关性打分: 规则分（名称命中 2 / 描述命中 1，阈值 {s.get('protect_min_score')}）",
            f"配置窗口: {window if window > 0 else '∞ (不裁剪)'}"
            f"  扩张上限: {s.get('max_window_size')}"
            f"  实际窗口: {s.get('window_target')}"
            + ("（已自动扩张）" if s.get("auto_expanded") else ""),
            f"候选 action: {s.get('candidates')}  →  暴露: {s.get('exposed_actions')}"
            f"  (core 工具另计)",
            f"保底工具: {s.get('mandatory')} 个（LRU 激活 + 与指令强相关工具，"
            f"阈值 {s.get('protect_min_score')} 分；强相关按窗口预算截断）",
            f"强相关保底: {', '.join(s.get('protected_names') or []) or '无'}"
            f"（候选 {s.get('relevant_candidates')} 个"
            + ("，已按窗口预算截断）" if s.get("relevance_budget_trimmed") else "）"),
            f"命中分类: {', '.join(s.get('keyword_domains') or []) or '无'}"
            f"｜相关词: {', '.join((s.get('relevance_terms') or [])[:12]) or '无'}",
            f"窗口外(被裁): {s.get('dropped')} 个",
        ]
        dropped_names = s.get("dropped_names") or []
        if dropped_names:
            lines.append(f"  裁剪名单: {', '.join(dropped_names)}")
        return "\n".join(lines)


# 策略规格（spec）解析：`lru` / `lru@20` / `lru@window=20,max=48`
# ---------------------------------------------------------------------------

SPEC_SEP = "@"


def strategy_base_name(spec: str) -> str:
    """取策略规格的注册名（``"lru@20"`` → ``"lru"``）。"""
    return (spec or "").split(SPEC_SEP, 1)[0].strip()


def _coerce_arg(raw: str):
    """把 spec 里的参数值转成 int/bool/str。"""
    text = (raw or "").strip()
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if text.lstrip("+-").isdigit():
        return int(text)
    return text


def parse_strategy_spec(spec: str) -> tuple:
    """解析策略规格 → (注册名, 参数字典)。

    支持（供基准实验直接对比同名策略的不同配置）：
    - ``"lru"``                  → ("lru", {})
    - ``"lru@20"``               → ("lru", {"window_size": 20})   位置参数=窗口大小
    - ``"lru@window=20,max=48"`` → ("lru", {"window_size": 20, "max_window_size": 48})

    键名支持简写别名：``window``/``w`` → ``window_size``；``max``/``cap`` → ``max_window_size``；
    ``protect``/``score`` → ``protect_min_score``（强相关阈值）。
    """
    name = strategy_base_name(spec)
    params: Dict[str, Any] = {}
    if SPEC_SEP not in (spec or ""):
        return name, params
    aliases = {
        "window": "window_size", "w": "window_size", "window_size": "window_size",
        "max": "max_window_size", "cap": "max_window_size",
        "max_window": "max_window_size", "max_window_size": "max_window_size",
        "protect": "protect_min_score", "protect_min_score": "protect_min_score",
        "score": "protect_min_score",
    }
    positional_used = False
    for part in spec.split(SPEC_SEP, 1)[1].split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            params[aliases.get(key.strip(), key.strip())] = _coerce_arg(value)
        elif not positional_used:
            params["window_size"] = _coerce_arg(part)
            positional_used = True
    return name, params


def create_strategy(name: str, manager: Any) -> ToolStrategy:
    """实例化一个已注册策略（支持 ``lru@20`` 形式的参数规格）。

    未注册抛 ValueError；策略不接受所传参数时同样抛 ValueError（便于基准脚本早失败）。
    """
    base, params = parse_strategy_spec(name)
    cls = lookup_strategy(base)
    if cls is None:
        raise ValueError(
            f"未知工具调度策略 '{name}'，可用：{list_strategies()}"
        )
    if not params:
        return cls(manager)
    try:
        return cls(manager, **params)
    except TypeError as e:
        raise ValueError(f"策略 '{base}' 不接受参数 {sorted(params)}: {e}") from e

