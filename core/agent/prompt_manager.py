"""
PromptManager — 插件提示词段注册与注入管理

负责：
- 接收插件注册的 PromptSection（按功能模块拆分的小粒度知识段）
- 提供 Agent 工具查询和注入知识段
- 维护当前对话的注入状态
- 支持根据用户输入自动匹配并注入

架构原则：
- 插件自治：插件自己声明提示词段，不改核心代码
- Agent 驱动：Agent 通过工具自主发现和注入所需知识
- 细粒度：每个 PromptSection 聚焦一个具体功能点，避免大段冗余
- 位置无关：不绑定当前界面，Agent 在任何界面都能获取任意插件的知识
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional
import threading

from core.agent.token_budget import estimate_tokens


@dataclass
class PromptSection:
    """一个独立可注入的提示词知识段

    分层设计：
    - 插件级命名空间：section_id = <plugin_id>.<domain>
    - category 标记知识类型，支持按类型筛选注入
    - keywords 支持语义匹配，用于 auto_inject
    - scenes 标记适用场景，用于按场景筛选注入

    section_id 示例:
      "annotation.concept.coordinates"  — 坐标系统概念
      "annotation.interact.sam"         — SAM 打点操作
      "annotation.concept.data_format"  — 标注数据格式
      "annotation.workflow.sam"         — SAM 交互分割流程
      "dashboard.project.manage"        — 项目管理
      "version.manage"                  — 版本管理

    category 取值:
      "concept"   — 概念说明（坐标系统、数据格式等）
      "guide"     — 操作指南（如何打点、如何绘制等）
      "example"   — 代码示例
      "reference" — 参考信息（参数表、枚举值等）
      "rule"      — 行为规则/约束

    scenes 取值示例:
      "sam"              — SAM 交互分割场景
      "manual_draw"      — 手动绘制标注场景
      "batch"            — 批量/自动标注场景
      "quality_check"    — 质量检查/审核场景
      "export"           — 数据导出场景
      "navigation"       — 图片导航/浏览场景
    """
    section_id: str
    plugin_id: str = ""
    title: str = ""
    description: str = ""
    category: str = "guide"                     # 知识类型：concept/guide/example/reference/rule
    keywords: List[str] = field(default_factory=list)
    scenes: List[str] = field(default_factory=list)  # 适用场景标签列表
    content: str = ""
    priority: int = 0
    is_injected: bool = False


class PromptManager:
    """管理所有插件的提示词段注册与注入状态（线程安全）"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._sections: List[PromptSection] = []
        self._injected_lru: OrderedDict[str, int] = OrderedDict()  # section_id -> token_count

    @classmethod
    def instance(cls) -> PromptManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---- 注册 ----

    def register_section(self, section: PromptSection):
        with self._lock:
            for existing in self._sections:
                if existing.section_id == section.section_id:
                    return
            self._sections.append(section)

    def register_sections(self, sections: List[PromptSection]):
        for s in sections:
            self.register_section(s)

    # ---- 查询 ----

    def get_section(self, section_id: str) -> Optional[PromptSection]:
        for s in self._sections:
            if s.section_id == section_id:
                return s
        return None

    def list_sections(self, keywords: str = "", category: str = "",
                      scene: str = "") -> List[PromptSection]:
        result = list(self._sections)
        if category:
            result = [s for s in result if s.category == category]
        if scene:
            result = [s for s in result if scene in s.scenes]
        if keywords:
            kw_lower = keywords.lower()
            filtered = []
            for s in result:
                for kw in s.keywords:
                    if kw.lower() in kw_lower or kw_lower in kw.lower():
                        filtered.append(s)
                        break
                else:
                    if kw_lower in s.title.lower() or kw_lower in s.description.lower():
                        filtered.append(s)
            result = filtered
        return result

    def list_scenes(self) -> List[str]:
        scenes = set()
        for s in self._sections:
            for sc in s.scenes:
                scenes.add(sc)
        return sorted(scenes)

    def list_categories(self) -> List[str]:
        cats = set()
        for s in self._sections:
            if s.category:
                cats.add(s.category)
        return sorted(cats)

    # ---- 注入管理 ----

    def inject(self, section_id: str) -> Optional[PromptSection]:
        with self._lock:
            section = self.get_section(section_id)
            if section:
                section.is_injected = True
                # 同步加入 LRU 队列最新位置
                token_count = estimate_tokens(section.content)
                self._injected_lru[section_id] = token_count
            return section

    def get_active_sections(self) -> List[PromptSection]:
        return [s for s in self._sections if s.is_injected]

    def get_active_formatted(self) -> str:
        active = self.get_active_sections()
        if not active:
            return ""
        parts = ["# 已注入的插件知识段\n"]
        for s in active:
            parts.append(f"## {s.title}\n")
            parts.append(s.content)
        return "\n\n".join(parts)

    # ---- LRU 生命周期管理 ----

    def touch_section(self, section_id: str):
        """更新指定段在 LRU 队列中的位置到最新（最热）"""
        with self._lock:
            if section_id in self._injected_lru:
                self._injected_lru.move_to_end(section_id)

    def touch_by_user_input(self, user_input: str):
        """根据用户输入文本匹配已注入段并刷新其 LRU 位置

        匹配规则：
        - 用户输入包含段的任一 keyword（不区分大小写）
        - 用户输入包含段的 title
        """
        if not user_input:
            return
        lowered = user_input.lower()
        with self._lock:
            for section in self._sections:
                if not section.is_injected:
                    continue
                if section.section_id not in self._injected_lru:
                    continue
                matched = False
                for kw in section.keywords:
                    if kw.lower() in lowered:
                        matched = True
                        break
                if not matched and section.title and section.title.lower() in lowered:
                    matched = True
                if matched:
                    self._injected_lru.move_to_end(section.section_id)

    def _calculate_eviction_priority(self) -> List[tuple]:
        """计算所有已注入段的淘汰优先级

        priority_score = lru_age_weight * token_count
        - lru_age_weight: 队列第一个（最旧）= 1.0，最后一个（最新）= 0.1，线性插值
        - 返回 [(section_id, priority_score), ...] 按优先级降序排列
        """
        if not self._injected_lru:
            return []
        items = list(self._injected_lru.items())  # [(section_id, token_count), ...] 从旧到新
        n = len(items)
        if n == 1:
            weight = 1.0
        else:
            # 线性插值：第 0 个（最旧）=1.0，第 n-1 个（最新）=0.1
            weight_for = lambda i: 1.0 - (1.0 - 0.1) * (i / (n - 1))
        priorities = []
        for i, (section_id, token_count) in enumerate(items):
            w = weight if n == 1 else weight_for(i)
            priorities.append((section_id, w * token_count))
        # 按优先级降序排列
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def evict_one(self) -> Optional[str]:
        """淘汰优先级最高的段

        - 将对应 section 的 is_injected 置为 False
        - 从 _injected_lru 中删除
        - 返回被淘汰的 section_id；若队列为空返回 None
        """
        with self._lock:
            priorities = self._calculate_eviction_priority()
            if not priorities:
                return None
            section_id = priorities[0][0]
            section = self.get_section(section_id)
            if section:
                section.is_injected = False
            self._injected_lru.pop(section_id, None)
            return section_id

    def evict_until_fit(self, system_prompt_limit: int, base_tokens: int) -> List[str]:
        """循环淘汰段，直到 base_tokens + 已注入 token 总数 <= system_prompt_limit

        Returns:
            被淘汰的 section_id 列表
        """
        evicted = []
        with self._lock:
            # 直接在持锁状态下执行淘汰逻辑（evict_one 内部会再次加锁，故不调用）
            while (self._injected_lru
                   and base_tokens + sum(self._injected_lru.values()) > system_prompt_limit):
                priorities = self._calculate_eviction_priority()
                if not priorities:
                    break
                section_id = priorities[0][0]
                section = self.get_section(section_id)
                if section:
                    section.is_injected = False
                self._injected_lru.pop(section_id, None)
                evicted.append(section_id)
        return evicted

    def evict_until_budget(self, token_budget, min_keep: int = 1) -> List[str]:
        """循环淘汰段，直到 token_budget.should_evict() 返回 False 或达到 min_keep 下限

        Args:
            token_budget: TokenBudget 实例，需提供 should_evict() 与更新接口
            min_keep: 至少保留的段数量

        Returns:
            被淘汰的 section_id 列表
        """
        evicted = []
        with self._lock:
            while (self._injected_lru
                   and len(self._injected_lru) > min_keep
                   and token_budget.should_evict()):
                priorities = self._calculate_eviction_priority()
                if not priorities:
                    break
                section_id = priorities[0][0]
                section = self.get_section(section_id)
                if section:
                    section.is_injected = False
                self._injected_lru.pop(section_id, None)
                evicted.append(section_id)
                # 重新估算并更新 token_budget（如有 update 接口）
                if hasattr(token_budget, "update_injected_tokens"):
                    token_budget.update_injected_tokens(sum(self._injected_lru.values()))
        return evicted

    def get_lru_order(self) -> List[str]:
        """返回 LRU 队列中的 section_id 列表（从最旧到最新）"""
        with self._lock:
            return list(self._injected_lru.keys())

    def get_section_token_count(self, section_id: str) -> int:
        """返回指定段的 token_count（不存在返回 0）"""
        with self._lock:
            return self._injected_lru.get(section_id, 0)

    # ---- 自动注入 ----

    # 场景 → 关键词映射，用于从用户输入推断场景
    _SCENE_KEYWORDS = {
        "sam": ["sam", "打点", "分割", "segment", "交互式"],
        "manual_draw": ["绘制", "画", "矩形", "多边形", "polygon", "矩形框", "draw"],
        "batch": ["全部", "批量", "所有图片", "一键", "自动标注", "全部图片", "批处理"],
        "quality_check": ["检查", "审核", "质量", "验证", "审查", "校验", "漏标"],
        "export": ["导出", "格式", "转换", "coco", "yolo", "voc", "划分"],
        "navigation": ["导航", "切换", "跳转", "下一张", "上一张", "浏览"],
    }

    def _infer_scene(self, user_input: str) -> str:
        """从用户输入推断最可能的使用场景"""
        lowered = user_input.lower()
        best_scene = ""
        best_score = 0
        for scene, kws in self._SCENE_KEYWORDS.items():
            score = sum(1 for kw in kws if kw in lowered)
            if score > best_score:
                best_score = score
                best_scene = scene
        return best_scene

    def auto_inject(self, user_input: str, max_count: int = 3,
                    scene: str = "", system_prompt_limit: int = 2000,
                    base_tokens: int = 0) -> List[str]:
        """根据用户输入自动注入相关的知识段

        Args:
            user_input: 用户输入文本
            max_count: 最多注入数量
            scene: 指定场景（可选），为空则自动推断
            system_prompt_limit: 系统提示词 token 上限
            base_tokens: 基础 token 占用（不含已注入段）
        """
        with self._lock:
            if not scene:
                scene = self._infer_scene(user_input)

            injected = []
            matches = []
            _CATEGORY_WEIGHT = {
                "guide": 1.2,
                "concept": 1.0,
                "example": 0.9,
                "reference": 0.8,
                "rule": 0.7,
            }
            for s in self._sections:
                if s.is_injected:
                    continue
                score = 0

                # 场景匹配加分
                if scene and scene in s.scenes:
                    score += 2

                # 关键词匹配加分
                for kw in s.keywords:
                    if kw.lower() in user_input.lower():
                        score += 1

                if score > 0:
                    cat_w = _CATEGORY_WEIGHT.get(s.category, 1.0)
                    matches.append((score * cat_w * s.priority, s))
            matches.sort(key=lambda x: x[0], reverse=True)
            for _, s in matches[:max_count]:
                new_section_tokens = estimate_tokens(s.content)
                # 注入前检查 token 上限
                if (base_tokens + sum(self._injected_lru.values())
                        + new_section_tokens > system_prompt_limit):
                    # 先尝试淘汰以腾出空间
                    # 直接在持锁状态下执行淘汰逻辑（evict_until_fit 内部会再次加锁）
                    while (self._injected_lru
                           and base_tokens + sum(self._injected_lru.values())
                           + new_section_tokens > system_prompt_limit):
                        priorities = self._calculate_eviction_priority()
                        if not priorities:
                            break
                        evict_id = priorities[0][0]
                        evict_section = self.get_section(evict_id)
                        if evict_section:
                            evict_section.is_injected = False
                        self._injected_lru.pop(evict_id, None)
                    # 淘汰后仍无法容纳则跳过该段
                    if (base_tokens + sum(self._injected_lru.values())
                            + new_section_tokens > system_prompt_limit):
                        continue
                s.is_injected = True
                self._injected_lru[s.section_id] = new_section_tokens
                injected.append(s.section_id)
            return injected

    def reset_injected(self):
        with self._lock:
            for s in self._sections:
                s.is_injected = False
            self._injected_lru.clear()
