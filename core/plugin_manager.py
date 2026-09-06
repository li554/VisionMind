"""
插件管理器，支持插件的物理隔离、动态发现、依赖排序加载和卸载
"""
import os
import json
import sys
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable


class IPlugin(ABC):
    """所有插件必须实现此接口"""

    @staticmethod
    @abstractmethod
    def plugin_id() -> str:
        """插件唯一标识"""

    @staticmethod
    @abstractmethod
    def plugin_name() -> str:
        """插件显示名称"""

    @staticmethod
    @abstractmethod
    def plugin_icon() -> str:
        """FluentIcon 名称"""

    @staticmethod
    def dependencies() -> list:
        """依赖的其他插件ID列表"""
        return []

    @abstractmethod
    def on_load(self, ctx: 'PluginContext'):
        """插件加载时调用"""

    @abstractmethod
    def on_unload(self, ctx: 'PluginContext'):
        """插件卸载时调用"""


class PluginContext:
    """插件运行时上下文，由核心 Shell 提供"""

    # 插件图标字符串 → FluentIcon 映射
    _ICON_MAP = None

    # 默认图标（在类加载时缓存，避免作用域问题）
    _DEFAULT_ICON = None

    @classmethod
    def _resolve_icon(cls, icon):
        """将字符串图标名解析为 FluentIcon 枚举实例"""
        if not isinstance(icon, str):
            return icon  # 已经是 FluentIcon / QIcon 实例

        if cls._ICON_MAP is None:
            from qfluentwidgets import FluentIcon as FIF
            from core.common.icons import AppIcon
            cls._DEFAULT_ICON = FIF.APPLICATION
            cls._ICON_MAP = {
                "HOME": AppIcon.DASHBOARD,
                "TAG": AppIcon.TAG,
                "HISTORY": AppIcon.HISTORY,
                "IMAGE": FIF.PHOTO,
                "VIEW": FIF.VIEW,
                "ROBOT": FIF.ROBOT,
                "RULER": FIF.UNIT,        # RULER 不存在，用 UNIT 替代
                "TEST": FIF.CHECKBOX,
                "FLOW": FIF.SYNC,
                "DIAGNOSE": FIF.SEARCH,   # DIAGNOSE 不存在，用 SEARCH 替代
                "TRAINING": FIF.EDUCATION,
                "SETTING": FIF.SETTING,
                "PEOPLE": FIF.PEOPLE,
                "FOLDER": FIF.FOLDER,
                "EDIT": FIF.EDIT,
                "ADD": FIF.ADD,
                "DELETE": FIF.DELETE,
                "SEARCH": FIF.SEARCH,
                "SAVE": FIF.SAVE,
                "CLOSE": FIF.CLOSE,
                "BRIGHTNESS": FIF.BRIGHTNESS,
            }

        return cls._ICON_MAP.get(icon.upper(), cls._DEFAULT_ICON)

    def __init__(self, main_window, event_bus):
        self._main_window = main_window
        self._event_bus = event_bus
        self._project_service = None
        self._config_cards = []  # [(order, title, widget_class), ...]
        self._state_router = None  # StateRouter 单例，由 main.py 注入

    def register_interface(self, widget, icon, text: str, order: int = 99):
        """注册导航页，同时自动注册界面的 @action 方法，并按域自动生成提示词知识段"""
        resolved_icon = self._resolve_icon(icon)
        self._main_window.add_plugin_interface(widget, resolved_icon, text, order)

        # 自动注册该界面实例的所有 @action 方法到 ActionRegistry
        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        object_name = widget.objectName()
        # 生成前缀：从 objectName 提取（如 "AnnotationInterface" → "annotation"）
        prefix = ""
        if object_name and object_name.endswith("Interface"):
            prefix = object_name[:-9].lower()  # 去掉 "Interface" 后转小写
        elif object_name:
            prefix = object_name.lower()

        registry.register_instance(widget, prefix=prefix)

        # 如果界面有 draw_area 子组件，也注册其 @action 方法
        draw_area = getattr(widget, 'draw_area', None)
        if draw_area is not None:
            registry.register_instance(draw_area, prefix=prefix)

        # 自动从 action 域前缀生成 PromptSection
        self._auto_register_prompt_sections(prefix)

    def register_service(self, service_instance, prefix: str = ""):
        """注册 service 实例的 @action 方法到 ActionRegistry

        应与 register_interface 使用相同的前缀，以便 service 的
        @action(scope="agent") 方法覆盖 interface 的同名 action。
        建议在 register_interface 之前调用，确保 service action 优先注册。

        Args:
            service_instance: Service 层实例
            prefix: action 名称前缀（建议与 register_interface 一致）
        """
        from core.common.action_registry import ActionRegistry
        registry = ActionRegistry.instance()
        registry.register_instance(service_instance, prefix=prefix)

    def _auto_register_prompt_sections(self, prefix: str):
        """按 action 名的域前缀自动生成提示词知识段

        将同一域前缀下的所有 action 的描述聚合为一个 PromptSection。
        域前缀即 action 名的第一个点之前的部分（如 "interact.add_sam_point" 的域是 "interact"）。
        """
        from core.common.action_registry import ActionRegistry
        from core.agent.prompt_manager import PromptManager, PromptSection

        registry = ActionRegistry.instance()

        # 按域分组：domain -> [(ActionMeta, full_name)]
        domain_actions = {}
        for full_name, (_, _, meta) in registry._actions.items():
            if not full_name.startswith(f"{prefix}."):
                continue
            if "." not in meta.name:
                continue  # 无域前缀的 action 跳过
            if getattr(meta, "scope", "ui") != "agent":
                continue  # 仅将 agent 可调工具的描述注入知识段，避免诱导 agent 调 scope=ui 的界面/录制 action
            domain = meta.name.split(".")[0]
            if domain not in domain_actions:
                domain_actions[domain] = []
            domain_actions[domain].append((meta, full_name))

        pm = PromptManager.instance()

        # 域显示名映射
        DOMAIN_TITLES = {
            "interact": "画布交互操作",
            "edit": "标注编辑操作",
            "nav": "图片导航操作",
            "batch": "批量自动标注",
            "resource": "辅助资源管理",
            "review": "质量审查",
            "category": "类别管理",
            "misc": "其他操作",
        }

        # 域关键词映射（每个域的核心关键词）
        DOMAIN_KEYWORDS = {
            "interact": ["交互", "绘制", "SAM", "打点", "视图", "缩放", "画布"],
            "edit": ["编辑", "删除", "撤销", "修改", "保存", "标注"],
            "nav": ["导航", "翻图", "切换", "文件", "项目"],
            "batch": ["批量", "一键标注", "自动标注", "全部图片"],
            "resource": ["模型", "示例", "提示词", "规则", "资源"],
            "review": ["筛选", "审查", "过滤", "检查"],
            "category": ["类别", "分类", "标签"],
        }

        # 域场景映射
        DOMAIN_SCENES = {
            "interact": ["sam", "manual_draw"],
            "edit": ["manual_draw", "quality_check"],
            "nav": ["navigation"],
            "batch": ["batch"],
            "resource": ["sam", "batch"],
            "review": ["quality_check"],
            "category": ["manual_draw", "batch", "quality_check"],
        }

        for domain, metas in domain_actions.items():
            section_id = f"{prefix}.{domain}"
            if pm.get_section(section_id):
                continue  # 已手动注册过则跳过

            title = DOMAIN_TITLES.get(domain, domain)

            # 聚合内容：每个 action 的描述即为提示词
            content_parts = [f"## {title}\n"]
            for meta, full_name in metas:
                # 与工具面 display_name 保持一致（去掉插件前缀、保留域前缀），如 "annotation.resource.list_examples" → "resource.list_examples"
                action_short = full_name.split(".", 1)[1] if "." in full_name else full_name
                content_parts.append(f"### {action_short}")
                content_parts.append(meta.description)

            content = "\n\n".join(content_parts)

            # 合并关键词
            keywords = list(DOMAIN_KEYWORDS.get(domain, []))
            for meta, full_name in metas:
                action_short = full_name.split(".", 1)[1] if "." in full_name else full_name
                keywords.append(action_short)

            section = PromptSection(
                section_id=section_id,
                plugin_id=prefix,
                title=title,
                description=f"{title}相关功能说明",
                keywords=keywords,
                scenes=DOMAIN_SCENES.get(domain, []),
                content=content,
                priority=6,
            )
            pm.register_section(section)

    def remove_interface(self, widget):
        """移除导航页"""
        self._main_window.remove_plugin_interface(widget)

    def register_config_card(self, title: str, widget_class, order: int = 100):
        """注册配置卡片"""
        self._config_cards.append((order, title, widget_class))

    def get_config_cards(self) -> list:
        """返回排序后的配置卡片列表"""
        return sorted(self._config_cards, key=lambda x: x[0])

    @property
    def project_service(self):
        """核心项目服务"""
        return self._project_service

    @project_service.setter
    def project_service(self, svc):
        self._project_service = svc

    @property
    def event_bus(self):
        """事件总线"""
        return self._event_bus

    def subscribe(self, event: str, callback: Callable):
        """订阅事件"""
        self._event_bus.subscribe(event, callback)

    def publish(self, event: str, data: Any = None):
        """发布事件"""
        self._event_bus.publish(event, data)

    # ---- 统一状态路由（StateRouter）接入 ----

    def set_state_router(self, router):
        """注入全局 StateRouter（由 main.py 调用）。"""
        self._state_router = router

    def register_state_handlers(self, mapping: dict):
        """新插件声明响应的 type -> 处理器，接入统一状态信号机制。

        用法（方案 A，type 空间开放）：
            ctx.register_state_handlers({
                "project_selected":    self.on_project_changed,
                "annotations_changed": self.interface.on_annotations_changed,
            })
        """
        if self._state_router is None:
            return
        self._state_router.register_many(mapping)

    def unregister_state_handlers(self, mapping: dict = None):
        """卸载插件注册的处理器（on_unload 清理，幂等）。mapping 为空时清除全部。"""
        if self._state_router is None:
            return
        if not mapping:
            self._state_router.unregister()
            return
        for t, h in mapping.items():
            self._state_router.unregister(t, h)

    def publish_state(self, type_: str, data: Any = None, extra: dict = None):
        """广播统一状态事件（service/interface 层均可）。"""
        self._event_bus.publish_state(type_, data, extra)

    def switch_to(self, widget):
        """切换到指定插件界面"""
        self._main_window.switchTo(widget)

    # ---- 提示词段注册 ----

    def register_prompt_section(self, section_id: str, title: str, content: str,
                                 description: str = "", category: str = "guide",
                                 keywords: list = None, scenes: list = None,
                                 priority: int = 0):
        """注册插件的提示词知识段，供 Agent 按需注入

        Args:
            section_id: 知识段 ID（建议格式 <plugin_id>.<domain>.<subdomain>，如 "annotation.interact.sam"）
            title: 知识段标题
            content: 知识段内容（Markdown 文本）
            description: 简短说明（用于搜索预览）
            category: 知识类型（concept/guide/example/reference/rule），默认 guide
            keywords: 关键词列表（用于搜索和自动匹配）
            scenes: 适用场景列表（如 sam/manual_draw/batch/quality_check/export）
            priority: 注入优先级（数值越高越优先）
        """
        from core.agent.prompt_manager import PromptManager, PromptSection
        section = PromptSection(
            section_id=section_id,
            plugin_id="",
            title=title,
            description=description,
            category=category,
            keywords=keywords or [],
            scenes=scenes or [],
            content=content,
            priority=priority,
        )
        # 自动从 section_id 提取 plugin_id（取第一个点之前的部分）
        if "." in section_id:
            section.plugin_id = section_id.split(".")[0]
        PromptManager.instance().register_section(section)


class PluginManager:
    """
    插件管理器

    加载规则：
    - 文件夹存在 + plugins.json 中 true  → 加载
    - 文件夹存在 + plugins.json 中 false → 忽略
    - 文件夹不存在                      → 忽略（不报错）
    - 新发现文件夹（plugins.json 无记录） → 默认 true，自动追加
    """

    def __init__(self, plugins_dir: str, config_path: str, context: PluginContext):
        self.plugins_dir = plugins_dir
        self.config_path = config_path
        self.context = context
        self.plugins: Dict[str, IPlugin] = {}
        self.plugin_meta: Dict[str, dict] = {}
        self.config: Dict[str, bool] = {}

    def load_config(self):
        """加载 plugins.json"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8-sig') as f:
                    self.config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[PluginManager] Failed to load config: {e}")
                self.config = {}
        else:
            self.config = {}

    def save_config(self):
        """保存配置到磁盘"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def discover_plugins(self):
        """
        扫描 plugins/ 目录，发现所有 plugin.json

        规则：
        - 文件夹存在 + config 未记录 → 默认 true，追加到 config
        - 文件夹存在 + config 为 true → 候选加载
        - 文件夹存在 + config 为 false → 跳过
        - 文件夹不存在 → 跳过
        """
        changed = False

        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir, exist_ok=True)
            return

        for folder_name in sorted(os.listdir(self.plugins_dir)):
            folder_path = os.path.join(self.plugins_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # 跳过 __pycache__ 等特殊目录
            if folder_name.startswith('_') or folder_name.startswith('.'):
                continue

            meta_path = os.path.join(folder_path, "plugin.json")
            if not os.path.exists(meta_path):
                continue

            # 新发现的插件，默认启用
            if folder_name not in self.config:
                self.config[folder_name] = True
                changed = True

            # 配置中禁用的，跳过
            if not self.config.get(folder_name, True):
                print(f"[PluginManager] 插件 {folder_name} 已禁用，跳过")
                continue

            # 读取元数据
            try:
                with open(meta_path, 'r', encoding='utf-8-sig') as f:
                    meta = json.load(f)
                meta['_folder_path'] = folder_path
                self.plugin_meta[folder_name] = meta
            except Exception as e:
                print(f"[PluginManager] 读取 {meta_path} 失败: {e}")

        if changed:
            self.save_config()

    def load_all(self):
        """按依赖顺序加载所有已启用的插件"""
        load_order = self._resolve_order()

        for plugin_id in load_order:
            self._load_plugin(plugin_id)

    def _resolve_order(self) -> List[str]:
        """拓扑排序，保证依赖先加载"""
        result = []
        visited = set()
        visiting = set()

        def visit(pid):
            if pid in visited:
                return
            if pid in visiting:
                print(f"[PluginManager] 循环依赖检测: {pid}")
                return
            visiting.add(pid)

            meta = self.plugin_meta.get(pid)
            if meta:
                for dep in meta.get("dependencies", []):
                    if dep in self.plugin_meta:
                        visit(dep)

            visiting.discard(pid)
            visited.add(pid)
            result.append(pid)

        for pid in self.plugin_meta:
            visit(pid)

        return result

    def _load_plugin(self, plugin_id: str):
        """动态加载单个插件"""
        meta = self.plugin_meta.get(plugin_id)
        if not meta:
            return

        folder_path = meta['_folder_path']
        entry = meta.get('entry', 'plugin.py')

        # 检查依赖是否已加载
        for dep in meta.get('dependencies', []):
            if dep not in self.plugins:
                print(f"[PluginManager] 插件 {plugin_id} 的依赖 {dep} 未加载，跳过")
                return

        # 动态 import
        try:
            # 解析 entry
            if ':' in entry:
                file_name, class_name = entry.split(':', 1)
            else:
                file_name, class_name = entry, None

            py_path = os.path.join(folder_path, file_name)
            if not os.path.exists(py_path):
                print(f"[PluginManager] 插件 {plugin_id} 入口文件不存在: {py_path}")
                return

            # 确保 plugins 包和插件子包在 sys.modules 中注册
            # 这样插件内的相对导入 (from .interfaces.xxx) 才能正常工作
            plugins_pkg_name = "plugins"
            plugin_pkg_name = f"plugins.{plugin_id}"
            
            if plugins_pkg_name not in sys.modules:
                import types
                plugins_pkg = types.ModuleType(plugins_pkg_name)
                plugins_pkg.__path__ = [self.plugins_dir]
                plugins_pkg.__package__ = plugins_pkg_name
                sys.modules[plugins_pkg_name] = plugins_pkg
            
            if plugin_pkg_name not in sys.modules:
                import types
                plugin_pkg = types.ModuleType(plugin_pkg_name)
                plugin_pkg.__path__ = [folder_path]
                plugin_pkg.__package__ = plugin_pkg_name
                sys.modules[plugin_pkg_name] = plugin_pkg

            module_name = f"plugins.{plugin_id}.{os.path.splitext(file_name)[0]}"
            spec = importlib.util.spec_from_file_location(module_name, py_path)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = plugin_pkg_name
            sys.modules[module_name] = module

            spec.loader.exec_module(module)

            # 找到入口类
            if class_name:
                plugin_class = getattr(module, class_name, None)
                if plugin_class is None:
                    print(f"[PluginManager] 插件 {plugin_id} 入口类 {class_name} 未找到")
                    return
            else:
                # 自动查找 IPlugin 子类
                plugin_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type)
                        and issubclass(attr, IPlugin)
                        and attr is not IPlugin):
                        plugin_class = attr
                        break
                if plugin_class is None:
                    print(f"[PluginManager] 插件 {plugin_id} 未找到 IPlugin 子类")
                    return

            # 实例化并加载
            instance = plugin_class()
            instance.on_load(self.context)
            self.plugins[plugin_id] = instance

            # 加载插件 AGENTS.md 知识段
            agents_md = os.path.join(folder_path, "AGENTS.md")
            if os.path.exists(agents_md):
                from core.agent.prompt_manager import PromptManager, PromptSection
                content = open(agents_md, "r", encoding="utf-8").read()
                section = PromptSection(
                    section_id=plugin_id,
                    plugin_id=plugin_id,
                    title=meta.get("name", plugin_id),
                    description=meta.get("description", ""),
                    category="guide",
                    keywords=["knowledge", plugin_id],
                    content=content,
                    priority=4,
                )
                PromptManager.instance().register_section(section)

            print(f"[PluginManager] 插件 {plugin_id} ({instance.plugin_name()}) 加载成功")

        except Exception as e:
            print(f"[PluginManager] 插件 {plugin_id} 加载失败: {e}")
            import traceback
            traceback.print_exc()

    def unload_plugin(self, plugin_id: str):
        """卸载插件"""
        # 检查是否有其他插件依赖它
        for pid, meta in self.plugin_meta.items():
            if pid in self.plugins and plugin_id in meta.get('dependencies', []):
                print(f"[PluginManager] 插件 {pid} 依赖 {plugin_id}，无法卸载")
                return

        plugin = self.plugins.get(plugin_id)
        if plugin:
            try:
                plugin.on_unload(self.context)
            except Exception as e:
                print(f"[PluginManager] 插件 {plugin_id} 卸载失败: {e}")
            del self.plugins[plugin_id]
            print(f"[PluginManager] 插件 {plugin_id} 已卸载")

    def enable_plugin(self, plugin_id: str):
        """启用插件（下次启动生效）"""
        self.config[plugin_id] = True
        self.save_config()

    def disable_plugin(self, plugin_id: str):
        """禁用插件（下次启动生效）"""
        self.config[plugin_id] = False
        self.save_config()

    def get_plugin_list(self) -> List[dict]:
        """获取所有插件的启用状态（供系统配置界面显示）"""
        result = []

        if not os.path.exists(self.plugins_dir):
            return result

        for folder_name in sorted(os.listdir(self.plugins_dir)):
            folder_path = os.path.join(self.plugins_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.startswith('_') or folder_name.startswith('.'):
                continue

            meta_path = os.path.join(folder_path, "plugin.json")
            if not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, 'r', encoding='utf-8-sig') as f:
                    meta = json.load(f)
                result.append({
                    "id": folder_name,
                    "name": meta.get("name", folder_name),
                    "version": meta.get("version", "?"),
                    "description": meta.get("description", ""),
                    "icon": meta.get("icon", ""),
                    "order": meta.get("order", 99),
                    "enabled": self.config.get(folder_name, True),
                    "loaded": folder_name in self.plugins,
                    "dependencies": meta.get("dependencies", [])
                })
            except Exception:
                pass

        return sorted(result, key=lambda x: x.get("order", 99))
