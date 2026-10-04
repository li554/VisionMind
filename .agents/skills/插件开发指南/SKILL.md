---
name: 插件开发指南
description: 指导如何创建新插件、理解插件架构、使用 PluginContext、@action 装饰器（scope/params/工具化）、领域服务注册、Prompt.md 与提示词知识段、EventBus 通信、配置卡片，以及插件与核心/Agent 的交互方式。
---

# 插件开发指南

VisionMind 的插件系统基于物理隔离的目录结构、依赖拓扑排序加载、事件总线通信。
插件不仅向 UI 注册界面，还同时向 **Agent（AI 智能体）** 暴露能力：`@action` 方法即 LLM 工具，`Prompt.md` 即 Agent 的领域说明书。

---

## 1. 插件目录结构

每个插件是一个独立文件夹，放在 `plugins/` 下：

```
plugins/<plugin_id>/
├── plugin.json          # 元数据
├── plugin.py            # IPlugin 实现
├── Prompt.md            # ★ 必备：插件级 Agent 领域说明书（见第 5 节）
├── interfaces/
│   └── <id>_interface.py  # UI 界面（Interface QFrame 子类）
├── services/            # 领域服务（@action(scope="agent") 的承载层）
│   └── *_service.py
├── settings_card.py     # （可选）配置页面
└── ...                  # 其他文件
```

### plugin.json

```json
{
    "id": "annotation",
    "name": "手动标注",
    "version": "1.0.0",
    "description": "图像手动标注与交互式分割",
    "icon": "TAG",
    "order": 2,
    "dependencies": ["dashboard"],
    "entry": "plugin.py:AnnotationPlugin"
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | √ | 插件唯一标识，也是目录名 |
| `name` | √ | 显示名称 |
| `version` | | 语义化版本号 |
| `description` | | 功能描述 |
| `icon` | √ | 图标名称字符串（见下方图标清单） |
| `order` | √ | 导航栏排序序号，越小越靠前 |
| `dependencies` | | 依赖的插件 ID 列表，依赖者后加载 |
| `entry` | √ | `文件名:类名`，如 `"plugin.py:AnnotationPlugin"` |

**可用图标字符串**（`PluginContext._resolve_icon` 解析）：`HOME`, `TAG`, `HISTORY`, `IMAGE`, `VIEW`, `ROBOT`, `RULER`, `TEST`, `FLOW`, `DIAGNOSE`, `TRAINING`, `SETTING`, `PEOPLE`, `FOLDER`, `EDIT`, `ADD`, `DELETE`, `SEARCH`, `SAVE`, `CLOSE`, `BRIGHTNESS`。
其中 `HOME`/`TAG`/`HISTORY` 已映射为项目自绘的主题化图标（`core/common/icons.py` 的 `AppIcon.DASHBOARD/TAG/HISTORY`，随明暗主题自动着色），其余解析为 FluentIcon。

### IPlugin 接口

`core/plugin_manager.py` 中定义的抽象基类：

```python
from core.plugin_manager import IPlugin, PluginContext

class MyPlugin(IPlugin):
    @staticmethod
    def plugin_id():      return "my_plugin"
    @staticmethod
    def plugin_name():    return "我的插件"
    @staticmethod
    def plugin_icon():    return "TAG"
    @staticmethod
    def dependencies():   return ["dashboard"]

    def on_load(self, ctx: PluginContext):
        # 1. 创建领域服务与界面
        # 2. 先注册 service（Agent action），再注册界面（UI action）
        # 3. 注册到导航栏
        # 4. 注册配置卡片（可选）/ 订阅事件（可选）
        ...

    def on_unload(self, ctx: PluginContext):
        # 清理：移除界面、断开信号
        ...
```

---

## 2. PluginContext API

`PluginContext` 是插件与核心交互的唯一入口，通过 `on_load(ctx)` 获取。

| 方法/属性 | 说明 |
|------|------|
| `register_service(service_instance, prefix)` | 注册领域服务的 `@action` 方法（Agent 能力面），**应在 `register_interface` 之前调用**，且前缀一致 |
| `register_interface(widget, icon, text, order)` | 注册导航页面，自动注册界面（及 `draw_area`）的 `@action` 方法，并按域自动生成提示词知识段 |
| `remove_interface(widget)` | 移除导航页面 |
| `register_config_card(title, widget_class, order)` | 注册配置卡片 |
| `get_config_cards()` | 获取所有配置卡片（排序后） |
| `register_prompt_section(section_id, title, content, ...)` | 手动注册 Agent 提示词知识段（见第 6 节） |
| `subscribe(event, callback)` | 订阅事件总线事件 |
| `publish(event, data)` | 发布事件到事件总线 |
| `switch_to(widget)` | 切换到指定插件界面 |
| `project_service` | 项目服务实例 |
| `event_bus` | 事件总线实例 |

---

## 3. Action 分层与 @action 装饰器（核心）

### 3.1 两层结构与 scope

VisionMind 的可调用操作分两层，`scope` 决定暴露对象：

| 层 | scope | 承载 | 用途 |
|----|-------|------|------|
| **领域服务层** | `scope="agent"` | `services/*_service.py`（如 `manual`/`auto`） | Agent（LLM）直接调用的核心逻辑，纯数据操作、无 UI 依赖 |
| **界面层** | `scope="ui"` | `interfaces/*_interface.py` | 工程回放（TestEngine）与 UI 入口；常见 `*_ui` 后缀包装（内部调用 service action 后刷新界面） |

> Agent 只看到 `scope="agent"` 的 action；`scope="ui"` 对 Agent 不可见。

### 3.2 装饰器签名（`core/common/action_registry.py`）

```python
def action(name: str, description: str = "", category: str = "通用",
           scope: str = "ui", params: Dict = None):
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `name` | √ | action 名称，**全局唯一**。建议带**域前缀**（`域.动作`，如 `interact.add_sam_point`、`nav.select_image`） |
| `description` | | 功能描述。**Agent 的工具说明直接取自这里**，请写清前提条件、参数枚举含义、坐标约定等（支持 `\n` 多行） |
| `category` | | 分类标签（如 `"标注"`、`"画布"`、`"导航"`、`"自动标注"`），用于 Agent 工具分组与工具面显示 |
| `scope` | | `"agent"`=AI 可调用；`"ui"`=仅供工程回放（默认） |
| `params` | | 参数 schema（见 3.4），生成 OpenAI Function Calling 格式 |

### 3.3 命名与自动注册

- `register_interface` 从 `widget.objectName()` 生成前缀：`AnnotationInterface` → `annotation`
- `meta.name` 已含插件前缀时不重复添加；最终注册名 = `插件前缀.meta.name`，如 `annotation.interact.add_sam_point`
- `draw_area` 子组件的 `@action` 方法也以同一前缀注册
- 调用时支持**后缀匹配**（`toggle_tool` → `annotation.toggle_tool`）；多候选时优先解析 `scope="ui"` 的 action
- `@action` 包装器自动做两件事：录制（TestEngine 回放，嵌套调用只录最外层）+ OperationLogger 状态快照（供意图分析）

### 3.4 params 的四种格式（生成 Tool Schema）

```python
# 格式1：简单类型字符串
params={"direction": "int"}

# 格式2：带描述
params={"role": {"type": "str", "description": "模型角色：interactive/auto/plain/refine"}}

# 格式3：静态 enum
params={"mode": {"type": "str", "enum": ["seg", "det", "obb"],
                 "description": "任务模式: seg=分割, det=检测, obb=旋转框"}}

# 格式4：动态 enum（运行时由 DynamicToolManager 解析）
params={"image_path": {"type": "str", "dynamic_enum": "image_paths"}}

# 必填标记：dict 中 "required": True → 进入 schema 的 required 数组
params={"mode": {"type": "str", "required": True}}
```

`description` 会进入 Tool Schema，`name` 全局唯一且调用名带插件前缀（如 `annotation.manual.get_state`）。

---

## 4. 领域服务注册（register_service）

Agent 能力应放在 **service 层**（纯逻辑、可测试、无 UI 耦合），通过 `register_service` 注册：

```python
def on_load(self, ctx: PluginContext):
    from .interfaces.annotation_interface import AnnotationInterface

    self.interface = AnnotationInterface(ctx.project_service, None)
    # 先注册领域服务，让 @action(scope="agent") 优先注册，
    # 从而覆盖 interface 层的同名 action（Agent 拿到的是纯逻辑版本）
    ctx.register_service(self.interface.manual, prefix="annotation")
    ctx.register_service(self.interface.auto, prefix="annotation")
    ctx.register_interface(self.interface, "TAG", "手动标注", order=2)
```

要点：
- `register_service` 与 `register_interface` 使用**相同前缀**；先注册者的同名 action 优先生效
- interface 层的同名 `*_ui` 包装 action（`scope="ui"）保留给 TestEngine 与菜单/快捷键使用
- 注册时自动打印 `[ActionRegistry] 注册 N 个 action 来自 <类名>`，可用于确认

---

## 5. Prompt.md（★ 必备）

每个插件必须提供 `Prompt.md` —— 这是 **Agent 的领域说明书**（运行时注入 Agent 的知识段由 `@action` 描述自动聚合与 `register_prompt_section` 生成，Prompt.md 是这些规则的权威源文档与约定载体），与 `core/Prompt.md`（全局基础提示词）配合工作。

格式参照 `plugins/annotation/Prompt.md`：

```markdown
# <plugin_id> — <插件名>

## 角色与定位
你是 <X> 插件的操作指南，负责……（一句话说明领域）

## 能力范围
- **分组名**：能力列表……

## 行为规则
1. **规则名**：约束与顺序要求（如"必须先 set_drawing_mode 再绘制"）……
   - 细则、反例、易错点（Agent 依据这些规则避免误操作）

## 核心概念与数据格式
### 坐标系统 / 任务模式 / 数据格式
（Agent 正确调用所需的概念与格式约定，含代码示例）
```

写作要求：
- **行为规则**要写"负面清单"（明确禁止什么），Agent 对反例的遵守远好于正面描述
- 数据格式给出 JSON 示例；枚举值给出完整含义映射
- 规则条目要可直接执行、可判定，避免空泛描述

此外还可用代码注册更细粒度的知识段（供 Agent 按需注入）：

```python
ctx.register_prompt_section(
    section_id="annotation.interact.sam",   # 建议格式 <plugin_id>.<domain>.<subdomain>
    title="SAM 交互分割详解",
    content="……（Markdown）",
    category="guide",                        # concept/guide/example/reference/rule
    keywords=["SAM", "打点"],
    scenes=["sam"],
    priority=6,
)
```

`register_interface` 还会**自动**按 action 名的域前缀（`interact/edit/nav/batch/resource/review/category/misc`）聚合生成 PromptSection；已手动注册过同 ID 段则跳过。Agent 侧通过 `ListPromptSectionsTool` 列出可用知识段。

---

## 6. 创建插件界面

### Interface 基类

定义在 `core/common/widgets/base.py`：

```python
from core.common.widgets.base import Interface

class MyInterface(Interface):
    def __init__(self, project_service=None, parent=None):
        super().__init__('MyInterface', parent)  # objectName = "MyInterface"
        self._init_state()
        self.init_ui()
        self.connect_signals()
```

**关键约定：**
- `objectName` 必须以 `Interface` 结尾（如 `"MyInterface"`），用于 ActionRegistry 前缀生成
- 构造函数接收 `project_service=None` 和 `parent=None`
- 如果实现 `set_project(project_info)` 方法，会在项目切换时自动被调用

---

## 7. 事件总线通信

```python
def on_load(self, ctx: PluginContext):
    ctx.subscribe("project:selected", self.on_project_changed)

def on_project_changed(self, project_info):
    if hasattr(self.interface, 'set_project'):
        self.interface.set_project(project_info)
```

发布事件：`ctx.publish("project:selected", project_info)`。

---

## 8. 配置卡片 / 导航切换 / 加载流程

配置卡片：

```python
ctx.register_config_card("我的设置", MySettingsCard, order=30)
```

导航与切换：

```python
ctx.switch_to(widget)              # 通过 PluginContext
main_window.switchTo(widget)       # 直接通过 MainWindow
```

加载流程：

```
main.py
  └─ PluginManager(plugins_dir, config, context)
       ├─ load_config()              # 读取 core/plugins.json
       ├─ discover_plugins()         # 扫描 plugins/ 目录（新目录默认启用）
       └─ load_all()                 # 依赖拓扑排序后逐个加载
            └─ _load_plugin(id)      # 动态导入 entry 模块
                 └─ IPlugin.on_load(ctx)
                      ├─ ctx.register_service()        # Agent action（先）
                      ├─ ctx.register_interface()      # UI action + 自动提示词段
                      ├─ ctx.register_config_card()
                      ├─ ctx.register_prompt_section()（可选）
                      └─ ctx.subscribe()
```

插件启停由 `core/plugins.json` 控制（新发现目录默认 `true`；置 `false` 跳过加载）。

---

## 9. 完整示例：创建新插件

### 目录

```
plugins/my_plugin/
├── plugin.json
├── Prompt.md          # ★ Agent 领域说明书
├── plugin.py
├── interfaces/
│   └── my_interface.py
├── services/
│   └── my_service.py  # 领域服务（Agent 能力）
└── settings_card.py   # 可选
```

### plugin.json

```json
{
    "id": "my_plugin",
    "name": "我的插件",
    "version": "1.0.0",
    "description": "示例插件",
    "icon": "TAG",
    "order": 15,
    "dependencies": ["dashboard"],
    "entry": "plugin.py:MyPlugin"
}
```

### plugin.py

```python
from core.plugin_manager import IPlugin, PluginContext


class MyPlugin(IPlugin):
    @staticmethod
    def plugin_id(): return "my_plugin"

    @staticmethod
    def plugin_name(): return "我的插件"

    @staticmethod
    def plugin_icon(): return "TAG"

    @staticmethod
    def dependencies(): return ["dashboard"]

    def on_load(self, ctx: PluginContext):
        from .interfaces.my_interface import MyInterface
        from .services.my_service import MyService

        self.service = MyService()
        self.interface = MyInterface(ctx.project_service)
        # 先 service（Agent action 优先），后 interface（UI action + 界面）
        ctx.register_service(self.service, prefix="my")
        ctx.register_interface(self.interface, "TAG", "我的插件", order=15)
        ctx.subscribe("project:selected", self.on_project_changed)

    def on_unload(self, ctx: PluginContext):
        ctx.remove_interface(self.interface)
        self.interface = None

    def on_project_changed(self, project_info):
        if hasattr(self.interface, 'set_project'):
            self.interface.set_project(project_info)
```

### services/my_service.py（Agent 能力层）

```python
from core.common.action_registry import action


class MyService:
    @action("data.process", description="处理一条数据。\n- data_id：数据唯一标识（必填）\n- mode：处理模式，fast=快速 / full=完整", category="数据处理",
            params={"data_id": {"type": "str", "required": True},
                    "mode": {"type": "str", "enum": ["fast", "full"]}},
            scope="agent")
    def process(self, data_id: str = "", mode: str = "fast"):
        # 纯逻辑，返回结构化结果（dict），Agent 直接消费
        return {"status": "ok", "data_id": data_id, "mode": mode}
```

### Prompt.md

```markdown
# my_plugin — 我的插件

## 角色与定位
你负责……（领域一句话）

## 能力范围
- **数据处理**：process 数据处理（fast/full 两种模式）

## 行为规则
1. **模式选择**：用户未说明时默认 fast；要求完整分析时用 full
```

### interfaces/my_interface.py（UI 层）

```python
from PySide6.QtWidgets import QVBoxLayout
from qfluentwidgets import BodyLabel
from core.common.widgets.base import Interface
from core.common.action_registry import action


class MyInterface(Interface):
    def __init__(self, project_service=None, parent=None):
        super().__init__('MyInterface', parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(BodyLabel("Hello from MyPlugin"))

    @action("my.run_ui", description="UI 包装：调用 service 后刷新界面（供菜单/快捷键/回放）",
            category="数据处理", scope="ui")
    def run_ui(self):
        ...
```

---

## 10. 现有插件清单

| ID | objectName | 导航名 | 依赖 | 配置卡片 |
|----|-----------|--------|------|---------|
| dashboard | `DashboardInterface` | 项目大厅 | 无 | 无 |
| annotation | `AnnotationInterface` | 手动标注 | dashboard | 标注设置 (order=20) |
| version | `VersionInterface` | 版本管理 | dashboard | 版本设置 (order=25) |
| *(built-in)* | `ConfigInterface` | 系统配置 | — | 由各插件注册 |

> generation / training / unsupervised / inference / flow / kachi / testing 等插件已移除（`core/plugins.json` 中未启用，无对应目录）。
> **新增或修改插件后，务必更新上表以保持文档准确。**
