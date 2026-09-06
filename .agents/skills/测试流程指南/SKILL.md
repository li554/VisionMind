---
name: 测试流程指南
description: 指导如何创建 @action 装饰器方法 和 编写 JSON 测试流程配置。包括 ActionRegistry 注册机制、装饰器参数说明、测试场景/步骤结构、断言用法、变量/表达式、最佳实践。
---

# 测试流程指南

指导创建 `@action` 方法（可被自动化测试调用的操作单元）和编写 JSON 测试配置文件。

---

## 1. 创建 Action

### 1.1 `@action` 装饰器

定义在 `core/common/action_registry.py`：

```python
def action(name: str, description: str = "", category: str = "通用",
           scope: str = "ui", params: Dict = None):
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `name` | √ | action 名称，**全局唯一**。建议用 `snake_case`，并带**域前缀**（`域.动作`，如 `interact.add_sam_point`） |
| `description` | | 功能描述。**Agent 的工具说明直接取自这里**，请写清前提条件、参数枚举含义与坐标约定（支持 `\n` 多行） |
| `category` | | 分类标签（如 `"标注"`、`"画布"`、`"导航"`），用于 Agent 工具分组 |
| `scope` | | `"agent"`=AI 可调用（Agent 工具面只暴露这些）；`"ui"`=仅供工程回放与 UI 入口（默认），Agent 不可见 |
| `params` | | 参数 schema，支持四种格式（见 1.1.1），用于生成 OpenAI Function Calling 工具描述 |

#### 1.1.1 params 的四种格式

```python
# 格式1：简单类型字符串
params={"direction": "int"}

# 格式2：带描述
params={"role": {"type": "str", "description": "模型角色：interactive/auto/plain/refine"}}

# 格式3：静态 enum
params={"mode": {"type": "str", "enum": ["seg", "det", "obb"]}}

# 格式4：动态 enum（运行时由 DynamicToolManager 解析）
params={"image_path": {"type": "str", "dynamic_enum": "image_paths"}}

# 必填标记：dict 中 "required": True → 进入工具 schema 的 required 数组
params={"mode": {"type": "str", "required": True}}
```

`get_tool_schema()` 据此生成 OpenAI Function Calling 格式的工具描述（`description` 一并带入）。

### 1.2 基本结构

```python
from core.common.action_registry import action

class MyActions:
    @action("my_action", description="我的操作", category="通用",
            params={"param1": "str", "param2": "int"})
    def my_action(self, param1: str = "", param2: int = 0):
        """
        实现逻辑...
        """
        # 失败时返回 False，成功时返回 True
        if something_wrong:
            return False
        return True
```

**关键规则：**

- 方法必须属于某个类的实例方法（`self` 为第一参数）
- **返回值建议为 `bool`**：`True` = 成功，`False` = 失败，测试引擎据此判定步骤是否通过。返回 `None` 视为成功
- 参数建议提供默认值，方便从 JSON 配置中缺省调用
- 方法名不限（和 `name` 无关），但建议与 action name 一致或相近

### 1.3 Action 注册

#### 自动注册（推荐）

`plugin_manager.py` 在 `register_interface()` 中自动扫描所有继承 QWidget 的插件界面：

```python
# 自动注册界面类的 @action 方法，前缀 = objectName 去掉 "Interface" 后小写
# 例如 AnnotationInterface → 前缀 "annotation"
# 同时注册该界面的 draw_area 子组件
registry.register_instance(widget, prefix=prefix)
registry.register_instance(draw_area, prefix=prefix)
```

所以你在 `AnnotationInterface` 中写 `@action("select_image")`，最终注册名为 `annotation.select_image`。

#### 手动注册

在 `core/main.py` 中：

```python
registry = ActionRegistry.instance()
registry.register_instance(BuiltinActions())                     # 无前缀 → "wait", "assert", ...
registry.register_instance(InputSimulator(main_window), prefix="input")  # 前缀 "input" → "input.mouse", ...
registry.register_instance(main_window, prefix="main")           # → "main.switch_to"
```

### 1.4 别名机制

测试配置中通常使用**无前缀短名称**（如 `"select_image"`）。`TestEngine._register_aliases()` 自动为 `prefix.name` 创建别名 `name`：

```python
# annotation.select_image → 别名 select_image
# input.mouse         → 别名 mouse
```

所有 `@action` 方法的注册名和别名可以通过在运行时打印 `registry._actions` 和 `registry._aliases` 查看。

### 1.5 完整示例

**真实示例** — `drawing_widget.py` 中的 action：

```python
class DrawingWidget(QWidget):
    # ... 继承自 QWidget，由 plugin_manager 自动注册

    @action("set_drawing_mode", description="设置绘制模式", category="画布",
            params={"mode": "str"})
    def set_mode(self, mode):
        if isinstance(mode, str):
            mode = DrawingMode.from_string(mode)
        if mode is None:
            return False
        self.mode = mode
        self.update()
        return True

    @action("draw_rect", description="画矩形框", category="画布",
            params={"x": "int", "y": "int", "w": "int", "h": "int", "label": "str"})
    def action_draw_rect(self, x=100, y=100, w=200, h=200, label=""):
        annotation = {
            "type": "rect", "points": [[x, y], [x+w, y+h]],
            "label": label or "object"
        }
        self.add_annotation(annotation)
        return True
```

**框架基础设施** — `BuiltinActions` 在 `core/common/builtin_actions.py`：

```python
class BuiltinActions:
    def __init__(self, engine=None):
        self._engine = engine

    @action("set_variable", description="设置变量", category="基础控制",
            params={"name": "str", "value": "str"})
    def set_variable(self, name="", value=""):
        if self._engine:
            self._engine._variables[name] = value
        return True

    @action("assert", description="通用断言（lambda表达式）", category="断言",
            params={"check": "str", "message": "str"})
    def assert_check(self, check="", message=""):
        if not check:
            return False
        fn = eval(check)            # "lambda e: ..." → callable
        result = fn(self._engine)   # e = TestEngine
        return bool(result)
```

### 1.6 最佳实践

1. **一个 action 做一件事**：保持方法职责单一
2. **参数都有默认值**：`name=""`、`index=-1`、`value=None`，让 JSON 配置可以只传需要的字段
3. **错误时返回 False，不要抛异常**：测试引擎捕获异常后会标记失败，但返回 False 更清晰
4. **每个 action 都写 print 日志**：方便在控制台跟踪执行流程
5. **`self._engine` 绑定**：如果 action 需要访问 TestEngine（变量、界面引用），在 `__init__` 中接收 engine 参数
6. **新增或修改 action 后务必更新本 Skill 第 3 节的 Action 完整清单**：保持清单与实际代码一致，否则后续使用时会遗漏可用 action

---

## 2. 创建测试流程

### 2.1 JSON 配置文件结构

配置文件放在 `config/` 目录下，以 `test_` 开头（如 `config/test_annotation.json`）：

```json
{
    "test_scenarios": [
        {
            "name": "场景名称",
            "description": "场景说明",
            "enabled": true,
            "steps": [
                { "action": "action_name", "param1": "value1", ... }
            ],
            "on_complete": { "action": "print_result", "message": "完成" }
        }
    ],
    "settings": {
        "default_wait_after_action": 200,
        "fail_on_error": false,
        "log_level": "info",
        "auto_close_on_complete": true
    }
}
```

### 2.2 场景（Scenario）字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | √ | 场景名称，测试报告中显示 |
| `description` | | 场景描述 |
| `enabled` | | `true` 执行，`false` 跳过。默认 `true` |
| `steps` | √ | 步骤数组 |
| `on_complete` | | 场景完成后的回调 action（常用于打印结果） |

### 2.3 步骤（Step）字段

每个 step 是一个 JSON 对象：

```json
{ "action": "action_name", "description": "说明", "wait_after": 500, ...参数 }
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `action` | √ | action 名称（支持短名称，如 `"select_image"`） |
| `description` | | 步骤说明，打印到控制台 |
| `wait_after` | | 执行后等待时间（ms），覆盖 settings 中的默认值 |
| 其他字段 | | 传入 action 方法的参数 |

`wait` action 是例外，它的参数是 `duration`：

```json
{ "action": "wait", "duration": 2000, "description": "等待2秒" }
```

### 2.4 断言（Assertion）

使用 `assert` action 在流程中插入验证：

```json
{ "action": "assert",
  "check": "lambda e: '类别A' in e.annotation_interface.categories",
  "message": "类别A应存在" }
```

- `check` 是 Python lambda 字符串，接收 `e`（TestEngine 实例），返回 bool
- 断言失败时引擎打印 `[Assert FAIL]`，增加 `error_count`
- 如果 `settings.fail_on_error: true`，断言失败会中止当前场景

**safe_builtins 可用函数：** 断言环境为受限 eval，支持 `len/str/int/float/bool/list/dict/set/tuple/abs/min/max/sum/range/enumerate/zip/isinstance/hasattr/getattr/normpath` 及 `True/False/None`。**注意：`any`/`all` 不在白名单内**，需用「列表推导 + len」替代：

```json
# ✅ 推荐：遍历控件查找（any 不可用，用 len(...)>0）
{ "action": "assert",
  "check": "lambda e: len([e.annotation_interface.category_list.item(i).text() for i in range(e.annotation_interface.category_list.count()) if e.annotation_interface.category_list.item(i).text() == '新类别']) > 0",
  "message": "界面类别列表应显示新类别" }
```

**常用断言模式：**

```json
# 验证 categories
{ "action": "assert", "check": "lambda e: 'cat' in e.annotation_interface.categories" }
{ "action": "assert", "check": "lambda e: 'cat' not in e.annotation_interface.categories" }

# 验证 current_category
{ "action": "assert", "check": "lambda e: e.annotation_interface.current_category == 'cat'" }

# 验证当前图片
{ "action": "assert", "check": "lambda e: e.annotation_interface.file_list.currentRow() == 0" }

# 验证标注数量
{ "action": "assert", "check": "lambda e: len(e.annotation_interface.draw_area.annotations) > 0" }

# 验证 mode
{ "action": "assert", "check": "lambda e: e.annotation_interface.draw_area.mode is not None" }

# 验证导出格式（version 插件管理）
{ "action": "assert", "check": "lambda e: e.version_interface.export_format_combo.currentText() == 'yoloseg'" }

# 验证 checkbox 状态
{ "action": "assert", "check": "lambda e: e.annotation_interface.cb_show_bbox.isChecked()" }

# 验证筛选
{ "action": "assert", "check": "lambda e: e.annotation_interface.filter_category is None" }
{ "action": "assert", "check": "lambda e: e.annotation_interface.filter_size_range is not None" }

# 验证 ROI
{ "action": "assert", "check": "lambda e: e.annotation_interface.draw_area.persistent_roi is None" }

# 验证剪贴板
{ "action": "assert", "check": "lambda e: __import__('PySide6').QtWidgets.QApplication.clipboard().text() == e.annotation_interface.current_image_path" }

# 验证任务模式
{ "action": "assert", "check": "lambda e: e.annotation_interface.draw_area.task_mode == 'det'" }

# 验证 selected_index
{ "action": "assert", "check": "lambda e: e.annotation_interface.draw_area.selected_index >= 0" }

# 验证 zoom
{ "action": "assert", "check": "lambda e: e.annotation_interface.draw_area.zoom > 0.1" }

# 验证变量
{ "action": "assert", "check": "lambda e: e._variables.get('myvar') == 'expected'" }
```

### 2.5 变量和表达式

#### 内置变量

| 变量 | 说明 |
|------|------|
| `${loop_index}` | 当前循环索引（在 loop 步骤内使用） |
| `${project_count}` | 项目总数 |

#### 自定义变量

通过 `set_variable` 设置：

```json
{ "action": "set_variable", "name": "current_img", "value": "photo.jpg" }
```

之后在后续步骤中用 `${current_img}` 引用。变量替换作用于所有字符串字段（包括 `action` 名称）。

#### 算术表达式

`${loop_index % project_count}` 会被自动求值，支持简单算术表达式。

### 2.6 循环控制

```json
{ "action": "loop", "count": 3, "steps": [
    { "action": "select_image", "image_index": "${loop_index}" },
    { "action": "wait", "duration": 500 },
    { "action": "switch_image", "direction": 1, "description": "循环索引 ${loop_index}" }
]}
```

- `count`：循环次数
- `steps`：每次循环执行的步骤
- `${loop_index}` 自动从 0 递增

### 2.7 条件分支

```json
{ "action": "if", "condition": "has_categories", "then": [
    { "action": "delete_category", "category_name": "test" }
], "else": [
    { "action": "add_category", "category_name": "test" }
]}
```

支持的 `condition`：
- `has_categories` — categories 非空
- `has_annotations` — 当前图片有标注
- `has_images` — 文件列表非空
- `${var_name}` — 变量值为 truthy

### 2.8 等待策略

- 每个步骤执行后默认等待 `default_wait_after_action` ms（settings 中配置，默认 200ms）
- 单个步骤可覆盖：`"wait_after": 1000`
- 使用 `wait` action 插入等待：`{ "action": "wait", "duration": 3000 }`

### 2.9 Settings 说明

```json
"settings": {
    "default_wait_after_action": 200,   // 步骤间默认等待（ms）
    "fail_on_error": false,              // true=断言失败时终止场景；false=继续执行
    "log_level": "info",                 // 日志级别
    "auto_close_on_complete": true       // 完成后自动关闭
}
```

### 2.10 Agent 指令测试（内嵌 AI Agent）

用于自动化验证「自然语言指令 → Agent 工具执行 → 应用状态变更」链路，无需人工打开程序：

```json
{
    "name": "Agent指令-切换到下一张图片",
    "description": "发送指令并断言状态",
    "enabled": true,
    "steps": [
        { "action": "agent.initialize", "description": "初始化Agent（幂等，需系统配置API Key）" },
        { "action": "agent.chat", "text": "切换到下一张图片", "description": "发送指令（异步）" },
        { "action": "wait_until",
          "check": "lambda e: not e.main_window._ai_agent.is_busy()",
          "timeout": 180000, "interval": 1000, "message": "等待Agent执行完成" },
        { "action": "agent.result", "message": "Agent回复", "description": "打印LLM回复" },
        { "action": "assert",
          "check": "lambda e: e.annotation_interface.file_list.currentRow() == 1",
          "message": "当前图片应为第2张" }
    ]
}
```

关键点：

- `agent.initialize`：从系统配置读取 `ai_api_key/ai_base_url/ai_model`，已初始化时幂等跳过；未配置 API Key 时返回 `False`
- `agent.chat`：异步发送指令，立即返回；Agent 未初始化或忙时返回 `False`
- `agent.wait_until`：轮询 `e.main_window._ai_agent.is_busy()` 直到 Agent 空闲（LLM 调用 + 工具执行期间主线程仍响应）
- `agent.result`：打印 `last_response`，便于人工核对 LLM 是否理解指令
- 场景间对话历史会累积；每个指令场景独立填写「发送 → 等待空闲 → 打印回复 → 断言状态」

### 2.11 Agent 会话质量审查（agent.review）

只靠 `assert` 最终状态不足以验证 Agent 执行过程。`agent.review` 会解析**会话记录**（`agent._agent.messages`），还原 Agent 本次指令后的完整工具调用序列并逐项检查，运行完一条测试命令后据此判断：
① Agent 是否正常完成任务没有报错；② 是否真正完成了任务；③ 有没有多余的额外操作；④ 是否遵循「先输出 ## 执行计划 再调用工具」的行为规则。

```json
{ "action": "agent.review",
  "expected_tools": ["manual.get_state", "manual.add_category", "interact.set_drawing_mode", "interact.draw_rect", "manual.save_current"],
  "forbidden_tools": ["manual.remove_category", "manual.rename_category", "manual.clear_annotations", "manual.delete_annotation"],
  "save_report": "outputs/agent_review_07.json",
  "description": "审查会话：覆盖四类操作且无多余动作" }
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `expected_tools` | | 期望被调用的工具**两段式名称**列表（如 `manual.navigate`/`manual.add_category`/`manual.get_state`），缺省不检查 |
| `forbidden_tools` | | 禁止调用的工具两段式名称列表，出现即判定为多余操作，缺省不检查 |
| `require_plan` | | 是否强制「按计划执行」检查，默认 `true`（严格）。单步简单任务（切图/加类别/切工具/查询）可设 `false` 豁免，报告仍如实显示 `plan_followed` 状态 |
| `save_report` | | 审查报告保存路径；留空自动保存到 `Config.OUTPUTS_DIR`（`outputs/agent_review_时间戳.json`） |

**5 项检查：**

| 检查项 | 含义 | 判定 |
|--------|------|------|
| `no_error` | 无工具报错 | 所有工具返回结果不以 `❌`/`Error` 开头 |
| `plan_followed` | 按计划执行 | 首次工具调用前助手消息须输出「## 执行计划」（规则来自 `core/AGENTS.md`） |
| `task_completed` | 任务完成 | 最后一条助手消息为纯文本回复（说明 Agent 正常收尾，而非卡在工具调用） |
| `expected_tools_ok` | 期望工具全部调用 | `expected_tools` 中的工具名都在调用序列中出现 |
| `forbidden_tools_ok` | 未调用禁止工具 | `forbidden_tools` 中的工具名都未出现 |

全部通过返回 `True`，任一失败返回 `False`（测试引擎计为步骤失败，场景随之标记失败），同时把**完整审查报告**打印到控制台并落盘：

```text
========== [Agent 会话审查] ==========
  指令: ...
  工具调用序列 (N 次):
    [1] ✅ get_state({})
    [2] ✅ add_category({"category_name": "巡检缺陷", "color": "#FF0000"})
    ...
  检查结果:
    ✅ 无工具报错
    ✅ 按计划执行（## 执行计划）
    ✅ 任务完成（有最终文本回复）
    ✅ 期望工具全部调用
    ✅ 无禁止工具被调用
  最终回复: ...
  审查结论: ✅ 通过
======================================
```

**审查范围：** 只审查**最后一条** user 指令之后的会话段（`agent.review` 自动定位），之前的场景消息不受影响。

**重要约束：**
- 审查发生在 `agent.chat` → `wait_until`（等 Agent 空闲）→ `agent.result` 之后
- `expected_tools`/`forbidden_tools` 必须用工具**两段式名称**：Agent 调用工具时使用的名称是 `display_name` = 完整注册名去掉第一段（如 `annotation.manual.navigate` → `manual.navigate`、`annotation.interact.draw_rect` → `interact.draw_rect`）。匹配时同时支持去掉前缀后的短名（如 `manual.get_state` 也能匹配 core 工具的 `get_state`），因此测试配置统一写两段式即可
- 只有 `scope="agent"` 的 action 才会暴露给 Agent；`scope="ui"` 的包装 action（`*_ui` 后缀）Agent 不可见
- Agent 会话中可能调用 `get_state`/`search_tools`/`activate_tools` 等核心工具，这些不应列入 `forbidden_tools`
- 若 Agent 在长任务中滥用通用工具（如 `bash` 查文件、`manual.clear_annotations` 清空标注、`edit.add_annotation` 重复添加），应将它们列入 `forbidden_tools`，使 review 自动捕获这些**多余操作**

### 2.11.1 通用工具自测（agent.test_action）—— 新插件 action 一键验证

新插件开发的核心痛点：**无法保证新注册的 `scope="agent"` action 能被 Agent 正确发现并调用**。`agent.test_action` 把它封装成一行调用：模拟一段真实对话，让 Agent 自己去「找到并调用」目标工具，再自动判定调用是否正确。

```json
{ "action": "agent.test_action",
  "name": "annotation.manual.get_state",
  "hint": false,
  "save_report": "outputs/action_test_get_state.json",
  "description": "验证 Agent 能正确调用 get_state" }
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `name` | √ | 目标 action **完整注册名**（如 `annotation.manual.get_state`）。只接受 `scope="agent"` 的 action，否则 Skip |
| `instruction` | | 自定义自然语言指令；缺省时**自动从 meta.description/params/category 生成**一条不泄露工具名的任务（测「能否被找到」） |
| `hint` | | `true` 指令中点名工具名（只测调用链路与参数；受 LLM 随机性影响小）；默认 `false` |
| `expect_tool` | | 期望被调用的工具名；缺省取该 action 的显示名（去掉前缀后一段，如 `get_state`） |
| `require_plan` | | 是否强制「按计划执行」检查，默认 `false`（单步/查询类工具常直接调用，不做流程编排审查；报告仍如实显示该状态），`--plan` 可强制开启 |
| `save_report` | | 报告保存路径；留空自动存到 `outputs/action_test_时间戳.json` |

**判定逻辑**（复用会话解析，与 `agent.review` 一致）：① 目标工具被调用（短名/完整名归一匹配）；② 调用无报错；③ 按计划执行（可豁免）；④ 任务完成（有最终回复）。全部通过返回 `True`。

**独立脚本**（`scripts/test_agent_action.py`）—— 无需手写 JSON，供开发期快速验证：

```bash
python scripts/test_agent_action.py --list                 # 列出所有 agent 可调用 action
python scripts/test_agent_action.py --name manual.get_state
python scripts/test_agent_action.py --name manual.get_state --hint
python scripts/test_agent_action.py --all                  # 遍历全部 agent action（耗时长）
python scripts/test_agent_action.py --name xxx --instruction "自定义指令"
```

脚本会生成临时 `config/test_agent_action_auto.json` 并交由 `core/main.py --test` 在软件窗口内执行，结果报告保存到 `outputs/action_test_<时间戳>/`。**前提：系统配置已填 AI API Key**，否则 Agent 初始化失败会 Skip。

### 2.12 Agent 指令设计建议

Agent 行为有 LLM 随机性，指令设计应尽量**明确操作方式与顺序**，避免二义性导致工具选择漂移：

- **短指令**（如「切换到下一张图片」）：验证单一工具调用链路，`expected_tools` 只放语义上必然调用的工具（如 `manual.navigate`）
- **复杂长指令**：串接多个操作，覆盖尽可能多的操作类别。例如「先查看状态 → 新增类别 → 切工具 → 画标注 → 保存」可一次覆盖 `manual.get_state`/`manual.add_category`/`interact.set_drawing_mode`/`interact.draw_rect`/`manual.save_current`
- **只读指令**：明确「不要修改任何数据」，`forbidden_tools` 列出全部修改/绘制/类别管理/导航类工具（含 `bash`），验证 Agent 无多余操作
- **巡检指令**：明确指定切换次数与回位方式（如「使用图片跳转功能回到第一张」），`expected_tools` 要求 `manual.navigate`/`manual.select_image` 全部被调用
- 若 LLM 用了不同但合理的实现方式导致 `expected_tools` 未命中，属正常现象：先看 review 报告中的工具序列，判断是否「真正完成任务」，再决定调整指令措辞还是调整期望

**expected_tools 自洽原则：** `expected_tools` 必须与指令语义严格自洽，否则 Agent 遵循指令反而会被判定违规。典型反例：指令写「画一个矩形，类别选择 X」，同时 `forbidden_tools` 列了 `manual.select_category`——Agent 正确的实现是直接 `interact.draw_rect(label="X")`（无需先选类别），若期望里又要求 `manual.select_category` 则必然失败。**当指令允许多种等价实现时，`expected_tools` 只放所有实现路径的共同必选工具**（如 `manual.add_category`/`interact.set_drawing_mode`/`interact.draw_rect`/`manual.save_current`），`forbidden_tools` 不要列指令语义允许的操作。

### 2.13 工具返回值增强（防 Agent 绕道验证）

Agent 的「多余操作」（如用 `bash` 查图片尺寸、反复检查标注文件、重复保存）往往不是恶意绕道，而是**工具返回值信息不足导致的「被迫验证」**：它不信任结果，就用通用工具自查。修复手段是让工具返回值「自证完成」：

| 场景 | 之前返回值 | 增强后 |
|------|-----------|--------|
| `manual.get_state` | 无图片尺寸/标注摘要 | 补 `image_width`/`image_height`（带缓存）与 `annotations` 摘要（label+bbox 列表） |
| `manual.save_current` | 仅 `{"status":"success"}` | 补 `file_path` 保存路径、`annotation_count`、`annotations` 明细 |
| `interact.draw_rect` | `True` | 补 `index`/`label`/`bbox`/`annotation_count` |
| `edit.add_annotation` | `None` | 补 `index`/`label`/`bbox` |
| `interact.zoom` / `set_mode` | `None` | 补缩放倍率/当前模式 |

增强后 Agent 从返回值即可确认「画好了/存好了/尺寸是多少」，不再需要 bash 查文件。**这是比「禁止 bash」更根本的治理手段**：不禁止任何工具，而是消除 Agent 绕道的动机。

### 2.14 压缩记忆保护

Agent 长任务失控的另一个诱因是**上下文压缩后失忆**：对话 token 超限触发消息压缩（67 条→6 条），Agent 忘记自己已经画过/保存过，于是重复执行。`core/ai/message_compressor.py` 的三个压缩函数（`compress_messages`/`emergency_truncate`）在压缩时会调用 `build_operation_summary()` 提取**已成功执行的操作摘要**（工具名 + 关键参数），以 system 消息注入上下文：

```text
## 已完成操作（压缩前）
- manual.add_category(category_name=巡检缺陷)
- manual.save_current(annotation_count=1)
```

Agent 压缩后仍能感知「这些步骤已完成」，避免重复操作。若 Agent 出现「重复调用相同工具」「反复验证同一结果」的失控模式，优先检查压缩是否生效、摘要是否注入。

### 2.15 三大扩展能力（深度分析 / 文档生成 / 意图分析）

为支撑真正有价值的复杂指令，系统补齐了三类底层能力（均对 Agent 可见，属 `_core_tools` 核心工具或 `意图分析` 全局分类 action）：

**① python 脚本执行沙箱（`run_script`，核心工具）**

受限沙箱执行 python 脚本做**深度图像分析**（非浅层统计）：可导入 `cv2`/`numpy` 做灰度直方图、Canny 边缘提取、Sobel/Scharr 梯度、膨胀腐蚀、阈值分割等。自动注入上下文变量 `image_path`/`image_width`/`image_height`/`annotations`/`categories`/`image_index`；脚本把结果赋给变量 `result`（可 JSON 序列化）即作为工具返回值；`print` 输出一并返回；`save_result_image(img, 'name.png')` 可把可视化图保存到 `reports/`。安全约束：仅白名单模块（cv2/numpy/json/math/statistics/collections/re），禁止写文件/删文件/网络/subprocess，默认 30s 超时。

```python
import cv2, numpy as np
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
result = {"mean_gray": round(float(gray.mean()),2),
          "edge_ratio": round(float((edges>0).mean()),4)}
```

**② Markdown 文档工具（核心工具）**

- `generate_doc(filename, content, mode)`：生成/追加 md 文档到**当前项目 `reports/` 目录**（未开项目时用用户目录 `VisionLeeQT_Reports`），返回完整路径
- `read_doc(filename)`：读取文档全文用于核对
- `list_docs()`：列出已生成文档清单
- `open_doc(filename)`：在界面打开**文档编辑器窗口**（编辑区 + Markdown 实时预览，可保存），非模态
- 界面入口：`nav.open_document`（scope=ui，path 省略时打开最近文档）——文档编辑器由 `core/common/widgets/document_editor_dialog.py` 实现

**③ 操作/状态实时意图分析（`intent.*`，category=意图分析，全局暴露）**

与对话 Agent 分工：对话 Agent 只处理用户主动发出的指令；`IntentAnalyzer` 实时监测用户操作历史（`OperationLogger` 已在 action 调用链中**始终后台记录**，不受录制模式限制，每 20 条限流落盘）+ 当前程序状态，规则引擎分析用户意图并主动提出可执行建议（界面上以建议卡片展示，点击即可执行）：

- `intent.get_recent(count)`（scope=agent）：查询最近用户操作历史（action/params/前后状态快照）
- `intent.analyze()`（scope=agent）：规则分析当前意图（标注中/筛选中/浏览中/待保存等）+ 建议列表（含可执行 action 名）
- 后台检测规则：连续新增标注未保存→建议保存；连续翻图未标注→建议一键标注；频繁切换类别→建议整理；多次清空→建议检查；大量标注未导出→建议生成版本

测试场景 10/11 覆盖：10 要求 Agent 用 `run_script` 做深度分析并用 `generate_doc`+`read_doc` 生成核对报告；11 先用测试引擎产生操作历史，再要求 Agent 用 `intent.get_recent`+`intent.analyze` 感知用户意图。

---

## 3. Action 完整清单

## 3. Action 完整清单

### Builtin (19)

| Action | params | 说明 |
|--------|--------|------|
| `wait` | `duration: int` | 等待 N ms |
| `wait_until` | `check: str, timeout: int, interval: int, message: str` | 轮询等待条件满足或超时 |
| `print` | `message: str` | 打印消息 |
| `print_result` | `message: str` | 打印结果 |
| `set_variable` | `name: str, value: str` | 设置变量 `${name}` |
| `close_app` | — | 关闭程序 |
| `loop` | `count: int, steps: list` | 循环子步骤 |
| `start` | `timer_name: str` | 开始计时 |
| `stop` | `timer_name: str` | 停止计时 |
| `assert_less_than` | `timer_name, max_ms, message` | 断言耗时 |
| `assert` | `check: str, message: str` | 通用断言（对应 `assert.generic`） |
| `if` | `condition, then, else` | 条件分支 |
| `ui_responsive` | `message: str` | 断言 UI 响应（对应 `assert.ui_responsive`） |
| `dialog.op` | `op: str, dialog_type: str, item: str` | 统一对话框交互入口（op=accept/reject/delete_item/clear_all/add_item） |
| `agent.initialize` | `api_key, base_url, model` | 初始化 AI Agent（从系统配置读取 API Key，幂等） |
| `agent.chat` | `text: str` | 向 AI Agent 发送自然语言指令（异步） |
| `agent.result` | `message: str` | 打印 AI Agent 最近一次完整回复 |
| `agent.review` | `expected_tools: list, forbidden_tools: list, require_plan: bool, save_report: str` | 审查最近一次 Agent 会话记录（详见 2.11）：无报错/按计划/任务完成/期望工具全部调用/无禁止工具 |
| `agent.test_action` | `name: str, instruction: str, hint: bool, expect_tool: str, require_plan: bool, save_report: str, max_wait: int` | 通用工具自测（详见 2.11.1）：模拟对话让 Agent 找到并调用指定 scope=agent action，判定调用正确性 |
| `intent.get_recent` | `count: int` | 查询最近用户操作历史（scope=agent，category=意图分析，全局暴露） |
| `intent.analyze` | | 规则分析用户当前意图与可执行建议（scope=agent） |

### Input (4)

`mouse`, `key_press`, `key_type`, `key_shortcut`

### Canvas — DrawingWidget (15)

`reset_view`, `set_drawing_mode`, `draw_rect`, `add_annotation`, `remove_annotation`, `select_annotation_at`, `select_in_rect`, `add_polygon_vertex`, `close_polygon`, `add_polygon_vertex_on_edge`, `delete_polygon_vertex`, `add_sam_point`, `confirm_sam_annotation`, `toggle_enhance`, `zoom`（direction=1/-1）

### Navigation — MainWindow (3)

`switch_to` (`target: str`), `refresh_all`, `toggle_theme`

### Annotation Interface (104)

短名（别名）清单，按域前缀分组（完整注册名 = `annotation.` + 下述名称）：

- **nav**: `change_save_path`, `copy_file_path`, `copy_image_to_clipboard`, `delete_image_file`, `open_directory`, `open_in_explorer`（target=file/output）, `refresh_file_list`, `select_image`（scope=ui）, `switch_image`（direction=1/-1，scope=ui）, `switch_project`
- **interact**: `cancel_current`, `clear_persistent_roi`, `refine_single_annotation`, `set_display_options`, `set_task_mode`, `toggle_panel`（side=left/right）, `toggle_recording`, `toggle_tool`, `toggle_view`
- **edit**: `accept_secondary_annotation`, `add_secondary_as_primary`, `change_annotation_category`, `clear_annotations`, `clear_secondary_annotations`, `delete_annotation`, `delete_selected`, `save_current`, `select_annotation`（direction=1/-1）, `set_annotation_visibility`, `step_history`（direction=-1/1）, `toggle_annotations_visibility`（scope=current/all）, `toggle_hard_sample`
- **category**: `add_category`, `delete_category`, `export_categories`, `import_categories`, `rename_category`, `select_category`, `switch_category`
- **batch**: `one_click_annotate`, `one_click_by_all`, `one_click_by_all_ui`, `one_click_by_current`, `one_click_by_current_ui`, `refresh_view`, `set_split`
- **review**: `clear_filters`, `search_images`, `set_filter`
- **resource**: `add_custom_model`, `add_prompt`, `clear_example_library`, `clear_examples_selection`, `clear_prompts`, `delete_prompt`, `get_rules`, `get_selected_examples`, `list_builtin_models`, `list_custom_models`, `list_examples`, `list_prompts`, `open_example_library`, `open_prompt_library`, `open_rule_config`, `remove_custom_model`, `remove_example`, `select_trained_model`, `set_as_example`, `set_examples`, `set_model`（role=interactive/auto/plain）, `set_rule`
- **manual**: `accept_secondary_annotation`, `add_category`, `add_secondary_as_primary`, `clear_annotations`, `clear_filters`, `clear_secondary_annotations`, `dataset_splits`（op=save/load）, `delete_annotation`, `export_categories`, `fit_check_current`（scope=agent，检查当前图片标注贴合度；小目标走CV、中大目标走SAM2，results 中 shape_type 区分 polygon/bbox 标注）, `get_state`, `import_categories`, `load_secondary_annotations`, `merge_categories`, `navigate`（direction=1/-1）, `open_directory`, `predict_mask_with_bbox`, `predict_mask_with_points`, `predict_similar_in_image`, `refine_annotation`（refine_method 可选 vitmatte/cascadepsp/hybrid；hybrid 混合分割，可替代原 apply_fit_check 的应用场景）, `remove_category`, `rename_category`, `save_current`, `search_images`, `select_category`, `select_image`, `set_filter`, `set_hard_sample`, `set_split`, `set_task_mode`, `step_history`（direction=-1/1）, `update_annotation_label`
- **annotate**: `by_current`
- 其他：`refresh_interface`

> 说明：`manual.*`、`annotate.*`、`batch.*`、`resource.*` 等为 service 层（scope="agent"）纯数据 action；`nav.*`、`interact.*`、`edit.*`、`category.*`、`review.*` 等为 interface 层（scope="ui"）action。短名冲突时后缀匹配优先解析为 scope="ui" 的 action（详见 `_resolve_name`）。

### Dashboard Interface (11)

| Action | params | 说明 |
|--------|--------|------|
| `click_project_card` | `project_name: str, project_index: int` | 点击项目卡片进入项目 |
| `refresh_interface` | | 刷新项目大厅界面（供 nav.refresh_all 调用） |
| `refresh_projects` | | 刷新项目列表 |
| `delete_project` | `project_name, skip_confirm` | 删除项目（UI 包装） |
| `import_images` | `project_name, source_dir` | 导入图片（UI 包装） |
| `create_project` | `project_name, task_type` | 创建项目（UI 包装） |
| `delete_project_data` | | 删除项目数据（service，scope="agent"） |
| `create_project_data` | | 创建项目数据（service，scope="agent"） |
| `load_project_data` | | 加载项目数据（service，scope="agent"） |
| `update_project_info` | | 更新项目信息（service，scope="agent"） |
| `get_image_count` | | 获取图片数量（service，scope="agent"） |

### Version Interface (20)

| Action | params | 说明 |
|--------|--------|------|
| `delete_version` | `version_name` | 删除指定的版本（service，scope="agent"） |
| `delete_version_ui` | `version_name` | 删除指定的版本（UI 包装，弹出确认对话框） |
| `export_error` | | export_error |
| `export_finished` | | export_finished |
| `export_format_changed` | | export_format_changed |
| `generate_version` | | generate_version |
| `list_versions` | | 列出当前项目的所有版本（service，scope="agent"） |
| `project_combo_changed` | | 项目切换事件 |
| `open_version` | | 从资源管理器中打开版本文件夹 |
| `refresh_project_list` | | 刷新项目列表 |
| `refresh_interface` | | 刷新版本管理界面（供 nav.refresh_all 调用） |
| `refresh_versions` | | 刷新版本列表 |
| `clear_example_sets_selection` | | 清除已选示例集（service，scope="agent"） |
| `get_selected_example_sets` | | 获取已选示例集（service，scope="agent"） |
| `list_example_sets` | | 列出示例集（service，scope="agent"） |
| `select_examples` | | 打开示例图片选择对话框 |
| `set_example_sets` | `example_names` | 设置示例集（service，scope="agent"） |
| `set_example_sets_ui` | `example_names` | 设置示例集（UI 包装，同步界面状态） |
| `split_mode_changed` | | 数据集划分模式切换 |
| `update_stats` | | 更新统计信息 |

### Test Interface (0)

已移除。测试引擎（`core/common/test_engine.py`）不再注册 `get_scenario`/`get_step_data`/`load_scenario` 等 action，场景加载由引擎内部方法 `load_scenarios()`/`load_scenario_dict()` 完成。

### Flow Interface (0)

已移除。flow 插件未启用（`core/plugins.json` 中 `"flow": false`），无对应插件目录，相关 action 均未注册。

### Inference Interface (0)

已移除。inference 插件未启用，无对应插件目录，相关 action 均未注册。

### Kachi Interface (0)

已移除。kachi 插件未启用，无对应插件目录，相关 action 均未注册。

### Training Interface (0)

已移除。training 插件未启用，无对应插件目录，相关 action 均未注册。

### Unsupervised Interface (0)

已移除。unsupervised 插件未启用，无对应插件目录，相关 action 均未注册。

### Generation Interface (0)

已移除。generation 插件未启用，无对应插件目录，相关 action 均未注册。

### Config Interface (2)

| Action | params | 说明 |
|--------|--------|------|
| `set_plugin_manager` | | 设置插件管理器实例（内部使用） |
| `set_project_service` | | 设置项目服务实例（内部使用） |
