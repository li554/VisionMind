# VisionMind

**Agent 驱动的工业级图像标注软件 · 基于 PySide6 + QFluentWidgets + LLM**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-GPL%203.0-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PySide6](https://img.shields.io/badge/PySide6-Qt6-green)]()

***

## 项目简介 / Overview

VisionMind 是一款 **Agent 驱动的工业级图像标注软件**：内置对话式智能体，用自然语言即可驱动全部标注操作——"说话即标注"。意图分析自动匹配操作，`@action` 动作体系全量暴露为 LLM 工具（Tool Call），支持多模态读图（原图 / ROI 裁剪 / 标注叠加），并结合 SAM 系列交互式分割与 YOLO 检测模型，提供从手动标注到 AI 辅助标注、从数据管理到模型推理的完整工作流。插件化架构设计，所有能力可扩展、可测试、可被 Agent 调用。

***

## 特性 / Features

### 多模式标注

- 矩形框（Bounding Box）

- 多边形（Polygon）

- 旋转矩形（OBB）

- SAM 交互式分割（点击/框选提示）

### AI 辅助标注

- 基于 SAM/SAM2/SAM3/HQ-SAM 的交互式分割

- 基于示例的少样本标注（Few-Shot）

- 文本引导标注（Text-Guided）

- YOLOv8/YOLOE 自动检测标注

### 多格式支持

- LabelMe JSON

- COCO（实例分割）

- VOC（Pascal VOC XML）

- YOLO（TXT 标注）

- SA1B（SAM 格式）

- Mask（二值掩码）

### 项目管理

- 项目模式统一管理数据集与标注

- 版本管理与格式转换

- 数据集导入导出

### 自动化

- 自动标注流程（Auto Annotation Pipeline）

- 切片推理（Sliced Inference）

- 推理规则过滤（Rule-Based Filtering）

### AI 智能体（Agent）

- 内置对话式 Agent 侧栏：多会话管理、流式输出、工具调用过程可视化

- 自然语言驱动标注：意图分析自动匹配操作，建议卡片一键执行

- `@action` 动作体系自动暴露为 LLM 工具（Tool Schema），Agent 可调用全部标注能力

- 多模态理解：附加图片/文件（回形针/图片按钮），支持原图、ROI 裁剪、标注叠加等读图模式

- 沙箱 Python 工具：受限环境中执行脚本，结合 cv2/numpy 对当前图片做深度分析

- 意图分析悬浮球：常驻状态提示与快捷操作入口

### 插件系统

- 完整的插件开发框架（`IPlugin` 接口 + `PluginContext` 上下文）

- 依赖排序加载与热插拔

- `EventBus` 事件总线实现插件间松耦合通信

- `ActionRegistry` 统一注册与调用插件动作

### 自动化测试

- 基于 `TestEngine` 的 UI 自动化测试框架

- JSON 配置驱动的测试场景

- `@action` 装饰器注册可测试动作

***

## 技术栈 / Tech Stack

| 类别       | 技术                             |
| -------- | ------------------------------ |
| UI 框架    | PySide6 (Qt6)                  |
| 组件库      | QFluentWidgets (Fluent Design) |
| 交互式分割    | SAM / SAM2 / SAM3 / HQ-SAM     |
| 检测模型     | YOLOv8 / YOLOE                 |
| 推理加速     | TensorRT                       |
| 模型部署     | ONNX Runtime                   |
| 大语言模型    | OpenAI 兼容接口（多模型可配）             |
| Agent 架构 | Tool Call 工具调用 / 意图分析 / 多模态消息  |
| 编程语言     | Python 3.10+                   |

***

## 项目结构 / Project Structure

```
VisionMind/
├── core/                        # 核心框架
│   ├── backend/                 # AI 后端
│   │   ├── autoannotation/      # 自动标注引擎
│   │   │   ├── formats/         # 标注格式（LabelMe/COCO/VOC/YOLO/SA1B/Mask）
│   │   │   └── models/          # 模型封装（SAM/SAM2/SAM3/HQ-SAM/YOLOv8/YOLOE）
│   │   └── ...                    # AI 后端扩展
│   ├── agent/                     # AI 智能体
│   │   ├── agent_dialog.py        # 对话侧栏（会话/流式/工具调用展示）
│   │   ├── visionmind_agent.py    # LLM 智能体（意图分析 + 工具调用循环）
│   │   ├── multimodal.py          # 多模态编解码（读图/ROI/标注叠加）
│   │   ├── tools/                 # Agent 工具（状态/读图/Python 沙箱/提示词）
│   │   └── ...
│   ├── common/                  # 公共组件
│   │   ├── widgets/             # 通用控件（绘图/示例选择/规则配置/参数调节）
│   │   ├── action_registry.py   # 动作注册中心
│   │   ├── test_engine.py       # 自动化测试引擎
│   │   ├── settings.py          # 全局设置
│   │   └── ...
│   ├── service/                 # 服务层（项目/数据集）
│   ├── view/                    # 核心界面
│   ├── event_bus.py             # 事件总线
│   ├── plugin_manager.py        # 插件管理器
│   ├── main_window.py           # 主窗口
│   └── main.py                  # 启动入口
├── plugins/                     # 插件目录
│   ├── annotation/              # 手动标注
│   ├── dashboard/               # 项目大厅
│   ├── version/                 # 版本管理
│   └── ...                      # 更多插件陆续开放（系统配置为核心内置模块）
├── config/                      # 测试流程配置
├── resources/                   # 资源文件（图标、主题、示例、配方）
└── docs/                        # 文档
```

***

## 快速开始 / Getting Started

### 环境要求

- Python 3.10+

- CUDA 11.8+（GPU 推理）

- Windows 10/11

### 安装

```bash
# 克隆项目
git clone https://github.com/li554/VisionMind.git
cd VisionMind

# 创建并激活 Conda 环境
conda create -n visionmind python=3.10 -y
conda activate visionmind

# 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
# 启动主程序
python -m core.main

# 以自动化测试模式启动
python -m core.main --test test_annotation.json
```

***

## 架构概览 / Architecture

VisionMind 采用**插件化架构**，核心框架提供基础服务，所有业务功能通过插件扩展：

```
┌──────────────────────────────────────────────┐
│                  MainWindow                   │
├──────────────────────────────────────────────┤
│  PluginManager  │  EventBus  │ ActionRegistry │
├──────────────────────────────────────────────┤
│              PluginContext                    │
│  ┌─────────┐ ┌──────────┐ ┌───────────────┐ │
│  │Project  │ │Dataset   │ │License        │ │
│  │Service  │ │Service   │ │Manager        │ │
│  └─────────┘ └──────────┘ └───────────────┘ │
├──────────────────────────────────────────────┤
│                 Plugins                       │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                │
│  │Anno│ │Dash│ │Vers│ │Conf │  ...陆续开放   │
│  └────┘ └────┘ └────┘ └────┘                │
└──────────────────────────────────────────────┘
```

### 核心组件

| 组件                | 职责                                           |
| ----------------- | -------------------------------------------- |
| `PluginManager`   | 插件发现、依赖排序、加载/卸载                              |
| `PluginContext`   | 插件运行上下文，注入主窗口、事件总线、服务                        |
| `EventBus`        | 插件间松耦合事件通信                                   |
| `ActionRegistry`  | 统一注册与调用 `@action` 方法，支撑自动化测试（后续将支持 AI 智能体调用） |
| `TestEngine`      | JSON 配置驱动的 UI 自动化测试引擎                        |
| `ProjectService`  | 项目与数据集管理                                     |
| `VisionMindAgent` | LLM 智能体：意图分析、工具调用循环、多模态消息管理                  |

### 插件开发

每个插件实现 `IPlugin` 接口，包含以下核心方法：

```python
class IPlugin(ABC):
    @staticmethod
    @abstractmethod
    def plugin_id() -> str: ...      # 唯一标识

    @staticmethod
    @abstractmethod
    def plugin_name() -> str: ...    # 显示名称

    @staticmethod
    @abstractmethod
    def plugin_icon() -> str: ...    # FluentIcon 名称

    @abstractmethod
    def on_load(self, ctx: PluginContext): ...   # 加载回调

    @abstractmethod
    def on_unload(self, ctx: PluginContext): ... # 卸载回调
```

> 详细的插件开发指南请参考 [插件开发文档](docs/action_reference.md) 和项目内嵌的插件开发 Skill。

***

## 使用指南 / Usage

### 标注工作流

1. **创建项目** — 在项目大厅（Dashboard）中新建项目，导入图像数据
2. **选择标注模式** — 矩形框 / 多边形 / OBB / SAM 交互式分割
3. **标注与编辑** — 手动标注或使用 AI 辅助（SAM 点击分割、YOLO 自动检测）
4. **导出结果** — 支持 LabelMe / COCO / VOC / YOLO / SA1B / Mask 多格式导出

### AI 辅助标注

- **交互式分割**：在图像上点击目标区域，SAM系列模型自动生成分割掩码

- **少样本标注**：提供示例图像，模型自动学习并标注相似目标

- **自动标注流程**：配置检测模型 + 推理规则，批量自动标注

### AI 智能体（自然语言标注）

1. 在 **设置 → AI 助手** 中填写提供商、 API Key、接口地址与模型
2. 点击标题栏星芒按钮打开 **Agent 侧栏**
3. 用自然语言下指令，例如：

   - 「帮我自动标注当前图片」

   - 「批量标注前 10 张，用切片推理」

   - 「把 object 类别的颜色改成绿色」
4. 添加图片或文件，Agent 可直接读图分析，支持自主读图
5. 支持程序内图片、标注、标注区域、ROI区域添加到对话
6. 支持plan指令和compact指令、支持读写工具
7. 支持意图分析，属于试验功能，暂不完善

### 自动化测试

编写 JSON 测试配置，启动时通过 `--test` 参数加载：

```json
{
  "scenarios": [
    {
      "name": "标注流程测试",
      "steps": [
        { "action": "annotation.create_rect", "args": {"x": 100, "y": 100, "w": 200, "h": 150} },
        { "action": "annotation.save", "assert": { "result": "success" } }
      ]
    }
  ]
}
```

***

## TODO

- [x] Action 工具化：将 @action 体系暴露为 AI 可调用的工具（Tool Schema）

- [x] AI 智能体接入：API Key 填写接口，接入大语言模型（Agent 侧栏已实现）

- [x] 自然语言驱动标注：用户通过对话式指令控制软件完成标注任务

- [ ] 开发DSH插件版：将标注能力接入 DeepSeek Harness（DSH），在 DSH 中直接交互打标

- [ ] 更多插件陆续开放（录制回放、训练平台 ... ）

- [ ] 开发Web端版本

- [ ] 支持云端推理

***

## 贡献指南 / Contributing

欢迎贡献！请遵循以下流程：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/your-feature`）
3. 提交更改（`git commit -m "feat: add your feature"`）
4. 推送到分支（`git push origin feature/your-feature`）
5. 创建 Pull Request

### 提交规范

| 前缀         | 说明      |
| ---------- | ------- |
| `feat`     | 新功能     |
| `fix`      | 修复缺陷    |
| `docs`     | 文档更新    |
| `refactor` | 代码重构    |
| `test`     | 测试相关    |
| `chore`    | 构建/工具变更 |

### 插件开发

如需开发自定义插件，请参考以下资源：

- 项目内嵌 Skill：`插件开发指南`、`测试流程指南`

- 现有插件示例：`plugins/annotation/`、`plugins/dashboard/`

***

## 致谢 / Acknowledgments

本项目借鉴并致敬以下优秀开源项目：

- [CoreCoder](https://github.com/he-yufeng/CoreCoder)（MIT）— 内置 Agent 内核（`core/corecoder/`）的 LLM 工具调用循环、上下文压缩、子代理等核心模式源自 CoreCoder 对 Claude Code 的精简重实现。

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（DSH，MIT）— Web 版 Agent 侧栏（`core/agent/webui/`）的视觉与交互设计参照 DSH 前端实现。

感谢以上项目作者与社区的出色工作。

***

## 许可证 / License

本项目基于 GPL 3.0 协议开源。详见 [LICENSE](LICENSE)。

***

<p align="center">
  Built with ❤️ by li554
</p>
