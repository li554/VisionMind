# 标注界面架构治理

本文档是 `plugins/annotation` 图像/标注生命周期架构的**目标形态 + 分阶段落地计划**，
以及一份**已确认缺陷清单**（每条都有 file:line 与复现/证据方式）。

起因：用户在快速前后切换图片一段时间后界面卡住，并频繁在 `logs/` 留下崩溃日志。
经排查，这**不是一个问题，而是两个独立问题**：

| 问题 | 机制 | 证据 |
|---|---|---|
| **崩溃** | 运行中的 `QThread` 被提前析构 -> `qFatal` -> `abort()` | 68 份 crash log 中 56 份含 `QThread: Destroyed while thread '' is still running`；已用真实 `switch_image()` 复现 |
| **卡住** | 列表重建泄漏子控件（每次切图 +120 控件）+ 每次切图数百 ms 的 GUI 线程 I/O 与重建 | `tests/annotation/test_label_list_leak.py` 实测 120 -> 3720 |

---

## 1. 治理原则（目标架构的 6 条不变量）

1. **唯一状态所有者** ✅（§4.8，`ImageSession`）：`(当前图片路径, 该图标注, 已提交像素与度量)`
   是一个原子整体，由**一个**对象持有；其余都是它的只读投影（赋值即 `AttributeError`）。
   禁止 interface / service / widget 各存一份。**按图作用域的编辑态也已并入**
   （§4.25：撤销栈 / 重做栈 / 可见性集合 / 副标注 —— service 侧只留同名委托 property）。
2. **异步结果只通过"代次提交协议"落地**：每次切换 +1 代次；工作线程产出 `(generation, payload)`；
   只有 `generation == 当前代次`（且路径一致）才允许提交。提交是**原子的**（像素与标注一起换）。
3. **线程生命周期不靠引用计数** ✅（标注插件内 **0 个 `QThread` 子类、0 处 `.terminate()`**，见 §4.11）：
   不继承 `QThread`、不遮蔽 Qt 内建信号名、调用方永不"丢弃线程引用"；
   一律 `QThreadPool` / `QObject+moveToThread`，并显式 `drain(timeout)`。
4. **不伪中断、不强杀**：`terminate()` 永久禁用；不可中断的长操作（`QImageReader.read()`）用
   "代次作废 + 结果丢弃"处理，绝不在 `QThread.wait()` 里阻塞 GUI 线程。
5. **GUI 线程只做 O(可见) 工作** 🚧 —— 所有**随数据集规模重复发生**的 I/O 都已在后台：
   图像解码、缩略图、批量标注、标注装载、图像增强、标注写入、搜索扫描。
   仅剩的同步 I/O 是**一次性目录列举**（`scan_image_files`），实测 5000 张 **8ms**，
   不构成实质冻结；`processEvents()` 在状态变更处理器内永久禁用。
6. **可确定性关闭**：`main_window.closeEvent` -> 插件 teardown -> 各 `drain(timeout)`；
   关闭前先作废代次，排空必须早于 Qt 对象树析构。

---

## 2. 已确认缺陷清单

分级：**P0** = 数据损坏/必崩；**P1** = 崩溃或冻结；**P2** = 正确性/资源。

### P0 数据损坏（可复现）

| # | 位置 | 缺陷 | 修复方向 |
|---|---|---|---|
| D1 | `manual_annotation_service.delete_image`（import 与 `os.remove` 顺序） | `from core.backend.formats import find_label_file` **必定 ImportError**（该符号未从 `core/backend/formats/__init__.py` 导出，实测复现），而 `os.remove(image_path)` 在它**之前**已执行 -> **图片被删、标注留下**，且调用方据 `success=False` 跳过列表清理 -> 界面残留失效条目 | 改 `from core.backend.utils import find_label_file`；先收集并校验全部待删路径再删；删除改确认 | **已修复**（§4.6） |
| D2 | `auto_annotation_service.annotate_by_current`（约 :424） | 批量跑完把 `manual.current_annotations` 无条件换成**最后一张批量图**的标注，不校验 `image_path`；此后 `save_current` 会把该标注写进**屏幕上那张图**的文件 | **已修复**（§4.14；阶段 2 的只读化先把它变成了显式 AttributeError，本轮改为受控入口） |
| D3 | `manual_annotation_service.undo/redo` | 撤销栈虽存了 `image_path` 却**从不比对**，`save_current` 用当前路径落盘 -> 切图/切项目后 Ctrl+Z **把旧图标注写进新图文件** | **已修复**（§4.14：出栈即校验路径，不一致则拒绝 + 清栈） |
| D4 | `manual_annotation_service.enter_non_project_mode` | 用 `if not self.context._non_project_save_dir` 判"首次进入" -> 开过目录 A 后再开目录 B，`output_dir` 仍是 **A**，B 的标注被存进 A（同名 basename 覆盖） | **已修复**（§4.15：进入时无条件更新保存目录） |
| D5 | `core/backend/formats/*` 全部写入点 | **零原子写**：一律 `open(path,'w')`（立即截断）+ 原地 `json.dump`/`tree.write`。与"平均每天崩多次"叠加 = 崩溃瞬间正在保存的那个标注文件被清空/半写 | 引入 `atomic_write_json/bytes`（同目录临时文件 + flush + `os.replace`，可选 fsync），全量替换 | **已修复**（§4.6，32 处全部替换） |
| D6 | `manual_annotation_service.delete_annotation_points` 的调用方（interface 单删分支） | 单标注删除路径不重算 `batch_selected_points`；之后按 Delete 会把**过期顶点三元组**交给删除函数 -> **删错顶点** | 任何标注集合变更都必须作废全部下标态 |
| D7 | `DrawingWidget` 瞬时状态跨图存活 | `dragging_* / rect_start,end / batch_* / obb_points / hover_point / is_near_first_point / right_* / ignore_next_left_press / last_mouse_pos / focused_index` 在 `set_image`/`clear` 都不清；按住左键用快捷键切图后，释放会用**旧图坐标**向**新图**提交标注，而 `annotation_modified` -> `save_current` **无条件写盘**且撤销栈已被清空 -> 静默、不可撤销的数据损坏 | 抽 `_reset_transient_state()`，`set_image`/`clear`/标注集合变更三处共用；`mouseReleaseEvent` 加自愈 | **已修复**（§4.8） |
| D8 | `interface.load_image` | 先提交 `current_image_path`，之后 `set_image` 失败会 `return`、标注加载失败会保留旧标注 -> **路径已换、标注是旧图的**，下一次自动保存即覆盖写 | **已修复**（§4.8：装载失败会禁止保存） |
| D9 | `auto_annotation_service.batch_ai_convert`（约 :1104 / :1168） | 调用的 `find_and_load_annotations` / `save_image_annotations` **只存在于 `ManualAnnotationService`**（实测确认）-> 第一次迭代必 `AttributeError`，"AI 全量转换"100% 不可用 | **已修复**（§4.15 委托给 manual 服务）；**取消按钮从未接线**也已修复（§4.21） |
| D10 | `interface.delete_image_file` | `skip_confirm` 是死参数、无确认框，而 action 描述写着"物理删除不可恢复" -> agent 可无确认删用户图片 | **已修复**（§4.6，确认门生效；回收站未做） |
| D34 | 删除正在解码的图片 | Windows 上解码进行中 `os.remove` 抛 `WinError 32 另一个程序正在使用此文件`。打开目录会为所有可见行一次性排入缩略图任务，因此"刚打开目录就删图"**必定失败**（实测：缩略图池活跃 3、解码池活跃 1，删除失败；约 200ms 后成功） | **已修复**（§4.6） |

### P1 崩溃 / 冻结

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| D11 | `drawing_widget.set_image`（原 `ImageLoader`） | 丢弃运行中的 `QThread` 引用 -> abort | **已修复**（见 §4） |
| D12 | `model_manager.load_model_async` | `ModelLoadThread.finished` 遮蔽 `QThread.finished` 且 `run()` 内提前 emit -> `_model_loading` 提前清 False -> 覆盖 `self.load_thread` 丢引用 -> 同类 abort。两次快速切模型即可命中；另有"旧线程完成信号清掉新线程状态"的误判 | **已修复**（§4.11） |
| D13 | `interface.refresh_label_list` 等列表重建 | `takeItem()` 不销毁 itemWidget -> 每次重建泄漏 6 控件/item；每次切图调用（且 `load_image` 内调 **2 次**）-> 控件线性累积 + 单次重建 ~120ms（20 标注） | **已修复泄漏**（见 §4）；重复调用与单次成本待优化（阶段 3） |
| D14 | `stop_and_wait` ↔ `run_on_main` 互等死锁 | 主线程 `QThread.wait(max_ms)` 等 worker，而 worker 正阻塞在 `main_thread_dispatcher.ctx.done.wait()`（无超时）-> 界面冻结最长 `max_wait`（默认 600000ms = 10 分钟）；**另有一层更隐蔽的缺陷**：abort 后只标记 abandoned，未把 ctx 从队列摘掉，残留记录会在下一次派发时被执行，照样占住主线程 | **已修复**（§4.16） |
| D15 | `search_images` 挂在 `textChanged` | 每次击键全列表 O(N)，有类别筛选时**逐图读盘 + json.load** -> 数千张即秒级~分钟级冻结；`_apply_search_tokens` 一次输入还会触发多轮全量对账 | **已修复**（§4.13） |
| D16 | `_apply_search_tokens` / `load_directory_images` / 复选框 / 类别按钮 | 在发射栈上重建列表、销毁发起行；`load_directory_images` 的 `search_box.clear()` 未 blockSignals -> 连续两次 `set_image` + 短暂错图 | **已修复**（§4.24：信号贯穿整个重建期屏蔽 + 发射栈内重建延后） |
| D17 | `_generate_enhanced_pixmap` | LibLLIE + numpy 全图运算在 **GUI 线程**同步执行，开启增强后每次切图冻结；且对增强器返回值零校验（dtype/通道数不符会按 3 字节行距越界读 numpy 缓冲） | **已修复**（§4.10） |
| D18 | `paintEvent` 不变量过弱 | `persistent_roi` 直接解包 4 元组（来源是 `project_info.json` 裸 JSON，无 schema 校验）-> 损坏值使**画布永久无法重绘**；`bbox=None`、`polygons=[1.0,2.0]` 等畸形数据在帧循环内抛未捕获异常 | **已修复**（§4.23：读取时校验 + 帧循环内的 bbox/多边形访问全部走安全访问器） |
| D19 | `ThumbnailManager._pending` | `run()` 在文件缺失/解码失败/异常三处**不发信号就 return**，而 `_pending.discard` 只在完成槽里 -> 该路径永久留在"在途"集合、**永不重试**；`clear_pending()` 无调用者 | **已修复**（§4.6） |
| D20 | `main_window.closeEvent` | 只保存 AI 会话（且 `sidebar.shutdown()` 是 `pass`、`agent_dialog` 无 `shutdown`）；全仓无 `aboutToQuit`、`plugins/` 下无 `def shutdown` -> **没有任何线程排空**；`drain()` 新增后仍无调用者 | **已修复**（§4.4、§4.5） |

### P2 正确性 / 资源

- **D21** 撤销/重做不维护 `hidden_indices`（`clear_annotations` 直接 `.clear()` -> 撤销后可见性永久丢失）。**【已修复 §4.19】**（撤销帧携带可见性 + 空列表也能写回）
- **D22** `_is_undo_redo` 是死标志（只被赋 False，从未置 True）-> undo 期间零重入保护。
- **D23** `annotation_cache` 写入端与失效端不对称：`save_current`/`delete_annotation`/`refine` 等**都不 invalidate** -> 筛选与"设为示例"读到陈旧标注。
- **D24** `_coco_cache` 生命周期不受控：换项目/换目录不失效，命中后跳过重读磁盘 -> 磁盘外部改动会被内存旧版反向覆盖。
- **D25** 无锁全局预测器：`core/backend/core.py` 的 `_interactive_predictor` 等懒创建无锁，且加载新模型前会 `release_models()` 释放**全部**模型 -> 若另一线程正在推理，模型被从脚下释放。触发路径明确（主线程贴合度检查 vs agent 批量）。
- **D26** `fit_check_current` 标 `read_only=True` 却在主线程跑整轮 SAM2/CV 推理 -> UI 完全冻结。
- **D27** `refine` 每次新建并加载细化模型且缓存键无人命中 -> 反复重载、并发时显存翻倍。
- **D28** `process_batch_all` 的 `selected_examples` 是死参数（UI 勾选对批量无效）；`one_click_running` 只写不读 -> 两次批量可真并发。
- **D29** `remove_annotation` 不落盘（`delete_annotation` 会）-> 批量删除后磁盘仍是旧标注，崩溃即丢。
- **D30** 运行时状态写进**受版本控制**的 `core/core.json`：`settings.set()` 整文件重写，且切图路径每次调用 `save_last_index` -> 每次切图一次落盘 + 该文件永久 dirty。`shortcut_manager` 也整文件读改写同一文件（无锁）。**【已修复 §4.17】**（会话态键不落盘 + 保存原子化/串行化；历史遗留键自动清理）
- **D31** `annotation_interface` 内 `except Exception: if settings.DEBUG: raise` 的 raise 会被 `StateRouter._dispatch` 再吞一次（只 print）-> DEBUG 下的"抛出来"完全无效，UI↔service 分歧永久静默。
- **D32** `Interface`/`IPlugin` 无生命周期钩子；`unload_plugin` 全仓无调用者；`remove_plugin_interface` 不 `deleteLater()`；`uninstall_global_exception_handler` 无调用者。
- **D33** `_last_ann_sig` 私有第二份标注状态，`clear_project`/`_do_set_project` 不重置，且签名不含多边形坐标 -> 抑制真实刷新。**【已修复 §4.18】**（签名纳入几何 + 新增 `reset_annotations_refresh_state()` 并在三个真值替换入口调用）

> 注：以上行号来自审计快照；`annotation_interface.py` 在本次修复后行号有偏移，
> 故优先按**函数名**定位。审计中标注 unverified 的项（PySide6 包装器身份比较、
> 跨线程 `activatedAmbiguously`、`takeItem` 所有权语义的边界）未列入上表。

---

## 3. 分阶段落地计划

每阶段的验收都是**可复现脚本 + 退出码断言**（见 §5）。

| 阶段 | 内容 | 验收断言 |
|---|---|---|
| **1 ✅** | 图像解码管线重写：删除 `ImageLoader(QThread)`，改为 `ImageDecodeService`（`QThreadPool`，有界并发 3、最新优先、代次由调用方持有、无 `terminate`、`drain(timeout)`） | I1 快速切图不 abort；I3 并发 ≤ max_workers；I4 最终图不丢 |
| **1.5 ✅** | 列表重建泄漏修复：`_clear_list_widget()` 显式 `removeItemWidget + deleteLater` | L1 viewport 子控件数零增长 |
| **2 ✅** | 单一状态所有者（`plugins/annotation/session.py::ImageSession`）+ 原子提交 ✅；交互/`get_state` 以"已提交"为闸门 ✅；装载失败禁止保存 ✅（§4.8）；**按图作用域编辑态并入** ✅（§4.25）；标注 dict 边界归一化 + `persistent_roi` schema 校验 ✅（§4.23） | ✅ 切换过程中不存在可观测的"(图 N-1 度量, 图 N 标注)"状态；三处投影恒等于唯一真值（`test_single_state_owner.py`）；✅ 畸形标注/ROI 不使画布失效（`test_canvas_invariants.py`）；✅ service 上无第二份编辑态（`test_edit_state_owner.py`） |
| **3 ✅** | **已完成**：逐图标注装载移入后台 ✅、去掉重复刷新 ✅、增强图后台 ✅、标注写入后台 ✅、搜索扫描去抖 + 后台 ✅（§4.13）、**D35/D36 各格式写入路径** ✅（§4.20）、导航可见集唯一真值 ✅（§4.22）。目录列举实测 5000 张 8ms（一次性），不列为缺口 | 已达成：切图/增强/自动保存/搜索都不再阻塞 GUI（分别 2.1ms / 0.1ms / 2.3ms / 1.4ms 返回）✅、mask/sa1b 非项目写入从"必然失败"变为可用且可读回 ✅、筛选后导航不跳到隐藏图 ✅ |
| **4 ✅** | 关闭协议 ✅（§4.4）；`processEvents` 重入 ✅（§4.7）；D14 `stop_and_wait` 互等已修 ✅（§4.18） | ✅「解码中关窗」无 Qt Fatal、排空期间不死锁、`shutdown()` 幂等（`test_shutdown.py`）；✅ 批量进行中界面可自由操作、取消有效、拒绝并发（`test_no_processevents.py`）；✅ 静态扫描 0 处 `processEvents(` 调用、0 个 `QThread` 子类、0 处 `.terminate()` |
| **5** | 数据安全修复：D1~D10 + D12（格式层原子写、删除顺序、撤销归属、模型线程状态机、缩略图 `_pending`） | 新增针对"删除图片副作用/跨图撤销/批量后当前标注归属/并发保存同路径"的回归测试 |
| **6** | 收尾：其余 QThread 反模式（`intent_agent`/`LLMWorker`）、运行时状态与受版本控制文件分离（D30）、日志轮转与工作线程异常入日志、`Interface` 生命周期钩子、`tests/` 套件补齐 | 全套测试通过；`core/core.json` 不再被运行时改写 |

---

## 4. 已完成的改动（含验证证据）

### 4.1 阶段 1：图像解码管线（消除 abort）

新增 `plugins/annotation/widgets/image_decode_service.py`：

- **不继承 `QThread`** —— 线程由 `QThreadPool` 持有并回收，调用方不需要也无法"丢弃线程引用"。
- **有界并发**（默认 3）+ **最新优先**：排队期间已过期的任务在 `run()` 入口直接返回，不做 I/O。
- **代次权威在调用方**：服务只通过构造时注入的 `is_current` 只读查询，服务自身**不维护第二个计数器**
  （避免再次出现"两处代次错位"这类缺陷的放大版）。
- **不遮蔽 Qt 信号名**（对外只有 `ready`/`failed`）、**不调用 `terminate()`**、提供 `drain(timeout_ms)`。

`drawing_widget.py`：删除 `ImageLoader`；`set_image` 只 +1 代次并提交请求；`_on_image_decoded`
用 **代次 + 路径双重守卫**提交；`clear()` 改为非阻塞作废（移除 `wait(500)`/`terminate()`）；
`paintEvent` 改用 `self._decoder.is_busy()`；新增 `cancel_pending_load()` 与 `drain()`。
`annotation_interface._do_set_project` 的 loader 停机块替换为 `cancel_pending_load()`。

验证：`tests/annotation/test_switch_stress.py`

```
run(40, 20ms) exit=0   I3 解码并发峰值 2（上限 3）   PASS I1/I2/I3/I4/I5
run(80,  5ms) exit=0   I3 解码并发峰值 3（上限 3）   PASS I1/I2/I3/I4/I5
```
修复前同一配置：第 2 次切换即 `exit=-1073740791`（abort）。

> 过程中回归测试抓到我自己引入的一个静默 bug：`_DecodeTask` 调用的 `is_current` 在服务端
> 名字写错成 `_is_current`，被 `except Exception: return False` 吞掉 -> **所有解码任务静默空转、
> 一张图都不显示**。已修，并移除那个吞异常的写法。

### 4.2 阶段 1.5：列表重建泄漏（消除"越来越卡"）

`annotation_interface.py`：新增 `_clear_list_widget()`（`removeItemWidget` + `setParent(None)` +
`deleteLater()`），用于 `refresh_label_list` 与 `refresh_category_list`。

验证：`tests/annotation/test_label_list_leak.py`

```
修复前: 120 -> 3720（+3600，每次重建泄漏 120 个控件）  FAIL      单次重建 128.5 ms
修复后: 120 -> 120  （+0）                          PASS      单次重建 ~120 ms
```

### 4.3 顺带确认的两条"必现"结论（独立复现）

- `from core.backend.formats import find_label_file` -> `ImportError`（实测）。
- `AutoAnnotationService` 无 `find_and_load_annotations` / `save_image_annotations`
  （仅 `ManualAnnotationService` 有）-> `batch_ai_convert` 必 `AttributeError`。

### 4.4 阶段 4（部分）：可确定性关闭

新增 `core/lifecycle/shutdown_coordinator.py`：

- 契约 `drain(timeout_ms) -> bool`；`register_drainable(name, obj)` / `shutdown(total_ms)`。
- **总预算**而非每项各一份超时（避免 N 个 drainable 把关闭拖成 N×timeout）；**LIFO**；**幂等**
  （`closeEvent` 与 `aboutToQuit` 都会调用）；`finally` 后**保持闸门关闭**，让"退出期还去投递
  主线程任务"快速失败而不是挂住解释器收尾。
- **排空期间禁止 `run_on_main`**：`core/common/main_thread_dispatcher.py` 新增关闭闸门，
  关闭期间工作线程的投递**快速失败为 `RuntimeError`**。这直接消灭了审计 D14 那类互等
  （主线程阻塞在排空里、工作线程阻塞在无超时的 `ctx.done.wait()`）。

接入点：`AnnotationInterface.drain()`（先解码后缩略图）→ 在 `init_ui` 登记为
`annotation.interface`；`MainWindow.closeEvent` **第一件事**就是排空（原先只保存 AI 会话，
而 `sidebar.shutdown()` 是 `pass`）；`core/main.py` 挂 `aboutToQuit` 兜底。

验证：`tests/annotation/test_shutdown.py`

```
已登记的 drainable: ['annotation.interface']
切图 120 次：解码并发峰值=3  缩略图并发峰值=4  缩略图在途峰值=4
排空耗时 40 ms -> test.marshal_probe=ok(2ms), annotation.interface=ok(38ms)
PASS: A/B/C/D/E/F 全部通过     退出码 0
```
其中 B 断言用一个 `drain()` 期间启动工作线程投递主线程任务的探针，验证**快速失败而非挂住**。

### 4.5 阶段 5（部分）：缩略图在途集合收敛 + 线程池排空（D19）

`thumbnail_manager.py`：`run()` 改为 `finally` 中**必定**发射 `done`（失败发 null QImage，
接收端已有 null 分支），因此 `_pending` 不再永久滞留；新增 `active_count()` /
`pending_count()`（可观测）与 `drain(timeout_ms)`（实现 Drainable 契约，刻意不调用
`QThreadPool.clear()` 以避开"Qt 删除 Python 持有的 QRunnable"这类边界）。

验证：`tests/annotation/test_thumbnail_pending.py`

```
P1 请求缺失文件: 在途(立即)=1 -> 收敛后=0  收敛=True     （修复前永久停在 1）
P2 再次请求同一缺失文件: 在途=1（可重试）
P3 正常文件: 收敛=True 在途=0
P4 drain(5000)=True 活跃线程=0
PASS  退出码 0
```

### 4.6 阶段 5（部分）：原子写 + 删图安全

**原子写（D5）** 新增 `core/common/atomic_io.py`：`atomic_open` / `atomic_writer_path` /
`atomic_write_text` / `atomic_write_bytes` / `atomic_write_json`。要点：

- 临时文件与目标**同目录**（`os.replace` 只有同卷才原子；放系统临时目录会跨卷退化成
  拷贝+删除，Windows 上还可能直接失败）。
- 临时文件名**改变扩展名结尾**（`foo.json` -> `foo.json.atomic-<pid>-<n>`），避免被
  按扩展名过滤的扫描/遍历（`scan_image_files`、`os.walk` 找图、`endswith('.json')` 的
  类别重写循环）当成真实数据。代价是调用方不能从路径推断格式，因此
  `imwrite_unicode` 新增 `ext=` 显式格式参数。
- 失败时删除临时文件并**保留原文件**，异常继续上抛（不吞）。
- 默认 `fsync=False`：要防的是**进程崩溃**（本项目是 `qFatal -> abort`），此刻
  `close()` 后 `os.replace` 的元数据操作即已完成；防掉电才需要 fsync。

**替换范围**：`core/backend/formats/{labelme,coco,yolo,voc,mask,sa1b}_format.py` +
`base.py` + `manual_annotation_service.py`，共 **32 处**（含 `tree.write`、`cv2.imwrite`、
`imwrite_unicode`、`json.dump` 四种形态）。用一次性脚本按精确文本替换并**校验命中次数**
（`scripts/_apply_atomic_writes.py`，CRLF/LF 双模式、幂等），避免静默半替换。

**删图安全（D1 / D10 / D34）**：

- `delete_image` 改为三阶段：**先只解析**待删清单（任一环节失败则一个文件都不删）→
  **先删图片本身，删不掉就直接返回**（避免"标注已删、图片还在"的反向孤儿）→ 再清理关联文件。
  import 修正为 `core.backend.utils.find_label_file`。
- 新增共享文件保护：`find_label_file` 可能返回**共享**标注文件（`_annotations.coco.json`）
  或共享掩码，盲删会毁掉整个数据集；现在只报告、不删除（`warnings`）。
- `_remove_with_retry`：对 `PermissionError` 做**有界**重试（4 × 30ms），避免 GUI 线程卡顿。
- `DrawingWidget.wait_io_idle` / `ImageDecodeService.wait_idle` / `ThumbnailManager.wait_idle`：
  删除前等在途解码结束（**无副作用**：不作废代次、不取消请求）。这三者是 D34 的修复。
- `delete_image_file` 的 `skip_confirm` 真正生效（默认弹确认框），并把共享文件警告回传给 agent。

验证：`tests/annotation/test_atomic_write.py`、`tests/annotation/test_delete_image.py`

```
A6 真实 save_annotations 中途崩溃: 原标注文件完好  OK      （修复前必然被截断+半写）
B1 刚打开目录即删 img1.png -> 'success'                （修复前 WinError 32）
B3 deleted=['img3.png'] warnings=['标注位于共享文件，已保留未删除...']
B4 success=False deleted=[]                            （修复前：图已删、标注留下）
B5 拒绝确认 -> '已取消删除' 文件仍在=True / 同意确认 -> 文件已删=True
```

---

### 4.7 阶段 4（完成）：消除 processEvents 重入

原状：`annotation_interface.py` 有 **7 处** `QApplication.processEvents()`，根源都是
"长任务在 GUI 线程同步执行"——不让出事件循环，进度条就不重绘、取消按钮就点不动。
而在**状态变更处理器内部**重入事件循环，会让用户在批量标注进行中继续触发
切图/保存/删除，界面状态与磁盘状态随之错乱（审计 P1）。

| 场景 | 处理 |
|---|---|
| 批量标注进度（2 处） | 批量整体搬进新的 `core/common/background_task.py::BackgroundTaskRunner`（QThreadPool / 单任务语义 / 实现 Drainable）。GUI 线程空闲后 Qt 自然重绘，取消按钮自然可用 |
| `update_save_path` 三步扫描（3 处） | 该进度对话框 `has_cancel=False`，本就不需要事件循环来处理点击；改为 `progress.repaint()` —— 同步重绘**该控件**但不派发任何输入/定时器/网络事件 |
| `convert_all_with_ai`（2 处） | 同上用 `repaint()`。附带发现：它的 `QProgressDialog` 带"取消"按钮却**从未接任何取消逻辑**，服务侧也没有 cancel 入口 -> 取消一直是假的（并入 D9，阶段 5c 处理） |

**行为变更（有意，且已核对全部调用点）**：`batch.one_click_by_all_ui` 现在**立即返回**
（"已在后台开始"），不再返回同步结果；7 个 UI 调用点都是菜单 lambda、忽略返回值，
需要同步结果的 agent 应使用 service 层 `batch.one_click_by_all`（`background=True`）。
同时补上"已有批量在跑则拒绝"的并发保护（原先 `one_click_running` 只写不读，
两次批量可真并发）。关闭流程会先 `cancel_batch()` 再排空 runner，避免长批量把关闭拖到超时。

验证：`tests/annotation/test_no_processevents.py`

```
N1 静态检查: plugins/annotation 下 0 处 processEvents 调用  OK
N2 发起返回耗时 0.8 ms -> '批量标注已在后台开始（共 6 张图片），进度见进度对话框'
N2 事件循环: 共 14 次 tick，其中批量运行期间 6 次      <- 同步执行时此断言必然失败
N3 取消点=处理到第 2 张，结果 processed=3 total=6 canceled=True
N4 并发第二次发起 -> '错误: 已有批量标注正在进行中，请等待结束或先取消'
N5 进度对话框已关闭=True
PASS: N1/N2/N3/N4/N5 全部通过
```

N2 是最有判别力的一条：若批量仍在 GUI 线程同步执行，事件循环会被阻塞，
`ticks_while_busy` 必为 0。N1 是**永久性不变量**，防止重入被重新引入。

### 4.8 阶段 2（完成）：唯一状态所有者 + 原子提交

**问题**：同一件事有三份真值 —— `interface.current_image_path`、
`service.current_image_path`、`widget._current_image_path`，再叠加
`service.current_annotations`（widget 通过 property 读到的是它的**别名**）。
于是"当前图片"与"当前标注"可由三处分别改写，中间出现

    (图 N-1 的 pixels / original_image_size / zoom / pan) + (图 N 的 path / annotations)

这种状态既不自洽、又会被任何一次自动保存写进磁盘（审计 W1/W2 / D8）。

**新增 `plugins/annotation/session.py::ImageSession`** —— 唯一所有者，持有
`(path, generation, annotations, image, pixmap, original_image_size, display_scale)`：

- `begin_load(path)`：换代次 + 改路径 + **清空"已提交的像素与度量"**。清空是关键：
  "像素/度量属于哪张图"不再靠约定，而由 `is_committed` 直接回答。
- `commit_pixels(gen, ...)`：**只有代次匹配**才落地（过期结果静默丢弃）。
- `set_annotations(anns, for_path=...)`：`for_path` 不匹配则**拒绝装载**——这是
  审计 D2/D3 那类"把 A 图标注装进 B 图当前状态"的直接拦截点。
- `is_save_allowed`：装载失败（文件损坏/编码错误）时为 False，`save_current` 拒绝落盘，
  避免把空标注覆盖到那张可恢复的坏文件上（D8）。
- `is_committed`：交互与坐标换算的唯一闸门。

**三处真值归一到只读投影**：service 与 interface 的 `current_image_path`、
`current_annotations`，widget 的 `image` / `pixmap` / `original_image_size` /
`display_scale` / `current_image_path` 全部改成**没有 setter 的 property**。
任何"我再存一份"的代码会立刻 `AttributeError`，而不是留下第二份真值
（测试 O1 断言 7 个属性赋值全部抛 `AttributeError`）。7 个赋值点已改为走
`session.begin_load()` / `session.clear()`。

**交互闸门**：`to_image_coords` 在未提交时返回 None（所有坐标换算被拒）；鼠标三事件、
滚轮、键盘、`add_sam_point` / `add_polygon_vertex`、`toggle_roi_zoom` 全部前置
`_interaction_ready()`；`mouseReleaseEvent` 被拒时调用新的
`_reset_transient_drag_state()`（同时修掉 D7：20 个下标/坐标型瞬时状态在切图时统一作废，
不再出现"按住左键切图后用旧坐标向新图提交标注"）。`get_state` 新增
`canvas_committed` 标志，避免 agent 在未提交时按旧分辨率生成坐标。

**行为变更**：`set_qimage` 现在也会**换代次**（原先不换，在途解码会静默覆盖刚设的图）。

验证：`tests/annotation/test_single_state_owner.py`

```
O1 三份真值均已只读（7 个属性赋值全部 AttributeError）  OK
O2 原子提交：5 次切换的窗口内均未提交、度量已清空、标注已属新图、坐标换算被拒  OK
O3 过期提交被拒 + set_qimage 换代次  OK
O4 未提交时交互被闸门拦住（坐标换算/SAM/多边形）  OK
O5 为其它图准备的标注被拒绝装载  OK
O7 装载失败禁止保存 / 装载成功正常保存  OK
O6 三处投影与 session 全程一致（24 次采样，246 ms）  OK
PASS: O1~O7 全部通过
```

O2 是本轮最有判别力的一条：它在 `load_image(i)` **返回后、事件循环跑之前**断言
画布未提交、度量已归零、坐标换算被拒、标注已属于图 i（用"标注自带图名标签 +
每张图尺寸各不相同"两个手段让归属可判别），并在提交后校验度量等于**那张图**的真实尺寸。

### 4.9 阶段 3（部分）：逐图标注装载移入后台 + 去掉重复刷新

**问题**：`load_image_annotations()` 在 GUI 线程**同步**读盘 + 解析（非项目模式还会
降级二次扫描），发生在每一次切图上 —— 这是"IO 全在后台"这条要求里最后一块同步 I/O。
紧随其后 `load_image` 又调一次 `refresh_label_list()`，于是每次切图全量重建两遍标注
列表（20 条标注实测约 240ms）。

**做法**：复用 `BackgroundTaskRunner`（阶段 4b 引入、已测）新建
`_annotation_runner`（单任务）：

- **带代次校验**：请求携带 `(generation, path)`，结果只有在二者都仍匹配时才提交；
  过期结果打印一行日志后丢弃。
- **最新优先**：正在跑时新请求只覆盖 `_pending_annotation_request`（深度 1），跑完再发起，
  因此连按方向键不会积压一串读盘任务，中间图直接跳过。
- **提交顺序修正**：`clear_visibility_state()` 改为 `set_annotations([], loaded=False)`，
  于是装载窗口内 `is_save_allowed` 为 False、`save_current` 被拒 —— 不会把上一张图的
  标注写进这张图的文件（这是异步化**必须**配套的保证，否则异步反而放大了 D8/D2/D3）。
- **交互闸门升级**：`_interaction_ready()` 改用新的 `session.is_ready`
  （= `is_committed and is_annotations_loaded`）。标注装载到位前不允许绘制，否则用户
  刚画的内容会被随后到达的"这张图既有标注"整体替换掉。
- 去掉 `load_image` 里重复的 `refresh_label_list()`；`drain()` 纳入装载器。

**顺带修掉 `BackgroundTaskRunner.drain()` 的一个语义漏洞**：`_inflight` 原本只在
队列化的交付槽里归零，而 `drain()` 是同步等待、不跑事件循环 —— 于是排空后 `busy()`
恒为 True，且**已经排在事件队列里的结果仍会在排空后派发**，把过期状态写进正在被拆掉的
界面。现在 `drain()` 会清掉在途标记与回调（排空 = 弃置在途结果）。

验证：`tests/annotation/test_annotation_io_async.py`

```
P1 load_image 返回耗时 2.1 ms（慢 loader 250 ms），装载期间 tick=12     <- 同步执行时必然失败
P2 采样 97 次；最终图=img02.png 标注=['tag:img02.png']；越权落地 0 例
P2 loader 调用: ['img00.png', 'img02.png']                             <- 中间图 B 被合并
P3 装载窗口内 is_save_allowed=False is_ready=False save_current=error
P4 每次切图的 refresh_label_list 次数: [1, 1, 1, 1]                     <- 修复前为 2
P5 drain(5000)=True  装载器仍忙碌=False
PASS: P1~P5 全部通过
```

P2 的 97 次采样逐次校验"任何时刻落地的标注都必须属于当时的当前图"，是"过期结果
绝不落地"的直接证据；P1 用"慢 loader 下切图 2.1ms 返回 + 事件循环照常 tick"证明
GUI 线程确实不再做标注 I/O。

### 4.10 阶段 3（部分）：增强图移出 GUI 线程 + 结果校验（D17）

**问题**（审计 D17 的两处）：
1. `_generate_enhanced_pixmap()` 在 **GUI 线程**同步执行 LibLLIE + numpy 全图运算，
   而 `_on_image_decoded` / `set_qimage` 每次提交都会调用它 —— 开启增强后每次切图都冻结。
2. 对增强器返回值**零校验**：`h, w, ch = enhanced_arr.shape` 后用 `ch * w` 当
   bytesPerLine 构造 QImage。返回 float32/uint16 时真实行距是 `12*w`/`6*w`，
   于是逐行越界读 numpy 缓冲（花屏或崩溃）；返回 1-D 则直接 ValueError。

**做法**：新增 `_enhancer_runner`（`BackgroundTaskRunner`，单任务）+
`_normalize_enhanced_array()`：

- 增强在**工作线程**执行（QImage 是可重入值类型、只读，且提交只替换引用不改写对象，
  因此跨线程读安全）。
- 结果规范化：RGBA 降成 RGB、非 uint8 先 clip 再转、**强制连续**、形状/尺寸非法直接抛错。
  因此构造 QImage 时行距必然等于 `3*width`，不可能越界。
- **代次 + 路径双重守卫**：过期结果绝不安装（否则会把 A 图的增强图贴到 B 图上），
  过期请求直接丢弃。
- `_enhance_requested`：用户请求增强后跨切图继续生成（原先只在 `_show_enhanced`
  为真时才生成，而 `_show_enhanced` 要等首次生成成功才置位）。
- 增强器不可用（缺 libllie）时置 `_enhancer_unavailable`，本次会话不再重试，避免每次
  切图刷一屏 traceback。
- `DrawingWidget.drain()` 纳入增强线程池。

副作用说明：`toggle_enhance()` 首次调用不再同步返回成功，而是返回
「增强图正在后台生成，完成后自动切换显示」，生成完成后自动置 `_show_enhanced`。

验证：`tests/annotation/test_enhance_async.py`

```
F3 结果校验：RGBA/float32/非连续/畸形形状 处理正确  OK
F1 _generate_enhanced_pixmap 返回 0.1 ms（慢增强器 250 ms），忙碌=True，生成期间 tick=12，结果已安装=True
F2 安装的增强结果=['img02.png']  被丢弃的过期结果=['img01.png']  过期到达时是否误装=False
F4 drain(5000)=True  增强线程仍忙碌=False
F5 已受理=True  增强器不可用标记=True
PASS: F1~F5 全部通过
```

F2 是代次守卫的直接证据：A 的增强结果在**切到 B 之后**才到达，被丢弃且未安装
（`过期到达时是否误装=False`），最终只安装了 B 的结果。F1 用"慢增强器下 0.1ms 返回 +
生成期间事件循环 tick 12 次"证明 GUI 线程不再被增强阻塞。

### 4.11 阶段 5（部分）：模型加载线程（D12）—— 标注插件内清零 QThread

**问题**（审计 D12，与阶段 1 修掉的 `ImageLoader` 是同一类）：`ModelLoadThread(QThread)`
把 `finished = Signal(bool, str)` 定义在自己身上，**遮蔽了 `QThread.finished`**，而且在
`run()` **内部** emit —— 线程还没真正结束就宣告完成，于是 `_model_loading` 被提前清成
False。之后若再调一次 `load_model_async`，就会
`self.load_thread = ModelLoadThread(...)` **覆盖仍在运行的 QThread 的引用** →
引用计数归零 → `QThread: Destroyed while thread is still running` → `abort()`。

**做法**：删掉 `ModelLoadThread`，改用 `BackgroundTaskRunner`（单任务、拒绝并发、
实现 Drainable、不遮蔽任何 Qt 信号名），完成状态用**代次跟踪**：

- 线程由 `QThreadPool` 持有，调用方**无法**丢掉线程引用。
- `start()` 被拒（已有一个在跑）时**保持** `_model_loading = True`，不谎报"未在加载"。
- **代次只在启动成功之后才提交**：这条是回归测试抓出来的 —— 最初我把
  `_load_generation += 1` 放在 `start()` 之前，于是被拒绝的重复请求会把代次推进，
  真正在途的那个任务的完成回调被判为"过期"而丢弃，`_model_loading` 永久停在 True。
- 过期完成回调不得清掉当前加载的状态（原先的"误判"）。
- 注册进 `AnnotationInterface.drain()`（`model_loader`）。

**据此得到的结论性证据**（目标的"不得再出现 QThread 析构崩溃或 terminate 死锁"）：

```
plugins/annotation 下的 QThread 引用：仅 QThread.currentThread()（只读查询）与 QThreadPool
QThread 子类：0 个
.terminate() 调用：0 处
```

验证：`tests/annotation/test_model_load_thread.py`

```
M0 静态检查：model_manager 代码内 0 处 QThread  OK
M1 加载期间重复请求被拒（只启动 1 个后台任务）  OK
M2 强制清零标志后连发 11 次：实际启动任务=1  标志=True
M6 连发结束：总任务=1  完成信号=1  标志=False
M3 过期完成回调：标志保持=True？True  完成信号数=0
M4 完成后可再次发起加载  OK
M5 drain(5000)=True  加载器仍忙碌=False
PASS: M0~M6 全部通过
```

M2/M6 是原缺陷路径的直接复现：强制把 `_model_loading` 清零后连发 11 次（等价于原先
"覆盖 `load_thread` 引用"的触发条件），新实现下**只启动 1 个后台任务、发 1 次完成信号、
进程正常退出（退出码 0）**；旧实现会在第 2 次覆盖时 abort。

### 4.12 阶段 3（部分）：标注写入移入后台（含排序与失败上报）—— 并暴露两个自查缺陷

**问题**：逐图标注的**写盘**原先在 GUI 线程同步执行，触发点是"每次标注修改"和
"每次切图的自动保存"（全部 10 处 `save_current(silent=True)` 的调用方都不使用返回值）；
`save_image_annotations` 内部还要 `imread_unicode` 读回整张图只为取宽高。

**做法**：新增 `plugins/annotation/services/annotation_writer.py::AnnotationWriter`
（标准库 `threading.Thread` + 条件变量）作为**标注写入的唯一出口**：

* **全局串行**：同一时刻只有一个写入任务在跑 -> 不可能乱序覆盖；
* **同路径最新优先**：排队期间又来一次保存，只保留最新快照（旧的直接丢弃）；
* **`wait_for_path(path)`**：标注装载在开始读之前调用它（装载本来就在工作线程上，
  阻塞等待无代价）—— 这是异步化**必须**配套的排序保证，否则"切走再切回同一张图"
  会读到旧内容；
* 显式保存（`silent=False`，Ctrl+S / agent action）先 `wait_for_pending_writes()` 再同步写，
  保证写入顺序不受排队影响；
* **失败必须上报**：异步路径没有返回值，写盘失败（例如编码失败）若不提示就是
  **静默丢数据**。`set_failure_handler()` 的回调在写入线程上被调用，界面通过
  `run_on_main` marshal 回 GUI 线程后 `InfoBar.error` + 状态栏提示。
* `drain()` 排空队列并 join 写入线程，已注册进 `AnnotationInterface.drain()`。

**自查缺陷 1（严重回归，阶段 2 引入）**：`find_and_load_annotations` 把"这张图**还没有**
标注文件"和"标注文件存在但装载失败"**共用同一个 `'error'` 状态**。阶段 2 的 D8 保护
（装载失败禁止保存）因此把**全新的图**也判成装载失败 -> **永远无法保存**——新数据集
根本无法标注。修复：新增 `'empty'` 状态表示"还没有标注（或文件本来就是空的）"，
只有"文件存在但解析抛错"才返回 `'error'` 并禁止保存。两个现有消费方
（`get_cached_annotations`、`auto_annotation_service.batch_ai_convert`）都只看
`status == 'success'`，新增状态对它们无影响。

**自查缺陷 2（D35，已在 §4.20 修复）**：`save_image_annotations` 的兜底分支
（`else:`，覆盖 `mask` / `sa1b` 等非 JSON 格式）把输出路径**硬编码为 `<base>.json`**：

```
core/backend/formats/mask_format.py:179
    imwrite_unicode(_tmp, mask, ext=os.path.splitext(output_path)[1])   # ext='.json'
-> OpenCV: could not find encoder for the specified extension
```

即**非项目模式下 mask/sa1b 格式的保存必然失败**（异常被 `save_image_annotations` 的
`except` 吞成 `{"status":"error"}`）。相关观察（D36）：目录里没有任何标注文件时，
`_non_project_format` 会被解析成 `mask`，与 D35 组合就是"新目录保存必然失败"。
本轮只把它记入清单并让**失败上报**把它显式暴露出来，不在同一轮里顺手改格式/子目录语义
（掩码落盘可能与本目录下的源图同名，需要单独确认 `subdir`/`ext` 语义后再动）。

验证：`tests/annotation/test_save_async.py`

```
S1 save_current(silent=True) 返回 2.3 ms（慢写盘 300 ms），ret=queued，在途=1，写盘期间 tick=15
S1 落盘内容正确: ['s1']  OK
S2 连续 5 次自动保存 -> 实际写盘 2 次；最后一次快照=img00.png|['s1', 's2']
S3 装载前该路径有在途写入=True；装载后标注=['s1', 's2', 's3']
S3 装载读到的是刚保存的内容（读等待写入）  OK
S4 装载失败后 schedule_save_current=error 写盘次数=0
S5 显式保存返回=保存成功: 当前 旋转检测 标注已保存；最终文件=['s1', 's2', 's3', 's5']
S7 写盘失败上报次数=1  内容=[('...\\img00.png', '模拟编码失败')]
S6 drain(5000)=True  剩余在途=0  写入线程存活=False
PASS: S1~S7 全部通过
```

S2 是合并效果的证据（5 次自动保存 -> 2 次写盘），S3 是排序保证的证据（立刻切回同一张图，
读到的是刚保存的 `s3` 而不是旧内容），S5 证明显式保存不会被排队中的旧快照覆盖
（最终文件同时含 `s5`），S7 证明异步失败不再被静默吞掉。

**同时修正的测试前置条件**：`test_label_list_leak` 的基线必须在**标注装载落地之后**再取
（阶段 3a 把装载改成异步后，迟到的装载结果会清空列表，产生"基线 0 -> 之后 120"的假泄漏；
修好后基线 120 -> 120、增长 +0）。

### 4.13 阶段 3（部分）：搜索扫描去抖 + 后台化（D15）

**问题**：`search_box.textChanged` 直接连到 `self.search_images`，于是**每次击键**都在
GUI 线程做一次全量扫描；一旦筛选器有效（类别/尺寸/标注状态/划分），扫描会逐图
`get_cached_annotations` → **读盘 + json.load**。数千张图片时这是秒级到分钟级冻结。

**做法**：

* `textChanged` 改连 `_on_search_text_changed`：只**置脏 + 去抖**（150ms 单次定时器），
  不扫描。纯数字输入（"跳转到第 N 张"）仍走同步导航路径，保持原有手感。
* 去抖到点 → `_request_background_search()`：在 GUI 线程取一份
  `manual.search_snapshot()`（image_files / image_splits / 6 个 filter 字段的副本），
  交给 `_search_runner`（`BackgroundTaskRunner`，单任务）在后台执行。
  **快照是必需的**：否则工作线程迭代 `image_files`、读 `filter_*` 时会被 GUI 线程的
  修改撕裂（列表被整体替换、筛选条件改到一半）→ 自相矛盾的可见性结果。
* **代次校验 + 最新优先**：结果代次不匹配直接丢弃；正在跑时新请求只覆盖待发请求。
* **显式搜索**（清筛选后的重算、agent/UI 内部调用）会先递增代次，使在途后台结果作废，
  自己仍同步返回 —— 调用方行为不变。
* `manual.search_images(..., snapshot=..., apply_state=False)`：快照执行时**不回写**
  `self.file_visibility`（那是 GUI 线程状态），由界面在提交结果时统一回写。
* 搜索扫描器已注册进 `AnnotationInterface.drain()`。

验证：`tests/annotation/test_search_async.py`（12 张图，每次"读盘"30ms，搜索框带
`@划分:未划分` token 以保持筛选器有效 —— 这正是逐图读盘的真实触发条件；
纯文本输入会因 `_apply_search_tokens` 清除筛选而不触发读盘，这一点测试中也断言了）

```
Q1 5 次击键最慢一次 1.4 ms；去抖窗口内读盘=0 次；实际扫描轮数=1.0（应≈1）；扫描期间 tick=27
Q5 去抖窗口内（击键之间）读盘次数=0
Q2 第一次扫描已启动=True；返回的关键词=['a', 'img0']；被应用次数=1
Q3 显式搜索返回=None；显式后立即应用=1 次；最终应用=1 次；后台关键词=['a']
Q4 drain(5000)=True  搜索扫描器仍忙碌=False
PASS: Q1~Q5 全部通过
```

Q1/Q5 是主断言：5 次击键最慢 1.4ms、去抖窗口内**一次盘都没读**、最终只跑 1 轮扫描
（修复前是 5 轮、每轮 12 次读盘），且扫描期间事件循环 tick 27 次 —— 扫描确实在后台。
Q2 证明确实取到了过期结果（两个关键词的结果都回来了）但只应用了最后一次。

### 4.14 阶段 5d：跨图写入的路径校验（D2 / D3）

这两个缺陷是同一类：**"当前"状态被别的图的数据污染，随后被保存写进当前图的文件**。
它们正是"带代次/路径校验的原子提交"这条要求在最后两处入口上的漏点。

**D3（撤销栈跨图）**：`_undo_stack` 里每帧都存了 `image_path`，但 `undo`/`redo`
**从不比对**，弹出后直接 `save_current(annotations=restored)` → 用**当前**路径落盘。
修法：

* 新增 `_pop_undo_frame(stack, kind)`：出栈即校验帧的 `image_path` 是否属于当前图，
  不一致就**清空撤销历史并拒绝**（栈已跨图失效，留着只会在下一次撤销时再次误写），
  返回的错误信息明确指出是"另一张图片"。
* 保留第一道防线（`load_image` → `clear_undo_history()`），并在测试里分别断言两道防线：
  U1a 切图后栈为空；U1b 即使栈里残留了属于别的图的帧（路径传入过期值，或将来某个入口
  忘了清栈），也必须被路径校验拦住。

**D2（批量写回污染当前状态）**：`auto_annotation_service.annotate_by_current` 末尾为了
让画布刷新，执行了 `self.manual.current_annotations = list(merged)` —— 无条件把
"最后一张批量图"的标注塞进"当前图"的状态。修法：改用受控入口
`set_current_annotations(merged, for_path=image_path)`，路径不匹配时**直接拒绝写回**
（画布保持不动），匹配时才写回（画布刷新依赖它，B2 断言这一点）。

顺带说明：阶段 2 把 `current_annotations` 变成**无 setter 的只读 property**之后，
D2 那行赋值会立刻抛 `AttributeError` —— 也就是说这个缺陷已经从"静默数据损坏"变成了
"批量标注必崩"。修复后两者都不再发生。

验证：`tests/annotation/test_cross_image_writes.py`

```
S1 静态检查：三层内 0 处对只读真值的赋值  OK
U1a 切图前栈=1；切图后栈=0
U1b 跨图撤销返回 err='错误: 撤销栈里的操作属于另一张图片(img00.png)，已清空撤销历史，以免把旧图标注写入当前图的标注文件'；img01 文件 前=None 后=None；栈剩=0；当前标注=[]
U2 同图撤销 err=None；恢复=['base']；文件=['base']
B1 为 img01 准备的标注写回 img00 当前状态：受理=False；当前标注未变=True
B2 为当前图准备的标注写回：受理=True；标签=['A', 'FROM_CURRENT']
B1 收尾：img00 文件=['A', 'FROM_CURRENT']（不得含 FROM_IMG01）
PASS: U1/U2/B1/B2/S1 全部通过
```

**S1 是一条常驻的静态不变量**（把阶段 2 那次的临时扫描变成回归）：三层内任何对
`current_image_path` / `current_annotations` / `original_image_size` / `display_scale`
的赋值都会让测试失败 —— 它本来就能在 D2 那行赋值上报警。

### 4.15 阶段 5d：目录串号（D4）与"AI 全量转换"不可用（D9）

**D4 非项目目录串号**：`enter_non_project_mode` 用
`if not self.context._non_project_save_dir:` 判断"是否首次进入"，于是**先开目录 A、
再开目录 B** 时 `output_dir` 仍然是 A —— B 的标注被写进 A 里，同名 basename 直接
**覆盖别人的标注文件**。修法：把"当前非项目目录的保存位置"与"上次打开过哪个目录"
彻底拆开 —— `_non_project_save_dir` 每次进入**无条件**指向本次目录
（"上次目录"由 `last_opened_dir` 这类设置项承担，不复用这个字段）。

**D9 "AI 全量转换" 100% 不可用**：`batch_ai_convert` 里以 `self.` 调用了
`find_and_load_annotations` / `save_image_annotations`，而这两个方法**只存在于
`ManualAnnotationService`** 上 —— 第一次迭代必然 `AttributeError`。修法：新增
`_load_annotations_for_convert` / `_save_converted_annotations` 两个薄委托方法，
显式走 `self.manual`，并在 manual 缺失时返回**可诊断的失败**而不是崩栈。

验证：`tests/annotation/test_non_project_dirs.py`

```
D9-1  静态检查：auto_annotation_service 内 0 处 self.<manual 专有方法>  OK
D4-1  A 的保存目录=dataset_A；进入 B 后的保存目录=dataset_B
D4-2  B 保存后：B 文件=['FROM_B']；A 文件 前=['FROM_A'] 后=['FROM_A']
D4-3  回到 A 后的保存目录=dataset_A
D9-2a 委托读路径：status=success 读到 1 条标注
D9-2b batch_ai_convert 结果={'converted_count': 0, 'empty_results_count': 1, ..., 'total': 1}；异常=[]
PASS: D4-1/D4-2/D4-3/D9-1/D9-2a/D9-2b 全部通过
```

两个目录用**同名图片**（`same.png`）刻意制造 basename 冲突 —— 这正是原缺陷
"覆盖别人标注文件"的触发条件，D4-2 断言 A 的文件前后一致。

**关于 D9-2b 的 0 转换**：`convert_bbox_to_polygon` 需要真实分割模型，本环境没有模型，
因此返回 `No objects found`、1 条标注记为"空结果"。这是**环境性降级**，不是缺陷复活 ——
所以 D9-2b 断言的是"不崩 + 返回结构完整 + 计数自洽（说明确实读到了源标注）"，
另用 D9-2a 直接证明委托读路径可用。测试里写清了这个区分，避免以后把环境问题读成回归。

### 4.16 阶段 6：`stop_and_wait` ↔ `run_on_main` 互等（D14）

**死锁路径**（先确认了它真实可达，不是理论推测）：

```
非 background 的 action 都经 ActionTool.execute -> run_on_main 派到主线程执行
        ↓
agent_chat(..., max_wait=600000) 本身也是这样一个 action -> 在**主线程**上执行
        ↓
它调用 stop_and_wait() -> QThread.wait(max_ms) 阻塞主线程（不跑事件循环）
        ↓
worker 此刻正 run_on_main(...) 等主线程 -> ctx.done.wait()（无超时）
        ↓
主线程在 wait() 里不会去执行队列里的任务 -> 互等，界面冻结最长 10 分钟
```

**修复（三处）**：

1. `main_thread_dispatcher.call()` 增加可选 `abort_event`：等待侧改为轮询
   （`ctx.done.wait(poll_ms)`），一旦 abort 立即抛 `MainThreadCallAborted`，
   不再无限等待。
2. **放弃后必须把 ctx 从队列里摘掉**（本轮新发现的第 2 层缺陷）。
   原实现只 `ctx.abandoned = True`，那条 record 与对应的 `_execute_all` 仍留在
   Qt 事件队列里，会在下一次派发时被取出执行 —— "调用方已放弃的任务"照样占住
   主线程（若它是阻塞型任务，界面真的卡死），同时破坏了"abandoned 任务绝不执行"
   的保证。新增 `_discard_abandoned(ctx)`，在 abort 时摘除该 record（以及其它已
   放弃的 record）并唤醒其等待方；`_execute_all` 保留 `abandoned` 判断作为二重保险。
3. `LLMWorker` 的停止标志改为 `threading.Event`（`_stop_event`，`_stop_requested`
   保留为只读 property 以兼容旧读法），并通过 `_set_tool_abort_event()` 登记到各工具上
   （`ActionTool.abort_event`），使停止请求能**立刻打断** worker 在主线程调度器里的等待。
4. `VisionMindAgent.stop_and_wait()` 按调用线程分支：主线程上不再用 `QThread.wait`
   （那正是互等的一方），改为 `_wait_pumping_events()` —— 泵事件循环 + `worker.wait(1)`
   直到超时，界面保持响应且 worker 能真正退出；工作线程上仍用阻塞 `QThread.wait`。

验证：`tests/annotation/test_stop_and_wait.py`

```
D1 正常 run_on_main 返回 42、执行 1 次  OK
D2 abort 通道：结果=aborted 耗时=186 ms
D3 abort 后的待执行记录数=0（必须为 0）
D4 被放弃的任务是否被执行: []（应为空）
   5 次并发放弃后的待执行记录数=0（必须为 0）
D6 stop_and_wait 已按调用线程分支（主线程 -> 泵事件循环）  OK
D7 工作线程 stop_and_wait：等待中=True 返回=True 耗时=502 ms worker 已退出=True
PASS: D1~D7 全部通过
```

**这个测试为什么不去真的占用自己的主线程**：一旦阻塞型任务被派发，
`processEvents()` 就不会返回，测试自身也会卡死、无法再做任何断言。这一点是实测
踩到的坑（连续几版 harness 都栽在这里），因此改为**直接断言调度器契约**：abort 有界、
abort 后队列为空、被放弃的任务不执行、多次放弃不累积、分支选择正确、工作线程分支
仍真正阻塞等待。

### 4.17 阶段 6：运行时状态与受版本控制的配置分离（D30）

**问题**（两层）：

1. `core/core.json` 是**受版本控制**的配置文件，而 `settings.set()` 会整文件重写它。
   切图路径每次都调用 `save_last_index` → **每次切图一次整文件落盘**；而且会话状态
   混进版本控制文件后，该文件**永久 dirty**（git diff 全是噪声，还可能被误提交）。
2. `save()` 用的是裸 `open(self.filepath, 'w')` —— 既不原子也不互斥，而
   `settings.set()` 会从工作线程被调用（例如模型训练路径），并发保存会互相截断/交错。

**做法**：

* 引入 `SESSION_KEY_PREFIXES = ("last_idx_", "last_opened_dir")` + `is_session_key()`：
  这些键只存在于内存 `_data` 里，**不落盘**；`save()` 还会顺带清理文件里历史遗留的
  会话态键（旧版本写进去的）。`last_trained_model_path` 是跨会话有意保留的**用户选择**，
  刻意不在前缀列表内。
* 新增 `set_session()/get_session()`，把"这是本次运行内的状态"变成显式意图；
  `set()` 对会话态键自动降级为内存更新，因此**没有改调用方行为**也能立刻止血。
* 切图路径改为 `settings.set_session(...)`（`save_last_index`、`load_directory` 的读回）；
  另有静态回归保证 annotation 层不会再用 `settings.set` 写这些键。
* `save()` 改为 `_save_lock` 串行化 + `atomic_write_text`（阶段 5b 已有的原子写原语）
  —— 并发保存不再互相截断，也不会留下半截文件。

验证：`tests/annotation/test_settings_session_state.py`

```
R1 内存读回: idx=7 dir='D:\\some\\dir'
R2 落盘泄露的会话态键: []（必须为空）
R3 普通键是否落盘: True
R4 20 次会话态写入后文件是否未变: True
R5 清理后仍残留的会话态键: []（必须为空）
R6 并发保存：异常=[] 文件可解析=True
R7 直接 settings.set 写会话态的调用点: []（必须为空）
PASS: R1~R7 全部通过
```

R2+R4 是主断言：切图热路径**不再产生任何写盘**；R5 覆盖旧版本遗留脏数据的自愈；
R6 覆盖工作线程并发保存的原子性；R7 是常驻静态不变量。

**过程中自查到的一个错误**：我第一次把 `is_session_key` 的编辑没有落盘（`old_string`
未匹配），于是 `set()` 抛 `AttributeError: 'CoreSettingsManager' object has no attribute
'is_session_key'`；补回时又因为一次编辑吞掉了结尾换行，把 `load()` 的 docstring 与
下一行拼在了一起（`SyntaxError`）。两者都由"先跑数值探针 + `py_compile`"立刻暴露。
这也说明：改共享基础设施时，探针比通读代码更快定位问题。

### 4.18 阶段 6：刷新去重签名盲于几何 + 派生缓存不作废（D33）

**问题**（两层，都会让用户看到"改了但界面没变"）：

1. **签名盲于几何**：`_annotations_state_signature()` 只记录
   `len(a.get("polygons"))`，于是**移动一个多边形顶点**（元素个数、label、bbox 全没变）
   算出的签名完全相同。实测证据：

   ```
   sig1 = (1, (('a', '[1, 2, 10, 20]', 1),), frozenset(), 0)
   sig2 = (1, (('a', '[1, 2, 10, 20]', 1),), frozenset(), 0)   # 顶点已移到 [99,99]
   顶点移动后签名是否变化: False
   ```

   结果：`on_annotations_changed` 提前 return → 画布与列表**停留在旧几何**上。

2. **派生缓存不清**：`_last_ann_sig` / `_last_ann_path` 是"上次刷新过的状态"缓存
   （派生状态，不是标注真值），但 `clear_project` / `_do_set_project` 不重置它 ——
   切换项目后若新数据恰好同签名，刷新同样被跳过。

**做法**：

* 签名改为覆盖"真实编辑"的所有维度：bbox 转**数值元组**（原先用 `str()`，格式差异会
  造成假变化）、每条多边形的**顶点坐标**、`shape_type`，外加原有的 hidden_indices 与
  副标注计数。签名只用于"同一次事件的重复刷新去重"，因此**宁可灵敏也不要漏判**。
* 新增 `reset_annotations_refresh_state()`，在三个"标注真值被整体替换"的入口调用：
  切图（`load_image`，与 `clear_undo_history` 同处）、`_do_set_project`、`clear_project`。

验证：`tests/annotation/test_refresh_signature.py`

```
G1 顶点移动前后签名是否不同: True
G2 bbox 变化改变签名: True
G3 标签/数量变化改变签名: True
G4 可见性变化改变签名: True
G5 发布两次后的刷新次数: baseline=1 after_move=2
G7 同一状态连续发布 4 次的刷新次数: 1（应只刷新一次）
G6a 切图后派生签名缓存已作废: True
G6b reset_annotations_refresh_state() 生效: True
G6c clear_project / _do_set_project 均已作废缓存: True
PASS: G1~G7 全部通过
```

G5 是主断言：**只移动顶点**、其余全同，仍然产生第二次刷新（修复前会被抑制）。
G7 守住反向能力——去重不能被修掉（否则每次嵌套发布都全量重建列表）。
G6c 用静态检查盯住两个项目入口，避免将来新增入口漏掉重置。

**测试写法上的一个坑**：独立构造的 `AnnotationInterface(None, None)` 没有
`event_bus`（由 `AnnotationPlugin` 注入），所以走事件总线发布是收不到的 —— 最初
G5/G7 计数恒为 0 就是这个原因。改为**直接驱动 `on_annotations_changed`**（被测的是
它内部的去重判定，不是事件总线本身）后才有效。另外 G7 一开始也失败：紧邻的 G5 已经
把同一签名缓存进去了，4 次发布理应全被去重（0 次刷新）—— 是**我的断言假设了冷缓存**，
先重置缓存再测，结果恰好 1 次。

### 4.19 阶段 6：撤销与可见性同帧回滚（D21）—— 顺带修掉"空列表写不回"

**问题**（D21 本体）：`hidden_indices` 是**按下标寻址**的集合，而撤销栈原先只快照
`annotations`：

* `clear_annotations` 先把 `hidden_indices` 清空、再置空标注并**只快照标注** →
  撤销后标注回来了，但可见性停在"空"，**原来的隐藏状态永久丢失**；
* `delete_annotation` 会让下标位移（service 用 `adjust_hidden_indices_after_deletion`
  维护），撤销只回滚列表、不回滚集合 → "列表=删除前、可见性下标=删除后"的不自洽组合，
  复选框与实际显示对不上，下次再删还会删错对象。

**做法**：撤销/重做帧增加 `hidden_indices` 字段（`set()` 独立拷贝，避免入栈后的原地
修改污染已入栈的帧）；`undo`/`redo` 在写回标注的同时把可见性一并回滚。

**顺带修掉的第二个真实缺陷（H2 抓到的）**：`undo`/`redo` 的写回守卫写成了

```python
if self.current_annotations:        # ← 拿"即将被替换掉的状态"当条件
    self.current_annotations[:] = restored
```

清除全部标注后 `current_annotations` 是**空列表**，守卫直接跳过写回 —— 于是撤销只恢复了
可见性、标注列表仍是空的（H2 第一版就是 `labels=[] hidden=[0,2]` 这种半恢复状态）。
改为统一走 session 的受控入口 `set_current_annotations(restored)`，空列表也能正常写回。

验证：`tests/annotation/test_undo_visibility.py`

```
H1 撤销帧字段=['annotations', 'hidden_indices', 'image_path'] 可见性=[1]
H2 清除前 labels=['x','y','z'] hidden=[0,2] → 清除后 labels=[] hidden=[]
   → 撤销后 labels=['x','y','z'] hidden=[0,2]
H3 删除前 hidden=[3] → 删除后 labels=['p0','p2','p3'] hidden=[2]（下标位移）
   → 撤销后 labels=['p0','p1','p2','p3'] hidden=[3]
H4 重做后 labels=['p0','p2','p3'] hidden=[2]
H5 空栈撤销 err='错误: 没有可撤销的操作'
H5 跨图撤销 err='...属于另一张图片(img00.png)...' 可见性=[0]（D3 保护仍有效，且被拒时不动可见性）
PASS: H1~H5 全部通过
```

H3 是最有价值的一条：它同时验证"标注列表恢复"与"可见性下标回到删除前的值"，
这正是下标寻址状态最容易出错的地方。H5 确认新字段没有破坏阶段 5d 的跨图拒绝路径。

### 4.20 阶段 3（收尾）：各标注格式的写入路径与写读闭环（D35 / D36）

**问题**：

* **D35**：`save_image_annotations` 的兜底分支把输出路径**硬编码为 `<base>.json`**，
  而 `MaskFormat.save()` 写的是**图像** —— 扩展名直接取自 output_path 交给 OpenCV：

  ```
  imwrite_unicode(<...>.json.atomic-1-2>, ext='.json')
  -> OpenCV: could not find encoder for the specified extension
  ```

  于是非项目模式下 mask/sa1b 保存 **100% 失败**，异常还被该函数的 `except` 吞成
  `{"status":"error"}`（阶段 9 加的失败上报才让它显形）。
* **D36**：目录里没有任何标注文件时 `_non_project_format` 会被解析成 `mask`，
  与 D35 组合就是"**新目录第一次保存必然失败**"。
* 附带风险：掩码若按 `<dir>/<base>.png` 落盘会与**源图同名同目录**，直接覆盖别人的图。

**做法**：新增 `_resolve_annotation_output(fmt, output_dir, base_name)`，"写什么文件、
放哪个子目录"一律从**格式类自身**读取（与它的加载实现绑定的事实），而非在调用点猜：

```
mask    -> masks/<base>.png      # MaskFormat.EXTENSIONS[0]，子目录与 scan_categories 的 'masks' 一致
sa1b    -> <base>.json
labelme -> <base>.json（项目内 annotation_formats 配置优先）
```

**顺带修掉的第二个缺陷（W6 抓到）**：读取端 `load_image_annotations` 依赖
`settings["annotation_formats"]`，而该配置**默认为空** -> `if not config: continue`
把 mask 等格式直接跳过。结果是"**写进去了却永远读不回来**"——同样属于数据完整性缺陷。
现在读写两端共用 `_annotation_output_candidates()`（基于同一份约定 + 兼容旧路径），
从此不会再漂移。

**过程中的一次误判（值得记下来）**：W6 第一版读回 0 条，我先怀疑写读不匹配，探针查出
`MaskFormat` 的语义是"**像素值 = class_id，0 表示背景**"——我测试里用的是 `cat: 0`，
写出来就是一张纯背景掩码。这是**测试数据问题，不是缺陷**；换成非 0 类别编号后即通过。

验证：`tests/annotation/test_format_write_paths.py`

```
W1 mask     -> masks\img00.png  OK
W1 sa1b     -> ds\img00.json  OK
W1 labelme  -> ds\img00.json  OK
W2 mask 保存 status=success；掩码是图像: True (96x64)；源图未被覆盖
W3 sa1b 保存 status=success
W4 labelme 保存 status=success（原有格式回归）
W6 掩码读回标注数=2（写-读闭环）
W5 新目录首次 mask 保存 status=success（D35+D36 组合场景）
PASS: W1~W6 全部通过
```

W6 是"写读闭环"的证明，也是发现读取端缺陷的那条断言；W5 覆盖原缺陷最典型的表现
（新目录首次保存）。

### 4.21 阶段 6：AI 全量转换的"取消"真正生效（D9 余项）

**问题**：`convert_all_with_ai` 的 `QProgressDialog` 上写着"取消"按钮，但
`progress.canceled` **没有接到任何逻辑**，服务侧也没有取消入口 —— 按下去毫无作用，
用户只能等整批跑完。这属于"看起来能停、实际停不下来"的用户可见失效路径。

**做法**：协作式取消，并且**只在图片边界检查**：

* 界面新建 `threading.Event`，`progress.canceled.connect(_request_cancel)` 置位它，
  并把 event 一路传给 `convert_all_with_ai` → `batch_ai_convert`；
* 服务在每张图的**开始**检查该 event，置位则 `cancelled=True` 并 `break`；
  同时把在途标记挂到 `self._batch_cancel_event`，提供 `cancel_batch_convert()`
  供程序化取消（无在途任务时返回 False）；
* **为什么只在边界检查**：绝不打断正在写盘的那张图，因此已落盘的标注始终完整，
  不会留下"半张图"的数据；
* 结果 dict 新增 `cancelled` 与 `processed_count`，界面据此提示
  「已在第 N/M 张停止（已处理的标注均已完整落盘）」，而不是谎报总数；
* 全过程**不引入 `processEvents`**（保持阶段 4b 的结论）。

验证：`tests/annotation/test_batch_cancel.py`

```
C1 cancelled=True processed=4/6 total=6
C3 已落盘产物数=0 processed_count=4（应满足 produced <= processed）
C2 已落盘产物是否有空文件: []（应为空）
C4 进度回调次数=4（取消后不应继续）
C6 无在途任务时 cancel_batch_convert()=False（应为 False）
C5 不取消：cancelled=False processed=6/6 回调末值=6
C7 界面侧已接线 canceled.connect + cancel_event: True
C7 是否真实调用 processEvents: False（应为 False）
PASS: C1~C7 全部通过
```

C2 是"取消不会造成半份数据"的直接证据；C5 守住"不取消时行为完全不变"；
C7 是静态接线检查（**只扫代码、不扫注释** —— 函数体内本来就有一条解释
`processEvents` 的注释，第一版因此误报，这一点也写进了测试注释）。

### 4.22 阶段 3（收尾）：导航可见集唯一真值来源（3e）

**问题**：可见性原先有**两个来源**：

* 界面 `switch_image` 从 `file_list` 的隐藏状态推导 —— 正确，会跳过被筛选隐藏的图；
* service `_navigate_ai` 硬编码 `set(range(len(self.image_files)))` —— 即**假设没有筛选**。

同一个"下一张"因此有两条真值：启用筛选后，**agent 走 `next_image` 会跳到被隐藏的图片上**，
而用户按方向键会正确跳过。反向验证（同一场景：隐藏奇数下标，从 0 向后）：

```
OLD(假设全可见) -> 1      # 落在被筛选隐藏的图片上
NEW(唯一真值)   -> 2      # 跳过隐藏项
```

**做法**：两侧各提供一个 `visible_image_indices()`，共用同一份语义 ——
界面以 `file_list` 的实际隐藏状态为准（筛选结果的可视真值），service 以界面回写的
`file_visibility` 为准；无筛选时都退化为"全部可见"。`switch_image` 与 `_navigate_ai`
都改为从这里取，导航不再有任何"假设全可见"的硬编码。

验证：`tests/annotation/test_navigation_single_source.py`

```
N1 无筛选: 界面可见=8 service 可见=8（应都为 8）
N2 筛选后: 界面可见=[0, 2, 4, 6] service 可见=[0, 2, 4, 6]（应一致）
N3 从 0 向后导航 -> 2（应为 2，修复前会得到 1 即被隐藏的那张）
N4 同一个动作：界面 -> 2，service -> 2（应相同）
N5 末张(6)向后 -> None（应为 None，不越界不循环）
N6 静态检查：两端都经 visible_image_indices()  OK
PASS: N1~N6 全部通过
```

N4 是最关键的一条：它断言"同一个动作不能有两个答案"。N6 用静态检查把这条约束固化，
避免将来新增导航入口时又各自造一份可见集。

### 4.23 阶段 6：画布不变量（D18）—— 畸形数据不得让画布永久停止重绘

**为什么这条最严重**：Qt 里 `paintEvent` 抛异常会**吞掉该次绘制**，用户看到的是
"画布卡住、之后再也不刷新"。而触发它的数据来自**持久化文件**
（`project_info.json` 的 ROI、标注 JSON 的 bbox/polygons），一次坏写入会**永久生效**。

**三处实测触发点**（本轮用真实 `QPainter` + 直接调用 `paintEvent` 逐一定位）：

```
CASE a FAIL at: drawing_widget.py line 974  TypeError   # bx,by,bw,bh = ann.get('bbox', [0,0,0,0])
CASE b FAIL at: drawing_widget.py line 902  TypeError   # any(len(p) >= 3 for p in ann['polygons'])
```

根因是同一个反模式：`ann.get('bbox', [0,0,0,0])` 的默认值**只在键缺失时生效** ——
键存在但值为 `None` / 长度不为 4 时照样返回坏值，随后解包抛异常。

**做法**（把校验收敛到唯一入口，帧循环只调用安全访问器）：

* `persistent_roi` 改为**读取时校验**（`_coerce_roi`）：None / 长度≠4 / 非数字 /
  NaN / bool 一律视为"无 ROI"，返回 None。5 处解包点本来都写成
  `if self.persistent_roi:` 再解包，因此天然安全；
* 新增 `_is_polygon_like(poly)`：显式判断"能否 len 且 >=3"，替换原来会抛异常的
  `any(len(p) >= 3 ...)`；
* 新增 `_safe_bbox(ann)`：保证返回 4 个数值（坏元素用默认值兜底），
  替换帧循环内两处 `ann.get('bbox', [0,0,0,0])`。

验证：`tests/annotation/test_canvas_invariants.py`

```
P1 损坏 ROI 全部被拒: True（8 种：None/长度3/长度5/字符串/标量/含字符串/含NaN/含bool）
P2 合法 ROI 正常保留: (1.0, 2.0, 3.0, 4.0)
P3 ROI=None / 长度3 / 含字符串 时重绘成功=True，且读取值均被规范为 None
P5 坏 ROI 下连续 5 次重绘均成功: True
P4 畸形标注#0..#4 重绘成功=True（bbox=None / polygons=[1.0,2.0] / polygons=[[[1,2],[3,4]]] / 缺字段 / polygons=[None]）
P4 全部畸形标注同时存在时仍可重绘  OK
P6 persistent_roi 读取经 _coerce_roi 校验  OK
PASS: P1~P6 全部通过
```

**测试写法上的两个坑（都实测踩到）**：

1. 第一版用 `canvas.grab()` 触发绘制，直接 **abort**：
   `QPaintDevice: Cannot destroy paint device that is being painted`（退出码
   `-1073740791`）。改为自己开 `QPainter(canvas)` 再直接调用 `paintEvent`
   （`paintEvent` 只使用 `painter`、不使用 `event`，传真实 `QPaintEvent` 即可）。
2. 修完 line 913 后**现象完全没变** —— 说明还有别的触发点。没有继续猜，而是打印
   traceback 取到精确行号（902/974），这才把三处一次修净。**"改完没变化"就该去取
   真实栈，而不是再加一层保护。**

### 4.24 阶段 6：列表重建的信号安全（D16）

**问题**（两层，都属于"无重入/无对象生命周期漏洞"）：

1. `load_directory_images()` 里 `search_box.clear()` **未 blockSignals**：`clear()`
   必然发射 `textChanged`（实测 `handler('')` + `textChanged('')` 都被触发），于是在
   "目录刚重建、currentRow 尚未确定"时启动一次搜索/去抖，状态被并发改写。
2. 列表重建（`refresh_label_list` 会销毁并重建列表行及其上的复选框/按钮）原先直接在
   信号槽里执行。若该槽由**列表行上的子控件**发射触发（删除按钮、类别按钮、
   双击），就等于在**发射栈上销毁发起控件** —— Qt 下是未定义行为。

**做法**：

* **信号贯穿整个重建期屏蔽**。原先只屏蔽 `takeItem` 循环，而 `addItem` 之后的
  `_populate_file_thumbnails()` 会设置 item 数据并触发 `itemChanged`（实测 5 行 = 5 次）
  —— 重建期间仍有信号打到外部，那正是"在发射栈上重建"的入口。现在
  `file_list` 与 `search_box` 的 `blockSignals` 用 `try/finally` 贯穿全过程。
* **新增 `request_list_rebuild(fn, from_signal=False)`**：在重建区内部直接执行；
  从信号槽发起的重建则 `QTimer.singleShot(0, fn)` **延后到下一轮事件循环** ——
  那时发射栈已展开完毕，销毁发起控件才安全。删除/类别切换路径已改为 `from_signal=True`。

**为什么必须显式传 `from_signal` 而不能自动判定**：最初我用 `self.sender() is not None`
判断"是否在发射栈内"，T4 直接证伪 —— 在 **lambda / 嵌套辅助函数**里 `sender()`
返回 None（Qt 只保证在**直接**槽函数里有效），于是重建仍在发射栈内同步执行。这类
"看起来能自动判断、实际会漏判"的写法必须用回归固定住。

验证：`tests/annotation/test_list_rebuild_safety.py`

```
T1 load_directory_images 期间 search_box.textChanged 次数=0（应为 0）
T2 列表重建期间 file_list 信号次数=0（应为 0，行数 5 -> 5）
T3 非信号栈内同步执行: ['sync']
T4 信号栈内顺序=['emitter_before_rebuild', 'emitter_after_request', 'rebuild']
T5 重建区内嵌套请求顺序=['nested', 'outer_done']
T6 clear() 前后均有 blockSignals: True
T6 删除路径经 request_list_rebuild: True
PASS: T1~T6 全部通过
```

T4 是核心断言：重建必须出现在 `emitter_after_request` **之后**（发射栈已展开）。
T2 的范围刻意只覆盖"列表重建区"，不含重建之后 `load_image -> setCurrentRow` 引发的
"选择已改变" —— 后者是导航的正常结果（用调用栈确认过），把它算进 D16 会得到
一个永远失败的断言。

### 4.25 收尾：编辑态并入唯一状态所有者

**问题**：`hidden_indices`、`_undo_stack`、`_redo_stack`、`secondary_annotations`、
`secondary_annotation_map` 原先各自存在于 `ManualAnnotationService`，与 `ImageSession`
并列成为**第二份状态**。它们全都是**按图作用域**的：

* 撤销栈 frame 记录某张图的标注 + 可见性，跨图存活就是 D3（把上一张图的标注写进当前图）；
* `hidden_indices` 是**按下标寻址**的集合，与当前标注列表同生共死（D21：只回滚列表
  不回滚它会出现"列表=删除前、可见性=删除后"的不自洽）；
* 副标注是当前图的对照标注，换图后必须整体失效。

**做法**：全部搬进 `ImageSession`，并在 service 上保留**同名 property 委托**——
既有约 40 处调用点无需改动，但写入会落到 session，**不可能再各存一份**。
`begin_load()` 与 `clear()` 统一调用 `clear_edit_state()`，"换图要清哪些"不再靠
service 逐个记得（少一个漏清的机会）。撤销栈的 `append`/`clear` 等就地变更也作用在
session 的同一个列表对象上（property 返回真实列表，不是副本）。

验证：`tests/annotation/test_edit_state_owner.py`

```
S1 service 上的独立副本=[]（应为空）
S2 hidden=True secondary=True map=True
S3 同一列表对象=True append 生效=True redo append=True
S4 界面路由到同一列表=True
S5 换图后 hidden=[] undo=0 redo=0 sec=0 map=0
S6 clear() 后 hidden=[] undo=0 sec=0
S7 跨图撤销 err=错误: 撤销栈里的操作属于另一张图片(img02.png)，已清空撤销历史…
PASS: S1~S7 全部通过
```

S1 是"没有第二份真值"的机器可检证据；S3/S4 确保委托**不是**复制（否则又变成两份）；
S7 确认这次搬家没有削弱 D3 的跨图保护。

### 4.26 长跑切图稳定性：覆盖原始症状形态

**为什么补这一条**：用户报告的原始症状是"**快速切换前后两张图片一段时间后**软件突然
卡住"。已有 `test_switch_stress.py` 是短促爆发（40/80 轮），断言的是"不 abort / 状态
一致 / 并发有界 / 最新优先"—— **没有任何一条断言资源不随时间增长**。也就是说
"跑一段时间才卡"这种形态在原覆盖下**可以漏过去**：单次爆发正常，不代表第 400 次也正常。
第 24 轮终检时我发现这个缺口，本轮补齐。

**两阶段**（对应两种真实操作形态）：

* **阶段一（长跑）**：默认 400 轮，每轮间隔 >= 解码耗时，在 1/4 进度点采样线程数与
  QObject 子对象数，看是否**单调增长**；
* **阶段二（连发）**：轮数 ×3，间隔 ~1ms **不等上一张解码完就继续发** —— 这才是
  "快速滑动"的真实形态，也是原始冻结报告的触发条件。

**断言**：L1 线程不随轮次增长且静止后回落；L2 子对象数不增长；L3 解码并发有界；
L4 事件循环无长冻结（>400ms 即失败）；L5 收尾 `drain` 干净且画布持最后请求的图；
L6 全程无 abort（退出码即断言）。

**反向验证（关键）**：一个只会通过的 soak 测试毫无价值，因此单独验证了检测器**非永真**：

```
正常: 2->2基线2 => 报警? False
泄漏: 2->9基线2 => 报警? True       ← 真的会失败
轻微: 2->5基线2 => 报警? False
实测泄漏可见性: 基线=1 起6线程后=7 差值=6   ← 泄漏确实被观测到
```

结果（200 轮 + 600 次连发）：

```
连发 600 次，墙钟 2.84s，最差间隔 8.0 ms（阈值 400）
L4 事件循环最差间隔 9.5 ms；L3 解码并发峰值 0（上限 3）
L1 线程采样: [(50,2),(100,2),(150,2),(200,2)]（基线 2）→ 增长 0
L2 子对象采样: [(50,9),(100,9),(150,9),(200,9)]（基线 9）
L5 drain=True 画布图=img007.png 期望=img007.png 一致=True
PASS: L1~L6 全部通过
```

线程数与对象数在 400 次切换 + 600 次连发后**完全平坦**，事件循环最差间隔 8~17ms
（相对 400ms 阈值有两个数量级余量），收尾排空干净 —— 原始"跑一会儿就卡"的形态
在修复后已无对应现象。

## 5. 回归测试约定**约定：退出码即断言。** 被测缺陷是 `qFatal -> abort()`，Python 层不可能捕获，
测试无法自报失败，因此：

| 退出码 | 含义 |
|---|---|
| `0` | 通过 |
| `1` | 断言失败 |
| `-1073740791` | `0xC0000409`，进程被 abort |
| 其他 | 异常退出 |

放在 `tests/`（**不要放 `scripts/` 或 `logs/`，那两个目录在 `.gitignore` 里，无法入库**）。

```powershell
$py = "D:\miniconda3\envs\sam3\python.exe"
& $py -u tests\run_all.py        # 一次跑完全部并汇总退出码（同时校验 core/core.json 未被改动）

# 或单独跑：
& $py -u tests\annotation\test_switch_stress.py 40 20       # I1~I5：切图不 abort、并发有界、状态一致、最终图不丢、可排空
& $py -u tests\annotation\test_label_list_leak.py 30 20     # L1~L2：列表重建不泄漏
& $py -u tests\annotation\test_shutdown.py 5 120            # A~F：关闭排空、排空期间不死锁、幂等
& $py -u tests\annotation\test_thumbnail_pending.py         # P1~P4：在途集合必定收敛、可重试
& $py -u tests\annotation\test_atomic_write.py              # A1~A7：原子写，崩溃不得毁掉已有标注
& $py -u tests\annotation\test_delete_image.py              # B1~B5：删图安全（专属/共享/半删/确认门）
& $py -u tests\annotation\test_no_processevents.py          # N1~N5：无重入 + 批量期间 GUI 空闲 + 取消/并发
& $py -u tests\annotation\test_single_state_owner.py 6      # O1~O7：唯一状态所有者 + 原子提交
& $py -u tests\annotation\test_annotation_io_async.py       # P1~P5：标注装载后台 I/O + 代次校验
& $py -u tests\annotation\test_enhance_async.py            # F1~F5：增强图后台 + 结果校验 + 代次守卫
& $py -u tests\annotation\test_model_load_thread.py        # M0~M6：模型加载不再用 QThread 子类
& $py -u tests\annotation\test_save_async.py               # S1~S7：标注后台写入 + 排序 + 失败上报
& $py -u tests\annotation\test_search_async.py             # Q1~Q5：搜索去抖 + 后台扫描（D15）
& $py -u tests\annotation\test_cross_image_writes.py       # U1/U2/B1/B2/S1：跨图写入路径校验（D2/D3）
& $py -u tests\annotation\test_non_project_dirs.py         # D4-1~3/D9-1~2：非项目目录隔离 + AI 全量转换
& $py -u tests\annotation\test_stop_and_wait.py            # D1~D7：stop_and_wait/run_on_main 互等（D14）
& $py -u tests\annotation\test_settings_session_state.py   # R1~R7：运行时状态不落盘 + 保存原子（D30）
& $py -u tests\annotation\test_refresh_signature.py        # G1~G7：刷新签名覆盖几何 + 缓存作废（D33）
& $py -u tests\annotation\test_undo_visibility.py          # H1~H5：撤销与可见性同帧回滚（D21）
& $py -u tests\annotation\test_format_write_paths.py      # W1~W6：各格式写入路径 + 写读闭环（D35/D36）
& $py -u tests\annotation\test_batch_cancel.py            # C1~C7：全量转换取消真正生效（D9 余项）
& $py -u tests\annotation\test_navigation_single_source.py # N1~N6：导航可见集唯一来源（阶段 3e）
& $py -u tests\annotation\test_canvas_invariants.py        # P1~P6：畸形数据不得让画布停止重绘（D18）
& $py -u tests\annotation\test_list_rebuild_safety.py      # T1~T6：列表重建的信号安全（D16）
& $py -u tests\annotation\test_edit_state_owner.py        # S1~S7：编辑态并入唯一状态所有者
& $py -u tests\annotation\test_long_switch_soak.py 400 20 # L1~L6：长跑+连发切图（原始症状形态）
```

**状态隔离是硬要求**：`isolate_state()` 把 `settings` 与 `project_settings` 重定向到临时目录后
才构造界面。这个应用默认会把运行时状态写进**真实文件**（`core/core.json` 受版本控制、
`projects/<name>/project_info.json`），没有隔离的测试会污染用户数据 —— 早期探针踩过这个坑。
测试运行后应校验 `core/core.json` 的 mtime 未变。

---

## 6. 待决策

阶段 2~6 涉及较大的结构改动（尤其阶段 2 的单一状态所有者与阶段 5 的格式层原子写）。
建议按上表顺序推进，每阶段独立验证；阶段 5 中的 **D1/D2/D3/D4** 是四条可复现的
数据损坏路径，若希望更快见效可提前插入。
