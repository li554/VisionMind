# 模型权重目录（weights/）

本目录存放 VisionMind **当前开源版本所需的全部预训练权重**（PyTorch / ONNX）。
**权重文件本身不进 Git**（见根目录 `.gitignore` 的 `weights/*`），仅本说明文件纳入版本管理。

---

## 一、总览

当前开源版本共 3 个插件，其中**只有 `annotation`（手动标注）会加载模型**：

| 插件 | 名称 | 是否使用模型权重 |
|------|------|------------------|
| `plugins/annotation` | 手动标注 | ✅ 需要（交互式分割 / 自动标注 / 标注细化） |
| `plugins/dashboard` | 项目大厅 | ❌ 不需要 |
| `plugins/version` | 版本管理 | ❌ 不需要 |

按用途分三类，按需下载即可：

| 用途 | 需要的权重 | 章节 |
|------|------------|------|
| 交互式分割（点击/框选打点） | MobileSAM（ONNX）、可选 SAM / SAM2 / HQ-SAM | 三 |
| 一键自动标注（文本 / 示例） | SAM3、YOLOE-26 系列 | 四 |
| 标注细化（边缘精修） | ViTMatte | 五 |

> ✅ **程序内已支持自动下载**：`ModelFactory.create()` 在本地权重缺失、`load()` 失败后，会按
> **文件名**查 `core/core.json → download_urls` 并自动下载（见 `core/backend/core.py`），
> 首次加载某个模型时会自动拉取对应权重；下载中断会保留分片并可断点续传。下表的章节说明
> 用于**手动核对 / 离线部署**，一般无需手动放置。
>
> ⚠️ 少数权重**无法自动下载**，仍需手动获取（详见第四、五节）：
> `sam3.pt`（需 HuggingFace 授权）、`trtsam3` 引擎（私有包）、VitMatte（HF 整目录）。
>
> ⚠️ Ultralytics 系权重（`sam_b.pt`、`sam2.1_b.pt`、`mobile_sam.pt`、`yolo*.pt`、
> `yoloe-26*-seg.pt`、`mobileclip2_b.ts` …）的下载链接规则统一为：
> `https://github.com/ultralytics/assets/releases/download/v8.4.0/<文件名>`
> （`sam3.pt` 例外，需在 HuggingFace 申请授权，见第四节）。

---

## 二、目录结构与配置对应关系

```
weights/
├── sam/                     # SAM / MobileSAM / HQ-SAM 权重及其 ONNX 导出
│   └── onnx/                #   ONNX encoder + decoder
├── sam2/                    # SAM2 权重及其 ONNX 导出
│   └── onnx/
├── sam3/                    # SAM3 权重与词表（含随权重下载的 README/LICENSE）
├── hqsam/                   # HQ-SAM / SAM2-HQ 的 PyTorch 权重（ONNX 版统一放 sam/onnx/）
├── yolo/                    # YOLO / YOLOE 检测分割权重
├── refine/                  # 边缘细化（ViTMatte）
│   └── vitmatte-small-composition-1k/
└── README.md                # 本文件
```

以上即开源版本涉及的全部权重目录；各目录下的配置文件（`config.json`、`preprocessor_config.json`、
`tokenizer.json` 等）随对应权重一起下载。

路径**不是硬编码**的，全部通过 `core/core.json` 的 `model_paths`（扁平数组，`key`/`value` 结构）
由 `core/backend/path_resolver.py` 解析；相对路径基于**项目根目录**。
也可在程序内 **设置 → 模型配置 → 模型路径** 中修改：

| 配置键 | 默认值 | 含义 |
|--------|--------|------|
| `weights_dir` | `weights` | 权重根目录 |
| `sam_dir` | `weights/sam` | SAM 权重目录 |
| `sam2_dir` | `weights/sam2` | SAM2 权重目录 |
| `yolo_dir` | `weights/yolo` | YOLO / YOLOE 权重目录 |
| `hqsam_dir` | `weights/hqsam` | HQ-SAM 权重目录 |
| `refine_dir` / `vitmatte_dir` | `weights/refine` / `…/vitmatte-small-composition-1k` | 细化模型目录 |
| `trt_dir` | `weights/trtsam3` | TensorRT 版 SAM3 引擎目录（`trtsam3` 依赖私有包 `trtsam`，开源版不可用，见第四节） |

---

## 三、交互式分割模型（annotation 插件 · 交互模型 `role=interactive`）

在「标注 → 交互模型」菜单中可选；默认使用 `mobilesam_onnx`。
**默认启用列表**见 `core/core.json → enabled_interactive_models`。

### 3.1 MobileSAM / SAM / SAM2（ONNX，推荐，无需 GPU 也能跑）

ONNX 版需要 **编码器 + 解码器** 两个文件，成对放置，缺一不可。
官方的 SAM / MobileSAM / HQ-SAM / SAM2 ONNX 权重由 **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)**
（GPL-3.0）在其 GitHub Release 上托管，可直接下载、无需自行导出（见文末「权重来源与致谢」）。
本项目 `sam_onnx_wrapper` / `sam2_onnx_wrapper` 使用的输入输出契约（`image_embeddings` /
`mask_input` / `has_mask_input` / `orig_im_size`，SAM2 为 `high_res_feats_0/1`）与 X-AnyLabeling
的导出件一致，因此其发布文件可直接放入下表路径使用。

链接统一规则：`https://github.com/CVHub520/X-AnyLabeling/releases/download/<tag>/<文件名>`，
下表已给出完整地址（实测可达，大小已核对）。

| 模型名（菜单） | 需要的文件（相对 `weights/`） | 大小 | 官方下载地址 |
|----------------|------------------------------|------|--------------|
| `mobilesam_onnx` | `sam/onnx/mobile_sam.encoder.onnx` | 26.85 MB | [`v0.2.0/mobile_sam.encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/mobile_sam.encoder.onnx) |
| `mobilesam_onnx` | `sam/onnx/sam_vit_h_4b8939.decoder.onnx` | 8.34 MB | [`v0.2.0/sam_vit_h_4b8939.decoder.quant.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_h_4b8939.decoder.quant.onnx)（量化版，直链即此文件；非量化 `…decoder.onnx` 为 15.74 MB） |
| `sam_onnx_b` | `sam/onnx/sam_vit_b_01ec64.encoder.onnx`<br>`sam/onnx/sam_vit_b_01ec64.decoder.onnx` | 342.58 MB<br>15.74 MB | [`v0.2.0/…encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_b_01ec64.encoder.onnx)<br>[`v0.2.0/…decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_b_01ec64.decoder.onnx) |
| `sam_onnx_l` | `sam/onnx/sam_vit_l_0b3195.encoder.onnx`<br>`sam/onnx/sam_vit_l_0b3195.decoder.onnx` | 1177.09 MB<br>15.74 MB | [`v0.2.0/…encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_l_0b3195.encoder.onnx)<br>[`v0.2.0/…decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_l_0b3195.decoder.onnx) |
| `hqsam_onnx_b` | `sam/onnx/sam_hq_vit_b_encoder.onnx`<br>`sam/onnx/sam_hq_vit_b_decoder.onnx` | 342.37 MB<br>19.74 MB | [`v2.0.0/sam_hq_vit_b_encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.0.0/sam_hq_vit_b_encoder.onnx)<br>[`v2.0.0/sam_hq_vit_b_decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.0.0/sam_hq_vit_b_decoder.onnx) |
| `hqsam_onnx_l` | `sam/onnx/sam_hq_vit_l_encoder.onnx`<br>`sam/onnx/sam_hq_vit_l_decoder.onnx` | 1176.63 MB<br>20.74 MB | [`v2.0.0/sam_hq_vit_l_encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.0.0/sam_hq_vit_l_encoder.onnx)<br>[`v2.0.0/sam_hq_vit_l_decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.0.0/sam_hq_vit_l_decoder.onnx) |
| `sam2_onnx_b` | `sam2/onnx/sam2.1_hiera_base_plus.encoder.onnx`<br>`sam2/onnx/sam2.1_hiera_base_plus.decoder.onnx` | 324.04 MB<br>19.68 MB | [`v2.5.0/…encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.5.0/sam2.1_hiera_base_plus.encoder.onnx)<br>[`v2.5.0/…decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.5.0/sam2.1_hiera_base_plus.decoder.onnx) |
| `sam2_onnx_l` | `sam2/onnx/sam2.1_hiera_large.encoder.onnx`<br>`sam2/onnx/sam2.1_hiera_large.decoder.onnx` | 848.16 MB<br>19.68 MB | [`v2.5.0/…encoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.5.0/sam2.1_hiera_large.encoder.onnx)<br>[`v2.5.0/…decoder.onnx`](https://github.com/CVHub520/X-AnyLabeling/releases/download/v2.5.0/sam2.1_hiera_large.decoder.onnx) |

> `sam_onnx_h`（SAM ViT-H）、`hqsam_onnx_h`、`sam2_onnx_s/t` 等 ONNX 模型**不在默认启用列表**
> （`core/core.json → enabled_interactive_models`）中，本表也未收录其编码器直链，一般无需下载；
> 如需使用，请用官方导出脚本自行生成并放入对应 `…/onnx/` 目录。
>
> 文件格式对不上时（例如下载后文件名不同），程序按 `core/core.json` 里的路径找文件，
> 重命名成上表「需要的文件」列的名字即可。
> 如需自行导出（例如换精度/换算子集），仍可用官方导出脚本：SAM / HQ-SAM / SAM2 各自仓库的
> `scripts/export_onnx_model.py`，MobileSAM 用
> https://github.com/ChaoningZhang/MobileSAM/blob/master/scripts/export_onnx_model.py 。

### 3.2 PyTorch 权重

| 模型名（菜单） | 目标文件（相对 `weights/`） | 大小 | 下载地址 |
|----------------|------------------------------|------|----------|
| `sam_b` | `sam/sam_b.pt` | 357.7 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt` |
| `sam_l` | `sam/sam_l.pt` | 1.19 GB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_l.pt` |
| `sam_h` | `sam/sam_vit_h_4b8939.pth` | 2.45 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| `sam2_b` | `sam2/sam2_b.pt` | 154.4 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_b.pt` |
| `sam2_l` | `sam2/sam2_l.pt` | 428.4 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt` |
| `hqsam_b` | `hqsam/sam_hq_vit_b.pth` | 361.8 MB | https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth |
| `hqsam_l` | `hqsam/sam_hq_vit_l.pth` | 1.20 GB | https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth |
| `hqsam_tiny` | `hqsam/sam_hq_vit_tiny.pth` | 40.6 MB | https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth |
| `sam2_hq_b`（配置项 `sam2_hq_b_path`） | `hqsam/sam2.1_hq_hiera_base_plus.pt` | ~320 MB | ⚠️ 上游仓库未提供该文件名的公开直链，当前**缺失**（见 5.4） |
| `sam2_hq_l`（配置项 `sam2_hq_l_path`） | `hqsam/sam2.1_hq_hiera_large.pt` | 857.2 MB | https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt |
| `MobileSAM` 原始权重 | `sam/mobile_sam.pt` | 38.8 MB | https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt<br>（亦可 `…/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt`） |

> 上表除 `sam_h` 外均为 **Ultralytics 格式**（`.pt`），由 `ultralytics` 的
> `Predictor / SAM2Predictor` 加载；`hqsam_*` 为 **HQ-SAM 原始格式**，由
> `segment-anything-hq` 包加载（可选依赖，`pip install segment-anything-hq`；未安装时这 3 个
> 模型不会注册，程序会在启动日志中提示，但**不影响**其他模型与 ONNX 系模型）。
> ONNX 系模型由 `onnxruntime-gpu` 加载，不需要上述两个 Python 包。

---

## 四、示例模型 / 一键自动标注（annotation 插件 · 示例模型 `role=example`）

「自动标注 → 示例模型」菜单使用；默认启用列表见 `core/core.json → enabled_auto_models`。

| 模型名（菜单） | 需要的文件（相对 `weights/`） | 大小 | 下载地址 / 说明 |
|----------------|------------------------------|------|------------------|
| `sam3` / `fs_sam3` | `sam3/sam3.pt` | 3.29 GB | **需申请授权**：https://huggingface.co/facebook/sam3 （SAM 3 权重不随 Ultralytics 自动下载） |
| `sam3` / `fs_sam3` | `sam3/bpe_simple_vocab_16e6.txt.gz` | ~1.3 MB | 配置项 `sam3_bpe_path`；来自 CLIP 仓库：<br>`https://github.com/ultralytics/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz` |
| （SAM3 文本提示依赖） | `sam/mobileclip2_b.ts` | 242.0 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts` |
| `yoloe-26x` | `yolo/yoloe-26x-seg.pt` | 163.7 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt` |
| `yoloe-26n` | `yolo/yoloe-26n-seg.pt` | 11.2 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt` |
| `yoloe-26s` | `yolo/yolo26s-seg.pt` | 22.4 MB | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt` |

另有 7 个 `fs_*`（少样本 / 示例复用）模型，复用上表 ONNX 权重对，无需额外下载：
`fs_sam2_onnx_b/l`、`fs_hqsam_onnx_b/l`、`fs_sam_onnx_b/l`、`fs_mobilesam_onnx`；
`fs_sam3` 则复用 `sam3.pt`（需按上表申请授权下载），并非复用 ONNX。

> `sam3` 是**示例模型**（不属于交互模型），同时支持**示例推理**（给定示例框找同类目标）与
> **文本推理**（自然语言概念描述，概念分割）；对应菜单为「自动标注 → 示例模型」，
> 交互模型菜单中不会出现它。
>
> `sam3` 依赖 `clip` 包的特殊 fork：`pip install git+https://github.com/ultralytics/CLIP.git`
> （否则会报 `'SimpleTokenizer' object is not callable`）。
>
> ⚠️ `trtsam3`（TensorRT 版 SAM3）依赖私有包 `trtsam`，**未发布到公共 PyPI，开源版不可用**；
> 但它默认出现在 `core/core.json → enabled_auto_models` 中，启动时会加载失败——可在
> 「设置 → 模型配置」里移除该项，或忽略该失败提示。
>
> `yoloe-26s` 对应的权重文件名是 `yolo26s-seg.pt`（不含 `e`，与配置 `yoloe_26s_path` 一致），
> 与 `yoloe-26x/n-seg.pt` 的命名不同，按上表/下载脚本放置即可。

---

## 五、细化模型（annotation 插件 · 细化模型 `role=refine`）

「标注细化 / refine」功能使用，默认 `vitmatte`（`core/core.json → refine_method`）。

### 5.1 ViTMatte（默认，ONNX + PyTorch 双格式）

目录：`weights/refine/vitmatte-small-composition-1k/`

| 文件 | 用途 | 大小 | 来源 |
|------|------|------|------|
| `config.json`、`preprocessor_config.json` | 处理器 / 模型配置 | 小 | 随权重一起下载 |
| `model.safetensors` | PyTorch 权重（`refine_use_onnx=关闭` 时使用） | 98.5 MB | https://huggingface.co/hustvl/vitmatte-small-composition-1k |
| `onnx/model_quantized.onnx` | 默认使用的量化 ONNX 模型（配置项 `vitmatte_onnx_path`） | 26.2 MB | 由上述权重转换（`optimum-cli export onnx`），或直接使用仓库内已有的导出件 |
| `onnx/model.onnx` | 非量化 ONNX（可选） | 99.1 MB | 同上 |

```bash
# 方式 A：直接下载 PyTorch 权重（HF 全量文件）
pip install -U "huggingface_hub[cli]"
hf download hustvl/vitmatte-small-composition-1k --local-dir weights/refine/vitmatte-small-composition-1k

# 方式 B：国内镜像
set HF_ENDPOINT=https://hf-mirror.com
hf download hustvl/vitmatte-small-composition-1k --local-dir weights/refine/vitmatte-small-composition-1k

# 量化 ONNX 由 PyTorch 权重导出（可选）
pip install optimum[exporters]
optimum-cli export onnx --model weights/refine/vitmatte-small-composition-1k \
  weights/refine/vitmatte-small-composition-1k/onnx
```

### 5.2 CascadePSP（可选细化模型）

`refine_method=cascadepsp` 时由 `segmentation-refinement` 包加载，权重目录为 `refine_dir`
（默认 `weights/refine`），包会**自动下载** CascadePSP 权重到该目录：

```bash
pip install segmentation-refinement
```

### 5.3 混合细化（hybrid）

`refine_method=hybrid` 不引入新权重，内部复用交互式模型 `sam2_l`（见 3.2）。

### 5.4 已知缺口

* 配置项 `sam2_hq_b_path` 指向的 `weights/hqsam/sam2.1_hq_hiera_base_plus.pt` 在上游仓库没有同名
  公开文件，运行到该配置项时会报文件不存在；如需使用请自行从 HQ-SAM 项目获取并重命名，
  或改用 `sam2_hq_l`。

---

## 六、普通模型 / 自定义模型（annotation 插件 · `role=plain` / 自定义）

用户自己训练好的 YOLO 权重（检测 / 分割 / OBB）通过
**「标注 → 普通模型 → 添加自定义模型」** 注册，路径写入 `core/core.json → custom_models`，
可放在任意目录（建议 `weights/yolo/` 或项目 `outputs/` 下）。
当前 `enabled_plain_models` 为空，即默认没有普通模型，`一键标注(mode=model)` 需先注册。

若要让训练任务的模型下拉框可用（`ModelFactory.TRAINING_MODELS`），建议预置以下基础权重：

| 文件（`weights/yolo/`） | 大小 | 下载地址（Ultralytics v8.4.0 assets） |
|--------------------------|------|----------------------------------------|
| `yolo11n.pt` / `yolo11n-seg.pt` | 5.4 / 5.9 MB | `…/v8.4.0/yolo11n.pt`、`…/v8.4.0/yolo11n-seg.pt` |
| `yolov8n.pt` / `yolov8n-seg.pt` | 6.2 / 6.7 MB | `…/v8.4.0/yolov8n.pt`、`…/v8.4.0/yolov8n-seg.pt` |
| `yolo26n.pt` / `yolo26n-seg.pt` | 5.3 / 6.4 MB | `…/v8.4.0/yolo26n.pt`、`…/v8.4.0/yolo26n-seg.pt` |
| `yolo26s.pt` / `yolo26s-seg.pt` | 19.5 / 22.4 MB | `…/v8.4.0/yolo26s.pt`、`…/v8.4.0/yolo26s-seg.pt` |
| `yolo26n-obb.pt` / `yolo26s-obb.pt` | 5.6 / 20.7 MB | `…/v8.4.0/yolo26n-obb.pt`、`…/v8.4.0/yolo26s-obb.pt` |

（`…` 均指 `https://github.com/ultralytics/assets/releases/download`。其余 `yolo11s/m/l/x`、
`yolov8s/m/l/x` 系列同理，把文件名替换即可。）
`yolov8n.onnx` 为导出件，可由 `yolo export model=weights/yolo/yolov8n.pt format=onnx` 生成。

---

## 七、批量下载脚本（可选，用于离线预下载）

```powershell
# 在项目根目录执行；下载当前开源版本需要的 Ultralytics 系权重
$base = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
$files = @{
  "weights/sam/sam_b.pt"           = "sam_b.pt"
  "weights/sam/sam_l.pt"           = "sam_l.pt"
  "weights/sam/mobile_sam.pt"      = "mobile_sam.pt"
  "weights/sam2/sam2_b.pt"         = "sam2.1_b.pt"
  "weights/sam2/sam2_l.pt"         = "sam2.1_l.pt"
  "weights/sam/mobileclip2_b.ts"   = "mobileclip2_b.ts"
  "weights/yolo/yoloe-26x-seg.pt"  = "yoloe-26x-seg.pt"
  "weights/yolo/yoloe-26n-seg.pt"  = "yoloe-26n-seg.pt"
  "weights/yolo/yolo26s-seg.pt"    = "yolo26s-seg.pt"
  "weights/yolo/yolo11n.pt"        = "yolo11n.pt"
  "weights/yolo/yolo11n-seg.pt"    = "yolo11n-seg.pt"
  "weights/yolo/yolov8n.pt"        = "yolov8n.pt"
  "weights/yolo/yolov8n-seg.pt"    = "yolov8n-seg.pt"
}
foreach ($dst in $files.Keys) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  Write-Host "下载 $($files[$dst]) ..."
  curl.exe -L --retry 3 -o $dst "$base/$($files[$dst])"
}

# 官方 SAM / HQ-SAM（非 Ultralytics 命名）
curl.exe -L -o weights/sam/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl.exe -L -o weights/hqsam/sam_hq_vit_b.pth   https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
curl.exe -L -o weights/hqsam/sam_hq_vit_l.pth   https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth
curl.exe -L -o weights/hqsam/sam_hq_vit_tiny.pth https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth
curl.exe -L -o weights/hqsam/sam2.1_hq_hiera_large.pt https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt
```

### ONNX 权重（交互式分割，来自 X-AnyLabeling）

```powershell
# X-AnyLabeling Release 托管；本地文件名即其发布名，可直接 -o 指定目标路径
$x = "https://github.com/CVHub520/X-AnyLabeling/releases/download"
$onnx = @{
  "weights/sam/onnx/mobile_sam.encoder.onnx"                = "$x/v0.2.0/mobile_sam.encoder.onnx"
  "weights/sam/onnx/sam_vit_h_4b8939.decoder.onnx"          = "$x/v0.2.0/sam_vit_h_4b8939.decoder.quant.onnx"   # 量化版 8.34MB，更省显存
  "weights/sam/onnx/sam_vit_b_01ec64.encoder.onnx"          = "$x/v0.2.0/sam_vit_b_01ec64.encoder.onnx"
  "weights/sam/onnx/sam_vit_b_01ec64.decoder.onnx"          = "$x/v0.2.0/sam_vit_b_01ec64.decoder.onnx"
  "weights/sam/onnx/sam_vit_l_0b3195.encoder.onnx"          = "$x/v0.2.0/sam_vit_l_0b3195.encoder.onnx"
  "weights/sam/onnx/sam_vit_l_0b3195.decoder.onnx"          = "$x/v0.2.0/sam_vit_l_0b3195.decoder.onnx"
  "weights/sam/onnx/sam_hq_vit_b_encoder.onnx"              = "$x/v2.0.0/sam_hq_vit_b_encoder.onnx"
  "weights/sam/onnx/sam_hq_vit_b_decoder.onnx"              = "$x/v2.0.0/sam_hq_vit_b_decoder.onnx"
  "weights/sam/onnx/sam_hq_vit_l_encoder.onnx"              = "$x/v2.0.0/sam_hq_vit_l_encoder.onnx"
  "weights/sam/onnx/sam_hq_vit_l_decoder.onnx"              = "$x/v2.0.0/sam_hq_vit_l_decoder.onnx"
  "weights/sam2/onnx/sam2.1_hiera_base_plus.encoder.onnx"   = "$x/v2.5.0/sam2.1_hiera_base_plus.encoder.onnx"
  "weights/sam2/onnx/sam2.1_hiera_base_plus.decoder.onnx"   = "$x/v2.5.0/sam2.1_hiera_base_plus.decoder.onnx"
  "weights/sam2/onnx/sam2.1_hiera_large.encoder.onnx"       = "$x/v2.5.0/sam2.1_hiera_large.encoder.onnx"
  "weights/sam2/onnx/sam2.1_hiera_large.decoder.onnx"       = "$x/v2.5.0/sam2.1_hiera_large.decoder.onnx"
}
foreach ($dst in $onnx.Keys) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  Write-Host "下载 $(Split-Path $dst -Leaf) ..."
  curl.exe -L --retry 3 -C - -o $dst $onnx[$dst]     # -C - 断点续传，大文件断了可重跑
}

# ViTMatte（细化模型）
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内网络建议启用，否则可省略
hf download hustvl/vitmatte-small-composition-1k --local-dir weights/refine/vitmatte-small-composition-1k
```

---

## 八、校验与排查

1. **校验文件大小**（对照本文表格中的 MB 值即可，无需哈希）：

   ```powershell
   Get-ChildItem -Recurse -File weights -Include *.pt,*.pth,*.onnx,*.ts,*.safetensors |
     Sort-Object Length -Descending |
     Select-Object @{n='MB';e={[math]::Round($_.Length/1MB,1)}}, FullName
   ```

2. **常见报错对照**

   | 现象 | 原因 / 处理 |
   |------|-------------|
   | 控制台打印「模型加载失败」 | 目标文件不存在或路径与 `core.json` 不一致，按第二节核对 |
   | `hqsam_b/l/tiny` 在菜单里找不到 | 未安装可选依赖：`pip install segment-anything-hq`（只影响 HQ-SAM 的 `.pt` 版；`hqsam_onnx_*` 不需要它） |
   | `trtsam3` 加载失败 / 提示缺少 `trtsam` | 属预期：`trtsam` 为私有包、未公开，开源版不支持；在设置中移除 `trtsam3` 或忽略即可（见第四节） |
   | `'SimpleTokenizer' object is not callable` | 装了标准 `clip` 而非 fork：`pip install git+https://github.com/ultralytics/CLIP.git` |
   | `FileNotFoundError: VitMatte模型目录不存在` | `vitmatte_dir` 下缺 `config.json` 等文件，按 5.1 重新下载整个目录 |
   | 自动下载失败 / 卡住 | 确认网络与 `core.json → download_urls` 是否含该文件名；中断会保留分片、重试可续传；无直链的 `sam3.pt` / `trtsam3` / VitMatte 仍需手动放置（见第四、五节） |

3. 权重文件缺失不影响程序启动，只是对应模型不可用；切换模型时会自动释放上一个模型占用的显存。

---

## 九、环境变量说明

程序**不会**自动设置任何下载相关的环境变量，ViTMatte 等 HuggingFace 下载默认走官方站点、
默认缓存目录（`~/.cache/huggingface`）。国内网络或需要统一缓存位置时，可在启动前自行设置：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"      # HF 镜像
$env:HF_HOME     = "$PWD\weights\huggingface"   # 可选：把 HF 缓存放进 weights/
$env:HF_HUB_DISABLE_SYMLINKS = "1"              # 可选：Windows 无符号链接权限时用复制代替
```

若在 Windows 上遇到 HuggingFace 下载报符号链接权限错误（`WinError 1314`），设置
`HF_HUB_DISABLE_SYMLINKS=1` 即可解决。

---

## 十、权重来源与致谢

本目录的权重**不由本仓库分发**，全部由用户按上文链接自行下载，因此各自遵循其上游协议
（本项目自身的 AGPL-3.0 协议不适用于这些权重）。各模型的来源与许可：

| 权重 | 来源 | 许可 |
|------|------|------|
| SAM / MobileSAM / HQ-SAM / SAM2 的 **ONNX 权重** | [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 的 GitHub Release（v0.2.0 / v2.0.0 / v2.5.0） | 权重源自 Meta 的 SAM / SAM2 及各上游项目（Apache-2.0 等）；托管方 X-AnyLabeling 项目本身为 GPL-3.0。本项目只引用其下载地址、不重新分发 |
| `sam_b.pt` / `sam_l.pt` / `sam2_*.pt` / `yoloe-26*-seg.pt` / `yolo*.pt` / `mobileclip2_b.ts` | [Ultralytics assets](https://github.com/ultralytics/assets/releases) | AGPL-3.0（Ultralytics） |
| `sam_vit_h_4b8939.pth` | [Meta SAM](https://github.com/facebookresearch/segment-anything) | Apache-2.0 |
| `sam_hq_vit_*.pth` / `sam2.1_hq_hiera_large.pt` | [HQ-SAM](https://github.com/lkeab/hq-sam) | Apache-2.0 |
| `mobile_sam.pt` | [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Apache-2.0 |
| `sam3.pt` | [facebook/sam3](https://huggingface.co/facebook/sam3)（需申请授权） | 见该模型页的许可条款 |
| ViTMatte | [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) | 见仓库页 |
| CascadePSP | [segmentation-refinement](https://github.com/li-plus/segmentation-refinement)（自动下载） | MIT |

特别感谢 **[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)** 项目——它为本项目的交互式
分割提供了可直接下载的 SAM / MobileSAM / HQ-SAM / SAM2 ONNX 导出件，省去了自行导出与校验的麻烦。
