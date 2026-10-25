# HYworld-image2world

**从单张图片生成三维高斯场景，支持分阶段训练与 Unity 资产导出。**

[English](README.md) · [简体中文](README_zh.md) · [上游 HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · [问题反馈](https://github.com/maohaotian/HYworld-image2world/issues)

HYworld-image2world 是基于腾讯 HY-World 2.0 的工程分支，将图像预处理、全景生成、全景超分、世界扩展、Gaussian Splatting 优化与 Unity 导出串成可配置的工作流。你可以先检查全景图，再生成完整场景；也可以复用已有结果，只运行指定阶段，并导出与场景配套的相机。

本项目沿用上游的预训练模型和整体架构，重点是高分辨率输入、生成流程的工程适配，以及向 Unity 交付场景资产。仓库没有发布新的基础模型，也不包含完整的 Unity/VR 应用。

## 目录

- [相对上游的修改](#相对上游的修改)
- [处理流程](#处理流程)
- [安装](#安装)
- [模型权重](#模型权重)
- [快速开始](#快速开始)
- [配置](#配置)
- [输出与查看](#输出与查看)
- [限制与常见问题](#限制与常见问题)
- [参与开发](#参与开发)
- [致谢与许可证](#致谢与许可证)

## 相对上游的修改

本分支保留上游模型，主要调整以下实现：

| 范围 | 本仓库的修改 |
| --- | --- |
| 流程入口 | 新增单图生成、仅生成全景供检查、从已确认全景继续生成、双卡重建等启动脚本。世界生成支持单独选择阶段。 |
| 输入与全景 | 保留原始图片；对工作副本处理 EXIF 方向、等比例缩放并记录元数据。新增带水平环绕填充的 2 倍全景放大，保存预览及实际后端、图像统计信息。 |
| 高分辨率 GS 数据 | 优先使用超分全景。帧分辨率变化时同步缩放相机内参和深度图，对齐天空深度与法线的尺寸，并记录训练帧来源。 |
| 模型与 VLM 配置 | 支持本地模型路径、显式指定 WorldStereo 基座、仅从本地加载及 CPU offload。支持可配置的 OpenAI 兼容视觉接口、API Key、备用模型和不调用 VLM 的确定性回退。 |
| GS 优化 | 开放训练步数、SH 阶数、初始点预算、损失权重和训练帧筛选。增加按 rank 加载 checkpoint 的续训与原训练时长处理，修复分布式进程的 bilateral-grid 导入；延长训练时不再同步改变增密频率。 |
| Unity 导出 | 导出坐标对齐后的 GS PLY 和相机 JSON，收录可用训练视角与生成轨迹；支持可选 SPZ，并提供从已有训练 PLY 导出的独立脚本。 |

主要改动位于 [`scripts/`](scripts/)、[`world_gs_trainer.py`](hyworld2/worldgen/world_gs_trainer.py)、[`gen_gs_data.py`](hyworld2/worldgen/gen_gs_data.py)，以及 worldgen 的相机、VLM 和模型加载工具。这些是工程实现与可配置实验，不代表每个场景的生成质量都优于上游。

## 处理流程

```text
图片 + 可选提示词
  → HY-Pano 全景 → 可选的 2 倍全景超分
  → 轨迹规划与渲染 → WorldStereo 世界扩展
  → RGB／深度／法线／相机数据准备 → GS 优化
  → 原生场景 PLY + Unity PLY／相机 JSON
```

全景生成后，worldgen 启动脚本包含五个阶段：

| 阶段 | 组件 | 作用 |
| --- | --- | --- |
| 1 | `traj_generate.py` | 分析全景，规划相机轨迹。 |
| 2 | `traj_render.py` | 渲染初始轨迹视图，准备描述信息。 |
| 3 | `video_gen.py` | 使用 WorldStereo 扩展场景。 |
| 4 | `gen_gs_data.py` | 整理训练视图、深度、法线与相机。 |
| 5 | `world_gs_trainer.py` | 优化高斯并导出场景资产。 |

上游的多视图／视频重建能力也保留在 [`hyworld2/worldrecon`](hyworld2/worldrecon)，无需先生成全景。

## 安装

### 环境要求

- Linux、Python 3.11、NVIDIA GPU 及兼容的 CUDA 工具链。参考安装目标为 CUDA 12.8。
- 根目录依赖固定为 PyTorch 2.7.1、torchvision 0.22.1；光栅化与几何扩展需要在匹配的环境中编译。
- 世界生成的第 2–4 阶段默认启动两个 GPU 进程。完整流程需要较大的显存、内存及磁盘空间；本分支尚未给出经过测量的消费级显卡最低配置。
- 能访问所需模型仓库；默认规划流程还需要支持图像输入的 VLM 接口。

### 创建环境并安装扩展

```bash
git clone --recurse-submodules https://github.com/maohaotian/HYworld-image2world.git
cd HYworld-image2world

conda create -n hyworld python=3.11 -y
conda activate hyworld

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements_git.txt

git submodule update --init --recursive
python -m pip install -e hyworld2/worldgen/third_party/gsplat_maskgaussian --no-build-isolation
python -m pip install ./hyworld2/worldgen/third_party/navmesh --no-build-isolation
python -m pip install flash-attn --no-build-isolation
```

世界生成训练使用仓库内的定制 `gsplat`；直接换成官方版本并不等价，部分训练选项依赖定制实现。Unity PLY 转换内部也会使用 `spz` 包，即使没有要求保存 SPZ 文件。

如需学习式全景超分，安装可选依赖：

```bash
python -m pip install realesrgan basicsr
```

其他全景后端见[全景模块文档](hyworld2/panogen/README_zh_CN.md)。该模块的独立 requirements 与根目录使用不同的 NumPy／Transformers 版本，不要未经处理直接覆盖安装到同一环境。

## 模型权重

本源码仓库不随附模型权重和生成场景。默认 Qwen 全景路线期望以下目录结构：

```text
checkpoints/
├── HY-World-2.0/
│   ├── HY-Pano-2.0/
│   └── HY-WorldMirror-2.0/
├── Qwen-Image-Edit-2509/
├── WorldStereo/
│   └── worldstereo-memory-dmd/
├── Wan2.1-I2V-14B-720P-Diffusers/
├── moge-2-vitl-normal/
│   └── model.pt
└── sam3/
```

| 组件 | 来源 |
| --- | --- |
| HY-Pano 适配权重、WorldMirror | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0) |
| Qwen 全景基座 | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |
| WorldStereo | [hanshanxue/WorldStereo](https://huggingface.co/hanshanxue/WorldStereo) |
| WorldStereo 基座 | [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) |
| 深度与法线 | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| 分割 | [facebook/sam3](https://huggingface.co/facebook/sam3) |

安装 Hugging Face CLI、获得相关模型访问权限后，可按以下方式下载：

```bash
hf download tencent/HY-World-2.0 --local-dir checkpoints/HY-World-2.0 \
  --include 'HY-Pano-2.0/*' 'HY-WorldMirror-2.0/*'
hf download Qwen/Qwen-Image-Edit-2509 --local-dir checkpoints/Qwen-Image-Edit-2509
hf download hanshanxue/WorldStereo --local-dir checkpoints/WorldStereo \
  --include 'worldstereo-memory-dmd/*'
hf download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
  --local-dir checkpoints/Wan2.1-I2V-14B-720P-Diffusers
hf download Ruicheng/moge-2-vitl-normal --local-dir checkpoints/moge-2-vitl-normal
hf download facebook/sam3 --local-dir checkpoints/sam3
```

规划阶段还会通过 Hugging Face 缓存查找 ZIM 和 Grounding DINO，超分工具可下载 Real-ESRGAN x2 权重。WorldStereo 默认 `WORLDSTEREO_LOCAL_FILES_ONLY=1`，其本地模型目录需要完整；这个开关不代表所有辅助模型都能完全离线运行。

## 快速开始

以下命令均在仓库根目录、已激活的环境中执行。

### 1. 配置运行路径与 VLM

当前启动脚本仍有开发机环境路径作为默认值。在自己的机器上显式指定解释器：

```bash
export PYTHON_BIN="$(command -v python)"
export TORCHRUN_BIN="$(command -v torchrun)"
export CHECKPOINT_DIR="$PWD/checkpoints"
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES_STAGE1=0

cp .env.example .env.local
```

编辑 `.env.local`，为支持图片输入的 OpenAI 兼容服务填写 `LLM_BASE_URL`、`LLM_NAME` 和 `OPENAI_API_KEY`。模型名必须与服务端一致，模板中的值仅为占位。可选配置 `LLM_FALLBACK_NAME`。脚本依次加载 `.env`、`.env.local`，文件中的值会覆盖先前导出的同名变量。不要提交 API Key。

### 2. 生成并检查全景

```bash
PANO_PROMPT="安静的校园庭院，道路连续，建筑风格一致。" \
PANO_SR_BACKEND=realesrgan \
bash scripts/run_image_to_panorama_sr.sh \
  /absolute/path/to/input.jpg "$PWD/inference_output/demo"
```

此命令生成基础全景、超分图和预览后停止。先检查空间布局与内容，再继续生成。默认 Qwen 后端经过边缘混合后的全景为 `1920×960`，2 倍超分后为 `3840×1920`。

`PANO_SR_BACKEND=lanczos` 使用普通插值，不依赖学习式超分组件。`auto` 会先尝试 Real-ESRGAN，失败时回退到 Lanczos，实际后端写入 `panorama_sr.png.json`。实验中若不接受静默回退，请显式使用 `realesrgan`。

### 3. 生成 GS 场景

```bash
bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"
```

脚本优先使用已有的 `panorama_sr.png`，否则使用 `panorama.png`。默认运行全部五个阶段，训练 10,000 步、SH 阶数为 0，并导出原生与 Unity GS。

如需从图片自动运行到最终场景：

```bash
PANO_SR=1 PANO_SR_BACKEND=realesrgan \
bash scripts/run_image_to_worldgen_2gpu.sh \
  /absolute/path/to/input.jpg "$PWD/inference_output/demo"
```

如果已有全景，将其放在 `inference_output/demo/panorama.png`，直接运行全景到 worldgen 的入口即可。超分图是可选项，需要单独生成。

### 4. 选择阶段或继续训练

```bash
# 保留已有世界扩展结果，重新准备 GS 数据并训练。
STAGES=45 bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"

# 从兼容的训练 checkpoint 继续到更长的训练时长。
STAGES=5 GS_RESUME_GS_PATH="$PWD/inference_output/demo/gs_result_2gpu/ckpts" \
GS_RESUME_SOURCE_MAX_STEPS=10000 GS_MAX_STEPS=20000 \
GS_SAVE_STEPS=20000 GS_EVAL_STEPS=20000 GS_PLY_STEPS=20000 \
bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"
```

阶段选择不会自动补回缺失的前置产物。续训需要训练数据和兼容的各 rank checkpoint；最终 PLY 本身不能替代训练 checkpoint。图像／全景入口可通过 `--help` 查看参数说明。

## 配置

[`run_worldgen_2gpu.sh`](scripts/run_worldgen_2gpu.sh) 的常用环境变量：

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `STAGES` | `all` | 选择阶段，如 `3`、`45` 或 `5`。 |
| `WORLDGEN_NPROC_PER_NODE` | `2` | 第 2–4 阶段的进程数，应与可见 GPU 匹配。 |
| `WORLDSTEREO_CPU_OFFLOAD` | `0` | 使用 CPU offload 替代 WorldStereo 默认 FSDP 路线。 |
| `SPLITTED_RESOLUTION` | `720` | 透视视图准备分辨率。 |
| `TRAJ_PRESET` | `auto` | `light`、`balanced`、`full`；分辨率 ≥720 时 auto 选择 light。 |
| `GS_MAX_STEPS` | `10000` | 训练时长；修改时同步设置保存、评估及 PLY 导出步数。 |
| `GS_SH_DEGREE` | `0` | 球谐阶数。 |
| `GS_DOWNSAMPLE_PTS_NUM` | `2000000` | 初始点预算，不是最终高斯数量。 |
| `GS_DEPTH_LAMBDA1` / `GS_LPIPS_LAMBDA1` | `0.1` / `0.2` | 损失权重。 |
| `GS_NORMAL_LOSS` | `1` | 法线监督。 |
| `GS_USE_MASK_GAUSSIAN` / `GS_USE_SCALE_REGULARIZATION` | `0` / `0` | 可选正则项。 |
| `GS_EXPORT_UNITY` / `GS_EXPORT_UNITY_SPZ` | `1` / `0` | Unity 导出及可选 SPZ 文件。 |

模型路径可通过 `PANO_QWEN_MODEL`、`PANO_LORA_PATH`、`WORLDSTEREO_REPO`、`WORLDSTEREO_BASE_MODEL`、`WORLDMIRROR_MODEL`、`MOGE_MODEL`、`SAM3_MODEL` 覆盖。

`HY_OFFLINE_VLM=1` 使用确定性的元数据／对象回退，跳过 VLM 调用。可为场景设置 `HY_OFFLINE_SCENE_TYPE` 和 `HY_OFFLINE_OBJECTS`。该模式会改变规划行为，也不会免除模型下载要求。

## 输出与查看

一次图片到场景的运行通常产生：

```text
inference_output/demo/
├── input/
│   ├── original.jpg                 # 原始输入副本，扩展名随输入而定
│   ├── input_rgb.png                # 处理方向与色彩格式后的工作图
│   └── input_meta.json
├── panorama.png
├── panorama_sr.png                  # 可选
├── panorama_sr.png.json             # 超分后端及统计信息
├── panorama_sr_preview.jpg
├── render_results/                  # 扩展结果与相机轨迹
├── gs_data/                         # RGB、深度、法线、相机及初始点
├── gs_result_2gpu/
│   ├── cfg.yml
│   ├── ckpts/                       # 训练 checkpoint
│   ├── stats/                       # 评估记录
│   ├── renders/                     # 验证渲染
│   ├── ply/
│   │   ├── point_cloud_9999.ply      # 10,000 步训练后的原生 GS
│   │   └── position_meta_info.json
│   └── unity_export/
│       ├── <asset>_9999_unity_dx12.ply
│       └── <asset>_9999_unity_dx12.json
└── run_summary.txt
```

以上为目录示意，中间文件和 mesh 产物随实际执行阶段变化。`run_summary.txt` 由图片入口写入。导出文件名使用从 0 开始的步数，因此 10,000 步训练对应 `9999`。Unity 资产名称由输入元数据或场景目录推导。

### 查看原生 GS

```bash
python hyworld2/worldgen/show_gs.py \
  --ckpt "$PWD/inference_output/demo/gs_result_2gpu/ply/point_cloud_9999.ply" \
  --port 8080
```

这个查看器需要 PLY 旁的 `position_meta_info.json`。已有场景的渲染无需保留大体积的全景／世界扩展模型权重。

### 交付给 Unity

在兼容的 Gaussian Splatting 渲染器和相机导入器中，配套使用 `*_unity_dx12.ply` 与对应 JSON。JSON 包含坐标变换后的相机位姿、视场角及可用的轨迹／训练视角。本仓库不提供 Unity 渲染器、XR 集成或应用交互逻辑。不要将 Unity 坐标系相机与原生坐标系 PLY 混用。

已有训练场景可单独重新导出，无需再次优化：

```bash
python scripts/export_unity_dx12_from_ply.py \
  --scene-dir "$PWD/inference_output/demo"
```

此脚本仍会读取 `gs_data`、配置以及相机／几何元数据，应在清理训练中间产物前完成导出。如果两种渲染路径都要使用，请同时保留原生与 Unity 资产。

精简归档应保留最终 PLY、坐标／相机元数据、原始图片、提示词、最终全景、配置和评估记录。若要续训，还需保留训练数据和 checkpoint；预览视频和训练缓存不是最终场景本体。

### 多视图／视频重建

```bash
INPUT_PATH="$PWD/examples/worldrecon/realistic/Messy_Room" \
OUTPUT_PATH="$PWD/inference_output/reconstruction_demo" \
bash scripts/run_worldrecon_2gpu.sh
```

此入口使用两个 worker 与 WorldMirror FSDP。更多选项见 [worldrecon 模块](hyworld2/worldrecon)和[继承的 CLI 参考](DOCUMENTATION_zh.md)。参考手册主要描述上游接口，本 README 中的入口与默认值以本分支为准。

## 限制与常见问题

- **找不到解释器：** 显式设置 `PYTHON_BIN`、`TORCHRUN_BIN`；启动脚本仍保留开发机路径作为回退值。
- **显存不足：** 可尝试 WorldStereo CPU offload、较轻的轨迹预设或较低视图分辨率。这些设置会影响性能与覆盖范围，单卡运行也不等价于默认配置。
- **模型加载失败：** 检查访问权限、本地路径及模型子目录完整性。WorldStereo 的 local-only 开关与辅助模型的缓存相互独立。
- **几何质量不理想：** 先检查全景，再检查扩展视图与相机。提高分辨率不保证几何正确，也不能恢复输入中不可见的真实结构。
- **清理后无法续训：** 需要继续优化的场景应保留训练数据和 checkpoint；仅归档 PLY 仍可用于渲染。
- **Unity 朝向或构图不同：** 使用配套导出与相机，检查坐标系和视场角，避免混用不同空间的资产。

这是研究型生成流程，结果可能有空洞、漂浮高斯，以及与输入不同的生成结构。本分支不承诺基准性能提升、精确度量重建，或自动交付可部署的 VR 应用。

## 参与开发

反馈问题时，请提供 commit、GPU／CUDA／PyTorch 版本、启动命令、不含密钥的相关配置，以及失败阶段的日志。涉及特定场景时，尽可能提供可公开的输入或最小复现。

提交应保持范围明确并说明验证方式。修改流程时，区分全景、扩展、训练和导出的结果，便于复现受影响阶段。请勿提交凭据、模型权重及生成的训练数据集。

## 致谢与许可证

本项目基于 [Tencent HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)，沿用 HY-Pano、WorldNav、WorldStereo、WorldMirror，并依赖 gsplat、MoGe、SAM3、Recast Navigation 及可选的 Real-ESRGAN 等项目。第三方代码、权重和资产分别遵循各自许可证。

继承项目适用 [Tencent HY-WORLD 2.0 Community License Agreement](License.txt)，包括其中的用途与地域限制。本分支没有将其替换为 MIT 或 Apache-2.0。上游归属声明见 [NOTICE](NOTICE)。

研究中使用底层模型时，请引用上游工作：

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint arXiv:2604.14268},
  year={2026}
}
```
