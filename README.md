# HYworld-image2world

**Single-image world generation, staged GS training, and Unity-ready scene exports.**

[English](README.md) · [简体中文](README_zh.md) · [Upstream HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) · [Issues](https://github.com/maohaotian/HYworld-image2world/issues)

HYworld-image2world is an engineering fork of Tencent's HY-World 2.0. It connects image preprocessing, panorama generation, panorama super-resolution, world expansion, Gaussian Splatting optimization, and Unity export in a configurable workflow. You can inspect a panorama before generating a full scene, rerun selected stages, and export the scene together with its cameras.

The pretrained models and overall architecture come from upstream. This repository focuses on adapting the pipeline for high-resolution inputs and downstream use in Unity. It does not introduce a new foundation model or include a Unity/VR application.

## Contents

- [What changed](#what-changed)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Model weights](#model-weights)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Outputs and viewing](#outputs-and-viewing)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [Contributing](#contributing)
- [Acknowledgments and license](#acknowledgments-and-license)

## What changed

The fork retains the upstream models and adds the following implementation changes:

| Area | Changes in this repository |
| --- | --- |
| Pipeline entry points | Wrappers for single-image generation, panorama-only review, generation from an accepted panorama, and two-GPU reconstruction. Worldgen stages can be selected independently. |
| Inputs and panoramas | Preserve the original image; apply EXIF orientation and proportional resizing to the working copy; record preprocessing metadata. Add optional 2× panorama upscaling with horizontal wrap padding, preview, and backend/statistics metadata. |
| High-resolution GS data | Prefer an SR panorama when present. Scale camera intrinsics with frame resolution, resize matching depth maps, reconcile sky-depth/normal resolutions, and record frame provenance. |
| Model and VLM configuration | Expose local checkpoint paths, an explicit WorldStereo base model, local-only loading, and CPU offload. Support configurable OpenAI-compatible vision endpoints, API keys, an optional fallback model, and deterministic VLM-free fallbacks. |
| GS optimization | Expose training duration, SH degree, point budget, losses, and frame selection. Add per-rank checkpoint resume and source-horizon handling; fix distributed bilateral-grid imports. Scale training duration without also changing densification cadence. |
| Unity export | Export coordinate-aligned GS PLYs and camera JSON, including available training views and generated trajectories. Add optional SPZ output and an exporter for existing trained PLYs. |

The changes live primarily in [`scripts/`](scripts/), [`world_gs_trainer.py`](hyworld2/worldgen/world_gs_trainer.py), [`gen_gs_data.py`](hyworld2/worldgen/gen_gs_data.py), and the worldgen camera, VLM, and model-loading utilities. These are engineering changes and configurable experiments, not a claim of improved quality on every scene.

## Pipeline

```text
Image + optional prompt
  → HY-Pano panorama → optional 2× panorama SR
  → trajectory planning/rendering → WorldStereo expansion
  → RGB/depth/normal/camera preparation → GS optimization
  → native scene PLY + Unity PLY/camera JSON
```

After panorama generation, the worldgen launcher exposes five stages:

| Stage | Component | Purpose |
| --- | --- | --- |
| 1 | `traj_generate.py` | Analyze the panorama and plan trajectories. |
| 2 | `traj_render.py` | Render initial trajectory views and prepare captions. |
| 3 | `video_gen.py` | Expand the scene using WorldStereo. |
| 4 | `gen_gs_data.py` | Prepare training views, depth, normals, and cameras. |
| 5 | `world_gs_trainer.py` | Optimize Gaussians and export assets. |

The inherited multi-view/video reconstruction path is also available through [`hyworld2/worldrecon`](hyworld2/worldrecon), without panorama generation.

## Installation

### Requirements

- Linux, Python 3.11, NVIDIA GPUs, and a compatible CUDA toolkit/compiler. CUDA 12.8 is the reference installation target.
- PyTorch 2.7.1 and torchvision 0.22.1, as pinned in the root requirements. Native rasterization and geometry extensions must match the environment.
- The worldgen wrappers use two GPU processes by default for stages 2–4. Full generation requires substantial GPU memory, host RAM, and disk space; this fork does not provide a measured consumer-GPU minimum.
- Access to the model repositories and, for the default planning route, an image-capable VLM endpoint.

### Environment and extensions

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

Worldgen training uses the custom `gsplat` variant; stock `gsplat` is not an equivalent backend for every option. The `spz` package is also used internally during Unity PLY conversion, even when saving an SPZ file is disabled.

For learned panorama SR, install the optional packages:

```bash
python -m pip install realesrgan basicsr
```

The [panorama module documentation](hyworld2/panogen/README.md) covers the alternative Hunyuan backend. Its standalone requirements have different NumPy/Transformers pins; do not install them over the shared environment without resolving those differences.

## Model weights

Weights and generated scenes are not distributed with this source repository. The default Qwen-based route expects:

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

| Component | Source |
| --- | --- |
| HY-Pano adapter and WorldMirror | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0) |
| Qwen panorama base model | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |
| WorldStereo | [hanshanxue/WorldStereo](https://huggingface.co/hanshanxue/WorldStereo) |
| WorldStereo base model | [Wan-AI/Wan2.1-I2V-14B-720P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) |
| Depth and normals | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) |
| Segmentation | [facebook/sam3](https://huggingface.co/facebook/sam3) |

For example, using the Hugging Face CLI after obtaining any required model access:

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

ZIM and Grounding DINO are resolved through the Hugging Face cache during planning. The SR helper can download Real-ESRGAN x2 weights. WorldStereo defaults to `WORLDSTEREO_LOCAL_FILES_ONLY=1`, so its local directories must be complete; that switch does not make all auxiliary models network-free.

## Quick start

Run commands from the repository root in the activated environment.

### 1. Configure runtime paths and the VLM

The launchers retain development-machine fallback paths. Override the executables explicitly:

```bash
export PYTHON_BIN="$(command -v python)"
export TORCHRUN_BIN="$(command -v torchrun)"
export CHECKPOINT_DIR="$PWD/checkpoints"
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES_STAGE1=0

cp .env.example .env.local
```

Edit `.env.local`: set `LLM_BASE_URL`, `LLM_NAME`, and `OPENAI_API_KEY` for an image-capable OpenAI-compatible endpoint. The model name must match your endpoint; the template is a placeholder. `LLM_FALLBACK_NAME` is optional. The wrappers load `.env`, then `.env.local`, so file values override exported values. Keep API keys out of commits.

### 2. Generate and inspect the panorama

```bash
PANO_PROMPT="A quiet campus courtyard with continuous paths and consistent architecture." \
PANO_SR_BACKEND=realesrgan \
bash scripts/run_image_to_panorama_sr.sh \
  /absolute/path/to/input.jpg "$PWD/inference_output/demo"
```

This stops after writing the panorama, SR image, and preview. Inspect layout and content before continuing. The default Qwen output is `1920×960` after edge blending; 2× SR produces `3840×1920`.

Use `PANO_SR_BACKEND=lanczos` for ordinary interpolation without learned SR dependencies. `auto` tries Real-ESRGAN and falls back to Lanczos; the actual backend is recorded in `panorama_sr.png.json`. Choose `realesrgan` explicitly when fallback would invalidate an experiment.

### 3. Generate the GS scene

```bash
bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"
```

The launcher prefers `panorama_sr.png` if present, otherwise `panorama.png`. Defaults are all five stages, 10,000 training steps, SH degree 0, and native/Unity GS exports.

For an unattended image-to-scene run:

```bash
PANO_SR=1 PANO_SR_BACKEND=realesrgan \
bash scripts/run_image_to_worldgen_2gpu.sh \
  /absolute/path/to/input.jpg "$PWD/inference_output/demo"
```

For an existing panorama, put it at `inference_output/demo/panorama.png` and run the panorama-to-worldgen wrapper. An SR image is optional and must be generated separately.

### 4. Select stages or resume optimization

```bash
# Rebuild GS data and train using the existing expansion results.
STAGES=45 bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"

# Resume a compatible training checkpoint to a longer horizon.
STAGES=5 GS_RESUME_GS_PATH="$PWD/inference_output/demo/gs_result_2gpu/ckpts" \
GS_RESUME_SOURCE_MAX_STEPS=10000 GS_MAX_STEPS=20000 \
GS_SAVE_STEPS=20000 GS_EVAL_STEPS=20000 GS_PLY_STEPS=20000 \
bash scripts/run_panorama_to_worldgen_2gpu.sh "$PWD/inference_output/demo"
```

Stage selection does not rebuild missing prerequisites. Resume requires training data and compatible per-rank checkpoints; a final PLY alone is not a training checkpoint. Use `--help` on the image/panorama wrappers for their argument reference.

## Configuration

Common settings for [`run_worldgen_2gpu.sh`](scripts/run_worldgen_2gpu.sh):

| Setting | Default | Purpose |
| --- | --- | --- |
| `STAGES` | `all` | Select stages, such as `3`, `45`, or `5`. |
| `WORLDGEN_NPROC_PER_NODE` | `2` | Worker count for stages 2–4; match visible GPUs. |
| `WORLDSTEREO_CPU_OFFLOAD` | `0` | CPU offload instead of WorldStereo FSDP. |
| `SPLITTED_RESOLUTION` | `720` | Perspective-view preparation resolution. |
| `TRAJ_PRESET` | `auto` | `light`, `balanced`, or `full`; auto selects light at resolution ≥720. |
| `GS_MAX_STEPS` | `10000` | Training horizon; adjust save/eval/PLY steps when changing it. |
| `GS_SH_DEGREE` | `0` | Spherical harmonics degree. |
| `GS_DOWNSAMPLE_PTS_NUM` | `2000000` | Initial point budget, not final Gaussian count. |
| `GS_DEPTH_LAMBDA1` / `GS_LPIPS_LAMBDA1` | `0.1` / `0.2` | Loss weights. |
| `GS_NORMAL_LOSS` | `1` | Normal supervision. |
| `GS_USE_MASK_GAUSSIAN` / `GS_USE_SCALE_REGULARIZATION` | `0` / `0` | Optional regularizers. |
| `GS_EXPORT_UNITY` / `GS_EXPORT_UNITY_SPZ` | `1` / `0` | Unity export and optional SPZ file. |

Override model paths with `PANO_QWEN_MODEL`, `PANO_LORA_PATH`, `WORLDSTEREO_REPO`, `WORLDSTEREO_BASE_MODEL`, `WORLDMIRROR_MODEL`, `MOGE_MODEL`, and `SAM3_MODEL` as needed.

`HY_OFFLINE_VLM=1` bypasses VLM calls using deterministic metadata/object fallbacks. Set `HY_OFFLINE_SCENE_TYPE` and `HY_OFFLINE_OBJECTS` for the scene. This changes planning behavior and does not eliminate model downloads.

## Outputs and viewing

Typical image-to-scene output:

```text
inference_output/demo/
├── input/
│   ├── original.jpg                 # Source copy; extension follows input
│   ├── input_rgb.png                # Oriented RGB working image
│   └── input_meta.json
├── panorama.png
├── panorama_sr.png                  # Optional
├── panorama_sr.png.json             # SR backend and statistics
├── panorama_sr_preview.jpg
├── render_results/                  # Expansion results and camera trajectories
├── gs_data/                         # RGB, depth, normals, cameras, initial points
├── gs_result_2gpu/
│   ├── cfg.yml
│   ├── ckpts/                       # Training checkpoints
│   ├── stats/                       # Evaluation records
│   ├── renders/                     # Validation renders
│   ├── ply/
│   │   ├── point_cloud_9999.ply      # Final native GS for 10,000 steps
│   │   └── position_meta_info.json
│   └── unity_export/
│       ├── <asset>_9999_unity_dx12.ply
│       └── <asset>_9999_unity_dx12.json
└── run_summary.txt
```

This tree is illustrative; intermediate/mesh outputs depend on executed stages. The image wrapper writes `run_summary.txt`. Export filenames use zero-based steps, so a 10,000-step run ends at `9999`. Unity asset names are inferred from input metadata or the scene directory.

### Native scene viewer

```bash
python hyworld2/worldgen/show_gs.py \
  --ckpt "$PWD/inference_output/demo/gs_result_2gpu/ply/point_cloud_9999.ply" \
  --port 8080
```

Keep `position_meta_info.json` beside the PLY for this viewer. Rendering a trained scene does not require the large panorama/world-expansion model weights.

### Unity handoff

Use the exported `*_unity_dx12.ply` and paired JSON with a compatible Gaussian Splatting renderer and camera importer. The JSON carries transformed camera poses, field of view, and available trajectories/training views. This repository does not supply the Unity renderer, XR integration, or application controls. Do not combine Unity-transformed cameras with the native-coordinate PLY.

To export an existing trained scene without rerunning optimization:

```bash
python scripts/export_unity_dx12_from_ply.py \
  --scene-dir "$PWD/inference_output/demo"
```

The exporter still uses `gs_data`, configuration, and camera/geometry metadata. Export before deleting training intermediates. Keep both native and Unity assets when both rendering paths are needed.

A compact archive should retain final PLYs, coordinate/camera metadata, original images, prompts, final panorama, configuration, and evaluation records. Also retain training data and checkpoints if optimization will resume. Preview videos and training caches are not the final scene representation.

### Multi-view/video reconstruction

```bash
INPUT_PATH="$PWD/examples/worldrecon/realistic/Messy_Room" \
OUTPUT_PATH="$PWD/inference_output/reconstruction_demo" \
bash scripts/run_worldrecon_2gpu.sh
```

This wrapper runs WorldMirror with two workers and FSDP. The [worldrecon module](hyworld2/worldrecon) and [inherited CLI reference](DOCUMENTATION.md) cover additional options. The reference describes upstream interfaces; the wrappers/defaults above describe this fork.

## Limitations and troubleshooting

- **Missing executable:** override `PYTHON_BIN` and `TORCHRUN_BIN`; machine-specific fallback paths remain in the launchers.
- **Out of GPU memory:** try WorldStereo CPU offload, lighter trajectories, or lower view resolution. These affect performance and coverage; single-GPU execution is not a drop-in equivalent of the default route.
- **Model loading fails:** check model access, local paths, and required subfolders. WorldStereo local-only loading is separate from auxiliary model caches.
- **Poor geometry:** inspect the panorama, expansion views, and cameras. Higher resolution does not guarantee correct geometry or recover unseen structure.
- **Cannot resume after cleanup:** retain training data and checkpoints for continued optimization. PLY archives remain renderable.
- **Unity orientation/framing differs:** use the paired export/cameras, respecting their coordinate system and field of view.

This is a research pipeline. Missing regions, floaters, and generated structures that differ from the input can occur. This fork does not claim a benchmark improvement, exact metric reconstruction, or an automatically deployable VR experience.

## Contributing

Report the commit, GPU/CUDA/PyTorch versions, launch command, relevant non-secret settings, and failing-stage logs. Include a shareable input or small reproduction for scene-specific issues where possible.

Keep changes scoped and describe verification. Distinguish panorama, expansion, training, and export results when reporting pipeline changes. Do not commit credentials, model weights, or generated training datasets.

## Acknowledgments and license

Built on [Tencent HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0), including HY-Pano, WorldNav, WorldStereo, and WorldMirror, with dependencies such as gsplat, MoGe, SAM3, Recast Navigation, and optionally Real-ESRGAN. Third-party code, weights, and assets retain their respective licenses.

The inherited project uses the [Tencent HY-WORLD 2.0 Community License Agreement](License.txt), including its use and geographic restrictions. This fork does not replace it with MIT or Apache-2.0. See [NOTICE](NOTICE) for attribution.

For research using the underlying models, cite the upstream work:

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint arXiv:2604.14268},
  year={2026}
}
```
