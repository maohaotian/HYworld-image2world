#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_image_to_worldgen_2gpu.sh /path/to/input_image /path/to/case_dir

Environment overrides:
  INPUT_IMAGE              Input image path. Used when arg1 is omitted.
  CASE_DIR                 Output scene directory. Used when arg2 is omitted.
  CASE_NAME                Scene directory name under OUTPUT_ROOT when CASE_DIR is omitted.
  OUTPUT_ROOT              Default: $ROOT_DIR/inference_output
  RUN_WORLDGEN             Run stages 1-5 after panorama generation. Default: 1
  OVERWRITE_PANORAMA       Regenerate panorama.png if it already exists. Default: 0

Input preprocessing:
  INPUT_MAX_EDGE           Resize input only if longest edge exceeds this value. Default: 4096
                           Set 0 to disable resizing.

Panorama generation:
  PANO_BACKEND             qwen or hunyuan. Default: qwen
  PANO_HEIGHT              Backend output height before edge blending. Default: 960
  PANO_WIDTH               Backend output width before edge blending. Default: 1952
  PANO_BLEND_WIDTH         Qwen backend edge blend/crop width. Default: 32
                           With 1952/960/32, final panorama is 1920x960.
  PANO_SEED                Default: 101
  PANO_PROMPT              Optional additional prompt.
  PANO_NEGATIVE_PROMPT     Qwen backend only.
  PANO_STEPS               Qwen --num-inference-steps. Default: 40
  PANO_GUIDANCE_SCALE      Qwen --guidance-scale. Default: 1.0
  PANO_TRUE_CFG_SCALE      Qwen --true-cfg-scale. Default: 7.5
  PANO_CROP_BORDER         Qwen input crop border fraction. Default: 0.0

Worldgen:
  STAGES                   Forwarded to scripts/run_worldgen_2gpu.sh. Default: all
  GS_STEPS_SCALER          Scales training duration only.
                           Use 2 to double the baseline run while keeping densify cadence fixed.
  LLM_BASE_URL/LLM_NAME    Forwarded to VLM stages.
  CUDA_VISIBLE_DEVICES     Default inherited by run_worldgen_2gpu.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for env_file in "$ROOT_DIR/.env" "$ROOT_DIR/.env.local"; do
  if [[ -f "$env_file" ]]; then
    set -a
    . "$env_file"
    set +a
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/torchrun}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
export PYTHON_BIN
export TORCHRUN_BIN

INPUT_IMAGE="${1:-${INPUT_IMAGE:-}}"
if [[ -z "$INPUT_IMAGE" ]]; then
  echo "ERROR: INPUT_IMAGE is required. Pass it as arg1 or set INPUT_IMAGE." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "ERROR: input image does not exist: $INPUT_IMAGE" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/inference_output}"
if [[ -n "${2:-}" ]]; then
  CASE_DIR="$2"
else
  CASE_DIR="${CASE_DIR:-}"
fi
if [[ -z "$CASE_DIR" ]]; then
  input_stem="$(basename "$INPUT_IMAGE")"
  input_stem="${input_stem%.*}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  CASE_NAME="${CASE_NAME:-${input_stem}_image_to_gs_${timestamp}}"
  CASE_DIR="$OUTPUT_ROOT/$CASE_NAME"
fi

RUN_WORLDGEN="${RUN_WORLDGEN:-1}"
OVERWRITE_PANORAMA="${OVERWRITE_PANORAMA:-0}"
INPUT_MAX_EDGE="${INPUT_MAX_EDGE:-4096}"

PANO_BACKEND="${PANO_BACKEND:-qwen}"
PANO_HEIGHT="${PANO_HEIGHT:-960}"
PANO_WIDTH="${PANO_WIDTH:-1952}"
PANO_BLEND_WIDTH="${PANO_BLEND_WIDTH:-32}"
PANO_SEED="${PANO_SEED:-101}"
PANO_PROMPT="${PANO_PROMPT:-}"
PANO_NEGATIVE_PROMPT="${PANO_NEGATIVE_PROMPT:-}"
PANO_STEPS="${PANO_STEPS:-40}"
PANO_GUIDANCE_SCALE="${PANO_GUIDANCE_SCALE:-1.0}"
PANO_TRUE_CFG_SCALE="${PANO_TRUE_CFG_SCALE:-7.5}"
PANO_CROP_BORDER="${PANO_CROP_BORDER:-0.0}"

PANO_QWEN_MODEL="${PANO_QWEN_MODEL:-$CHECKPOINT_DIR/Qwen-Image-Edit-2509}"
PANO_LORA_PATH="${PANO_LORA_PATH:-$CHECKPOINT_DIR/HY-World-2.0}"
PANO_LORA_SUBFOLDER="${PANO_LORA_SUBFOLDER:-HY-Pano-2.0}"
PANO_HUNYUAN_MODEL="${PANO_HUNYUAN_MODEL:-$CHECKPOINT_DIR/HY-World-2.0}"
PANO_HUNYUAN_SUBFOLDER="${PANO_HUNYUAN_SUBFOLDER:-HY-Pano-2.0}"

mkdir -p "$CASE_DIR/input"

INPUT_IMAGE_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$INPUT_IMAGE")"
CASE_DIR_ABS="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$CASE_DIR")"
CASE_DIR="$CASE_DIR_ABS"

PROCESSED_IMAGE="$CASE_DIR/input/input_rgb.png"
INPUT_META="$CASE_DIR/input/input_meta.json"
PANORAMA_PATH="$CASE_DIR/panorama.png"

echo "[image-to-worldgen] case dir: $CASE_DIR"
echo "[image-to-worldgen] input image: $INPUT_IMAGE_ABS"

"$PYTHON_BIN" -c '
from pathlib import Path
from PIL import Image, ImageOps
import json
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
max_edge = int(sys.argv[4])
original_copy = dst.parent / ("original" + src.suffix.lower())

dst.parent.mkdir(parents=True, exist_ok=True)
if src.resolve() != original_copy.resolve():
    shutil.copy2(src, original_copy)

with Image.open(src) as im:
    original_size = list(im.size)
    im = ImageOps.exif_transpose(im)
    exif_oriented_size = list(im.size)
    if im.mode != "RGB":
        im = im.convert("RGB")

    resized = False
    if max_edge > 0 and max(im.size) > max_edge:
        scale = max_edge / float(max(im.size))
        new_size = (max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale)))
        im = im.resize(new_size, Image.Resampling.LANCZOS)
        resized = True

    processed_size = list(im.size)
    im.save(dst)

meta = {
    "source_image": str(src),
    "original_copy": str(original_copy),
    "processed_image": str(dst),
    "original_size": original_size,
    "exif_oriented_size": exif_oriented_size,
    "processed_size": processed_size,
    "input_max_edge": max_edge,
    "resized": resized,
}
meta_path.write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
' "$INPUT_IMAGE_ABS" "$PROCESSED_IMAGE" "$INPUT_META" "$INPUT_MAX_EDGE"

if [[ ! -f "$PANORAMA_PATH" || "$OVERWRITE_PANORAMA" =~ ^(1|true|yes|on)$ ]]; then
  echo "[image-to-worldgen] generating panorama with backend: $PANO_BACKEND"
  if [[ "$PANO_BACKEND" == "qwen" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/hyworld2/panogen/pipeline_with_qwen_image.py" \
      --image "$PROCESSED_IMAGE" \
      --save "$PANORAMA_PATH" \
      --height "$PANO_HEIGHT" \
      --width "$PANO_WIDTH" \
      --blend-width "$PANO_BLEND_WIDTH" \
      --seed "$PANO_SEED" \
      --prompt "$PANO_PROMPT" \
      --negative-prompt "$PANO_NEGATIVE_PROMPT" \
      --num-inference-steps "$PANO_STEPS" \
      --guidance-scale "$PANO_GUIDANCE_SCALE" \
      --true-cfg-scale "$PANO_TRUE_CFG_SCALE" \
      --crop-border "$PANO_CROP_BORDER" \
      --pretrained-model-name-or-path "$PANO_QWEN_MODEL" \
      --lora-path "$PANO_LORA_PATH" \
      --lora-subfolder "$PANO_LORA_SUBFOLDER" \
      --reproduce
  elif [[ "$PANO_BACKEND" == "hunyuan" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/hyworld2/panogen/pipeline.py" \
      --image "$PROCESSED_IMAGE" \
      --save "$PANORAMA_PATH" \
      --height "$PANO_HEIGHT" \
      --width "$PANO_WIDTH" \
      --seed "$PANO_SEED" \
      --prompt "$PANO_PROMPT" \
      --pretrained-model-name-or-path "$PANO_HUNYUAN_MODEL" \
      --subfolder "$PANO_HUNYUAN_SUBFOLDER" \
      --reproduce
  else
    echo "ERROR: unsupported PANO_BACKEND=$PANO_BACKEND. Expected qwen or hunyuan." >&2
    exit 2
  fi
else
  echo "[image-to-worldgen] panorama already exists, skipping: $PANORAMA_PATH"
fi

"$PYTHON_BIN" -c '
from pathlib import Path
from PIL import Image
import json
import sys
p = Path(sys.argv[1])
with Image.open(p) as im:
    meta = {"panorama": str(p), "size": list(im.size)}
print(json.dumps(meta, indent=2))
' "$PANORAMA_PATH"

if [[ "$RUN_WORLDGEN" =~ ^(1|true|yes|on)$ ]]; then
  echo "[image-to-worldgen] running worldgen stages: ${STAGES:-all}"
  TARGET_PATH="$CASE_DIR" \
  RESULT_DIR="${RESULT_DIR:-$CASE_DIR/gs_result_2gpu}" \
  bash "$ROOT_DIR/scripts/run_worldgen_2gpu.sh"
else
  echo "[image-to-worldgen] RUN_WORLDGEN=$RUN_WORLDGEN, stopping after panorama generation."
fi

cat > "$CASE_DIR/run_summary.txt" <<EOF
case_dir=$CASE_DIR
input_image=$INPUT_IMAGE_ABS
processed_image=$PROCESSED_IMAGE
panorama=$PANORAMA_PATH
gs_result_dir=${RESULT_DIR:-$CASE_DIR/gs_result_2gpu}
unity_export=${RESULT_DIR:-$CASE_DIR/gs_result_2gpu}/unity_export
EOF

echo "[image-to-worldgen] done"
echo "[image-to-worldgen] summary: $CASE_DIR/run_summary.txt"
