#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/torchrun}"
MODEL_ROOT="${WORLDRECON_MODEL_ROOT:-$CHECKPOINT_DIR/HY-World-2.0}"

: "${INPUT_PATH:?Set INPUT_PATH to an image directory or video path}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/inference_output/worldrecon_2gpu}"

cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
"$TORCHRUN_BIN" --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --pretrained_model_name_or_path "$MODEL_ROOT" \
  --subfolder HY-WorldMirror-2.0 \
  --use_fsdp --enable_bf16
