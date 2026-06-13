#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for env_file in "$ROOT_DIR/.env" "$ROOT_DIR/.env.local"; do
  if [[ -f "$env_file" ]]; then
    set -a
    . "$env_file"
    set +a
  fi
done

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/maohaotian/.conda/envs/hyworld2/bin/torchrun}"
export PYTHON_BIN
export TORCHRUN_BIN

: "${TARGET_PATH:?Set TARGET_PATH to a scene directory containing panorama.png}"

RESULT_DIR="${RESULT_DIR:-$TARGET_PATH/gs_result_2gpu}"
LLM_ADDR="${LLM_ADDR:-localhost}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_NAME="${LLM_NAME:-gpt-5.4}"
# Set OPENAI_API_KEY or LLM_API_KEY when using a closed-source OpenAI-compatible endpoint.
LLM_BASE_URL="${LLM_BASE_URL:-http://${LLM_ADDR}:${LLM_PORT}/v1}"
STAGES="${STAGES:-all}"
HY_OFFLINE_VLM="${HY_OFFLINE_VLM:-0}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29621}"
MASTER_PORT_STAGE3="${MASTER_PORT_STAGE3:-29631}"
MASTER_PORT_STAGE4="${MASTER_PORT_STAGE4:-29641}"
WORLD_MIRROR_MASTER_PORT="${WORLD_MIRROR_MASTER_PORT:-29651}"
export WORLD_MIRROR_MASTER_PORT

GS_STEPS_SCALER="${GS_STEPS_SCALER:-1}"

WORLDSTEREO_REPO="${WORLDSTEREO_REPO:-$CHECKPOINT_DIR/WorldStereo}"
WORLDSTEREO_BASE_MODEL="${WORLDSTEREO_BASE_MODEL:-$CHECKPOINT_DIR/Wan2.1-I2V-14B-480P-Diffusers}"
WORLDMIRROR_MODEL="${WORLDMIRROR_MODEL:-$CHECKPOINT_DIR/HY-World-2.0}"
MOGE_MODEL="${MOGE_MODEL:-$CHECKPOINT_DIR/moge-2-vitl-normal/model.pt}"
SAM3_MODEL="${SAM3_MODEL:-$CHECKPOINT_DIR/sam3}"

export HY_WORLDSTEREO_PATH="$WORLDSTEREO_REPO"
export HY_WORLDSTEREO_BASE_MODEL="$WORLDSTEREO_BASE_MODEL"
export HY_WORLDMIRROR_PATH="$WORLDMIRROR_MODEL"
export HY_MOGE_PATH="$MOGE_MODEL"
export HY_SAM3_PATH="$SAM3_MODEL"

cd "$ROOT_DIR/hyworld2/worldgen"

OFFLINE_VLM_ARGS=()
if [[ "${HY_OFFLINE_VLM,,}" =~ ^(1|true|yes|on)$ ]]; then
  OFFLINE_VLM_ARGS+=(--offline_vlm)
fi

run_stage() {
  [[ "$STAGES" == "all" || "$STAGES" == *"$1"* ]]
}

if run_stage 1; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_STAGE1:-0}" \
  "$PYTHON_BIN" traj_generate.py \
    --target_path "$TARGET_PATH" \
    --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME" \
    --llm_base_url "$LLM_BASE_URL" \
    "${OFFLINE_VLM_ARGS[@]}" \
    --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
fi

if run_stage 2; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$TORCHRUN_BIN" --nproc_per_node=2 --master_port "$MASTER_PORT_STAGE2" traj_render.py \
    --target_path "$TARGET_PATH" \
    --llm_addr "$LLM_ADDR" --llm_port "$LLM_PORT" --llm_name "$LLM_NAME" \
    --llm_base_url "$LLM_BASE_URL" \
    "${OFFLINE_VLM_ARGS[@]}"
fi

if run_stage 3; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$TORCHRUN_BIN" --nproc_per_node=2 --master_port "$MASTER_PORT_STAGE3" video_gen.py \
    --target_path "$TARGET_PATH" \
    --worldstereo_repo "$WORLDSTEREO_REPO" \
    --worldstereo_base_model "$WORLDSTEREO_BASE_MODEL" \
    --moge_model "$MOGE_MODEL" \
    --sam3_model "$SAM3_MODEL" \
    --local_files_only --fsdp --skip_exist
fi

if run_stage 4; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$TORCHRUN_BIN" --nproc_per_node=2 --master_port "$MASTER_PORT_STAGE4" gen_gs_data.py \
    --root_path "$TARGET_PATH" --save_normal --split_sky
fi

if run_stage 5; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$PYTHON_BIN" -m world_gs_trainer default \
    --data_dir "$TARGET_PATH/gs_data" --result_dir "$RESULT_DIR" \
    --steps_scaler "$GS_STEPS_SCALER" \
    --save_ply --convert_to_spz --disable_video \
    --use_scale_regularization --antialiased \
    --depth_loss --normal_loss --sky_depth_from_pcd \
    --use_mask_gaussian --mask_export_stochastic \
    --no-mask-export-anchor-protection --use_anchor_protection --export_mesh \
    --strategy.refine-start-iter 400 --strategy.refine-stop-iter 2000 \
    --strategy.refine-every 200 --strategy.refine-scale2d-stop-iter 2000 \
    --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 \
    --strategy.prune-scale3d 0.1
fi
