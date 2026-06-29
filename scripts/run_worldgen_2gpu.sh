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
SPLITTED_RESOLUTION="${SPLITTED_RESOLUTION:-720}"
TRAJ_SKIP_EXIST="${TRAJ_SKIP_EXIST:-0}"
TRAJ_PRESET="${TRAJ_PRESET:-auto}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29621}"
MASTER_PORT_STAGE3="${MASTER_PORT_STAGE3:-29631}"
MASTER_PORT_STAGE4="${MASTER_PORT_STAGE4:-29641}"
WORLD_MIRROR_MASTER_PORT="${WORLD_MIRROR_MASTER_PORT:-29651}"
export WORLD_MIRROR_MASTER_PORT

GS_STEPS_SCALER="${GS_STEPS_SCALER:-1}"
GS_MAX_STEPS="${GS_MAX_STEPS:-10000}"
GS_SAVE_STEPS="${GS_SAVE_STEPS:-4000 5000 6000 8000 10000}"
GS_EVAL_STEPS="${GS_EVAL_STEPS:-1000 2000 3000 4000 5000 6000 7000 8000 9000 10000}"
GS_PLY_STEPS="${GS_PLY_STEPS:-4000 5000 6000 8000 10000}"
GS_RESUME_SOURCE_MAX_STEPS="${GS_RESUME_SOURCE_MAX_STEPS:-}"
GS_SH_DEGREE="${GS_SH_DEGREE:-0}"
GS_DOWNSAMPLE_PTS_NUM="${GS_DOWNSAMPLE_PTS_NUM:-2000000}"
GS_NORMAL_LOSS="${GS_NORMAL_LOSS:-1}"
GS_DEPTH_LAMBDA1="${GS_DEPTH_LAMBDA1:-0.1}"
GS_LPIPS_LAMBDA1="${GS_LPIPS_LAMBDA1:-0.2}"
GS_USE_MASK_GAUSSIAN="${GS_USE_MASK_GAUSSIAN:-0}"
GS_USE_SCALE_REGULARIZATION="${GS_USE_SCALE_REGULARIZATION:-0}"
GS_MEANS_LR="${GS_MEANS_LR:-3.2e-4}"
GS_VIDEO_ONLY="${GS_VIDEO_ONLY:-0}"
GS_PANO_NAME="${GS_PANO_NAME:-auto}"
GS_HIGH_RES_PANO="${GS_HIGH_RES_PANO:-auto}"
GS_RESUME_GS_PATH="${GS_RESUME_GS_PATH:-}"

WORLDSTEREO_REPO="${WORLDSTEREO_REPO:-$CHECKPOINT_DIR/WorldStereo}"
# Prefer the checked-in local Wan2.1 checkpoint to avoid stalls from remote fetches.
WORLDSTEREO_BASE_MODEL="${WORLDSTEREO_BASE_MODEL:-$CHECKPOINT_DIR/Wan2.1-I2V-14B-720P-Diffusers}"
WORLDSTEREO_LOCAL_FILES_ONLY="${WORLDSTEREO_LOCAL_FILES_ONLY:-1}"
WORLDMIRROR_MODEL="${WORLDMIRROR_MODEL:-$CHECKPOINT_DIR/HY-World-2.0}"
MOGE_MODEL="${MOGE_MODEL:-$CHECKPOINT_DIR/moge-2-vitl-normal/model.pt}"
SAM3_MODEL="${SAM3_MODEL:-$CHECKPOINT_DIR/sam3}"

export HY_WORLDSTEREO_PATH="$WORLDSTEREO_REPO"
export HY_WORLDSTEREO_BASE_MODEL="$WORLDSTEREO_BASE_MODEL"
export HY_WORLDMIRROR_PATH="$WORLDMIRROR_MODEL"
export HY_MOGE_PATH="$MOGE_MODEL"
export HY_SAM3_PATH="$SAM3_MODEL"

cd "$ROOT_DIR/hyworld2/worldgen"

is_truthy() {
  [[ "${1,,}" =~ ^(1|true|yes|on)$ ]]
}

if [[ "$TRAJ_PRESET" == "auto" ]]; then
  if [[ "$SPLITTED_RESOLUTION" =~ ^[0-9]+$ && "$SPLITTED_RESOLUTION" -ge 720 ]]; then
    TRAJ_PRESET="light"
  else
    TRAJ_PRESET="full"
  fi
fi

case "$TRAJ_PRESET" in
  light)
    TRAJ_APPLY_NAV_TRAJ="${TRAJ_APPLY_NAV_TRAJ:-0}"
    TRAJ_APPLY_UP_ROUTE="${TRAJ_APPLY_UP_ROUTE:-0}"
    TRAJ_APPLY_RECON_ITERATION="${TRAJ_APPLY_RECON_ITERATION:-0}"
    TRAJ_WONDER_TOPK="${TRAJ_WONDER_TOPK:-1}"
    TRAJ_RECON_TOPK="${TRAJ_RECON_TOPK:-2}"
    ;;
  balanced)
    TRAJ_APPLY_NAV_TRAJ="${TRAJ_APPLY_NAV_TRAJ:-1}"
    TRAJ_APPLY_UP_ROUTE="${TRAJ_APPLY_UP_ROUTE:-0}"
    TRAJ_APPLY_RECON_ITERATION="${TRAJ_APPLY_RECON_ITERATION:-0}"
    TRAJ_WONDER_TOPK="${TRAJ_WONDER_TOPK:-1}"
    TRAJ_RECON_TOPK="${TRAJ_RECON_TOPK:-2}"
    ;;
  full)
    TRAJ_APPLY_NAV_TRAJ="${TRAJ_APPLY_NAV_TRAJ:-1}"
    TRAJ_APPLY_UP_ROUTE="${TRAJ_APPLY_UP_ROUTE:-1}"
    TRAJ_APPLY_RECON_ITERATION="${TRAJ_APPLY_RECON_ITERATION:-1}"
    TRAJ_WONDER_TOPK="${TRAJ_WONDER_TOPK:-3}"
    TRAJ_RECON_TOPK="${TRAJ_RECON_TOPK:-5}"
    ;;
  *)
    echo "Unknown TRAJ_PRESET=$TRAJ_PRESET. Use auto, light, balanced, or full." >&2
    exit 2
    ;;
esac

TRAJ_FORCE_VLM="${TRAJ_FORCE_VLM:-1}"
TRAJ_SPLIT_VIEW_NUM="${TRAJ_SPLIT_VIEW_NUM:-3}"
TRAJ_UP_RIGHT="${TRAJ_UP_RIGHT:-60}"

OFFLINE_VLM_ARGS=()
if is_truthy "$HY_OFFLINE_VLM"; then
  OFFLINE_VLM_ARGS+=(--offline_vlm)
fi

TRAJ_SKIP_EXIST_ARGS=()
if is_truthy "$TRAJ_SKIP_EXIST"; then
  TRAJ_SKIP_EXIST_ARGS+=(--skip_exist)
fi

TRAJ_ARGS=(
  --split_view_num "$TRAJ_SPLIT_VIEW_NUM"
  --up_right "$TRAJ_UP_RIGHT"
  --wonder_topk "$TRAJ_WONDER_TOPK"
  --recon_topk "$TRAJ_RECON_TOPK"
)
if is_truthy "$TRAJ_APPLY_NAV_TRAJ"; then
  TRAJ_ARGS+=(--apply_nav_traj)
fi
if is_truthy "$TRAJ_APPLY_UP_ROUTE"; then
  TRAJ_ARGS+=(--apply_up_route)
fi
if is_truthy "$TRAJ_APPLY_RECON_ITERATION"; then
  TRAJ_ARGS+=(--apply_recon_iteration)
fi
if is_truthy "$TRAJ_FORCE_VLM"; then
  TRAJ_ARGS+=(--force_vlm)
fi

GS_NORMAL_LOSS_ARGS=()
if is_truthy "$GS_NORMAL_LOSS"; then
  GS_NORMAL_LOSS_ARGS+=(--normal_loss)
fi

GS_MASK_GAUSSIAN_ARGS=()
if is_truthy "$GS_USE_MASK_GAUSSIAN"; then
  GS_MASK_GAUSSIAN_ARGS+=(--use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection)
fi

GS_SCALE_REGULARIZATION_ARGS=()
if is_truthy "$GS_USE_SCALE_REGULARIZATION"; then
  GS_SCALE_REGULARIZATION_ARGS+=(--use_scale_regularization)
fi

GS_VIDEO_ONLY_ARGS=()
if is_truthy "$GS_VIDEO_ONLY"; then
  GS_VIDEO_ONLY_ARGS+=(--video_only)
fi

GS_RESUME_ARGS=()
if [[ -n "$GS_RESUME_GS_PATH" ]]; then
  GS_RESUME_ARGS+=(--resume_gs_path "$GS_RESUME_GS_PATH")
fi
if [[ -n "$GS_RESUME_SOURCE_MAX_STEPS" ]]; then
  GS_RESUME_ARGS+=(--resume_source_max_steps "$GS_RESUME_SOURCE_MAX_STEPS")
fi

WORLDSTEREO_LOCAL_FILES_ONLY_ARGS=()
if is_truthy "$WORLDSTEREO_LOCAL_FILES_ONLY"; then
  WORLDSTEREO_LOCAL_FILES_ONLY_ARGS+=(--local_files_only)
fi

if [[ "$GS_PANO_NAME" == "auto" ]]; then
  if [[ -f "$TARGET_PATH/panorama_sr.png" ]]; then
    GS_PANO_NAME="panorama_sr"
  else
    GS_PANO_NAME="panorama"
  fi
fi

GS_PANO_ARGS=(--pano_name "$GS_PANO_NAME")
if [[ "${GS_HIGH_RES_PANO,,}" == "auto" ]]; then
  if [[ "$GS_PANO_NAME" == "panorama_sr" ]]; then
    GS_PANO_ARGS+=(--high_res)
  fi
elif is_truthy "$GS_HIGH_RES_PANO"; then
  GS_PANO_ARGS+=(--high_res)
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
    --splitted_resolution "$SPLITTED_RESOLUTION" \
    "${OFFLINE_VLM_ARGS[@]}" \
    "${TRAJ_SKIP_EXIST_ARGS[@]}" \
    "${TRAJ_ARGS[@]}"
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
    "${WORLDSTEREO_LOCAL_FILES_ONLY_ARGS[@]}" \
    --fsdp --skip_exist
fi

if run_stage 4; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$TORCHRUN_BIN" --nproc_per_node=2 --master_port "$MASTER_PORT_STAGE4" gen_gs_data.py \
    --root_path "$TARGET_PATH" --save_normal --split_sky \
    "${GS_PANO_ARGS[@]}"
fi

if run_stage 5; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  "$PYTHON_BIN" -m world_gs_trainer default \
    --data_dir "$TARGET_PATH/gs_data" --result_dir "$RESULT_DIR" \
    --max_steps "$GS_MAX_STEPS" \
    --save_steps $GS_SAVE_STEPS \
    --eval_steps $GS_EVAL_STEPS \
    --ply_steps $GS_PLY_STEPS \
    --steps_scaler "$GS_STEPS_SCALER" \
    --sh_degree "$GS_SH_DEGREE" \
    --downsample_pts_num "$GS_DOWNSAMPLE_PTS_NUM" \
    --means_lr "$GS_MEANS_LR" \
    --save_ply --convert_to_spz --disable_video \
    "${GS_SCALE_REGULARIZATION_ARGS[@]}" --antialiased \
    --depth_loss --depth_lambda1 "$GS_DEPTH_LAMBDA1" \
    --lpips_lambda1 "$GS_LPIPS_LAMBDA1" \
    "${GS_NORMAL_LOSS_ARGS[@]}" --sky_depth_from_pcd \
    "${GS_MASK_GAUSSIAN_ARGS[@]}" \
    "${GS_VIDEO_ONLY_ARGS[@]}" \
    "${GS_RESUME_ARGS[@]}" \
    --use_anchor_protection --export_mesh \
    --strategy.refine-start-iter 400 --strategy.refine-stop-iter 2000 \
    --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 2000 \
    --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 \
    --strategy.prune-scale3d 0.1
fi
