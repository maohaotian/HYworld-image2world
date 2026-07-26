#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_image_to_panorama_sr.sh /path/to/input_image /path/to/case_dir

Generates panorama.png and panorama_sr.png, then stops before worldgen.

Environment overrides are forwarded to run_image_to_worldgen_2gpu.sh.
Common options:
  PANO_PROMPT              Additional text prompt for panorama generation.
  PANO_BACKEND             qwen or hunyuan. Default inherited from the base script.
  OVERWRITE_PANORAMA       Regenerate panorama.png if it already exists. Default: 0
  OVERWRITE_PANORAMA_SR    Regenerate panorama_sr.png if it already exists. Default: 0
  PANO_SR_BACKEND          auto, realesrgan, or lanczos. Default inherited.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  echo "ERROR: input image and case dir are required." >&2
  usage >&2
  exit 2
fi

PANO_SR=1 \
RUN_WORLDGEN=0 \
bash "$ROOT_DIR/scripts/run_image_to_worldgen_2gpu.sh" "$1" "$2"
