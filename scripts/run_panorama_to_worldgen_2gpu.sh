#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_panorama_to_worldgen_2gpu.sh /path/to/case_dir

Continues worldgen stages 1-5 from an approved case directory containing
panorama.png, preferably with panorama_sr.png already generated.

Environment overrides are forwarded to run_worldgen_2gpu.sh.
Common options:
  STAGES                   Default: all
  CUDA_VISIBLE_DEVICES     Default inherited by run_worldgen_2gpu.sh.
  GS_MAX_STEPS             Default inherited by run_worldgen_2gpu.sh.
  GS_SH_DEGREE             Default inherited by run_worldgen_2gpu.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: case dir is required." >&2
  usage >&2
  exit 2
fi

CASE_DIR="$1"
if [[ ! -d "$CASE_DIR" ]]; then
  echo "ERROR: case dir does not exist: $CASE_DIR" >&2
  exit 2
fi
CASE_DIR="$(cd "$CASE_DIR" && pwd)"
if [[ ! -f "$CASE_DIR/panorama.png" ]]; then
  echo "ERROR: panorama.png is required before worldgen: $CASE_DIR/panorama.png" >&2
  exit 2
fi
if [[ ! -f "$CASE_DIR/panorama_sr.png" ]]; then
  echo "WARNING: panorama_sr.png not found; worldgen will use panorama.png." >&2
fi

TARGET_PATH="$CASE_DIR" bash "$ROOT_DIR/scripts/run_worldgen_2gpu.sh"
