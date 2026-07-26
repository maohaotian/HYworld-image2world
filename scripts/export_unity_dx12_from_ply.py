#!/usr/bin/env python3
"""Export Unity DX12 assets from an existing trained 3DGS PLY."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


WORLDGEN_DIR = _repo_root() / "hyworld2" / "worldgen"
sys.path.insert(0, str(WORLDGEN_DIR))

from gs.opencv import Parser  # noqa: E402
from world_gs_trainer import export_unity_dx12_assets  # noqa: E402


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def _read_cfg_scalars(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("-"):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip()] = _parse_scalar(value)
    return values


def _bool_cfg(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    value = cfg.get(key, default)
    return bool(value)


def _int_cfg(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    return default if value is None else int(value)


def _latest_ply(result_dir: Path) -> tuple[Path, int]:
    candidates = []
    for path in (result_dir / "ply").glob("point_cloud_*.ply"):
        match = re.search(r"point_cloud_(\d+)\.ply$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No point_cloud_*.ply found under {result_dir / 'ply'}")
    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing gs_data and render_results.")
    parser.add_argument("--result-dir", help="GS result directory. Defaults to SCENE_DIR/gs_result_2gpu.")
    parser.add_argument("--data-dir", help="GS data directory. Defaults to SCENE_DIR/gs_data.")
    parser.add_argument("--ply-path", help="Existing trained point_cloud_*.ply path. Defaults to latest in result dir.")
    parser.add_argument("--step", type=int, help="Export step used in output filenames. Defaults to the PLY step.")
    parser.add_argument("--write-spz", action="store_true", help="Also write a Unity-aligned SPZ.")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    result_dir = Path(args.result_dir).resolve() if args.result_dir else scene_dir / "gs_result_2gpu"
    data_dir = Path(args.data_dir).resolve() if args.data_dir else scene_dir / "gs_data"
    cfg = _read_cfg_scalars(result_dir / "cfg.yml")

    if args.ply_path:
        ply_path = Path(args.ply_path).resolve()
        step = args.step
        if step is None:
            match = re.search(r"point_cloud_(\d+)\.ply$", ply_path.name)
            step = int(match.group(1)) if match else 0
    else:
        ply_path, step = _latest_ply(result_dir)
        if args.step is not None:
            step = args.step

    print(f"[unity-export] scene_dir={scene_dir}")
    print(f"[unity-export] data_dir={data_dir}")
    print(f"[unity-export] result_dir={result_dir}")
    print(f"[unity-export] ply_path={ply_path}")

    gs_parser = Parser(
        data_dir=str(data_dir),
        factor=_int_cfg(cfg, "data_factor", 1),
        normalize=not _bool_cfg(cfg, "no_normalize", False),
        test_every=_int_cfg(cfg, "test_every", 8),
        downsample_pts_num=cfg.get("downsample_pts_num", None),
        downsample_mode=str(cfg.get("downsample_mode", "geometry_aware")),
        detect_anchor_candidates=_bool_cfg(cfg, "use_anchor_protection", False),
        world_rank=0,
        world_size=1,
        local_rank=0,
        align_sky_only=_bool_cfg(cfg, "init_align_sky_only", False),
    )

    exported = export_unity_dx12_assets(
        ply_path=str(ply_path),
        data_dir=str(data_dir),
        result_dir=str(result_dir),
        step=int(step),
        up_direction_rdf=np.array(gs_parser.up_direction),
        world_transform=np.array(gs_parser.transform),
        antialiased=_bool_cfg(cfg, "antialiased", True),
        write_spz=args.write_spz,
        unity_camera_trajectory=cfg.get("unity_camera_trajectory", None),
        unity_camera_fps=float(cfg.get("unity_camera_fps", 30.0) or 30.0),
        test_every=_int_cfg(cfg, "test_every", 8),
        pano_only=_bool_cfg(cfg, "pano_only", False),
        video_only=_bool_cfg(cfg, "video_only", False),
        video_prefix_filter=cfg.get("video_prefix_filter", None),
        video_exclude_substring=cfg.get("video_exclude_substring", None),
        pano_azimuth_interval=_int_cfg(cfg, "pano_azimuth_interval", 1),
        polar_up_azimuth_interval=_int_cfg(cfg, "polar_up_azimuth_interval", 1),
        polar_up_extra_layers=_bool_cfg(cfg, "polar_up_extra_layers", True),
        polar_down_azimuth_interval=_int_cfg(cfg, "polar_down_azimuth_interval", 1),
    )
    print("[unity-export] wrote:")
    for key, value in exported.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
