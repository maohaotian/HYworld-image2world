#!/usr/bin/env python3
"""Create a high-resolution panorama_sr.png with horizontal wrap padding.

The script treats equirectangular panoramas as horizontally periodic images:
it copies the right edge to the left and the left edge to the right before
upscaling, then crops the padding away after upscaling. This keeps ordinary
image SR models from seeing the panorama seam as a hard image boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REALESRGAN_X2_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale panorama.png to panorama_sr.png with left-right wrap padding."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Scene directory. Defaults input to panorama.png and output to panorama_sr.png.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root for default checkpoints/RealESRGAN location.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input panorama image.")
    parser.add_argument("--output", type=Path, default=None, help="Output super-res panorama image.")
    parser.add_argument(
        "--backend",
        choices=("auto", "realesrgan", "lanczos"),
        default="auto",
        help="Upscaling backend. auto tries Real-ESRGAN, then falls back to Lanczos.",
    )
    parser.add_argument("--scale", type=int, default=2, choices=(2,), help="Upscale factor.")
    parser.add_argument(
        "--pad",
        type=int,
        default=128,
        help="Horizontal wrap padding in source pixels before upscaling.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Real-ESRGAN x2 model path. Downloaded under checkpoints/RealESRGAN if omitted.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download Real-ESRGAN weights when missing.",
    )
    parser.add_argument("--tile", type=int, default=400, help="Real-ESRGAN tile size. 0 disables tiling.")
    parser.add_argument("--tile-pad", type=int, default=32, help="Real-ESRGAN tile overlap padding.")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable half precision for Real-ESRGAN.",
    )
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device id for Real-ESRGAN.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Optional side-by-side crop preview path.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional metadata JSON path. Defaults to <output>.json.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.scene_dir is not None:
        scene_dir = args.scene_dir
        input_path = args.input or scene_dir / "panorama.png"
        output_path = args.output or scene_dir / "panorama_sr.png"
    else:
        if args.input is None:
            raise SystemExit("ERROR: pass --scene-dir or --input")
        input_path = args.input
        output_path = args.output or input_path.with_name(f"{input_path.stem}_sr{input_path.suffix}")
        scene_dir = input_path.parent

    preview_path = args.preview or output_path.with_name(f"{output_path.stem}_preview.jpg")
    meta_path = args.meta or output_path.with_suffix(output_path.suffix + ".json")
    return scene_dir.resolve(), input_path.resolve(), output_path.resolve(), preview_path.resolve(), meta_path.resolve()


def wrap_pad_lr(image: Image.Image, pad_px: int) -> tuple[Image.Image, int]:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    pad_px = max(0, min(int(pad_px), max(0, w // 4)))
    if pad_px == 0:
        return Image.fromarray(arr), 0
    padded = np.concatenate([arr[:, -pad_px:], arr, arr[:, :pad_px]], axis=1)
    return Image.fromarray(padded), pad_px


def crop_unpad_lr(image: Image.Image, orig_w: int, orig_h: int, pad_px: int, scale: int) -> Image.Image:
    left = pad_px * scale
    right = left + orig_w * scale
    bottom = orig_h * scale
    return image.crop((left, 0, right, bottom))


def download_realesrgan_x2(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"[pano-sr] downloading Real-ESRGAN x2 weights: {REALESRGAN_X2_URL}")
    urllib.request.urlretrieve(REALESRGAN_X2_URL, tmp_path)
    tmp_path.replace(model_path)


def install_torchvision_compat_shim() -> None:
    """BasicSR 1.4.x imports torchvision.transforms.functional_tensor.

    Newer torchvision versions removed that module. The functions BasicSR uses
    are available from torchvision.transforms.functional, so aliasing the module
    keeps the optional Real-ESRGAN backend usable without patching site-packages.
    """
    try:
        import torchvision.transforms.functional as functional
    except Exception:
        return
    sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)


def upscale_realesrgan(
    image: Image.Image,
    *,
    scale: int,
    model_path: Path,
    tile: int,
    tile_pad: int,
    fp32: bool,
    gpu_id: int | None,
) -> Image.Image:
    install_torchvision_compat_shim()
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        raise RuntimeError(
            "Real-ESRGAN backend requires optional packages: realesrgan and basicsr. "
            "Install them or use --backend lanczos."
        ) from exc

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    half = (not fp32) and torch.cuda.is_available()
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=half,
        gpu_id=gpu_id,
    )
    # Real-ESRGAN expects BGR numpy input.
    rgb = np.asarray(image.convert("RGB"))
    bgr = rgb[:, :, ::-1]
    output_bgr, _ = upsampler.enhance(bgr, outscale=scale)
    output_rgb = output_bgr[:, :, ::-1]
    return Image.fromarray(output_rgb)


def upscale_lanczos(image: Image.Image, *, scale: int) -> Image.Image:
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)


def seam_mae(image: Image.Image, width: int = 16) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    w = arr.shape[1]
    width = max(1, min(width, w // 8))
    return float(np.abs(arr[:, :width] - arr[:, -width:]).mean())


def laplacian_variance(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    center = gray[1:-1, 1:-1] * -4.0
    lap = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def save_preview(original: Image.Image, sr: Image.Image, path: Path, *, scale: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ow, oh = original.size
    sw, sh = sr.size
    crop_w = min(512, ow // 3)
    crop_h = min(320, oh // 2)
    boxes = [
        (0, max(0, oh // 2 - crop_h // 2), crop_w, max(0, oh // 2 - crop_h // 2) + crop_h),
        (
            max(0, ow // 2 - crop_w // 2),
            max(0, oh // 2 - crop_h // 2),
            max(0, ow // 2 - crop_w // 2) + crop_w,
            max(0, oh // 2 - crop_h // 2) + crop_h,
        ),
        (max(0, ow - crop_w), max(0, oh // 2 - crop_h // 2), ow, max(0, oh // 2 - crop_h // 2) + crop_h),
    ]
    rows = []
    for box in boxes:
        orig_crop = original.crop(box)
        orig_up = orig_crop.resize((orig_crop.width * scale, orig_crop.height * scale), Image.Resampling.LANCZOS)
        sr_box = tuple(int(v * scale) for v in box)
        sr_crop = sr.crop(sr_box)
        row = Image.new("RGB", (orig_up.width + sr_crop.width, max(orig_up.height, sr_crop.height)), (0, 0, 0))
        row.paste(orig_up, (0, 0))
        row.paste(sr_crop, (orig_up.width, 0))
        rows.append(row)
    preview = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (0, 0, 0))
    y = 0
    for row in rows:
        preview.paste(row, (0, y))
        y += row.height
    preview.save(path, quality=95)


def main() -> int:
    args = parse_args()
    scene_dir, input_path, output_path, preview_path, meta_path = resolve_paths(args)
    if not input_path.exists():
        raise SystemExit(f"ERROR: input panorama does not exist: {input_path}")
    if output_path.exists() and not args.force:
        raise SystemExit(f"ERROR: output exists, pass --force to overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = args.repo_root.resolve() if args.repo_root is not None else Path.cwd().resolve()
    model_path = args.model_path
    if model_path is None:
        model_path = repo_root / "checkpoints" / "RealESRGAN" / "RealESRGAN_x2plus.pth"
    model_path = model_path.resolve()

    original = Image.open(input_path).convert("RGB")
    orig_w, orig_h = original.size
    padded, pad_px = wrap_pad_lr(original, args.pad)

    backend_used = args.backend
    started = time.time()
    if args.backend in ("auto", "realesrgan"):
        if not model_path.exists() and args.download:
            download_realesrgan_x2(model_path)
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Real-ESRGAN model not found: {model_path}")
            sr_padded = upscale_realesrgan(
                padded,
                scale=args.scale,
                model_path=model_path,
                tile=args.tile,
                tile_pad=args.tile_pad,
                fp32=args.fp32,
                gpu_id=args.gpu_id,
            )
            backend_used = "realesrgan"
        except Exception as exc:
            if args.backend == "realesrgan":
                raise
            print(f"[pano-sr] Real-ESRGAN unavailable, falling back to Lanczos: {exc}")
            sr_padded = upscale_lanczos(padded, scale=args.scale)
            backend_used = "lanczos"
    else:
        sr_padded = upscale_lanczos(padded, scale=args.scale)
        backend_used = "lanczos"

    sr = crop_unpad_lr(sr_padded, orig_w, orig_h, pad_px, args.scale)
    sr.save(output_path)
    save_preview(original, sr, preview_path, scale=args.scale)

    meta: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "preview": str(preview_path),
        "backend": backend_used,
        "scale": args.scale,
        "pad_px": pad_px,
        "tile": args.tile if backend_used == "realesrgan" else None,
        "tile_pad": args.tile_pad if backend_used == "realesrgan" else None,
        "model_path": str(model_path) if backend_used == "realesrgan" else None,
        "input_size": [orig_w, orig_h],
        "padded_size": list(padded.size),
        "output_size": list(sr.size),
        "input_seam_mae_16px": seam_mae(original, width=16),
        "output_seam_mae_32px": seam_mae(sr, width=32),
        "input_laplacian_var": laplacian_variance(original),
        "output_laplacian_var": laplacian_variance(sr),
        "elapsed_sec": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
