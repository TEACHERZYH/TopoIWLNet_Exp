#!/usr/bin/env python
"""Generate waterline boundary, buffer, skeleton, and signed-distance labels from binary masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.utils.io import list_files, read_mask, save_mask  # noqa: E402
from topoiwl.utils.morphology import (  # noqa: E402
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    has_fast_morphology,
    skeletonize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--buffer-width", type=int, default=3)
    parser.add_argument("--distance-trunc", type=float, default=20.0)
    return parser.parse_args()


def save_distance_png(path: Path, distance: np.ndarray, trunc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = (np.clip(distance, -trunc, trunc) + trunc) / (2 * trunc)
    Image.fromarray((scaled * 65535).astype(np.uint16)).save(path)


def main() -> None:
    args = parse_args()
    if not has_fast_morphology():
        print("Warning: scipy/scikit-image not fully available; using slower NumPy fallbacks.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir = args.out_dir / "boundary"
    buffer_dir = args.out_dir / "buffer"
    skeleton_dir = args.out_dir / "skeleton"
    distance_npy_dir = args.out_dir / "distance_npy"
    distance_png_dir = args.out_dir / "distance_png"

    rows = []
    files = list_files(args.mask_dir)
    if not files:
        raise SystemExit(f"No masks found in {args.mask_dir}")

    for idx, path in enumerate(files, start=1):
        mask = read_mask(path).astype(bool)
        dilated = binary_dilation(mask, iterations=args.boundary_width)
        eroded = binary_erosion(mask, iterations=args.boundary_width)
        boundary = np.logical_and(dilated, ~eroded)
        buffer_mask = binary_dilation(boundary, iterations=args.buffer_width)
        skel = skeletonize(boundary)
        dist_inside = distance_transform_edt(mask)
        dist_outside = distance_transform_edt(~mask)
        signed_distance = np.clip(dist_outside - dist_inside, -args.distance_trunc, args.distance_trunc).astype(np.float32)
        signed_distance = signed_distance / float(args.distance_trunc)

        stem = path.stem
        save_mask(boundary_dir / f"{stem}.png", boundary)
        save_mask(buffer_dir / f"{stem}.png", buffer_mask)
        save_mask(skeleton_dir / f"{stem}.png", skel)
        distance_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(distance_npy_dir / f"{stem}.npy", signed_distance)
        save_distance_png(distance_png_dir / f"{stem}.png", signed_distance, 1.0)
        rows.append({"stem": stem, "water_pixels": int(mask.sum()), "boundary_pixels": int(boundary.sum())})
        if idx % 200 == 0:
            print(f"Processed {idx}/{len(files)}")

    with (args.out_dir / "label_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "water_pixels", "boundary_pixels"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated labels for {len(files)} masks -> {args.out_dir}")


if __name__ == "__main__":
    main()
