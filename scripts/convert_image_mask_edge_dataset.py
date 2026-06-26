#!/usr/bin/env python
"""Convert image/mask/(optional edge) datasets to the TopoIWL-Net format."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion, distance_transform_edt, skeletonize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--edge-dir", type=Path, default=None)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--stem-prefix", default="")
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--edge-threshold", type=int, default=128)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--buffer-width", type=int, default=3)
    parser.add_argument("--distance-trunc", type=float, default=20.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes], key=lambda p: p.stem)


def save_image(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary(path: Path, threshold: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= threshold


def generated_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def save_distance_png(path: Path, distance: np.ndarray, trunc: float) -> None:
    ensure_dir(path.parent)
    scaled = (np.clip(distance, -trunc, trunc) + trunc) / (2 * trunc)
    Image.fromarray((scaled * 65535).astype(np.uint16)).save(path)


def split_stems(stems: list[str], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    return train, val, test


def write_split(path: Path, stems: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem"])
        writer.writeheader()
        for stem in stems:
            writer.writerow({"stem": stem})


def compute_image_stats(image_dir: Path) -> dict[str, object]:
    paths = sorted(image_dir.glob("*.png"))
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += (flat**2).sum(axis=0)
        pixel_count += flat.shape[0]
    mean = channel_sum / max(pixel_count, 1)
    var = np.maximum(channel_sq_sum / max(pixel_count, 1) - mean**2, 0.0)
    return {"num_images": len(paths), "mean": mean.tolist(), "std": np.sqrt(var).tolist()}


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    converted_stems: list[str] = []

    for index, image_path in enumerate(image_paths, start=1):
        mask_path = args.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            mask_path = args.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")

        edge_path = None
        if args.edge_dir is not None:
            candidate = args.edge_dir / f"{image_path.stem}.png"
            if candidate.exists():
                edge_path = candidate

        stem = f"{args.stem_prefix}{image_path.stem}"
        out_image = args.out_root / "images" / f"{stem}.png"
        out_mask = args.out_root / "masks" / f"{stem}.png"
        out_boundary = args.out_root / "boundary" / f"{stem}.png"
        out_buffer = args.out_root / "buffer" / f"{stem}.png"
        out_skeleton = args.out_root / "skeleton" / f"{stem}.png"
        out_distance = args.out_root / "distance_npy" / f"{stem}.npy"
        out_distance_png = args.out_root / "distance_png" / f"{stem}.png"

        if args.overwrite or not out_image.exists():
            save_image(out_image, read_rgb(image_path))
        mask = read_binary(mask_path, args.mask_threshold)
        if edge_path is not None:
            boundary = read_binary(edge_path, args.edge_threshold)
        else:
            boundary = generated_boundary(mask, args.boundary_width)
        buffer_mask = binary_dilation(boundary, iterations=args.buffer_width)
        skel = skeletonize(boundary)
        dist_inside = distance_transform_edt(mask)
        dist_outside = distance_transform_edt(~mask)
        signed_distance = np.clip(dist_outside - dist_inside, -args.distance_trunc, args.distance_trunc).astype(np.float32)
        signed_distance = signed_distance / float(args.distance_trunc)

        if args.overwrite or not out_mask.exists():
            save_image(out_mask, mask.astype(np.uint8) * 255)
            save_image(out_boundary, boundary.astype(np.uint8) * 255)
            save_image(out_buffer, buffer_mask.astype(np.uint8) * 255)
            save_image(out_skeleton, skel.astype(np.uint8) * 255)
            ensure_dir(out_distance.parent)
            np.save(out_distance, signed_distance)
            save_distance_png(out_distance_png, signed_distance, 1.0)

        water_pixels = int(mask.sum())
        boundary_pixels = int(boundary.sum())
        rows.append(
            {
                "stem": stem,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "source_edge": str(edge_path) if edge_path else "",
                "water_pixels": water_pixels,
                "boundary_pixels": boundary_pixels,
                "water_ratio": water_pixels / max(mask.size, 1),
                "boundary_ratio": boundary_pixels / max(mask.size, 1),
            }
        )
        converted_stems.append(stem)
        if index % 200 == 0:
            print(f"Converted {index}/{len(image_paths)}")

    with (args.out_root / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    train, val, test = split_stems(converted_stems, args.train_ratio, args.val_ratio, args.seed)
    write_split(args.out_root / "splits" / "train.csv", train)
    write_split(args.out_root / "splits" / "val.csv", val)
    write_split(args.out_root / "splits" / "test.csv", test)

    stats = compute_image_stats(args.out_root / "images")
    with (args.out_root / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(stats, handle, indent=2)
    print(f"Converted {len(converted_stems)} samples -> {args.out_root}")
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    print(stats)


if __name__ == "__main__":
    main()
