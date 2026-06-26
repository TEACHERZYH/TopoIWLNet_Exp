#!/usr/bin/env python
"""Convert large GLH-Water image/mask tiles into TopoIWL patch format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion, distance_transform_edt, skeletonize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--buffer-width", type=int, default=3)
    parser.add_argument("--distance-trunc", type=float, default=20.0)
    parser.add_argument("--min-boundary-ratio", type=float, default=0.0005)
    parser.add_argument("--max-boundary-per-source", type=int, default=0)
    parser.add_argument("--nonboundary-keep-prob", type=float, default=0.02)
    parser.add_argument("--max-nonboundary-per-source", type=int, default=16)
    parser.add_argument("--max-sources-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem-prefix", default="glh_")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes], key=lambda p: p.stem)


def save_image(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def generated_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def signed_distance(mask: np.ndarray, trunc: float) -> np.ndarray:
    if mask.all():
        return np.full(mask.shape, -1.0, dtype=np.float32)
    if not mask.any():
        return np.full(mask.shape, 1.0, dtype=np.float32)
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    distance = np.clip(dist_outside - dist_inside, -trunc, trunc).astype(np.float32)
    return distance / float(trunc)


def save_distance_png(path: Path, distance: np.ndarray) -> None:
    ensure_dir(path.parent)
    scaled = (np.clip(distance, -1.0, 1.0) + 1.0) / 2.0
    Image.fromarray((scaled * 65535).astype(np.uint16)).save(path)


def write_split(path: Path, stems: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem"])
        writer.writeheader()
        for stem in stems:
            writer.writerow({"stem": stem})


def image_stats(paths: list[Path]) -> dict[str, object]:
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


def candidate_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    tail = length - patch_size
    if positions and positions[-1] != tail:
        positions.append(tail)
    return positions


def should_keep(
    boundary_ratio: float,
    boundary_count: int,
    nonboundary_count: int,
    rng: random.Random,
    min_boundary_ratio: float,
    max_boundary_per_source: int,
    nonboundary_keep_prob: float,
    max_nonboundary_per_source: int,
) -> tuple[bool, bool]:
    if boundary_ratio >= min_boundary_ratio:
        if max_boundary_per_source > 0 and boundary_count >= max_boundary_per_source:
            return False, False
        return True, False
    if nonboundary_count >= max_nonboundary_per_source:
        return False, True
    return rng.random() < nonboundary_keep_prob, True


def convert_split(args: argparse.Namespace, split: str, rng: random.Random) -> tuple[list[dict[str, object]], list[str]]:
    image_dir = args.raw_root / split / "img"
    mask_dir = args.raw_root / split / "label"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing GLH split folders under {args.raw_root / split}")

    rows: list[dict[str, object]] = []
    stems: list[str] = []
    image_paths = list_images(image_dir)
    if args.max_sources_per_split > 0:
        image_paths = image_paths[: args.max_sources_per_split]

    for image_index, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        if image.size != mask_image.size:
            raise ValueError(f"Image/mask size mismatch: {image_path} {image.size} vs {mask_path} {mask_image.size}")

        xs = candidate_positions(image.size[0], args.patch_size, args.stride)
        ys = candidate_positions(image.size[1], args.patch_size, args.stride)
        boundary_count = 0
        nonboundary_count = 0
        source_kept = 0

        positions = [(x, y) for y in ys for x in xs]
        rng.shuffle(positions)

        for x, y in positions:
            box = (x, y, x + args.patch_size, y + args.patch_size)
            mask = np.asarray(mask_image.crop(box), dtype=np.uint8) >= args.mask_threshold
            boundary = generated_boundary(mask, args.boundary_width)
            boundary_ratio = float(boundary.mean())
            keep, is_nonboundary = should_keep(
                boundary_ratio,
                boundary_count,
                nonboundary_count,
                rng,
                args.min_boundary_ratio,
                args.max_boundary_per_source,
                args.nonboundary_keep_prob,
                args.max_nonboundary_per_source,
            )
            if not keep:
                continue
            if not is_nonboundary:
                boundary_count += 1
            if is_nonboundary:
                nonboundary_count += 1

            stem = f"{args.stem_prefix}{split}_{image_path.stem}_{y:05d}_{x:05d}"
            out_image = args.out_root / "images" / f"{stem}.png"
            out_mask = args.out_root / "masks" / f"{stem}.png"
            out_boundary = args.out_root / "boundary" / f"{stem}.png"
            out_buffer = args.out_root / "buffer" / f"{stem}.png"
            out_skeleton = args.out_root / "skeleton" / f"{stem}.png"
            out_distance = args.out_root / "distance_npy" / f"{stem}.npy"
            out_distance_png = args.out_root / "distance_png" / f"{stem}.png"

            if args.overwrite or not out_image.exists():
                save_image(out_image, np.asarray(image.crop(box), dtype=np.uint8))
                save_image(out_mask, mask.astype(np.uint8) * 255)
                save_image(out_boundary, boundary.astype(np.uint8) * 255)
                save_image(out_buffer, binary_dilation(boundary, iterations=args.buffer_width).astype(np.uint8) * 255)
                save_image(out_skeleton, skeletonize(boundary).astype(np.uint8) * 255)
                ensure_dir(out_distance.parent)
                dist = signed_distance(mask, args.distance_trunc)
                np.save(out_distance, dist)
                save_distance_png(out_distance_png, dist)

            rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "x": x,
                    "y": y,
                    "water_ratio": float(mask.mean()),
                    "boundary_ratio": boundary_ratio,
                }
            )
            stems.append(stem)
            source_kept += 1

            boundary_cap_reached = args.max_boundary_per_source > 0 and boundary_count >= args.max_boundary_per_source
            nonboundary_cap_reached = nonboundary_count >= args.max_nonboundary_per_source
            if boundary_cap_reached and nonboundary_cap_reached:
                break
        print(f"{split}: {image_index} {image_path.name}, kept {source_kept} patches")
    return rows, stems


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_rows: list[dict[str, object]] = []
    split_stems: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        rows, stems = convert_split(args, split, rng)
        all_rows.extend(rows)
        split_stems[split] = stems
        write_split(args.out_root / "splits" / f"{split}.csv", stems)

    manifest = args.out_root / "source_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    train_images = [args.out_root / "images" / f"{stem}.png" for stem in split_stems["train"]]
    stats = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "min_boundary_ratio": args.min_boundary_ratio,
        "max_boundary_per_source": args.max_boundary_per_source,
        "nonboundary_keep_prob": args.nonboundary_keep_prob,
        "max_nonboundary_per_source": args.max_nonboundary_per_source,
        "split_counts": {split: len(stems) for split, stems in split_stems.items()},
        "train_image_stats_0_1": image_stats(train_images),
    }
    with (args.out_root / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
