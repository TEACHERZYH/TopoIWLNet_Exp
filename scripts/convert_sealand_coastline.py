#!/usr/bin/env python
"""Convert SeaLand_Coastline_2025 into the unified TopoIWL-Net format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.utils.morphology import binary_dilation, distance_transform_edt, skeletonize  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("F:/2026/Remote Sensing_codex/datasets/SeaLand_Coastline_2025/raw"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("F:/2026/Remote Sensing_codex/datasets/SeaLand_Coastline_2025/processed/topoiwl_format"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem-prefix", default="sealand_")
    parser.add_argument("--buffer-width", type=int, default=3)
    parser.add_argument("--distance-trunc", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask.astype(bool).astype(np.uint8) * 255)).save(path)


def write_split(path: Path, stems: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem"])
        writer.writeheader()
        for stem in stems:
            writer.writerow({"stem": stem})


def stratified_split(rows: list[dict[str, object]], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    bins: dict[int, list[str]] = {}
    for row in rows:
        ratio = float(row["water_ratio"])
        bin_id = min(int(ratio * 10), 9)
        bins.setdefault(bin_id, []).append(str(row["stem"]))

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for stems in bins.values():
        rng.shuffle(stems)
        n = len(stems)
        n_test = max(1, round(n * test_ratio)) if n >= 3 and test_ratio > 0 else 0
        n_val = max(1, round(n * val_ratio)) if n - n_test >= 3 and val_ratio > 0 else 0
        test.extend(stems[:n_test])
        val.extend(stems[n_test : n_test + n_val])
        train.extend(stems[n_test + n_val :])

    for split in (train, val, test):
        split.sort()
    return train, val, test


def compute_image_stats(image_dir: Path) -> dict[str, object]:
    paths = sorted(image_dir.glob("*.png"))
    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0
    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
        flat = arr.reshape(-1, arr.shape[-1])
        if channel_sum is None:
            channel_sum = np.zeros(flat.shape[-1], dtype=np.float64)
            channel_sq_sum = np.zeros(flat.shape[-1], dtype=np.float64)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += (flat**2).sum(axis=0)
        pixel_count += flat.shape[0]
    if channel_sum is None or channel_sq_sum is None:
        return {"num_images": 0, "mean": [], "std": []}
    mean = channel_sum / max(pixel_count, 1)
    var = np.maximum(channel_sq_sum / max(pixel_count, 1) - mean**2, 0.0)
    return {"num_images": len(paths), "mean": mean.tolist(), "std": np.sqrt(var).tolist()}


def main() -> None:
    args = parse_args()
    image_dir = args.raw_root / "imgs"
    mask_dir = args.raw_root / "masks"
    edge_dir = args.raw_root / "img_edge"
    for required in (image_dir, mask_dir, edge_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    out_dirs = {
        "images": args.out_root / "images",
        "masks": args.out_root / "masks",
        "boundary": args.out_root / "boundary",
        "buffer": args.out_root / "buffer",
        "skeleton": args.out_root / "skeleton",
        "distance_npy": args.out_root / "distance_npy",
        "distance_png": args.out_root / "distance_png",
    }
    for path in out_dirs.values():
        ensure_dir(path)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        raw_stem = image_path.stem
        mask_path = mask_dir / image_path.name
        edge_path = edge_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing edge for {image_path.name}")

        stem = f"{args.stem_prefix}{raw_stem}"
        out_image = out_dirs["images"] / f"{stem}.png"
        if args.overwrite or not out_image.exists():
            shutil.copy2(image_path, out_image)

        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        boundary = np.asarray(Image.open(edge_path).convert("L")) > 127
        buffer_mask = binary_dilation(boundary, iterations=args.buffer_width)
        skel = skeletonize(boundary)
        dist_inside = distance_transform_edt(mask)
        dist_outside = distance_transform_edt(~mask)
        signed_distance = np.clip(dist_outside - dist_inside, -args.distance_trunc, args.distance_trunc).astype(np.float32)
        signed_distance = signed_distance / float(args.distance_trunc)

        save_mask(out_dirs["masks"] / f"{stem}.png", mask)
        save_mask(out_dirs["boundary"] / f"{stem}.png", boundary)
        save_mask(out_dirs["buffer"] / f"{stem}.png", buffer_mask)
        save_mask(out_dirs["skeleton"] / f"{stem}.png", skel)
        np.save(out_dirs["distance_npy"] / f"{stem}.npy", signed_distance)
        scaled = (np.clip(signed_distance, -1.0, 1.0) + 1.0) / 2.0
        Image.fromarray((scaled * 65535).astype(np.uint16)).save(out_dirs["distance_png"] / f"{stem}.png")

        rows.append(
            {
                "stem": stem,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "source_edge": str(edge_path),
                "water_pixels": int(mask.sum()),
                "boundary_pixels": int(boundary.sum()),
                "total_pixels": int(mask.size),
                "water_ratio": float(mask.mean()),
            }
        )
        if index % 200 == 0:
            print(f"Converted samples: {index}/{len(image_paths)}")

    ensure_dir(args.out_root)
    with (args.out_root / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    train, val, test = stratified_split(rows, args.val_ratio, args.test_ratio, args.seed)
    write_split(args.out_root / "splits" / "train.csv", train)
    write_split(args.out_root / "splits" / "val.csv", val)
    write_split(args.out_root / "splits" / "test.csv", test)

    summary = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "num_samples": len(rows),
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "buffer_width": args.buffer_width,
        "distance_trunc": args.distance_trunc,
        "image_stats_0_1": compute_image_stats(out_dirs["images"]),
    }
    (args.out_root / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
