#!/usr/bin/env python
"""Convert the GF6_TCUNet dataset into the unified TopoIWL-Net format."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("F:/2026/Remote Sensing_codex/datasets/GF6_TCUNet/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("F:/2026/Remote Sensing_codex/datasets/GF6_TCUNet/processed/topoiwl_format"))
    parser.add_argument("--image-source", choices=["img_345", "images"], default="img_345")
    parser.add_argument("--label-threshold", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem-prefix", default="gf6_")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert-unlabeled-test", action="store_true")
    return parser.parse_args()


def read_tiff(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[-1] > 16:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype == np.uint8:
        out = arr
    elif np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        out = np.clip(arr.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [1, 99])
            out = np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out


def save_png(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


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

    for part in (train, val, test):
        part.sort()
    return train, val, test


def convert_labeled(args: argparse.Namespace) -> list[dict[str, object]]:
    image_dir = args.raw_root / "train" / args.image_source
    label_dir = args.raw_root / "train" / "labels"
    out_image_dir = args.out_root / "images"
    out_mask_dir = args.out_root / "masks"
    image_paths = sorted(image_dir.glob("*.tif"), key=lambda p: p.name)
    rows: list[dict[str, object]] = []

    if not image_paths:
        raise SystemExit(f"No training images found in {image_dir}")

    for index, image_path in enumerate(image_paths, start=1):
        label_path = label_dir / image_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

        stem = f"{args.stem_prefix}{image_path.stem}"
        out_image_path = out_image_dir / f"{stem}.png"
        out_mask_path = out_mask_dir / f"{stem}.png"
        if args.overwrite or not out_image_path.exists():
            image = to_uint8_image(read_tiff(image_path))
            save_png(out_image_path, image)
        label = read_tiff(label_path)
        mask = (label >= args.label_threshold).astype(np.uint8) * 255
        if args.overwrite or not out_mask_path.exists():
            save_png(out_mask_path, mask)

        water_pixels = int((mask > 0).sum())
        total_pixels = int(mask.size)
        rows.append(
            {
                "stem": stem,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "water_pixels": water_pixels,
                "total_pixels": total_pixels,
                "water_ratio": water_pixels / max(total_pixels, 1),
            }
        )
        if index % 200 == 0:
            print(f"Converted labeled samples: {index}/{len(image_paths)}")
    return rows


def convert_unlabeled_test(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.convert_unlabeled_test:
        return []
    image_dir = args.raw_root / "test" / args.image_source
    mndwi_dir = args.raw_root / "test" / "mndwi"
    out_root = args.out_root.parent / "gf6_unlabeled_test"
    out_image_dir = out_root / "images"
    out_mndwi_dir = out_root / "mndwi_masks"
    rows: list[dict[str, object]] = []
    image_paths = sorted(image_dir.glob("*.tif"), key=lambda p: p.name)
    for index, image_path in enumerate(image_paths, start=1):
        stem = f"{args.stem_prefix}test_{image_path.stem}"
        out_image_path = out_image_dir / f"{stem}.png"
        if args.overwrite or not out_image_path.exists():
            save_png(out_image_path, to_uint8_image(read_tiff(image_path)))

        mndwi_path = mndwi_dir / image_path.name
        water_ratio = ""
        if mndwi_path.exists():
            mndwi = read_tiff(mndwi_path)
            mndwi_mask = (mndwi >= args.label_threshold).astype(np.uint8) * 255
            if args.overwrite or not (out_mndwi_dir / f"{stem}.png").exists():
                save_png(out_mndwi_dir / f"{stem}.png", mndwi_mask)
            water_ratio = float((mndwi_mask > 0).mean())

        rows.append({"stem": stem, "source_image": str(image_path), "source_mndwi": str(mndwi_path), "mndwi_water_ratio": water_ratio})
        if index % 100 == 0:
            print(f"Converted unlabeled test samples: {index}/{len(image_paths)}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_image_stats(image_dir: Path) -> dict[str, object]:
    paths = sorted(image_dir.glob("*.png"))
    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0
    for path in paths:
        arr = np.asarray(Image.open(path)).astype(np.float64) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
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
    std = np.sqrt(var)
    return {"num_images": len(paths), "mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_root)
    rows = convert_labeled(args)
    write_manifest(args.out_root / "source_manifest.csv", rows)

    train, val, test = stratified_split(rows, args.val_ratio, args.test_ratio, args.seed)
    write_split(args.out_root / "splits" / "train.csv", train)
    write_split(args.out_root / "splits" / "val.csv", val)
    write_split(args.out_root / "splits" / "test.csv", test)

    stats = compute_image_stats(args.out_root / "images")
    summary = {
        "raw_root": str(args.raw_root),
        "out_root": str(args.out_root),
        "image_source": args.image_source,
        "label_threshold": args.label_threshold,
        "num_labeled_samples": len(rows),
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "image_stats_0_1": stats,
    }
    (args.out_root / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    unlabeled_rows = convert_unlabeled_test(args)
    if unlabeled_rows:
        unlabeled_root = args.out_root.parent / "gf6_unlabeled_test"
        write_manifest(unlabeled_root / "source_manifest.csv", unlabeled_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
