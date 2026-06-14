#!/usr/bin/env python
"""Create a tiny synthetic water-land dataset for pipeline smoke tests.

The generated samples are not intended for reporting paper results. They only
exercise data loading, label generation, training, and evaluation code before
real manually downloaded datasets are converted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_sample(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    phase = rng.uniform(0, 2 * np.pi)
    slope = rng.uniform(-0.22, 0.22)
    base = rng.uniform(0.35 * size, 0.65 * size)
    amplitude = rng.uniform(4.0, 12.0)
    frequency = rng.uniform(1.0, 2.7)
    boundary = base + slope * (xx - size / 2) + amplitude * np.sin((xx / size) * 2 * np.pi * frequency + phase)
    boundary += rng.normal(0.0, 1.0, size=(size, size))
    water_below = rng.random() > 0.5
    mask = yy >= boundary if water_below else yy <= boundary

    land_color = np.array([0.46, 0.41, 0.29]) + rng.normal(0, 0.03, size=3)
    water_color = np.array([0.12, 0.34, 0.48]) + rng.normal(0, 0.03, size=3)
    image = np.where(mask[..., None], water_color, land_color)

    texture = rng.normal(0, 0.035, size=(size, size, 1))
    illumination = 0.08 * np.sin(xx[..., None] / size * np.pi + rng.uniform(0, np.pi))
    image = image + texture + illumination

    for _ in range(rng.integers(1, 4)):
        cx = rng.uniform(0, size)
        cy = rng.uniform(0, size)
        radius = rng.uniform(size * 0.05, size * 0.16)
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
        delta = rng.uniform(-0.06, 0.08)
        image[blob] += delta

    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8), mask.astype(np.uint8) * 255


def main() -> None:
    args = parse_args()
    image_dir = args.root / "images"
    mask_dir = args.root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for index in range(args.num_samples):
        image, mask = make_sample(rng, args.size)
        stem = f"synthetic_{index:04d}"
        Image.fromarray(image).save(image_dir / f"{stem}.png")
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")
    print(f"Wrote {args.num_samples} synthetic samples to {args.root}")


if __name__ == "__main__":
    main()
