#!/usr/bin/env python
"""Run prediction on an unlabeled image folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.io import list_files, normalize_image, read_image, save_prob  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def channel_values(values: list[float] | None, channels: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.size > channels:
        arr = arr[:channels]
    elif arr.size < channels:
        arr = np.pad(arr, (0, channels - arr.size), mode="edge")
    return arr.reshape(1, 1, channels)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    channels = int(cfg["model"].get("in_channels", 3))
    mean = channel_values(cfg.get("dataset", {}).get("image_mean"), channels)
    std = channel_values(cfg.get("dataset", {}).get("image_std"), channels)
    out_dir = ensure_dir(args.out_dir)

    paths = list_files(args.image_dir)
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            image = normalize_image(read_image(path))
            if image.shape[-1] > channels:
                image = image[..., :channels]
            elif image.shape[-1] < channels:
                image = np.concatenate([image, np.repeat(image[..., -1:], channels - image.shape[-1], axis=-1)], axis=-1)
            if mean is not None and std is not None:
                image = (image - mean) / np.maximum(std, 1e-6)
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            pred = model(tensor)
            save_prob(out_dir / "mask_prob" / f"{path.stem}.png", torch.sigmoid(pred["mask"])[0, 0].cpu().numpy())
            save_prob(out_dir / "boundary_prob" / f"{path.stem}.png", torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy())
            if index % 100 == 0:
                print(f"Predicted {index}/{len(paths)}")
            if args.max_images and index >= args.max_images:
                break
    print(f"Wrote predictions to: {out_dir}")


if __name__ == "__main__":
    main()
