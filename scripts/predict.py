#!/usr/bin/env python
"""Run prediction and save probability maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.io import save_prob  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--boundary-source", default="head", choices=["head", "mask"])
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--mask-boundary-width", type=int, default=1)
    return parser.parse_args()


def mask_to_boundary(mask, width: int = 1):
    mask = mask.astype(bool)
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return (dilated & ~eroded).astype(float)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mask_threshold = float(args.mask_threshold) if args.mask_threshold is not None else float(cfg["eval"]["mask_threshold"])
    split_key = f"{args.split}_split"
    ds_cfg = cfg["dataset"]
    ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg[split_key],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_dir = ensure_dir(args.out_dir or Path(cfg["output"]["exp_dir"]) / f"pred_{args.split}")
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            pred = model(batch["image"].to(device))
            stem = batch["stem"][0]
            mask_prob = torch.sigmoid(pred["mask"])[0, 0].cpu().numpy()
            if args.boundary_source == "mask":
                boundary_prob = mask_to_boundary(mask_prob >= mask_threshold, width=args.mask_boundary_width)
            else:
                boundary_prob = torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy()
            save_prob(out_dir / "mask_prob" / f"{stem}.png", mask_prob)
            save_prob(out_dir / "boundary_prob" / f"{stem}.png", boundary_prob)
            if args.max_samples and sample_index >= args.max_samples:
                break
    print(f"Wrote predictions to: {out_dir}")


if __name__ == "__main__":
    main()
