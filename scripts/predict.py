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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
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
            save_prob(out_dir / "mask_prob" / f"{stem}.png", torch.sigmoid(pred["mask"])[0, 0].cpu().numpy())
            save_prob(out_dir / "boundary_prob" / f"{stem}.png", torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy())
            if args.max_samples and sample_index >= args.max_samples:
                break
    print(f"Wrote predictions to: {out_dir}")


if __name__ == "__main__":
    main()
