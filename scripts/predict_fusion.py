#!/usr/bin/env python
"""Save final waterline predictions with validation-selected fusion parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.io import save_prob  # noqa: E402
from topoiwl.utils.morphology import binary_dilation, binary_erosion, label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--final-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mask-boundary-width", type=int, default=1)
    parser.add_argument("--mask-buffer-iters", type=int, default=0)
    parser.add_argument("--gap-bridge-iters", type=int, default=0)
    parser.add_argument("--min-component-size", type=int, default=0)
    return parser.parse_args()


def build_dataset_config(cfg: dict, dataset_root: Path | None) -> dict:
    ds_cfg = dict(cfg["dataset"])
    if dataset_root is not None:
        root = Path(dataset_root)
        ds_cfg["root"] = str(root)
        ds_cfg["train_split"] = str(root / "splits" / "train.csv")
        ds_cfg["val_split"] = str(root / "splits" / "val.csv")
        ds_cfg["test_split"] = str(root / "splits" / "test.csv")
    return ds_cfg


def mask_to_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    dilated = binary_dilation(mask, iterations=width)
    eroded = binary_erosion(mask, iterations=width)
    return np.logical_and(dilated, ~eroded)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool)
    labels, n_labels = label(mask)
    if n_labels == 0:
        return mask.astype(bool)
    out = np.zeros(mask.shape, dtype=bool)
    for idx in range(1, n_labels + 1):
        comp = labels == idx
        if int(comp.sum()) >= min_size:
            out |= comp
    return out


def bridge_gaps(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask.astype(bool)
    bridged = binary_dilation(mask.astype(bool), iterations=iters)
    bridged = binary_erosion(bridged, iterations=iters)
    return bridged.astype(bool)


def fuse_boundary(
    mask_prob: np.ndarray,
    boundary_prob: np.ndarray,
    mask_threshold: float,
    final_threshold: float,
    alpha: float,
    mask_boundary_width: int,
    mask_buffer_iters: int,
    gap_bridge_iters: int,
    min_component_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_pred = mask_prob >= mask_threshold
    mask_boundary = mask_to_boundary(mask_pred, width=mask_boundary_width)
    final_prob = alpha * boundary_prob + (1.0 - alpha) * mask_boundary.astype(np.float32)
    final_pred = final_prob >= final_threshold
    if mask_buffer_iters > 0:
        gate = binary_dilation(mask_boundary, iterations=mask_buffer_iters)
        final_pred = np.logical_and(final_pred, gate)
    final_pred = bridge_gaps(final_pred, gap_bridge_iters)
    final_pred = remove_small_components(final_pred, min_component_size)
    return mask_pred, final_prob, final_pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    mask_threshold = float(args.mask_threshold) if args.mask_threshold is not None else float(cfg["eval"]["mask_threshold"])
    ds_cfg = build_dataset_config(cfg, args.dataset_root)
    split_key = f"{args.split}_split"
    ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg[split_key],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")

    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    model_cfg["pretrained_weights"] = None
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = ensure_dir(args.out_dir)
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            pred = model(batch["image"].to(device))
            stem = batch["stem"][0]
            mask_prob = torch.sigmoid(pred["mask"])[0, 0].cpu().numpy()
            boundary_prob = torch.sigmoid(pred["boundary"])[0, 0].cpu().numpy()
            mask_pred, final_prob, final_pred = fuse_boundary(
                mask_prob=mask_prob,
                boundary_prob=boundary_prob,
                mask_threshold=mask_threshold,
                final_threshold=args.final_threshold,
                alpha=args.alpha,
                mask_boundary_width=args.mask_boundary_width,
                mask_buffer_iters=args.mask_buffer_iters,
                gap_bridge_iters=args.gap_bridge_iters,
                min_component_size=args.min_component_size,
            )
            save_prob(out_dir / "mask_prob" / f"{stem}.png", mask_prob)
            save_prob(out_dir / "boundary_head_prob" / f"{stem}.png", boundary_prob)
            save_prob(out_dir / "fusion_prob" / f"{stem}.png", final_prob)
            save_prob(out_dir / "mask_pred" / f"{stem}.png", mask_pred.astype(np.float32))
            save_prob(out_dir / "boundary_prob" / f"{stem}.png", final_pred.astype(np.float32))
            if args.max_samples is not None and sample_index >= args.max_samples:
                break
    print(f"Wrote fusion predictions to: {out_dir}")


if __name__ == "__main__":
    main()
