#!/usr/bin/env python
"""Train a mask-only segmentation baseline."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.losses import dice_loss_with_logits  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def mask_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    if pos_weight == 1.0:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    else:
        weight = torch.where(target > 0.5, torch.full_like(target, pos_weight), torch.ones_like(target))
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    return bce + dice_loss_with_logits(logits, target)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, pos_weight: float, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        pred = model(batch["image"])
        loss = mask_loss(pred["mask"], batch["mask"], pos_weight)
        losses.append(float(loss.detach().cpu()))
        if max_batches and batch_index >= max_batches:
            break
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")

    ds_cfg = cfg["dataset"]
    train_ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg["train_split"],
        augment=True,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    val_ds = WaterlineDataset(
        ds_cfg["root"],
        ds_cfg["val_split"],
        augment=False,
        input_channels=cfg["model"]["in_channels"],
        image_mean=ds_cfg.get("image_mean"),
        image_std=ds_cfg.get("image_std"),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_memory,
    )

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    amp_enabled = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=float(cfg["train"].get("amp_init_scale", 1024.0)))
    max_train_batches = cfg["train"].get("max_train_batches")
    max_val_batches = cfg["train"].get("max_val_batches")
    pos_weight = float(cfg.get("loss", {}).get("mask_pos_weight", 1.0))

    exp_dir = ensure_dir(cfg["output"]["exp_dir"])
    best_path = exp_dir / "best.pt"
    last_path = exp_dir / "last.pt"
    log_path = exp_dir / "train_log.csv"
    done_path = exp_dir / "training_complete.txt"
    best_val = float("inf")

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            model.train()
            epoch_losses = []
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg['train']['epochs']}",
                disable=not sys.stderr.isatty(),
                leave=False,
            )
            for batch_index, batch in enumerate(progress, start=1):
                batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred = model(batch["image"])
                    loss = mask_loss(pred["mask"], batch["mask"], pos_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_value = float(loss.detach().cpu())
                epoch_losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.4f}")
                if max_train_batches and batch_index >= max_train_batches:
                    break
            scheduler.step()
            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_loss = validate(model, val_loader, device, pos_weight, max_batches=max_val_batches)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, last_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}, best_path)
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val:.4f}")
    done_path.write_text(f"epochs={cfg['train']['epochs']}\nbest_val={best_val}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
