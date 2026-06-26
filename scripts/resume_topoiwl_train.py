#!/usr/bin/env python
"""Resume TopoIWL-Net training from a checkpoint that stores model weights and epoch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.data.dataset import WaterlineDataset  # noqa: E402
from topoiwl.losses import TopoIWLLoss  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402
from topoiwl.utils.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, criterion: TopoIWLLoss, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model(batch["image"])
        loss, _ = criterion(pred, batch)
        losses.append(float(loss.detach().cpu()))
        if max_batches and batch_index >= max_batches:
            break
    model.train()
    return sum(losses) / max(len(losses), 1)


def previous_best(log_path: Path, best_path: Path) -> float:
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        if "val_loss" in ckpt:
            return float(ckpt["val_loss"])
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return min(float(row["val_loss"]) for row in rows)
    return float("inf")


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
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=pin_memory)

    model = build_model(cfg["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    total_epochs = int(cfg["train"]["epochs"])
    if start_epoch > total_epochs:
        print(f"checkpoint already at epoch {start_epoch - 1}; nothing to resume")
        return

    criterion = TopoIWLLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    for _ in range(start_epoch - 1):
        scheduler.step()
    amp_enabled = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=float(cfg["train"].get("amp_init_scale", 1024.0)))
    max_train_batches = cfg["train"].get("max_train_batches")
    max_val_batches = cfg["train"].get("max_val_batches")

    exp_dir = ensure_dir(cfg["output"]["exp_dir"])
    best_path = exp_dir / "best.pt"
    last_path = exp_dir / "last.pt"
    log_path = exp_dir / "train_log.csv"
    best_val = previous_best(log_path, best_path)

    mode = "a" if log_path.exists() else "w"
    with log_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            epoch_losses = []
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
            for batch_index, batch in enumerate(progress, start=1):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred = model(batch["image"])
                    loss, logs = criterion(pred, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(logs["loss"])
                progress.set_postfix(loss=f"{logs['loss']:.4f}")
                if max_train_batches and batch_index >= max_train_batches:
                    break
            scheduler.step()
            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_loss = validate(model, val_loader, criterion, device, max_batches=max_val_batches)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
            writer.writerow(row)
            handle.flush()
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, last_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_loss": val_loss}, best_path)
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best={best_val:.4f}")


if __name__ == "__main__":
    main()
