#!/usr/bin/env python
"""Measure model size, approximate FLOPs, latency, FPS, and peak memory."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from topoiwl.config import ensure_dir, load_config  # noqa: E402
from topoiwl.models.topoiwl_net import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def conv2d_macs(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    x = inputs[0]
    batch = x.shape[0]
    out_h, out_w = output.shape[-2:]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
    return int(batch * module.out_channels * out_h * out_w * kernel_ops)


def linear_macs(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    x = inputs[0]
    batch = x.shape[0] if x.ndim > 1 else 1
    return int(batch * module.in_features * module.out_features)


def measure_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    handles: list[Any] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += conv2d_macs(module, inputs, output)
        elif isinstance(module, nn.Linear):
            macs += linear_macs(module, inputs, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        _ = model(sample)
    for handle in handles:
        handle.remove()
    return macs


def benchmark(model: nn.Module, sample: torch.Tensor, warmup: int, iters: int, amp: bool) -> tuple[float, float, float]:
    device = sample.device
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iters):
                with torch.amp.autocast("cuda", enabled=amp):
                    _ = model(sample)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_event.elapsed_time(end_event))
            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        else:
            start = time.perf_counter()
            for _ in range(iters):
                _ = model(sample)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            peak_mb = 0.0
    latency_ms = elapsed_ms / max(iters, 1)
    fps = 1000.0 * sample.shape[0] / latency_ms
    return latency_ms, fps, peak_mb


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    model = build_model(cfg["model"]).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
    sample = torch.randn(args.batch_size, int(cfg["model"].get("in_channels", 3)), args.height, args.width, device=device)

    total_params, trainable_params = count_params(model)
    macs = measure_macs(model, sample)
    latency_ms, fps, peak_mb = benchmark(model, sample, args.warmup, args.iters, args.amp)
    row = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "device": str(device),
        "input_shape": f"{args.batch_size}x{int(cfg['model'].get('in_channels', 3))}x{args.height}x{args.width}",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_m": total_params / 1e6,
        "macs_g": macs / 1e9,
        "flops_g_2x_macs": 2.0 * macs / 1e9,
        "latency_ms": latency_ms,
        "fps": fps,
        "peak_memory_mb": peak_mb,
        "amp": bool(args.amp),
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if args.out_csv:
        ensure_dir(args.out_csv.parent)
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
