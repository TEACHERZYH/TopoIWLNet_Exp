"""Image and mask IO utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def list_files(folder: str | Path, exts: set[str] | None = None) -> list[Path]:
    folder = Path(folder)
    exts = exts or IMAGE_EXTS
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        arr = np.asarray(tifffile.imread(path))
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[-1] > 16:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def read_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        arr = np.asarray(tifffile.imread(path))
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    if max_value > 1.0:
        arr = arr / max_value
    return (arr >= threshold).astype(np.uint8)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask.astype(np.float32) > 0.5).astype(np.uint8) * 255
    Image.fromarray(out).save(path)


def save_prob(path: str | Path, prob: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(prob.astype(np.float32), 0.0, 1.0)
    Image.fromarray((out * 255).astype(np.uint8)).save(path)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    original_dtype = arr.dtype
    arr = arr.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    if max_value > 1.5:
        if np.issubdtype(original_dtype, np.integer):
            arr = arr / float(np.iinfo(original_dtype).max)
        else:
            arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr
