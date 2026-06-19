"""Dataset implementation for instantaneous waterline extraction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from topoiwl.data.transforms import random_brightness_contrast, random_flip_rotate
from topoiwl.utils.io import normalize_image, read_image, read_mask


class WaterlineDataset(Dataset):
    """Dataset expecting aligned image/mask/boundary/distance files with matching stems."""

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        image_dir: str = "images",
        mask_dir: str = "masks",
        boundary_dir: str = "boundary",
        distance_dir: str = "distance_npy",
        augment: bool = False,
        input_channels: int = 3,
        image_mean: list[float] | tuple[float, ...] | None = None,
        image_std: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.boundary_dir = self.root / boundary_dir
        self.distance_dir = self.root / distance_dir
        self.augment = augment
        self.input_channels = input_channels
        self.image_mean = self._channel_values(image_mean) if image_mean is not None else None
        self.image_std = self._channel_values(image_std) if image_std is not None else None

        if split_file:
            self.items = self._read_split(split_file)
        else:
            self.items = self._discover_items()

        if not self.items:
            raise RuntimeError(f"No samples found under {self.root}")

    def _channel_values(self, values: list[float] | tuple[float, ...]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size > self.input_channels:
            arr = arr[: self.input_channels]
        elif arr.size < self.input_channels:
            arr = np.pad(arr, (0, self.input_channels - arr.size), mode="edge")
        return arr.reshape(1, 1, self.input_channels)

    def _read_split(self, split_file: str | Path) -> list[dict[str, Path]]:
        items: list[dict[str, Path]] = []
        with Path(split_file).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stem = row.get("stem") or Path(row["image"]).stem
                items.append(self._paths_from_stem(stem))
        return items

    def _discover_items(self) -> list[dict[str, Path]]:
        stems = sorted(p.stem for p in self.image_dir.iterdir() if p.is_file())
        return [self._paths_from_stem(stem) for stem in stems]

    def _find_with_stem(self, folder: Path, stem: str, exts: tuple[str, ...]) -> Path:
        for ext in exts:
            path = folder / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find {stem} in {folder}")

    def _paths_from_stem(self, stem: str) -> dict[str, Path]:
        return {
            "stem": Path(stem),
            "image": self._find_with_stem(self.image_dir, stem, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")),
            "mask": self._find_with_stem(self.mask_dir, stem, (".png", ".tif", ".tiff", ".npy")),
            "boundary": self._find_with_stem(self.boundary_dir, stem, (".png", ".tif", ".tiff", ".npy")),
            "distance": self._find_with_stem(self.distance_dir, stem, (".npy",)),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = normalize_image(read_image(item["image"]))
        mask = read_mask(item["mask"])[..., None].astype(np.float32)
        boundary = read_mask(item["boundary"])[..., None].astype(np.float32)
        distance = np.load(item["distance"]).astype(np.float32)
        if distance.ndim == 2:
            distance = distance[..., None]

        if image.shape[-1] > self.input_channels:
            image = image[..., : self.input_channels]
        elif image.shape[-1] < self.input_channels:
            pad = np.repeat(image[..., -1:], self.input_channels - image.shape[-1], axis=-1)
            image = np.concatenate([image, pad], axis=-1)

        sample = {"image": image, "mask": mask, "boundary": boundary, "distance": distance}
        if self.augment:
            sample = random_flip_rotate(sample)
            sample["image"] = random_brightness_contrast(sample["image"])
        if self.image_mean is not None and self.image_std is not None:
            sample["image"] = (sample["image"] - self.image_mean) / np.maximum(self.image_std, 1e-6)

        return {
            "stem": item["stem"].name,
            "image": torch.from_numpy(sample["image"].transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(sample["mask"].transpose(2, 0, 1)).float(),
            "boundary": torch.from_numpy(sample["boundary"].transpose(2, 0, 1)).float(),
            "distance": torch.from_numpy(sample["distance"].transpose(2, 0, 1)).float(),
        }
