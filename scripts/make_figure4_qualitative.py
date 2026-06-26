#!/usr/bin/env python
"""Create the manuscript qualitative comparison figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "GF6": {
        "root": PROJECT_ROOT / "data/GF6_TCUNet/processed/topoiwl_format",
        "topoiwl": PROJECT_ROOT / "outputs/remote_gf6_ablate_full80_test_predictions",
        "mobilenet": PROJECT_ROOT / "outputs/baseline_remote_gf6_mobilenetv3_unet80_test_predictions",
        "resnet": PROJECT_ROOT / "outputs/baseline_remote_gf6_deeplabv3_resnet50_80_test_predictions",
        "role": "success",
    },
    "SeaLand": {
        "root": Path("/data/zyh/datasets/SeaLand_Coastline_2025/processed/topoiwl_format"),
        "topoiwl": PROJECT_ROOT / "outputs/remote_sealand_full80_test_predictions",
        "mobilenet": PROJECT_ROOT / "outputs/baseline_remote_sealand_mobilenetv3_unet80_test_predictions",
        "resnet": PROJECT_ROOT / "outputs/baseline_remote_sealand_deeplabv3_resnet50_80_test_predictions",
        "role": "success",
    },
    "GLH": {
        "root": Path("/data/zyh/datasets/GLH-Water/processed/topoiwl_format"),
        "topoiwl": PROJECT_ROOT / "outputs/remote_glh_boundary_metric_fusion_min64_test_predictions",
        "mobilenet": PROJECT_ROOT / "outputs/baseline_remote_glh_mobilenetv3_unet80_test_predictions",
        "resnet": PROJECT_ROOT / "outputs/baseline_remote_glh_deeplabv3_resnet50_80_test_predictions",
        "role": "success",
    },
}


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr


def read_binary(path: Path, threshold: float = 0.5) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr >= threshold


def find_existing(folder: Path, stem: str, exts: tuple[str, ...]) -> Path:
    for ext in exts:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {stem} in {folder}")


def pred_path(pred_dir: Path, stem: str) -> Path:
    return pred_dir / "boundary_prob" / f"{stem}.png"


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def available_stems(info: dict[str, object]) -> list[str]:
    root = info["root"]
    assert isinstance(root, Path)
    pred_dirs = [info["topoiwl"], info["mobilenet"], info["resnet"]]
    stems = None
    for pred_dir in pred_dirs:
        assert isinstance(pred_dir, Path)
        current = {p.stem for p in (pred_dir / "boundary_prob").glob("*.png")}
        stems = current if stems is None else stems & current
    assert stems is not None
    image_stems = {p.stem for p in (root / "images").iterdir() if p.is_file()}
    boundary_stems = {p.stem for p in (root / "boundary").iterdir() if p.is_file()}
    return sorted(stems & image_stems & boundary_stems)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    gt_buffer = dilate(gt, radius=tolerance)
    pred_buffer = dilate(pred, radius=tolerance)
    precision = (pred & gt_buffer).sum() / max(pred.sum(), 1)
    recall = (gt & pred_buffer).sum() / max(gt.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def select_sample(name: str, info: dict[str, object]) -> tuple[str, dict[str, float]]:
    root = info["root"]
    assert isinstance(root, Path)
    candidates = []
    for stem in available_stems(info):
        gt = read_binary(find_existing(root / "boundary", stem, (".png", ".tif", ".tiff", ".npy")))
        if gt.sum() < 50:
            continue
        scores = {}
        for method in ("mobilenet", "resnet", "topoiwl"):
            pred_dir = info[method]
            assert isinstance(pred_dir, Path)
            scores[method] = boundary_f1(read_binary(pred_path(pred_dir, stem)), gt)
        baseline_best = max(scores["mobilenet"], scores["resnet"])
        margin = scores["topoiwl"] - baseline_best
        if info["role"] == "stress":
            rank_score = -margin
        else:
            rank_score = margin
        candidates.append((rank_score, stem, scores))
    if not candidates:
        raise RuntimeError(f"No candidates found for {name}")
    candidates.sort(reverse=True)
    _, stem, scores = candidates[0]
    return stem, scores


def overlay(base: np.ndarray, gt: np.ndarray | None = None, pred: np.ndarray | None = None) -> np.ndarray:
    out = base.copy()
    if gt is not None:
        gt_buf = dilate(gt, radius=1)
        out[gt_buf] = np.array([0.0, 0.95, 0.25])
    if pred is not None:
        pred_buf = dilate(pred, radius=1)
        overlap = pred_buf & (dilate(gt, radius=1) if gt is not None else False)
        out[pred_buf] = np.array([1.0, 0.0, 0.75])
        out[overlap] = np.array([1.0, 1.0, 1.0])
    return out


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_panel(panel: np.ndarray, size: tuple[int, int]) -> Image.Image:
    arr = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int, int] = (0, 0, 0, 150),
) -> None:
    x, y = xy
    padding = 7
    w, h = text_size(draw, text, font)
    draw.rounded_rectangle((x, y, x + w + padding * 2, y + h + padding * 2), radius=5, fill=bg)
    draw.text((x + padding, y + padding - 1), text, font=font, fill=fill)


def wrap_id(stem: str, width: int = 18) -> list[str]:
    if len(stem) <= width:
        return [f"ID: {stem}"]
    parts = [stem[i : i + width] for i in range(0, len(stem), width)]
    return ["ID: " + parts[0], *parts[1:]]


def build_figure() -> None:
    selected = []
    for name, info in DATASETS.items():
        stem, scores = select_sample(name, info)
        selected.append((name, stem, scores, info))

    out_dir = PROJECT_ROOT / "outputs/figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "RGB image",
        "GT waterline",
        "MobileNetV3-UNet",
        "DeepLabV3-ResNet50",
        "TopoIWL-Net",
    ]
    panel_size = (512, 512)
    gap = 10
    left_margin = 230
    right_margin = 22
    top_margin = 68
    bottom_margin = 50
    width = left_margin + len(columns) * panel_size[0] + (len(columns) - 1) * gap + right_margin
    height = top_margin + len(selected) * panel_size[1] + (len(selected) - 1) * gap + bottom_margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = load_font(27, bold=True)
    row_font = load_font(24, bold=True)
    small_font = load_font(18)
    legend_font = load_font(22)

    for col, title in enumerate(columns):
        x = left_margin + col * (panel_size[0] + gap)
        tw, _ = text_size(draw, title, title_font)
        draw.text((x + (panel_size[0] - tw) / 2, 22), title, font=title_font, fill=(20, 20, 20))

    for row, (name, stem, scores, info) in enumerate(selected):
        root = info["root"]
        assert isinstance(root, Path)
        image = read_image(find_existing(root / "images", stem, (".png", ".jpg", ".jpeg", ".tif", ".tiff")))
        gt = read_binary(find_existing(root / "boundary", stem, (".png", ".tif", ".tiff", ".npy")))
        preds = {
            "mobilenet": read_binary(pred_path(info["mobilenet"], stem)),
            "resnet": read_binary(pred_path(info["resnet"], stem)),
            "topoiwl": read_binary(pred_path(info["topoiwl"], stem)),
        }
        panels = [
            image,
            overlay(image, gt=gt),
            overlay(image, gt=gt, pred=preds["mobilenet"]),
            overlay(image, gt=gt, pred=preds["resnet"]),
            overlay(image, gt=gt, pred=preds["topoiwl"]),
        ]
        row_lines = [f"({chr(97 + row)}) {name}", *wrap_id(stem)]
        y = top_margin + row * (panel_size[1] + gap)
        for idx, line in enumerate(row_lines):
            draw.text((18, y + 30 + idx * 34), line, font=row_font if idx == 0 else small_font, fill=(20, 20, 20))

        for col, panel in enumerate(panels):
            x = left_margin + col * (panel_size[0] + gap)
            canvas.paste(fit_panel(panel, panel_size), (x, y))
            draw.rectangle((x, y, x + panel_size[0] - 1, y + panel_size[1] - 1), outline=(65, 65, 65), width=2)

        score_text = (
            f"BF1@3: MBV3 {scores['mobilenet']:.2f}, "
            f"R50 {scores['resnet']:.2f}, Topo {scores['topoiwl']:.2f}"
        )
        x = left_margin + (len(columns) - 1) * (panel_size[0] + gap) + 12
        draw_label_box(draw, (x, y + 14), score_text, small_font)

    draw.text(
        (left_margin, height - 37),
        "Overlay colors: green = reference waterline, magenta = predicted waterline, white = overlap.",
        font=legend_font,
        fill=(20, 20, 20),
    )
    png = out_dir / "figure4_qualitative_comparison.png"
    pdf = out_dir / "figure4_qualitative_comparison.pdf"
    canvas.save(png, dpi=(450, 450))
    canvas.save(pdf, "PDF", resolution=450.0)
    print("Selected samples:")
    for name, stem, scores, _ in selected:
        print(name, stem, scores)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    build_figure()
