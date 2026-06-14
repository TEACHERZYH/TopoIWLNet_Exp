# Dataset Conversion Guide

本文件说明不同来源数据下载后如何转换为 TopoIWL-Net 统一格式。

## 1. 统一格式

```text
topoiwl_format/
  images/
  masks/
  boundary/
  buffer/
  skeleton/
  distance_npy/
  splits/
```

要求：

- `images` 和 `masks` 文件名 stem 必须一致。
- `masks` 中水体建议为 255，陆地为 0；如果相反，转换脚本需要反转。
- `boundary`、`buffer`、`skeleton`、`distance_npy` 可由 `scripts/generate_waterline_labels.py` 自动生成。

## 2. GF-6 / TCUNet

手动下载目录：

```text
F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\raw
```

转换目标：

```text
F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format
```

后续需要检查：

- 已完成检查和转换。
- 原始训练集包含 1886 个带标签样本。
- 原始测试集包含 215 个无人工标签样本，可用于预测展示。
- 主实验使用 `train/img_345` 三通道影像和 `train/labels` 标签。
- 标签阈值为 `>=128` water，`<128` non-water。
- 转换详情见 `GF6_TCUNET_DATA_PREPARATION_REPORT.md`。

## 3. SeaLand_Coastline_2025

手动下载目录：

```text
F:\2026\Remote Sensing_codex\datasets\SeaLand_Coastline_2025\raw
```

转换目标：

```text
F:\2026\Remote Sensing_codex\datasets\SeaLand_Coastline_2025\processed\topoiwl_format
```

后续需要检查：

- 是否包含海陆分割标签；
- 是否包含海岸线边缘标签；
- 边缘标签能否直接作为 boundary；
- 标签是否需要形态学修正。

## 4. SWED

手动下载目录：

```text
F:\2026\Remote Sensing_codex\datasets\SWED\raw
```

转换目标：

```text
F:\2026\Remote Sensing_codex\datasets\SWED\processed\topoiwl_format
```

后续需要检查：

- sample/full 解压结构；
- Sentinel-2 波段文件组织；
- water/non-water mask 文件名；
- 是否需要选择 RGB/NIR/SWIR 波段。

## 5. GLH-Water

手动下载目录：

```text
F:\2026\Remote Sensing_codex\datasets\GLH-Water\raw
```

转换目标：

```text
F:\2026\Remote Sensing_codex\datasets\GLH-Water\processed\topoiwl_format
```

后续需要检查：

- 图像与标签配对方式；
- 标签编码；
- 数据场景类型；
- 是否适合加入主实验或仅作泛化实验。
