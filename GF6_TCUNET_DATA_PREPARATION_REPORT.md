# GF6_TCUNet Data Preparation Report

## 原始数据位置

```text
F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\raw
```

原始结构：

```text
raw/
  train/
    images/    # 8-band GF-6 TIFF
    img_345/   # 3-channel band-composite TIFF
    labels/    # water/land labels
  test/
    images/    # 8-band GF-6 TIFF
    img_345/   # 3-channel band-composite TIFF
    mndwi/     # binary MNDWI reference/pseudo mask
```

## 关键检查结果

- `train/images`: 1886 files
- `train/img_345`: 1886 files
- `train/labels`: 1886 files
- `test/images`: 215 files
- `test/img_345`: 215 files
- `test/mndwi`: 215 files

影像尺寸与编码：

- `images/*.tif`: `(512, 512, 8)`, `uint8`
- `img_345/*.tif`: `(512, 512, 3)`, `uint8`
- `labels/*.tif`: `(512, 512)`, `uint8`

标签阈值：

- `label >= 128`: water
- `label < 128`: non-water

说明：标签接近二值，但边缘存在少量 1-7 和 248-254 的过渡值，因此采用 128 阈值二值化。

## 统一格式输出

监督训练数据：

```text
F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format
```

统一格式：

```text
topoiwl_format/
  images/
  masks/
  boundary/
  buffer/
  skeleton/
  distance_npy/
  distance_png/
  splits/
  source_manifest.csv
  label_manifest.csv
  dataset_stats.json
```

无标签原始 test 预测数据：

```text
F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\gf6_unlabeled_test
```

## 划分结果

使用 `train` 中 1886 个带人工标签样本重新划分：

```text
train: 1318
val:   284
test:  284
```

原始 `test` 目录没有人工 label，不用于主表定量评估；可用于无标签预测、可视化展示或与 MNDWI 伪参考做补充分析。

## 图像统计

基于 `img_345` 转换后的 PNG，按 `[0, 1]` 归一化后统计：

```json
{
  "mean": [0.13079419699530181, 0.20402835376150294, 0.12039565361876671],
  "std": [0.07995418927406385, 0.15305796551761253, 0.07822307254291326]
}
```

从零训练配置已使用上述 GF6 数据集统计；MobileNetV3 ImageNet 预训练配置保留 ImageNet mean/std。

## 已执行命令

转换：

```powershell
conda run -n dl_env python scripts\convert_gf6_tcunet.py `
  --raw-root "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\raw" `
  --out-root "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format" `
  --image-source img_345 `
  --label-threshold 128 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --seed 42 `
  --stem-prefix gf6_ `
  --overwrite `
  --convert-unlabeled-test
```

辅助标签生成：

```powershell
conda run -n dl_env python scripts\generate_waterline_labels.py `
  --mask-dir "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format\masks" `
  --out-dir "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format" `
  --boundary-width 1 `
  --buffer-width 3 `
  --distance-trunc 20
```

结构检查：

```text
images: 1886 files
masks: 1886 files
boundary: 1886 files
distance_npy: 1886 files
matched samples: 1886
```

## 已完成真实数据调试

轻量随机初始化模型：

```text
config: configs/topoiwl_gf6_debug.yaml
checkpoint: experiments/gf6_debug/best.pt
train_loss: 3.4204
val_loss: 3.3498
```

MobileNetV3-Large ImageNet 预训练模型：

```text
config: configs/topoiwl_gf6_mobilenetv3_pretrained_debug.yaml
checkpoint: experiments/gf6_mobilenetv3_pretrained_debug/best.pt
train_loss: 3.4699
val_loss: 3.0425
```

说明：以上只是少量 batch 的链路调试结果，不作为论文结果。

## 预览图

转换预览：

```text
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\outputs\gf6_conversion_preview.png
```

无标签 test 调试预测：

```text
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\outputs\gf6_unlabeled_debug_predictions
```

## 下一步正式实验

建议先跑两个主实验：

1. 从零训练轻量模型：

```powershell
conda run -n dl_env python scripts\train.py --config configs\topoiwl_default.yaml
```

2. MobileNetV3-Large ImageNet 预训练模型：

```powershell
conda run -n dl_env python scripts\train.py --config configs\topoiwl_mobilenetv3_pretrained.yaml
```

训练完成后分别运行：

```powershell
conda run -n dl_env python scripts\evaluate.py `
  --config configs\topoiwl_default.yaml `
  --checkpoint experiments\topoiwl_default\best.pt `
  --split test

conda run -n dl_env python scripts\evaluate.py `
  --config configs\topoiwl_mobilenetv3_pretrained.yaml `
  --checkpoint experiments\topoiwl_mobilenetv3_pretrained\best.pt `
  --split test
```
