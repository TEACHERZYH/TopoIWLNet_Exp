# TopoIWLNet_Exp

TopoIWL-Net 实验工程，用于论文 **TopoIWL-Net: A Lightweight Geometry-Topology Preserving Network for Instantaneous Waterline Extraction from Remote Sensing Images**。

## 重要规则

数据集由用户手动下载。Codex 不自动下载数据集文件。

本工程只提供：

- 数据结构检查；
- 标签生成；
- 数据划分；
- 模型训练；
- 评估；
- 预测输出；
- 后续数据转换脚本的放置位置。

## 工程结构

```text
TopoIWLNet_Exp/
  configs/
    topoiwl_default.yaml
    topoiwl_mobilenetv3_pretrained.yaml
    topoiwl_mobilenetv3_smoke.yaml
    topoiwl_smoke.yaml
  scripts/
    make_synthetic_dataset.py
    check_dataset.py
    create_splits.py
    generate_waterline_labels.py
    train.py
    evaluate.py
    predict.py
  src/topoiwl/
    data/
    models/
    utils/
    config.py
    losses.py
    metrics.py
  experiments/
  outputs/
  logs/
```

## 统一数据格式

训练代码期望数据整理为：

```text
topoiwl_format/
  images/
    sample_000001.png
  masks/
    sample_000001.png
  boundary/
    sample_000001.png
  buffer/
    sample_000001.png
  skeleton/
    sample_000001.png
  distance_npy/
    sample_000001.npy
  splits/
    train.csv
    val.csv
    test.csv
```

其中 `images`、`masks`、`boundary`、`distance_npy` 是训练必须项。

## 环境安装

建议在项目级环境中安装：

```powershell
cd "F:\2026\Remote Sensing_codex\TopoIWLNet_Exp"
pip install -r requirements.txt
```

如果 PyTorch 需要 CUDA 版本，后续根据本机 CUDA 情况安装对应版本。

当前本机 `dl_env` 环境已经可以运行本工程。若 `scipy` / `scikit-image` 暂时安装失败，代码会自动使用 NumPy 后备形态学实现；正式大规模生成标签时建议再补装这两个包以提升速度。

## 预训练模型

本工程支持从预训练编码器开始训练，不必完全从零训练。首选配置：

```powershell
conda run -n dl_env python scripts\train.py --config configs\topoiwl_mobilenetv3_pretrained.yaml
```

该配置使用 `torchvision` 的 MobileNetV3-Large ImageNet 预训练编码器，并保留 TopoIWL-Net 的多任务解码头。详细筛选依据见：

```text
PRETRAINED_MODEL_SELECTION.md
```

## 烟雾测试

在没有真实数据时，可以用合成小样本检查整条流水线：

```powershell
conda run -n dl_env python scripts\make_synthetic_dataset.py `
  --root outputs\smoke_dataset `
  --num-samples 8 `
  --size 96

conda run -n dl_env python scripts\generate_waterline_labels.py `
  --mask-dir outputs\smoke_dataset\masks `
  --out-dir outputs\smoke_dataset `
  --boundary-width 1 `
  --buffer-width 3 `
  --distance-trunc 20

conda run -n dl_env python scripts\create_splits.py `
  --root outputs\smoke_dataset `
  --out-dir outputs\smoke_dataset\splits `
  --train-ratio 0.5 `
  --val-ratio 0.25

conda run -n dl_env python scripts\train.py --config configs\topoiwl_smoke.yaml

conda run -n dl_env python scripts\evaluate.py `
  --config configs\topoiwl_smoke.yaml `
  --checkpoint experiments\smoke\best.pt `
  --split test
```

合成数据只用于代码检查，不能作为论文实验结果。

## 典型流程

### 1. 用户手动下载数据

把数据放到：

```text
F:\2026\Remote Sensing_codex\datasets\<DATASET_NAME>\raw
```

### 2. 数据转换

下载完成后，根据实际数据结构编写转换脚本，将数据整理到 `processed/topoiwl_format`。

### 3. 生成 waterline 辅助标签

如果已经有 `images/` 和 `masks/`：

```powershell
python scripts\generate_waterline_labels.py `
  --mask-dir "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format\masks" `
  --out-dir "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format" `
  --boundary-width 2 `
  --buffer-width 3 `
  --distance-trunc 20
```

### 4. 检查数据

```powershell
python scripts\check_dataset.py `
  --root "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format"
```

### 5. 划分数据集

```powershell
python scripts\create_splits.py `
  --root "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format" `
  --out-dir "F:\2026\Remote Sensing_codex\datasets\GF6_TCUNet\processed\topoiwl_format\splits"
```

### 6. 训练

```powershell
python scripts\train.py --config configs\topoiwl_default.yaml
```

### 7. 评估

```powershell
python scripts\evaluate.py `
  --config configs\topoiwl_default.yaml `
  --checkpoint experiments\topoiwl_default\best.pt `
  --split test
```

### 8. 预测

```powershell
python scripts\predict.py `
  --config configs\topoiwl_default.yaml `
  --checkpoint experiments\topoiwl_default\best.pt `
  --split test
```

## 当前状态

本工程已经完成代码框架，并通过合成小样本烟雾测试。GF6_TCUNet 数据集已经转换为统一格式，且真实数据短训练、短评估、无标签预测链路均已跑通。转换与检查详情见：

```text
GF6_TCUNET_DATA_PREPARATION_REPORT.md
```

最新中等长度探测训练结果见：

```text
GF6_PROBE20_REPORT.md
```

## 论文实验可复现记录

Remote Sensing 投稿稿件使用的远程训练路径、数据集根目录、核心配置、baseline/TopoIWL/PVTv2/GLH optimized fusion 结果文件、预测输出目录和标准评估命令见：

```text
REPRODUCIBILITY_REMOTE_EXPERIMENTS.md
```
