# Smoke Test Report

生成时间：2026-06-13

## 目的

在真实数据集尚未下载和转换前，使用合成小样本验证 TopoIWL-Net 工程的基本流水线：

- 合成影像与水陆掩膜生成；
- 水边线、缓冲区、骨架和 signed distance 标签生成；
- train / val / test 划分；
- 数据结构检查；
- 1 epoch 训练；
- checkpoint 保存；
- test split 评估；
- mask / boundary 概率图预测导出。

## 使用环境

- Conda 环境：`dl_env`
- Python 调用方式：`conda run -n dl_env python`
- 备注：当前 `scipy` / `scikit-image` 未成功安装，标签生成使用 NumPy 后备实现。

## 关键命令

```powershell
conda run -n dl_env python scripts\make_synthetic_dataset.py --root outputs\smoke_dataset --num-samples 8 --size 96 --seed 42
conda run -n dl_env python scripts\generate_waterline_labels.py --mask-dir outputs\smoke_dataset\masks --out-dir outputs\smoke_dataset --boundary-width 1 --buffer-width 3 --distance-trunc 20
conda run -n dl_env python scripts\create_splits.py --root outputs\smoke_dataset --out-dir outputs\smoke_dataset\splits --train-ratio 0.5 --val-ratio 0.25 --seed 42
conda run -n dl_env python scripts\check_dataset.py --root outputs\smoke_dataset
conda run -n dl_env python scripts\train.py --config configs\topoiwl_smoke.yaml
conda run -n dl_env python scripts\evaluate.py --config configs\topoiwl_smoke.yaml --checkpoint experiments\smoke\best.pt --split test
conda run -n dl_env python scripts\predict.py --config configs\topoiwl_smoke.yaml --checkpoint experiments\smoke\best.pt --split test --out-dir outputs\smoke_predictions
```

## 输出结果

数据检查：

```text
images: 8 files
masks: 8 files
boundary: 8 files
distance_npy: 8 files
matched samples: 8
```

训练结果：

```text
epoch=1 train_loss=3.2800 val_loss=3.2970 best=3.2970
```

评估文件：

```text
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\experiments\smoke\metrics_test.csv
```

预测输出：

```text
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\outputs\smoke_predictions\mask_prob
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\outputs\smoke_predictions\boundary_prob
```

## 解释

本次评估指标不用于论文，因为合成数据只有 8 张且仅训练 1 个 epoch；其意义是证明工程入口、模型前向传播、损失函数、指标计算、checkpoint 与预测导出均能正常执行。正式实验需要在用户手动下载数据集后，将数据转换为统一格式再运行 `configs/topoiwl_default.yaml` 或后续数据集专用配置。
