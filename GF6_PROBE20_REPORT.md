# GF6 Probe20 Training Report

运行时间：2026-06-14

## 目的

在 quick run 之后，进一步验证 MobileNetV3-Large ImageNet 预训练版 TopoIWL-Net 在 GF6_TCUNet 数据集上的中等长度训练趋势，并判断：

- loss 是否继续下降；
- 默认阈值 `0.5` 是否合适；
- 是否值得进入全量正式训练；
- 当前主要误差来自模型未收敛还是阈值/过分割。

## 训练配置

```text
configs/topoiwl_gf6_mobilenetv3_probe20.yaml
```

核心设置：

```yaml
model:
  encoder: mobilenet_v3_large
  pretrained: true
  width: 64

train:
  epochs: 20
  batch_size: 2
  amp: true
  max_train_batches: 80
  max_val_batches: 24
```

实际训练量：

```text
20 epochs x 80 train batches x 2 images = 3200 training image iterations
```

## 训练结果

训练日志：

```text
experiments/gf6_mobilenetv3_probe20/train_log.csv
```

最优验证 loss：

```text
best val_loss = 2.8954 at epoch 15
```

最终记录：

```text
epoch  train_loss  val_loss
1      3.3694      3.2032
5      3.0669      2.9635
10     3.0360      3.1223
15     3.1009      2.8954
20     3.0109      2.9957
```

loss 曲线：

```text
outputs/gf6_probe20_loss_curve.png
```

## 默认阈值评估

配置阈值：

```text
mask_threshold = 0.5
boundary_threshold = 0.5
```

测试范围：

```text
test split first 64 samples
```

结果：

```text
IoU:       0.4427
F1:        0.4711
Precision: 0.4479
Recall:    0.8501
BF1@3:     0.5070
BF1@5:     0.6044
```

解释：默认阈值下 recall 高于 precision，说明仍存在明显过分割。

## 阈值扫描

验证范围：

```text
val split first 64 samples
```

输出：

```text
experiments/gf6_mobilenetv3_probe20/threshold_sweep_val64.csv
```

发现：

```text
best mask IoU threshold ~= 0.7
best boundary BF1 threshold ~= 0.8
```

验证集最佳 mask 设置：

```text
mask_threshold: 0.7
IoU:       0.5624
F1:        0.6316
Precision: 0.6847
Recall:    0.6068
```

验证集最佳 boundary 设置：

```text
boundary_threshold: 0.8
BF1@3: 0.6389
```

## 调阈值后测试结果

配置：

```text
configs/topoiwl_gf6_mobilenetv3_probe20_tuned.yaml
mask_threshold = 0.7
boundary_threshold = 0.8
```

64 张 test 样本：

```text
IoU:       0.5401
F1:        0.6099
Precision: 0.6587
Recall:    0.5839
BF1@3:     0.6359
BF1@5:     0.6732
```

完整 test split，284 张样本：

```text
IoU:       0.5997
F1:        0.6556
Precision: 0.7096
Recall:    0.6269
BF1@1:     0.4867
BF1@2:     0.5646
BF1@3:     0.6039
BF1@5:     0.6323
Chamfer:   51.6212
ASSD:      25.8106
Hausdorff: 231.8986
```

输出文件：

```text
experiments/gf6_mobilenetv3_probe20/metrics_test_full_tuned.csv
```

## 预测预览

预测概率图：

```text
outputs/gf6_probe20_test_predictions
```

预览图：

```text
outputs/gf6_probe20_test_preview.png
```

观察：

- 大水体轮廓已基本可识别；
- 默认阈值下过分割明显；
- 调高阈值后 precision 和 IoU 明显改善；
- 细碎纹理仍会产生误检，component difference 仍偏高。

## 结论

Probe20 证明 MobileNetV3-Large 预训练方案值得进入正式训练。当前主要问题不是训练链路，而是：

- 模型仍未充分收敛；
- 低阈值导致过分割；
- 细碎地物纹理会触发边界误检；
- component-level 拓扑指标仍需改善。

## 下一步建议

进入全量正式训练，先跑 40 epoch 版本：

```text
configs/topoiwl_gf6_mobilenetv3_formal40.yaml
```

运行命令：

```powershell
conda run -n dl_env python scripts\train.py --config configs\topoiwl_gf6_mobilenetv3_formal40.yaml
```

正式训练结束后：

```powershell
conda run -n dl_env python scripts\evaluate.py `
  --config configs\topoiwl_gf6_mobilenetv3_formal40.yaml `
  --checkpoint experiments\gf6_mobilenetv3_formal40\best.pt `
  --split test
```

如果 formal40 仍然过分割，下一步再调：

- `boundary_pos_weight: 6.0`
- `lambda_boundary: 0.7`
- `lambda_topology: 0.3`
- 增加 post-processing small component removal。
