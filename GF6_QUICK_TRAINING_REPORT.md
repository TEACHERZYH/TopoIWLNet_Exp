# GF6 Quick Training Report

运行时间：2026-06-14

## 目的

在完整正式训练前，使用 GF6_TCUNet 真实数据和 MobileNetV3-Large ImageNet 预训练编码器做一次简短训练，验证：

- 真实数据训练链路是否稳定；
- GPU + AMP 设置是否可用；
- loss 是否开始下降；
- 预测结果是否从随机状态进入可学习状态。

## 配置

```text
configs/topoiwl_gf6_mobilenetv3_quick.yaml
```

核心设置：

```yaml
model:
  encoder: mobilenet_v3_large
  pretrained: true
  width: 64

train:
  epochs: 3
  batch_size: 2
  amp: true
  max_train_batches: 20
  max_val_batches: 8
```

实际训练量：

```text
3 epochs x 20 train batches x 2 images = 120 training image iterations
```

## 训练日志

```text
epoch,train_loss,val_loss,lr
1,3.097722887992859,3.2218832969665527,0.000225
2,2.987492597103119,2.8526518642902374,0.000075
3,3.0971890807151796,2.6371697783470154,0.0
```

结论：

- 验证 loss 从 `3.2219` 降到 `2.6372`；
- 模型已经开始学习；
- 当前训练量太少，训练 loss 有波动是正常现象。

## 小样本测试评估

测试范围：

```text
test split first 32 samples
```

输出文件：

```text
experiments/gf6_mobilenetv3_quick/metrics_test_32.csv
```

平均结果：

```text
IoU:       0.3748
F1:        0.4099
Precision: 0.3750
Recall:    0.9593
BF1@1:     0.4350
BF1@2:     0.5271
BF1@3:     0.5953
BF1@5:     0.6530
Chamfer:   21.8241
ASSD:      10.9120
Hausdorff: 109.9208
```

解释：

- Recall 很高，说明模型倾向于把真实水体覆盖进去；
- Precision 偏低，说明短训后仍明显过分割；
- boundary 指标已经非零，说明边界分支开始产生有效响应；
- 组件数偏多，说明边界/水体预测还有大量纹理误检。

## 预测预览

预测输出：

```text
outputs/gf6_quick_test_predictions
```

预览图：

```text
outputs/gf6_quick_test_preview.png
```

预览图列顺序：

```text
image | gt mask | pred mask probability | gt edge overlay | pred edge overlay
```

## 判断

本次 short run 成功证明真实数据、预训练模型、AMP、训练、评估和预测输出均可用。模型表现已经脱离随机状态，但远未收敛，不可作为论文结果。

下一步建议进行正式训练：

```powershell
conda run -n dl_env python scripts\train.py --config configs\topoiwl_mobilenetv3_pretrained.yaml
```

如果希望先更稳妥地试跑，可将正式配置复制为 20 epoch 版本，观察 val loss 和 BF1 曲线后再开 160 epoch。
