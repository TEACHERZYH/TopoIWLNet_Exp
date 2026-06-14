# Pretrained Model Selection for TopoIWL-Net

更新时间：2026-06-13

## 结论

可以不从零开始训练。建议采用“两阶段路线”：

1. 主实验优先使用 `torchvision` 的 `MobileNetV3-Large` ImageNet 预训练编码器。
2. 后续扩展实验再比较遥感基础模型或水体专用预训练模型。

这样做的好处是：第一阶段能最快接入现有 TopoIWL-Net 工程，保持轻量化与边界/距离/拓扑多任务头不变；第二阶段再用遥感预训练模型增强论文创新性和泛化实验。

## 推荐优先级

### 方案 A：MobileNetV3-Large ImageNet 预训练

推荐程度：最高，作为主实验默认预训练方案。

适配方式：

- 使用 `torchvision.models.mobilenet_v3_large` 的 ImageNet 权重初始化编码器；
- 取 stride 2 / 4 / 8 / 16 四级特征；
- 保留 TopoIWL-Net 的 FPN 融合、mask head、boundary head、distance head；
- 使用 ImageNet mean / std 归一化。

优点：

- 本机 `dl_env` 已安装 `torchvision==0.22.1+cu128`；
- 权重小、训练稳定、速度快；
- 与轻量网络定位一致；
- 适合作为从零训练的直接对照。

当前工程配置：

```text
configs/topoiwl_mobilenetv3_pretrained.yaml
```

### 方案 B：MobileNetV3-Small ImageNet 预训练

推荐程度：高，作为边缘部署/轻量消融实验。

适配方式与方案 A 一致，只把编码器改为：

```yaml
model:
  encoder: mobilenet_v3_small
  pretrained: true
```

优点：

- 参数量和计算量更小；
- 适合论文中与 Paper 2 的边缘部署方向衔接。

不足：

- 表达能力弱于 MobileNetV3-Large，复杂岸线/窄水道场景可能下降。

### 方案 C：Hydro Foundation Model

推荐程度：中高，适合作为“水体专用预训练”扩展实验。

来源：

- GitHub: https://github.com/isaaccorley/hydro-foundation-model

潜在价值：

- 任务主题与水体识别更接近；
- 可作为“水文遥感预训练优于通用 ImageNet 预训练吗”的对比。

主要风险：

- 与当前轻量 FPN 结构不一定能直接对齐；
- 需要检查权重格式、输入波段、许可证和依赖环境；
- 工程改动大于 MobileNetV3。

### 方案 D：Prithvi-EO / Prithvi-EO-2.0

推荐程度：中，适合遥感基础模型泛化实验，不建议作为第一版主干。

来源：

- Hugging Face: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M

潜在价值：

- 遥感时空基础模型，论文叙事价值强；
- 可作为“foundation model backbone / feature extractor”对比实验；
- 如果后续使用 Landsat / Sentinel 多光谱数据，价值更大。

主要风险：

- 模型较重，与本文“轻量网络”定位存在张力；
- 输入波段、分辨率和 patch 组织方式可能需要额外适配；
- 训练成本高于 MobileNetV3。

### 方案 E：SatMAE / SeCo / RemoteCLIP

推荐程度：中，适合作为第二阶段探索。

来源：

- SatMAE: https://github.com/sustainlab-group/SatMAE
- SeCo: https://github.com/ServiceNow/seasonal-contrast
- RemoteCLIP: https://github.com/ChenDelong1999/RemoteCLIP

潜在价值：

- SatMAE / SeCo 更偏遥感自监督特征；
- RemoteCLIP 可用于语义增强或跨域泛化分析。

主要风险：

- 需要额外适配编码器特征输出；
- 可能引入较多依赖；
- 不一定比 MobileNetV3 更适合像素级细边界提取。

## 当前已完成的工程接入

已新增配置：

```text
configs/topoiwl_mobilenetv3_pretrained.yaml
configs/topoiwl_mobilenetv3_smoke.yaml
```

已新增能力：

- `model.encoder: lightweight`
- `model.encoder: mobilenet_v3_large`
- `model.encoder: mobilenet_v3_small`
- `model.pretrained: true / false`
- `model.pretrained_weights: DEFAULT`
- `model.freeze_encoder: true / false`
- `dataset.image_mean`
- `dataset.image_std`

已完成验证：

```text
mobilenet_v3_large, pretrained=false, synthetic smoke test
epoch=1 train_loss=3.3045 val_loss=3.3415 best=3.3415
```

GF6 真实数据调试也已完成，MobileNetV3-Large ImageNet 权重已成功下载到本机 PyTorch 缓存，并在 batch size 2 + AMP 设置下通过真实样本短训练。

```text
config: configs/topoiwl_gf6_mobilenetv3_pretrained_debug.yaml
epoch=1 train_loss=3.4699 val_loss=3.0425
```

评估文件：

```text
F:\2026\Remote Sensing_codex\TopoIWLNet_Exp\experiments\smoke_mobilenetv3\metrics_test.csv
```

## 正式实验建议

正式数据到位后，至少做三组主实验：

1. `TopoIWL-Net-Random`
   - 配置：`topoiwl_default.yaml`
   - 作用：证明从零训练基线。

2. `TopoIWL-Net-MobileNetV3-ImageNet`
   - 配置：`topoiwl_mobilenetv3_pretrained.yaml`
   - 作用：证明预训练对水边线提取的提升。

3. `TopoIWL-Net-MobileNetV3-FrozenWarmup`
   - 前 5 到 10 epoch 冻结 encoder，仅训练融合层和任务头；
   - 后续解冻全模型微调；
   - 作用：小数据集条件下减少过拟合。

论文表格可报告：

- mIoU / F1；
- BF1@1、BF1@2、BF1@3、BF1@5；
- Chamfer / ASSD / Hausdorff；
- broken segments / component difference；
- 参数量、FLOPs、推理速度。

## 权重下载策略

本次没有下载任何预训练权重。`torchvision` 权重会在第一次运行 `pretrained: true` 时自动下载到 PyTorch 缓存目录。若网络不稳定，可以后续改为手动下载权重文件并放入缓存。

数据集仍遵守既定规则：由用户手动下载，Codex 只负责寻找下载途径、整理转换和实验代码。
