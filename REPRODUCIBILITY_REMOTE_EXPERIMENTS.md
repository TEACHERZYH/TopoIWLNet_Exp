# TopoIWL-Net Remote Experiment Reproducibility Notes

Date: 2026-06-26

This file records the experiment provenance for the Remote Sensing manuscript draft. Dataset archives are not distributed with this repository; users must obtain the original datasets from their providers and organize them into the documented `topoiwl_format`.

The public repository contains code, configurations, small result-summary CSV files, and manuscript figure source-data CSV files. It intentionally excludes original imagery, dataset archives, model checkpoints, full experiment directories, prediction rasters, and remote host logs.

## Remote Execution Environment

- Remote host used for full training and evaluation: `zhangl@180.209.128.66`
- Remote project path: `/data/zyh/projects/TopoIWLNet_Exp/`
- Remote dataset root: `/data/zyh/datasets/`
- Python environment used in the completed remote runs: `/data/zhangl/.conda/envs/rs_train/bin/python`
- Local synchronized project path: `F:\2026\Remote Sensing_codex\TopoIWLNet_Exp`

## Dataset Roots

| Dataset | Processed root expected by configs |
|---|---|
| GF6_TCUNet | `/data/zyh/datasets/GF6_TCUNet/processed/topoiwl_format` |
| SeaLand_Coastline_2025 | `/data/zyh/datasets/SeaLand_Coastline_2025/processed/topoiwl_format` |
| GLH-Water | `/data/zyh/datasets/GLH-Water/processed/topoiwl_format` |

The manuscript uses GF6_TCUNet and SeaLand_Coastline_2025 as the two main waterline benchmarks. GLH-Water is treated as a large-scene heterogeneous-label stress test.

## Included Result and Figure Source Data

- `results/baseline_real_results_summary.csv` and `.json`: same-split baseline summary values.
- `results/sensitivity_summary.csv`: real threshold/radius sensitivity summary.
- `results/sensitivity_best_thresholds.csv`: validation-selected threshold summary.
- `paper_source_data/`: small CSV files used for manuscript figure source data, qualitative manifests, and numerical traceability audits.

## Main TopoIWL-Net Runs

| Role | Config | Experiment directory | Key test/evaluation file |
|---|---|---|---|
| GF6 Full / ablation full | `configs/topoiwl_remote_gf6_mobilenetv3_ablate_full80.yaml` | `experiments/remote_gf6_mobilenetv3_ablate_full80` | `metrics_test_val_selected_thresholds.csv` |
| GF6 Lite | `configs/topoiwl_remote_gf6_mobilenetv3_small48_full80.yaml` | `experiments/remote_gf6_mobilenetv3_small48_full80` | `metrics_test_val_selected_thresholds.csv` |
| GF6 PVTv2 | `configs/topoiwl_remote_gf6_pvtv2_b2_sota120.yaml` | `experiments/remote_gf6_pvtv2_b2_sota120` | `metrics_test_val_best_bf1.csv` |
| SeaLand Full | `configs/topoiwl_remote_sealand_mobilenetv3_full80.yaml` | `experiments/remote_sealand_mobilenetv3_full80` | `metrics_test_val_best_bf1.csv` |
| SeaLand Lite | `configs/topoiwl_remote_sealand_mobilenetv3_small48_full80.yaml` | `experiments/remote_sealand_mobilenetv3_small48_full80` | `metrics_test_val_best_bf1.csv` |
| SeaLand PVTv2 | `configs/topoiwl_remote_sealand_pvtv2_b2_sota120.yaml` | `experiments/remote_sealand_pvtv2_b2_sota120` | `metrics_test_val_best_bf1.csv` |
| GLH Full | `configs/topoiwl_remote_glh_mobilenetv3_full80.yaml` | `experiments/remote_glh_mobilenetv3_full80` | `metrics_test_val_best_bf1.csv` |
| GLH Lite | `configs/topoiwl_remote_glh_mobilenetv3_small48_full80.yaml` | `experiments/remote_glh_mobilenetv3_small48_full80` | `metrics_test_val_best_bf1.csv` |
| GLH PVTv2 | `configs/topoiwl_remote_glh_pvtv2_b2_sota120.yaml` | `experiments/remote_glh_pvtv2_b2_sota120` | `metrics_test_val_best_bf1.csv` |
| GLH Optimized | `configs/topoiwl_remote_glh_mobilenetv3_boundary_metric50.yaml` | `experiments/remote_glh_mobilenetv3_boundary_metric50` | `fusion_metric_continuity_grid/metrics_test_fusion_val_best.csv` |

## Baseline Runs

The manuscript baseline rows use same-split runs from:

- `configs/baseline_remote_gf6_unet80.yaml`
- `configs/baseline_remote_gf6_mobilenetv3_unet80.yaml`
- `configs/baseline_remote_gf6_deeplabv3_mobilenet80.yaml`
- `configs/baseline_remote_gf6_deeplabv3_resnet50_80.yaml`
- `configs/baseline_remote_sealand_unet80.yaml`
- `configs/baseline_remote_sealand_mobilenetv3_unet80.yaml`
- `configs/baseline_remote_sealand_deeplabv3_mobilenet80.yaml`
- `configs/baseline_remote_sealand_deeplabv3_resnet50_80.yaml`
- `configs/baseline_remote_glh_unet80.yaml`
- `configs/baseline_remote_glh_mobilenetv3_unet80.yaml`
- `configs/baseline_remote_glh_deeplabv3_mobilenet80.yaml`
- `configs/baseline_remote_glh_deeplabv3_resnet50_80.yaml`

Each baseline experiment directory is named after its config stem under `experiments/`. Standard files include:

- `best.pt`
- `threshold_sweep_val_fast.csv`
- `metrics_test_mask_boundary_val_best_bf1.csv`
- `efficiency_v100_fp32_b1_512_fair.csv`

## GLH Optimized Fusion Protocol

The optimized GLH row in the manuscript uses the same network complexity as TopoIWL-Net Full but selects the checkpoint and final waterline generation settings on the validation set.

- Config: `configs/topoiwl_remote_glh_mobilenetv3_boundary_metric50.yaml`
- Checkpoint: `experiments/remote_glh_mobilenetv3_boundary_metric50/best_metric.pt`
- Fusion summary: `experiments/remote_glh_mobilenetv3_boundary_metric50/fusion_metric_continuity_grid/fusion_best_summary.json`
- Test metrics: `experiments/remote_glh_mobilenetv3_boundary_metric50/fusion_metric_continuity_grid/metrics_test_fusion_val_best.csv`
- Validation-selected parameters:
  - mask threshold: `0.65`
  - final threshold: `0.65`
  - alpha: `0.40`
  - mask boundary width: `1`
  - mask buffer iterations: `0`
  - gap bridge iterations: `0`
  - minimum component size: `64`

Representative command for the final prediction maps:

```bash
CUDA_VISIBLE_DEVICES=7 /data/zhangl/.conda/envs/rs_train/bin/python scripts/predict_fusion.py \
  --config configs/topoiwl_remote_glh_mobilenetv3_boundary_metric50.yaml \
  --checkpoint experiments/remote_glh_mobilenetv3_boundary_metric50/best_metric.pt \
  --split test \
  --out-dir outputs/remote_glh_boundary_metric_fusion_min64_test_predictions \
  --mask-threshold 0.65 \
  --final-threshold 0.65 \
  --alpha 0.40 \
  --mask-boundary-width 1 \
  --mask-buffer-iters 0 \
  --gap-bridge-iters 0 \
  --min-component-size 64 \
  --num-workers 4
```

## Prediction Outputs Used by Qualitative Figures

| Dataset / method | Output directory |
|---|---|
| GF6 TopoIWL-Net | `outputs/remote_gf6_ablate_full80_test_predictions` |
| SeaLand TopoIWL-Net | `outputs/remote_sealand_full80_test_predictions` |
| GLH TopoIWL-Net Optimized | `outputs/remote_glh_boundary_metric_fusion_min64_test_predictions` |
| GLH MobileNetV3-UNet baseline | `outputs/baseline_remote_glh_mobilenetv3_unet80_test_predictions` |
| GLH DeepLabV3-ResNet50 baseline | `outputs/baseline_remote_glh_deeplabv3_resnet50_80_test_predictions` |

The refreshed qualitative Figure 8 uses GLH sample `glh_test_74_02560_12288` and the optimized fusion output.

## Standard Evaluation Commands

Threshold search:

```bash
/data/zhangl/.conda/envs/rs_train/bin/python scripts/sweep_thresholds_fast.py \
  --config <config.yaml> \
  --checkpoint <experiment_dir>/best.pt \
  --out-csv <experiment_dir>/threshold_sweep_val_fast.csv
```

Test evaluation:

```bash
/data/zhangl/.conda/envs/rs_train/bin/python scripts/evaluate.py \
  --config <config.yaml> \
  --checkpoint <experiment_dir>/best.pt \
  --split test \
  --boundary-source head \
  --out-csv <experiment_dir>/metrics_test_val_best_bf1.csv
```

Efficiency measurement:

```bash
/data/zhangl/.conda/envs/rs_train/bin/python scripts/measure_efficiency.py \
  --config <config.yaml> \
  --checkpoint <experiment_dir>/best.pt \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --warmup 100 \
  --iters 500 \
  --out-csv <experiment_dir>/efficiency_v100_fp32_b1_512_fair.csv
```

## Manuscript Result Provenance

- Table 2: GF6 and SeaLand main comparison.
- Table 3: GLH stress-test comparison, including the optimized GLH fusion row.
- Table 4: Params/FLOPs/latency/FPS.
- Tables 5-6: GF6 and SeaLand ablations.
- Figure 6: `Paper1_TopoIWL-Net_Instantaneous_Waterline/figures/source_data/figure6_accuracy_efficiency_tradeoff_source_data.csv`
- Figure 8: `Paper1_TopoIWL-Net_Instantaneous_Waterline/figures/source_data/figure8_qualitative_manifest.csv`
- Supplementary Figure S3: `Paper1_TopoIWL-Net_Instantaneous_Waterline/figures/source_data/figure_s3_threshold_radius_sensitivity_source_data.csv`
