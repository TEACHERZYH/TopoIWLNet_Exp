#!/usr/bin/env bash
set -euo pipefail

cd /data/zyh/projects/TopoIWLNet_Exp

PYTHON_BIN="${PYTHON_BIN:-/data/zhangl/.conda/envs/rs_train/bin/python}"
GPU_ID="${GPU_ID:-3}"
LOG="experiments/fusion_phase1_glh.log"
mkdir -p experiments

echo "[$(date -Is)] Starting GLH final-waterline fusion search" | tee "$LOG"
echo "python=$PYTHON_BIN gpu=$GPU_ID" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/sweep_final_fusion.py \
  --config configs/topoiwl_remote_glh_mobilenetv3_full80.yaml \
  --checkpoint experiments/remote_glh_mobilenetv3_full80/best.pt \
  --dataset-name GLH-Water \
  --dataset-root /data/zyh/datasets/GLH-Water/processed/topoiwl_format \
  --out-dir experiments/remote_glh_mobilenetv3_full80/fusion_phase1 \
  --mask-thresholds 0.30,0.40,0.50,0.60,0.70,0.80 \
  --final-thresholds 0.20,0.30,0.40,0.50,0.60,0.70,0.80 \
  --alphas 0.00,0.25,0.50,0.75,1.00 \
  --mask-buffer-iters 0 \
  --gap-bridge-iters 0 \
  --min-component-sizes 0 \
  --max-val-samples 160 \
  --num-workers 4 \
  --progress-every 20 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] Completed GLH final-waterline fusion search" | tee -a "$LOG"
