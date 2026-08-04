#!/bin/bash
# R4b — 5% s0 labeled-only fine-tune (no SSL), eval val_full=6019.
# Provides the apples-to-apples 5% no-SSL baseline for the ceiling figure.
set -u
LOG=/root/BMVC2026/scripts/r4b_pipeline.log
WD=/root/BMVC2026/work_dirs/R4b_sup_5p_s0_eval_full
CFG=/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0_eval_full.py
GPU=4

mkdir -p "$WD"
echo "[$(date)] R4b start, GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$CFG" --work-dir "$WD" \
    >> "$WD/train.log" 2>&1
echo "[$(date)] train exited rc=$?" >> "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: epoch_10.pth missing" >> "$LOG"
    exit 1
fi

# External eval
echo "[$(date)] external eval" >> "$LOG"
EVAL_DIR=/root/BMVC2026/work_dirs/R4b_external_eval
mkdir -p "$EVAL_DIR"
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/test.py "$CFG" "$WD/epoch_10.pth" \
    --eval bbox \
    --eval-options "jsonfile_prefix=${EVAL_DIR}/results" \
    > "${EVAL_DIR}/eval.log" 2>&1
echo "[$(date)] R4b complete" >> "$LOG"
