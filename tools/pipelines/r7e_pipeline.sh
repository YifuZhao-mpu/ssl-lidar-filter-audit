#!/bin/bash
# R7e — Proper per-camera column-shuffle (decorrelation) consensus.
# Fixes the row-shuffle bug in R7d which preserved joint distribution.
set -u
LOG=/root/BMVC2026/scripts/r7e_pipeline.log
DECOR_PKL=/root/BMVC2026/cache/pseudo_5p_s0_filtered/consensus_decorr.pkl
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_5p_s0_consensus_decorr.pkl
WD=/root/BMVC2026/work_dirs/R7e_ssl_5p_s0_consensus_decorr
CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_consensus_decorr.py
GPU=0

mkdir -p "$WD"
echo "[$(date)] R7e pipeline start GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

if [ ! -f "$DECOR_PKL" ]; then
    echo "[$(date)] stage 1: column-shuffle filter" >> "$LOG"
    cd /root/BMVC2026
    /root/miniconda3/envs/teocc/bin/python -u tools/g2_consensus_decorr_filter.py \
        --out "$DECOR_PKL" --seed 0 >> "$LOG" 2>&1
fi

if [ ! -f "$PSEUDO_INFOS" ]; then
    echo "[$(date)] stage 2: passthrough merge" >> "$LOG"
    /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$DECOR_PKL" \
        --unlabeled-infos /root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl \
        --variant passthrough \
        --out-pkl "$PSEUDO_INFOS" >> "$LOG" 2>&1
fi

echo "[$(date)] stage 3: SSL train (10 epochs)" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py "$CFG" \
    --work-dir "$WD" >> "$WD/train.log" 2>&1
echo "[$(date)] stage 3 rc=$?" >> "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "FATAL: ckpt missing" >> "$LOG"; exit 1
fi

EVAL_DIR=/root/BMVC2026/work_dirs/R7e_external_eval
mkdir -p "$EVAL_DIR"
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/test.py "$CFG" "$WD/epoch_10.pth" \
    --eval bbox \
    --eval-options "jsonfile_prefix=${EVAL_DIR}/results" \
    > "${EVAL_DIR}/eval.log" 2>&1
echo "[$(date)] R7e complete" >> "$LOG"
