#!/bin/bash
# R7d — Support-shuffle consensus intervention.
# Stage 1: compute shuffled-consensus filter pkl
# Stage 2: merge to infos PKL
# Stage 3: train R7d SSL student on GPU 0
# Stage 4: external eval
set -u
LOG=/root/BMVC2026/scripts/r7d_pipeline.log
SHUF_PKL=/root/BMVC2026/cache/pseudo_5p_s0_filtered/consensus_shuf.pkl
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_5p_s0_consensus_shuf.pkl
WD=/root/BMVC2026/work_dirs/R7d_ssl_5p_s0_consensus_shuf
CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_consensus_shuf.py
GPU=0

mkdir -p "$WD"
echo "[$(date)] R7d pipeline start GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# Stage 1: shuffled filter
if [ ! -f "$SHUF_PKL" ]; then
    echo "[$(date)] stage 1: shuffled consensus filter" >> "$LOG"
    cd /root/BMVC2026
    /root/miniconda3/envs/teocc/bin/python -u tools/g2_consensus_shuf_filter.py \
        --out "$SHUF_PKL" --seed 0 >> "$LOG" 2>&1
fi

# Stage 2: passthrough merge
if [ ! -f "$PSEUDO_INFOS" ]; then
    echo "[$(date)] stage 2: passthrough merge" >> "$LOG"
    /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$SHUF_PKL" \
        --unlabeled-infos /root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl \
        --variant passthrough \
        --out-pkl "$PSEUDO_INFOS" >> "$LOG" 2>&1
fi

# Stage 3: SSL train
echo "[$(date)] stage 3: SSL train" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py "$CFG" \
    --work-dir "$WD" >> "$WD/train.log" 2>&1
echo "[$(date)] stage 3 rc=$?" >> "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: ckpt missing" >> "$LOG"; exit 1
fi

# Stage 4: external eval
EVAL_DIR=/root/BMVC2026/work_dirs/R7d_external_eval
mkdir -p "$EVAL_DIR"
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/test.py "$CFG" "$WD/epoch_10.pth" \
    --eval bbox \
    --eval-options "jsonfile_prefix=${EVAL_DIR}/results" \
    > "${EVAL_DIR}/eval.log" 2>&1
echo "[$(date)] R7d complete" >> "$LOG"
