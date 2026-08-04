#!/bin/bash
# R7c-s1 pipeline: 5% s1 consensus filter → merge → SSL student → eval.
# Codex round-5: confirms D2 family-gap holds at a second seed.
set -u
LOG=/root/BMVC2026/scripts/r7c_seed1_pipeline.log
WD=/root/BMVC2026/work_dirs/R7c_ssl_5p_s1_consensus
CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_consensus.py
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_5p_s1_raw_full.pkl
DET_CACHE=/root/BMVC2026/cache/fasterrcnn_v2_train.pkl
SOURCE_INFOS=/root/BMVC2026/infos/nuscenes_infos_train_5p_s1_unlabeled.pkl
FILTER_OUTDIR=/root/BMVC2026/cache/pseudo_5p_s1_filtered
CONSENSUS_PKL="${FILTER_OUTDIR}/consensus.pkl"
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_consensus.pkl
GPU=7

mkdir -p "$WD" "$FILTER_OUTDIR"
echo "[$(date)] R7c-s1 start, GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# Stage 1: consensus filter on 5% s1 raw pseudo
if [ ! -f "$CONSENSUS_PKL" ]; then
    echo "[$(date)] stage 1: consensus filter" >> "$LOG"
    cd /root/BMVC2026
    /root/miniconda3/envs/teocc/bin/python tools/g2_filter_pseudo.py \
        --raw-pseudo "$RAW_PSEUDO" \
        --det-cache "$DET_CACHE" \
        --source-infos "$SOURCE_INFOS" \
        --out-dir "$FILTER_OUTDIR" \
        --variants consensus >> "$LOG" 2>&1
    echo "[$(date)] stage 1 exited rc=$?" >> "$LOG"
fi

# Stage 2: merge consensus → infos
if [ ! -f "$PSEUDO_INFOS" ]; then
    echo "[$(date)] stage 2: passthrough merge" >> "$LOG"
    /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$CONSENSUS_PKL" \
        --unlabeled-infos "$SOURCE_INFOS" \
        --variant passthrough \
        --out-pkl "$PSEUDO_INFOS" >> "$LOG" 2>&1
    echo "[$(date)] stage 2 exited rc=$?" >> "$LOG"
fi
[ ! -f "$PSEUDO_INFOS" ] && { echo "FATAL infos missing" >> "$LOG"; exit 1; }

# Stage 3: SSL train
echo "[$(date)] stage 3: SSL train" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$CFG" --work-dir "$WD" >> "$WD/train.log" 2>&1
echo "[$(date)] stage 3 exited rc=$?" >> "$LOG"
[ ! -f "$WD/epoch_10.pth" ] && { echo "FATAL ckpt missing" >> "$LOG"; exit 1; }

# Stage 4: external eval
echo "[$(date)] stage 4: external eval" >> "$LOG"
EVAL_DIR=/root/BMVC2026/work_dirs/R7c_s1_external_eval
mkdir -p "$EVAL_DIR"
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/test.py "$CFG" "$WD/epoch_10.pth" \
    --eval bbox \
    --eval-options "jsonfile_prefix=${EVAL_DIR}/results" \
    > "${EVAL_DIR}/eval.log" 2>&1
echo "[$(date)] R7c-s1 complete" >> "$LOG"
