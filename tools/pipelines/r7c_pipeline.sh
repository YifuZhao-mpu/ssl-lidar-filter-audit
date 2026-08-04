#!/bin/bash
# R7c — 5% s0 SSL student with the canonical 6-camera consensus filter.
# Stages:
#   1. Run g2_filter_pseudo.py with --variants consensus on R5a raw 5% pseudo
#   2. Merge consensus.pkl into infos via passthrough
#   3. Train R7c SSL student on GPU 5
#   4. External eval on val_full=6019
set -u
LOG=/root/BMVC2026/scripts/r7c_pipeline.log
WD=/root/BMVC2026/work_dirs/R7c_ssl_5p_s0_consensus
CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_consensus.py
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_5p_s0_raw_full.pkl
DET_CACHE=/root/BMVC2026/cache/fasterrcnn_v2_train.pkl
SOURCE_INFOS=/root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl
FILTER_OUTDIR=/root/BMVC2026/cache/pseudo_5p_s0_filtered
CONSENSUS_PKL="${FILTER_OUTDIR}/consensus.pkl"
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_5p_s0_consensus.pkl
GPU=5

mkdir -p "$WD"
echo "[$(date)] R7c start, GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# ---------- Stage 1: regenerate consensus filter pkl ---------- #
if [ ! -f "$CONSENSUS_PKL" ]; then
    echo "[$(date)] stage 1: g2_filter_pseudo.py --variants consensus" >> "$LOG"
    cd /root/BMVC2026
    /root/miniconda3/envs/teocc/bin/python tools/g2_filter_pseudo.py \
        --raw-pseudo "$RAW_PSEUDO" \
        --det-cache "$DET_CACHE" \
        --source-infos "$SOURCE_INFOS" \
        --out-dir "$FILTER_OUTDIR" \
        --variants consensus \
        >> "$LOG" 2>&1
    rc=$?
    echo "[$(date)] stage 1 exited rc=$rc" >> "$LOG"
    if [ ! -f "$CONSENSUS_PKL" ]; then
        echo "[$(date)] FATAL: consensus.pkl missing" >> "$LOG"
        exit 1
    fi
else
    echo "[$(date)] stage 1: consensus.pkl already exists, skipping" >> "$LOG"
fi

# ---------- Stage 2: merge consensus.pkl into infos ---------- #
if [ ! -f "$PSEUDO_INFOS" ]; then
    echo "[$(date)] stage 2: passthrough merge consensus → infos" >> "$LOG"
    /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$CONSENSUS_PKL" \
        --unlabeled-infos "$SOURCE_INFOS" \
        --variant passthrough \
        --out-pkl "$PSEUDO_INFOS" \
        >> "$LOG" 2>&1
    rc=$?
    echo "[$(date)] stage 2 exited rc=$rc" >> "$LOG"
    if [ ! -f "$PSEUDO_INFOS" ]; then
        echo "[$(date)] FATAL: pseudo infos pkl missing" >> "$LOG"
        exit 1
    fi
else
    echo "[$(date)] stage 2: pseudo infos already exists, skipping" >> "$LOG"
fi

# ---------- Stage 3: train R7c SSL student ---------- #
echo "[$(date)] stage 3: training R7c SSL student" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$CFG" --work-dir "$WD" \
    >> "$WD/train.log" 2>&1
echo "[$(date)] stage 3 exited rc=$?" >> "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: student epoch_10.pth missing" >> "$LOG"
    exit 1
fi

# ---------- Stage 4: external eval ---------- #
echo "[$(date)] stage 4: external eval R7c" >> "$LOG"
EVAL_DIR=/root/BMVC2026/work_dirs/R7c_external_eval
mkdir -p "$EVAL_DIR"
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/test.py "$CFG" "$WD/epoch_10.pth" \
    --eval bbox \
    --eval-options "jsonfile_prefix=${EVAL_DIR}/results" \
    > "${EVAL_DIR}/eval.log" 2>&1
echo "[$(date)] R7c complete" >> "$LOG"
