#!/bin/bash
# Train + external-eval one arm of the 2026-08-12 batch.
#   usage: r19r20_train.sh <GPU> <TAG> <CFG-basename>
#
# Arms (review-stage/PREREG_2026-08-12.md):
#   GPU 3  R19_ssl_5p_s1_perclass_and_consensus   conjunctive, seed 1
#   GPU 6  R19_ssl_5p_s2_perclass_and_consensus   conjunctive, seed 2
#   GPU 7  R20_ssl_5p_s0_perclass_random_prune    volume-matched random pruning
#
# Identical protocol to every reported run: same base config, same
# initialization, same schedule, 10 epochs, external eval on val_full=6019 with
# the patched _format_bbox against the RAW ep10 checkpoint (not the EMA copy).
set -u

ROOT=/root/BMVC2026
PYTHON=/root/miniconda3/envs/teocc/bin/python
GPU=$1
TAG=$2
CFG=$ROOT/configs/$3

WD=$ROOT/work_dirs/$TAG
ED=${WD}_external_eval
LOG=$ROOT/scripts/logs20/${TAG}.log

mkdir -p "$WD" "$ED"

echo "[$(date)] TRAIN start  gpu=$GPU  tag=$TAG" | tee -a "$LOG"
( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT \
    "$PYTHON" tools/train.py "$CFG" --work-dir "$WD" >> "$WD/train.log" 2>&1 )
echo "[$(date)] TRAIN done rc=$?" | tee -a "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: $WD/epoch_10.pth missing - not evaluating" | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] EVAL start" | tee -a "$LOG"
( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT \
    "$PYTHON" tools/test.py "$CFG" "$WD/epoch_10.pth" --eval bbox \
    --eval-options "jsonfile_prefix=$ED/results" > "$ED/eval.log" 2>&1 )
echo "[$(date)] EVAL done rc=$?" | tee -a "$LOG"

NDS=$(python3 -c "import json;print(json.load(open('$ED/results/pts_bbox/metrics_summary.json'))['nd_score'])" 2>/dev/null)
echo "[$(date)] COMPLETE $TAG  NDS=$NDS" | tee -a "$LOG"
