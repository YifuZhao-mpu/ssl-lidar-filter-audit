#!/bin/bash
# R19 — conjunctive control: perclass_only AND six-camera consensus, 5% s0.
#
# Why this arm exists: the audited camera-support family (R8b/R8c/R8d/R7c)
# applies no teacher-confidence condition, so its 7-11 NDS deficit against the
# score family confounds "camera evidence is bad" with "the confidence floor
# was removed". R19 keeps the confidence floor and layers the R7c consensus
# rule on top as a pure veto, so its kept set is a strict subset of R7b
# (perclass_only). Read against R7b it isolates the camera veto; read against
# R18 (positive-only rescue, camera may only ADD) it closes the other half of
# the subordinated-camera space.
#
# Same base config, same initialization, same schedule, same epoch count as
# every other arm. Only the pseudo-label ann_file differs.
set -u

ROOT=/root/BMVC2026
PYTHON=/root/miniconda3/envs/teocc/bin/python
GPU=${1:-0}

TAG=R19_ssl_5p_s0_perclass_and_consensus
CFG=$ROOT/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_perclass_and_consensus.py
WD=$ROOT/work_dirs/$TAG
ED=${WD}_external_eval
LOG=$ROOT/scripts/r19_conjunctive_pipeline.log

mkdir -p "$WD" "$ED"

echo "[$(date)] R19 TRAIN start on GPU $GPU  cfg=$CFG" | tee -a "$LOG"
( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT \
    "$PYTHON" tools/train.py "$CFG" --work-dir "$WD" >> "$WD/train.log" 2>&1 )
rc=$?
echo "[$(date)] R19 TRAIN done rc=$rc" | tee -a "$LOG"

if [ ! -f "$WD/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: $WD/epoch_10.pth missing - skipping eval" | tee -a "$LOG"
    exit 1
fi

# External eval on val_full=6019 with the patched _format_bbox, raw ep10
# checkpoint (not the EMA copy) - identical protocol to every reported run.
echo "[$(date)] R19 EVAL start" | tee -a "$LOG"
( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ROOT \
    "$PYTHON" tools/test.py "$CFG" "$WD/epoch_10.pth" --eval bbox \
    --eval-options "jsonfile_prefix=$ED/results" > "$ED/eval.log" 2>&1 )
echo "[$(date)] R19 EVAL done rc=$?" | tee -a "$LOG"
echo "[$(date)] R19 COMPLETE" | tee -a "$LOG"
