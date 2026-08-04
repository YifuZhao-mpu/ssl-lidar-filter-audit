#!/bin/bash
# R10 pipeline: 10% s0 teacher → pseudo dump → perclass_only filter → SSL student.
# Same structure as r17_pipeline.sh, scaled to 10% labels (2813 labeled,
# ~25317 unlabeled). Started: 2026-05-08.
set -u
LOG=/root/BMVC2026/scripts/r10_pipeline.log
TEACHER_DIR=/root/BMVC2026/work_dirs/R10a_sup_10p_s0
STUDENT_DIR=/root/BMVC2026/work_dirs/R10b_ssl_10p_s0_perclass_only
TEACHER_CFG=/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_10p_s0.py
STUDENT_CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_10p_s0_perclass_only.py
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_10p_s0_raw_full.pkl
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_10p_s0_perclass_only.pkl
GPU=3

mkdir -p "$TEACHER_DIR" "$STUDENT_DIR"
echo "[$(date)] R10 pipeline start, GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# ---------- Stage 1: train R10a teacher ---------- #
echo "[$(date)] stage 1: training R10a teacher" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$TEACHER_CFG" --work-dir "$TEACHER_DIR" \
    >> "$TEACHER_DIR/train.log" 2>&1
rc=$?
echo "[$(date)] stage 1 exited rc=$rc" >> "$LOG"
if [ ! -f "$TEACHER_DIR/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: teacher epoch_10.pth missing — abort" >> "$LOG"
    exit 1
fi

# ---------- Stage 2: dump pseudo on 10% unlabeled ---------- #
echo "[$(date)] stage 2: dumping pseudo on 10p_s0 unlabeled" >> "$LOG"
cd /root/BMVC2026
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/g2_dump_pseudo.py \
    --config "$TEACHER_CFG" \
    --ckpt "$TEACHER_DIR/epoch_10.pth" \
    --unlabeled-pkl /root/BMVC2026/infos/nuscenes_infos_train_10p_s0_unlabeled.pkl \
    --out "$RAW_PSEUDO" \
    >> "$LOG" 2>&1
rc=$?
echo "[$(date)] stage 2 exited rc=$rc" >> "$LOG"
if [ ! -f "$RAW_PSEUDO" ]; then
    echo "[$(date)] FATAL: raw pseudo missing — abort" >> "$LOG"
    exit 1
fi

# ---------- Stage 3: perclass_only filter + merge ---------- #
echo "[$(date)] stage 3: perclass_only filter+merge" >> "$LOG"
/root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
    --raw-pseudo "$RAW_PSEUDO" \
    --unlabeled-infos /root/BMVC2026/infos/nuscenes_infos_train_10p_s0_unlabeled.pkl \
    --det-cache /root/BMVC2026/cache/fasterrcnn_v2_train.pkl \
    --variant perclass_only \
    --out-pkl "$PSEUDO_INFOS" \
    >> "$LOG" 2>&1
rc=$?
echo "[$(date)] stage 3 exited rc=$rc" >> "$LOG"
if [ ! -f "$PSEUDO_INFOS" ]; then
    echo "[$(date)] FATAL: pseudo infos pkl missing — abort" >> "$LOG"
    exit 1
fi

# ---------- Stage 4: train R10b SSL student ---------- #
echo "[$(date)] stage 4: training R10b SSL student" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$STUDENT_CFG" --work-dir "$STUDENT_DIR" \
    >> "$STUDENT_DIR/train.log" 2>&1
rc=$?
echo "[$(date)] stage 4 exited rc=$rc" >> "$LOG"

if [ ! -f "$STUDENT_DIR/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: student epoch_10.pth missing — abort eval" >> "$LOG"
    exit 1
fi

# ---------- Stage 5: external eval both checkpoints ---------- #
echo "[$(date)] stage 5: external eval R10a (baseline) + R10b (claim)" >> "$LOG"
cd /root/TEOcc
PYTHON=/root/miniconda3/envs/teocc/bin/python

eval_one () {
    local tag="$1"
    local cfg="$2"
    local ckpt="$3"
    local out_dir="/root/BMVC2026/work_dirs/${tag}_external_eval"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
        "$PYTHON" tools/test.py "$cfg" "$ckpt" \
        --eval bbox \
        --eval-options "jsonfile_prefix=${out_dir}/results" \
        > "${out_dir}/eval.log" 2>&1
    echo "[$(date)] eval ${tag} rc=$?" >> "$LOG"
}

eval_one R10a "$TEACHER_CFG" "$TEACHER_DIR/epoch_10.pth"
eval_one R10b "$STUDENT_CFG" "$STUDENT_DIR/epoch_10.pth"

echo "[$(date)] R10 pipeline complete" >> "$LOG"
