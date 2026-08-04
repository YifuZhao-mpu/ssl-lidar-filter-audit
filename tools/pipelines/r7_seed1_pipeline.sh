#!/bin/bash
# R7-s1 pipeline: 5% s1 teacher → pseudo dump → make TWO infos PKLs
# (perclass_boost + perclass_only) → train R7-s1 (GPU 0) and R7b-s1 (GPU 1) in
# parallel → external eval both.
# Started: 2026-05-08 — strengthens D1 to two-seed evidence per Codex round 2.
set -u
LOG=/root/BMVC2026/scripts/r7_seed1_pipeline.log
TEACHER_DIR=/root/BMVC2026/work_dirs/R4_sup_5p_s1
TEACHER_CFG=/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s1.py
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_5p_s1_raw_full.pkl
PSEUDO_BOOST=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_perclass_boost.pkl
PSEUDO_ONLY=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_perclass_only.pkl
SOURCE_INFOS=/root/BMVC2026/infos/nuscenes_infos_train_5p_s1_unlabeled.pkl
DET_CACHE=/root/BMVC2026/cache/fasterrcnn_v2_train.pkl
GPU_TEACHER=0
GPU_BOOST=0
GPU_ONLY=1

mkdir -p "$TEACHER_DIR"
echo "[$(date)] R7-s1 pipeline start" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# ---------- Stage 1: train R4-s1 teacher ---------- #
echo "[$(date)] stage 1: training R4-s1 teacher on GPU ${GPU_TEACHER}" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU_TEACHER} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$TEACHER_CFG" --work-dir "$TEACHER_DIR" \
    >> "$TEACHER_DIR/train.log" 2>&1
rc=$?
echo "[$(date)] stage 1 exited rc=$rc" >> "$LOG"
if [ ! -f "$TEACHER_DIR/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: teacher epoch_10.pth missing — abort" >> "$LOG"
    exit 1
fi

# ---------- Stage 2: dump pseudo on 5% s1 unlabeled ---------- #
echo "[$(date)] stage 2: pseudo dump on 5p_s1 unlabeled" >> "$LOG"
cd /root/BMVC2026
CUDA_VISIBLE_DEVICES=${GPU_TEACHER} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/g2_dump_pseudo.py \
    --config "$TEACHER_CFG" \
    --ckpt "$TEACHER_DIR/epoch_10.pth" \
    --unlabeled-pkl "$SOURCE_INFOS" \
    --out "$RAW_PSEUDO" \
    >> "$LOG" 2>&1
rc=$?
echo "[$(date)] stage 2 exited rc=$rc" >> "$LOG"
if [ ! -f "$RAW_PSEUDO" ]; then
    echo "[$(date)] FATAL: raw pseudo missing — abort" >> "$LOG"
    exit 1
fi

# ---------- Stage 3a: perclass_boost merge ---------- #
echo "[$(date)] stage 3a: perclass_boost filter+merge" >> "$LOG"
/root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
    --raw-pseudo "$RAW_PSEUDO" \
    --unlabeled-infos "$SOURCE_INFOS" \
    --det-cache "$DET_CACHE" \
    --variant perclass_boost \
    --out-pkl "$PSEUDO_BOOST" \
    >> "$LOG" 2>&1
echo "[$(date)] stage 3a exited rc=$?" >> "$LOG"

# ---------- Stage 3b: perclass_only merge ---------- #
echo "[$(date)] stage 3b: perclass_only filter+merge" >> "$LOG"
/root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
    --raw-pseudo "$RAW_PSEUDO" \
    --unlabeled-infos "$SOURCE_INFOS" \
    --det-cache "$DET_CACHE" \
    --variant perclass_only \
    --out-pkl "$PSEUDO_ONLY" \
    >> "$LOG" 2>&1
echo "[$(date)] stage 3b exited rc=$?" >> "$LOG"

if [ ! -f "$PSEUDO_BOOST" ] || [ ! -f "$PSEUDO_ONLY" ]; then
    echo "[$(date)] FATAL: pseudo infos missing" >> "$LOG"
    exit 1
fi

# ---------- Stage 4: train R7-s1 (boost) on GPU 0 + R7b-s1 (only) on GPU 1 in parallel ---------- #
echo "[$(date)] stage 4: parallel SSL training" >> "$LOG"
WD_BOOST=/root/BMVC2026/work_dirs/R7_ssl_5p_s1_perclass_boost
WD_ONLY=/root/BMVC2026/work_dirs/R7b_ssl_5p_s1_perclass_only
CFG_BOOST=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_perclass.py
CFG_ONLY=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_perclass_only.py
mkdir -p "$WD_BOOST" "$WD_ONLY"

cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU_BOOST} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$CFG_BOOST" --work-dir "$WD_BOOST" \
    >> "$WD_BOOST/train.log" 2>&1 &
PID_BOOST=$!
echo "[$(date)] launched R7-s1 PID=$PID_BOOST GPU=$GPU_BOOST" >> "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ONLY} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$CFG_ONLY" --work-dir "$WD_ONLY" \
    >> "$WD_ONLY/train.log" 2>&1 &
PID_ONLY=$!
echo "[$(date)] launched R7b-s1 PID=$PID_ONLY GPU=$GPU_ONLY" >> "$LOG"

wait $PID_BOOST
echo "[$(date)] R7-s1 train exited rc=$?" >> "$LOG"
wait $PID_ONLY
echo "[$(date)] R7b-s1 train exited rc=$?" >> "$LOG"

# ---------- Stage 5: external eval both ---------- #
echo "[$(date)] stage 5: external eval R7-s1 + R7b-s1" >> "$LOG"
PYTHON=/root/miniconda3/envs/teocc/bin/python
eval_one () {
    local tag="$1"; local cfg="$2"; local ckpt="$3"; local gpu="$4"
    local out_dir="/root/BMVC2026/work_dirs/${tag}_external_eval"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/root/BMVC2026 \
        "$PYTHON" tools/test.py "$cfg" "$ckpt" \
        --eval bbox \
        --eval-options "jsonfile_prefix=${out_dir}/results" \
        > "${out_dir}/eval.log" 2>&1
    echo "[$(date)] eval ${tag} rc=$?" >> "$LOG"
}

if [ -f "$WD_BOOST/epoch_10.pth" ]; then
    eval_one R7_s1   "$CFG_BOOST" "$WD_BOOST/epoch_10.pth" "$GPU_BOOST"
fi
if [ -f "$WD_ONLY/epoch_10.pth" ]; then
    eval_one R7b_s1  "$CFG_ONLY"  "$WD_ONLY/epoch_10.pth"  "$GPU_ONLY"
fi

echo "[$(date)] R7-s1 pipeline complete" >> "$LOG"
