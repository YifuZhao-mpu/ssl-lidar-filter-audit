#!/bin/bash
# R10-s1 pipeline: 10% s1 teacher → pseudo dump → perclass_only filter+merge →
# SSL student → external eval both. Codex round-4 ask: paired replication of
# R10a/R10b on a different seed/split to test the "lucky teacher" hypothesis.
set -u
LOG=/root/BMVC2026/scripts/r10_seed1_pipeline.log
TEACHER_DIR=/root/BMVC2026/work_dirs/R10a_sup_10p_s1
STUDENT_DIR=/root/BMVC2026/work_dirs/R10b_ssl_10p_s1_perclass_only
TEACHER_CFG=/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_10p_s1.py
STUDENT_CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_10p_s1_perclass_only.py
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_10p_s1_raw_full.pkl
PSEUDO_INFOS=/root/BMVC2026/infos/nuscenes_pseudo_10p_s1_perclass_only.pkl
SOURCE_INFOS=/root/BMVC2026/infos/nuscenes_infos_train_10p_s1_unlabeled.pkl
DET_CACHE=/root/BMVC2026/cache/fasterrcnn_v2_train.pkl
GPU=6

mkdir -p "$TEACHER_DIR" "$STUDENT_DIR"
echo "[$(date)] R10-s1 pipeline start, GPU=${GPU}" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

echo "[$(date)] stage 1: training R10a-s1 teacher" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$TEACHER_CFG" --work-dir "$TEACHER_DIR" \
    >> "$TEACHER_DIR/train.log" 2>&1
echo "[$(date)] stage 1 exited rc=$?" >> "$LOG"
[ ! -f "$TEACHER_DIR/epoch_10.pth" ] && { echo "FATAL teacher missing" >> "$LOG"; exit 1; }

echo "[$(date)] stage 2: pseudo dump on 10p_s1 unlabeled" >> "$LOG"
cd /root/BMVC2026
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/g2_dump_pseudo.py \
    --config "$TEACHER_CFG" \
    --ckpt "$TEACHER_DIR/epoch_10.pth" \
    --unlabeled-pkl "$SOURCE_INFOS" \
    --out "$RAW_PSEUDO" >> "$LOG" 2>&1
echo "[$(date)] stage 2 exited rc=$?" >> "$LOG"
[ ! -f "$RAW_PSEUDO" ] && { echo "FATAL raw pseudo missing" >> "$LOG"; exit 1; }

echo "[$(date)] stage 3: perclass_only filter+merge" >> "$LOG"
/root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
    --raw-pseudo "$RAW_PSEUDO" \
    --unlabeled-infos "$SOURCE_INFOS" \
    --det-cache "$DET_CACHE" \
    --variant perclass_only \
    --out-pkl "$PSEUDO_INFOS" >> "$LOG" 2>&1
echo "[$(date)] stage 3 exited rc=$?" >> "$LOG"
[ ! -f "$PSEUDO_INFOS" ] && { echo "FATAL pseudo infos missing" >> "$LOG"; exit 1; }

echo "[$(date)] stage 4: training R10b-s1 SSL student" >> "$LOG"
cd /root/TEOcc
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    "$STUDENT_CFG" --work-dir "$STUDENT_DIR" \
    >> "$STUDENT_DIR/train.log" 2>&1
echo "[$(date)] stage 4 exited rc=$?" >> "$LOG"
[ ! -f "$STUDENT_DIR/epoch_10.pth" ] && { echo "FATAL student missing" >> "$LOG"; exit 1; }

echo "[$(date)] stage 5: external eval both" >> "$LOG"
PYTHON=/root/miniconda3/envs/teocc/bin/python
eval_one () {
    local tag="$1"; local cfg="$2"; local ckpt="$3"
    local out_dir="/root/BMVC2026/work_dirs/${tag}_external_eval"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/root/BMVC2026 \
        "$PYTHON" tools/test.py "$cfg" "$ckpt" \
        --eval bbox \
        --eval-options "jsonfile_prefix=${out_dir}/results" \
        > "${out_dir}/eval.log" 2>&1
    echo "[$(date)] eval ${tag} rc=$?" >> "$LOG"
}
eval_one R10a_s1 "$TEACHER_CFG" "$TEACHER_DIR/epoch_10.pth"
eval_one R10b_s1 "$STUDENT_CFG" "$STUDENT_DIR/epoch_10.pth"
echo "[$(date)] R10-s1 pipeline complete" >> "$LOG"
