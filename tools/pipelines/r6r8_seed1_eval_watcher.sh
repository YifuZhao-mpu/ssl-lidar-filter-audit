#!/bin/bash
# Watcher: wait for the 4 family-s1 epoch_10.pth files, then run external eval
# on each. The original pipeline's wait/eval phase mis-fired due to a subshell
# bug; this watcher cleans up.
set -u
LOG=/root/BMVC2026/scripts/r6r8_seed1_eval_watcher.log
echo "[$(date)] watcher start" > "$LOG"

declare -A CFG WD
CFG[R6_s1]=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_single_thr.py
WD[R6_s1]=/root/BMVC2026/work_dirs/R6_ssl_5p_s1_single_thr

CFG[R8b_s1]=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_best_view.py
WD[R8b_s1]=/root/BMVC2026/work_dirs/R8b_ssl_5p_s1_best_view

CFG[R8c_s1]=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_any_view.py
WD[R8c_s1]=/root/BMVC2026/work_dirs/R8c_ssl_5p_s1_any_view

CFG[R8d_s1]=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_max_cam.py
WD[R8d_s1]=/root/BMVC2026/work_dirs/R8d_ssl_5p_s1_max_cam

declare -A GPU
GPU[R6_s1]=0
GPU[R8b_s1]=1
GPU[R8c_s1]=4
GPU[R8d_s1]=5

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc
cd /root/TEOcc

eval_one () {
    local tag="$1"
    local out_dir="/root/BMVC2026/work_dirs/${tag}_external_eval"
    rm -rf "$out_dir"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=${GPU[$tag]} PYTHONPATH=/root/BMVC2026 \
        /root/miniconda3/envs/teocc/bin/python tools/test.py \
        "${CFG[$tag]}" "${WD[$tag]}/epoch_10.pth" \
        --eval bbox \
        --eval-options "jsonfile_prefix=${out_dir}/results" \
        > "${out_dir}/eval.log" 2>&1
    echo "[$(date)] eval ${tag} rc=$?" >> "$LOG"
}

# Wait for each ckpt then immediately eval (parallel-friendly: as each finishes
# training, its GPU frees up and the eval runs on the same GPU).
for tag in R6_s1 R8b_s1 R8c_s1 R8d_s1; do
    until [ -f "${WD[$tag]}/epoch_10.pth" ]; do sleep 60; done
    echo "[$(date)] ${tag} ckpt ready, evaluating..." >> "$LOG"
    eval_one "$tag" &
done
wait
echo "[$(date)] watcher done" >> "$LOG"
