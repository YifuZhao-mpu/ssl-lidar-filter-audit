#!/bin/bash
# Wait for max_cam@K=5 filter rerun → re-merge → launch R8d on GPU 5.
set -u
LOG=/root/BMVC2026/scripts/r8d_launch_max_cam_k5.log
echo "[$(date)] start" > "$LOG"

until grep -q "Metrics →" /root/BMVC2026/cache/logs/R8_filter_max_cam_k5.log 2>/dev/null; do
    sleep 15
done
echo "[$(date)] filter K=5 complete" >> "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc
cd /root/BMVC2026

python tools/g2_make_pseudo_infos.py \
    --raw-pseudo /root/BMVC2026/cache/pseudo_5p_s0_filtered/max_cam.pkl \
    --unlabeled-infos /root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl \
    --variant passthrough \
    --out-pkl /root/BMVC2026/infos/nuscenes_pseudo_5p_s0_max_cam.pkl \
    >> "$LOG" 2>&1
echo "[$(date)] max_cam@K=5 merge done" >> "$LOG"

WD=/root/BMVC2026/work_dirs/R8d_ssl_5p_s0_max_cam
# Wipe stale partial artifacts from the killed K=50 run
rm -rf "$WD"
mkdir -p "$WD"

cd /root/TEOcc
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/root/BMVC2026 \
    /root/miniconda3/envs/teocc/bin/python tools/train.py \
    /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_max_cam.py \
    --work-dir "$WD" \
    >> "$WD/train.log" 2>&1 &
echo "[$(date)] launched R8d (K=5) on GPU 5 pid=$!" >> "$LOG"
wait
echo "[$(date)] R8d exited" >> "$LOG"
