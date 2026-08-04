#!/bin/bash
# Coordinator: after filter rerun completes, re-merge max_cam (the only variant
# whose content changed in v2), then launch R8b/c/d in parallel on GPUs 0/4/5.
# best_view and any_view infos PKLs from the v1 filter are byte-identical to
# what v2 produces (filter logic unchanged for those variants), so no re-merge.
set -u
LOGDIR=/root/BMVC2026/scripts
mkdir -p "$LOGDIR"
COORD_LOG="$LOGDIR/r8_launch_alt_pack.log"

echo "[$(date)] coordinator start" > "$COORD_LOG"

# Wait for filter rerun to complete
until grep -q "Metrics →" /root/BMVC2026/cache/logs/R8_filter_alt_v2.log 2>/dev/null; do
    sleep 15
done
echo "[$(date)] filter v2 complete" >> "$COORD_LOG"

# Re-merge max_cam with the corrected content
source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc
cd /root/BMVC2026

python tools/g2_make_pseudo_infos.py \
    --raw-pseudo /root/BMVC2026/cache/pseudo_5p_s0_filtered/max_cam.pkl \
    --unlabeled-infos /root/BMVC2026/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl \
    --variant passthrough \
    --out-pkl /root/BMVC2026/infos/nuscenes_pseudo_5p_s0_max_cam.pkl \
    >> "$COORD_LOG" 2>&1
echo "[$(date)] max_cam merge done" >> "$COORD_LOG"

# Launch trainings — one per GPU, all in background
launch_r8 () {
    local tag="$1"
    local gpu="$2"
    local cfg="$3"
    local wd="/root/BMVC2026/work_dirs/${tag}"
    mkdir -p "$wd"
    cd /root/TEOcc
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/root/BMVC2026 \
        /root/miniconda3/envs/teocc/bin/python tools/train.py \
        "$cfg" --work-dir "$wd" \
        >> "$wd/train.log" 2>&1 &
    local pid=$!
    echo "[$(date)] launched ${tag} on GPU ${gpu} pid=${pid}" >> "$COORD_LOG"
    cd /root/BMVC2026
}

launch_r8 R8b_ssl_5p_s0_best_view  0  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_best_view.py
launch_r8 R8c_ssl_5p_s0_any_view   4  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_any_view.py
launch_r8 R8d_ssl_5p_s0_max_cam    5  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_max_cam.py

echo "[$(date)] all R8 alt students launched" >> "$COORD_LOG"
wait
echo "[$(date)] all R8 alt students exited" >> "$COORD_LOG"
