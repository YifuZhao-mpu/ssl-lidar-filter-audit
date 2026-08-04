#!/bin/bash
# Run external eval on R7b/R8b/R8c/R8d epoch_10.pth (raw, not EMA) to match
# the R6/R7 protocol in the tracker. val_full=6019, patched _format_bbox.
set -u
cd /root/TEOcc
PYTHON=/root/miniconda3/envs/teocc/bin/python

eval_one () {
    local tag="$1"
    local gpu="$2"
    local cfg="$3"
    local ckpt="$4"
    local out_dir="/root/BMVC2026/work_dirs/${tag}_external_eval"
    mkdir -p "$out_dir"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/root/BMVC2026 \
        "$PYTHON" tools/test.py "$cfg" "$ckpt" \
        --eval bbox \
        --eval-options "jsonfile_prefix=${out_dir}/results" \
        > "${out_dir}/eval.log" 2>&1 &
    echo "launched ${tag} eval on GPU ${gpu} pid=$!"
}

eval_one R7b 4 \
  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_perclass_only.py \
  /root/BMVC2026/work_dirs/R7b_ssl_5p_s0_perclass_only/epoch_10.pth

eval_one R8b 6 \
  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_best_view.py \
  /root/BMVC2026/work_dirs/R8b_ssl_5p_s0_best_view/epoch_10.pth

eval_one R8c 7 \
  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_any_view.py \
  /root/BMVC2026/work_dirs/R8c_ssl_5p_s0_any_view/epoch_10.pth

eval_one R8d 3 \
  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_max_cam.py \
  /root/BMVC2026/work_dirs/R8d_ssl_5p_s0_max_cam/epoch_10.pth

wait
echo "all evals exited"
