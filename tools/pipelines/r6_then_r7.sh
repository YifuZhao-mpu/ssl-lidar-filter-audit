#!/bin/bash
# Chain script: wait for R6 (PID 3962212) to finish, then launch R7 sequentially on GPU 6.
# Started: 2026-04-25 09:37
set -u

R6_PID=3962212
R6_DIR=/root/BMVC2026/work_dirs/R6_ssl_5p_s0_baseline
R7_DIR=/root/BMVC2026/work_dirs/R7_ssl_5p_s0_perclass
R7_CFG=/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_perclass.py
LOG=/root/BMVC2026/scripts/r6_then_r7.log

mkdir -p "$R7_DIR"

echo "[$(date)] waiting for R6 PID=$R6_PID" >> "$LOG"
while kill -0 "$R6_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] R6 PID exited" >> "$LOG"

# Verify R6 produced epoch_10.pth before launching R7
if [ -f "$R6_DIR/epoch_10.pth" ]; then
    echo "[$(date)] R6 epoch_10.pth present — launching R7" >> "$LOG"
    cd /root/TEOcc
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/root/BMVC2026 \
        /root/miniconda3/envs/teocc/bin/python tools/train.py \
        "$R7_CFG" --work-dir "$R7_DIR" \
        >> "$R7_DIR/train.log" 2>&1
    echo "[$(date)] R7 exited rc=$?" >> "$LOG"
else
    echo "[$(date)] R6 epoch_10.pth MISSING — aborting R7 launch" >> "$LOG"
    ls -la "$R6_DIR" >> "$LOG"
fi
