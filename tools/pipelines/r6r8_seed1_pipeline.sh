#!/bin/bash
# Full family @ seed 1: R6-s1 single-thr, R8b-s1 best-view, R8c-s1 any-view,
# R8d-s1 max-cam K=5. Codex Round 1+2 ask: upgrade D2 from "single-seed
# full-family + 1-filter seed-1 confirmation" to "two-seed full-family".
set -u
LOG=/root/BMVC2026/scripts/r6r8_seed1_pipeline.log
RAW_PSEUDO=/root/BMVC2026/cache/pseudo_5p_s1_raw_full.pkl
DET_CACHE=/root/BMVC2026/cache/fasterrcnn_v2_train.pkl
SOURCE_INFOS=/root/BMVC2026/infos/nuscenes_infos_train_5p_s1_unlabeled.pkl
FILTER_OUTDIR=/root/BMVC2026/cache/pseudo_5p_s1_filtered

mkdir -p "$FILTER_OUTDIR"
echo "[$(date)] full-family s1 pipeline start" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

# ---------- Stage 1: regen filter variants on 5% s1 raw pseudo ---------- #
echo "[$(date)] stage 1: filter regen (best_view, any_view, max_cam)" >> "$LOG"
cd /root/BMVC2026
/root/miniconda3/envs/teocc/bin/python tools/g2_filter_pseudo.py \
    --raw-pseudo "$RAW_PSEUDO" \
    --det-cache "$DET_CACHE" \
    --source-infos "$SOURCE_INFOS" \
    --out-dir "$FILTER_OUTDIR" \
    --variants best_view any_view max_cam \
    --max-cam-topk 5 \
    >> "$LOG" 2>&1
echo "[$(date)] stage 1 exited rc=$?" >> "$LOG"

# ---------- Stage 2: build infos PKLs for all 4 variants ---------- #
declare -A VARIANT_PKL
VARIANT_PKL[single_thr]=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_single_thr.pkl
VARIANT_PKL[best_view]=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_best_view.pkl
VARIANT_PKL[any_view]=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_any_view.pkl
VARIANT_PKL[max_cam]=/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_max_cam.pkl

echo "[$(date)] stage 2: build infos PKLs" >> "$LOG"
# single_thr — direct from raw pseudo
if [ ! -f "${VARIANT_PKL[single_thr]}" ]; then
    /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$RAW_PSEUDO" \
        --unlabeled-infos "$SOURCE_INFOS" \
        --variant single_thr \
        --single-thr 0.40 \
        --out-pkl "${VARIANT_PKL[single_thr]}" >> "$LOG" 2>&1
fi
# best_view, any_view, max_cam — passthrough merge of filtered pkls
for v in best_view any_view max_cam; do
    if [ ! -f "${VARIANT_PKL[$v]}" ]; then
        /root/miniconda3/envs/teocc/bin/python tools/g2_make_pseudo_infos.py \
            --raw-pseudo "${FILTER_OUTDIR}/${v}.pkl" \
            --unlabeled-infos "$SOURCE_INFOS" \
            --variant passthrough \
            --out-pkl "${VARIANT_PKL[$v]}" >> "$LOG" 2>&1
    fi
done
echo "[$(date)] stage 2 done" >> "$LOG"

# ---------- Stage 3: 4 SSL trainings in parallel ---------- #
echo "[$(date)] stage 3: launching 4 parallel SSL trainings" >> "$LOG"
cd /root/TEOcc

launch_ssl () {
    local tag="$1"; local gpu="$2"; local cfg="$3"
    local wd="/root/BMVC2026/work_dirs/${tag}"
    mkdir -p "$wd"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/root/BMVC2026 \
        /root/miniconda3/envs/teocc/bin/python tools/train.py \
        "$cfg" --work-dir "$wd" \
        >> "$wd/train.log" 2>&1 &
    local pid=$!
    echo "[$(date)] launched ${tag} on GPU ${gpu} pid=${pid}" >> "$LOG"
    echo $pid
}

PID_R6=$(launch_ssl  R6_ssl_5p_s1_single_thr  0  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_single_thr.py)
PID_R8B=$(launch_ssl R8b_ssl_5p_s1_best_view  1  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_best_view.py)
PID_R8C=$(launch_ssl R8c_ssl_5p_s1_any_view   4  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_any_view.py)
PID_R8D=$(launch_ssl R8d_ssl_5p_s1_max_cam    5  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_max_cam.py)

wait $PID_R6
echo "[$(date)] R6-s1 train exited rc=$?" >> "$LOG"
wait $PID_R8B
echo "[$(date)] R8b-s1 train exited rc=$?" >> "$LOG"
wait $PID_R8C
echo "[$(date)] R8c-s1 train exited rc=$?" >> "$LOG"
wait $PID_R8D
echo "[$(date)] R8d-s1 train exited rc=$?" >> "$LOG"

# ---------- Stage 4: external eval all 4 ---------- #
echo "[$(date)] stage 4: external eval" >> "$LOG"
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

eval_one R6_s1   /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_single_thr.py /root/BMVC2026/work_dirs/R6_ssl_5p_s1_single_thr/epoch_10.pth   0 &
eval_one R8b_s1  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_best_view.py  /root/BMVC2026/work_dirs/R8b_ssl_5p_s1_best_view/epoch_10.pth  1 &
eval_one R8c_s1  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_any_view.py   /root/BMVC2026/work_dirs/R8c_ssl_5p_s1_any_view/epoch_10.pth   4 &
eval_one R8d_s1  /root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s1_max_cam.py    /root/BMVC2026/work_dirs/R8d_ssl_5p_s1_max_cam/epoch_10.pth    5 &
wait
echo "[$(date)] full-family s1 pipeline complete" >> "$LOG"
