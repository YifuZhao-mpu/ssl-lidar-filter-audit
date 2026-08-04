#!/bin/bash
# Package B pipeline (WACV 2027 E&D redirect) — see PACKAGE_B_PROTOCOL.md.
#   Phase A1 [GPU4]: s2 teacher (10 ep) -> s2 pseudo dump -> teacher external eval
#   Phase A2 [CPU]:  hybrid_posonly infos from existing s0 dump -> queue R18-s0
#   Phase B  [CPU]:  all 8 s2 filter/infos variants -> queue 8 s2 SSL jobs
#   Workers  [GPU3/4/5]: pop queue -> train (10 ep) -> external eval -> repeat
# Launched 2026-07-16. GPUs 3/4/5 only (0/2/6/7 occupied by others, 1 has residual).
set -u
ROOT=/root/BMVC2026
LOG=$ROOT/scripts/pkgB_pipeline.log
QUEUE=$ROOT/scripts/pkgB_queue.txt
LOCK=$ROOT/scripts/pkgB_queue.lock
ALL_PUSHED_FLAG=$ROOT/scripts/pkgB_all_pushed.flag
PYTHON=/root/miniconda3/envs/teocc/bin/python

TEACHER_CFG=$ROOT/configs/g2_sup_centerpoint_01voxel_nus_5p_s2.py
TEACHER_DIR=$ROOT/work_dirs/R4_sup_5p_s2
RAW_S2=$ROOT/cache/pseudo_5p_s2_raw_full.pkl
RAW_S0=$ROOT/cache/pseudo_5p_s0_raw_full.pkl
UNLAB_S2=$ROOT/infos/nuscenes_infos_train_5p_s2_unlabeled.pkl
UNLAB_S0=$ROOT/infos/nuscenes_infos_train_5p_s0_unlabeled.pkl
DET_CACHE=$ROOT/cache/fasterrcnn_v2_train.pkl
FILTER_OUTDIR=$ROOT/cache/pseudo_5p_s2_filtered

rm -f "$QUEUE" "$ALL_PUSHED_FLAG"
touch "$QUEUE" "$LOCK"
echo "[$(date)] Package B pipeline start" > "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

push_job () {  # $1=tag $2=cfg
    ( flock 9; echo "$1|$2" >> "$QUEUE" ) 9>"$LOCK"
    echo "[$(date)] queued: $1" >> "$LOG"
}

pop_job () {
    ( flock 9
      line=$(head -n1 "$QUEUE")
      if [ -n "$line" ]; then sed -i '1d' "$QUEUE"; fi
      echo "$line"
    ) 9>"$LOCK"
}

worker () {
    local gpu=$1
    echo "[$(date)] worker gpu${gpu} started" >> "$LOG"
    while true; do
        local job
        job=$(pop_job)
        if [ -z "$job" ]; then
            if [ -f "$ALL_PUSHED_FLAG" ]; then break; fi
            sleep 60
            continue
        fi
        local tag=${job%%|*}
        local cfg=${job##*|}
        local wd=$ROOT/work_dirs/$tag
        mkdir -p "$wd"
        echo "[$(date)] worker gpu${gpu} TRAIN start: $tag" >> "$LOG"
        ( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT \
            $PYTHON tools/train.py "$cfg" --work-dir "$wd" >> "$wd/train.log" 2>&1 )
        local rc=$?
        echo "[$(date)] worker gpu${gpu} TRAIN done: $tag rc=$rc" >> "$LOG"
        if [ ! -f "$wd/epoch_10.pth" ]; then
            echo "[$(date)] FATAL: $tag epoch_10.pth missing - skipping eval" >> "$LOG"
            continue
        fi
        local ed=$ROOT/work_dirs/${tag}_external_eval
        mkdir -p "$ed"
        echo "[$(date)] worker gpu${gpu} EVAL start: $tag" >> "$LOG"
        ( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT \
            $PYTHON tools/test.py "$cfg" "$wd/epoch_10.pth" --eval bbox \
            --eval-options "jsonfile_prefix=$ed/results" > "$ed/eval.log" 2>&1 )
        echo "[$(date)] worker gpu${gpu} EVAL done: $tag rc=$?" >> "$LOG"
    done
    echo "[$(date)] worker gpu${gpu} exit" >> "$LOG"
}

# ---------- lane workers on GPU 3 and 5 start immediately ---------- #
worker 3 & W3=$!
worker 5 & W5=$!

# ---------- Phase A2 (CPU): hybrid-s0 infos from existing s0 dump ---------- #
(
    $PYTHON $ROOT/tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$RAW_S0" \
        --unlabeled-infos "$UNLAB_S0" \
        --det-cache "$DET_CACHE" \
        --variant hybrid_posonly \
        --out-pkl $ROOT/infos/nuscenes_pseudo_5p_s0_hybrid_posonly.pkl \
        >> "$LOG" 2>&1
    rc=$?
    echo "[$(date)] A2 hybrid-s0 infos rc=$rc" >> "$LOG"
    if [ $rc -eq 0 ] && [ -f $ROOT/infos/nuscenes_pseudo_5p_s0_hybrid_posonly.pkl ]; then
        push_job R18_ssl_5p_s0_hybrid_posonly $ROOT/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_hybrid_posonly.py
    else
        echo "[$(date)] FATAL: A2 hybrid-s0 infos failed" >> "$LOG"
    fi
) &
A2_PID=$!

# ---------- Phase A1 (GPU4): s2 teacher ---------- #
echo "[$(date)] A1 teacher training start (GPU4)" >> "$LOG"
mkdir -p "$TEACHER_DIR"
( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$ROOT \
    $PYTHON tools/train.py "$TEACHER_CFG" --work-dir "$TEACHER_DIR" \
    >> "$TEACHER_DIR/train.log" 2>&1 )
echo "[$(date)] A1 teacher rc=$?" >> "$LOG"
if [ ! -f "$TEACHER_DIR/epoch_10.pth" ]; then
    echo "[$(date)] FATAL: teacher epoch_10.pth missing - abort" >> "$LOG"
    touch "$ALL_PUSHED_FLAG"
    exit 1
fi

echo "[$(date)] A1 pseudo dump start (GPU4)" >> "$LOG"
( cd $ROOT && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$ROOT \
    $PYTHON tools/g2_dump_pseudo.py \
    --config "$TEACHER_CFG" \
    --ckpt "$TEACHER_DIR/epoch_10.pth" \
    --unlabeled-pkl "$UNLAB_S2" \
    --out "$RAW_S2" >> "$LOG" 2>&1 )
echo "[$(date)] A1 dump rc=$?" >> "$LOG"
if [ ! -f "$RAW_S2" ]; then
    echo "[$(date)] FATAL: raw s2 pseudo missing - abort" >> "$LOG"
    touch "$ALL_PUSHED_FLAG"
    exit 1
fi

# Teacher external eval on GPU4, concurrent with Phase B (CPU)
(
    ed=$ROOT/work_dirs/R4_s2_external_eval
    mkdir -p "$ed"
    cd /root/TEOcc && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$ROOT \
        $PYTHON tools/test.py "$TEACHER_CFG" "$TEACHER_DIR/epoch_10.pth" \
        --eval bbox --eval-options "jsonfile_prefix=$ed/results" \
        > "$ed/eval.log" 2>&1
    echo "[$(date)] A1 teacher external eval rc=$?" >> "$LOG"
) &
TEVAL_PID=$!

# ---------- Phase B (CPU): all 8 s2 filter/infos variants ---------- #
echo "[$(date)] B camera-variant filtering start" >> "$LOG"
mkdir -p "$FILTER_OUTDIR"
$PYTHON $ROOT/tools/g2_filter_pseudo.py \
    --raw-pseudo "$RAW_S2" \
    --det-cache "$DET_CACHE" \
    --source-infos "$UNLAB_S2" \
    --out-dir "$FILTER_OUTDIR" \
    --variants best_view any_view max_cam consensus \
    --max-cam-topk 5 >> "$LOG" 2>&1
echo "[$(date)] B camera-filters rc=$?" >> "$LOG"

# score-based + hybrid variants
$PYTHON $ROOT/tools/g2_make_pseudo_infos.py \
    --raw-pseudo "$RAW_S2" --unlabeled-infos "$UNLAB_S2" \
    --variant single_thr --single-thr 0.40 \
    --out-pkl $ROOT/infos/nuscenes_pseudo_5p_s2_single_thr.pkl >> "$LOG" 2>&1
echo "[$(date)] B single_thr rc=$?" >> "$LOG"

for VAR in perclass_boost perclass_only hybrid_posonly; do
    $PYTHON $ROOT/tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$RAW_S2" --unlabeled-infos "$UNLAB_S2" \
        --det-cache "$DET_CACHE" \
        --variant $VAR \
        --out-pkl $ROOT/infos/nuscenes_pseudo_5p_s2_${VAR}.pkl >> "$LOG" 2>&1
    echo "[$(date)] B $VAR rc=$?" >> "$LOG"
done

# passthrough merges for camera variants
for VAR in best_view any_view max_cam consensus; do
    $PYTHON $ROOT/tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$FILTER_OUTDIR/${VAR}.pkl" \
        --unlabeled-infos "$UNLAB_S2" \
        --variant passthrough \
        --out-pkl $ROOT/infos/nuscenes_pseudo_5p_s2_${VAR}.pkl >> "$LOG" 2>&1
    echo "[$(date)] B merge $VAR rc=$?" >> "$LOG"
done

# verify all 8 pseudo infos exist
MISSING=0
for F in single_thr perclass_boost perclass_only hybrid_posonly best_view any_view max_cam consensus; do
    if [ ! -f $ROOT/infos/nuscenes_pseudo_5p_s2_${F}.pkl ]; then
        echo "[$(date)] FATAL: missing nuscenes_pseudo_5p_s2_${F}.pkl" >> "$LOG"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo "[$(date)] FATAL: phase B incomplete - queue only what exists" >> "$LOG"
fi

# ---------- queue s2 SSL jobs (family-gap anchors first) ---------- #
CFGD=$ROOT/configs
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_perclass_only.pkl ] && push_job R7b_ssl_5p_s2_perclass_only $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_perclass_only.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_consensus.pkl ]     && push_job R7c_ssl_5p_s2_consensus     $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_consensus.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_single_thr.pkl ]    && push_job R6_ssl_5p_s2_single_thr     $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_single_thr.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_best_view.pkl ]     && push_job R8b_ssl_5p_s2_best_view     $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_best_view.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_any_view.pkl ]      && push_job R8c_ssl_5p_s2_any_view      $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_any_view.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_max_cam.pkl ]       && push_job R8d_ssl_5p_s2_max_cam       $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_max_cam.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_perclass_boost.pkl ] && push_job R7_ssl_5p_s2_perclass_boost $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_perclass.py
[ -f $ROOT/infos/nuscenes_pseudo_5p_s2_hybrid_posonly.pkl ] && push_job R18_ssl_5p_s2_hybrid_posonly $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_hybrid_posonly.py
touch "$ALL_PUSHED_FLAG"
echo "[$(date)] all jobs pushed" >> "$LOG"

# GPU4 joins the worker pool once its teacher external eval finishes
wait $TEVAL_PID
worker 4 & W4=$!

wait $W3 $W5 $W4
wait $A2_PID 2>/dev/null
echo "[$(date)] Package B pipeline complete" >> "$LOG"
