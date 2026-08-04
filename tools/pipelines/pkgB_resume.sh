#!/bin/bash
# Package B resume after the 2026-07-16 /root -> /workspace migration killed
# wave-1 trainings. Teacher/dump/8 pseudo-infos survived; this script only
# re-runs the 9 SSL jobs. Paths go through the /root symlink shims so the
# original configs work verbatim.
#
# Usage:
#   bash pkgB_resume.sh main        # verify assets, build queue, spawn workers on $GPUS
#   bash pkgB_resume.sh worker N    # attach one extra worker on GPU N (same queue)
set -u
ROOT=/root/BMVC2026
LOG=$ROOT/scripts/pkgB_pipeline.log      # append to the ORIGINAL log (monitor follows it)
QUEUE=$ROOT/scripts/pkgB_queue.txt
LOCK=$ROOT/scripts/pkgB_queue.lock
ALL_PUSHED_FLAG=$ROOT/scripts/pkgB_all_pushed.flag
PIDFILE=$ROOT/scripts/pkgB_resume.pid
PYTHON=/root/miniconda3/envs/teocc/bin/python
CFGD=$ROOT/configs
GPUS="4 5"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate teocc

push_job () {  # $1=tag $2=cfg $3=extra train args (may be empty)
    # Emit 2-field lines when extra is empty: the surviving pre-migration
    # workers parse cfg as ${job##*|} (last field), so a trailing "|" hands
    # them an empty config path and they insta-fail. 2-field lines parse
    # identically under both worker generations.
    ( flock 9; echo "$1|$2${3:+|$3}" >> "$QUEUE" ) 9>"$LOCK"
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
    echo "[$(date)] worker gpu${gpu} started (resume)" >> "$LOG"
    while true; do
        local job
        job=$(pop_job)
        if [ -z "$job" ]; then
            if [ -f "$ALL_PUSHED_FLAG" ]; then break; fi
            sleep 60
            continue
        fi
        local tag=$(echo "$job" | cut -d'|' -f1)
        local cfg=$(echo "$job" | cut -d'|' -f2)
        local extra=$(echo "$job" | cut -d'|' -f3)
        local wd=$ROOT/work_dirs/$tag
        mkdir -p "$wd"
        echo "[$(date)] worker gpu${gpu} TRAIN start: $tag ${extra:+(extra: $extra)}" >> "$LOG"
        ( cd /root/TEOcc && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$ROOT \
            $PYTHON tools/train.py "$cfg" --work-dir "$wd" $extra >> "$wd/train.log" 2>&1 )
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

if [ "${1:-main}" = "worker" ]; then
    worker "$2"
    exit 0
fi

# ---------------- main ---------------- #
echo "[$(date)] Package B RESUME start (post /root->/workspace migration)" >> "$LOG"
echo $$ > "$PIDFILE"

# Verify surviving assets before queueing anything
FAIL=0
for F in \
    "$ROOT/work_dirs/R4_sup_5p_s2/epoch_10.pth" \
    "$ROOT/infos/nuscenes_pseudo_5p_s0_hybrid_posonly.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_single_thr.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_perclass_boost.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_perclass_only.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_hybrid_posonly.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_best_view.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_any_view.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_max_cam.pkl" \
    "$ROOT/infos/nuscenes_pseudo_5p_s2_consensus.pkl"; do
    if [ ! -f "$F" ]; then echo "[$(date)] FATAL: missing asset $F" >> "$LOG"; FAIL=1; fi
done
[ $FAIL -eq 1 ] && { echo "[$(date)] FATAL: asset check failed - not queueing" >> "$LOG"; exit 1; }

# Archive dead partial runs (R7b/R7c restart clean; R18-s0 resumes from epoch_2)
for TAG in R7b_ssl_5p_s2_perclass_only R7c_ssl_5p_s2_consensus; do
    WD=$ROOT/work_dirs/$TAG
    [ -f "$WD/train.log" ] && mv "$WD/train.log" "$WD/train.log.killed_by_move_20260716"
    rm -f "$WD"/epoch_*_ema.pth "$WD"/latest.pth
done

rm -f "$QUEUE" "$ALL_PUSHED_FLAG"
touch "$QUEUE" "$LOCK"

# Queue: balanced lanes; R18-s0 resume first (least remaining), anchors next.
push_job R18_ssl_5p_s0_hybrid_posonly $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s0_hybrid_posonly.py "--resume-from $ROOT/work_dirs/R18_ssl_5p_s0_hybrid_posonly/latest.pth"
push_job R7b_ssl_5p_s2_perclass_only  $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_perclass_only.py ""
push_job R7c_ssl_5p_s2_consensus      $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_consensus.py ""
push_job R6_ssl_5p_s2_single_thr      $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_single_thr.py ""
push_job R8b_ssl_5p_s2_best_view      $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_best_view.py ""
push_job R8c_ssl_5p_s2_any_view       $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_any_view.py ""
push_job R8d_ssl_5p_s2_max_cam        $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_max_cam.py ""
push_job R7_ssl_5p_s2_perclass_boost  $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_perclass.py ""
push_job R18_ssl_5p_s2_hybrid_posonly $CFGD/g2_ssl_centerpoint_01voxel_nus_5p_s2_hybrid_posonly.py ""
touch "$ALL_PUSHED_FLAG"
echo "[$(date)] all jobs pushed (resume, 9 jobs)" >> "$LOG"

WPIDS=""
for G in $GPUS; do
    worker "$G" &
    WPIDS="$WPIDS $!"
done
wait $WPIDS
echo "[$(date)] Package B pipeline complete" >> "$LOG"
rm -f "$PIDFILE"
