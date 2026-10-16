#!/bin/bash
# True-5% initialization replication (T5) -- see review-stage/PREREG_2026-10-06_TRUE5P.md
#
# One job queue, one worker per GPU (0-3). Job types:
#   TEACHER|<seed>          train T5 teacher from scratch on L (20 ep) -> val_full eval
#                           -> sanity gate -> dump pseudo on U -> (background, CPU)
#                           build filters/infos and queue that seed's students
#   STUDENT|<tag>|<cfg>     train 10 ep from the T5 teacher -> external eval (raw epoch_10)
# A worker only takes a job when its GPU carries no foreign compute process
# (>1000 MiB, not ours): GPUs 1/2 were running the user's ICRA2027 jobs at launch,
# and this pipeline never preempts or co-runs with them.
# Re-runnable: finished evals are skipped, interrupted trainings resume from latest.pth.
set -u
ROOT=/home/zyf/paper/BMVC2026
TEO=/home/zyf/TEOcc
PY=/home/zyf/anaconda3/envs/mm3drc4/bin/python
CFGD=$ROOT/configs/true5p
LOGD=$ROOT/scripts/logs_t5
LOG=$LOGD/pipeline.log
QUEUE=$LOGD/queue.txt
LOCK=$LOGD/queue.lock
FLAGS=$LOGD/flags
DET=$ROOT/cache/fasterrcnn_v2_train.pkl
DATA=$ROOT/data/nuscenes
GPUS="0 1 2 3"
SEEDS="0 1 2"
MIN_TEACHER_NDS=0.10          # pre-registered abort gate

mkdir -p "$LOGD" "$FLAGS"
touch "$QUEUE" "$LOCK"
log () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

push_job () { ( flock 9; echo "$1" >> "$QUEUE" ) 9>"$LOCK"; log "queued: $1"; }
pop_job () {  # seed-2 jobs (anchors-only extension) are taken only when nothing else is queued
    ( flock 9
      line=$(grep -v -e '^TEACHER|2$' -e '_s2_' "$QUEUE" | head -n1)
      [ -z "$line" ] && line=$(head -n1 "$QUEUE")
      if [ -n "$line" ]; then
          grep -vxF -- "$line" "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
      fi
      echo "$line"
    ) 9>"$LOCK"
}

gpu_foreign_busy () {  # 0 (true) if GPU $1 runs a compute process that is not ours
    local uuid
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$1" '$1==g{print $2}')
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits |
    while IFS=', ' read -r pid u mem; do
        [ "$u" = "$uuid" ] || continue
        [ "${mem:-0}" -gt 1000 ] || continue
        if ! tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q -e BMVC2026 -e TEOcc; then
            echo busy; break
        fi
    done | grep -q busy
}

nds_of () { $PY -c "import json;print(json.load(open('$1/results/pts_bbox/metrics_summary.json'))['nd_score'])" 2>/dev/null; }

train_eval () {  # gpu tag cfg final_ckpt
    local gpu=$1 tag=$2 cfg=$3 ck=$4
    local wd=$ROOT/work_dirs/$tag ed=$ROOT/work_dirs/${tag}_external_eval
    mkdir -p "$wd" "$ed"
    if [ -f "$ed/results/pts_bbox/metrics_summary.json" ]; then
        log "skip $tag (already evaluated, NDS=$(nds_of "$ed"))"; return 0
    fi
    if [ ! -f "$wd/$ck" ]; then
        local resume=""
        [ -e "$wd/latest.pth" ] && resume="--resume-from $wd/latest.pth"
        log "TRAIN start $tag gpu$gpu $resume"
        ( cd "$TEO" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$TEO:$ROOT \
            "$PY" tools/train.py "$cfg" --work-dir "$wd" $resume >> "$wd/train.log" 2>&1 )
        log "TRAIN done $tag rc=$?"
    fi
    if [ ! -f "$wd/$ck" ]; then log "FATAL $tag: $ck missing, not evaluating"; return 1; fi
    log "EVAL start $tag ($ck, val_full)"
    ( cd "$TEO" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$TEO:$ROOT \
        "$PY" tools/test.py "$cfg" "$wd/$ck" --eval bbox \
        --eval-options "jsonfile_prefix=$ed/results" > "$ed/eval.log" 2>&1 )
    log "EVAL done $tag rc=$? NDS=$(nds_of "$ed")"
}

build_and_queue () {  # seed  (CPU only)
    local s=$1
    local raw=$ROOT/cache/pseudo_t5_5p_s${s}_raw_full.pkl
    local un=$ROOT/infos/nuscenes_infos_train_5p_s${s}_unlabeled.pkl
    local fo=$ROOT/cache/pseudo_t5_5p_s${s}_filtered
    local bl=$LOGD/build_s${s}.log
    local inf=$ROOT/infos/nuscenes_pseudo_t5_5p_s${s}
    mkdir -p "$fo"
    log "BUILD start s$s"
    cd "$ROOT"
    # The camera filter, the score filter and the two support-based variants are
    # independent, so they run concurrently; support computation uses a worker pool.
    ( [ -s "$fo/best_view.pkl" ] && [ -s "$fo/consensus.pkl" ] ) || \
        ( PYTHONPATH=$ROOT "$PY" tools/g2_filter_pseudo.py --raw-pseudo "$raw" --det-cache "$DET" \
            --source-infos "$un" --out-dir "$fo" --variants best_view consensus \
            --max-cam-topk 5 > "$bl.camera" 2>&1; log "BUILD s$s camera filters rc=$?" ) &
    [ -s "${inf}_single_thr.pkl" ] || ( PYTHONPATH=$ROOT "$PY" tools/g2_make_pseudo_infos.py \
        --raw-pseudo "$raw" --unlabeled-infos "$un" --variant single_thr --single-thr 0.40 \
        --out-pkl "${inf}_single_thr.pkl" > "$bl.single_thr" 2>&1; log "BUILD s$s single_thr rc=$?" ) &
    for V in perclass_only perclass_and_consensus; do
        [ -s "${inf}_${V}.pkl" ] || ( PYTHONPATH=$ROOT "$PY" tools/g2_make_pseudo_infos.py \
            --raw-pseudo "$raw" --unlabeled-infos "$un" --det-cache "$DET" --variant "$V" \
            --workers 8 --out-pkl "${inf}_${V}.pkl" > "$bl.$V" 2>&1; log "BUILD s$s $V rc=$?" ) &
    done
    wait
    for V in best_view consensus; do
        [ -s "${inf}_${V}.pkl" ] || PYTHONPATH=$ROOT "$PY" tools/g2_make_pseudo_infos.py \
            --raw-pseudo "$fo/${V}.pkl" --unlabeled-infos "$un" --variant passthrough \
            --out-pkl "${inf}_${V}.pkl" > "$bl.merge_$V" 2>&1
    done
    log "BUILD done s$s: $(ls ${inf}_*.pkl 2>/dev/null | wc -l) infos"
    # queue students (family-gap anchors first). Seed 2 = anchors only (pre-registered).
    local vlist="perclass_only best_view single_thr consensus perclass_and_consensus"
    [ "$s" = "2" ] && vlist="perclass_only best_view"
    for V in $vlist; do
        if [ -s "${inf}_${V}.pkl" ]; then
            push_job "STUDENT|T5_ssl_5p_s${s}_${V}|$CFGD/t5_ssl_5p_s${s}_${V}.py"
        else
            log "FATAL s$s: infos for $V missing, not queued"
        fi
    done
    [ "$s" != "2" ] && push_job "STUDENT|T5_supcont_5p_s${s}|$CFGD/t5_supcont_5p_s${s}.py"
    touch "$FLAGS/seed${s}.done"
}

teacher_job () {  # gpu seed
    local gpu=$1 s=$2
    local tag=T5_sup_5p_s${s}
    local raw=$ROOT/cache/pseudo_t5_5p_s${s}_raw_full.pkl
    if ! train_eval "$gpu" "$tag" "$CFGD/t5_sup_5p_s${s}.py" epoch_20.pth; then
        log "ABORT seed $s: teacher failed"; touch "$FLAGS/seed${s}.done"; return
    fi
    local nds; nds=$(nds_of "$ROOT/work_dirs/${tag}_external_eval")
    if ! $PY -c "import sys; sys.exit(0 if float('${nds:-0}') >= $MIN_TEACHER_NDS else 1)"; then
        log "ABORT seed $s: teacher NDS=${nds:-NA} < $MIN_TEACHER_NDS (pre-registered gate)"
        touch "$FLAGS/seed${s}.done"; return
    fi
    if [ ! -s "$raw" ]; then
        log "DUMP start s$s gpu$gpu"
        ( cd "$TEO" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$TEO:$ROOT \
            "$PY" "$ROOT/tools/g2_dump_pseudo.py" --config "$CFGD/t5_sup_5p_s${s}.py" \
            --ckpt "$ROOT/work_dirs/$tag/epoch_20.pth" \
            --unlabeled-pkl "$ROOT/infos/nuscenes_infos_train_5p_s${s}_unlabeled.pkl" \
            --data-root "$DATA" --out "$raw" > "$LOGD/dump_s${s}.log" 2>&1 )
        log "DUMP done s$s rc=$?"
    fi
    if [ ! -s "$raw" ]; then log "ABORT seed $s: dump missing"; touch "$FLAGS/seed${s}.done"; return; fi
    ( build_and_queue "$s" ) &
}

all_settled () {
    for s in $SEEDS; do [ -f "$FLAGS/seed${s}.done" ] || return 1; done
    return 0
}

worker () {
    local gpu=$1 waited=0
    log "worker gpu$gpu up"
    while true; do
        if gpu_foreign_busy "$gpu"; then
            [ $waited -eq 0 ] && log "worker gpu$gpu: GPU has a foreign process, waiting"
            waited=1; sleep 300; continue
        fi
        [ $waited -eq 1 ] && { log "worker gpu$gpu: GPU free, joining"; waited=0; }
        local job; job=$(pop_job)
        if [ -z "$job" ]; then
            all_settled && break
            sleep 120; continue
        fi
        IFS='|' read -r kind a b <<< "$job"
        case "$kind" in
            TEACHER) teacher_job "$gpu" "$a" ;;
            STUDENT) train_eval "$gpu" "$a" "$b" epoch_10.pth ;;
            *) log "worker gpu$gpu: unknown job '$job'" ;;
        esac
    done
    log "worker gpu$gpu exit"
}

log "=== T5 pipeline start (pid $$) ==="
if [ ! -s "$QUEUE" ] && ! ls "$FLAGS"/seed*.done >/dev/null 2>&1; then
    for s in $SEEDS; do push_job "TEACHER|$s"; done
fi
for g in $GPUS; do worker "$g" & done
wait
log "=== T5 pipeline complete ==="
