#!/bin/bash
# Build the three pseudo-infos for the 2026-08-12 arms (PREREG_2026-08-12.md):
#   R19 seed 1, R19 seed 2  -- conjunctive keep rule replicated off seed 0
#   R20 seed 0              -- volume-matched random pruning control
#
# teocc python (numpy 1.26.4) is mandatory: pickles written by system python
# (numpy 2.x) reference numpy._core and the training env cannot import them.
# This has cost the project two rebuilds; see MANIFEST 2026-08-05.
set -u
ROOT=/root/BMVC2026
PY=/root/miniconda3/envs/teocc/bin/python
LOGD=$ROOT/scripts/logs20
mkdir -p "$LOGD"

build () {   # $1 seed  $2 variant  $3 out-basename
    echo "[$(date)] build start $3"
    "$PY" "$ROOT/tools/g2_make_pseudo_infos.py" \
        --raw-pseudo      "$ROOT/cache/pseudo_5p_$1_raw_full.pkl" \
        --unlabeled-infos "$ROOT/infos/nuscenes_infos_train_5p_$1_unlabeled.pkl" \
        --det-cache       "$ROOT/cache/fasterrcnn_v2_train.pkl" \
        --variant         "$2" \
        --prune-seed      19 \
        --out-pkl         "$ROOT/infos/$3" \
        > "$LOGD/build_$3.log" 2>&1
    echo "[$(date)] build done  $3 rc=$?"
}

build s1 perclass_and_consensus nuscenes_pseudo_5p_s1_perclass_and_consensus.pkl &
build s2 perclass_and_consensus nuscenes_pseudo_5p_s2_perclass_and_consensus.pkl &
build s0 perclass_random_prune  nuscenes_pseudo_5p_s0_perclass_random_prune.pkl &
wait
echo "[$(date)] ALL BUILDS COMPLETE"
