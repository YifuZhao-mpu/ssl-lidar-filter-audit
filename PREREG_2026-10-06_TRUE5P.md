# Pre-registration — True-5% initialization replication (T5)

**Written 2026-10-06, before any T5 job had started.** No T5 checkpoint, dump, filter output or evaluation existed when this was written. The point of locking it now is that the reported claim cannot be chosen after seeing the numbers. Nothing in this file may be edited once the first T5 job starts; corrections go in a dated appendix at the bottom.

Pipeline: `scripts/t5_pipeline.sh`. Configs: `configs/true5p/`. Logs: `scripts/logs_t5/`.

## Why this experiment

Six role-separated journal-fit reviews on 2026-10-06 (same model family) (`review-stage/VENUE_UPGRADE_ASSESSMENT_2026-10-06.md`) named the same lead defect.

Every arm in the paper starts from the public mmdet3d CenterPoint-0.1voxel checkpoint. That checkpoint was trained with full supervision on all 700 train scenes, so it had already seen the labels of the scenes our split calls "unlabeled". The paper states this itself (§III-A: "5% names the fine-tuning partition, not the supervision budget").

The leaked labels could favour confidence gating, because a teacher that has seen U's labels is better calibrated on U. T5 removes the leak and re-asks the paper's headline question: does score-based gating beat camera-support gating when no arm has seen U's labels?

## Protocol (fixed now)

**Teacher (per seed).**
- CenterPoint-0.1voxel trained **from scratch** (`load_from=None`) on the seed's 5% labeled subset L only. These are the same subsets as the paper: `infos/nuscenes_infos_train_5p_s{0,1,2}_labeled.pkl`.
- Recipe: the stock mmdet3d `cyclic_20e` schedule, run to completion. Otherwise identical to the published teacher config: CBGS, `samples_per_gpu=4`, 1 GPU, EMA hook.
- Teacher = raw `epoch_20.pth`. No epoch selection.
- Rationale for epoch 20 instead of the paper's epoch 10: a from-scratch model needs the full schedule. This is the only schedule change, and it applies to every T5 arm.

**Labeled-only controls.**
- (a) The teacher itself, evaluated on val_full.
- (b) `T5_supcont`: the teacher continued for 10 epochs on L only, with exactly a student's schedule. This controls for the students' extra training.
- (a) is the paper-equivalent control. (b) is secondary.

**Pseudo-labels.**
- `tools/g2_dump_pseudo.py` on the seed's unlabeled split U, using the T5 teacher's raw epoch_20.

**Keep rules.**
- **Every constant is unchanged from the paper**: `PER_CLASS_THR` as published; single-thr τ=0.40; camera τc=0.15, γ=0.20, m(n), the passthrough branch and the no-valid-camera fallback; the same frozen COCO Faster R-CNN V2 detection cache (`cache/fasterrcnn_v2_train.pkl`); the same tools.
- **Nothing is re-tuned on the T5 teacher**, including the per-class thresholds. They are kept as published constants, not re-derived by the oracle cocoF1 search. This removes the score family's oracle-tuning advantage instead of re-creating it.

**Arms.**

| Seed | Students (10 epochs from T5 teacher epoch_20, CBGS over L ∪ P) | Controls |
|---|---|---|
| 0 | single_thr, perclass_only, best_view, consensus, perclass_and_consensus | teacher; supcont |
| 1 | single_thr, perclass_only, best_view, consensus, perclass_and_consensus | teacher; supcont |
| 2 | perclass_only, best_view (anchors only) | teacher |

Seed 2 is an anchors-only extension of the pairwise contrast. It is scheduled last and is not part of the primary decision rules.

**Evaluation.**
- Raw `epoch_10.pth`, never the EMA copy and never another epoch.
- val_full = 6019, with the patched `_format_bbox` (paper §III-E).

**Software stack.**
- 4× V100-32GB, torch 1.12.1, mmcv-full 1.6.1, TEOcc fork of mmdet3d 1.0.0rc4 (env `mm3drc4`).
- This differs from the stack behind the published numbers. All T5-internal comparisons are same-stack.
- Cross-stack comparisons to the published arms are descriptive only. The stack-to-stack anchor is the consensus retrain, which reproduced the original to 0.17 NDS (paper §IV-E).

## Statistics (fixed now)

- Paired scene-level block bootstrap: B=1000, 150 val scenes, the same driver as every published interval (`tools/g2_bootstrap_ci.py`), percentile intervals as primary. Basic intervals are reported alongside, as in the paper.
- **Primary quantity per seed:** the T5 family gap `G_s = min(NDS single_thr, NDS perclass_only) − max(NDS best_view, NDS consensus)`.
- **The ordering check uses all 4 individual score×camera contrasts per seed**, each with its own paired interval (paper §III-D).

## Decision rules (fixed now; each outcome reported with equal prominence)

Primary seeds: 0 and 1.

- **F1 — replicates.** All 4 score×camera contrasts are positive with paired 95% intervals excluding zero on both seeds.
  - Claim: "the score-based ordering holds when no arm has seen the unlabeled scenes' labels".
  - The initialization caveat moves from Limitations to a resolved control.
- **F2 — attenuated.** The point-estimate family gap `G_s` is > 0 on both seeds, but at least one contrast's interval includes zero, or `G_s` is less than half of the published same-seed gap (s0: 9.15, s1: 7.23 NDS).
  - Claim: the ordering survives, with its magnitude reported as initialization-dependent.
  - The paper reports both regimes side by side.
- **F3 — vanishes or reverses.** `G_s ≤ 0` on at least one primary seed.
  - Claim: the published ordering is conditional on the supervised initialization.
  - The paper is reframed around that dependence. The headline "score > camera" may not be stated without the initialization qualifier.
  - This outcome is reported at the same prominence as F1.

**Secondary analyses** (reported whatever they show; no decision rides on them):
1. SSL vs labeled-only: each student vs (a) its teacher and (b) supcont, with paired intervals. No claim of an SSL gain unless the interval excludes zero.
2. Conjunctive cost: `perclass_and_consensus − perclass_only` per seed, compared descriptively with the published −8.18 / −7.54 NDS.
3. Pre-flight checklist C1–C3 recomputed on the T5 dumps (CPU, descriptive).
4. Kept-box counts and per-class kept / P / R per arm (the paper's audit protocol).
5. Seed 2 anchors: `perclass_only − best_view` with paired interval, as a third-seed replication of the pairwise contrast.

## Abort and deviation criteria (fixed now)

- **Teacher gate:** if a teacher's val_full NDS is below 0.10, that seed's students are not trained (the pipeline enforces this). The seed is reported as degenerate. No retuning or schedule change follows without a dated appendix here.
- **Empty variant:** if any variant keeps zero boxes, it is not trained, and this is reported.
- **Divergence:** on NaN loss or a crash, rerun once with the same seed and config, and record it here.
- **Re-runs:** an interrupted run resumes from its last checkpoint (`--resume-from latest.pth`); this is not a deviation.
- **GPU sharing:** jobs start only on GPUs without foreign compute processes. Wall-clock time is not a reported quantity.

## Appendix — deviations and corrections

- **2026-10-06 11:16, before any T5 job had started** (all four workers were still waiting for free GPUs): engineering change only. In `scripts/t5_pipeline.sh` the per-seed filter build now runs its four independent builds concurrently, and support computation passes `--workers 8` to `tools/g2_make_pseudo_infos.py`. The tools, arguments, constants and outputs are unchanged; support is computed per keyframe, so the result does not depend on the worker count. The pipeline was stopped and relaunched for this. A new hash line was appended to `scripts/logs_t5/prereg_lock.sha256`; the original lines are kept.
- **2026-10-06 11:22, before any T5 job had started:** clarification, not a change. A pipeline audit found that every info file in the study (`infos/nuscenes_infos_*.pkl`, TEOcc/BEVDet format) has an empty LiDAR `sweeps` list. Every published model was therefore trained and evaluated on the single keyframe sweep (mAVE 1.15–1.45 m/s across all 41 evaluated runs), and `g2_dump_pseudo.py` also passes `sweeps=[]`. T5 keeps this regime unchanged, so initialization stays the only manipulated factor. The single-sweep regime is disclosed in the manuscript (§III-A, Limitations vii). A multi-sweep check would be a separate experiment.
