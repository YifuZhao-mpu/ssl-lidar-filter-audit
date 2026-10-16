# Score-based versus camera-support pseudo-label filtering for semi-supervised LiDAR 3D object detection

Code, configurations and per-run results for the paper *"Score-based versus
camera-support pseudo-label filtering for semi-supervised LiDAR 3D object
detection: a controlled study with design implications"* (under review).

> **Status: released at submission time.** The code, the configurations and the
> per-run and analysis outputs behind every number in the paper are here now.
> Bulk artifacts are not — nuScenes itself, the teacher checkpoints (~36 MB
> each), the raw pseudo-label dumps (~4.0 M boxes per seed) and the cached
> per-run detections are too large for this repository and are regenerable
> from the configs and tools below. Because several analysis scripts read
> those bulk paths directly, not everything re-runs unmodified from a bare
> clone; "What does and does not re-run from this repository" below says
> exactly which does.

## What the study does

Semi-supervised learning for LiDAR 3D detection turns on a single design choice:
the *filter* that decides which teacher proposals become pseudo-labels. A
recurring idea is to use the surrounding cameras as independent verifiers —
project each LiDAR pseudo-box into every camera and keep it only if enough views
agree.

We test that idea under controlled conditions. On nuScenes with a CenterPoint EMA
mean-teacher at 5% labels, the learner, the data splits and the training recipe
are held fixed and **only the keep rule varies**, across a seven-filter family
(three score-based rules, four camera-support rules) and three random seeds. The
paper reports the outcome, three diagnostics that explain it, a constructive
control, and a pre-flight checklist that predicts the outcome before any student
is trained.

A side contribution is independent of the rest: a ≤30-line patch to the
`NuScenesDataset._format_bbox` of the **BEVDet-lineage mmdet3d forks** — here
[TEOcc](https://github.com/VDIGPKU/TEOcc), the codebase these runs are built on —
correcting a 90° `lidar2ego` pose misalignment that silently yields `NDS = 0` when
such a fork is driven LiDAR-only. **Upstream mmdetection3d 1.0.0rc4 does not have
this defect**: its `lidar_nusc_box_to_global` composes `lidar2ego` and then
`ego2global` correctly, and the identifier `ego_cam` does not appear in its
formatter. The fork substitutes a single camera-time `ego2global`, which is right
for its own radar–camera pipeline and silently wrong for LiDAR-only detection.
**The operative part of the patch and its derivation are reproduced in Appendix A
of the paper**, so the fix is usable without cloning anything; the complete file is
here in `patch/`.

## Layout

| Path | Contents |
|------|----------|
| `patch/` | `nusc_lidar_ssl_dataset.py` — the `lidar2ego` `_format_bbox` fix for the TEOcc/BEVDet-lineage fork (not needed on upstream mmdet3d), as a drop-in `NuScenesDataset` subclass override |
| `configs/` | 40 mmdetection3d configs: per-seed supervised teachers and every SSL student in the filter family |
| `configs/local/` | 6 configs as run on the second machine (`/home/zyf/paper/BMVC2026` root): the split-gate arm and its matched shuffle control, the same-stack consensus retrain, and the two bases the true-5% configs inherit from |
| `configs/true5p/` | 21 configs of the true-5% replication: from-scratch teachers, schedule-matched labeled-only continuations and students, per seed |
| `tools/` | Pseudo-label dumping, the seven filter implementations, the class-aware positive-only hybrid, the gate-matched conjunctive control, the 2D-detection cache, the support-correlation and per-class kept/precision/recall audits, the scene-level block bootstrap and the pre-flight diagnostic |
| `tools/pipelines/` | The shell drivers that sequenced the runs, kept verbatim as a record of execution order |
| `results/analysis/` | Bootstrap confidence intervals, support-correlation statistics, pilot cocoF1 per seed, per-class kept/P/R, pre-flight diagnostic — all JSON |
| `results/t5/` | The true-5% replication: `T5_ANALYSIS.md` (pre-registered decision and every contrast), `t5_analysis.json`, `bootstrap_ci_T5.json`, the secondary analyses (`T5_SECONDARY.md`, `t5_secondary.json`, per-seed `perclass_kept_pr_t5_*.json` and `d2_correlation_conditional_t5_*.json`), the pipeline log, and the prereg hash lock |
| `results/per_run/` | 58 files, one per evaluated run: each is that run's mmdetection3d `metrics_summary.json`, renamed to `<TAG>__results__pts_bbox.json` (arms added from 2026-08-05 on, including the split-gate arms and the 17 `T5_*` runs, are plain `<TAG>.json`), under the patched protocol on `val_full = 6019` |
| `figures/` | Deterministic figure and table generators that emit the paper's figures and LaTeX tables. They read the *working tree* (`work_dirs/`, `cache/`), not `results/` — see "What does and does not re-run from this repository" below |

## What does and does not re-run from this repository

Be aware of the distinction between *auditing* a number and *recomputing* it.
Every number in the paper can be audited here: `results/per_run/` holds the
evaluation output of each of the 58 runs and `results/analysis/` holds the
output of every analysis pass. Recomputing them from scratch is a different
matter, because the analysis scripts were written to run inside the working
tree and read `work_dirs/` and `cache/`, which hold the bulk artifacts this
repository excludes. Concretely:

| Script | Reads | Re-runs from a bare clone? |
|---|---|---|
| `tools/g2_preflight_diagnostic.py` | three JSONs that ship in `results/analysis/`: `perclass_kept_pr_5p_s0.json`, `d2_correlation_analysis.json`, `d2_correlation_perclass.json` | **Yes**, and it is the only one — after repointing `ROOT` (line 37, hardcoded to `/workspace/BMVC2026`) at your clone and copying `results/analysis/*.json` into `<ROOT>/cache/`. Seconds. |
| `tools/g2_bootstrap_ci.py` | `work_dirs/<TAG>/results/pts_bbox/results_nusc.json` (~12 GB over 38 runs) and `cache/bootstrap_prep/` (~3.1 GB) | **No.** Its *output* ships as `results/analysis/bootstrap_ci*.json`. B = 1000 replicates, run in four passes (26-run main, 8-run supplement, and two contrast passes for the 2026-08-12 arms), ≈9.6 h CPU in total when the detections are present. All 38 evaluated runs of the main study carry an interval; a fifth pass over the 17 true-5% runs (`--runs t5`, ≈19 h on 32 CPU workers) ships as `results/t5/bootstrap_ci_T5.json`; point estimates reproduce the published NDS, 26 of 30 bit-identical in the merged table, max deviation 5.6 × 10⁻¹⁷. |
| `tools/g2_perclass_kept_pr.py` | the raw pseudo-label dumps and the unlabeled `infos` pickles | **No** — both are bulk artifacts. Output ships as `results/analysis/perclass_kept_pr_5p_s0.json`. |
| `figures/gen_*.py` | per-run `eval.log` under `work_dirs/`, plus `cache/` | **No** as written. The same NDS values are in `results/per_run/*.json` under `nd_score` and `label_aps`, so the plots are re-derivable, but you would have to repoint `figures/paper_plot_style.py:load_run` at `results/per_run/`. |

Regenerating the pseudo-label dumps and retraining the students needs a GPU;
the full grid is ≈420 GPU-h on a single 96 GB card (Appendix E of the paper).

## Pre-registration

`PREREG_2026-10-06_TRUE5P.md` fixes, before any of its jobs started, the
protocol and decision rule of the true-5% replication; its hashes, plus dated
appendix entries, are in `results/t5/prereg_lock.sha256`. The pre-registered
outcome was **F1 (replicates)**: see `results/t5/T5_ANALYSIS.md`.

`PREREG_2026-08-12.md` fixes, *before* the runs existed, the decision rules for
the two controls added on 2026-08-12: the seed replication of the conjunctive
arm and the volume-matched random-pruning control. It records which branch each
rule actually fired, including one argument the control forced us to withdraw.

## Notes for anyone reproducing

**Two different Python environments, and they are not interchangeable.**
Training and pseudo-label generation run in the mmdetection3d environment
(`numpy 1.26`); the bootstrap driver runs in an environment with `numpy 2.x`.
The two write incompatible pickles in *both* directions — numpy 2.x pickles
reference `numpy._core`, which numpy 1.26 cannot import, and the cached
`bootstrap_prep/gt_val_filtered.pkl` in this study was written by numpy 2.x.
So: generate pseudo-label infos with the **mmdet3d** interpreter, and run
`tools/g2_bootstrap_ci.py` with the **numpy-2.x** one. Mixing them fails at
dataset construction or at bootstrap worker init, with a
`ModuleNotFoundError: No module named 'numpy._core.numeric'` that does not
name the real cause.

**Absolute paths are preserved deliberately, and there are three of them.** The
configs and most tool defaults point at `/root/BMVC2026/...`, which is where
the main-study runs executed. Five later analysis scripts —
`tools/g2_{make_bootstrap_outputs,pilot_cocoF1,preflight_diagnostic}.py`
and `figures/gen_{tables,fig1_family_gap}.py` — point at `/workspace/BMVC2026`
instead: the project was relocated mid-study, and `/root/BMVC2026` became a
symlink to the new location, so both roots resolved on the original machine.
The split-gate arms and the true-5% replication ran on a second machine under
`/home/zyf/paper/BMVC2026` (with the TEOcc checkout at `/home/zyf/TEOcc`):
`configs/local/`, `configs/true5p/` and `tools/pipelines/t5_pipeline.sh` use
that root. They are kept verbatim rather than rewritten so the released files
remain an exact record of what produced the reported numbers. Substitute your
own data root, `work_dir` and checkpoint paths before running — and note that
most scripts hardcode these roots rather than deriving them, so an unmodified
clone elsewhere will not find its inputs. The exceptions are
`tools/g2_bootstrap_ci.py` and `tools/t5_secondary_diagnostics.py`, which derive
the project root from their own location (override with the `ARIS_PROJECT_ROOT`
environment variable), and `tools/t5_analysis.py`, whose inputs and outputs are
arguments relative to the working directory. Running
`python3 tools/t5_analysis.py --boot results/t5/bootstrap_ci_T5.json` from a bare
clone reproduces `results/t5/T5_ANALYSIS.md`.

**Initialization is not label-free.** Every arm — labeled-only controls,
teachers and SSL students alike — loads the public mmdetection3d CenterPoint
nuScenes checkpoint (`load_from` in the supervised configs), which was trained
with full supervision on the entire 700-scene train split, including the scenes
the `5%` split then treats as unlabeled. The absolute NDS values are therefore
not clean 5%-budget numbers. Because the same checkpoint initializes every arm
it is a constant of the controlled comparison rather than a confound between
the two filter families. This is stated in §3.1 and Limitations (iii) of the
paper; it is repeated here so it is visible from the code.

The **true-5% replication** removes it for the core contrast. Its teachers
(`configs/true5p/t5_sup_5p_s*.py`, `load_from = None`) are trained from random
initialization on the 1406 labeled keyframes of each seed, and every filter
constant is reused unchanged. Driver: `tools/pipelines/t5_pipeline.sh`; decision:
`tools/t5_analysis.py`; secondary analyses: `tools/t5_secondary_diagnostics.py`
and `tools/g2_support_correlation_conditional.py`. The labeled and unlabeled sets
are split by keyframe, not by scene, in both regimes.

**Single-sweep input.** The info files used throughout carry an empty LiDAR
`sweeps` list, so every model in both regimes sees only the keyframe sweep
(mAVE 1.15–1.68 m/s, velocity term of NDS clipped to zero). Absolute NDS is
therefore below published multi-sweep CenterPoint numbers; see §3.1 of the paper.

## Known issues

- `tools/g2_dump_pseudo.py` writes a `class_score_floor` of 0.25–0.30 into each
  dump's metadata, but **never applies it** to the scores. The operative floor
  is the detector's own NMS threshold, 0.10. The reported numbers come from the
  un-floored dumps, so the behaviour is left as-is; treat the metadata field as
  documentation of an intent that was not executed. The consequence — the
  camera-support filters inherit no confidence gate — is analysed in §3.2 of
  the paper.

## Environment

Main study: mmdetection3d 1.0.0rc4, PyTorch 2.7 / CUDA 12.8, single NVIDIA RTX
PRO 6000 (96 GB). Split-gate arms and the true-5% replication: the TEOcc fork of
mmdetection3d 1.0.0rc4, PyTorch 1.12.1, mmcv-full 1.6.1, one V100-32GB per job.
nuScenes v1.0-trainval.

## Citation

A BibTeX entry will be added here once the paper has a DOI.

## License

Released under the Apache License 2.0 — chosen for compatibility with
mmdetection3d, from which the evaluation patch derives.
