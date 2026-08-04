# Score-based versus camera-support pseudo-label filtering for semi-supervised LiDAR 3D object detection

Code, configurations and per-run results for the paper *"Score-based versus
camera-support pseudo-label filtering for semi-supervised LiDAR 3D object
detection: a controlled study with design implications"* (under review).

> **Status: released at submission time.** Everything needed to reproduce the
> analysis, the figures and the tables is here now. Bulk artifacts are not —
> nuScenes itself, the teacher checkpoints (~36 MB each), and the raw
> pseudo-label dumps (~4.0 M boxes per seed) are too large for this repository
> and are regenerable from the configs and tools below.

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

A side contribution is independent of the rest: a ≤30-line patch to mmdetection3d's
`NuScenesDataset._format_bbox` correcting a 90° `lidar2ego` pose misalignment that
silently yields `NDS = 0` on LiDAR-only evaluation. **The full patch and its
derivation are also reproduced in Appendix A of the paper**, so that fix is usable
without cloning anything.

## Layout

| Path | Contents |
|------|----------|
| `patch/` | `nusc_lidar_ssl_dataset.py` — the `lidar2ego` `_format_bbox` fix, as a drop-in `NuScenesDataset` subclass override |
| `configs/` | 36 mmdetection3d configs: per-seed supervised teachers and every SSL student in the filter family |
| `tools/` | Pseudo-label dumping, the seven filter implementations, the class-aware positive-only hybrid, the 2D-detection cache, the support-correlation and per-class kept/precision/recall audits, the scene-level block bootstrap and the pre-flight diagnostic |
| `tools/pipelines/` | The shell drivers that sequenced the runs, kept verbatim as a record of execution order |
| `results/analysis/` | Bootstrap confidence intervals, support-correlation statistics, pilot cocoF1 per seed, per-class kept/P/R, pre-flight diagnostic — all JSON |
| `results/per_run/` | 34 `metrics_summary.json` files, one per evaluated run, under the patched protocol on `val_full = 6019` |
| `figures/` | Deterministic figure and table generators; they read `results/` and emit the paper's figures and LaTeX tables |

## Reproducing without a GPU

The analysis layer is CPU-only and runs off `results/analysis/`:

- **Scene-level block bootstrap** (`tools/g2_bootstrap_ci.py`) — B = 1000
  replicates × 26 runs, ≈4.3 h CPU. Point estimates reproduce the published NDS
  from cached detections; 23 of 26 are bit-identical, maximum deviation
  5.6 × 10⁻¹⁷.
- **Pre-flight diagnostic** (`tools/g2_preflight_diagnostic.py`) — the three
  checks (vocabulary coverage, view independence, gate selectivity), seconds.
- **Per-class kept/P/R audit** (`tools/g2_perclass_kept_pr.py`).
- **Figures and tables** (`figures/gen_*.py`).

Regenerating the pseudo-label dumps and retraining the students needs a GPU;
the full grid is ≈380 GPU-h on a single 96 GB card (Appendix E of the paper).

## Notes for anyone reproducing

**Absolute paths are preserved deliberately.** The configs and tool defaults
still point at `/root/BMVC2026/...`, which is where these runs actually
executed. They are kept verbatim rather than rewritten so the released configs
remain an exact record of what produced the reported numbers. Substitute your
own data root, `work_dir` and checkpoint paths before running.

**Initialization is not label-free.** Every arm — labeled-only controls,
teachers and SSL students alike — loads the public mmdetection3d CenterPoint
nuScenes checkpoint (`load_from` in the supervised configs), which was trained
with full supervision on the entire 700-scene train split, including the scenes
the `5%` split then treats as unlabeled. The absolute NDS values are therefore
not clean 5%-budget numbers. Because the same checkpoint initializes every arm
it is a constant of the controlled comparison rather than a confound between
the two filter families. This is stated in §3.1 and limitation (vi) of the
paper; it is repeated here so it is visible from the code.

## Known issues

- `tools/g2_dump_pseudo.py` writes a `class_score_floor` of 0.25–0.30 into each
  dump's metadata, but **never applies it** to the scores. The operative floor
  is the detector's own NMS threshold, 0.10. The reported numbers come from the
  un-floored dumps, so the behaviour is left as-is; treat the metadata field as
  documentation of an intent that was not executed. The consequence — the
  camera-support filters inherit no confidence gate — is analysed in §3.2 and
  limitation (vii) of the paper.

## Environment

mmdetection3d 1.0.0rc4, PyTorch 2.7 / CUDA 12.8, single NVIDIA RTX PRO 6000
(96 GB). nuScenes v1.0-trainval.

## Citation

A BibTeX entry will be added here once the paper has a DOI.

## License

Released under the Apache License 2.0 — chosen for compatibility with
mmdetection3d, from which the evaluation patch derives.
