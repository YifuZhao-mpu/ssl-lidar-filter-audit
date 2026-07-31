# Score-based versus camera-support pseudo-label filtering for semi-supervised LiDAR 3D object detection

Code and artifacts for the paper *"Score-based versus camera-support pseudo-label
filtering for semi-supervised LiDAR 3D object detection: a controlled study with
design implications"* (under review).

> **Status: code and data will be released upon acceptance.**
> This repository is a placeholder so that the paper can cite a stable URL. The
> directory layout below is what will be populated; nothing is withheld
> permanently.

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
derivation are reproduced in Appendix A of the paper**, so that fix is usable
today without waiting for this release.

## Planned layout

| Path | Contents |
|------|----------|
| `patch/` | The `lidar2ego` `_format_bbox` patch, as a drop-in subclass override |
| `configs/` | Per-seed supervised and SSL configs for the full filter family |
| `tools/` | Pseudo-label dumping, the seven filter implementations, the class-aware hybrid, and the per-class kept/precision/recall audit |
| `results/` | Per-run evaluation metrics and the scene-level bootstrap confidence intervals |
| `figures/` | Deterministic figure and table generators |

## Environment

The experiments run on mmdetection3d 1.0.0rc4 with PyTorch 2.7 / CUDA 12.8.
Exact versions, the nuScenes preparation steps and the per-run compute accounting
will accompany the release.

## Citation

A BibTeX entry will be added here once the paper has a DOI.

## License

Released under the Apache License 2.0 — chosen for compatibility with
mmdetection3d, from which the evaluation patch derives.
