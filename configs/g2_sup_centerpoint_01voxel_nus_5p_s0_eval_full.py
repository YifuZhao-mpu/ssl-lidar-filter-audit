# R4b — 5% s0 labeled-only fine-tune, eval on val_full=6019.
# Same as R4 (g2_sup_centerpoint_01voxel_nus_5p_s0.py) but:
#   - val_full=6019 instead of valmini=4850 (avoids R4's mismatch crash)
#   - max_epochs=10 + interval=10 (only end-of-train eval, save GPU time)
# Doubles as the no-SSL baseline at 5% scale (apples-to-apples with R17a/R10a).
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

data = dict(
    val=dict(
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
    ),
    test=dict(
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=10, pipeline={{_base_.eval_pipeline}})

work_dir = '/root/BMVC2026/work_dirs/R4b_sup_5p_s0_eval_full'
