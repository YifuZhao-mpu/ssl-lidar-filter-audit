# R17a — 1% s0 labeled-only CenterPoint teacher.
# Doubles as the no-SSL baseline for the 1% probe.
# Inherits everything from the 5% sup config; overrides labeled PKL + work_dir.
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

data = dict(
    train=dict(
        dataset=dict(
            ann_file='/root/BMVC2026/infos/nuscenes_infos_train_1p_s0_labeled.pkl',
        ),
    ),
    val=dict(
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
    ),
    test=dict(
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
    ),
)

# Train fewer epochs because labeled set is 5x smaller; 10ep matches R4's effective
# teacher checkpoint regime.
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=5)
# In-train eval costs ~10 min on val_full=6019; only run at the end to save time.
evaluation = dict(interval=10, pipeline={{_base_.eval_pipeline}})

work_dir = '/root/BMVC2026/work_dirs/R17a_sup_1p_s0'
