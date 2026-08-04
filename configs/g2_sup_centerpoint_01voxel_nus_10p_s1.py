# 10% s1 labeled-only teacher (R10a-s1; doubles as 10% no-SSL baseline seed-1).
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

data = dict(
    train=dict(
        dataset=dict(
            ann_file='/root/BMVC2026/infos/nuscenes_infos_train_10p_s1_labeled.pkl',
        ),
    ),
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

work_dir = '/root/BMVC2026/work_dirs/R10a_sup_10p_s1'
seed = 1
