# 5% s2 labeled-only teacher (Package B third seed, 1407 keyframes from seed 2).
# Mirrors g2_sup_centerpoint_01voxel_nus_5p_s1.py but with s2 PKL + seed 2.
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

data = dict(
    train=dict(
        dataset=dict(
            ann_file='/root/BMVC2026/infos/nuscenes_infos_train_5p_s2_labeled.pkl',
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

work_dir = '/root/BMVC2026/work_dirs/R4_sup_5p_s2'
seed = 2
