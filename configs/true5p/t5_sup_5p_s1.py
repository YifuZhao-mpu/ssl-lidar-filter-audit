# T5 teacher, seed 1 -- true-5% replication (PREREG_2026-10-06_TRUE5P.md).
# CenterPoint-0.1voxel trained FROM SCRATCH on the seed-1 5% labeled subset only.
# Identical to configs/local/g2_sup_centerpoint_01voxel_nus_5p_s0.py (the published
# teacher config) except: load_from=None (no public checkpoint, which was trained
# with full supervision on all 700 train scenes), the stock 20-epoch cyclic
# schedule run to completion (teacher = raw epoch_20), seed-1 labeled subset,
# val/test on val_full, no in-training evaluation.
_base_ = ['/home/zyf/paper/BMVC2026/configs/local/g2_sup_centerpoint_01voxel_nus_5p_s0.py']
data = dict(
    train=dict(dataset=dict(ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_train_5p_s1_labeled.pkl')),
    val=dict(ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl'),
    test=dict(ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl'),
)
load_from = None
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=1000)
work_dir = '/home/zyf/paper/BMVC2026/work_dirs/T5_sup_5p_s1'
seed = 1
