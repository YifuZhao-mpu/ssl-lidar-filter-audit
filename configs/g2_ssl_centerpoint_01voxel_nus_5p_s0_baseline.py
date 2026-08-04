# R6 — 5% s0 SSL baseline: labeled (1407) + pseudo-filtered@single_thr=0.40 (~26k subset)
# Trains CenterPoint from scratch (load_from teacher ep10) with ConcatDataset of
# labeled + pseudo-labeled subsets. ep20, GPU 6.
#
# Differs from g2_sup_centerpoint_01voxel_nus_5p_s0.py only in:
#   - data.train: ConcatDataset of labeled + pseudo
#   - work_dir
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesLidarSSLDataset'
input_modality = dict(use_lidar=True, use_camera=False,
                      use_radar=False, use_map=False, use_external=False)

train_pipeline = {{_base_.train_pipeline}}

test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='/root/BMVC2026/infos/nuscenes_infos_train_5p_s0_labeled.pkl',
                    pipeline=train_pipeline,
                    classes=class_names,
                    test_mode=False,
                    use_valid_flag=True,
                    box_type_3d='LiDAR',
                    modality=input_modality,
                ),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='/root/BMVC2026/infos/nuscenes_pseudo_5p_s0_single_thr.pkl',
                    pipeline=train_pipeline,
                    classes=class_names,
                    test_mode=False,
                    use_valid_flag=True,
                    box_type_3d='LiDAR',
                    modality=input_modality,
                ),
            ],
            separate_eval=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/root/BMVC2026/infos/nuscenes_infos_val_full.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=5, pipeline={{_base_.eval_pipeline}})

# Init from teacher ep10 (rather than centerpoint_01voxel_nus.pth) since it's
# already adapted to 5% labeled distribution
load_from = '/root/BMVC2026/work_dirs/R4_sup_5p_s0/epoch_10.pth'

work_dir = '/root/BMVC2026/work_dirs/R6_ssl_5p_s0_baseline'
seed = 0
gpu_ids = [0]
