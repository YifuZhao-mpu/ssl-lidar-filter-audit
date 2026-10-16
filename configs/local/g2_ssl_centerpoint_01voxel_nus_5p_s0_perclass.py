# R7 — 5% s0 SSL claim arm: labeled (1407) + pseudo-filtered@perclass+boost (~26k subset)
# Same as R6 baseline but pseudo ann_file uses per-class+boost filter.
_base_ = ['/home/zyf/paper/BMVC2026/configs/local/g2_sup_centerpoint_01voxel_nus_5p_s0.py']

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
                    ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_train_5p_s0_labeled.pkl',
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
                    ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_pseudo_5p_s0_perclass_boost.pkl',
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
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl',
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

load_from = '/home/zyf/paper/BMVC2026/work_dirs/R4_sup_5p_s0/epoch_10.pth'
work_dir = '/home/zyf/paper/BMVC2026/work_dirs/R7_ssl_5p_s0_perclass'
seed = 0
gpu_ids = [0]
