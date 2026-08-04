# R7-s1 — 5% s1 SSL with perclass+boost filter (same as R7 but seed 1).
_base_ = ['/root/BMVC2026/configs/g2_sup_centerpoint_01voxel_nus_5p_s1.py']

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
                    ann_file='/root/BMVC2026/infos/nuscenes_infos_train_5p_s1_labeled.pkl',
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
                    ann_file='/root/BMVC2026/infos/nuscenes_pseudo_5p_s1_perclass_boost.pkl',
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

load_from = '/root/BMVC2026/work_dirs/R4_sup_5p_s1/epoch_10.pth'
work_dir = '/root/BMVC2026/work_dirs/R7_ssl_5p_s1_perclass_boost'
seed = 1
