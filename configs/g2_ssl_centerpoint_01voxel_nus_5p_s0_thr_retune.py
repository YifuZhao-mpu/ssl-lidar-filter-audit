# R8a — 5% s0 SSL simple-alt: threshold_retune (raise teacher floor 0.30→0.40 via
# delta=0.10). Strongest non-camera baseline already identified in R5b grid; this
# is the apples-to-apples NDS test for C3.
_base_ = ['/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s0_baseline.py']

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
                    ann_file='/root/BMVC2026/infos/nuscenes_pseudo_5p_s0_thr_retune.pkl',
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

work_dir = '/root/BMVC2026/work_dirs/R8a_ssl_5p_s0_thr_retune'
