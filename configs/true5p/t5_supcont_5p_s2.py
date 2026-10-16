# T5 labeled-only continuation, seed 2 -- true-5% replication.
# Same schedule as a T5 student (10 epochs from the T5 teacher's raw epoch_20,
# CBGS, samples_per_gpu=4) but trained on L only: controls for the students'
# extra training epochs when reading SSL-vs-labeled-only.
_base_ = ['/home/zyf/paper/BMVC2026/configs/local/g2_ssl_centerpoint_01voxel_nus_5p_s0_perclass.py']
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
            _delete_=True,
            type=dataset_type, data_root=data_root,
            ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_train_5p_s2_labeled.pkl',
            pipeline=train_pipeline, classes=class_names,
            test_mode=False, use_valid_flag=True,
            box_type_3d='LiDAR', modality=input_modality),
    ),
    val=dict(
        type=dataset_type, data_root=data_root,
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl',
        pipeline=test_pipeline, classes=class_names, test_mode=True,
        box_type_3d='LiDAR', modality=input_modality),
    test=dict(
        type=dataset_type, data_root=data_root,
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_val_full.pkl',
        pipeline=test_pipeline, classes=class_names, test_mode=True,
        box_type_3d='LiDAR', modality=input_modality),
)
load_from = '/home/zyf/paper/BMVC2026/work_dirs/T5_sup_5p_s2/epoch_20.pth'
work_dir = '/home/zyf/paper/BMVC2026/work_dirs/T5_supcont_5p_s2'
seed = 2
