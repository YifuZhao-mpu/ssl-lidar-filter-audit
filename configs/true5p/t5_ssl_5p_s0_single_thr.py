# T5 student, seed 0, filter `single_thr` -- true-5% replication.
# Published student protocol (configs/local/g2_ssl_centerpoint_01voxel_nus_5p_s0_perclass.py:
# CBGS over L u P, samples_per_gpu=4, 10 epochs, eval raw epoch_10 on val_full) except:
# initialized from the T5 teacher (raw epoch_20, trained from scratch on L only), and
# pseudo-labels dumped from that teacher; every filter constant is unchanged.
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
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type, data_root=data_root,
                    ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_train_5p_s0_labeled.pkl',
                    pipeline=train_pipeline, classes=class_names,
                    test_mode=False, use_valid_flag=True,
                    box_type_3d='LiDAR', modality=input_modality),
                dict(
                    type=dataset_type, data_root=data_root,
                    ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_pseudo_t5_5p_s0_single_thr.pkl',
                    pipeline=train_pipeline, classes=class_names,
                    test_mode=False, use_valid_flag=True,
                    box_type_3d='LiDAR', modality=input_modality),
            ],
            separate_eval=False,
        ),
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
load_from = '/home/zyf/paper/BMVC2026/work_dirs/T5_sup_5p_s0/epoch_20.pth'
work_dir = '/home/zyf/paper/BMVC2026/work_dirs/T5_ssl_5p_s0_single_thr'
seed = 0
