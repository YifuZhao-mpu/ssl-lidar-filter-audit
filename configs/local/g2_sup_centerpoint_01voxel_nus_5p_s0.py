# R4 — 5% s0 labeled-only CenterPoint teacher training config
# Fine-tune from pretrained centerpoint_01voxel_nus (mmdet3d rc4) on 5% labeled
# subset, 20 epochs. Run from cwd = /home/zyf/TEOcc so the data/nuscenes symlink
# resolves paths like './data/nuscenes/samples/...' that R_infos pkl contains.

_base_ = [
    '/home/zyf/TEOcc/configs/_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '/home/zyf/TEOcc/configs/_base_/datasets/nus-3d.py',
    '/home/zyf/TEOcc/configs/_base_/schedules/cyclic_20e.py',
    '/home/zyf/TEOcc/configs/_base_/default_runtime.py',
]

# Keep standard nus-3d class order from the dataset base
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]

# Patch: add point_cloud_range to pts_voxel_layer (missing in TEOcc base config)
model = dict(
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    train_cfg=dict(
        pts=dict(
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        # TEOcc fork: CenterPointBBoxCoder now needs pc_range positionally
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
    ),
)

# Dataset: use our 5% seed0 labeled subset; val on val-mini for fast gates.
# Data root is relative to cwd = /home/zyf/TEOcc which has data/nuscenes symlink.
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesLidarSSLDataset'
# Register our custom dataset class
custom_imports = dict(
    imports=['tools.nusc_lidar_ssl_dataset'],
    allow_failed_imports=False,
)
file_client_args = dict(backend='disk')

# Train pipeline pulled from nus-3d base; kept identical to avoid model drift.
input_modality = dict(use_lidar=True, use_camera=False,
                      use_radar=False, use_map=False, use_external=False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_train_5p_s0_labeled.pkl',
            pipeline={{_base_.train_pipeline}},
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR',
            modality=input_modality,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_valmini.pkl',
        pipeline={{_base_.test_pipeline}},
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_infos_valmini.pkl',
        pipeline={{_base_.test_pipeline}},
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        modality=input_modality,
    ),
)
evaluation = dict(interval=5, pipeline={{_base_.eval_pipeline}})

# Override train/test pipelines to keep 5 dims from multi-sweeps (base default is
# use_dim=[0,1,2,4]=4 dims which mismatches HardSimpleVFE num_features=5)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10,
         use_dim=[0, 1, 2, 3, 4], file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
         file_client_args=file_client_args),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10,
         use_dim=[0, 1, 2, 3, 4], file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='GlobalRotScaleTrans',
                  rot_range=[0, 0], scale_ratio_range=[1., 1.],
                  translation_std=[0, 0, 0]),
             dict(type='RandomFlip3D'),
             dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
             dict(type='DefaultFormatBundle3D', class_names=class_names,
                  with_label=False),
             dict(type='Collect3D', keys=['points']),
         ]),
]
data['train']['dataset']['pipeline'] = train_pipeline
data['val']['pipeline'] = test_pipeline
data['test']['pipeline'] = test_pipeline

# Load pretrained CenterPoint ckpt for fine-tuning
load_from = '/home/zyf/paper/BMVC2026/pilot/ckpts/centerpoint_01voxel_nus.pth'

# Shorter schedule: just 20 epochs, cyclic LR from base
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# EMA hook for teacher (MEGVIIEMAHook already in TEOcc)
custom_hooks = [dict(type='MEGVIIEMAHook', init_updates=0, priority='NORMAL')]

# Disable TensorBoard for simplicity on a single GPU
work_dir = '/home/zyf/paper/BMVC2026/work_dirs/R4_sup_5p_s0'
seed = 0
gpu_ids = [0]
