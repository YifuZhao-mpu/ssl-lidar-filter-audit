# R21ctl — volume- and class-matched control for R21. Same split-gate keep rule,
# applied to a row-wise shuffled support tensor (support and validity permuted
# jointly within class), so per-class kept counts are IDENTICAL to R21 (337,655
# boxes, verified) and only WHICH boxes are kept differs. This is the same
# construction that makes R7d the matched partner of R7c. Barrier/traffic_cone
# read scores, which are not shuffled, so those two classes are byte-identical
# to R21 and inert as a control — as the paper already found for passthrough
# classes under the R20 volume match.
#
# Why this arm exists. R7c's keep rule is
#     keep = consensus | passthrough | (~has_valid)
# and on the 5% s0 dump the passthrough branch supplies 81% of the kept boxes
# at 13-27% pseudo-precision, inheriting only the teacher's 0.10 NMS floor.
# So on 27% of the label space the score-vs-camera family comparison is not
# score gate versus camera gate but score gate versus NO gate. Every control in
# the paper (R7d row-shuffle, R20 volume-matched prune) attributes most of the
# measured cost to supervision volume and class composition rather than to the
# camera's selections; none of them removes this particular confound.
#
# R21 removes it and nothing else: exactly one variable moves against R7c.
#     COCO-mapped classes   keep = consensus | (~has_valid)   [R7c verbatim]
#     barrier, traffic_cone keep = score >= PER_CLASS_THR[cls]
# Constants are reused from the existing family; nothing is newly tuned.
#
# What it decides: whether the camera gate fails on the classes it can
# actually see, or whether the family gap was mostly the passthrough flood.
# Read against R7c (0.3059) and R7b (0.4150) on the same seed.
#
# NOTE: paths below follow the /home/zyf/paper/BMVC2026 convention of every other config
# in this directory; rewrite the prefix to match the training host.
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
                    ann_file='/home/zyf/paper/BMVC2026/infos/nuscenes_pseudo_5p_s0_split_gate_shuf.pkl',
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

work_dir = '/home/zyf/paper/BMVC2026/work_dirs/R21ctl_ssl_5p_s0_split_gate_shuf'
