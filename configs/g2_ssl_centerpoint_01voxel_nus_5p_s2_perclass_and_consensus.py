# Seed-2 replication of the R19 conjunctive control, launched 2026-08-12 under
# review-stage/PREREG_2026-08-12.md. The seed-0 arm is the paper's co-headline
# result and carries an explicit "seed 0 only" caveat; these two remove it.
# R19 — 5% s2 SSL: CONJUNCTIVE control, per-class threshold AND six-camera
# consensus.
#
# This is the gate-matched counterpart to the audited camera-support family.
# The camera-support arms (R8b/R8c/R8d/R7c) apply NO teacher-confidence
# condition, so their 7-11 NDS deficit against the score family confounds
# "camera evidence is bad" with "the confidence floor was removed". R19
# removes that confound: the keep rule is perclass_only AND the R7c consensus
# rule, so the kept set is a strict subset of perclass_only (R7b) and camera
# support can only ever *veto*, never admit.
#
# Read against its two neighbours:
#   R7b  perclass_only        camera plays no role            (the base)
#   R18  hybrid_posonly       camera may only ADD boxes       -> -0.26/+0.11
#   R19  perclass_and_consensus  camera may only REMOVE boxes (this run)
# R18 and R19 together bracket the whole space of camera support subordinated
# to a confidence gate. R18 already reads null; if R19 also reads null, the
# claim "camera support adds nothing once confidence is gated" no longer
# depends on the un-gated family comparison at all.
#
# Everything except the pseudo-label ann_file is inherited from the same base
# as every other arm: same initialization, same schedule, same CBGS wrapper,
# same epoch count.
_base_ = ['/root/BMVC2026/configs/g2_ssl_centerpoint_01voxel_nus_5p_s2_perclass.py']

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
                    ann_file='/root/BMVC2026/infos/nuscenes_infos_train_5p_s2_labeled.pkl',
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
                    ann_file='/root/BMVC2026/infos/nuscenes_pseudo_5p_s2_perclass_and_consensus.pkl',
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

work_dir = '/root/BMVC2026/work_dirs/R19_ssl_5p_s2_perclass_and_consensus'
