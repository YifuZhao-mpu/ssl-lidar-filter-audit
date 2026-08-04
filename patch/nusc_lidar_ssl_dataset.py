"""Custom NuScenesDataset subclass + numpy monkeypatches for TEOcc mmdet3d.

Fixes:
  1. `ann_info` not populated when `use_camera=False` — always populate in
     training mode via get_ann_info.
  2. `np.bool` was deprecated in NumPy 1.20 and removed in 1.24+; TEOcc still
     uses it in transforms_3d. Monkeypatch to keep compatibility.
  3. _format_bbox in TEOcc fork applies only ego2global (cam-time) — for
     LiDAR-only detection, predictions are in raw nuScenes LIDAR frame which is
     ~-90° rotated from ego frame; missing lidar2ego puts predictions 90° away
     from GT → NDS=0 at all distance thresholds. Override to apply
     lidar2ego ∘ ego2global from lidar-time pose.
"""
import os.path as osp
import tempfile

import mmcv
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore

import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.builder import DATASETS


@DATASETS.register_module(name='NuScenesLidarSSLDataset')
class NuScenesLidarSSLDataset(NuScenesDataset):
    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        # Always populate ann_info in training mode (TEOcc fork only sets it
        # when use_camera=True, breaking LiDAR-only training)
        if input_dict is not None and not self.test_mode and 'ann_info' not in input_dict:
            try:
                input_dict['ann_info'] = self.get_ann_info(index)
            except Exception as e:
                print(f'[WARN] get_ann_info failed at idx {index}: {e}')
        return input_dict

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format (lidar2ego patched)...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            info = self.data_infos[sample_id]
            sample_token = info['token']

            l2e_t = np.array(info['lidar2ego_translation'])
            l2e_r = pyquaternion.Quaternion(info['lidar2ego_rotation'])
            e2g_t = np.array(info['ego2global_translation'])
            e2g_r = pyquaternion.Quaternion(info['ego2global_rotation'])

            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0.0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                # lidar → ego
                nusc_box.rotate(l2e_r)
                nusc_box.translate(l2e_t)
                # ego → global
                nusc_box.rotate(e2g_r)
                nusc_box.translate(e2g_t)

                vel_norm = float(np.sqrt(nusc_box.velocity[0] ** 2 +
                                         nusc_box.velocity[1] ** 2))
                if vel_norm > 0.2:
                    if name in ('car', 'construction_vehicle', 'bus',
                                'truck', 'trailer'):
                        attr = 'vehicle.moving'
                    elif name in ('bicycle', 'motorcycle'):
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name == 'pedestrian':
                        attr = 'pedestrian.standing'
                    elif name == 'bus':
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
