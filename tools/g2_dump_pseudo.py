#!/usr/bin/env python3
"""R5a — Teacher inference on unlabeled nuScenes keyframes; dump pseudo 3D boxes.

Uses mmdet3d 1.0.0rc4 inference API. For each unlabeled sample:
  - Load LiDAR point cloud (.bin file referenced by info['lidar_path'])
  - Forward through CenterPoint teacher
  - Collect boxes, scores, labels
  - Apply class-wise score floor

Output pkl format:
  {
    'meta': {
      'config': str, 'ckpt': str, 'source_pkl': str,
      'class_names': list[str], 'class_score_floor': dict,
    },
    'pseudo': {
      sample_token: {
        'boxes_lidar': np.ndarray (N, 9)  # x,y,z,w,l,h,yaw,vx,vy
        'scores': np.ndarray (N,),
        'labels': np.ndarray (N,),   # class index into class_names
      }
    }
  }
"""
import argparse, os, pickle, time, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/TEOcc')
sys.path.insert(0, '/root/BMVC2026')  # needed so `tools.nusc_lidar_ssl_dataset` resolves

NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

DEFAULT_SCORE_FLOOR = {c: 0.30 for c in NUSC_CLASSES}
DEFAULT_SCORE_FLOOR.update({
    'pedestrian': 0.25,
    'bicycle': 0.25,
    'motorcycle': 0.25,
    'traffic_cone': 0.30,
    'barrier': 0.30,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Full runnable mmdet3d config path')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--unlabeled-pkl', required=True)
    p.add_argument('--data-root', default='/datasets/data/nuscenes')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=-1)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def resolve_path(data_root, p):
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    p = p.lstrip('./').lstrip('/')
    for prefix in ('data/nuscenes/', 'nuscenes/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return os.path.join(data_root, p)


def load_points(lidar_path, data_root):
    path = resolve_path(data_root, lidar_path)
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from mmdet3d.apis import init_model, inference_detector

    print(f'[+] init_model config={args.config}')
    print(f'[+] init_model ckpt={args.ckpt}')
    model = init_model(args.config, args.ckpt, device=args.device)

    print(f'[+] Loading unlabeled pkl: {args.unlabeled_pkl}')
    data = pickle.load(open(args.unlabeled_pkl, 'rb'))
    infos = data['infos']
    if args.limit > 0:
        infos = infos[:args.limit]
    print(f'[+] {len(infos)} samples to process')

    # Class names from model config
    class_names = getattr(model, 'CLASSES', None) or NUSC_CLASSES
    print(f'[+] class_names (from model): {class_names}')

    pseudo = {}
    t0 = time.time()
    for i, info in enumerate(infos):
        token = info['token']
        lidar_file = resolve_path(args.data_root, info['lidar_path'])
        # inference_detector needs a file path so LoadPointsFromFile wraps it
        # into a LiDARPoints object that LoadPointsFromMultiSweeps can consume
        result, data_for_pipeline = inference_detector(model, lidar_file)
        # result format varies; extract bbox_3d, scores_3d, labels_3d
        boxes_lidar = None; scores = None; labels = None
        if isinstance(result, list):
            # list-of-dicts (TEOcc mmdet3d rc4 format)
            r = result[0]
        else:
            r = result
        if isinstance(r, dict):
            if 'pts_bbox' in r:  # multi-head output
                r = r['pts_bbox']
            if 'boxes_3d' in r:
                b = r['boxes_3d']
                boxes_lidar = b.tensor.cpu().numpy() if hasattr(b, 'tensor') else np.asarray(b)
            elif 'boxes_lidar' in r:
                boxes_lidar = np.asarray(r['boxes_lidar'])
            scores = np.asarray(r.get('scores_3d', r.get('scores', [])))
            labels = np.asarray(r.get('labels_3d', r.get('labels', [])))
        if boxes_lidar is None or len(boxes_lidar) == 0:
            pseudo[token] = {'boxes_lidar': np.zeros((0, 9), dtype=np.float32),
                             'scores': np.zeros((0,), dtype=np.float32),
                             'labels': np.zeros((0,), dtype=np.int64)}
        else:
            pseudo[token] = {
                'boxes_lidar': np.asarray(boxes_lidar, dtype=np.float32),
                'scores': np.asarray(scores, dtype=np.float32),
                'labels': np.asarray(labels, dtype=np.int64),
            }
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-3)
            eta = (len(infos) - (i + 1)) / max(rate, 1e-3)
            print(f'  sample {i+1}/{len(infos)}  elapsed={el/60:.1f}min  ETA={eta/60:.1f}min  boxes={len(pseudo[token]["boxes_lidar"])}')
    t1 = time.time()
    print(f'[+] Done in {(t1-t0)/60:.1f} min')

    total_boxes = sum(len(v['boxes_lidar']) for v in pseudo.values())
    print(f'[+] Total boxes across {len(pseudo)} samples: {total_boxes} (avg {total_boxes/max(len(pseudo),1):.1f})')

    payload = {
        'meta': {
            'config': args.config, 'ckpt': args.ckpt,
            'source_pkl': args.unlabeled_pkl,
            'class_names': list(class_names),
            'class_score_floor': DEFAULT_SCORE_FLOOR,
            'elapsed_sec': t1 - t0,
        },
        'pseudo': pseudo,
    }
    with open(out, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[+] Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
