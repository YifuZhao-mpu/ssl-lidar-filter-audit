#!/usr/bin/env python3
"""R2/R3 — Cache frozen Faster R-CNN V2 (COCO pretrained) detections on all
6 cameras for every keyframe in the given nuScenes infos pkl.

Output pkl format (compact):
  {
    'meta': {
        'detector': 'fasterrcnn_resnet50_fpn_v2',
        'source_pkl': str,
        'score_thr': float,
        'coco_categories': list[str],  # 91 names (COCO label set)
    },
    'dets': {
        sample_token: {
            cam_name: {
                'boxes':  np.ndarray (N, 4) xyxy,
                'scores': np.ndarray (N,),
                'labels': np.ndarray (N,) int64 COCO label,
                'img_wh': (W, H)
            }
        }
    }
  }

Notes:
  * Detections filtered by --score-thr at cache time; downstream can re-threshold.
  * Torch inference on GPU (CUDA_VISIBLE_DEVICES), batched per camera.
"""
import argparse, os, pickle, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.transforms import functional as TF

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--infos', required=True, help='Path to nuScenes infos pkl')
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--score-thr', type=float, default=0.05)
    p.add_argument('--batch-size', type=int, default=6,
                   help='Batch size per step (images across cams)')
    p.add_argument('--workers', type=int, default=4)  # not used yet
    p.add_argument('--limit', type=int, default=-1)
    return p.parse_args()


def resolve_path(data_root, p):
    """Resolve a nuScenes relative path. R_infos from TEOcc stores paths like
    './data/nuscenes/samples/CAM_FRONT/...'. Strip that prefix and rejoin with
    the real data_root (e.g. /datasets/data/nuscenes)."""
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    p = p.lstrip('./').lstrip('/')
    for prefix in ('data/nuscenes/', 'nuscenes/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return os.path.join(data_root, p)


def resolve_cam_path(data_root, cam_entry):
    return resolve_path(data_root, cam_entry['data_path'])


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'[+] Loading infos from {args.infos}')
    data = pickle.load(open(args.infos, 'rb'))
    infos = data['infos']
    if args.limit > 0:
        infos = infos[:args.limit]
    print(f'[+] {len(infos)} keyframes to process')

    device = torch.device('cuda:0')
    print(f'[+] Loading Faster R-CNN V2 on {device}')
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    coco_categories = weights.meta['categories']
    print(f'[+] {len(coco_categories)} COCO categories loaded')

    dets = {}
    t0 = time.time()
    batch_imgs = []      # list of tensors on device
    batch_meta = []      # list of (token, cam, W, H) for each image in batch

    def flush_batch():
        if not batch_imgs:
            return
        with torch.no_grad():
            preds = model(batch_imgs)
        for (token, cam, W, H), pred in zip(batch_meta, preds):
            scores = pred['scores'].cpu().numpy()
            keep = scores >= args.score_thr
            boxes = pred['boxes'][keep].cpu().numpy().astype(np.float32)
            scores = scores[keep].astype(np.float32)
            labels = pred['labels'][keep].cpu().numpy().astype(np.int64)
            dets.setdefault(token, {})[cam] = {
                'boxes': boxes, 'scores': scores,
                'labels': labels, 'img_wh': (W, H),
            }
        batch_imgs.clear()
        batch_meta.clear()

    missing = 0
    for i, info in enumerate(infos):
        token = info['token']
        for cam in CAMERAS:
            cam_entry = info['cams'][cam]
            path = resolve_cam_path(args.data_root, cam_entry)
            if not os.path.isfile(path):
                missing += 1
                continue
            img = Image.open(path).convert('RGB')
            W, H = img.size
            batch_imgs.append(TF.to_tensor(img).to(device))
            batch_meta.append((token, cam, W, H))
            if len(batch_imgs) >= args.batch_size:
                flush_batch()
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-3)
            eta = (len(infos) - (i + 1)) / max(rate, 1e-3)
            print(f'  sample {i+1}/{len(infos)}  elapsed={el/60:.1f}min  ETA={eta/60:.1f}min')
    flush_batch()
    t1 = time.time()

    print(f'[+] Finished in {(t1 - t0) / 60:.1f} min ({(t1 - t0) / 3600:.2f} GPU-hr)')
    print(f'[+] Missing images: {missing}')
    print(f'[+] Tokens cached:  {len(dets)}')

    payload = {
        'meta': {
            'detector': 'fasterrcnn_resnet50_fpn_v2',
            'source_pkl': args.infos,
            'score_thr': args.score_thr,
            'coco_categories': coco_categories,
            'cameras': CAMERAS,
            'n_keyframes': len(infos),
            'n_missing_imgs': missing,
            'elapsed_sec': t1 - t0,
        },
        'dets': dets,
    }
    print(f'[+] Writing cache to {out}')
    with open(out, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out.stat().st_size / 1e6
    print(f'[+] Wrote {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
