#!/usr/bin/env python3
"""R1 — generate deterministic labeled/unlabeled subsets of nuScenes trainval.

Reuses the existing nuscenes_R_infos_train.pkl from TEOcc (radar fields are
kept but ignored by LiDAR-only training). Keyframe-level sampling to match
standard semi-supervised 3D detection protocol (3DIoUMatch, MVAT).

Outputs per (ratio, seed):
  - nuscenes_infos_train_<R>p_s<S>_labeled.pkl    # size = round(ratio * 28130)
  - nuscenes_infos_train_<R>p_s<S>_unlabeled.pkl  # size = 28130 - labeled

Also emits:
  - nuscenes_infos_valmini.pkl  # first N val keyframes for fast gates
  - subset_manifest.json        # reproducibility manifest

Note: pkl structure preserved identically to input — {'infos': [...], 'metadata': {...}}
"""
import argparse, json, pickle, random
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src-train-pkl', required=True)
    p.add_argument('--src-val-pkl', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ratios', nargs='+', type=float, default=[0.01, 0.05, 0.10])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--val-mini-count', type=int, default=4850,
                   help='Keep first N val keyframes for fast-gate val-mini')
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[+] Loading train: {args.src_train_pkl}')
    train = pickle.load(open(args.src_train_pkl, 'rb'))
    print(f'[+] Loading val: {args.src_val_pkl}')
    val = pickle.load(open(args.src_val_pkl, 'rb'))

    infos_train = train['infos']
    infos_val = val['infos']
    n_train = len(infos_train)
    n_val = len(infos_val)
    print(f'[+] n_train={n_train}, n_val={n_val}')

    manifest = {
        'source_train': args.src_train_pkl,
        'source_val': args.src_val_pkl,
        'n_train': n_train,
        'n_val': n_val,
        'sampling': 'keyframe-level, deterministic with np.random.RandomState',
        'splits': {},
    }

    # Generate labeled/unlabeled for each (ratio, seed)
    for ratio in args.ratios:
        n_lab = int(round(ratio * n_train))
        rp = f'{int(round(ratio * 100))}p'
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            idx_lab = sorted(rng.choice(n_train, size=n_lab, replace=False).tolist())
            idx_set = set(idx_lab)
            idx_unlab = [i for i in range(n_train) if i not in idx_set]

            lab_infos = [infos_train[i] for i in idx_lab]
            unlab_infos = [infos_train[i] for i in idx_unlab]

            lab_out = out / f'nuscenes_infos_train_{rp}_s{seed}_labeled.pkl'
            unlab_out = out / f'nuscenes_infos_train_{rp}_s{seed}_unlabeled.pkl'
            pickle.dump({'infos': lab_infos, 'metadata': train.get('metadata', {})},
                        open(lab_out, 'wb'))
            pickle.dump({'infos': unlab_infos, 'metadata': train.get('metadata', {})},
                        open(unlab_out, 'wb'))

            key = f'{rp}_s{seed}'
            manifest['splits'][key] = {
                'ratio': ratio,
                'seed': seed,
                'n_labeled': len(lab_infos),
                'n_unlabeled': len(unlab_infos),
                'labeled_pkl': str(lab_out),
                'unlabeled_pkl': str(unlab_out),
                'first5_labeled_tokens': [lab_infos[i]['token'] for i in range(min(5, len(lab_infos)))],
            }
            print(f'  [{key}] labeled={len(lab_infos)}  unlabeled={len(unlab_infos)}  -> {lab_out.name}')

    # Val-mini
    n_vm = min(args.val_mini_count, n_val)
    infos_vm = infos_val[:n_vm]
    vm_out = out / 'nuscenes_infos_valmini.pkl'
    pickle.dump({'infos': infos_vm, 'metadata': val.get('metadata', {})},
                open(vm_out, 'wb'))
    manifest['val_mini'] = {'n': n_vm, 'pkl': str(vm_out)}
    print(f'[+] val-mini: {n_vm} -> {vm_out.name}')

    # Copy full val for final eval convenience
    full_val_out = out / 'nuscenes_infos_val_full.pkl'
    pickle.dump({'infos': infos_val, 'metadata': val.get('metadata', {})},
                open(full_val_out, 'wb'))
    manifest['val_full'] = {'n': n_val, 'pkl': str(full_val_out)}
    print(f'[+] val-full: {n_val} -> {full_val_out.name}')

    man_out = out / 'subset_manifest.json'
    json.dump(manifest, open(man_out, 'w'), indent=2)
    print(f'[+] Manifest: {man_out}')

    # Sanity
    expected = {0.01: 281, 0.05: 1407, 0.10: 2813}
    for ratio in args.ratios:
        n_lab = int(round(ratio * n_train))
        rp = f'{int(round(ratio * 100))}p'
        want = expected.get(ratio)
        if want is not None and n_lab != want:
            print(f'[WARN] {rp}: got {n_lab}, expected {want} (off by {n_lab - want})')
        else:
            print(f'[OK]   {rp}: {n_lab}')


if __name__ == '__main__':
    main()
