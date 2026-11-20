"""Fixed evaluation protocol for checkpoints produced by train_mfif.py."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from mfif_baseline.core import (MFIFTriplets, dataset_root, atomic_json, build_model,
                               ddim, evaluate, preview, to_u8)


@torch.inference_mode()
def native_fusion(model, a, b, sid, steps, seed):
    """Official test.py: reflect-pad to 16, sample, crop, then min-max normalize."""
    h, w = a.shape[-2:]
    hp, wp = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    a = torch.nn.functional.pad(a * 2 - 1, (0, wp-w, 0, hp-h), mode='reflect')
    b = torch.nn.functional.pad(b * 2 - 1, (0, wp-w, 0, hp-h), mode='reflect')
    y = ddim(model, a, b, sid, steps, seed, raw=True)[:, :, :h, :w]
    return (y - y.min()) / (y.max() - y.min())


def real_evaluation(model, output, steps, seed, rank, world):
    reports = []
    for name, pattern, replacement in [('Lytro', '*-A.jpg', ('-A.jpg', '-B.jpg')),
                                        ('MFFW', '*_A.jpg', ('_A.jpg', '_B.jpg'))]:
        root = dataset_root() / 'test' / name
        pairs = [(p, p.with_name(p.name.replace(*replacement))) for p in sorted(root.glob(pattern))]
        for i in range(rank, len(pairs), world):
            ap, bp = pairs[i]
            arrays = [np.array(Image.open(p).convert('RGB'), copy=True) for p in [ap, bp]]
            a, b = [torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda()/255 for x in arrays]
            assert a.shape == b.shape
            start = time.monotonic()
            fusion = to_u8(native_fusion(model, a, b, ap.stem, steps, seed)[0])
            out = output / name
            (out / 'fused').mkdir(parents=True, exist_ok=True)
            Image.fromarray(fusion).save(out / 'fused' / (ap.stem + '.png'))
            preview(out / 'previews' / (ap.stem + '.png'), [*arrays, fusion], ('Source A', 'Source B', 'Fusion'))
            reports.append(dict(dataset=name, sample=ap.stem, height=a.shape[-2], width=a.shape[-1],
                                seconds=time.monotonic()-start))
    if world > 1:
        chunks = [None] * world
        dist.all_gather_object(chunks, reports)
        reports = [r for chunk in chunks for r in chunk]
    report = dict(protocol='official native-size reflect-pad/sample/crop/minmax; no GT metrics',
                  steps=steps, eta=1, seed=seed, pairs=len(reports), records=reports)
    if rank == 0:
        atomic_json(output / 'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--datasets', nargs='+', default=['val', 'test', 'disk', 'wide', 'real'])
    p.add_argument('--limit', type=int, default=0, help='Smoke testing only; 0 evaluates all')
    args = p.parse_args()
    rank, world, local = [int(os.environ.get(k, default)) for k, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl', timeout=dt.timedelta(minutes=10))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    assert checkpoint['format'] in ('mfif-scratch-weights-v1', 'mfif-scratch-full-v1')
    assert checkpoint['initialization'] == 'random'
    model = build_model(checkpoint['config']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    step = checkpoint['step']
    del checkpoint
    roots = dict(val='val/MFIF', test='test/COME-MFIF',
                 disk='test/COME-MFIF/ood_disk', wide='test/COME-MFIF/ood_wide')
    output = Path(args.output)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=step, initialization='random', results={})
    for name in args.datasets:
        if name == 'real':
            result = real_evaluation(model, output / name, args.steps, args.seed, rank, world)
        else:
            dataset = MFIFTriplets(dataset_root() / roots[name])
            indices = list(range(min(args.limit, len(dataset)) if args.limit else len(dataset)))
            result = evaluate(model, dataset, indices, output / name, args.steps, args.seed,
                              previews=16, save_outputs=True, rank=rank, world=world)
        report['results'][name] = result
        if rank == 0:
            print(json.dumps(dict(event='evaluation', dataset=name, **result)), flush=True)
            atomic_json(output / 'report.json', report)
    if rank == 0:
        atomic_json(output / 'DONE.json', dict(status='complete', step=step,
                    finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
