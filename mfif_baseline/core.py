import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
UPSTREAM = REPO / 'third_party' / 'Mask-DiFuser'
sys.path.insert(0, str(UPSTREAM))
from mfif_baseline.data import MFIFTriplets
from Diffusion.model import UNet
from Diffusion.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler


def read_config(path):
    config = json.loads(Path(path).read_text())
    replacements = {'${PROJECT_ROOT}': str(REPO),
                    '${DATA_ROOT}': os.environ.get('MFIF_DATA_ROOT', str(WORKSPACE / 'dataset'))}
    for key in ('output', 'train_root', 'val_root'):
        if key in config:
            for token, value in replacements.items():
                config[key] = config[key].replace(token, value)
    return config


def dataset_root():
    return Path(os.environ.get('MFIF_DATA_ROOT', str(WORKSPACE / 'dataset')))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def build_model(config):
    # Intentionally no pretrained argument or checkpoint lookup here.
    return UNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2],
                num_res_blocks=2, dropout=config.get('dropout', 0.1))


class Objective(GaussianDiffusionTrainer):
    def __init__(self, model, change_epoch=200):
        super().__init__(model, 1e-4, 0.02, 1000, args=SimpleNamespace(change_epoch=change_epoch))
        self.loss_fn_vgg.requires_grad_(False).eval()

    def forward(self, gt, cond, epoch):
        return [loss.mean() for loss in super().forward(gt, cond, epoch)]


@torch.inference_mode()
def ddim(model, a, b, sample_id, steps=5, seed=3407, raw=False):
    """Call the unchanged official sampler with its MFF inference settings.

    Preserve t+1 indexing, final alpha, [-1.5,1] clipping, eta=1, common seed,
    and min-max output normalization. Outer evaluation saves/restores training RNG.
    """
    assert a.shape[0] == 1 and b.shape == a.shape
    sampler = GaussianDiffusionSampler(model, 1e-4, .02, 1000).to(a.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y, _ = sampler(torch.cat([a, b], 1), ddim=True, ddim_step=steps,
                       ddim_eta=1., seed=seed, type='MFF')
    return y if raw else (y - y.min()) / (y.max() - y.min())


def to_u8(tensor):
    return tensor.detach().float().clamp(0, 1).mul(255).round().byte().cpu().permute(1, 2, 0).numpy()


def metrics(pred, gt):
    p, g = pred.astype(np.float32) / 255, gt.astype(np.float32) / 255
    error = p - g
    pg = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    gg = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    return dict(psnr=float(-10 * np.log10(max(float(np.mean(error**2)), 1e-12))),
                ssim=float(structural_similarity(p, g, channel_axis=2, data_range=1.,
                           gaussian_weights=True, sigma=1.5, win_size=11,
                           use_sample_covariance=False)),
                mae=float(np.abs(error).mean()),
                laplacian_mae=float(np.abs(cv2.Laplacian(pg, cv2.CV_32F) -
                                          cv2.Laplacian(gg, cv2.CV_32F)).mean()))


def preview(path, images, names=('Source A', 'Source B', 'Average', 'Fusion', 'GT')):
    w, h = images[0].shape[1], images[0].shape[0]
    canvas = Image.new('RGB', (w * len(images), h + 24), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, name) in enumerate(zip(images, names)):
        canvas.paste(Image.fromarray(im), (i * w, 24))
        draw.text((i * w + 5, 5), name, fill='black')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def monitor_indices(dataset, count, seed):
    # One variant from each content, then deterministic sampling across contents.
    groups = {}
    for i, row in enumerate(dataset.rows):
        content = row['sample_id'].rsplit('_v', 1)[0]
        groups.setdefault(content, []).append(i)
    rng = random.Random(seed)
    selected = [rng.choice(indices) for _, indices in sorted(groups.items())]
    rng.shuffle(selected)
    return selected[:count] if count else list(range(len(dataset)))


@torch.inference_mode()
def evaluate(model, dataset, indices, out_dir, steps=5, seed=3407, previews=8,
             save_outputs=False, rank=0, world=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training = model.training
    model.eval()
    records = []
    for position in range(rank, len(indices), world):
        (a, b, gt), sid = dataset[indices[position]]
        pred = ddim(model, a[None].cuda() * 2 - 1, b[None].cuda() * 2 - 1,
                    sid, steps, seed)[0]
        au, bu, gu, pu = [to_u8(x) for x in [a, b, gt, pred]]
        avg = to_u8((a + b) / 2)
        records.append(dict(sample_id=sid, fusion=metrics(pu, gu), average=metrics(avg, gu)))
        if save_outputs:
            (out_dir / 'fused').mkdir(exist_ok=True)
            Image.fromarray(pu).save(out_dir / 'fused' / (sid + '.png'))
        if position < previews:
            preview(out_dir / 'previews' / (sid + '.png'), [au, bu, avg, pu, gu])
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [row for chunk in gathered for row in chunk]
    assert len(records) == len(indices) and len({r['sample_id'] for r in records}) == len(indices)
    summary = {method: {metric: float(np.mean([r[method][metric] for r in records]))
                        for metric in ('psnr', 'ssim', 'mae', 'laplacian_mae')}
               for method in ('fusion', 'average')}
    summary.update(count=len(records), ddim_steps=steps, eta=1, seed=seed,
                   protocol='official-MFF-sampler_minmax_rgb-u8_v1')
    if rank == 0:
        atomic_json(out_dir / 'summary.json', summary)
        with (out_dir / 'per_image.jsonl').open('w') as f:
            for row in sorted(records, key=lambda r: r['sample_id']):
                f.write(json.dumps(row, allow_nan=False) + '\n')
    model.train(training)
    return summary


class ResumeBatchSampler:
    """Deterministic epoch shuffle and rank shard; skip consumed indices before decoding."""
    def __init__(self, size, seed, epoch, rank, world, batch, accumulation, cursor=0):
        generator = torch.Generator().manual_seed(seed + epoch)
        unit = world * batch * accumulation
        usable = size // unit * unit
        order = torch.randperm(size, generator=generator).tolist()[:usable]
        local = order[rank::world]
        self.batches = [local[i:i + batch] for i in range(0, len(local), batch)][cursor:]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
