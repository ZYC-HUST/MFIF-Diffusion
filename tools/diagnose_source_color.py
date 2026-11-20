"""Source-only output corrections, measured on the existing validation monitor.

No model change or retraining. GT is read only by evaluation functions.
No coefficients are fitted to GT. Original output files are never overwritten.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mfif_baseline.core import metrics, preview, atomic_json, dataset_root


def read(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255


def u8(x):
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    cv2.setNumThreads(1)
    run = ROOT / 'experiments/mfif_scratch_20261119'
    out = ROOT / 'experiments/source_color_diagnostic_20261120'
    out.mkdir(exist_ok=True)
    ids = json.loads((run / 'monitor_samples.json').read_text())
    data = dataset_root() / 'val/MFIF'
    sources = {'best': run / 'final_evaluation/val/fused',
               'last': run / 'audit_20261120/last_full_validation/val/fused'}
    records = []
    for ckpt, folder in sources.items():
        for index, sid in enumerate(ids):
            a, b = [read(data / kind / (sid + '.png')) for kind in ['A', 'B']]
            f = read(folder / (sid + '.png'))
            reference = (a + b) / 2
            mean_aligned = f + reference.mean((0, 1), keepdims=True) - f.mean((0, 1), keepdims=True)
            low = lambda x: cv2.GaussianBlur(x, (0, 0), 16., borderType=cv2.BORDER_REFLECT_101)
            low_aligned = f + .5 * (low(a) + low(b)) - low(f)
            fy, ay, by = [cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb) for x in [f,a,b]]
            fy[:, :, 1:] = .5 * (ay[:, :, 1:] + by[:, :, 1:])
            chroma = cv2.cvtColor(fy, cv2.COLOR_YCrCb2RGB)
            variants = dict(official=f, source_mean=mean_aligned, source_lowfreq_sigma16=low_aligned,
                            source_chroma_mean=chroma, source_average=reference)
            gt = read(data / 'GT' / (sid + '.png'))
            gu, lab = u8(gt), rgb2lab(gt)
            for name, pred in variants.items():
                pu = u8(pred)
                p = pu.astype(np.float32) / 255
                row = dict(checkpoint=ckpt, sample_id=sid, method=name, **metrics(pu,gu),
                    deltaE00=float(deltaE_ciede2000(rgb2lab(p),lab).mean()),
                    rgb_mean_bias=float(np.abs((p-gt).mean((0,1))).mean()),
                    Y_mae=float(np.abs(cv2.cvtColor(p,cv2.COLOR_RGB2YCrCb)[:,:,0] -
                                       cv2.cvtColor(gt,cv2.COLOR_RGB2YCrCb)[:,:,0]).mean()))
                records.append(row)
            if index < 8:
                preview(out / 'previews' / f'{ckpt}_{sid}.png',
                    [u8(x) for x in [f,mean_aligned,low_aligned,chroma,gt]],
                    ['Official','Source mean','Source low frequency','Source chroma','GT'])
        print('Finished',ckpt,flush=True)
    summary = {}
    for ckpt in sources:
        summary[ckpt] = {}
        for name in variants:
            selected = [r for r in records if r['checkpoint']==ckpt and r['method']==name]
            summary[ckpt][name] = {k:float(np.mean([r[k] for r in selected])) for k in
                ['psnr','ssim','mae','laplacian_mae','deltaE00','rgb_mean_bias','Y_mae']}
    atomic_json(out/'summary.json',dict(protocol='64 fixed validation-monitor images; sRGB float to Lab D65 via skimage; saved uint8 metrics',
        scope='Independent postprocessing diagnostic; no GT in corrections; no retraining or baseline replacement',
        sigma=16, results=summary))
    (out/'per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
