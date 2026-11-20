"""Read-only checkpoint diagnosis. Alternative outputs are NOT baseline results."""
import contextlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mfif_baseline.core import MFIFTriplets, build_model, metrics, to_u8, preview, atomic_json
from Diffusion.diffusion import GaussianDiffusionSampler


@torch.inference_mode()
def main():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group('nccl')
    run = ROOT / 'experiments/mfif_scratch_20261119'
    out = run / 'audit_20261120'
    dataset = MFIFTriplets(ROOT.parent/'dataset/val/MFIF')
    ids = json.loads((run/'monitor_samples.json').read_text())
    lookup = {r['sample_id']: i for i, r in enumerate(dataset.rows)}
    all_records = []
    for ckpt_name in ['best', 'last']:
        ckpt = torch.load(run/f'checkpoints/{ckpt_name}.pt', map_location='cpu')
        model = build_model(ckpt['config']).cuda().eval()
        model.load_state_dict(ckpt['model'])
        step = ckpt['step']
        del ckpt
        sampler = GaussianDiffusionSampler(model, 1e-4, .02, 1000).cuda()
        for position in range(rank, len(ids), world):
            (a,b,gt), sid = dataset[lookup[ids[position]]]
            cond = torch.cat([a[None], b[None]], 1).cuda()*2-1
            gu = to_u8(gt)
            variants = [('bf16_5_seed3407',5,3407,True),('fp32_5_seed3407',5,3407,False),
                        ('bf16_20_seed3407',20,3407,True),('bf16_5_seed2026',5,2026,True)]
            for label, steps, seed, mixed in variants:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
                    raw,_ = sampler(cond,ddim=True,ddim_step=steps,ddim_eta=1.,seed=seed,type='MFF')
                assert torch.isfinite(raw).all()
                raw=raw[0].float()
                official=(raw-raw.min())/(raw.max()-raw.min())
                fixed=((raw+1)/2).clamp(0,1)
                predictions=dict(official_minmax=official,fixed_inverse_DIAGNOSTIC=fixed)
                rec=dict(checkpoint=ckpt_name,step=step,sample_id=sid,setting=label,
                    raw_min=raw.min().item(),raw_max=raw.max().item(),
                    raw_outside_training_range=((raw<-1)|(raw>1)).float().mean().item())
                for name,pred in predictions.items():
                    pu=to_u8(pred)
                    e=pu.astype(np.float64)/255-gu.astype(np.float64)/255
                    rgb_bias=e.mean((0,1))
                    rec[name]=dict(**metrics(pu,gu),mean_rgb_bias=rgb_bias.tolist(),
                        mean_abs_channel_bias=float(np.abs(rgb_bias).mean()),
                        dc_mse_fraction=float(np.mean(rgb_bias**2)/max(np.mean(e**2),1e-12)))
                # Ground-truth affine alignment is ONLY an error attribution oracle, not inference.
                if label=='bf16_5_seed3407':
                    p=official.cpu().permute(1,2,0).numpy().reshape(-1,3)
                    g=gu.astype(np.float32).reshape(-1,3)/255
                    corrected=np.empty_like(p)
                    for ch in range(3):
                        coeff=np.linalg.lstsq(np.stack([p[:,ch],np.ones(len(p))],1),g[:,ch],rcond=None)[0]
                        corrected[:,ch]=coeff[0]*p[:,ch]+coeff[1]
                    u=np.rint(corrected.reshape(gu.shape).clip(0,1)*255).astype(np.uint8)
                    rec['GT_affine_ORACLE_NOT_METHOD']=metrics(u,gu)
                    if position<8:
                        preview(out/f'previews/{ckpt_name}_{sid}.png',
                            [to_u8(a),to_u8(b),to_u8(official),to_u8(fixed),gu],
                            ['A','B','Official output','Fixed inverse (diagnostic)','GT'])
                all_records.append(rec)
            if position%8==rank:
                print(json.dumps(dict(checkpoint=ckpt_name,position=position,rank=rank)),flush=True)
        del sampler, model
        torch.cuda.empty_cache()
    gathered=[None]*world
    dist.all_gather_object(gathered,all_records)
    if rank==0:
        rows=[r for chunk in gathered for r in chunk]
        (out/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        summary={}
        for ckpt in ['best','last']:
            summary[ckpt]={}
            for setting in sorted({r['setting'] for r in rows}):
                selected=[r for r in rows if r['checkpoint']==ckpt and r['setting']==setting]
                result={name:{k:float(np.mean([r[name][k] for r in selected]))
                    for k in ['psnr','ssim','laplacian_mae','mean_abs_channel_bias','dc_mse_fraction']}
                    for name in ['official_minmax','fixed_inverse_DIAGNOSTIC']}
                result.update(n=len(selected),raw_min_mean=float(np.mean([r['raw_min'] for r in selected])),
                    raw_max_mean=float(np.mean([r['raw_max'] for r in selected])),
                    outside_training_range=float(np.mean([r['raw_outside_training_range'] for r in selected])))
                if setting=='bf16_5_seed3407':
                    result['GT_affine_ORACLE_NOT_METHOD']={k:float(np.mean([r['GT_affine_ORACLE_NOT_METHOD'][k] for r in selected])) for k in ['psnr','ssim','laplacian_mae']}
                summary[ckpt][setting]=result
        atomic_json(out/'diagnostics_summary.json',summary)
        print(json.dumps(summary),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
