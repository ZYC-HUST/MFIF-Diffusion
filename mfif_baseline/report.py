"""Write an honest, local training status report and scientific learning curves."""
import argparse
import json
from pathlib import Path


def make_report(run):
    run = Path(run)
    events = []
    path = run / 'metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrently appended final line may be incomplete.
    train = {r['step']: r for r in events if r['event'] == 'train'}
    val = {r['step']: r for r in events if r['event'] == 'validation'}
    train, val = [train[k] for k in sorted(train)], [val[k] for k in sorted(val)]
    config = json.loads((run / 'config.json').read_text())
    complete = run / 'completion.json'
    lines = ['Mask-DiFuser / COME-MFIF scratch experiment', '',
             'Initialization: RANDOM. Official fusion checkpoint is never loaded.',
             'Frozen LPIPS/VGG is used only as the original perceptual loss.',
             'Official model, losses and MFF sampler are used unchanged.',
             'Inference: DDIM 5 steps, eta=1, seed=3407, official clipping/indexing/min-max.',
             'Metrics: mean per-image RGB uint8 PSNR/SSIM; these are not the original gray SSIM.',
             'Training: 256x256, BF16, 8 GPUs, 1 image/GPU, accumulation=4 (global batch 32).',
             'AdamW lr=5e-5, wd=1e-4; official warmup/cosine; no gradient clipping.',
             f"Schedule epoch = {config['schedule_epoch_steps']} updates; auxiliary losses after epoch {config['change_epoch']}.",
             'Main schedule mapping: DIV2K 800 / original global batch 32 = 25 updates/reference epoch.',
             'Full MFIF data pass is tracked separately; final incomplete global batch is dropped.',
             'The checkpoint selected by fixed validation-monitor PSNR is used for final evaluation.', '']
    lines.append('Training status: ' + ('finished (see completion.json)' if complete.exists() else 'in progress; inspect job_status.json and status.json'))
    if train:
        last = train[-1]
        lines += [f"Last logged step: {last['step']}; samples seen: {last['samples_seen']}",
                  f"Last logged time (UTC): {last['time_utc']}",
                  f"Noise MSE: {last['noise_mse']:.6f}; loss phase: {last['loss_phase']}",
                  f"Recent step time: {last['seconds_per_step']:.3f}s"]
    lines += [f"Training deadline (UTC): {config.get('deadline_utc', 'not set')}", '',
              'Fixed validation monitor (not the complete validation set):',
              'step    PSNR(dB)    SSIM       Laplacian-MAE    Average-PSNR(dB)']
    for r in val:
        f, a = r['fusion'], r['average']
        lines.append(f"{r['step']:6d}  {f['psnr']:9.4f}  {f['ssim']:.6f}   {f['laplacian_mae']:.6f}         {a['psnr']:.4f}")
    final = run / 'final_evaluation/report.json'
    if final.exists():
        report = json.loads(final.read_text())
        lines += ['', f"Final evaluation checkpoint step: {report['step']}",
                  'dataset    N     PSNR(dB)    SSIM       Laplacian-MAE   Average-PSNR(dB)']
        for name, r in report['results'].items():
            if name == 'real':
                lines.append(f"Real images: {r['pairs']} pairs; native-size previews only, no GT metrics.")
            else:
                f, a = r['fusion'], r['average']
                lines.append(f"{name:8s} {r['count']:5d}  {f['psnr']:9.4f}  {f['ssim']:.6f}   {f['laplacian_mae']:.6f}        {a['psnr']:.4f}")
        lines.append('Final evaluation status: ' + ('complete' if (run/'final_evaluation/DONE.json').exists() else 'in progress or interrupted'))
    lines += ['', 'Do not interpret lower training loss as successful fusion or convergence.',
              'Compare fusion with the simple source average and inspect detail/brightness in previews.',
              'OOD results are synthetic robustness checks; real-image previews are qualitative.',
              'Paths: checkpoints/{last,best}.pt, validation/, final_evaluation/, metrics.jsonl, training_curves.png.']
    tmp = run / 'REPORT.txt.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(run / 'REPORT.txt')
    if not train:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].semilogy([x['step'] for x in train], [x['noise_mse'] for x in train])
    axes[0, 0].set_title('Training noise MSE (log scale)')
    for ax, key, title in [(axes[0,1], 'psnr', 'Validation monitor PSNR (dB)'),
                            (axes[1,0], 'ssim', 'Validation monitor SSIM'),
                            (axes[1,1], 'laplacian_mae', 'Validation monitor Laplacian MAE')]:
        if val:
            ax.plot([x['step'] for x in val], [x['fusion'][key] for x in val], 'o-', label='Fusion')
            ax.axhline(val[0]['average'][key], ls='--', color='gray', label='Source average')
            ax.legend()
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=.2)
    fig.suptitle('Mask-DiFuser: random initialization, original MFF sampler')
    fig.savefig(run / 'training_curves.tmp.png', dpi=150)
    (run / 'training_curves.tmp.png').replace(run / 'training_curves.png')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    make_report(p.parse_args().run)
