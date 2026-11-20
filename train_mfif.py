"""Scratch-only MFIF adaptation; launch with torchrun or plain python on one GPU."""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from mfif_baseline.core import (MFIFTriplets, Objective, ResumeBatchSampler, REPO, UPSTREAM, read_config,
    atomic_json, atomic_torch, build_model, evaluate,
    monitor_indices, restore_rng, rng_state, seed_all)
from Scheduler import GradualWarmupScheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', help='Only a complete checkpoint produced by this scratch trainer')
    parser.add_argument('--stop-after', type=int, help='Testing/controlled stop; does not alter LR schedule')
    args = parser.parse_args()
    config = read_config(args.config)
    if any(config.get(k) for k in ('pretrained', 'pretrained_path', 'weights', 'init_checkpoint')):
        raise ValueError('This entry point permits random initialization and own full-state resume only')
    rank, world, local = [int(os.environ.get(k, default)) for k, default in
                          [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    torch.cuda.set_device(local)
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl', timeout=dt.timedelta(minutes=8))
    out = Path(config['output'])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'config.json').exists() and not args.resume:
            raise FileExistsError(f'{out} already contains a run; choose a fresh output or --resume')
        atomic_json(out / 'config.json', config)
        if not args.resume:
            snap = out / 'code'
            shutil.copytree(REPO / 'mfif_baseline', snap / 'mfif_baseline', ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('train_mfif.py', 'evaluate_mfif.py', 'run_mfif_overnight.py'):
                if (REPO / name).exists():
                    shutil.copy2(REPO / name, snap / name)
            shutil.copytree(UPSTREAM / 'Diffusion', snap / 'third_party/Mask-DiFuser/Diffusion', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(UPSTREAM / 'loss', snap / 'third_party/Mask-DiFuser/loss', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(UPSTREAM / 'Scheduler.py', snap / 'third_party/Mask-DiFuser/Scheduler.py')
    if world > 1:
        dist.barrier()
    seed_all(config['seed'])
    model = build_model(config).cuda()
    fingerprint = hashlib.sha256()
    for name, param in model.named_parameters():
        fingerprint.update(name.encode())
        fingerprint.update(param.detach().flatten()[:16].float().cpu().numpy().tobytes())
    initial_fingerprint = fingerprint.hexdigest()
    if world > 1:
        wrapped = DDP(model, device_ids=[local], find_unused_parameters=True,
                      gradient_as_bucket_view=True, broadcast_buffers=False)
    else:
        wrapped = model
    objective = Objective(wrapped, config['change_epoch']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2., warm_epoch=1000,
                                      after_scheduler=cosine)
    train = MFIFTriplets(config['train_root'])
    train_indices = None
    if config.get('train_limit'):
        generator = torch.Generator().manual_seed(config['seed'] + 99)
        train_indices = torch.randperm(len(train), generator=generator)[:config['train_limit']].tolist()
        train = Subset(train, train_indices)
    val = MFIFTriplets(config['val_root'])
    val_indices = monitor_indices(val, config['monitor_count'], config['eval_seed'])
    batch, accumulation = config['batch_per_gpu'], config['accumulation']
    assert len(train) >= world * batch * accumulation
    step, epoch, cursor, best = 0, 0, 0, -float('inf')
    training_seconds = 0.
    seed_all(config['seed'] + rank * 100003 + 1)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        assert checkpoint['format'] == 'mfif-scratch-full-v1'
        assert checkpoint['initialization'] == 'random'
        assert checkpoint['world_size'] == world
        # Output/deadline/workers may change, numerical/data settings may not.
        for key in ('seed', 'batch_per_gpu', 'accumulation', 'lr', 'max_steps',
                    'schedule_epoch_steps', 'change_epoch', 'train_root', 'train_limit', 'dropout'):
            assert checkpoint['config'].get(key) == config.get(key), f'Resume mismatch: {key}'
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict({k: v for k, v in checkpoint['scheduler'].items() if k != 'after_scheduler'})
        cosine.load_state_dict(checkpoint['cosine_scheduler'])
        step, epoch, cursor = [checkpoint[k] for k in ('step', 'epoch', 'cursor')]
        best, training_seconds = checkpoint['best_psnr'], checkpoint['training_seconds']
        restore_rng(checkpoint['rng_by_rank'][rank])
        initial_fingerprint = checkpoint['initial_fingerprint']
        del checkpoint
    if rank == 0:
        if not args.resume:
            provenance = dict(initialization='random', official_weights_loaded=False,
                initial_parameter_fingerprint=initial_fingerprint, seed=config['seed'],
                architecture_parameters=sum(p.numel() for p in model.parameters()),
                frozen_perceptual_loss='LPIPS VGG (pretrained loss network only)',
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                upstream_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip(),
                torch=torch.__version__, cuda=torch.version.cuda, world_size=world,
                effective_batch=world * batch * accumulation, train_triplets=len(train),
                started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                nccl_p2p_disable=os.environ.get('NCCL_P2P_DISABLE'),
                sampler='unchanged official MFF DDIM, eta=1, common seed, min-max normalization',
                official_file_hashes={str(p.relative_to(UPSTREAM)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [UPSTREAM/'Diffusion/model.py', UPSTREAM/'Diffusion/diffusion.py', UPSTREAM/'loss/Myloss.py']})
            atomic_json(out / 'provenance.json', provenance)
            atomic_json(out / 'monitor_samples.json', [val.rows[i]['sample_id'] for i in val_indices])
            if train_indices is not None:
                atomic_json(out / 'train_indices.json', train_indices)
        print(json.dumps(dict(event='started', step=step, world=world, output=str(out),
                              initialization='random', resumed=bool(args.resume))), flush=True)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_checkpoint(reason):
        states = [rng_state()]
        if world > 1:
            states = [None] * world
            dist.all_gather_object(states, rng_state())
        if rank == 0:
            saved_at = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_torch(out / 'checkpoints/last.pt', dict(format='mfif-scratch-full-v1',
                initialization='random', model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler={k: v for k, v in scheduler.state_dict().items() if k != 'after_scheduler'},
                cosine_scheduler=cosine.state_dict(),
                amp=dict(dtype='bfloat16', scaler=None), step=step, epoch=epoch, cursor=cursor,
                best_psnr=best, world_size=world, config=config, rng_by_rank=states,
                initial_fingerprint=initial_fingerprint, training_seconds=training_seconds, reason=reason,
                saved_utc=saved_at))
            atomic_json(out / 'checkpoints/last.json', dict(step=step, epoch=epoch, cursor=cursor,
                reason=reason, saved_utc=saved_at, initialization='random'))
            print(json.dumps(dict(event='checkpoint', step=step, reason=reason)), flush=True)
        if world > 1:
            dist.barrier()

    def validate():
        nonlocal best
        # Evaluation must not perturb subsequent training, including future library internals.
        state = rng_state()
        report = evaluate(model, val, val_indices, out / f'validation/step_{step:07d}',
                          config['ddim_steps'], config['eval_seed'], rank=rank, world=world)
        restore_rng(state)
        improved = report['fusion']['psnr'] > best
        best = max(best, report['fusion']['psnr'])
        if rank == 0:
            event = dict(event='validation', step=step, **report)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)
            if improved:
                atomic_torch(out / 'checkpoints/best.pt', dict(format='mfif-scratch-weights-v1',
                    initialization='random', model=model.state_dict(), step=step,
                    config=config, validation=report))
                atomic_json(out / 'checkpoints/best.json', dict(step=step, validation=report,
                                                              initialization='random'))
        if world > 1:
            dist.barrier()

    if step == 0 and config.get('initial_eval', True):
        validate()
    last_validation = step if step == 0 and config.get('initial_eval', True) else -1
    log_values = torch.zeros(9, device='cuda', dtype=torch.float64)
    log_count, log_start = 0, time.monotonic()
    stop_reason = 'max_steps'
    while step < config['max_steps']:
        sampler = ResumeBatchSampler(len(train), config['seed'], epoch, rank, world,
                                     batch, accumulation, cursor)
        if not len(sampler):
            epoch, cursor = epoch + 1, 0
            continue
        loader = DataLoader(train, batch_sampler=sampler, num_workers=config['workers'],
                            pin_memory=True, generator=torch.Generator().manual_seed(config['seed'] + epoch),
                            persistent_workers=False)
        iterator = iter(loader)
        for _ in range(len(sampler) // accumulation):
            started = time.monotonic()
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad(set_to_none=True)
            values = torch.zeros(7, device='cuda')
            schedule_epoch = step // config['schedule_epoch_steps'] + 1
            auxiliary = schedule_epoch > config['change_epoch']
            for micro in range(accumulation):
                images, _ids = next(iterator)
                a, b, gt = [x.cuda(non_blocking=True) * 2 - 1 for x in images]
                sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else contextlib.nullcontext()
                with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = objective(gt, torch.cat([a, b], 1), schedule_epoch)
                    loss = losses[0] / accumulation
                loss.backward()
                values += torch.stack([x.detach().float() for x in losses]) / accumulation
                cursor += 1
            # Original train.py leaves gradient clipping commented out. Measure only.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            finite = torch.tensor(int(torch.isfinite(grad_norm) and torch.isfinite(values).all()), device='cuda')
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Non-finite loss/gradient at step {step}; last checkpoint retained')
            optimizer.step()
            step += 1
            if step % config['schedule_epoch_steps'] == 0:
                scheduler.step()
            elapsed = time.monotonic() - started
            training_seconds += elapsed
            log_values[:7] += values.double()
            log_values[7] += grad_norm.double()
            log_values[8] += elapsed
            log_count += 1
            if step % config['log_every'] == 0 or step <= 3:
                if world > 1:
                    dist.all_reduce(log_values)
                means = (log_values / (log_count * world)).tolist()
                if rank == 0:
                    record = dict(event='train', step=step, epoch=epoch, cursor=cursor,
                        schedule_epoch=schedule_epoch,
                        loss=means[0], noise_mse=means[1]/10, color_x10=means[2],
                        image_mse_x10=means[3], exposure_x10=means[4], ssim_x10=means[5], lpips_x10=means[6],
                        grad_norm=means[7], seconds_per_step=means[8], lr=lr,
                        loss_phase='full' if auxiliary else 'epsilon', best_psnr=best if best > -float('inf') else None,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        time_utc=dt.datetime.now(dt.timezone.utc).isoformat(), status='running',
                        deadline_utc=config.get('deadline_utc'), samples_seen=step*batch*world*accumulation)
                    with (out / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record, allow_nan=False) + '\n')
                    atomic_json(out / 'status.json', record)
                    print(json.dumps(record), flush=True)
                log_values.zero_()
                log_count, log_start = 0, time.monotonic()
            should_stop = int(bool(stop_requested or (args.stop_after and step >= args.stop_after) or
                              (config.get('deadline_unix') and time.time() >= config['deadline_unix'])))
            stop_tensor = torch.tensor(should_stop, device='cuda')
            if world > 1:
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())
            if step % config['eval_every'] == 0:
                validate()
                last_validation = step
                log_start = time.monotonic()
            if step % config['checkpoint_every'] == 0:
                save_checkpoint('periodic')
            if should_stop or step >= config['max_steps']:
                stop_reason = 'deadline_or_requested_stop' if should_stop else 'max_steps'
                break
        del iterator, loader
        if should_stop or step >= config['max_steps']:
            break
        epoch, cursor = epoch + 1, 0
    if last_validation != step:
        validate()
    save_checkpoint(stop_reason)
    if rank == 0:
        atomic_json(out / 'completion.json', dict(status='training_finished', step=step,
            best_monitor_psnr=best, reason=stop_reason, training_seconds=training_seconds,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
