"""Detached, bounded overnight training and final evaluation supervisor."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from mfif_baseline.report import make_report
from mfif_baseline.core import read_config


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NCCL_P2P_DISABLE='1',
               PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8']
    atomic(out / 'supervisor.json', dict(pid=os.getpid(), start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                       config=str(config_path)))

    def run(command, log, phase, max_seconds):
        started = time.monotonic()
        last_report = 0.
        with (out / log).open('a') as stream:
            stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
            stream.flush()
            proc = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                atomic(out / 'job_status.json', dict(status='running', phase=phase,
                    supervisor_pid=os.getpid(), worker_pid=proc.pid,
                    heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic()-started))
                if time.monotonic()-last_report > 300 and (out/'metrics.jsonl').exists():
                    try:
                        make_report(out)
                    except Exception:
                        traceback.print_exc()
                    last_report = time.monotonic()
                if time.monotonic()-started > max_seconds:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    return -124
                time.sleep(30)
            return proc.returncode

    try:
        command = launch + [str(repo/'train_mfif.py'), '--config', str(config_path)]
        last = out / 'checkpoints/last.pt'
        if last.exists():
            command += ['--resume', str(last)]
        code = run(command, 'train.log', 'training', max(300, config['deadline_unix']-time.time()+1200))
        # One bounded recovery attempt; never repeatedly restart a failing experiment.
        if code and last.exists() and time.time() < config['deadline_unix']-600:
            code = run(launch + [str(repo/'train_mfif.py'), '--config', str(config_path), '--resume', str(last)],
                       'train.log', 'training_resume', config['deadline_unix']-time.time()+1200)
        atomic(out/'training_exit.json', dict(exit_code=code, time_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        best = out / 'checkpoints/best.pt'
        if not best.exists():
            raise RuntimeError('No validation-selected scratch checkpoint is available')
        evaluation_code = run(launch + [str(repo/'evaluate_mfif.py'), '--checkpoint', str(best),
            '--output', str(out/'final_evaluation'), '--steps', str(config['ddim_steps'])],
            'evaluate.log', 'final_evaluation', 3600)
        make_report(out)
        atomic(out / 'job_status.json', dict(status='complete' if code == evaluation_code == 0 else 'partial_failure',
            training_exit_code=code, evaluation_exit_code=evaluation_code,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), supervisor_pid=os.getpid()))
    except Exception as exc:
        traceback.print_exc()
        atomic(out / 'job_status.json', dict(status='failed', error=str(exc), supervisor_pid=os.getpid(),
                                             time_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
