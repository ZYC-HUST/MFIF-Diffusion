MFIF-Diffusion
=============

Multi-focus image fusion research: source observations in diffusion sampling,
and preservation of real image details. Current baseline: a scratch-trained
Mask-DiFuser adaptation on COME-derived RGB multi-focus triplets.

Project layout
--------------
train_mfif.py / evaluate_mfif.py   Our training and evaluation entry points
run_mfif_overnight.py             Bounded training/evaluation supervisor
mfif_baseline/                   Dataset, configuration, logging, evaluation helpers
configs/                        Versioned run configurations
third_party/Mask-DiFuser/         Upstream code pinned as a Git submodule
docs/                           Protocols and implementation/research notes
reports/baseline_v0/             Small historical logs, curves and audit summaries
tools/                          Reproducible diagnostic tools
experiments/                    Local checkpoints/images/runtime logs; ignored by Git

The original sibling Mask-DiFuser checkout is a reference repository. New work
belongs here. The submodule pins 9574766fe0ed96c2b8be3b1a63e6998c0a08b768;
its model, diffusion sampler and losses remain upstream code. Future algorithm
changes should be made in this project with explicit experiment variants.
Upstream rights and attribution remain with the original authors; the submodule
does not imply a new license for third-party source or model weights.

Setup
-----
git clone --recurse-submodules https://github.com/ZYC-HUST/MFIF-Diffusion.git
cd MFIF-Diffusion
python -m venv .venv
.venv/bin/pip install -r requirements.txt

On the current server, the tested environment is ../.envs/gmff/bin/python.
requirements.txt records direct runtime packages from that environment; it is
not a claim that installation was retested on a clean machine.

Data and paths
--------------
By default, datasets are read from ../dataset. On another machine set
MFIF_DATA_ROOT=/absolute/path/to/dataset.
Expected roots: train/MFIF, val/MFIF, test/COME-MFIF, test/Lytro, test/MFFW.
Synthetic roots contain manifest.jsonl plus A/, B/, GT/ with matching PNG names.
No dataset images or checkpoints are published in this repository.

Configuration paths support ${PROJECT_ROOT} and ${DATA_ROOT}. Numerical settings
in historical configs are retained. Existing runtime snapshots and checkpoint
metadata contain their ORIGINAL paths under Mask-DiFuser; these are provenance,
not current execution paths. See docs/MIGRATION_20261120.txt.

Training
--------
Copy configs/mfif_scratch_template.json to a new config and choose a new output
name before training. The template intentionally has no wall-clock deadline;
its default 87500-step schedule can take substantially longer than one night.

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCCL_P2P_DISABLE=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  train_mfif.py --config configs/YOUR_RUN.json

The fusion network starts randomly. LPIPS uses a frozen pretrained VGG only
for the original perceptual loss. Resume only our complete checkpoint with:
  --resume experiments/RUN/checkpoints/last.pt
Resume requires the same world size and numerical/data settings. If using a
deadline config for continuation, update the deadline; elapsed deadlines stop
training after one update. First-night configs retain their historical deadlines.

Evaluation
----------
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCCL_P2P_DISABLE=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  evaluate_mfif.py --checkpoint experiments/RUN/checkpoints/best.pt \
  --output experiments/RUN/evaluation_NEW --steps 5

Current evaluate_mfif.py implements synthetic GT metrics plus real-image native
inference previews. It DOES NOT yet compute the inherited real-fusion metric
suite automatically. docs/evaluation_protocol_inherited.json and the adapter
mfif_baseline/ivif_reference_metrics.py preserve the existing IVIF_ZOO convention;
IVIF_ZOO is an external sibling reference, or pass reference_dir explicitly.
PI remains a separately labeled pyiqa 0.1.13 exception; that adapter provides
only EN/AG/SF/SD/SCD. Proposed color metrics are not silently added to old runs.

Baseline v0 findings
-------------------
The first overnight run ended at step31247; best-PSNR was step9000. The pipeline
works, but official RGB outputs have significant color/brightness drift and
are not yet a strong final MFIF baseline. Full test PSNR=21.5886 dB versus
source-average=31.9754 dB. Later checkpoints improve detail while color drifts.
Read reports/baseline_v0/AUDIT_REPORT_zh.txt before interpreting results.

Version management
------------------
Git tracks our code, configs, protocols, small reports and the upstream commit.
experiments/, dataset/data, model binaries, caches and credentials are ignored.
Private SSH credentials live outside the project and are never committed.
The server uses a repository-specific deploy key; other clones configure their
own authentication. Historical reports are retained, not rewritten as new results.

Reference
---------
Tang, Li and Ma. Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised
Image Fusion. IEEE TPAMI. https://github.com/Linfeng-Tang/Mask-DiFuser
