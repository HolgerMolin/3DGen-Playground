# Gaussian Splat Diffusion baseline

`bash train.sh` trains the recipe in [`configs/train.yml`](../configs/train.yml).
The model predicts clean, Hilbert-ordered Gaussian splats conditioned on frozen
FLAN-T5-Large tokens. It uses a 128-channel tokenizer bottleneck, a 768-wide,
17-block multimodal transformer, full text refinement, and image-stream AdaLN.
EMA decay is **0.999 from the first optimizer step**.

## Installation

Install a CUDA-compatible PyTorch/torchvision pair, then:

```bash
pip install -r requirements-training.txt
```

The renderer requires a CUDA GPU and a compatible CUDA compiler for gsplat's
first-use kernel compilation. Keep the active environment's `bin` directory on
`PATH` so `ninja` is available. The default patch Sinkhorn loss uses PyTorch;
PyTorch3D and KeOps are optional for other matching modes.

## Configure your data paths

Download and unpack the GaussianVerse data and captions using the
[data preparation instructions](../data/README.md). The object list maps object
hashes to `directory/file.tar.gz`; each unpacked source directory
`GS_PATH/directory/file/` must contain `point_cloud.ply` and `gs2sphere.npy`.

Copy the local training example and edit its input and output paths:

```bash
cp configs/local.example.yml configs/local.yml
```

`configs/local.yml` is ignored by Git. Set `obj_list`, `gs_path`, and
`caption_path` to your downloaded data. Place `cache_dir` (~106 GiB for the
reference dataset), `t5_token_dir` (~47 GB), and `output_dir` (checkpoints)
on a disk with enough space. Set `num_workers` for the CPUs available **per GPU
process**. The example disables W&B; set `wandb: true` and `project` to enable it.

Data and output paths are relative to the repository root unless absolute
(reference assets use `asset_dir` as described below);
`~` and environment variables such as `$SCRATCH` expand. `_base` is relative
to the YAML file containing it. No paths depend on the original author's checkout.
Configuration precedence is built-in/environment defaults, then YAML, then CLI
flags. The optional `.env` is loaded by `train.sh`; direct Python commands read
only exported environment variables. Using the local YAML works for both.

## Supply the local reference assets

Set `asset_dir` in `configs/local.yml` to a local directory with this layout:

```text
baseline_assets/
├── ref_cameras.tar.gz
├── sphere2plane.npy
└── stats/
    ├── all_mean_postfix.pt
    ├── all_std_postfix.pt
    ├── clip_thresholds_opacity_scales.pt
    └── rank_quantiles_8ch_clipped.pt
```

These binary assets are **not stored in Git or downloaded automatically**.
For now, you must supply them from an existing compatible training setup.
The [reference asset manifest](../data/baseline_assets.json) records file sizes
and SHA-256 hashes for the exact files used by the baseline. The normalization,
clipping, and rank tables are specific to this preprocessing; the original
GaussianVerse mean/std files are not interchangeable with them.

Paths for these six files are relative to `asset_dir`, with absolute paths also
accepted. Other data/output paths remain relative to the repository root.
`DGEN_ASSET_ROOT` sets the default asset directory when no YAML override is used.
The text outlier list remains in the repository. The per-object framing table
is generated locally by the next command, because it depends on your objects.
A fresh checkout requires these local assets before framing or training can run.

## Prepare training inputs

One command encodes captions and builds the camera framing table with the same
paths used by training:

```bash
python prepare.py --config-file configs/local.yml
```

Text encoding uses CUDA when available. Use `--text-batch-size 32` to reduce its
memory use or `--device cpu` to run it on CPU. Run stages separately with
`--only text` or `--only framing`; framing runs on CPU. Text encoding resumes
compatible partial outputs; framing recomputes its table.

Text encoding options, including sharding, are available through
`python tools/encode_t5_tokens.py --help`.

For an offline compute node, download `google/flan-t5-large`,
`openai/clip-vit-large-patch14`, and the torchvision VGG16 weights used by LPIPS
on an internet-connected machine first. Preserve the Hugging Face and torch
cache directories when launching the job.

## Train and resume

```bash
bash train.sh --config-file configs/local.yml
```

`NPROC_PER_NODE` controls GPU count; otherwise all visible GPUs are used.
The global batch stays **512**. With the default microbatch of 32, accumulation
is 16 on one GPU, 8 on two GPUs, and 4 on four GPUs. To change memory usage
without changing the number of samples per optimizer step:

```bash
NPROC_PER_NODE=2 bash train.sh --config-file configs/local.yml --set micro_batch_size=64
```

Set both `batch_size` and the microbatch/accumulation fields explicitly to change
the global batch. Inconsistent explicit values fail before training.

The underlying entry point is `python train.py`. For your own launcher:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config-file configs/local.yml
```

Inspect defaults without loading data or creating outputs:

```bash
python train.py --print-config
```

Override any configuration field with `--set KEY=VALUE`, or use
`--config-file path/to/config.yml`. Unknown configuration keys are rejected.
YAML files can inherit another file through `_base`, resolved relative to the
file containing it.

The loss combines v-space MSE, hard patch Sinkhorn matching, and RGB/alpha/LPIPS
render losses with weights 8/4/8. Render losses start at step 5,000, retain
samples with `t >= 0.31`, and allocate half their pixel weight to foreground.
Muon updates matrix weights; AdamW updates the remaining parameters. Learning
rates warm up for 5,000 steps and stay constant. Gradient norm is capped at 50.

Atomic checkpoints contain raw and EMA weights, optimizer state, step,
configuration, and repository provenance. Training resumes the latest checkpoint
in its output directory automatically. Use a fresh `--output-dir` for a new
experiment, `--set auto_resume=false` to disable resume, or
`--set resume_from=/path/to/checkpoint.pt` to choose a checkpoint.
W&B can be disabled with `--no-wandb`.

## Sample and evaluate

```bash
python -m mini3d sample --config-file configs/local.yml --checkpoint /path/to/checkpoint.pt \
  --prompt "a blue phone booth" --out output/phone_booth.png

python -m mini3d eval --config-file configs/local.yml --checkpoint /path/to/checkpoint.pt
```

Both commands use EMA weights unless `--checkpoint-target model` is specified.
Sampling defaults to CFG **4.0**, 100 Euler steps, and exact final-step prediction.
Evaluation uses 2,000 held-out samples and 50 steps, matching the reference score.
The held-out split is the last 5,000 objects after sorting by caption stem;
objects without cached text tokens are dropped before splitting.

## Reference result

The reference checkpoint was evaluated at training step **134,000**.
With 2,000 samples, 50 Euler steps, CFG 4.0, and exact final-step prediction,
its CMMD was **0.1882318047**. The same checkpoint at CFG 2.5 scored
0.1922075725. These are checkpoint measurements, not a guarantee for a fresh run.
The historical run shortened EMA after 100k steps; this default uses the
short EMA from the start. The architecture, losses, optimizer, and
preprocessing follow the winning setup.

This recipe retains the reference lazy-cache target construction and clipping/rank assets; it does not apply
quality-score exclusions. The ordinary uncached loader applies an explicit
clipping step that the lazy cache path skips, so turning off lazy caching can
change targets. Keep the default caching path when comparing with this result.

## Verification

```bash
pip install -r requirements-test.txt
python -m pytest tests -q
# Create configs/local-smoke.yml with _base: smoke.yml and the same paths
# as configs/local.yml, then:
bash train.sh --config-file configs/local-smoke.yml --output-dir output/smoke --no-wandb
```

The four-step smoke uses a small transformer but exercises the same data,
matching, render losses, optimizer, checkpoints, and sampling paths. It is a
wiring check and does not measure generation quality.

The model and flow-matching implementation are adapted from
[MiniT2I](https://github.com/Hope7Happiness/minit2i-torch). The data and renderer
implementations come from this repository's experimental training path.
