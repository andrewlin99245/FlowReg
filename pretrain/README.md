# ImageNet64-50 CFM Pretraining

This directory contains a clean class-conditioned Conditional Flow Matching pretraining pipeline for a 50-class subset of downsampled ImageNet64.

For the full end-to-end workflow, including shared env setup, dataset download, reward-model bootstrap, and fine-tuning, use the shared project README at `../README.md` and the root `scripts/` wrappers.

## Environment

The checked-in `environment.yml` now matches the shared working env versions used for development on this repo. The codepath is CUDA-compatible; on the actual training machine you should install the matching CUDA-enabled PyTorch build there.

```bash
mamba env create -p .conda-env -f environment.yml
```

If you want the exact tested dev snapshot from this machine, use the exported lock file instead:

```bash
mamba env create -p .conda-env -f environment.verified.lock.yml
```

That lock file is mainly for matching the verified development env on a similar platform. For general CUDA deployment, prefer `environment.yml`.

## Prepare the dataset

```bash
mamba run -p .conda-env \
  python prepare_imagenet64_subset.py \
  --cache-dir data/hf_cache \
  --prepared-root data/imagenet64_subset50
```

This downloads `ChocolateDave/imagenet-64` from Hugging Face, filters the requested 50 ImageNet classes, remaps them to local labels `0..49`, and writes a memory-mapped local copy under `data/imagenet64_subset50/`.

## Train

```bash
mamba run -p .conda-env \
  python train_cfm.py \
  --data-root data/imagenet64_subset50 \
  --output-dir outputs/imagenet64_subset50_cfm
```

If `data/imagenet64_subset50/` is missing, the training script can prepare it automatically when `--auto-prepare true` is left enabled.

## Tests

```bash
mamba run -p .conda-env \
  python -m unittest discover -s tests -v
```
