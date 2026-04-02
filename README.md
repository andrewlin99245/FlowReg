# FlowReg

This repo has a shared end-to-end workflow for:

1. setting up one shared env for pretraining and fine-tuning
2. downloading and preparing the ImageNet64 subset
3. bootstrapping the pretrained reward models and validating class mapping
4. pretraining the class-conditioned flow model
5. fine-tuning it with Flow-GRPO and FlowReg regularizers

The shared env lives at:

`pretrain/.conda-env`

Project-local caches also live under `pretrain/`:

- `pretrain/.torch-cache`
- `pretrain/.hf-cache`
- `pretrain/.pip-cache`
- `pretrain/.xdg-cache`
- `pretrain/.mpl-cache`

## Quick Start

From the repo root:

```bash
./scripts/setup_env.sh
./scripts/download_dataset.sh
./scripts/load_reward_models.sh
./scripts/run_pretrain.sh --data-root data/imagenet64_subset50 --output-dir outputs/imagenet64_subset50_cfm
./scripts/run_finetune.sh configs/experiments/no_reg_classifier.yaml
```

## Shared Scripts

### 1. Setup the shared env

```bash
./scripts/setup_env.sh
```

What it does:

- creates `pretrain/.conda-env` from [`pretrain/environment.yml`](pretrain/environment.yml)
- updates that same env with [`finetune/environment.yml`](finetune/environment.yml) so MUSIQ support is available
- both env files are now pinned to the versions used in the verified shared dev env

If you want the exact exported dev snapshot from the working machine instead of the cross-platform pinned env files:

```bash
./scripts/setup_env.sh --verified-lock
```

That uses [`pretrain/environment.verified.lock.yml`](pretrain/environment.verified.lock.yml). It is the closest match to the verified dev env, but it is less portable across OS/arch combinations than the default setup path.

### 2. Download and prepare the dataset

```bash
./scripts/download_dataset.sh
```

Default output:

- `pretrain/data/imagenet64_subset50`

You can forward any extra args directly to [`prepare_imagenet64_subset.py`](pretrain/prepare_imagenet64_subset.py), for example:

```bash
./scripts/download_dataset.sh --prepared-root data/imagenet64_subset50 --cache-dir data/hf_cache
```

### 3. Load pretrained reward models and verify mapping

```bash
./scripts/load_reward_models.sh
```

What it verifies:

- the actual pretrained classifier reward model loads
- the actual pretrained MUSIQ reward model loads
- the `classifier_plus_musiq` reward path runs a forward pass
- all 50 selected classes map correctly to ImageNet-1k labels

It writes a summary to:

- `pretrain/outputs/reward_bootstrap/reward_bootstrap_summary.json`

## Pretraining

Run pretraining through the shared wrapper:

```bash
./scripts/run_pretrain.sh --data-root data/imagenet64_subset50 --output-dir outputs/imagenet64_subset50_cfm
```

This calls [`train_cfm.py`](pretrain/train_cfm.py) inside `pretrain/`.

## Fine-Tuning

Run fine-tuning through the shared wrapper:

```bash
./scripts/run_finetune.sh configs/experiments/rfr_classifier_plus_musiq.yaml
```

This calls [`train_finetune.py`](finetune/train_finetune.py) inside `finetune/`.

## Class Mapping

The default classifier reward uses the ImageNet-1k classifier probability for the target class, not a separate 50-way classifier head. The mapping is defined once in [`dataset.py`](pretrain/dataset.py) and reused by fine-tuning.

Important alias-resolved classes:

- `tabby cat -> tabby`
- `horse -> sorrel`
- `trumpet -> cornet`
- `saxophone -> sax`
- `lamp -> table lamp`
- `vacuum cleaner -> vacuum`

## Common Commands

Evaluate samples from a fine-tuned checkpoint:

```bash
cd finetune
../pretrain/.conda-env/bin/python -m eval.sample \
  --config configs/experiments/rfr_classifier.yaml \
  --checkpoint outputs/rfr_classifier/checkpoints/latest.pt
```

Score a fine-tuned checkpoint:

```bash
cd finetune
../pretrain/.conda-env/bin/python -m eval.score \
  --config configs/experiments/batchot_classifier_plus_musiq.yaml \
  --checkpoint outputs/batchot_classifier_plus_musiq/checkpoints/latest.pt
```
