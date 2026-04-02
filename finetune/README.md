# FlowReg Fine-Tuning

This directory implements online RL fine-tuning for the pretrained ImageNet64-50 class-conditioned flow model from `../pretrain`, following the Flow-GRPO update structure and adding the FlowReg regularizers:

For the full end-to-end workflow, including shared env setup, dataset download, reward-model bootstrap, and pretraining, use the shared project README at `../README.md` and the root `scripts/` wrappers.

- `no_reg`
- `w2`
- `rfr`
- `batchot`

Supported reward settings:

- `classifier`
- `classifier_plus_musiq`

## What the code does

- Loads the same pretrained checkpoint for every experiment.
- Collects on-policy stochastic rollouts with the Flow-GRPO ODE-to-SDE conversion.
- Computes group-relative advantages within each class group.
- Optimizes a clipped GRPO objective with optional reference-KL weight `rl.beta_kl`.
- Adds one of the four regularizer modes through a unified interface.
- Saves JSONL logs, checkpoints, and per-class sample grids.
- Provides standalone sample-generation and reward-evaluation scripts.

## Important implementation note

Flow-GRPO is written for reverse-time denoising trajectories. This code adapts the same objective to the forward-time rectified-flow convention already used in `../pretrain`, where generation integrates from Gaussian noise at `t = 0` to images at `t = 1`. The clipped-ratio RL objective is unchanged; only the transition direction and the conditional Gaussian used for log-probabilities are mapped to the local forward-time rectified-flow dynamics.

## Environment

Use the same env as pretraining. The default shared setup path now pins the fine-tune overlay to the exact versions that were loaded successfully in the shared dev env:

```bash
cd ../pretrain
mamba env create -p .conda-env -f environment.yml
```

If you want the `classifier_plus_musiq` reward setting, install the extra fine-tune dependency into that same env:

```bash
cd ../finetune
mamba env update -p ../pretrain/.conda-env -f environment.yml
```

If you want the exact exported shared dev snapshot instead, run `../scripts/setup_env.sh --verified-lock` from the repo root. That snapshot is closer to the working local env, but the default pinned env files remain the safer path for a different CUDA machine.

## Run fine-tuning

From `FlowReg/finetune`:

```bash
mamba run -p ../pretrain/.conda-env \
  python train_finetune.py --config configs/experiments/no_reg_classifier.yaml
```

Example variants:

```bash
mamba run -p ../pretrain/.conda-env \
  python train_finetune.py --config configs/experiments/w2_classifier.yaml

mamba run -p ../pretrain/.conda-env \
  python train_finetune.py --config configs/experiments/rfr_classifier_plus_musiq.yaml

mamba run -p ../pretrain/.conda-env \
  python train_finetune.py --config configs/experiments/batchot_classifier_plus_musiq.yaml
```

## Generate samples from a checkpoint

```bash
mamba run -p ../pretrain/.conda-env \
  python -m eval.sample \
  --config configs/experiments/rfr_classifier.yaml \
  --checkpoint outputs/rfr_classifier/checkpoints/latest.pt
```

## Score a checkpoint

```bash
mamba run -p ../pretrain/.conda-env \
  python -m eval.score \
  --config configs/experiments/batchot_classifier_plus_musiq.yaml \
  --checkpoint outputs/batchot_classifier_plus_musiq/checkpoints/latest.pt
```

## Most important config knobs

- `pretrained.checkpoint_path`: pretrained flow checkpoint shared across all runs.
- `sample.rollout_steps`: number of SDE rollout steps for online RL data collection.
- `sample.train_steps`: number of rollout steps kept for training updates.
- `sample.num_groups_per_outer_step` and `sample.group_size`: class-group layout for GRPO.
- `sample.noise_level`: Flow-GRPO stochasticity parameter.
- `rl.clip_range`: ratio clipping width.
- `rl.beta_kl`: optional reference-KL weight inside the RL objective.
- `reward.setting`: `classifier` or `classifier_plus_musiq`.
- `reward.alpha` and `reward.beta`: reward mixing weights for classifier + MUSIQ.
- `regularizer.type`: `no_reg`, `w2`, `rfr`, or `batchot`.
- `regularizer.lambda_w2`, `regularizer.lambda_rfr`, `regularizer.lambda_batchot`: regularizer weights.
- `regularizer.sinkhorn_epsilon` and `regularizer.sinkhorn_iters`: BatchOT solver parameters.
- `train.total_outer_steps`, `train.num_inner_epochs`, `train.minibatch_size`, `optim.lr`.
