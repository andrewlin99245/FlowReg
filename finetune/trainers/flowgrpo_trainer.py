from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid, save_image

from models import (
    compute_transition_logprob,
    deterministic_euler_sample,
    load_flow_checkpoint,
    rollout_sde_with_logprobs,
    select_training_step_indices,
)
from regularizers import build_regularizer
from regularizers.base import RegularizerInputs
from rewards import build_reward_function
from utils import (
    WarmupCosineScheduler,
    append_jsonl,
    configure_cuda_runtime,
    save_json,
    select_device,
    set_seed,
    slugify,
)
from utils.misc import collect_rng_state, restore_rng_state


def _save_image_grid(images: torch.Tensor, path: Path, nrow: int) -> None:
    normalized = images.clamp(-1.0, 1.0).add(1.0).div(2.0)
    grid = make_grid(normalized, nrow=nrow)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)


class FlowGRPOTrainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = select_device(config.get("device", "auto"))
        configure_cuda_runtime(self.device)
        set_seed(int(config["train"]["seed"]))

        self.output_dir = Path(config["output"]["root"]).resolve() / config["experiment_name"]
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.eval_dir = self.output_dir / "eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.model, self.metadata = load_flow_checkpoint(
            checkpoint_path=config["pretrained"]["checkpoint_path"],
            device=self.device,
            freeze=False,
        )
        self.reference_model, _ = load_flow_checkpoint(
            checkpoint_path=config["pretrained"]["checkpoint_path"],
            device=self.device,
            freeze=True,
        )
        self.amp_enabled = bool(config["train"].get("mixed_precision", True)) and self.device.type == "cuda"
        from models.checkpoint import resolve_amp_settings  # local import keeps package init lean

        self.amp_settings = resolve_amp_settings(self.device, enabled=self.amp_enabled)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        optim_cfg = config["optim"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(optim_cfg["lr"]),
            betas=(float(optim_cfg.get("beta1", 0.9)), float(optim_cfg.get("beta2", 0.999))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            base_lr=float(optim_cfg["lr"]),
            warmup_steps=int(optim_cfg.get("warmup_steps", 0)),
            total_steps=max(1, int(config["train"]["total_outer_steps"])),
        )
        self.reward_fn = build_reward_function(config["reward"], device=self.device)
        self.regularizer = build_regularizer(config["regularizer"])
        self.outer_step = 0
        self.optimizer_step = 0
        self.metrics_path = self.output_dir / "metrics.jsonl"

        save_json(self.output_dir / "config_snapshot.json", config)
        save_json(self.output_dir / "metadata_snapshot.json", self.metadata)

        resume_path = str(config["train"].get("resume", "")).strip()
        if resume_path:
            self.load_checkpoint(resume_path)
        else:
            latest = self.checkpoint_dir / "latest.pt"
            if latest.is_file():
                self.load_checkpoint(latest)

    def sample_conditioning_labels(self) -> torch.Tensor:
        sample_cfg = self.config["sample"]
        num_groups = int(sample_cfg["num_groups_per_outer_step"])
        group_size = int(sample_cfg["group_size"])
        group_labels = torch.randint(0, len(self.metadata["requested_class_names"]), (num_groups,), device=self.device)
        return group_labels.repeat_interleave(group_size)

    def compute_group_advantages(self, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample_cfg = self.config["sample"]
        group_size = int(sample_cfg["group_size"])
        if rewards.numel() % group_size != 0:
            raise ValueError("Rewards batch size must be divisible by group_size.")
        reward_groups = rewards.view(-1, group_size)
        means = reward_groups.mean(dim=1, keepdim=True)
        stds = reward_groups.std(dim=1, unbiased=False, keepdim=True)
        advantages = (reward_groups - means) / (stds + float(self.config["rl"].get("advantage_eps", 1e-4)))
        clip_value = float(self.config["rl"].get("advantage_clip", 5.0))
        advantages = advantages.clamp(-clip_value, clip_value)
        valid_groups = stds.squeeze(1) > 0.0
        advantages = advantages * valid_groups.view(-1, 1)
        return advantages.reshape(-1), valid_groups.repeat_interleave(group_size)

    @torch.no_grad()
    def collect_rollout(self):
        self.model.eval()
        labels = self.sample_conditioning_labels()
        rollout = rollout_sde_with_logprobs(
            model=self.model,
            labels=labels,
            num_steps=int(self.config["sample"]["rollout_steps"]),
            noise_level=float(self.config["sample"]["noise_level"]),
            device=self.device,
            image_shape=tuple(self.config["model"]["image_shape"]),
            t_eps=float(self.config["rl"].get("t_eps", 1e-3)),
            amp_settings=self.amp_settings,
        )
        reward_outputs = self.reward_fn(rollout.terminal_states, labels)
        advantages, valid_mask = self.compute_group_advantages(reward_outputs.total)
        rollout.rewards = reward_outputs.total.detach()
        rollout.reward_classifier = reward_outputs.classifier.detach()
        rollout.reward_musiq = None if reward_outputs.musiq is None else reward_outputs.musiq.detach()
        rollout.advantages = advantages.detach()
        return rollout, valid_mask

    def _flatten_rollout_batch(self, rollout, trajectory_indices: torch.Tensor, step_indices: torch.Tensor):
        states = rollout.states[trajectory_indices]
        x_t = states[:, step_indices]
        x_next = states[:, step_indices + 1]
        batch_size, train_steps = x_t.shape[:2]
        labels = rollout.labels[trajectory_indices].view(-1, 1).expand(batch_size, train_steps)
        times = rollout.transition_times[step_indices].view(1, -1).expand(batch_size, train_steps)
        old_log_probs = rollout.log_probs[trajectory_indices][:, step_indices]
        advantages = rollout.advantages[trajectory_indices].view(-1, 1).expand(batch_size, train_steps)
        terminal_states = rollout.terminal_states[trajectory_indices].unsqueeze(1).expand_as(x_t)
        return {
            "x_t": x_t.reshape(-1, *x_t.shape[2:]),
            "x_next": x_next.reshape(-1, *x_next.shape[2:]),
            "labels": labels.reshape(-1),
            "times": times.reshape(-1),
            "old_log_probs": old_log_probs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "terminal_states": terminal_states.reshape(-1, *terminal_states.shape[2:]),
        }

    def _compute_minibatch_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rl_cfg = self.config["rl"]
        clip_range = float(rl_cfg["clip_range"])
        noise_level = float(self.config["sample"]["noise_level"])
        dt = 1.0 / float(self.config["sample"]["rollout_steps"])

        new_log_probs, new_mean, step_std, predicted_velocity, _ = compute_transition_logprob(
            model=self.model,
            x_t=batch["x_t"],
            x_next=batch["x_next"],
            t=batch["times"],
            labels=batch["labels"],
            dt=dt,
            noise_level=noise_level,
            t_eps=float(rl_cfg.get("t_eps", 1e-3)),
            amp_settings=self.amp_settings,
        )
        log_ratio = new_log_probs - batch["old_log_probs"]
        ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
        unclipped = ratio * batch["advantages"]
        clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * batch["advantages"]
        loss_pg = -torch.minimum(unclipped, clipped).mean()

        with torch.no_grad():
            ref_mean, _, _, _ = compute_transition_logprob(
                model=self.reference_model,
                x_t=batch["x_t"],
                x_next=batch["x_next"],
                t=batch["times"],
                labels=batch["labels"],
                dt=dt,
                noise_level=noise_level,
                t_eps=float(rl_cfg.get("t_eps", 1e-3)),
                amp_settings=self.amp_settings,
            )[1:]
        variance = step_std.square()
        kl = 0.5 * (new_mean - ref_mean).square().flatten(1).sum(dim=1) / variance
        kl_loss = kl.mean()
        loss_rl = loss_pg + float(rl_cfg.get("beta_kl", 0.0)) * kl_loss

        reg_inputs = RegularizerInputs(
            predicted_velocity=predicted_velocity,
            x_t=batch["x_t"],
            times=batch["times"],
            labels=batch["labels"],
            terminal_states=batch["terminal_states"],
        )
        reg_value = self.regularizer.compute(reg_inputs)
        total_loss = loss_rl + float(getattr(self.regularizer, "weight", 0.0)) * reg_value
        clipfrac = ((ratio > 1.0 + clip_range) | (ratio < 1.0 - clip_range)).float().mean()
        approx_kl = 0.5 * log_ratio.square().mean()
        return {
            "loss_total": total_loss,
            "loss_rl": loss_rl.detach(),
            "loss_pg": loss_pg.detach(),
            "loss_reg": (float(getattr(self.regularizer, "weight", 0.0)) * reg_value).detach(),
            "reg_value": reg_value.detach(),
            "kl_loss": kl_loss.detach(),
            "clipfrac": clipfrac.detach(),
            "approx_kl_proxy": approx_kl.detach(),
            "adv_mean": batch["advantages"].mean().detach(),
        }

    def save_checkpoint(self, tag: str) -> Path:
        path = self.checkpoint_dir / f"{tag}.pt"
        payload = {
            "config": self.config,
            "metadata": self.metadata,
            "model": self.model.state_dict(),
            "reference_model": self.reference_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "outer_step": self.outer_step,
            "optimizer_step": self.optimizer_step,
            "rng_state": collect_rng_state(),
            "regularizer": self.regularizer.state_dict(),
        }
        torch.save(payload, path)
        torch.save(payload, self.checkpoint_dir / "latest.pt")
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if "reference_model" in checkpoint:
            self.reference_model.load_state_dict(checkpoint["reference_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint["rng_state"])
        self.outer_step = int(checkpoint.get("outer_step", 0))
        self.optimizer_step = int(checkpoint.get("optimizer_step", 0))
        if "regularizer" in checkpoint:
            self.regularizer.load_state_dict(checkpoint["regularizer"])

    @torch.no_grad()
    def save_sample_grids(self, step_tag: str, num_steps: int | None = None) -> None:
        self.model.eval()
        eval_cfg = self.config["eval"]
        class_ids = eval_cfg.get("class_ids", [])
        if not class_ids:
            class_ids = list(range(len(self.metadata["requested_class_names"])))
        samples_per_class = int(eval_cfg["samples_per_class"])
        num_steps = int(num_steps or eval_cfg["sample_steps"])
        nrow = int(math.sqrt(samples_per_class))
        if nrow * nrow != samples_per_class:
            nrow = min(4, samples_per_class)
        output_dir = self.sample_dir / step_tag
        for class_id in class_ids:
            labels = torch.full((samples_per_class,), int(class_id), device=self.device, dtype=torch.long)
            samples = deterministic_euler_sample(
                model=self.model,
                labels=labels,
                num_steps=num_steps,
                device=self.device,
                image_shape=tuple(self.config["model"]["image_shape"]),
                amp_settings=self.amp_settings,
            )
            class_name = self.metadata["requested_class_names"][int(class_id)]
            _save_image_grid(
                samples,
                output_dir / f"{int(class_id):02d}_{slugify(class_name)}.png",
                nrow=nrow,
            )

    def train(self) -> None:
        train_cfg = self.config["train"]
        total_outer_steps = int(train_cfg["total_outer_steps"])
        log_every = int(train_cfg["log_every"])
        checkpoint_every = int(train_cfg["checkpoint_every"])
        eval_every = int(train_cfg["eval_every"])
        train_steps = select_training_step_indices(
            total_steps=int(self.config["sample"]["rollout_steps"]),
            train_steps=int(self.config["sample"]["train_steps"]),
        ).to(self.device)

        while self.outer_step < total_outer_steps:
            outer_start = time.time()
            rollout, valid_mask = self.collect_rollout()
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            self.model.train()
            metrics_sum = {
                "loss_rl": 0.0,
                "loss_pg": 0.0,
                "loss_reg": 0.0,
                "reg_value": 0.0,
                "kl_loss": 0.0,
                "clipfrac": 0.0,
                "approx_kl_proxy": 0.0,
                "adv_mean": 0.0,
            }
            update_count = 0

            if valid_indices.numel() > 0:
                minibatch_size = int(train_cfg["minibatch_size"])
                for _ in range(int(train_cfg["num_inner_epochs"])):
                    permutation = valid_indices[torch.randperm(valid_indices.numel(), device=valid_indices.device)]
                    for start in range(0, permutation.numel(), minibatch_size):
                        trajectory_indices = permutation[start : start + minibatch_size]
                        batch = self._flatten_rollout_batch(rollout, trajectory_indices=trajectory_indices, step_indices=train_steps)
                        self.optimizer.zero_grad(set_to_none=True)
                        losses = self._compute_minibatch_losses(batch)
                        total_loss = losses["loss_total"]
                        if self.amp_enabled:
                            self.scaler.scale(total_loss).backward()
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(train_cfg["grad_clip"]))
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(train_cfg["grad_clip"]))
                            self.optimizer.step()
                        self.optimizer_step += 1
                        for key in metrics_sum:
                            metrics_sum[key] += float(losses[key].detach().cpu())
                        update_count += 1

            self.scheduler.step()
            self.outer_step += 1
            self.regularizer.update_reference_batch(rollout.terminal_states)

            average_metrics = {key: (value / max(update_count, 1)) for key, value in metrics_sum.items()}
            reward_total = float(rollout.rewards.mean().cpu())
            reward_classifier = float(rollout.reward_classifier.mean().cpu())
            reward_musiq = None if rollout.reward_musiq is None else float(rollout.reward_musiq.mean().cpu())
            metric_payload = {
                "outer_step": self.outer_step,
                "optimizer_step": self.optimizer_step,
                "reward_total": reward_total,
                "reward_classifier": reward_classifier,
                "reward_musiq": reward_musiq,
                "loss_rl": average_metrics["loss_rl"],
                "loss_pg": average_metrics["loss_pg"],
                "loss_reg": average_metrics["loss_reg"],
                "reg_value": average_metrics["reg_value"],
                "kl_loss": average_metrics["kl_loss"],
                "clipfrac": average_metrics["clipfrac"],
                "approx_kl_proxy": average_metrics["approx_kl_proxy"],
                "adv_mean": average_metrics["adv_mean"],
                "lr": self.scheduler.lr,
                "steps_per_second": 1.0 / max(time.time() - outer_start, 1e-6),
                "valid_fraction": float(valid_mask.float().mean().cpu()),
            }
            append_jsonl(self.metrics_path, metric_payload)
            if self.outer_step == 1 or self.outer_step % log_every == 0:
                print(
                    f"outer_step={self.outer_step} reward={reward_total:.6f} "
                    f"loss_rl={average_metrics['loss_rl']:.6f} reg={average_metrics['reg_value']:.6f} "
                    f"lr={self.scheduler.lr:.6e}"
                )
            if self.outer_step % eval_every == 0 or self.outer_step == 1:
                self.save_sample_grids(step_tag=f"step_{self.outer_step:06d}")
            if self.outer_step % checkpoint_every == 0 or self.outer_step == total_outer_steps:
                checkpoint_path = self.save_checkpoint(tag=f"step_{self.outer_step:06d}")
                print(f"saved checkpoint: {checkpoint_path}")
