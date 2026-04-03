from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from models.checkpoint import ClassConditionedUNet, TARGET_CLASS_NAMES
from models.rollout import rollout_sde_with_logprobs, select_training_step_indices
from regularizers.batchot_reg import BatchOTRegularizer
from regularizers.rfr_reg import RFRRegularizer
from regularizers.w2_reg import W2Regularizer
from regularizers.base import RegularizerInputs
from rewards.base import RewardOutputs
from trainers.flowgrpo_trainer import FlowGRPOTrainer
from utils.config import load_config


class DummyReward:
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> RewardOutputs:
        del labels
        scores = images.mean(dim=(1, 2, 3))
        return RewardOutputs(total=scores, classifier=scores, musiq=None)


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.label_embed = nn.Embedding(len(TARGET_CLASS_NAMES), 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bias = self.label_embed(labels).view(-1, 3, 1, 1)
        return self.conv(x) + bias + t.view(-1, 1, 1, 1)


def make_fake_checkpoint(path: Path) -> None:
    model = ClassConditionedUNet()
    payload = {
        "model": model.state_dict(),
        "metadata": {
            "requested_class_names": TARGET_CLASS_NAMES,
        },
    }
    torch.save(payload, path)


class FineTunePipelineTest(unittest.TestCase):
    def test_select_training_step_indices(self) -> None:
        indices = select_training_step_indices(total_steps=10, train_steps=4)
        self.assertEqual(indices.tolist(), [0, 3, 6, 9])

    def test_rollout_shapes(self) -> None:
        model = ClassConditionedUNet()
        device = torch.device("cpu")
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        rollout = rollout_sde_with_logprobs(
            model=model,
            labels=labels,
            num_steps=4,
            noise_level=0.8,
            device=device,
            amp_settings=None,
        )
        self.assertEqual(tuple(rollout.states.shape), (4, 5, 3, 64, 64))
        self.assertEqual(tuple(rollout.log_probs.shape), (4, 4))
        self.assertTrue(torch.isfinite(rollout.log_probs).all().item())
        self.assertTrue(
            torch.allclose(
                rollout.transition_times.cpu(),
                torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float32),
            )
        )

    def test_regularizers(self) -> None:
        predicted_velocity = torch.randn(6, 3, 8, 8)
        x_t = torch.randn(6, 3, 8, 8)
        terminal = torch.randn(6, 3, 8, 8)
        times = torch.full((6,), 0.5)
        labels = torch.zeros(6, dtype=torch.long)
        inputs = RegularizerInputs(
            predicted_velocity=predicted_velocity,
            x_t=x_t,
            times=times,
            labels=labels,
            terminal_states=terminal,
        )
        self.assertGreater(float(W2Regularizer(weight=1.0).compute(inputs)), 0.0)
        self.assertGreaterEqual(float(RFRRegularizer(weight=1.0).compute(inputs)), 0.0)
        batchot = BatchOTRegularizer(weight=1.0, epsilon=0.05, num_iters=5)
        self.assertEqual(float(batchot.compute(inputs)), 0.0)
        batchot.update_reference_batch(terminal)
        batchot.prepare_rollout(terminal)
        prepared_anchors = batchot.select_anchor_states(
            trajectory_indices=torch.arange(terminal.shape[0]),
            device=terminal.device,
        )
        cached_inputs = RegularizerInputs(
            predicted_velocity=predicted_velocity,
            x_t=x_t,
            times=times,
            labels=labels,
            terminal_states=terminal,
            anchor_states=prepared_anchors,
        )
        with mock.patch.object(batchot, "_compute_anchors", wraps=batchot._compute_anchors) as patched:
            self.assertGreaterEqual(float(batchot.compute(cached_inputs)), 0.0)
            self.assertGreaterEqual(float(batchot.compute(cached_inputs)), 0.0)
        self.assertEqual(patched.call_count, 0)

    def test_trainer_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        base_config_path = repo_root / "configs" / "base.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            checkpoint_path = temp_root / "fake_pretrain.pt"
            make_fake_checkpoint(checkpoint_path)

            config = load_config(base_config_path)
            config["experiment_name"] = "smoke"
            config["pretrained"]["checkpoint_path"] = str(checkpoint_path)
            config["output"]["root"] = str(temp_root / "outputs")
            config["device"] = "cpu"
            config["train"]["mixed_precision"] = False
            config["train"]["total_outer_steps"] = 2
            config["train"]["log_every"] = 1
            config["train"]["checkpoint_every"] = 1
            config["train"]["eval_every"] = 1
            config["train"]["num_inner_epochs"] = 1
            config["train"]["minibatch_size"] = 2
            config["sample"]["rollout_steps"] = 4
            config["sample"]["train_steps"] = 2
            config["sample"]["num_groups_per_outer_step"] = 2
            config["sample"]["group_size"] = 2
            config["eval"]["class_ids"] = [0]
            config["eval"]["samples_per_class"] = 4
            config["eval"]["sample_steps"] = 2
            config["optim"]["warmup_steps"] = 0
            config["reward"]["setting"] = "classifier"
            config["regularizer"]["type"] = "batchot"
            config["regularizer"]["lambda_batchot"] = 1.0e-4
            config["regularizer"]["sinkhorn_iters"] = 4

            with mock.patch("trainers.flowgrpo_trainer.build_reward_function", return_value=DummyReward()):
                with mock.patch.object(FlowGRPOTrainer, "save_sample_grids", autospec=True, return_value=None):
                    with mock.patch(
                        "trainers.flowgrpo_trainer.load_flow_checkpoint",
                        side_effect=lambda checkpoint_path, device, freeze=False: (
                            TinyFlow().to(device),
                            {"requested_class_names": TARGET_CLASS_NAMES},
                        ),
                    ):
                        trainer = FlowGRPOTrainer(copy.deepcopy(config))
                        trainer.train()

            latest = temp_root / "outputs" / "smoke" / "checkpoints" / "latest.pt"
            self.assertTrue(latest.is_file())
            metrics = temp_root / "outputs" / "smoke" / "metrics.jsonl"
            self.assertTrue(metrics.is_file())


if __name__ == "__main__":
    unittest.main()
