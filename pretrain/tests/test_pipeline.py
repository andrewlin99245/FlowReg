from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import optim

from dataset import (
    CLASS_RECORDS,
    LOCAL_TO_NAME,
    PreparedImageNet64Subset,
    expected_metadata,
    save_prepared_metadata,
)
from model import ClassConditionedUNet
from sampling import AMPSettings, euler_sample
from train_cfm import TrainConfig, WarmupCosineScheduler, compute_cfm_loss, maybe_resume, save_checkpoint


class PipelineTests(unittest.TestCase):
    def create_prepared_dataset(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        metadata = expected_metadata()
        metadata["dataset_name"] = "synthetic"
        metadata["prepared_root"] = str(root)
        metadata["cache_dir"] = str(root / "cache")
        metadata["hf_splits"] = {"train": "train", "val": "validation"}
        metadata["split_sizes"] = {"train": 2, "val": 1}
        metadata["class_counts"] = {
            "train": {record.requested_name: 0 for record in CLASS_RECORDS},
            "val": {record.requested_name: 0 for record in CLASS_RECORDS},
        }
        metadata["class_counts"]["train"]["goldfish"] = 1
        metadata["class_counts"]["train"]["lion"] = 1
        metadata["class_counts"]["val"]["goldfish"] = 1
        save_prepared_metadata(root, metadata)

        train_images = np.random.randint(0, 256, size=(2, 3, 64, 64), dtype=np.uint8)
        train_local = np.asarray([0, 10], dtype=np.int64)
        train_global = np.asarray([CLASS_RECORDS[0].global_index, CLASS_RECORDS[10].global_index], dtype=np.int64)
        val_images = np.random.randint(0, 256, size=(1, 3, 64, 64), dtype=np.uint8)
        val_local = np.asarray([0], dtype=np.int64)
        val_global = np.asarray([CLASS_RECORDS[0].global_index], dtype=np.int64)

        np.save(root / "train_images.npy", train_images)
        np.save(root / "train_local_labels.npy", train_local)
        np.save(root / "train_global_labels.npy", train_global)
        np.save(root / "val_images.npy", val_images)
        np.save(root / "val_local_labels.npy", val_local)
        np.save(root / "val_global_labels.npy", val_global)

    def test_prepared_dataset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.create_prepared_dataset(root)
            dataset = PreparedImageNet64Subset(root, split="train")
            image, local_label, class_name, global_label = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertEqual(local_label, 0)
            self.assertEqual(class_name, "goldfish")
            self.assertEqual(global_label, CLASS_RECORDS[0].global_index)
            self.assertGreaterEqual(float(image.min()), -1.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_model_forward_and_class_conditioning(self) -> None:
        model = ClassConditionedUNet(base_channels=32, num_res_blocks=1)
        x = torch.randn(2, 3, 64, 64)
        t = torch.tensor([0.2, 0.7], dtype=torch.float32)
        labels_a = torch.tensor([0, 1], dtype=torch.long)
        labels_b = torch.tensor([2, 1], dtype=torch.long)

        out_a = model(x, t, labels_a)
        out_b = model(x, t, labels_b)

        self.assertEqual(tuple(out_a.shape), (2, 3, 64, 64))
        self.assertFalse(torch.allclose(out_a[0], out_b[0]))

    def test_training_step_sampling_and_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = ClassConditionedUNet(base_channels=32, num_res_blocks=1)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = WarmupCosineScheduler(optimizer, base_lr=1e-4, warmup_steps=2, total_steps=10)
            scaler = torch.amp.GradScaler("cpu", enabled=False)

            images = torch.randn(2, 3, 64, 64)
            labels = torch.tensor([0, 1], dtype=torch.long)
            loss = compute_cfm_loss(model, images, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            scheduler.step()

            samples = euler_sample(
                model=model,
                labels=labels,
                num_steps=2,
                device=torch.device("cpu"),
                amp_settings=AMPSettings(enabled=False, device_type="cpu", dtype=None),
            )
            self.assertEqual(tuple(samples.shape), (2, 3, 64, 64))
            self.assertTrue(torch.isfinite(samples).all())

            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            config = TrainConfig(max_steps=10, checkpoint_steps=(5, 10))
            metadata = {"requested_class_names": list(LOCAL_TO_NAME.values())}
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, 3, config, metadata)

            resumed_model = ClassConditionedUNet(base_channels=32, num_res_blocks=1)
            resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=1e-4)
            resumed_scheduler = WarmupCosineScheduler(
                resumed_optimizer, base_lr=1e-4, warmup_steps=2, total_steps=10
            )
            resumed_scaler = torch.amp.GradScaler("cpu", enabled=False)
            step = maybe_resume(
                checkpoint_path,
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_scaler,
            )
            self.assertEqual(step, 3)


if __name__ == "__main__":
    unittest.main()
