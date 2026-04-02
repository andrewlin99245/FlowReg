from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from dataset import (
    CLASS_RECORDS,
    GLOBAL_TO_LOCAL,
    LOCAL_TO_NAME,
    expected_metadata,
    is_prepared_dataset,
    load_prepared_metadata,
    prepared_split_paths,
    save_prepared_metadata,
)


DEFAULT_DATASET_NAME = "ChocolateDave/imagenet-64"
DEFAULT_HF_SPLITS = {"train": "train", "val": "val"}


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value}")


def import_datasets_module():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for dataset download and preparation. "
            "Install the project environment from environment.yml first."
        ) from exc
    return load_dataset


def extract_image_array(sample: Mapping[str, Any]) -> np.ndarray:
    image = sample["image"]
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported image payload type: {type(image)!r}")

    if array.shape[:2] != (64, 64):
        raise ValueError(f"Expected 64x64 image, got shape {array.shape}")
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def prepare_split(raw_split, split: str, prepared_root: str | Path) -> dict[str, Any]:
    selected_images: list[np.ndarray] = []
    selected_local_labels: list[int] = []
    selected_global_labels: list[int] = []
    class_counts = {LOCAL_TO_NAME[record.local_index]: 0 for record in CLASS_RECORDS}

    iterator = tqdm(raw_split, total=len(raw_split), desc=f"Preparing {split}", dynamic_ncols=True)
    for sample in iterator:
        global_label = int(sample["label"])
        local_label = GLOBAL_TO_LOCAL.get(global_label)
        if local_label is None:
            continue

        image = extract_image_array(sample)
        selected_images.append(np.transpose(image, (2, 0, 1)))
        selected_local_labels.append(local_label)
        selected_global_labels.append(global_label)
        class_counts[LOCAL_TO_NAME[local_label]] += 1

    if not selected_images:
        raise RuntimeError(f"No matching samples found while preparing split '{split}'.")

    images_array = np.stack(selected_images, axis=0).astype(np.uint8, copy=False)
    local_labels_array = np.asarray(selected_local_labels, dtype=np.int64)
    global_labels_array = np.asarray(selected_global_labels, dtype=np.int64)

    split_paths = prepared_split_paths(prepared_root, split)
    np.save(split_paths["images"], images_array)
    np.save(split_paths["local_labels"], local_labels_array)
    np.save(split_paths["global_labels"], global_labels_array)

    return {
        "num_examples": int(images_array.shape[0]),
        "class_counts": class_counts,
    }


def prepare_split_with_limits(
    raw_split,
    split: str,
    prepared_root: str | Path,
    max_raw_examples: int | None = None,
) -> dict[str, Any]:
    selected_images: list[np.ndarray] = []
    selected_local_labels: list[int] = []
    selected_global_labels: list[int] = []
    class_counts = {LOCAL_TO_NAME[record.local_index]: 0 for record in CLASS_RECORDS}

    if max_raw_examples is not None:
        iterator_source = itertools.islice(raw_split, max_raw_examples)
        total = max_raw_examples
    else:
        iterator_source = raw_split
        total = len(raw_split)

    iterator = tqdm(iterator_source, total=total, desc=f"Preparing {split}", dynamic_ncols=True)
    for sample in iterator:
        global_label = int(sample["label"])
        local_label = GLOBAL_TO_LOCAL.get(global_label)
        if local_label is None:
            continue

        image = extract_image_array(sample)
        selected_images.append(np.transpose(image, (2, 0, 1)))
        selected_local_labels.append(local_label)
        selected_global_labels.append(global_label)
        class_counts[LOCAL_TO_NAME[local_label]] += 1

    if not selected_images:
        raise RuntimeError(f"No matching samples found while preparing split '{split}'.")

    images_array = np.stack(selected_images, axis=0).astype(np.uint8, copy=False)
    local_labels_array = np.asarray(selected_local_labels, dtype=np.int64)
    global_labels_array = np.asarray(selected_global_labels, dtype=np.int64)

    split_paths = prepared_split_paths(prepared_root, split)
    np.save(split_paths["images"], images_array)
    np.save(split_paths["local_labels"], local_labels_array)
    np.save(split_paths["global_labels"], global_labels_array)

    return {
        "num_examples": int(images_array.shape[0]),
        "class_counts": class_counts,
        "raw_examples_scanned": total,
    }


def prepare_dataset(
    cache_dir: str | Path,
    prepared_root: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    hf_splits: Mapping[str, str] | None = None,
    streaming: bool = False,
    max_raw_examples: Mapping[str, int | None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    prepared_root = Path(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_splits = dict(hf_splits or DEFAULT_HF_SPLITS)
    max_raw_examples = dict(max_raw_examples or {})

    if is_prepared_dataset(prepared_root) and not force:
        metadata = load_prepared_metadata(prepared_root)
        metadata["dataset_name"] = dataset_name
        metadata["prepared_root"] = str(prepared_root)
        metadata["cache_dir"] = str(cache_dir)
        metadata["hf_splits"] = hf_splits
        return metadata

    load_dataset = import_datasets_module()

    split_stats: dict[str, Any] = {}
    for split, hf_split in hf_splits.items():
        raw_split = load_dataset(
            dataset_name,
            split=hf_split,
            cache_dir=str(cache_dir),
            streaming=streaming,
        )
        split_stats[split] = prepare_split_with_limits(
            raw_split,
            split,
            prepared_root,
            max_raw_examples=max_raw_examples.get(split),
        )

    metadata = expected_metadata()
    metadata["dataset_name"] = dataset_name
    metadata["prepared_root"] = str(prepared_root)
    metadata["cache_dir"] = str(cache_dir)
    metadata["hf_splits"] = hf_splits
    metadata["streaming"] = streaming
    metadata["max_raw_examples"] = max_raw_examples
    metadata["split_sizes"] = {split: split_stats[split]["num_examples"] for split in hf_splits}
    metadata["class_counts"] = {split: split_stats[split]["class_counts"] for split in hf_splits}
    metadata["raw_examples_scanned"] = {
        split: split_stats[split]["raw_examples_scanned"] for split in hf_splits
    }
    save_prepared_metadata(prepared_root, metadata)
    return metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare the 50-class ImageNet64 subset.")
    parser.add_argument("--cache-dir", type=str, default="data/hf_cache")
    parser.add_argument("--prepared-root", type=str, default="data/imagenet64_subset50")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--train-split", type=str, default=DEFAULT_HF_SPLITS["train"])
    parser.add_argument("--val-split", type=str, default=DEFAULT_HF_SPLITS["val"])
    parser.add_argument("--streaming", type=parse_bool, default=False)
    parser.add_argument("--max-raw-train-examples", type=int, default=None)
    parser.add_argument("--max-raw-val-examples", type=int, default=None)
    parser.add_argument("--force", type=parse_bool, default=False)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    metadata = prepare_dataset(
        cache_dir=args.cache_dir,
        prepared_root=args.prepared_root,
        dataset_name=args.dataset_name,
        hf_splits={"train": args.train_split, "val": args.val_split},
        streaming=args.streaming,
        max_raw_examples={
            "train": args.max_raw_train_examples,
            "val": args.max_raw_val_examples,
        },
        force=args.force,
    )
    print(f"Prepared dataset written to {metadata['prepared_root']}")
    for split, count in metadata["split_sizes"].items():
        print(f"{split}: {count} examples")


if __name__ == "__main__":
    main()
