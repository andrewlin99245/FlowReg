from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights


TARGET_CLASS_NAMES = [
    "goldfish",
    "stingray",
    "jellyfish",
    "starfish",
    "sea lion",
    "bald eagle",
    "great grey owl",
    "peacock",
    "flamingo",
    "toucan",
    "lion",
    "tiger",
    "zebra",
    "giant panda",
    "African elephant",
    "tabby cat",
    "Persian cat",
    "Siamese cat",
    "golden retriever",
    "horse",
    "box turtle",
    "American alligator",
    "tree frog",
    "axolotl",
    "Komodo dragon",
    "airliner",
    "school bus",
    "fire engine",
    "sports car",
    "bicycle-built-for-two",
    "acoustic guitar",
    "grand piano",
    "violin",
    "trumpet",
    "saxophone",
    "teapot",
    "lamp",
    "vacuum cleaner",
    "umbrella",
    "backpack",
    "pizza",
    "cheeseburger",
    "hotdog",
    "pretzel",
    "ice cream",
    "banana",
    "pineapple",
    "lemon",
    "pomegranate",
    "cauliflower",
]

CLASS_NAME_ALIASES = {
    "tabby cat": "tabby",
    "horse": "sorrel",
    "trumpet": "cornet",
    "saxophone": "sax",
    "lamp": "table lamp",
    "vacuum cleaner": "vacuum",
}

EXPECTED_IMAGE_SHAPE = (3, 64, 64)
METADATA_VERSION = 1


@dataclass(frozen=True)
class ClassRecord:
    local_index: int
    requested_name: str
    imagenet_name: str
    global_index: int


def get_imagenet_categories() -> list[str]:
    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def build_class_records() -> list[ClassRecord]:
    categories = get_imagenet_categories()
    records: list[ClassRecord] = []
    for local_index, requested_name in enumerate(TARGET_CLASS_NAMES):
        imagenet_name = CLASS_NAME_ALIASES.get(requested_name, requested_name)
        global_index = categories.index(imagenet_name)
        records.append(
            ClassRecord(
                local_index=local_index,
                requested_name=requested_name,
                imagenet_name=imagenet_name,
                global_index=global_index,
            )
        )
    return records


CLASS_RECORDS = build_class_records()
GLOBAL_TO_LOCAL = {record.global_index: record.local_index for record in CLASS_RECORDS}
LOCAL_TO_GLOBAL = {record.local_index: record.global_index for record in CLASS_RECORDS}
LOCAL_TO_NAME = {record.local_index: record.requested_name for record in CLASS_RECORDS}


def prepared_split_paths(root: str | Path, split: str) -> dict[str, Path]:
    base = Path(root)
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    return {
        "images": base / f"{split}_images.npy",
        "local_labels": base / f"{split}_local_labels.npy",
        "global_labels": base / f"{split}_global_labels.npy",
    }


def metadata_path(root: str | Path) -> Path:
    return Path(root) / "metadata.json"


def expected_metadata() -> dict[str, Any]:
    return {
        "metadata_version": METADATA_VERSION,
        "num_classes": len(CLASS_RECORDS),
        "image_shape": list(EXPECTED_IMAGE_SHAPE),
        "requested_class_names": TARGET_CLASS_NAMES,
        "alias_map": CLASS_NAME_ALIASES,
        "class_records": [asdict(record) for record in CLASS_RECORDS],
    }


def validate_prepared_metadata(metadata: Mapping[str, Any]) -> None:
    expected = expected_metadata()
    for key in ("metadata_version", "num_classes", "image_shape", "requested_class_names", "alias_map", "class_records"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Prepared dataset metadata mismatch for key '{key}'.")


def load_prepared_metadata(root: str | Path) -> dict[str, Any]:
    path = metadata_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_prepared_metadata(metadata)
    return metadata


def save_prepared_metadata(root: str | Path, metadata: Mapping[str, Any]) -> None:
    path = metadata_path(root)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def is_prepared_dataset(root: str | Path) -> bool:
    try:
        load_prepared_metadata(root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    for split in ("train", "val"):
        for path in prepared_split_paths(root, split).values():
            if not path.is_file():
                return False
    return True


class PreparedImageNet64Subset(Dataset[tuple[torch.Tensor, int, str, int]]):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self.metadata = load_prepared_metadata(self.root)
        split_paths = prepared_split_paths(self.root, split)
        self.images = np.load(split_paths["images"], mmap_mode="r")
        self.local_labels = np.load(split_paths["local_labels"], mmap_mode="r")
        self.global_labels = np.load(split_paths["global_labels"], mmap_mode="r")

        if self.images.ndim != 4:
            raise ValueError(f"Expected 4D images array, got shape {self.images.shape}")
        if tuple(self.images.shape[1:]) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(f"Expected image shape {EXPECTED_IMAGE_SHAPE}, got {self.images.shape[1:]}")
        if len(self.images) != len(self.local_labels) or len(self.images) != len(self.global_labels):
            raise ValueError("Prepared dataset arrays have inconsistent lengths.")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, int]:
        image_uint8 = np.array(self.images[index], copy=True)
        image = torch.tensor(image_uint8, dtype=torch.float32).div_(127.5).sub_(1.0)
        local_label = int(self.local_labels[index])
        global_label = int(self.global_labels[index])
        class_name = LOCAL_TO_NAME[local_label]
        return image, local_label, class_name, global_label
