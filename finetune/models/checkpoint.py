from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch


THIS_FILE = Path(__file__).resolve()
if THIS_FILE.parents[2].name == "pretrain":
    PRETRAIN_ROOT = THIS_FILE.parents[2]
    FLOWREG_ROOT = PRETRAIN_ROOT.parent
else:
    FLOWREG_ROOT = THIS_FILE.parents[2]
    PRETRAIN_ROOT = FLOWREG_ROOT / "pretrain"
if str(PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ROOT))

from dataset import LOCAL_TO_GLOBAL, LOCAL_TO_NAME, TARGET_CLASS_NAMES  # type: ignore  # noqa: E402
from model import ClassConditionedUNet  # type: ignore  # noqa: E402
from sampling import autocast_context, resolve_amp_settings  # type: ignore  # noqa: E402


def load_flow_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    freeze: bool = False,
) -> tuple[ClassConditionedUNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ClassConditionedUNet()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if freeze:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    metadata = copy.deepcopy(checkpoint.get("metadata", {}))
    metadata.setdefault("requested_class_names", TARGET_CLASS_NAMES)
    metadata.setdefault("local_to_global", {str(k): int(v) for k, v in LOCAL_TO_GLOBAL.items()})
    metadata.setdefault("local_to_name", {str(k): v for k, v in LOCAL_TO_NAME.items()})
    return model, metadata


__all__ = [
    "ClassConditionedUNet",
    "LOCAL_TO_GLOBAL",
    "LOCAL_TO_NAME",
    "TARGET_CLASS_NAMES",
    "autocast_context",
    "load_flow_checkpoint",
    "resolve_amp_settings",
]
