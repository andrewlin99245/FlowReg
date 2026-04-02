from __future__ import annotations

from .batchot_reg import BatchOTRegularizer
from .none_reg import NoRegularizer
from .rfr_reg import RFRRegularizer
from .w2_reg import W2Regularizer


def build_regularizer(config: dict) -> object:
    reg_type = config["type"].lower()
    if reg_type == "no_reg":
        return NoRegularizer()
    if reg_type == "w2":
        return W2Regularizer(weight=float(config.get("lambda_w2", 0.0)))
    if reg_type == "rfr":
        return RFRRegularizer(weight=float(config.get("lambda_rfr", 0.0)))
    if reg_type == "batchot":
        return BatchOTRegularizer(
            weight=float(config.get("lambda_batchot", 0.0)),
            epsilon=float(config.get("sinkhorn_epsilon", 0.05)),
            num_iters=int(config.get("sinkhorn_iters", 50)),
        )
    raise ValueError(f"Unsupported regularizer type: {reg_type}")
