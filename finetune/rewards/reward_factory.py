from __future__ import annotations

import torch

from .base import BaseRewardFunction, RewardOutputs
from .classifier_reward import ImageNetClassifierReward
from .musiq_reward import MUSIQReward


class ClassifierOnlyReward(BaseRewardFunction):
    def __init__(self, classifier: ImageNetClassifierReward):
        self.classifier = classifier

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> RewardOutputs:
        classifier_reward = self.classifier(images, labels)
        return RewardOutputs(total=classifier_reward, classifier=classifier_reward, musiq=None)


class ClassifierPlusMUSIQReward(BaseRewardFunction):
    def __init__(
        self,
        classifier: ImageNetClassifierReward,
        musiq: MUSIQReward,
        alpha: float,
        beta: float,
    ):
        self.classifier = classifier
        self.musiq = musiq
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> RewardOutputs:
        classifier_reward = self.classifier(images, labels)
        _, musiq_reward = self.musiq(images)
        total = self.alpha * classifier_reward + self.beta * musiq_reward
        return RewardOutputs(total=total, classifier=classifier_reward, musiq=musiq_reward)


def build_reward_function(config: dict, device: torch.device) -> BaseRewardFunction:
    classifier = ImageNetClassifierReward(config=config["classifier"], device=device)
    setting = str(config["setting"]).lower()
    if setting == "classifier":
        return ClassifierOnlyReward(classifier=classifier)
    if setting == "classifier_plus_musiq":
        musiq = MUSIQReward(config=config["musiq"], device=device)
        return ClassifierPlusMUSIQReward(
            classifier=classifier,
            musiq=musiq,
            alpha=float(config.get("alpha", 1.0)),
            beta=float(config.get("beta", 0.1)),
        )
    raise ValueError(f"Unsupported reward setting: {setting}")
