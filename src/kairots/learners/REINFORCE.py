"""REINFORCE (Williams, 1992) on the Battery environment.

Each window is one episode: actions are sampled closed loop from the policy's
logits (the row of the current charge), the environment pays each step's reward
from the hidden prices, and the loss is the
negative log-probability of the actions weighted by the return to go minus a
baseline (the batch mean return to go at that step). `loss_function` scores evaluation
only; this learner owns its objective.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kairots.environments.Battery import Battery
from kairots.learners.Gradient import Gradient


class REINFORCE(Gradient):
    @dataclass(kw_only=True)
    class Config(Gradient.Config):
        name: str = "REINFORCE"
        discount: float = 1.0

    def __init__(self, *, config: Config) -> None:
        self.config: REINFORCE.Config = config

    def fit_epoch(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[tuple[Tensor, Tensor]],
        device: torch.device,
        loss_function: Any,
        regularizer: Any,
    ) -> float:
        """One update per batch; returns the negative mean episode return."""
        environment = Battery()
        model.train()
        return_sum = torch.zeros((), device=device)
        episodes = 0
        for batch_x, price in batches:
            batch_x = batch_x.to(device=device)
            price = price.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            actions, rewards, chosen = environment.rollout(
                logits=model(batch_x=batch_x),
                price=price,
                choose=lambda logits: torch.distributions.Categorical(
                    logits=logits
                ).sample(),
            )
            policy = torch.distributions.Categorical(logits=chosen)
            to_go = torch.zeros_like(rewards)
            running = torch.zeros_like(rewards[:, 0])
            for index in reversed(range(rewards.shape[1])):
                running = rewards[:, index] + self.config.discount * running
                to_go[:, index] = running
            advantage = to_go - to_go.mean(dim=0)
            loss = -(policy.log_prob(actions) * advantage).sum(dim=1).mean()
            (loss + regularizer()).backward()
            optimizer.step()
            return_sum += rewards.sum()
            episodes += rewards.shape[0]
        return float((-return_sum / max(episodes, 1)).item())
