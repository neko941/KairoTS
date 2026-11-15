"""Battery arbitrage against a load-driven price.

One episode is the `output_length` steps after a window. The price of a step is the
last channel of the series at that step (the load in the units the store keeps,
raw under the `Identity` scaler) floored at zero, hidden from the policy until the
reward is paid. Each step the policy discharges (0), idles (1) or charges (2) at
most `rate`; charging stores `efficiency` of what it buys and the charge stays
within [0, capacity]. The reward is the cost saved against no battery: price times
energy sold minus energy bought. Episodes end at the horizon (truncation); there is
no termination and no randomness.

A policy acts closed loop through `rollout`: its logits hold one row per charge
bin and each step reads the row of the current charge. `foresight` is the best
return of an episode whose prices are known in advance, the upper bound of every
policy.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


class Battery:
    ACTIONS = 3

    @dataclass(kw_only=True)
    class Config:
        name: str = "Battery"
        capacity: float = 1.0
        rate: float = 0.25
        efficiency: float = 0.9
        initial: float = 0.5

    def __init__(self, *, config: Config | None = None) -> None:
        self.config = config or self.Config()

    def reset(self, *, batch_size: int, device: torch.device) -> Tensor:
        """The charge of each episode at its first step."""
        return torch.full(
            (batch_size,), self.config.initial * self.config.capacity, device=device
        )

    def step(
        self, *, charge: Tensor, action: Tensor, price: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The next charge and this step's reward."""
        config = self.config
        bought = torch.where(
            action == 2,
            torch.clamp(
                (config.capacity - charge) / config.efficiency, max=config.rate
            ),
            0.0,
        )
        sold = torch.where(action == 0, torch.clamp(charge, max=config.rate), 0.0)
        return (
            charge + bought * config.efficiency - sold,
            price.clamp(min=0.0) * (sold - bought),
        )

    def rollout(
        self, *, logits: Tensor, price: Tensor, choose: Callable[[Tensor], Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Play `[episodes, steps, bins, actions]` logits (or `[episodes, steps,
        actions]`, one bin) against their prices. Returns the actions, the rewards
        and the logits each action was chosen from, all per step.
        """
        if logits.dim() == 3:
            logits = logits[:, :, None]
        bins = logits.shape[2]
        rows = torch.arange(logits.shape[0], device=logits.device)
        charge = self.reset(batch_size=logits.shape[0], device=logits.device)
        actions, rewards, chosen = [], [], []
        for index in range(logits.shape[1]):
            # Nearest bin (charges are non-negative).
            level = (charge / self.config.capacity * (bins - 1) + 0.5).long()
            step_logits = logits[rows, index, level]
            action = choose(step_logits)
            charge, reward = self.step(
                charge=charge, action=action, price=price[:, index]
            )
            actions.append(action)
            rewards.append(reward)
            chosen.append(step_logits)
        return (
            torch.stack(actions, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(chosen, dim=1),
        )

    def foresight(self, *, price: Tensor) -> Tensor:
        """Each episode's best return with every price known: dynamic programming
        over the charges reachable from the initial one.
        """
        # Charges evolve the same way whatever the prices, so the reachable charges
        # (keyed to 1e-9) and each action's energy balance are computed once.
        charges = [self.config.initial * self.config.capacity]
        slots = {int(charges[0] * 1e9 + 0.5): 0}
        successors: list[list[int]] = []
        balances: list[list[float]] = []
        while len(successors) < len(charges):
            following, balance = self.step(
                charge=torch.full(
                    (self.ACTIONS,), charges[len(successors)], dtype=torch.float64
                ),
                action=torch.arange(self.ACTIONS),
                price=torch.ones(1, dtype=torch.float64),
            )
            row = []
            for charge in following.tolist():
                key = int(charge * 1e9 + 0.5)
                if key not in slots:
                    slots[key] = len(charges)
                    charges.append(charge)
                row.append(slots[key])
            successors.append(row)
            balances.append(balance.tolist())
            if len(charges) > 100_000:
                raise ValueError("the reachable battery charges do not close")
        following = torch.tensor(successors)
        balance = torch.tensor(balances, dtype=price.dtype)
        floor = price.clamp(min=0.0).cpu()
        value = torch.zeros(price.shape[0], len(charges), dtype=price.dtype)
        for column in reversed(range(price.shape[1])):
            value = (floor[:, column, None, None] * balance + value[:, following]).amax(
                dim=-1
            )
        return value[:, 0].to(price.device)
