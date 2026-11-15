from __future__ import annotations

import torch
from torch import Tensor

from kairots.environments.Battery import Battery
from kairots.learners.REINFORCE import REINFORCE
from kairots.metrics.Return import Return
from kairots.models.Policy import Policy
from kairots.tasks.Control import Control


def test_battery_keeps_the_charge_in_bounds_and_pays_saved_cost() -> None:
    environment = Battery(
        config=Battery.Config(capacity=0.5, rate=0.5, efficiency=0.5, initial=0.0)
    )
    # Two charges fill it at half efficiency, the third buys nothing; one sale
    # empties it; idling earns nothing.
    actions = torch.tensor([[2, 2, 2, 0, 0, 0, 1]])
    price = torch.tensor([[1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 9.0]])
    assert _rewards(environment=environment, actions=actions, price=price).tolist() == [
        [-0.5, -0.5, -0.0, 1.5, 0.0, 0.0, 0.0]
    ]
    empty = _rewards(
        environment=environment, actions=torch.zeros_like(actions), price=price
    )
    assert empty.sum() == 0
    # A negative load is a price of zero.
    assert _rewards(
        environment=environment, actions=actions, price=-price
    ).tolist() == [[0.0] * 7]


def _rewards(*, environment: Battery, actions: Tensor, price: Tensor) -> Tensor:
    """Rewards of fixed actions, played as one-hot open-loop logits."""
    _, rewards, _ = environment.rollout(
        logits=torch.nn.functional.one_hot(actions, Battery.ACTIONS).float(),
        price=price,
        choose=lambda logits: logits.argmax(dim=-1),
    )
    return rewards


def test_rollout_reads_the_row_of_the_current_charge() -> None:
    environment = Battery(
        config=Battery.Config(capacity=1.0, rate=0.5, efficiency=1.0, initial=0.0)
    )
    # Two bins (empty, full): charge when empty, sell when full.
    logits = torch.zeros(1, 4, 2, Battery.ACTIONS)
    logits[:, :, 0, 2] = 1.0
    logits[:, :, 1, 0] = 1.0
    actions, _, _ = environment.rollout(
        logits=logits,
        price=torch.ones(1, 4),
        choose=lambda logits: logits.argmax(dim=-1),
    )
    # 0 -> 0.5 (bin 1 by nearest) -> sell -> 0.0 -> charge.
    assert actions.tolist() == [[2, 0, 2, 0]]


def test_foresight_is_the_best_of_every_action_sequence() -> None:
    import itertools

    environment = Battery()
    price = torch.rand(3, 5, generator=torch.Generator().manual_seed(2)) - 0.2
    best = torch.stack(
        [
            _rewards(
                environment=environment,
                actions=torch.tensor([sequence] * 3),
                price=price,
            ).sum(dim=1)
            for sequence in itertools.product(range(Battery.ACTIONS), repeat=5)
        ]
    ).amax(dim=0)
    assert torch.allclose(environment.foresight(price=price), best, atol=1e-6)


def test_control_keeps_horizon_prices_out_of_the_model_input() -> None:
    task = Control(config=Control.Config())
    batch_x = torch.randn(2, 4, 2)
    batch_y = torch.randn(2, 3, 2)
    observed, price = task.prepare(batch_x=batch_x, batch_y=batch_y, output_length=3)
    assert observed is batch_x
    assert torch.equal(price, batch_y[..., -1])


def test_reinforce_learns_to_buy_low_and_sell_high() -> None:
    torch.manual_seed(0)
    model = Policy(
        config=Policy.Config(
            input_length=4, output_length=8, input_channels=1, hidden=16
        )
    )
    learner = REINFORCE(
        config=REINFORCE.Config(learning_rate=0.05, fused_optimizer=False)
    )
    optimizer = learner.optimizer(model=model)
    batch_x = torch.randn(64, 4, 1)
    price = torch.tensor([0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0]).repeat(64, 1)

    def greedy() -> float:
        metric = Return()
        with torch.no_grad():
            metric.update(prediction=model(batch_x=batch_x), target=price)
        return metric.compute()

    before = greedy()
    for _ in range(150):
        learner.fit_epoch(
            model=model,
            optimizer=optimizer,
            batches=[(batch_x, price)],
            device=torch.device("cpu"),
            loss_function=None,
            regularizer=lambda: torch.zeros(()),
        )
    # Selling the initial 0.5 at 2 earns 1.0; filling first at 0.5 (cost 0.28)
    # and selling 1.0 at 2 earns 1.72.
    assert greedy() > max(before, 1.0)


def test_return_merges_like_one_pool() -> None:
    torch.manual_seed(1)
    logits = torch.randn(6, 5, 3)
    price = torch.randn(6, 5)
    pooled, left, right = Return(), Return(), Return()
    pooled.update(prediction=logits, target=price)
    left.update(prediction=logits[:2], target=price[:2])
    right.update(prediction=logits[2:], target=price[2:])
    left.merge(other=right)
    assert abs(left.compute() - pooled.compute()) < 1e-6
