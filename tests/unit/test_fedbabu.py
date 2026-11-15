from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from kairots.config.schema import Experiment
from kairots.learners.Gradient import Gradient
from kairots.optimizers.SGD import SGD
from kairots.runtime.contracts import ClientState, Message
from kairots.strategies.FedBABU import FedBABU


class _BodyHead(nn.Module):
    def __init__(self, *, head: bool = True) -> None:
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1) if head else nn.Identity()

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.head(self.body(batch_x))


def _mse(*, prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def _fit(*, model: nn.Module) -> None:
    learner = Gradient(
        config=Gradient.Config(
            optimizer=SGD.Config(), learning_rate=0.1, fused_optimizer=False
        )
    )
    FedBABU.Client().fit(
        learner=learner,
        model=model,
        optimizer=learner.optimizer(model=model),
        batches=[(torch.randn(4, 2), torch.randn(4, 1)) for _ in range(2)],
        epochs=2,
        device=torch.device("cpu"),
        loss_function=_mse,
        reference_parameters={},
    )


def test_fit_trains_the_body_and_freezes_the_head() -> None:
    torch.manual_seed(0)
    model = _BodyHead()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    _fit(model=model)
    after = model.state_dict()
    assert not torch.equal(after["body.weight"], before["body.weight"])
    assert torch.equal(after["head.weight"], before["head.weight"])
    assert torch.equal(after["head.bias"], before["head.bias"])


def test_fit_refuses_a_head_without_parameters() -> None:
    with pytest.raises(ValueError, match="head has parameters"):
        _fit(model=_BodyHead(head=False))


def test_only_the_body_travels_and_the_head_keeps_its_start() -> None:
    strategy = FedBABU(config=FedBABU.Config())
    global_params = {"body.w": torch.tensor([1.0]), "head.w": torch.tensor([5.0])}
    assert set(
        strategy.build_downlink(global_params=global_params, client_id=0).tensors
    ) == {"body.w"}
    state = strategy.initial_state(global_params=global_params)
    assert set(state.model_state) == {"head.w"}
    uplink, kept = strategy.split_state(parameters=global_params)
    assert set(uplink) == {"body.w"} and set(kept) == {"head.w"}
    assert (
        strategy.client_parameters(global_params=global_params, state=ClientState())
        is global_params
    )
    results = [
        SimpleNamespace(
            uplink=Message(
                sender="",
                receiver="",
                tensors={"body.w": value},
                scalars={"train_samples": float(samples)},
            ),
            train_samples=samples,
        )
        for value, samples in ((torch.tensor([0.0]), 1), (torch.tensor([4.0]), 3))
    ]
    aggregated = strategy.aggregate(
        uplinks=[result.uplink for result in results], global_params=global_params
    )
    assert torch.equal(aggregated["body.w"], torch.tensor([3.0]))
    assert torch.equal(aggregated["head.w"], torch.tensor([5.0]))


def test_schema_requires_fine_tuning_epochs() -> None:
    with pytest.raises(ValueError, match="finetune_epochs > 0"):
        Experiment.from_dict(values={"strategy": {"name": "FedBABU"}})
    with pytest.raises(ValueError, match=">= 0"):
        Experiment.from_dict(values={"evaluation": {"finetune_epochs": -1}})
