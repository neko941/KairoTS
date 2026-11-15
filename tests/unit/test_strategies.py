from __future__ import annotations

import inspect

import pytest
import torch
from torch import Tensor, nn

from kairots.learners.Gradient import Gradient
from kairots.metrics.MSE import MSE
from kairots.registry import get, names
from kairots.runtime.contracts import ClientState, Message, Result
from kairots.strategies.base import Strategy
from kairots.strategies.Decentralized import Decentralized
from kairots.strategies.DFedProx import DFedProx
from kairots.strategies.Ditto import Ditto
from kairots.strategies.FedAvg import FedAvg


def test_fedavg_is_exact_sample_weighted_mean() -> None:
    strategy = FedAvg(config=FedAvg.Config())
    results = [
        Result(
            operation_id=(0, index, "fit"),
            uplink=Message(
                sender="client",
                receiver="server",
                tensors={"p": torch.tensor([value])},
                scalars={"train_samples": float(count)},
            ),
            state=ClientState(version=1),
            metrics={},
            # Unsent counts never weight the average: only the uplink does.
            train_samples=100,
        )
        for index, value, count in ((0, 1.0, 1), (1, 4.0, 3))
    ]
    assert (
        strategy.aggregate(
            uplinks=[result.uplink for result in results], global_params={}
        )["p"].item()
        == 3.25
    )
    for result in results:
        result.uplink.scalars = {}
    with pytest.raises(KeyError, match="train_samples"):
        strategy.aggregate(
            uplinks=[result.uplink for result in results], global_params={}
        )


def test_decentralized_exchange_gossips_metropolis_weighted_models() -> None:
    strategy = Decentralized(config=Decentralized.Config(topology="Ring"))
    results = [
        Result(
            operation_id=(0, client_id, "fit"),
            uplink=Message(sender=f"client:{client_id}", receiver="server"),
            state=ClientState(model_state={"p": torch.tensor(value)}),
            metrics={},
            train_samples=1,
        )
        for client_id, value in enumerate((0.0, 3.0, 6.0))
    ]
    messages = strategy.exchange(results=results)
    assert [result.state.model_state["p"].item() for result in results] == [
        3.0,
        3.0,
        3.0,
    ]
    assert len(messages) == 6
    assert sum(message.billed_bytes for message in messages) == 24


def test_fedprox_client_can_add_proximal_term_without_worker_changes() -> None:
    class FedProx(FedAvg):
        class Client(Strategy.Client):
            def proximal_term(
                self, *, model: nn.Module, reference_parameters: dict[str, Tensor]
            ) -> Tensor:
                return sum(
                    (
                        (parameter - reference_parameters[name].to(parameter.device))
                        .square()
                        .sum()
                        for name, parameter in model.named_parameters()
                    ),
                    torch.zeros(()),
                )

    model = nn.Linear(1, 1)
    reference = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert (
        FedProx.Client()
        .proximal_term(model=model, reference_parameters=reference)
        .item()
        > 0
    )


class _TinyModel(torch.nn.Module):
    def __init__(self, *, layer: torch.nn.Linear) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x)


def test_ditto_with_zero_personal_epochs_keeps_initial_personal_model() -> None:
    torch.manual_seed(42)
    linear = torch.nn.Linear(2, 1)
    model = _TinyModel(layer=linear)
    initial_personal = {
        key: value.detach().clone() for key, value in linear.state_dict().items()
    }
    initial_global = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    client = Ditto.Client(
        personal={key: value.clone() for key, value in initial_personal.items()},
        mu=0.1,
        personal_epochs=0,
    )
    learner = Gradient(config=Gradient.Config())
    optimizer = learner.optimizer(model=model)
    batches = [(torch.randn(4, 2), torch.randn(4, 1))]
    loss_function = MSE(config=MSE.Config())
    reference_parameters = {
        key: value.detach().clone() for key, value in initial_global.items()
    }
    client.fit(
        learner=learner,
        model=model,
        optimizer=optimizer,
        batches=batches,
        epochs=1,
        device=torch.device("cpu"),
        loss_function=loss_function,
        reference_parameters=reference_parameters,
    )
    for key, value in initial_personal.items():
        assert torch.equal(client.personal[key], value)
    assert any(
        not torch.equal(model.state_dict()[key], initial_global[key])
        for key in initial_global
    )


def test_ditto_proximal_term_pulls_personal_toward_global() -> None:
    torch.manual_seed(42)
    reference_parameters = {
        "layer.weight": torch.full((1, 2), 5.0),
        "layer.bias": torch.full((1,), 5.0),
    }
    initial_personal = {
        "layer.weight": torch.zeros((1, 2)),
        "layer.bias": torch.zeros((1,)),
    }
    batches = [(torch.randn(16, 2), torch.full((16, 1), -5.0))]
    learner = Gradient(config=Gradient.Config(learning_rate=0.001))
    loss_function = MSE(config=MSE.Config())

    linear_zero = torch.nn.Linear(2, 1)
    model_zero = _TinyModel(layer=linear_zero)
    optimizer_zero = learner.optimizer(model=model_zero)
    client_zero = Ditto.Client(
        personal={key: value.clone() for key, value in initial_personal.items()},
        mu=0.0,
        personal_epochs=1,
    )
    client_zero.fit(
        learner=learner,
        model=model_zero,
        optimizer=optimizer_zero,
        batches=batches,
        epochs=1,
        device=torch.device("cpu"),
        loss_function=loss_function,
        reference_parameters=reference_parameters,
    )

    linear_regularized = torch.nn.Linear(2, 1)
    model_regularized = _TinyModel(layer=linear_regularized)
    optimizer_regularized = learner.optimizer(model=model_regularized)
    client_regularized = Ditto.Client(
        personal={key: value.clone() for key, value in initial_personal.items()},
        mu=100.0,
        personal_epochs=1,
    )
    client_regularized.fit(
        learner=learner,
        model=model_regularized,
        optimizer=optimizer_regularized,
        batches=batches,
        epochs=1,
        device=torch.device("cpu"),
        loss_function=loss_function,
        reference_parameters=reference_parameters,
    )

    distance_zero = (
        torch.stack(
            [
                (client_zero.personal[key] - reference_parameters[key]).square().sum()
                for key in reference_parameters
            ]
        )
        .sum()
        .sqrt()
        .item()
    )
    distance_regularized = (
        torch.stack(
            [
                (client_regularized.personal[key] - reference_parameters[key])
                .square()
                .sum()
                for key in reference_parameters
            ]
        )
        .sum()
        .sqrt()
        .item()
    )
    assert distance_regularized < distance_zero


def test_parent_families_that_override_one_hook_clash() -> None:
    class Keeper(Strategy):
        def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
            return ClientState(model_state=dict(global_params))

    class Holder(Strategy):
        def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
            return ClientState(extra={"held": dict(global_params)})

    with pytest.raises(TypeError, match="initial_state"):

        class Clash(Keeper, Holder):
            pass

    class Resolved(Keeper, Holder):
        def initial_state(self, *, global_params: dict[str, Tensor]) -> ClientState:
            return Holder.initial_state(self, global_params=global_params)

    assert "held" in Resolved().initial_state(global_params={}).extra


def test_disjoint_families_combine_and_proximal_term_is_half_mu_distance() -> None:
    strategy = DFedProx(config=DFedProx.Config(mu=4.0, topology="Ring"))
    assert type(strategy).exchange is Decentralized.exchange
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(1.0)
    reference = {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)}
    term = DFedProx.Client(mu=strategy.config.mu).proximal_term(
        model=model, reference_parameters=reference
    )
    assert term.item() == 20.0


def test_fedpmt_trains_only_its_suffix_and_averages_who_trained() -> None:
    from kairots.strategies.FedPMT import FedPMT

    strategy = FedPMT(config=FedPMT.Config())
    groups = ["first", "second", "third"]
    assert strategy.trainable(client_id=0, groups=groups) == ["third"]
    assert strategy.trainable(client_id=4, groups=groups) == ["second", "third"]
    with pytest.raises(ValueError, match="depths"):
        FedPMT(config=FedPMT.Config(depths=(4,))).trainable(client_id=0, groups=groups)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)

        def forward(self, *, batch_x: Tensor) -> Tensor:
            return self.second(self.first(batch_x))

    torch.manual_seed(0)
    model = Tiny()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    learner = Gradient(config=Gradient.Config(learning_rate=0.1))
    client = FedPMT.Client(client_id=0, trainable=strategy.trainable)
    client.fit(
        learner=learner,
        model=model,
        optimizer=learner.optimizer(model=model),
        batches=[(torch.randn(4, 2), torch.randn(4, 2))],
        epochs=1,
        device=torch.device("cpu"),
        loss_function=MSE(config=MSE.Config()),
        reference_parameters=before,
    )
    assert client.trained == ["second.weight", "second.bias"]
    assert torch.equal(model.first.weight, before["first.weight"])
    assert not torch.equal(model.second.weight, before["second.weight"])

    results = [
        Result(
            operation_id=(0, index, "fit"),
            uplink=Message(sender="client", receiver="server", tensors=tensors),
            state=ClientState(version=1),
            metrics={},
            train_samples=count,
        )
        for index, tensors, count in (
            (0, {"a": torch.tensor([2.0])}, 1),
            (1, {"a": torch.tensor([4.0]), "b": torch.tensor([7.0])}, 9),
        )
    ]
    merged = strategy.aggregate(
        uplinks=[result.uplink for result in results],
        global_params={
            "a": torch.tensor([0.0]),
            "b": torch.tensor([0.0]),
            "c": torch.tensor([5.0]),
        },
    )
    assert [merged[name].item() for name in "abc"] == [3.0, 7.0, 5.0]


def test_fedpaq_quantizer_is_unbiased_on_its_grid_and_bills_encoded_bits() -> None:
    from kairots.strategies.FedPAQ import FedPAQ

    strategy = FedPAQ(config=FedPAQ.Config(levels=4))
    vector = torch.tensor([3.0, -4.0, 0.5, 0.0])
    generator = torch.Generator().manual_seed(0)
    draws = torch.stack(
        [strategy.quantize(vector=vector, generator=generator) for _ in range(4000)]
    )
    # Every value is sign * norm * k / levels, and the mean recovers the vector.
    grid = draws.abs() / vector.norm() * 4
    assert torch.allclose(grid, grid.round(), atol=1e-5)
    assert torch.allclose(draws.mean(dim=0), vector, atol=0.1)
    # 4 levels: 5 magnitudes in 3 bits + 1 sign bit per value, plus a 32-bit norm.
    assert strategy.encoded_bytes(numel=100) == 54
    off = FedPAQ(config=FedPAQ.Config(levels=0))
    assert torch.equal(off.quantize(vector=vector, generator=generator), vector)
    assert off.encoded_bytes(numel=100) is None
    message = Message(
        sender="client:0", receiver="server", tensors={"p": vector}, encoded_bytes=42
    )
    assert message.billed_bytes == 42


def test_server_hooks_see_only_uplink_messages() -> None:
    # The server is billed for and may read only what crossed the network.
    for name in names(category="strategies"):
        strategy_type = get(category="strategies", name=name)
        for hook in ("aggregate", "broadcast"):
            parameters = inspect.signature(getattr(strategy_type, hook)).parameters
            assert "uplinks" in parameters and "results" not in parameters, (name, hook)


def test_messages_carry_only_billable_values() -> None:
    message = Message(
        sender="client:0",
        receiver="server",
        tensors={"p": torch.zeros(3)},
        scalars={"train_samples": 5, "tau": 2.5},
    )
    assert message.billed_bytes == 3 * 4 + 2 * 4
    with pytest.raises(TypeError, match="scalar 'mask'"):
        Message(sender="a", receiver="b", scalars={"mask": [1, 0]})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="scalar 'flag'"):
        Message(sender="a", receiver="b", scalars={"flag": True})
    with pytest.raises(TypeError, match="tensor 'p'"):
        Message(sender="a", receiver="b", tensors={"p": [0.0]})  # type: ignore[dict-item]
