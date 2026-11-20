import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.runtime.contracts import ClientState
from kairots.strategies.MoEFL import MoEFL


class _Line(nn.Module):
    @dataclass(kw_only=True)
    class Config:
        name: str = "Line"
        num_classes: int = 3

    def __init__(self, *, config: "_Line.Config | None" = None) -> None:
        super().__init__()
        self.layer = nn.Linear(6, (config or _Line.Config()).num_classes)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x.flatten(1))


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules[MoEFL.__module__], "get", lambda *, category, name: _Line
    )


def _experiment() -> SimpleNamespace:
    return SimpleNamespace(model=_Line.Config())


def _mixture(*, reference: str = "code") -> MoEFL.Mixture:
    strategy = MoEFL(config=MoEFL.Config(reference=reference))
    return strategy._mixture(expert=_Line(), experiment=_experiment())


def test_gate_is_the_expert_architecture_with_one_sigmoid_output() -> None:
    mixture = _mixture()
    assert isinstance(mixture.gate.body, _Line)
    weight = mixture.gate(batch_x=torch.randn(5, 3, 2))
    assert weight.shape == (5, 1) and ((weight > 0) & (weight < 1)).all()


@pytest.mark.parametrize("reference", ["code", "paper"])
def test_mixture_rule_per_reference(reference: str) -> None:
    torch.manual_seed(0)
    mixture = _mixture(reference=reference).eval()
    with torch.no_grad():
        cast(Any, mixture.local_expert).layer.bias += 1.0
    batch_x = torch.randn(4, 3, 2)
    weight = mixture.gate(batch_x=batch_x)
    local = functional.log_softmax(mixture.local_expert(batch_x=batch_x), dim=-1)
    shared = functional.log_softmax(mixture.global_expert(batch_x=batch_x), dim=-1)
    if reference == "code":
        expected = weight * local + (1 - weight) * shared
    else:
        expected = torch.log(weight * local.exp() + (1 - weight) * shared.exp())
        assert torch.allclose(expected.exp().sum(dim=-1), torch.ones(4))
    torch.testing.assert_close(mixture(batch_x=batch_x), expected)


def test_without_a_mixture_a_client_is_scored_with_the_global_model() -> None:
    torch.manual_seed(0)
    strategy = MoEFL(config=MoEFL.Config())
    model = _Line()
    mixture = strategy.personal_model(model=model, experiment=_experiment()).eval()
    global_params = {key: value + 0.3 for key, value in model.state_dict().items()}
    mixture.load_state_dict(
        strategy.client_parameters(global_params=global_params, state=ClientState())
    )
    model.load_state_dict(global_params)
    batch_x = torch.randn(4, 3, 2)
    torch.testing.assert_close(
        mixture(batch_x=batch_x),
        functional.log_softmax(model(batch_x=batch_x), dim=-1),
    )


def _batches(*, count: int, seed: int) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(8, 3, 2, generator=generator),
            torch.randint(0, 3, (8,), generator=generator),
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize("gate_only", [False, True])
def test_mixture_freezes_global_finetunes_specialist_and_trains_gate(
    gate_only: bool,
) -> None:
    torch.manual_seed(0)
    config = MoEFL.Config(
        specialist_epochs=3, mixture_epochs=3, train_gate_only=gate_only
    )
    strategy = MoEFL(config=config)
    global_params = {key: value.clone() for key, value in _Line().state_dict().items()}
    client = MoEFL.Client(
        mixture=strategy._mixture(expert=_Line(), experiment=_experiment()),
        config=config,
        validation=_batches(count=1, seed=1),
        mixture_state={},
    )
    gate_before = {
        key: value.clone() for key, value in client.mixture.gate.state_dict().items()
    }
    _, state = client.fit_mixture(
        batches=_batches(count=3, seed=0),
        device=torch.device("cpu"),
        global_parameters=global_params,
    )
    for key, value in global_params.items():
        assert torch.equal(state[f"global_expert.{key}"], value)
    # Step 2 fine-tunes the specialist from the global model in both settings.
    assert not torch.equal(
        state["local_expert.layer.weight"], global_params["layer.weight"]
    )
    assert any(
        not torch.equal(state[f"gate.{key}"], value)
        for key, value in gate_before.items()
    )


def test_paper_mode_restores_the_best_validated_weights() -> None:
    torch.manual_seed(0)
    config = MoEFL.Config(
        reference="paper",
        specialist_epochs=0,
        mixture_epochs=12,
        validation_interval=5,
        stopping_checks=10,
        mixture_learning_rate=0.5,
    )
    strategy = MoEFL(config=config)
    client = MoEFL.Client(
        mixture=strategy._mixture(expert=_Line(), experiment=_experiment()),
        config=config,
        validation=_batches(count=1, seed=1),
        mixture_state={},
    )
    snapshots = []
    original = client._validation_loss

    def recording(*, module: nn.Module, device: torch.device) -> float:
        loss = original(module=module, device=device)
        snapshots.append(
            (
                loss,
                {
                    key: value.clone()
                    for key, value in client.mixture.gate.state_dict().items()
                },
            )
        )
        return loss

    client._validation_loss = recording  # type: ignore[method-assign]
    client.fit_mixture(
        batches=_batches(count=2, seed=0),
        device=torch.device("cpu"),
        global_parameters=_Line().state_dict(),
    )
    # Checks at epochs 0, 5, 10; the kept state is the first that improved most.
    best_loss, best_gate = snapshots[0]
    for loss, gate in snapshots[1:]:
        if loss < best_loss - config.minimum_improvement:
            best_loss, best_gate = loss, gate
    for key, value in client.mixture.gate.state_dict().items():
        torch.testing.assert_close(value, best_gate[key])


def test_rounds_select_opt_in_clients_then_everyone_mixes() -> None:
    strategy = MoEFL(
        config=MoEFL.Config(iterations=4, join_ratio=0.2, opt_out_fraction=0.5)
    )
    clients = list(range(10))
    opted_out = strategy.opted_out(client_ids=clients)
    assert len(opted_out) == 5
    for iteration in range(3):
        selected = strategy.select_clients(
            client_ids=clients, current_iteration=iteration, seed=iteration
        )
        assert len(selected) == 2 and not set(selected) & opted_out
    assert (
        strategy.select_clients(client_ids=clients, current_iteration=3, seed=0)
        == clients
    )


def test_validate_rejects_finetuning_and_a_single_round() -> None:
    experiment = SimpleNamespace(
        evaluation=SimpleNamespace(finetune_epochs=1),
        groups=(),
        client_model=None,
        strategy=MoEFL.Config(iterations=3),
        model=_Line.Config(),
    )
    with pytest.raises(ValueError, match="finetune_epochs"):
        MoEFL.validate(experiment=experiment)
    experiment.evaluation.finetune_epochs = 0
    experiment.strategy = MoEFL.Config(iterations=1)
    with pytest.raises(ValueError, match="FedAvg round"):
        MoEFL.validate(experiment=experiment)


def test_classification_only() -> None:
    assert MoEFL.TASKS == ("Classification",)


def test_billing_fedavg_rounds_send_model_and_count_mixture_round_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairots.runtime.contracts import Job, Message, Result

    strategy = MoEFL(
        config=MoEFL.Config(iterations=3, specialist_epochs=0, mixture_epochs=0)
    )
    global_params = {key: value.clone() for key, value in _Line().state_dict().items()}
    model_bytes = sum(
        value.numel() * value.element_size() for value in global_params.values()
    )
    monkeypatch.setattr(
        MoEFL, "_validation", staticmethod(lambda **_: _batches(count=1, seed=2))
    )

    class _Worker:
        def client_experiment(self, *, client_id: int) -> SimpleNamespace:
            return _experiment()

        def update(self, *, job: Job, client: MoEFL.Client) -> Result:
            client.fit(
                learner=None,
                model=_Line(),
                optimizer=None,
                batches=_batches(count=1, seed=3),
                epochs=0,
                device=torch.device("cpu"),
                loss_function=None,
                reference_parameters=job.downlink.tensors,
            )
            tensors = {key: value.clone() for key, value in global_params.items()}
            return Result(
                operation_id=job.operation_id,
                uplink=Message(
                    sender="client:0",
                    receiver="server",
                    tensors=tensors,
                    scalars=strategy.uplink_scalars(tensors=tensors, train_samples=8),
                ),
                state=ClientState(),
                metrics={},
                train_samples=8,
            )

    def job(iteration: int) -> Job:
        downlink = strategy.build_downlink(global_params=global_params, client_id=0)
        return Job(
            operation_id=(iteration, 0, "fit"),
            operation="fit",
            downlink=downlink,
            state=ClientState(),
            seed=0,
            client_id=0,
        )

    # Every round, the client receives the global model (paper Step 1; opt-out
    # clients get it at the end, here the mixture round).
    assert job(0).downlink.billed_bytes == model_bytes
    fedavg = strategy.client_update(worker=_Worker(), job=job(0))
    assert fedavg.uplink.billed_bytes == model_bytes + 4  # model + sample count
    mixing = strategy.client_update(worker=_Worker(), job=job(2))
    # The specialist and gate stay on the client; the mixture round sends nothing.
    assert mixing.uplink.billed_bytes == 0 and not mixing.uplink.tensors
    assert set(mixing.state.extra["mixture"]) >= {
        "gate.body.layer.weight",
        "local_expert.layer.weight",
    }
    # The server leaves the global model unchanged when nothing was sent.
    assert (
        strategy.aggregate(uplinks=[mixing.uplink], global_params=global_params)
        is global_params
    )
