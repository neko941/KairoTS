from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import sys

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.FedTREND import SYNTHETIC, FedTREND

INPUT, OUTPUT = 4, 3


class _Line(nn.Module):
    """A univariate forecaster: [batch, input, 1] -> [batch, output, 1]."""

    @dataclass(kw_only=True)
    class Config:
        name: str = "Line"

    def __init__(self, *, config: "_Line.Config | None" = None) -> None:
        super().__init__()
        self.layer = nn.Linear(INPUT, OUTPUT)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x[..., 0])[..., None]


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys.modules[FedTREND.__module__], "get", lambda *, category, name: _Line
    )


def _experiment(**changes: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "model": _Line.Config(),
        "lengths": lambda: (INPUT, OUTPUT, 0),
        "runtime": SimpleNamespace(seed=0, device="cpu", num_gpus=0),
        "groups": (),
        "client_model": None,
        "learner": SimpleNamespace(name="Gradient"),
        "data": SimpleNamespace(source=SimpleNamespace(batch_size=4)),
    }
    return SimpleNamespace(**{**values, **changes})


def _strategy(**changes: Any) -> FedTREND:
    small = {
        "construction_iterations": 2,
        "multistep": 1,
        "synthetic_steps": 2,
        "client_synthetic_size": 5,
    }
    strategy = FedTREND(config=FedTREND.Config(**{**small, **changes}))
    strategy.setup(experiment=_experiment())
    return strategy


def _uplink(*, client_id: int, state: dict[str, Tensor], samples: int = 0) -> Message:
    return Message(
        sender=f"client:{client_id}",
        receiver="server",
        tensors=state,
        scalars={"train_samples": float(samples)} if samples else {},
    )


def _shifted(*, state: dict[str, Tensor], by: Tensor | float) -> dict[str, Tensor]:
    return {key: value + by for key, value in state.items()}


def test_forecasting_on_univariate_clients_only() -> None:
    assert FedTREND.TASKS == ("Forecasting",)
    assert FedTREND.SOURCES == ("RowConcat", "Collection")
    FedTREND.validate(experiment=_experiment())
    with pytest.raises(ValueError, match="offset_length"):
        FedTREND.validate(experiment=_experiment(lengths=lambda: (4, 3, 1)))
    with pytest.raises(ValueError, match="one model architecture"):
        FedTREND.validate(experiment=_experiment(groups=(1,)))
    assert FedTREND.REQUIRES == {"learner.name": ("Gradient",)}


def test_synthetic_learning_rate_follows_the_reference() -> None:
    assert FedTREND.Config().synthetic_learning_rate == 0.1
    assert FedTREND.Config(reference="paper").synthetic_learning_rate == 3e-4
    assert FedTREND.Config(synthetic_learning_rate=0.5).synthetic_learning_rate == 0.5
    with pytest.raises(ValueError, match="global_interval"):
        FedTREND.Config(global_interval=2, global_segment=2)


def _official_mask(
    *, history: dict[str, Tensor] | None, update: dict[str, Tensor], epoch: int
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """`FedTSClients.consistency_mask` of the official code."""
    if epoch == 0:
        return {k: torch.ones_like(v) for k, v in update.items()}, {
            k: (v >= 0).float() for k, v in update.items()
        }
    assert history is not None
    mask = {
        k: (torch.where(update[k] >= 0, history[k], 1 - history[k]) > 0.3).float()
        for k in update
    }
    history = {
        k: (history[k] * epoch + (update[k] >= 0).float()) / (epoch + 1) for k in update
    }
    return mask, history


def test_code_aggregation_averages_sign_masked_updates_unweighted() -> None:
    strategy = _strategy(client_synthetic=False, global_refinement=False)
    torch.manual_seed(0)
    global_params = _Line().state_dict()
    histories: dict[int, Any] = {0: None, 1: None}
    for epoch in range(4):
        deltas = {
            client: {k: torch.randn_like(v) for k, v in global_params.items()}
            for client in (0, 1)
        }
        uplinks = [
            _uplink(client_id=client, state=_shifted(state=global_params, by=0.0))
            for client in (0, 1)
        ]
        for client, message in zip((0, 1), uplinks, strict=True):
            message.tensors = {
                k: global_params[k] + deltas[client][k] for k in global_params
            }
        expected = {k: v.clone() for k, v in global_params.items()}
        masked = []
        for client in (0, 1):
            mask, histories[client] = _official_mask(
                history=histories[client], update=deltas[client], epoch=epoch
            )
            masked.append({k: mask[k] * deltas[client][k] for k in mask})
        expected = {
            k: global_params[k] + (masked[0][k] + masked[1][k]) / 2
            for k in global_params
        }
        result = strategy.aggregate(uplinks=uplinks, global_params=global_params)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key])
        global_params = result
    assert strategy.uplink_scalars(tensors=global_params, train_samples=9) == {}


def test_paper_aggregation_is_the_sample_weighted_mean() -> None:
    strategy = _strategy(
        reference="paper", client_synthetic=False, global_refinement=False
    )
    global_params = _Line().state_dict()
    uplinks = [
        _uplink(client_id=0, state=_shifted(state=global_params, by=1.0), samples=1),
        _uplink(client_id=1, state=_shifted(state=global_params, by=-1.0), samples=3),
    ]
    result = strategy.aggregate(uplinks=uplinks, global_params=global_params)
    for key, value in global_params.items():
        torch.testing.assert_close(result[key], value - 0.5)
    assert strategy.uplink_scalars(tensors=global_params, train_samples=9) == {
        "train_samples": 9.0
    }


def _official_step(
    *,
    series: Tensor,
    start: Tensor,
    end: Tensor,
    config: FedTREND.Config,
    forward: Any,
    client: bool,
) -> Tensor:
    """One outer step of the official `server_MTT` / `client_MTT` with the whole
    synthetic set in each inner batch (so the draw order does not matter)."""
    series = series.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([series], lr=config.synthetic_learning_rate)
    syn_x = series[:, :INPUT, None]
    syn_y = series[:, INPUT:, None].clone().detach()
    student = start.clone().requires_grad_(True)
    for _ in range(config.synthetic_steps):
        target_y = forward(end, syn_x)
        y_pred = forward(student, syn_x)
        soft = syn_y * config.soft_label_ratio + target_y * (
            1 - config.soft_label_ratio
        )
        grad = torch.autograd.grad(
            functional.mse_loss(y_pred, soft), student, create_graph=True
        )[0]
        student = student - config.inner_learning_rate * grad
    if client:
        loss = functional.cosine_embedding_loss(
            student[None], end[None], torch.tensor([1])
        )
    else:
        param_loss = functional.mse_loss(student, end, reduction="sum")
        param_loss = param_loss / functional.mse_loss(start, end, reduction="sum")
        loss = param_loss / param_loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return series.detach()


@pytest.mark.parametrize("client", [False, True])
def test_construction_is_the_official_mtt_step(client: bool) -> None:
    strategy = _strategy(
        construction_iterations=0, synthetic_batch_size=64, synthetic_steps=3
    )
    torch.manual_seed(1)
    start = torch.randn(INPUT * OUTPUT + OUTPUT)
    end = start + 0.1 * torch.randn_like(start)
    distance = strategy.client_distance if client else strategy.global_distance
    # Code mode runs construction_iterations + 1 = 1 outer step.
    built = strategy.construct(
        size=6,
        segment=lambda: (start, end, torch.ones_like(start)),
        distance=distance,
        generator=torch.Generator().manual_seed(5),
    )
    initial = torch.randn(6, INPUT + OUTPUT, generator=torch.Generator().manual_seed(5))
    expected = _official_step(
        series=initial,
        start=start,
        end=end,
        config=strategy.config,
        forward=lambda flat, batch_x: strategy._forward(flat=flat, batch_x=batch_x),
        client=client,
    )
    torch.testing.assert_close(built, expected)
    assert not torch.equal(built, initial)


def test_client_synthetic_is_built_every_interval_and_billed_once_per_client() -> None:
    strategy = _strategy(client_interval=2, global_refinement=False)
    global_params = _Line().state_dict()
    model_bytes = sum(v.numel() * v.element_size() for v in global_params.values())
    for current in range(4):
        downlinks = [
            strategy.build_downlink(global_params=global_params, client_id=client)
            for client in (0, 1)
        ]
        extra = 5 * (INPUT + OUTPUT) * 4 if current in (1, 3) else 0
        assert [message.billed_bytes for message in downlinks] == [
            model_bytes + extra
        ] * 2
        assert (SYNTHETIC in downlinks[0].tensors) == bool(extra)
        uplinks = [
            _uplink(
                client_id=client,
                state=_shifted(state=global_params, by=0.01 * (client + 1)),
            )
            for client in (0, 1)
        ]
        global_params = strategy.aggregate(uplinks=uplinks, global_params=global_params)
    assert strategy.client_version == 2
    assert strategy.client_synthetic is not None
    assert strategy.client_synthetic.shape == (5, INPUT + OUTPUT)


def test_code_segment_is_one_round_paper_segment_spans_the_interval() -> None:
    global_params = _Line().state_dict()
    starts = {}
    for reference in ("code", "paper"):
        strategy = _strategy(
            reference=reference, client_interval=3, global_refinement=False
        )
        state = global_params
        for current in range(3):
            uplink = _uplink(
                client_id=0, state=_shifted(state=state, by=0.1), samples=1
            )
            state = strategy.aggregate(uplinks=[uplink], global_params=state)
            starts[reference, current] = strategy.client_start[0].clone()
    flat = strategy._flat
    # After round 1 (not a construction round) the code's start is the round-1
    # model; the paper keeps the round-0 upload (the last construction's end).
    assert not torch.equal(starts["code", 1], starts["code", 0])
    torch.testing.assert_close(starts["paper", 1], starts["paper", 0])
    torch.testing.assert_close(
        starts["paper", 0], flat(state=_shifted(state=global_params, by=0.1))
    )


def test_paper_mask_keeps_elements_whose_update_sign_repeats() -> None:
    strategy = _strategy(reference="paper", client_interval=2, global_refinement=False)
    global_params = {
        key: torch.zeros_like(value) for key, value in _Line().state_dict().items()
    }
    first = {key: torch.ones_like(value) for key, value in global_params.items()}
    second = {key: value.clone() for key, value in first.items()}
    second["layer.bias"] = -second["layer.bias"]
    seen: list[Tensor] = []
    strategy.construct = lambda **kwargs: (
        seen.append(kwargs["segment"]()[2]) or torch.zeros(5, 7)
    )  # type: ignore[method-assign]
    for update in (first, second, first):
        uplink = _uplink(
            client_id=0, state=_shifted(state=global_params, by=0.0), samples=1
        )
        uplink.tensors = {key: global_params[key] + update[key] for key in update}
        strategy.aggregate(uplinks=[uplink], global_params=global_params)
    # Round 0 has no previous update; round 2 flips only the bias back.
    assert seen[0].all()
    bias = OUTPUT
    assert not seen[1][-bias:].any() and seen[1][:-bias].all()


@pytest.mark.parametrize(("reference", "refined"), [("code", 2), ("paper", 4)])
def test_global_model_is_refined_after_each_construction(
    reference: str, refined: int
) -> None:
    strategy = _strategy(
        reference=reference,
        client_synthetic=False,
        global_interval=3,
        global_segment=1,
        refinement_rounds=2,
    )
    calls: list[int] = []
    original = strategy.refine

    def refine(*, state: dict[str, Tensor]) -> dict[str, Tensor]:
        calls.append(strategy.current_iteration)
        return original(state=state)

    strategy.refine = refine  # type: ignore[method-assign]
    state = _Line().state_dict()
    for client_ids in ((0, 1),) * 7:
        uplinks = [
            _uplink(client_id=c, state=_shifted(state=state, by=0.01), samples=1)
            for c in client_ids
        ]
        state = strategy.aggregate(uplinks=uplinks, global_params=state)
    # D_gt is built in rounds 3 and 6; code refines the 2 rounds after each one,
    # paper every round.
    expected = [3, 4, 6] if reference == "code" else [3, 4, 5, 6]
    assert calls == expected[: len(calls)] and len(calls) == (
        3 if reference == "code" else refined
    )
    assert strategy.global_synthetic is not None
    assert strategy.global_synthetic.shape == (2, INPUT + OUTPUT)


def test_refinement_is_sgd_on_the_global_synthetic_pairs() -> None:
    strategy = _strategy(refinement_batch_size=64)
    torch.manual_seed(2)
    strategy.global_synthetic = torch.randn(3, INPUT + OUTPUT)
    model = _Line()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    result = strategy.refine(state=state)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    series = strategy.global_synthetic
    loss = functional.mse_loss(
        model(batch_x=series[:, :INPUT, None]), series[:, INPUT:, None]
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(result[key], value)


class _Learner:
    def __init__(self) -> None:
        self.batches: list[tuple[Tensor, Tensor]] = []

    def fit_epoch(self, *, batches: Any, **_: Any) -> float:
        self.batches = list(batches)
        return 0.0


def test_client_trains_on_local_and_synthetic_pairs_in_shuffled_batches() -> None:
    synthetic = torch.arange(2 * (INPUT + OUTPUT), dtype=torch.float32).view(2, -1)
    client = FedTREND.Client(
        synthetic=synthetic, input_length=INPUT, batch_size=4, seed=0
    )
    local = [
        (torch.zeros(4, INPUT, 1), torch.zeros(4, OUTPUT, 1)),
        (torch.zeros(1, INPUT, 1), torch.zeros(1, OUTPUT, 1)),
    ]
    learner = _Learner()
    client.fit(
        learner=learner,
        model=_Line(),
        optimizer=None,
        batches=local,
        epochs=1,
        device=torch.device("cpu"),
        loss_function=None,
        reference_parameters={},
    )
    inputs = torch.cat([batch_x for batch_x, _ in learner.batches])
    targets = torch.cat([batch_y for _, batch_y in learner.batches])
    assert [len(batch_y) for _, batch_y in learner.batches] == [4, 3]
    assert inputs.shape == (7, INPUT, 1) and targets.shape == (7, OUTPUT, 1)
    rows = {
        tuple(torch.cat([x[:, 0], y[:, 0]]).tolist())
        for x, y in zip(inputs, targets, strict=True)
    }
    assert {tuple(row.tolist()) for row in synthetic} <= rows


def test_client_strips_the_synthetic_downlink_and_keeps_it() -> None:
    strategy = _strategy()
    synthetic = torch.randn(5, INPUT + OUTPUT)
    seen: list[Any] = []

    class _Worker:
        def client_experiment(self, *, client_id: int) -> SimpleNamespace:
            return _experiment()

        def update(self, *, job: Job, client: FedTREND.Client) -> Result:
            seen.append((set(job.downlink.tensors), client.synthetic))
            return Result(
                operation_id=job.operation_id,
                uplink=Message(sender="client:0", receiver="server"),
                state=ClientState(extra=job.state.extra),
                metrics={},
                train_samples=1,
            )

    params = _Line().state_dict()
    job = Job(
        operation_id=(0, 0, "fit"),
        operation="fit",
        downlink=Message(
            sender="server",
            receiver="client:0",
            tensors={**params, SYNTHETIC: synthetic},
        ),
        state=ClientState(),
        seed=0,
        client_id=0,
    )
    result = strategy.client_update(worker=_Worker(), job=job)
    assert seen[0][0] == set(params) and seen[0][1] is synthetic
    # The orchestrator bills the job's downlink, which keeps D_ct.
    assert SYNTHETIC in job.downlink.tensors
    kept = Job(
        **{
            **vars(job),
            "downlink": Message(sender="server", receiver="client:0", tensors=params),
            "state": result.state,
        }
    )
    strategy.client_update(worker=_Worker(), job=kept)
    assert seen[1][1] is synthetic


@pytest.mark.parametrize("reference", ["code", "paper"])
def test_billed_bytes_per_direction_against_fedavg(reference: str) -> None:
    from kairots.strategies.FedAvg import FedAvg

    fedavg = FedAvg(config=FedAvg.Config())
    strategy = _strategy(
        reference=reference, client_interval=2, global_refinement=False
    )
    global_params = _Line().state_dict()
    synthetic_bytes = 5 * (INPUT + OUTPUT) * 4
    for current in range(3):
        for client in (0, 1):
            baseline = fedavg.build_downlink(
                global_params=global_params, client_id=client
            )
            sent = strategy.build_downlink(
                global_params=global_params, client_id=client
            )
            # Downlink: the FedAvg model, plus D_ct once after each construction.
            extra = synthetic_bytes if current == 1 else 0
            assert sent.billed_bytes == baseline.billed_bytes + extra
        uplink = {key: value + 0.01 for key, value in global_params.items()}
        fedavg_up = Message(
            sender="client:0",
            receiver="server",
            tensors=uplink,
            scalars=fedavg.uplink_scalars(tensors=uplink, train_samples=8),
        )
        ours = Message(
            sender="client:0",
            receiver="server",
            tensors=uplink,
            scalars=strategy.uplink_scalars(tensors=uplink, train_samples=8),
        )
        # Uplink: the whole model (D_gt, banks and masks stay on the server); the
        # sample count (4 bytes) only in paper mode, which averages sample-weighted.
        count = 0 if reference == "code" else 4
        assert ours.billed_bytes == fedavg_up.billed_bytes - 4 + count
        global_params = strategy.aggregate(
            uplinks=[
                _uplink(client_id=c, state=uplink, samples=8 if count else 0)
                for c in (0, 1)
            ],
            global_params=global_params,
        )
