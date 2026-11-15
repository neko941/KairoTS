from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from kairots.learners.Gradient import Gradient
from kairots.optimizers.SGD import SGD
from kairots.strategies.SCAFFOLD import CONTROL, SCAFFOLD


class _Line(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x)


def _mse(*, prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def _gradient(
    *, parameters: dict[str, Tensor], batch: tuple[Tensor, Tensor]
) -> dict[str, Tensor]:
    leaves = {key: value.clone().requires_grad_() for key, value in parameters.items()}
    batch_x, batch_y = batch
    prediction = batch_x @ leaves["layer.weight"].T + leaves["layer.bias"]
    _mse(prediction=prediction, target=batch_y).backward()
    return {key: value.grad for key, value in leaves.items() if value.grad is not None}


def _fit(
    *, control_update: str, server: dict[str, Tensor], own: dict[str, Tensor]
) -> tuple[_Line, dict[str, Tensor], list[tuple[Tensor, Tensor]], SCAFFOLD.Client]:
    torch.manual_seed(0)
    model = _Line()
    start = {key: value.detach().clone() for key, value in model.state_dict().items()}
    batches = [(torch.randn(4, 2), torch.randn(4, 1)) for _ in range(3)]
    learner = Gradient(
        config=Gradient.Config(
            optimizer=SGD.Config(), learning_rate=0.1, fused_optimizer=False
        )
    )
    client = SCAFFOLD.Client(server=server, own=own, control_update=control_update)
    client.fit(
        learner=learner,
        model=model,
        optimizer=learner.optimizer(model=model),
        batches=batches,
        epochs=2,
        device=torch.device("cpu"),
        loss_function=_mse,
        reference_parameters=start,
    )
    return model, start, batches, client


def _controls(*, seed: int) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "layer.weight": torch.randn(1, 2, generator=generator),
        "layer.bias": torch.randn(1, generator=generator),
    }


def test_option_ii_steps_with_the_correction_and_keeps_the_mean_gradient() -> None:
    server, own = _controls(seed=1), _controls(seed=2)
    model, start, batches, client = _fit(control_update="II", server=server, own=own)
    expected = dict(start)
    gradients = []
    for _ in range(2):
        for batch in batches:
            gradient = _gradient(parameters=expected, batch=batch)
            gradients.append(gradient)
            expected = {
                key: value - 0.1 * (gradient[key] - own[key] + server[key])
                for key, value in expected.items()
            }
    for key, value in model.state_dict().items():
        assert torch.allclose(value, expected[key], atol=1e-6)
    # Option II equals the mean raw gradient of the K local steps.
    for key in start:
        mean = torch.stack([gradient[key] for gradient in gradients]).mean(dim=0)
        assert torch.allclose(client.updated[key], mean, atol=1e-5)


def test_option_i_keeps_the_gradient_at_the_server_model() -> None:
    zero = {key: torch.zeros_like(value) for key, value in _controls(seed=1).items()}
    _, start, batches, client = _fit(control_update="I", server=zero, own=zero)
    for key in start:
        mean = torch.stack(
            [_gradient(parameters=start, batch=batch)[key] for batch in batches]
        ).mean(dim=0)
        assert torch.allclose(client.updated[key], mean, atol=1e-6)


def test_server_steps_the_model_and_the_control_variate() -> None:
    strategy = SCAFFOLD(config=SCAFFOLD.Config(global_step_size=2.0))
    strategy.select_clients(client_ids=[0, 1, 2, 3], current_iteration=0, seed=0)
    global_params = {"w": torch.tensor([1.0])}
    strategy.build_downlink(global_params=global_params, client_id=0)
    results = [
        SimpleNamespace(
            uplink=SimpleNamespace(
                tensors={"w": torch.tensor([delta]), CONTROL + "w": torch.tensor([c])}
            )
        )
        for delta, c in ((0.5, 1.0), (1.5, 3.0))
    ]
    updated = strategy.aggregate(
        uplinks=[result.uplink for result in results], global_params=global_params
    )
    assert updated["w"].item() == pytest.approx(1.0 + 2.0 * 1.0)
    assert strategy.control["w"].item() == pytest.approx((1.0 + 3.0) / 4)


def test_refuses_an_adaptive_optimizer() -> None:
    from kairots.config.schema import Experiment

    with pytest.raises(ValueError, match=r"learner\.optimizer\.name"):
        Experiment.from_dict(
            values={
                "strategy": {"name": "SCAFFOLD"},
                "learner": {"optimizer": {"name": "Adam"}},
            }
        )


def test_option_ii_divides_by_the_summed_step_sizes() -> None:
    torch.manual_seed(0)
    model = _Line()
    start = {key: value.detach().clone() for key, value in model.state_dict().items()}
    zero = {key: torch.zeros_like(value) for key, value in start.items()}
    learner = Gradient(
        config=Gradient.Config(
            optimizer=SGD.Config(), learning_rate=0.1, fused_optimizer=False
        )
    )
    optimizer = learner.optimizer(model=model)
    # A within-update schedule: the rate halves after every step (sum 0.1875).
    optimizer.register_step_post_hook(
        lambda *_: optimizer.param_groups[0].update(
            lr=optimizer.param_groups[0]["lr"] / 2
        )
    )
    client = SCAFFOLD.Client(server=zero, own=zero, control_update="II")
    client.fit(
        learner=learner,
        model=model,
        optimizer=optimizer,
        batches=[(torch.randn(4, 2), torch.randn(4, 1)) for _ in range(2)],
        epochs=2,
        device=torch.device("cpu"),
        loss_function=_mse,
        reference_parameters=start,
    )
    for key, value in model.state_dict().items():
        assert torch.allclose(client.updated[key], (start[key] - value) / 0.1875)
