from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from kairots.learners.Gradient import Gradient
from kairots.optimizers.SGD import SGD
from kairots.runtime.contracts import ClientState
from kairots.strategies.APFL import APFL


class _Line(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x)


def _mse(*, prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean()


def _loss(*, parameters: dict[str, Tensor], batch: tuple[Tensor, Tensor]) -> Tensor:
    batch_x, batch_y = batch
    prediction = batch_x @ parameters["layer.weight"].T + parameters["layer.bias"]
    return _mse(prediction=prediction, target=batch_y)


def test_steps_the_global_model_the_local_model_and_alpha() -> None:
    torch.manual_seed(0)
    model = _Line()
    w = {key: value.detach().clone() for key, value in model.state_dict().items()}
    v = {key: value + 0.5 for key, value in w.items()}
    batches = [(torch.randn(4, 2), torch.randn(4, 1)) for _ in range(2)]
    learner = Gradient(
        config=Gradient.Config(
            optimizer=SGD.Config(), learning_rate=0.1, fused_optimizer=False
        )
    )
    client = APFL.Client(
        personal={key: value.clone() for key, value in v.items()},
        alpha=0.3,
        adaptive_alpha=True,
    )
    client.fit(
        learner=learner,
        model=model,
        optimizer=learner.optimizer(model=model),
        batches=batches,
        epochs=2,
        device=torch.device("cpu"),
        loss_function=_mse,
        reference_parameters=w,
    )
    alpha = 0.3
    for step in range(4):
        batch = batches[step % 2]
        leaves_w = {key: value.clone().requires_grad_() for key, value in w.items()}
        leaves_v = {key: value.clone().requires_grad_() for key, value in v.items()}
        weight = torch.tensor(alpha, requires_grad=True)
        mixed = {key: weight * leaves_v[key] + (1 - weight) * w[key] for key in w}
        _loss(parameters=mixed, batch=batch).backward()
        _loss(parameters=leaves_w, batch=batch).backward()
        w = {key: w[key] - 0.1 * leaves_w[key].grad for key in w}
        v = {key: v[key] - 0.1 * leaves_v[key].grad for key in v}
        if step == 0:
            assert weight.grad is not None
            alpha = min(max(alpha - 0.1 * weight.grad.item(), 0.0), 1.0)
    assert client.alpha == pytest.approx(alpha)
    assert client.alpha != pytest.approx(0.3)
    for key, value in model.state_dict().items():
        assert torch.allclose(value, w[key], atol=1e-6)
        assert torch.allclose(client.personal[key], v[key], atol=1e-6)


def test_scores_the_mixed_model_and_averages_unweighted() -> None:
    strategy = APFL(config=APFL.Config(alpha=0.25))
    global_params = {"w": torch.tensor([4.0])}
    state = strategy.initial_state(global_params=global_params)
    state = ClientState(extra={**state.extra, "personal": {"w": torch.tensor([8.0])}})
    mixed = strategy.client_parameters(global_params=global_params, state=state)
    assert mixed["w"].item() == pytest.approx(0.25 * 8.0 + 0.75 * 4.0)
    results = [
        SimpleNamespace(uplink=SimpleNamespace(tensors={"w": torch.tensor([value])}))
        for value in (1.0, 3.0)
    ]
    updated = strategy.aggregate(
        uplinks=[result.uplink for result in results], global_params=global_params
    )
    assert updated["w"].item() == pytest.approx(2.0)
