from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from kairots.config.schema import Experiment
from kairots.learners.Gradient import Gradient
from kairots.optimizers.SGD import SGD
from kairots.runtime.contracts import ClientState, Job, Message, Result
from kairots.strategies.FML import FML


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 3)

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return self.layer(batch_x)


def _cross_entropy(*, prediction: Tensor, target: Tensor) -> Tensor:
    return functional.cross_entropy(prediction, target)


def _step(
    *,
    private: nn.Module,
    meme: nn.Module,
    private_optimizer: torch.optim.Optimizer,
    meme_optimizer: torch.optim.Optimizer,
    batch: tuple[Tensor, Tensor],
) -> None:
    batch_x, batch_y = batch
    output = private(batch_x=batch_x)
    output_g = meme(batch_x=batch_x)
    loss = 0.4 * _cross_entropy(
        prediction=output, target=batch_y
    ) + 0.6 * functional.kl_div(
        input=functional.log_softmax(output, dim=1),
        target=functional.softmax(output_g.detach(), dim=1),
        reduction="batchmean",
    )
    loss_g = 0.7 * _cross_entropy(
        prediction=output_g, target=batch_y
    ) + 0.3 * functional.kl_div(
        input=functional.log_softmax(output_g, dim=1),
        target=functional.softmax(output.detach(), dim=1),
        reduction="batchmean",
    )
    private_optimizer.zero_grad(set_to_none=True)
    meme_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    loss_g.backward()
    nn.utils.clip_grad_norm_(private.parameters(), 10)
    nn.utils.clip_grad_norm_(meme.parameters(), 10)
    private_optimizer.step()
    meme_optimizer.step()


def test_steps_the_private_and_meme_models_as_mutual_learning() -> None:
    torch.manual_seed(2)
    private = _Linear()
    meme = _Linear()
    expected_private = _Linear()
    expected_meme = _Linear()
    expected_private.load_state_dict(private.state_dict())
    expected_meme.load_state_dict(meme.state_dict())
    batches = [(torch.randn(4, 2), torch.tensor([0, 1, 2, 1])) for _ in range(2)]
    learner = Gradient(
        config=Gradient.Config(
            optimizer=SGD.Config(), learning_rate=0.05, fused_optimizer=False
        )
    )
    # The private optimizer takes the meme optimizer's (scheduled) learning rate.
    meme_optimizer = learner.optimizer(model=meme)
    expected_meme_optimizer = learner.optimizer(model=expected_meme)
    expected_private_optimizer = learner.optimizer(model=expected_private)
    for optimizer in (
        meme_optimizer,
        expected_meme_optimizer,
        expected_private_optimizer,
    ):
        optimizer.param_groups[0]["lr"] = 0.02
    for batch in batches:
        _step(
            private=expected_private,
            meme=expected_meme,
            private_optimizer=expected_private_optimizer,
            meme_optimizer=expected_meme_optimizer,
            batch=batch,
        )

    strategy = FML(config=FML.Config(alpha=0.4, beta=0.7))
    client = strategy.Client(
        personal={
            key: value.detach().clone() for key, value in private.state_dict().items()
        },
        personal_optimizer={},
        private_config=SimpleNamespace(name="_Linear"),
        alpha=0.4,
        beta=0.7,
    )
    # Replace registry construction with the keyword-only tiny linear model.
    with patch(
        "kairots.strategies.FML.get", lambda **kwargs: lambda *, config: _Linear()
    ):
        client.fit(
            learner=learner,
            model=meme,
            optimizer=meme_optimizer,
            batches=batches,
            epochs=1,
            device=torch.device("cpu"),
            loss_function=_cross_entropy,
            reference_parameters=meme.state_dict(),
        )
    assert client.personal_optimizer["param_groups"][0]["lr"] == 0.02
    for key in private.state_dict():
        assert torch.allclose(
            client.personal[key], expected_private.state_dict()[key], atol=1e-6
        )
    for key, value in meme.state_dict().items():
        assert torch.allclose(value, expected_meme.state_dict()[key], atol=1e-6)


def test_scores_the_private_model_and_averages_unweighted() -> None:
    strategy = FML(config=FML.Config())
    global_params = {"weight": torch.tensor([4.0])}
    state = ClientState(extra={"personal": {"weight": torch.tensor([8.0])}})
    assert (
        strategy.client_parameters(global_params=global_params, state=state)[
            "weight"
        ].item()
        == 8.0
    )
    results = [
        SimpleNamespace(uplink=SimpleNamespace(tensors={"weight": torch.tensor([v])}))
        for v in (1.0, 3.0)
    ]
    assert strategy.aggregate(
        uplinks=[result.uplink for result in results], global_params=global_params
    )["weight"].item() == pytest.approx(2.0)


def test_meme_optimizer_keeps_only_its_param_groups_after_client_update() -> None:
    strategy = FML(config=FML.Config())
    state = ClientState(optimizer_state={"stale": True})
    job = Job(
        operation_id=(0, 0, "fit"),
        operation="fit",
        downlink=Message(sender="server", receiver="client:0"),
        state=state,
        seed=0,
        client_id=0,
    )
    experiment = SimpleNamespace(
        client_model=None, model=SimpleNamespace(name="DLinear")
    )

    class _Worker:
        def client_experiment(self, *, client_id: int) -> SimpleNamespace:
            return experiment

        def update(self, *, job: Job, client: FML.Client) -> Result:
            return Result(
                operation_id=job.operation_id,
                uplink=Message(sender="client:0", receiver="server"),
                state=ClientState(
                    optimizer_state={
                        "state": {0: {"momentum_buffer": torch.ones(1)}},
                        "param_groups": [{"lr": 0.1, "params": [0]}],
                    }
                ),
                metrics={},
                train_samples=1,
            )

    result = strategy.client_update(worker=_Worker(), job=job)
    assert result.state.optimizer_state == {
        "state": {},
        "param_groups": [{"lr": 0.1, "params": [0]}],
    }
    assert result.state.extra["personal"] == job.downlink.tensors


def test_fml_refuses_forecasting() -> None:
    with pytest.raises(ValueError):
        Experiment.from_dict(values={"strategy": {"name": "FML"}})
