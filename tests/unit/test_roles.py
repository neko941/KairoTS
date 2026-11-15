"""Loss and metric roles share formulas while enforcing their config contracts."""

from typing import Any

import pytest
import torch

from kairots.config.schema import Experiment


def test_metric_only_component_cannot_be_a_loss() -> None:
    with pytest.raises(ValueError, match="loss"):
        Experiment.from_dict(values={"evaluation": {"loss": "R2"}})
    with pytest.raises(ValueError, match="metric"):
        Experiment.from_dict(
            values={
                "task": {"name": "Classification"},
                "model": {"name": "DLinear"},
                "data": {"source": {"name": "BasicMotions"}},
                "evaluation": {"metrics": {"test": ["CrossEntropy"]}},
            }
        )


def test_sample_crps_roles_use_one_shared_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module("kairots.metrics.SampleCRPS")
    calls: list[int] = []

    def controlled(
        *, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(1)
        return prediction.new_tensor(12.0), prediction.new_tensor(6)

    monkeypatch.setattr(module, "sample_crps", controlled)
    quantity = module.SampleCRPS()
    prediction, target = torch.zeros(2, 3, 1, 4), torch.zeros(2, 3, 1)
    assert set(quantity.ROLES) == {"loss", "metric"}
    assert quantity(prediction=prediction, target=target).item() == 2.0
    quantity.update(prediction=prediction, target=target)
    assert quantity.compute() == 2.0
    assert len(calls) == 2


def test_return_loss_and_metric_match_episode_rewards() -> None:
    from kairots.environments.Battery import Battery
    from kairots.metrics.Return import Return

    generator = torch.Generator().manual_seed(21)
    prediction = torch.randn(4, 5, 3, generator=generator)
    target = torch.rand(4, 5, generator=generator)
    _, rewards, _ = Battery().rollout(
        logits=prediction, price=target, choose=lambda logits: logits.argmax(dim=-1)
    )
    quantity = Return(config=Return.Config())
    loss = quantity(prediction=prediction, target=target)
    quantity.update(prediction=prediction, target=target)
    assert loss.item() == pytest.approx(-rewards.sum(dim=1).mean().item())
    assert quantity.compute() == pytest.approx(-loss.item())


def test_mse_supports_batch_gradients_and_merged_metrics() -> None:
    from kairots.metrics.MSE import MSE

    prediction = torch.tensor([1.0, 4.0, 9.0], requires_grad=True)
    target = torch.tensor([0.0, 2.0, float("nan")])
    quantity, right = MSE(), MSE()
    loss = quantity(prediction=prediction, target=target)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.0, 2.0, 0.0]))
    quantity.update(prediction=prediction[:1], target=target[:1])
    right.update(prediction=prediction[1:], target=target[1:])
    quantity.merge(other=right)
    assert quantity.compute() == pytest.approx(loss.item())
    Experiment.from_dict(values={"evaluation": {"metrics": {"test": ["MSE"]}}})


def test_mae_loss_matches_l1_and_masks_unscored_gradients() -> None:
    from kairots.metrics.MAE import MAE

    prediction = torch.tensor([1.0, -4.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 2.0, 2.0])
    quantity = MAE()
    loss = quantity(prediction=prediction, target=target)
    reference = torch.nn.functional.l1_loss(prediction, target)
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(loss, prediction)[0],
        torch.autograd.grad(reference, prediction)[0],
        rtol=0,
        atol=0,
    )
    target[-1] = torch.nan
    masked = quantity(prediction=prediction, target=target)
    torch.testing.assert_close(
        torch.autograd.grad(masked, prediction)[0], torch.tensor([0.5, -0.5, 0.0])
    )
    assert quantity.absolute_error is None and quantity.count is None
    quantity.update(prediction=prediction.detach(), target=target)
    assert quantity.compute() == masked.item()
    Experiment.from_dict(values={"evaluation": {"loss": "MAE"}})
    with pytest.raises(ValueError, match="no scored targets"):
        quantity(prediction=prediction, target=torch.full_like(target, torch.nan))


def test_frozen_reuses_return_loss_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairots.environments.Battery import Battery
    from kairots.evaluation.Frozen import Frozen
    from kairots.metrics.Return import Return
    from kairots.registry import get

    experiment = Experiment.from_dict(
        values={
            "task": {"name": "Control"},
            "model": {"name": "Policy"},
            "learner": {"name": "REINFORCE"},
            "data": {
                "source": {"input_channels": ["OT"]},
                "split": {"input_length": 4, "output_length": 2},
            },
            "evaluation": {"loss": "Return", "metrics": {"test": ["Return"]}},
        }
    )

    class Actions(torch.nn.Module):
        def forward(self, *, batch_x: torch.Tensor) -> torch.Tensor:
            return batch_x.new_zeros(len(batch_x), 2, 3)

    calls: list[int] = []
    rollout = Battery.rollout

    def recorded(self: Battery, **kwargs: Any) -> Any:
        calls.append(1)
        return rollout(self, **kwargs)

    monkeypatch.setattr(Battery, "rollout", recorded)
    Frozen(config=experiment.evaluation, loss_function=Return()).evaluate(
        phase="test",
        model=Actions(),
        clients=[(torch.ones(10, 1), {"test": [(0, 4), (1, 5)]})],
        experiment=experiment,
        task=get(category="tasks", name="Control")(config=experiment.task),
        view=get(category="data/views", name="Offline")(config=experiment.data.view),
        device=torch.device("cpu"),
    )
    assert len(calls) == 1
