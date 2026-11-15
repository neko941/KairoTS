from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fixtures.synthetic import ett_experiment, write_ett_sources
from torch import Tensor, nn

from kairots.analysis.landscape import (
    Config,
    filter_normalized,
    hessian_trace,
    interpolation,
    landscape,
    mean_loss,
    surface,
    top_eigenvalue,
)
from kairots.config.schema import Experiment
from kairots.metrics.MSE import MSE
from kairots.runtime.orchestrator import run


def test_plot_dimensions_preserve_grid_coordinates(
    workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kairots.analysis.landscape import one_d, three_d, two_d

    pyplot = MagicMock()
    figure, axes = MagicMock(), MagicMock()
    pyplot.subplots.return_value = figure, axes
    pyplot.figure.return_value = figure
    figure.add_subplot.return_value = axes
    package = MagicMock()
    package.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    axis = np.array([-1.0, 0.0, 1.0])
    values = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    one_d.plot(axis=axis, values=values[:, 0], path=workspace_tmp_path / "line.png")
    np.testing.assert_array_equal(axes.plot.call_args.args[1], [0, 10, 20])
    two_d.plot(axis=axis, values=values, path=workspace_tmp_path / "contour.png")
    np.testing.assert_array_equal(axes.contourf.call_args.args[2], values.T)
    three_d.plot(axis=axis, values=values, path=workspace_tmp_path / "surface.png")
    first, second, height = axes.plot_surface.call_args.args
    np.testing.assert_array_equal(first[:, 0], axis)
    np.testing.assert_array_equal(second[0, :], axis)
    np.testing.assert_array_equal(height, values)
    assert pyplot.close.call_count == 3


def test_all_landscape_views_render_png(workspace_tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from kairots.analysis.landscape import one_d, three_d, two_d

    axis = np.array([-1.0, 0.0, 1.0])
    values = np.array([[2.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]])
    for name, module in (("1d", one_d), ("2d", two_d), ("3d", three_d)):
        path = workspace_tmp_path / f"{name}.png"
        module.plot(
            axis=axis, values=values[:, 0] if name == "1d" else values, path=path
        )
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


class Quadratic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.2, -0.1, 0.3]]))

    def forward(self, *, batch_x: Tensor) -> Tensor:
        return (batch_x @ self.weight.T).reshape(-1, 1, 1)


class FilterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)


def test_quadratic_surface_and_hessian_estimators() -> None:
    model = Quadratic()
    batch_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    batch_y = torch.zeros((3, 1, 1))
    batches = [(batch_x, batch_y)]
    loss_function = MSE(config=MSE.Config())
    hessian = 2 * batch_x.T @ batch_x / batch_x.shape[0]
    assert math.isclose(
        top_eigenvalue(
            model=model,
            batches=batches,
            loss_function=loss_function,
            iterations=20,
            generator=torch.Generator().manual_seed(4),
        ),
        float(torch.linalg.eigvalsh(hessian)[-1]),
        rel_tol=1e-5,
    )
    trace = hessian_trace(
        model=model,
        batches=batches,
        loss_function=loss_function,
        samples=2000,
        generator=torch.Generator().manual_seed(5),
    )
    assert math.isclose(trace, float(torch.trace(hessian)), rel_tol=0.1)
    direction = {"weight": torch.tensor([[0.1, 0.2, -0.1]])}
    points = 5
    values = surface(
        model=model,
        batches=batches,
        loss_function=loss_function,
        directions=(direction, None),
        points=points,
        span=1.0,
    )
    axis = torch.linspace(-1, 1, points)
    expected = torch.stack(
        [
            (
                (batch_x @ (model.weight.detach() + point * direction["weight"]).T) ** 2
            ).mean()
            for point in axis
        ]
    )
    torch.testing.assert_close(values, expected)
    assert mean_loss(
        model=model, batches=batches, loss_function=loss_function
    ) == float(expected[2])


def test_filter_normalization_zeros_bias_and_norm_parameters() -> None:
    model = FilterModel()
    directions = filter_normalized(
        model=model, generator=torch.Generator().manual_seed(7)
    )
    for index in range(4):
        torch.testing.assert_close(
            directions["linear.weight"][index].norm(),
            model.linear.weight[index].norm(),
        )
    assert torch.count_nonzero(directions["linear.bias"]) == 0
    assert torch.count_nonzero(directions["norm.weight"]) == 0
    assert torch.count_nonzero(directions["norm.bias"]) == 0


def test_saved_run_reproducibility_and_analysis_only_writes(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw, runs_dir=workspace_tmp_path / "runs", save_checkpoints=True
    ).to_dict()
    experiment["evaluation"]["metrics"]["train"] = ["MAE"]
    result = run(experiment=Experiment.from_dict(values=experiment))
    run_dir = Path(result.run_dir)
    snapshot = {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file() and "analysis" not in path.relative_to(run_dir).parts
    }
    settings = Config(points=3, iterations=3, samples=2, max_windows=10000, seed=9)
    first = landscape(run_dir=run_dir, config=settings)
    arrays = {
        name: value.copy()
        for name, value in np.load(run_dir / "analysis" / "landscape.npz").items()
    }
    # The saved run's prepared data is reused: the raw files are not needed.
    shutil.rmtree(raw)
    second = landscape(run_dir=run_dir, config=settings)
    repeated = np.load(run_dir / "analysis" / "landscape.npz")
    assert list(arrays) == repeated.files
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, repeated[name])
    assert first == second
    rounds = [
        json.loads(line) for line in (run_dir / "rounds.jsonl").read_text().splitlines()
    ]
    assert math.isclose(
        first["center_loss"],
        rounds[-1]["generalization_train_pooled_mse"],
        rel_tol=1e-5,
    )
    after = {
        path.relative_to(run_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file() and "analysis" not in path.relative_to(run_dir).parts
    }
    assert snapshot == after


def test_interpolation_runs_from_the_model_to_the_end_point() -> None:
    model = Quadratic()
    batches = [(torch.eye(3), torch.tensor([1.0, 0.0, 2.0]).reshape(3, 1, 1))]
    loss_function = MSE(config=MSE.Config())
    start = model.weight.detach().clone()
    end = {"weight": torch.tensor([[1.0, 0.0, 2.0]])}
    losses = interpolation(
        model=model, batches=batches, loss_function=loss_function, end=end, points=5
    )
    assert math.isclose(
        losses[0].item(),
        mean_loss(model=model, batches=batches, loss_function=loss_function),
    )
    assert losses[-1].item() == 0.0
    assert torch.equal(model.weight, start)


def test_personalized_run_gets_client_sharpness_and_interpolation(
    workspace_tmp_path: Path,
) -> None:
    raw = workspace_tmp_path / "raw"
    write_ett_sources(path=raw)
    experiment = ett_experiment(
        raw_path=raw,
        runs_dir=workspace_tmp_path / "runs",
        strategy="LocalOnly",
        save_checkpoints=True,
    )
    run_dir = Path(run(experiment=experiment).run_dir)
    payload = landscape(
        run_dir=run_dir,
        config=Config(points=3, iterations=2, samples=1, dimensions=1),
    )
    assert len(payload["clients"]) == 2
    assert {f"client_{key}" for key in payload["clients"]} == set(
        payload["interpolation"]
    )
    assert all(len(losses) == 3 for losses in payload["interpolation"].values())
