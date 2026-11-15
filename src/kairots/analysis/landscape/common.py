"""Loss surfaces and Hessian sharpness for a saved model.

Directions follow Li et al., "Visualizing the Loss Landscape of Neural Nets"
(NeurIPS 2018). Sharpness uses PyHessian's Hessian-vector products and
Hutchinson estimator (Yao et al., 2020).

Hessian estimates use the pooled loss of at most `max_windows` windows per client,
and each client's own loss for its per-client sharpness. A run that keeps client
parameters (`<checkpoint>_clients.pt`: Ditto, LocalOnly) also gets each client's
loss along the line from the global model to its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kairots.config.schema import Experiment
from kairots.registry import get


@dataclass(kw_only=True)
class Config:
    # Grid points per axis, covering [-span, span] along each direction.
    points: int = 21
    span: float = 1.0
    # Seeds the directions, the power-iteration start and the Hutchinson probes.
    seed: int = 0
    # Windows per client, from the start of the phase.
    max_windows: int = 512
    # checkpoints/<checkpoint>.pt: final or best_val.
    checkpoint: str = "final"
    # train, validation, or test (only with allow_test).
    phase: str = "train"
    allow_test: bool = False
    # 1: a line along one direction; 2: a plane.
    dimensions: int = 2
    iterations: int = 20
    samples: int = 20


def filter_normalized(
    *, model: nn.Module, generator: torch.Generator
) -> dict[str, Tensor]:
    """Draw Gaussian directions scaled to each weight filter's norm."""
    directions: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.dim() <= 1:
            directions[name] = torch.zeros_like(parameter)
            continue
        direction = torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype, device="cpu"
        ).to(parameter.device)
        for index in range(parameter.shape[0]):
            direction[index] *= parameter[index].norm() / direction[
                index
            ].norm().clamp_min(1e-12)
        directions[name] = direction
    return directions


def _batches_loss(
    *, model: nn.Module, batches: list[tuple[Tensor, Tensor]], loss_function: Any
) -> Tensor:
    total = torch.zeros((), dtype=next(model.parameters()).dtype)
    count = torch.zeros((), dtype=total.dtype)
    for batch_x, batch_y in batches:
        batch_y = batch_y.to("cpu")
        prediction = model(batch_x=batch_x.to("cpu"))
        scored = batch_y.isfinite().sum()
        total = total + loss_function(prediction=prediction, target=batch_y) * scored
        count = count + scored
    if count.item() == 0:
        raise ValueError("phase has no scored positions")
    return total / count


def mean_loss(
    *, model: nn.Module, batches: list[tuple[Tensor, Tensor]], loss_function: Any
) -> float:
    """Return the scored-position-weighted mean loss across batches."""
    with torch.no_grad():
        return float(
            _batches_loss(
                model=model, batches=batches, loss_function=loss_function
            ).item()
        )


def surface(
    *,
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    loss_function: Any,
    directions: tuple[dict[str, Tensor], dict[str, Tensor] | None],
    points: int,
    span: float,
) -> Tensor:
    """Evaluate pooled loss on a one or two dimensional parameter grid."""
    parameters = dict(model.named_parameters())
    original = {name: value.detach().clone() for name, value in parameters.items()}
    axis = torch.linspace(-span, span, points)
    shape = (points,) if directions[1] is None else (points, points)
    result = torch.empty(shape)
    try:
        with torch.no_grad():
            for row, first in enumerate(axis):
                columns = (None,) if directions[1] is None else axis
                for column, second in enumerate(columns):
                    for name, parameter in parameters.items():
                        parameter.copy_(original[name] + first * directions[0][name])
                        if directions[1] is not None:
                            assert second is not None
                            parameter.add_(second * directions[1][name])
                    if directions[1] is None:
                        result[row] = mean_loss(
                            model=model, batches=batches, loss_function=loss_function
                        )
                    else:
                        result[row, column] = mean_loss(
                            model=model, batches=batches, loss_function=loss_function
                        )
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(original[name])
    return result


def _dot(*, first: list[Tensor], second: list[Tensor]) -> Tensor:
    return sum(
        ((left * right).sum() for left, right in zip(first, second, strict=True)),
        torch.zeros(()),
    )


def top_eigenvalue(
    *,
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    loss_function: Any,
    iterations: int,
    generator: torch.Generator,
) -> float:
    """Estimate the largest Hessian eigenvalue by power iteration."""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    vector = [
        torch.randn(parameter.shape, generator=generator).to(parameter)
        for parameter in parameters
    ]
    norm = torch.sqrt(sum((item.square().sum() for item in vector), torch.zeros(())))
    vector = [item / norm for item in vector]
    eigenvalue = torch.zeros(())
    for _ in range(iterations):
        objective = _batches_loss(
            model=model, batches=batches, loss_function=loss_function
        )
        gradient = torch.autograd.grad(objective, parameters, create_graph=True)
        product = torch.autograd.grad(
            _dot(first=list(gradient), second=vector), parameters
        )
        eigenvalue = _dot(first=list(product), second=vector)
        norm = torch.sqrt(
            sum((item.square().sum() for item in product), torch.zeros(()))
        )
        vector = [item / norm.clamp_min(1e-12) for item in product]
    return float(eigenvalue.detach().item())


def hessian_trace(
    *,
    model: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
    loss_function: Any,
    samples: int,
    generator: torch.Generator,
) -> float:
    """Estimate the Hessian trace with Rademacher Hutchinson probes."""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    estimates = []
    for _ in range(samples):
        objective = _batches_loss(
            model=model, batches=batches, loss_function=loss_function
        )
        gradient = torch.autograd.grad(objective, parameters, create_graph=True)
        probe = [
            torch.randint(0, 2, parameter.shape, generator=generator)
            .to(parameter)
            .mul_(2)
            .sub_(1)
            for parameter in parameters
        ]
        product = torch.autograd.grad(
            _dot(first=list(gradient), second=probe), parameters
        )
        estimates.append(_dot(first=list(product), second=probe))
    return float(torch.stack(estimates).mean().detach().item())


def landscape(*, run_dir: Path, config: Config) -> dict[str, Any]:
    """Rebuild a saved run, compute its surface and sharpness, and save artifacts."""
    from kairots.analysis.landscape import one_d, three_d, two_d

    if config.phase not in {"train", "validation", "test"}:
        raise ValueError("phase must be train, validation or test")
    if config.phase == "test" and not config.allow_test:
        raise ValueError("test phase requires allow_test=True")
    if config.dimensions not in {1, 2}:
        raise ValueError("dimensions must be 1 or 2")
    values = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))[
        "resolved"
    ]
    experiment = Experiment.from_dict(values=values)
    store_type = get(category="data/stores", name=experiment.data.store.name)
    records = store_type.records(
        manifest=json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        experiment=experiment,
    )
    view_type = get(category="data/views", name=experiment.data.view.name)
    view = view_type(config=experiment.data.view)
    input_length, output_length, offset_length = experiment.lengths()
    task_type = get(category="tasks", name=experiment.task.name)
    task = task_type(config=experiment.task)
    loaded: list[tuple[int, list[tuple[Tensor, Tensor]]]] = []
    for record in records:
        stored_values, splits = store_type.load(record=record)
        windows = splits[config.phase][: config.max_windows]
        batches = view.batches(
            values=stored_values,
            windows=windows,
            input_length=input_length,
            output_length=output_length,
            offset_length=offset_length,
            batch_size=experiment.data.source.batch_size,
            shuffle=False,
            generator=torch.Generator(),
        )
        # Masks (imputation) start from the task's seed per client, as in scoring.
        mask_generator = torch.Generator().manual_seed(getattr(task.config, "seed", 0))
        prepared = [
            task.prepare(
                batch_x=batch_x,
                batch_y=batch_y,
                output_length=output_length,
                generator=mask_generator,
            )
            for batch_x, batch_y in batches
        ]
        loaded.append((record.client_id, prepared))
    if not loaded or not any(batches for _, batches in loaded):
        raise ValueError(f"{config.phase} phase has no windows")
    model_type = get(category="models", name=experiment.model.name)
    model = model_type(config=experiment.model).cpu()
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("landscape requires a model with trainable parameters")
    model.load_state_dict(
        torch.load(
            run_dir / "checkpoints" / f"{config.checkpoint}.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    loss_type = get(category="metrics", name=experiment.evaluation.loss)
    loss_function = loss_type(config=loss_type.Config())
    generator = torch.Generator().manual_seed(config.seed)
    first = filter_normalized(model=model, generator=generator)
    second = (
        filter_normalized(model=model, generator=generator)
        if config.dimensions == 2
        else None
    )
    directions = (first, second)
    combined = [batch for _, batches in loaded for batch in batches]
    center = mean_loss(model=model, batches=combined, loss_function=loss_function)
    surfaces = {
        f"client_{client_id}": surface(
            model=model,
            batches=batches,
            loss_function=loss_function,
            directions=directions,
            points=config.points,
            span=config.span,
        )
        for client_id, batches in loaded
    }
    pooled = surface(
        model=model,
        batches=combined,
        loss_function=loss_function,
        directions=directions,
        points=config.points,
        span=config.span,
    )
    largest = top_eigenvalue(
        model=model,
        batches=combined,
        loss_function=loss_function,
        iterations=config.iterations,
        generator=generator,
    )
    trace = hessian_trace(
        model=model,
        batches=combined,
        loss_function=loss_function,
        samples=config.samples,
        generator=generator,
    )
    clients = {
        str(client_id): {
            "top_eigenvalue": top_eigenvalue(
                model=model,
                batches=batches,
                loss_function=loss_function,
                iterations=config.iterations,
                generator=generator,
            ),
            "hessian_trace": hessian_trace(
                model=model,
                batches=batches,
                loss_function=loss_function,
                samples=config.samples,
                generator=generator,
            ),
        }
        for client_id, batches in loaded
        if batches
    }
    personal_path = run_dir / "checkpoints" / f"{config.checkpoint}_clients.pt"
    personal = (
        torch.load(personal_path, map_location="cpu", weights_only=True)
        if personal_path.is_file()
        else {}
    )
    interpolations = {
        f"interpolation_client_{client_id}": one_d.interpolation(
            model=model,
            batches=batches,
            loss_function=loss_function,
            end=personal[client_id],
            points=config.points,
        )
        for client_id, batches in loaded
        if batches and client_id in personal
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    axis = torch.linspace(-config.span, config.span, config.points).numpy()
    grid: Any = (
        axis
        if second is None
        else np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    )
    savez: Any = np.savez
    savez(
        analysis_dir / "landscape.npz",
        grid=grid,
        pooled=pooled.numpy(),
        **{key: value.numpy() for key, value in surfaces.items()},
        **{key: value.numpy() for key, value in interpolations.items()},
    )
    payload = {
        "config": asdict(config),
        "center_loss": center,
        "top_eigenvalue": largest,
        "hessian_trace": trace,
        "clients": clients,
        "interpolation": {
            key.removeprefix("interpolation_"): value.tolist()
            for key, value in interpolations.items()
        },
        "phase": config.phase,
        "checkpoint": config.checkpoint,
        "seed": config.seed,
    }
    (analysis_dir / "landscape.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    try:
        if second is None:
            one_d.plot(
                axis=axis, values=pooled.numpy(), path=analysis_dir / "landscape.png"
            )
        else:
            two_d.plot(
                axis=axis, values=pooled.numpy(), path=analysis_dir / "landscape.png"
            )
            three_d.plot(
                axis=axis, values=pooled.numpy(), path=analysis_dir / "landscape_3d.png"
            )
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
    return payload
