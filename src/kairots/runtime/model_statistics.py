"""Model size and compute, measured on one batch."""

from __future__ import annotations

from math import prod
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode


def _first_tensor(*, value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(value=item)
            if found is not None:
                return found
    return None


def model_statistics(*, model: nn.Module, batch_x: Tensor) -> dict[str, int]:
    """Parameter, memory and compute counts for one forward pass on `batch_x`.

    `mult_adds_torchinfo`, `forward_backward_bytes` and the byte counts follow
    torchinfo 1.8.0: a layer's mult-adds are its batch size times its parameter
    count (convolutions: times every output position but the channels; recurrent
    inner weights: times batch and steps), and the forward/backward size is twice
    the output size of leaf layers that hold parameters. This undercounts layers
    applied over extra dimensions, such as a Linear on [batch, channels, steps];
    `mult_adds` is PyTorch's FLOP count of the same pass, halved.
    """
    counts = {"mult_adds_torchinfo": 0, "output_bytes": 0}

    def hook(  # positional: torch forward hook.
        module: nn.Module, _: Any, output: Any
    ) -> None:
        tensor = _first_tensor(value=output)
        parameters = dict(module.named_parameters(recurse=False))
        if tensor is None or not parameters:
            return
        size = list(tensor.shape)
        if not any(True for _ in module.children()):
            counts["output_bytes"] += tensor.element_size() * tensor.numel()
        for name, parameter in parameters.items():
            if name in ("weight", "bias"):
                positions = (
                    prod(size[:1] + size[2:])
                    if "Conv" in type(module).__name__
                    else size[0]
                )
            elif "weight" in name or "bias" in name:
                positions = prod(size[:2])
            else:
                continue
            counts["mult_adds_torchinfo"] += positions * parameter.numel()

    handles = [module.register_forward_hook(hook) for module in model.modules()]
    training = model.training
    model.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as flops:
            model(batch_x=batch_x)
    finally:
        model.train(training)
        for handle in handles:
            handle.remove()
    parameters = list(model.parameters())
    input_bytes = batch_x.element_size() * batch_x.numel()
    param_bytes = sum(item.element_size() * item.numel() for item in parameters)
    forward_backward_bytes = 2 * counts["output_bytes"]
    return {
        "total_params": sum(item.numel() for item in parameters),
        "trainable_params": sum(
            item.numel() for item in parameters if item.requires_grad
        ),
        "mult_adds": flops.get_total_flops() // 2,
        "mult_adds_torchinfo": counts["mult_adds_torchinfo"],
        "input_bytes": input_bytes,
        "forward_backward_bytes": forward_backward_bytes,
        "param_bytes": param_bytes,
        "total_bytes": input_bytes + forward_backward_bytes + param_bytes,
    }
