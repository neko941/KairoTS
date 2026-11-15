"""Quantized-uplink family: each client uploads its whole update
`w_local - w_start` as one flat vector through a lossy codec, bills the encoded size,
and the server adds the unweighted mean of the decoded updates to the global model.
Leaves define the codec (`quantize`, `encoded_bytes`).
"""

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from kairots.runtime.contracts import Job, Message, Result
from kairots.strategies.Traditional import Traditional


class Quantized(Traditional):
    ABSTRACT = True

    def quantize(self, *, vector: Tensor, generator: torch.Generator) -> Tensor:
        """Return the decoded values the server receives for `vector`."""
        raise NotImplementedError

    def encoded_bytes(self, *, numel: int) -> int | None:
        """Wire size of `numel` encoded values; None bills raw tensors."""
        raise NotImplementedError

    def client_update(self, *, worker: Any, job: Job) -> Result:
        result: Result = worker.update(job=job, client=self.Client())
        start = job.downlink.tensors
        # Every state entry is treated as a float update; integer buffers
        # (batch-norm counters) need their own rule when such a model arrives.
        updates = {
            name: value.float() - start[name].float()
            for name, value in result.uplink.tensors.items()
        }
        flat = torch.cat([update.reshape(-1) for update in updates.values()])
        decoded = self.quantize(
            vector=flat, generator=torch.Generator().manual_seed(job.seed)
        )
        tensors = {}
        offset = 0
        for name, update in updates.items():
            tensors[name] = decoded[offset : offset + update.numel()].view_as(update)
            offset += update.numel()
        uplink = result.uplink
        return replace(
            result,
            uplink=Message(
                sender=uplink.sender,
                receiver=uplink.receiver,
                tensors=tensors,
                encoded_bytes=self.encoded_bytes(numel=flat.numel()),
            ),
        )

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """Unweighted averaging: no sample count is sent."""
        return {}

    def aggregate(
        self, *, uplinks: list[Message], global_params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Global plus the unweighted mean update."""
        return {
            name: value
            + torch.stack([message.tensors[name] for message in uplinks])
            .mean(dim=0)
            .to(value)
            for name, value in global_params.items()
        }
