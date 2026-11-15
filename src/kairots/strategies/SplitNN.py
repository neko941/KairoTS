"""SplitNN vertical federated training (Vepakomma et al., 2018), with the
`SplitNN` model.

Parties share one timeline and own different channels, so there is one client
(the aligned table) and synchronous split learning on it equals end-to-end
training of the block-structured model, which this strategy runs. Each training
step, every feature party sends its cut-layer activations to the label party and
receives their gradients; both are billed as the round's uplink. Parameters never
leave their owner, and evaluation-time activations are not billed. The simulation
holds every party's data in one process, so it makes no privacy claim.
"""

from dataclasses import dataclass
from typing import Any

from kairots.runtime.contracts import Message
from kairots.strategies.Centralized import Centralized

FLOAT_BYTES = 4


class SplitNN(Centralized):
    MODELS = ("SplitNN",)

    @dataclass(kw_only=True)
    class Config(Centralized.Config):
        name: str = "SplitNN"

    def client_update(self, *, worker: Any, job: Any) -> Any:
        result = super().client_update(worker=worker, job=job)
        model = worker.experiment.model
        # Activations up and gradients down, per window, epoch and feature party.
        cut_bytes = (
            2
            * worker.experiment.learner.epochs
            * result.train_samples
            * len(model.party_channels)
            * model.embedding_dimension
            * FLOAT_BYTES
        )
        result.uplink = Message(
            sender="feature_parties",
            receiver="label_party",
            tensors=result.uplink.tensors,
            encoded_bytes=cut_bytes,
        )
        return result
