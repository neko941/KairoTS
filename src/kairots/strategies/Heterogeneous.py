"""Model-heterogeneous family: clients may train and be scored with their own model
(`client_model`, per client group) while the server shares another one or parts of
it. Partial training (submodels), mutual learning and distillation are its branches.
A leaf that trains a `client_model` declares `CLIENT_MODEL`.
"""

from torch import Tensor

from kairots.strategies.Traditional import Traditional


class Heterogeneous(Traditional):
    ABSTRACT = True

    def uplink_scalars(
        self, *, tensors: dict[str, Tensor], train_samples: int
    ) -> dict[str, float]:
        """Unweighted averaging: no sample count is sent."""
        return {}
