"""Loss and phase scoring selectors shared by evaluation protocols."""

from dataclasses import dataclass, field


class Scoring:
    @dataclass(kw_only=True)
    class Config:
        name: str = "Scoring"
        loss: str = "MSE"
        # Rounds without a better validation score before training stops; 0 never stops.
        patience: int = 0
        # Metric whose best value picks the checkpoint; empty selects by `loss`. A
        # named metric also keeps the later of equal scores, as TSLib's EarlyStopping.
        selection_metric: str = ""
        # The historical benchmark protocol: pick the checkpoint on the TEST fold
        # instead of validation. The reported test score is then optimistically
        # biased and is no evidence of generalization. Needs a split without
        # validation; never enable it for a result presented as clean.
        leaky_test_selection: bool = False
        metrics: dict[str, tuple[str, ...]] = field(
            default_factory=lambda: {"validation": ("MAE",), "test": ("MAE",)}
        )
