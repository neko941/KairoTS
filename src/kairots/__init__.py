"""KairoTS: modular time-series learning, with a lightweight import surface."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kairots.config.schema import Experiment
    from kairots.runtime.orchestrator import RunResult, evaluate, fit, run

__all__ = ["Experiment", "RunResult", "evaluate", "fit", "run"]
__version__ = "0.0.0"


def __getattr__(name: str) -> Any:  # positional: Python module attribute hook.
    if name == "Experiment":
        from kairots.config.schema import Experiment

        return Experiment
    if name in {"RunResult", "evaluate", "fit", "run"}:
        from kairots.runtime.orchestrator import RunResult, evaluate, fit, run

        return {
            "RunResult": RunResult,
            "evaluate": evaluate,
            "fit": fit,
            "run": run,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
