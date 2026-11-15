"""Pretrained: start from a saved run's checkpoint (pretraining, then fine-tuning).

Every parameter or buffer whose name and shape match the checkpoint takes its
value; the rest (a new head, a length-dependent projection) keep native values.
`run_dir` is one run, or a run group whose latest run holding the checkpoint is
taken.

The pretraining run must not have seen this run's held-out period: temporal clients
compare first unseen timestamps, while panel evaluation compares source training
and target query sample ranges. For timestamped clients, the source's first unseen
timestamp (validation, else test; test for a `best_val` checkpoint, which validation
selected) may not be later than that of the same-named client here or, for a client
of another population (a transfer source of the same dataset family), than this
run's earliest one.
Known different families skip only that cross-population comparison; same-named
clients and unknown or composite families retain the check. The manifest records
the source run and whether the comparison was skipped. Series sharing a calendar can
carry information about each other's later values, so a source is held to the
target's time cut too.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kairots.initializers.base import Policy


def _held_out(*, client: dict[str, Any], selected: bool = False) -> Any:
    """The client's first timestamp training never saw; with `selected`, the first
    one model selection never saw either."""
    cuts = client["cut_timestamps"]
    if selected or cuts["validation"] is None:
        return cuts["test"]
    return cuts["validation"]


class Pretrained(Policy):
    @dataclass(kw_only=True)
    class Config(Policy.Config):
        name: str = "Pretrained"
        run_dir: str = ""
        # checkpoints/<checkpoint>.pt: final or best_val.
        checkpoint: str = "final"

    def __init__(self, *, config: Config) -> None:
        self.config: Pretrained.Config = config

    def _run(self) -> Path:
        if not self.config.run_dir:
            raise ValueError("Pretrained needs initialization.run_dir")
        path = Path(self.config.run_dir)
        if (path / "checkpoints").is_dir():
            return path
        checkpoint = f"checkpoints/{self.config.checkpoint}.pt"
        runs = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and (child / checkpoint).is_file()
        )
        if not runs:
            raise FileNotFoundError(f"no run under {path} holds {checkpoint}")
        return runs[-1]

    def apply(
        self, *, model: nn.Module, manifest: dict[str, Any] | None = None
    ) -> dict[str, str]:
        if manifest is None:
            raise ValueError("Pretrained needs this run's manifest")
        run_dir = self._run()
        source = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        own_cut = (
            "test"
            if manifest.get("data_permissions", {}).get("mode") == "evaluate"
            else None
        )
        clients = manifest["clients"]
        source_clients = source["clients"]
        same_family = (
            source.get("dataset_family") is None
            or manifest.get("dataset_family") is None
            or source["dataset_family"] == manifest["dataset_family"]
        )
        panel_clients: set[str] = set()
        own: dict[str, Any] = {}
        if own_cut is None:
            own = {client["name"]: _held_out(client=client) for client in clients}
        else:
            source_by_name = {client["name"]: client for client in source_clients}
            for client in clients:
                name = client["name"]
                cuts = client.get("cut_timestamps")
                if cuts is not None:
                    own[name] = cuts[own_cut]
                    continue
                if not same_family:
                    continue
                source_client = source_by_name.get(name)
                source_windows = (source_client or {}).get("windows", {})
                test_windows = client.get("windows", {}).get("test", ())
                train_windows = source_windows.get("train", ())
                if (
                    source.get("data_permissions", {}).get("mode") != "fit"
                    or source_client is None
                    or not train_windows
                    or not test_windows
                ):
                    raise ValueError(
                        f"cannot verify held-out samples for panel client {name!r}"
                    )
                if any(
                    max(train_first, test_first) <= min(train_last, test_last)
                    for train_first, train_last in train_windows
                    for test_first, test_last in test_windows
                ):
                    raise ValueError(
                        f"pretraining samples overlap evaluation samples for panel "
                        f"client {name!r}"
                    )
                panel_clients.add(name)
        earliest = min(own.values()) if own else None
        check = "sample-disjoint" if panel_clients else "enforced"
        reason = "fit and evaluation sample ranges are disjoint"
        if not panel_clients:
            reason = "shared or unknown dataset family"
        if not same_family:
            check, reason = "skipped", "different dataset families"
        manifest["pretraining_calendar_check"] = {
            "cross_population": check,
            "reason": reason,
            "source_run": str(run_dir),
        }
        for client in source_clients:
            if client["name"] in panel_clients:
                continue
            if client["name"] not in own and not same_family:
                continue
            seen = _held_out(
                client=client, selected=self.config.checkpoint == "best_val"
            )
            held_out = own.get(client["name"], earliest)
            if held_out is None:
                raise ValueError(
                    f"cannot verify held-out timestamps for {client['name']!r}"
                )
            if seen is None:
                raise ValueError(
                    f"pretraining on {client['name']!r} held out nothing, so it may "
                    "have seen this run's held-out period"
                )
            if type(seen) is not type(held_out):
                raise ValueError(
                    f"pretraining timestamps of {client['name']!r} ({seen!r}) are not "
                    f"comparable with this run's ({held_out!r})"
                )
            if seen > held_out:
                raise ValueError(
                    f"pretraining on {client['name']!r} ran up to {seen}, past this "
                    f"run's first held-out timestamp {held_out}"
                )
        state = torch.load(
            run_dir / "checkpoints" / f"{self.config.checkpoint}.pt",
            map_location="cpu",
            weights_only=True,
        )
        own_state = model.state_dict()
        loaded = {
            name: value
            for name, value in state.items()
            if name in own_state and own_state[name].shape == value.shape
        }
        if not loaded:
            raise ValueError(f"{run_dir} shares no parameter with this model")
        model.load_state_dict(loaded, strict=False)
        return {name: "pretrained" for name in loaded}
