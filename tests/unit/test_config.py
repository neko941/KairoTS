from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kairots.__main__ import experiments, overlay
from kairots.config.schema import Experiment, Reproduces


def test_cli_resolver_namespaces_have_defaults() -> None:
    from kairots.config.schema import _CATEGORIES, _DEFAULTS

    assert set(_CATEGORIES) <= set(_DEFAULTS)


def test_unknown_config_fields_fail() -> None:
    with pytest.raises(TypeError):
        Experiment.from_dict(values={"strategy": {"iteratons": 3}})


def test_ray_gpu_resources_match_explicit_device_policy() -> None:
    with pytest.raises(ValueError, match="needs num_gpus > 0"):
        Experiment.from_dict(values={"runtime": {"name": "Ray", "device": "cuda"}})
    with pytest.raises(ValueError, match="needs num_gpus=0"):
        Experiment.from_dict(
            values={"runtime": {"name": "Ray", "device": "cpu", "num_gpus": 0.5}}
        )
    with pytest.raises(ValueError, match="only supported by Ray"):
        Experiment.from_dict(values={"runtime": {"name": "Serial", "num_gpus": 0.5}})
    accepted = Experiment.from_dict(
        values={"runtime": {"name": "Ray", "device": "auto", "num_gpus": 0.5}}
    )
    assert accepted.runtime.num_gpus == 0.5


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ({"device": "tpu"}, "runtime.device"),
        ({"num_workers": 0}, "runtime.num_workers"),
        ({"num_workers": 1.5}, "runtime.num_workers"),
        ({"num_gpus": -0.5}, "runtime.num_gpus"),
        ({"num_gpus": float("inf")}, "runtime.num_gpus"),
    ],
)
def test_runtime_resource_fields_reject_invalid_values(
    runtime: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Experiment.from_dict(values={"runtime": runtime})


@pytest.mark.parametrize(
    ("old", "new"),
    [("input_len", "input_length"), ("output_len", "output_length")],
)
def test_abbreviated_window_keys_suggest_the_readable_name(old: str, new: str) -> None:
    with pytest.raises(TypeError, match=new):
        Experiment.from_dict(values={"data": {"split": {old: 12}}})


def test_second_source_variant_resolves_without_runtime_changes() -> None:
    experiment = Experiment.from_dict(values={"data": {"source": {"name": "ETTh1"}}})
    assert experiment.data.source.name == "ETTh1"


def test_channel_order_is_stable_across_hash_seeds() -> None:
    script = (
        "from kairots.config.schema import Experiment; "
        "print(','.join(Experiment.from_dict(values={}).data.source.input_channels))"
    )
    outputs = []
    for seed in ("1", "77"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script], env=environment, text=True
            ).strip()
        )
    assert outputs[0] == outputs[1] == "HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"


def test_cli_name_override_resolves_the_new_component_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"strategy": {"name": "FedAvg"}}), encoding="utf-8")
    experiment = overlay(path=str(config), arguments=["--strategy.name", "LocalOnly"])
    assert experiment.strategy.name == "LocalOnly"
    assert not hasattr(experiment.strategy, "join_ratio")
    assert (
        overlay(
            path=str(config), arguments=["--strategy.iterations", "7"]
        ).strategy.iterations
        == 7
    )
    # A field only the newly named component has is typed by that component.
    selected = overlay(
        path=str(config),
        arguments=["--model.num_trees", "5", "--model.name", "XGBoost"]
        + ["--strategy.name", "LocalOnly"],
    )
    assert selected.model.num_trees == 5
    # The file's fields for the old component are dropped, not passed to the new one.
    config.write_text(
        json.dumps({"initialization": {"name": "Pretrained", "run_dir": "runs/x"}}),
        encoding="utf-8",
    )
    native = overlay(path=str(config), arguments=["--initialization.name", "Random"])
    assert native.initialization.name == "Random"
    # A JSON array in the file is overridden by a JSON array on the command line.
    config.write_text(
        json.dumps({"strategy": {"name": "Transfer", "frozen": ["a.*"]}}),
        encoding="utf-8",
    )
    cleared = overlay(path=str(config), arguments=["--strategy.frozen", "[]"])
    assert tuple(cleared.strategy.frozen) == ()


def test_model_that_does_not_support_the_task_fails() -> None:
    with pytest.raises(ValueError, match="Naive"):
        Experiment.from_dict(
            values={
                "task": {"name": "Classification"},
                "data": {"source": {"name": "BasicMotions"}},
                "model": {"name": "Naive"},
            }
        )


def test_augmentation_is_optional_and_round_trips() -> None:
    plain = Experiment.from_dict(values={})
    assert plain.data.augmentation is None
    assert Experiment.from_dict(values=plain.to_dict()).data.augmentation is None
    jittered = Experiment.from_dict(
        values={"data": {"augmentation": {"name": "Jitter", "sigma": 0.1}}}
    )
    assert jittered.data.augmentation.sigma == 0.1
    assert Experiment.from_dict(values=jittered.to_dict()) == jittered


def test_xgboost_rejects_averaging_strategy() -> None:
    with pytest.raises(ValueError, match="LocalOnly.*Centralized"):
        Experiment.from_dict(
            values={"model": {"name": "XGBoost"}, "strategy": {"name": "FedAvg"}}
        )
    experiment = Experiment.from_dict(
        values={"model": {"name": "XGBoost"}, "strategy": {"name": "LocalOnly"}}
    )
    assert experiment.model.name == "XGBoost"


def test_groups_override_data_and_resolve_their_own_model_shapes() -> None:
    experiment = Experiment.from_dict(
        values={
            "data": {"source": {"name": "ETDatasetHour", "batch_size": 8}},
            "model": {"name": "DLinear"},
            "groups": [
                {},
                {"data": {"source": {"input_channels": ["OT"]}}},
                {"data": {"source": {"name": "ETTh1"}}},
            ],
        }
    )
    first, second, third = experiment.groups
    assert first.data.source == experiment.data.source
    assert second.data.source.input_channels == ("OT",)
    assert second.data.source.batch_size == 8
    assert second.model.input_channels == 1
    # Another name replaces the component, so its fields start from their defaults.
    assert (third.data.source.name, third.data.source.batch_size) == ("ETTh1", 32)
    assert experiment.member(group=1) is second
    assert experiment.groups[0].groups == ()


def test_groups_and_client_model_are_checked() -> None:
    with pytest.raises(TypeError, match="only data, client_model, clients"):
        Experiment.from_dict(values={"groups": [{"strategy": {}}]})
    with pytest.raises(ValueError, match="CLIENT_MODEL"):
        Experiment.from_dict(values={"client_model": {"name": "DLinear"}})
    with pytest.raises(ValueError, match="Frozen"):
        Experiment.from_dict(
            values={"groups": [{}], "evaluation": {"name": "Prequential"}}
        )


def test_strategy_constraints_match_a_name_or_a_family_class() -> None:
    from kairots.config.schema import _named
    from kairots.registry import get

    source = get(category="data/sources", name="ETTh1")
    assert _named(component=source, names=("ETTh1",))
    assert _named(component=source, names=("ETDataset", "Panel"))
    assert _named(component=source, names=("Multivariate",))
    assert not _named(component=source, names=("Panel",))


def test_model_shape_fields_must_match_the_data_and_task() -> None:
    with pytest.raises(ValueError, match="model input_length=97"):
        Experiment.from_dict(
            values={
                "data": {"split": {"input_length": 96}},
                "model": {"name": "DLinear", "input_length": 97},
            }
        )
    explicit = Experiment.from_dict(
        values={
            "data": {"split": {"input_length": 96}},
            "model": {"name": "DLinear", "input_length": 96},
        }
    )
    assert explicit.model.input_length == 96


def test_reproduces_block_resolves_into_the_resolved_config() -> None:
    block = {"paper": "arXiv:2205.13504v3"}
    experiment = Experiment.from_dict(values={"reproduces": block})
    assert experiment.to_dict()["reproduces"] == {**block, "code": ""}
    assert Experiment.from_dict(values={}).reproduces is None


@pytest.mark.parametrize(
    ("block", "error"),
    [
        ({"paper": ""}, ValueError),
        ({"paper": "arXiv:1", "table": "Table 1"}, TypeError),
    ],
)
def test_reproduces_block_rejects_incomplete_or_unknown_fields(
    *, block: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Experiment.from_dict(values={"reproduces": block})


def test_paper_configs_name_their_table_and_examples_claim_none() -> None:
    """configs/<paper>/Table<N>-<Model>[-<Dataset>].json reproduce; examples never."""
    root = Path(__file__).parents[2] / "configs"
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        values = json.loads(path.read_text(encoding="utf-8"))
        if relative.parts[0] == "examples":
            assert "reproduces" not in values and "sweep" not in values, relative
            continue
        assert len(relative.parts) == 2, relative
        assert relative.stem.startswith("Table"), relative
        Reproduces(**values["reproduces"])


def test_sweep_zips_fields_and_seeds_expand_in_order(tmp_path: Path) -> None:
    config = tmp_path / "Table2-DLinear-ETTh1.json"
    config.write_text(
        json.dumps(
            {
                "reproduces": {"paper": "arXiv:2205.13504v3"},
                "seeds": [1, 2],
                "sweep": {
                    "data.split.output_length": [96, 192],
                    "strategy.iterations": [3, 7],
                },
            }
        ),
        encoding="utf-8",
    )
    selected = experiments(path=str(config), arguments=["--model.name", "DLinear"])
    assert [label for label, _ in selected] == [
        "96/seed1",
        "96/seed2",
        "192/seed1",
        "192/seed2",
    ]
    assert [item.runtime.seed for _, item in selected] == [1, 2, 1, 2]
    assert [item.strategy.iterations for _, item in selected] == [3, 3, 7, 7]
    ((label, only),) = experiments(path=str(config), arguments=[], run="192", seed=2)
    assert (label, only.data.split.output_length) == ("192/seed2", 192)
    with pytest.raises(ValueError):
        experiments(path=str(config), arguments=[], run="336")
    with pytest.raises(ValueError):
        experiments(path=str(config), arguments=[], seed=3)
    with pytest.raises(ValueError):
        overlay(path=str(config), arguments=[])
    config.write_text(
        json.dumps(
            {
                "model": {"individual": True},
                "sweep": [
                    {"model.name": ["Linear", "DLinear"]},
                    {"data.split.output_length": [96, 192]},
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = experiments(path=str(config), arguments=[])
    assert [label for label, _ in selected] == [
        "Linear-96",
        "Linear-192",
        "DLinear-96",
        "DLinear-192",
    ]
    assert selected[2][1].model.name == "DLinear"
    # An unnamed base model block is shared by every model the sweep selects.
    assert all(item.model.individual for _, item in selected)
    config.write_text(
        json.dumps({"sweep": {"a.b": [1, 2], "c.d": [1]}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        experiments(path=str(config), arguments=[])
