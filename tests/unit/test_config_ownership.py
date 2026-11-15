"""Shared knobs have a shared declaration, even when a profile overrides a default."""

import ast
import shutil
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
from fixtures.config_fields import config_field_homes

_SHARED_HOMES = {
    "beta_start": {"kairots.models.base.Diffusion.Config"},
    "beta_end": {"kairots.models.base.Diffusion.Config"},
    "diffusion_steps": {"kairots.models.base.Diffusion.Config"},
    "samples": {"kairots.models.base.ConditionalDiffusion.Config"},
    "replay": {"kairots.replay.base.Replay.Config"},
    "replay_size": {"kairots.replay.base.Replay.Config"},
    "optimizer": {"kairots.learners.base.Learning.Config"},
    "learning_rate": {"kairots.learners.base.Learning.Config"},
    "epochs": {"kairots.learners.base.Learner.Config"},
    "weight_decay": {"kairots.optimizers.base.Optimizer.Config"},
    "device": {"kairots.runtime.executors.base.Executor.Config"},
    "num_workers": {"kairots.runtime.executors.base.Executor.Config"},
    "num_gpus": {"kairots.runtime.executors.base.Executor.Config"},
    "deterministic": {"kairots.runtime.executors.base.Executor.Config"},
    "runs_dir": {"kairots.runtime.executors.base.Executor.Config"},
    "save_checkpoints": {"kairots.runtime.executors.base.Executor.Config"},
    "resume": {"kairots.runtime.executors.base.Executor.Config"},
    "loss": {"kairots.evaluation.base.Scoring.Config"},
    "metrics": {"kairots.evaluation.base.Scoring.Config"},
    "iterations": {"kairots.strategies.base.Strategy.Config"},
    "mu": {"kairots.strategies.Proximal.Proximal.Config"},
    "steps": {"kairots.optimizers.base.UpdateBudget.Config"},
    "calendar": {"kairots.tasks.base.Timed.Config"},
    "num_clients": {"kairots.data.partitions._labeled.Labeled.Config"},
    "classes_per_client": {"kairots.data.partitions._labeled.Coverage.Config"},
    "offset_length": {"kairots.data.splits.base.Offset.Config"},
    "min_windows": {"kairots.data.splits.base.Minimum.Config"},
    "train_stride": {"kairots.data.splits.base.Strided.Config"},
    "width": {"kairots.models.base.FeatureWidth.Config"},
    "layers": {"kairots.models.base.ResidualStack.Config"},
    "hidden": {"kairots.models.base.HiddenWidth.Config"},
    "member": {"kairots.data.sources.base.SingleMember.Config"},
    "raw_path": {"kairots.data.sources.base.Source.Config"},
    "seasonality": {"kairots.data.sources.base.Source.Config"},
    "inverse": {"kairots.evaluation.Frozen.Frozen.Config"},
}

# Only derived model fields have a category-wide exception. Other owners are exact.
_REVIEWED_HOMES = {
    "seed": {
        "kairots.data.partitions._labeled.Labeled.Config": "partition assignment RNG",
        "kairots.data.splits.Predefined.Predefined.Config": "stratified fold RNG",
        "kairots.models.CSDI.CSDI.Config": "imputation target-pattern RNG",
        "kairots.models.TimeGrad.TimeGrad.Config": "forecast sampling RNG",
        "kairots.models.XGBoost.XGBoost.Config": "native booster RNG",
        "kairots.replay.Reservoir.Reservoir.Config": "buffer admission/draw RNG",
        "kairots.runtime.executors.base.Executor.Config": "job/model seed stream",
        "kairots.tasks.Generation.Generation.Config": "synthesis evaluation RNG",
        "kairots.tasks.Imputation.Imputation.Config": "artificial missingness RNG",
    },
    "alpha": {
        "kairots.data.partitions._labeled.Concentrated.Config": (
            "Dirichlet concentration"
        ),
        "kairots.strategies.APFL.APFL.Config": "personal model mixing coefficient",
        "kairots.strategies.FLwF2.FLwF2.Config": "supervised loss coefficient",
        "kairots.strategies.FML.FML.Config": "private mutual-learning weight",
    },
    "beta": {
        "kairots.strategies.FLwF2.FLwF2.Config": "previous-client teacher weight",
        "kairots.strategies.FML.FML.Config": "shared mutual-learning weight",
    },
    "capacity": {
        "kairots.replay.base.Buffer.Config": "buffer window slots",
        "kairots.environments.Battery.Battery.Config": "physical battery energy",
    },
    "input_channels": {
        "kairots.data.sources.base.Source.Config": "source channel names/order",
        "kairots.tasks.Forecasting.Forecasting.Config": "channel-role selection",
        "kairots.tasks.ProbabilisticImputation.ProbabilisticImputation.Config": (
            "value/mask channel split"
        ),
        "kairots.tasks.MaskedImputation.MaskedImputation.Config": (
            "value/mask channel split"
        ),
        "kairots.models.*": "model width filled and checked from data/task",
    },
    "input_length": {
        "kairots.data.sources.Panel.Panel.Config": "official panel sequence length",
        "kairots.data.splits.base.Window.Config": "forecast history window",
        "kairots.data.splits.CSDIFiveFold.CSDIFiveFold.Config": (
            "PhysioNet CSDI imputation sequence length"
        ),
        "kairots.models.*": "model history dimension derived from data/task",
    },
    "output_length": {
        "kairots.data.splits.base.Window.Config": "forecast target window",
        "kairots.data.splits.CSDIFiveFold.CSDIFiveFold.Config": (
            "PhysioNet CSDI imputation target length"
        ),
        "kairots.models.*": "model target dimension derived from data/task",
    },
    "num_classes": {
        "kairots.data.sources.Panel.Panel.Config": "source label cardinality",
        "kairots.models.*": "classifier size derived from source labels",
    },
    "num_quantiles": {
        "kairots.tasks.ProbabilisticForecasting.ProbabilisticForecasting.Config": (
            "quantile axis"
        ),
        "kairots.models.*": "output axis derived from task",
    },
    "target_pattern": {
        "kairots.tasks.ProbabilisticImputation.ProbabilisticImputation.Config": (
            "task conditioning pattern"
        ),
        "kairots.models.*": "conditioning pattern derived and checked from task",
    },
    "head": {"kairots.models.*": "head derived and checked from task HEAD"},
    "reconstruct": {"kairots.models.*": "derived from task RECONSTRUCTS"},
    "output_channels": {"kairots.models.*": "derived from target channel selection"},
    "calendar_channels": {"kairots.models.*": "derived from calendar selection"},
    "moving_average_kernel": {
        "kairots.models.DLinear.DLinear.Config": "public decomposition setting",
        "kairots.models.DLinear.Body.Body.Config": "private recipe adapter",
    },
    "individual": {
        "kairots.models.DLinear.DLinear.Config": "public channel-sharing choice",
        "kairots.models.DLinear.Body.Body.Config": "private recipe adapter",
    },
    "d_model": {
        "kairots.models.base.EmbeddingDimensions.Config": "public embedding dimension",
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
    "d_ff": {
        "kairots.models.base.EmbeddingDimensions.Config": (
            "public intermediate dimension"
        ),
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
    "encoder_layers": {
        "kairots.models.base.EncoderDepth.Config": "public encoder block count",
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
    "k": {
        "kairots.models.TimesNet.TimesNet.Config": "public frequency count",
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
    "num_inception_kernels": {
        "kairots.models.TimesNet.TimesNet.Config": "public inception branch count",
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
    "dropout": {
        "kairots.models.base.Dropout.Config": "public dropout probability",
        "kairots.models.TimesNet.Body.Body.Config": "private recipe adapter",
    },
}


def _check_home(*, field: str, homes: set[str]) -> None:
    assert homes == _SHARED_HOMES[field], (field, sorted(homes))


def _check_collision(*, field: str, homes: set[str]) -> None:
    assert field in _REVIEWED_HOMES, (field, sorted(homes))
    for owner in homes:
        assert any(fnmatchcase(owner, pattern) for pattern in _REVIEWED_HOMES[field]), (
            field,
            owner,
        )


def test_shared_config_fields_have_one_home() -> None:
    root = Path(__file__).parents[2] / "src" / "kairots"
    homes = config_field_homes(root=root)
    for field in _SHARED_HOMES:
        _check_home(field=field, homes=homes[field])
    for field, owners in homes.items():
        if len(owners) > 1:
            _check_collision(field=field, homes=owners)
    tree = ast.parse((root / "config" / "schema.py").read_text())
    derived = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "derived"
            for target in node.targets
        )
    ]
    assert len(derived) == 1 and isinstance(derived[0], ast.Dict)
    derived_fields = {
        key.value for key in derived[0].keys if isinstance(key, ast.Constant)
    }
    wildcard_fields = set()
    for field, reviewed in _REVIEWED_HOMES.items():
        assert field in homes, field
        for pattern, reason in reviewed.items():
            assert reason.strip(), (field, pattern)
            assert any(fnmatchcase(owner, pattern) for owner in homes[field]), (
                field,
                pattern,
            )
            if "*" in pattern:
                assert pattern == "kairots.models.*" and field in derived_fields, (
                    field,
                    pattern,
                )
                wildcard_fields.add(field)
    assert wildcard_fields == derived_fields


def test_new_strategy_cannot_duplicate_an_existing_shared_knob(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "src" / "kairots"
    root = tmp_path / "kairots"
    (root / "learners").mkdir(parents=True)
    shutil.copy2(source / "learners" / "base.py", root / "learners" / "base.py")
    (root / "strategies").mkdir()
    (root / "strategies" / "Example.py").write_text(
        "class Example:\n    class Config:\n        learning_rate: float = 0.5\n"
    )
    with pytest.raises(AssertionError, match="learning_rate"):
        _check_home(
            field="learning_rate", homes=config_field_homes(root=root)["learning_rate"]
        )


@pytest.mark.parametrize(
    ("field", "owner"),
    [
        ("alpha", "kairots.strategies.Example.Example.Config"),
        ("input_length", "kairots.strategies.Example.Example.Config"),
        ("unreviewed", "kairots.models.Example.Example.Config"),
    ],
)
def test_collision_exceptions_reject_unreviewed_owners(field: str, owner: str) -> None:
    with pytest.raises(AssertionError, match=field):
        _check_collision(field=field, homes={owner})


def test_derived_shape_exception_accepts_a_new_model() -> None:
    _check_collision(
        field="input_length", homes={"kairots.models.Example.Example.Config"}
    )


def test_inherited_config_lookup_follows_python_class_order(tmp_path: Path) -> None:
    root = tmp_path / "kairots"
    root.mkdir()
    (root / "example.py").write_text(
        "class Root:\n    class Config:\n        root_value: float = 1.0\n"
        "class Left(Root): pass\n"
        "class Right(Root):\n    class Config:\n        right_value: float = 2.0\n"
        "class Child(Left, Right): pass\n"
        "class Recipe:\n    class Config(Child.Config):\n"
        "        right_value: float = 3.0\n"
    )
    assert config_field_homes(root=root)["right_value"] == {
        "kairots.example.Right.Config"
    }
