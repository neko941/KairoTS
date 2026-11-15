"""Partition ownership and direct comparisons with pinned original code."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from kairots.data.partitions._labeled import Labeled
from kairots.data.partitions.ClassPerClient import ClassPerClient
from kairots.data.partitions.Dirichlet import Dirichlet
from kairots.data.partitions.ExtendedDirichlet import ExtendedDirichlet
from kairots.data.partitions.Pathological import Pathological
from kairots.data.stores.Parquet import code


def test_dirichlet_matches_author_balancing() -> None:
    path = (
        Path(__file__).parents[2]
        / "tmp/probabilistic-federated-neural-matching/experiment.py"
    )
    if not path.is_file():
        pytest.skip("pinned original PFNM checkout is absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ec384c92b67471833f46db9b8505d22846df3e4738bfdf04b4502b2bf2b6178d"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "partition_data"
    )
    labels = np.arange(1000) % 10
    environment: dict[str, Any] = {
        "np": SimpleNamespace(**{**vars(np), "random": np.random.default_rng(3)}),
        "load_mnist_data": lambda _path: (np.zeros((1000, 1)), labels, None, None),
        "record_net_data_stats": lambda *args: None,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        environment,
    )
    expected = environment["partition_data"]("mnist", "", "", "hetero-dir", 4, 0.5)[4]
    frame = pl.DataFrame(
        {"sample": np.arange(1000), "label": labels, "step": np.zeros(1000, dtype=int)}
    )
    source = SimpleNamespace(name="Toy", clients=[("Toy", frame)])
    actual = Dirichlet(
        config=Dirichlet.Config(num_clients=4, min_samples=10, seed=3)
    ).clients(source=source)
    for client_id, _, client in actual:
        assert sorted(client["sample"].to_list()) == sorted(expected[client_id])


@pytest.mark.parametrize(
    "values",
    [{"num_clients": 0}, {"min_samples": 0}, {"alpha": 0}, {"alpha": float("nan")}],
)
def test_dirichlet_rejects_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Dirichlet.Config(**values)


def test_dirichlet_refuses_impossible_sizes_and_bounds_rejection_sampling() -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {
                        "sample": range(20),
                        "label": [0] * 20,
                        "step": [0] * 20,
                    }
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="too few samples"):
        Dirichlet(config=Dirichlet.Config(num_clients=3, min_samples=10)).clients(
            source=source
        )
    with pytest.raises(ValueError, match="100 draws"):
        Dirichlet(
            config=Dirichlet.Config(num_clients=2, min_samples=10, alpha=0.001)
        ).clients(source=source)


def test_dirichlet_refuses_numerically_empty_balanced_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = SimpleNamespace(
        shuffle=lambda _values: None, dirichlet=lambda _alpha: np.array([1.0, 0.0])
    )
    monkeypatch.setattr(np.random, "default_rng", lambda _seed: generator)
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {"sample": range(6), "label": [0] * 3 + [1] * 3, "step": [0] * 6}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="underflow"):
        Dirichlet(config=Dirichlet.Config(min_samples=1)).clients(source=source)


@pytest.mark.parametrize("seed", [0, 3, 4, 17])
def test_extended_dirichlet_matches_author(seed: int) -> None:
    # Seed4's minimum label coverage9 distinguishes divisor5 from divisor2.
    path = Path(__file__).parents[2] / "tmp/convergence/sim/data/partition.py"
    if not path.is_file():
        pytest.skip("pinned original extended Dirichlet checkout is absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "93834e91005aa1c32067d697c7088c3f234bb98f169f1f2707a285454a48c5c6"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExDirPartitioner"
    )
    environment: dict[str, Any] = {
        "np": SimpleNamespace(**{**vars(np), "random": np.random.default_rng(seed)}),
        "Partitioner": object,
    }
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        environment,
    )
    labels = np.arange(10000) % 10
    expected = environment["ExDirPartitioner"](C=2, alpha=10).partition_data(
        labels, 100, 10
    )
    frame = pl.DataFrame(
        {
            "sample": np.arange(len(labels)),
            "label": labels,
            "step": np.zeros(len(labels), dtype=int),
        }
    )
    source = SimpleNamespace(name="Toy", clients=[("Toy", frame)])
    partition = ExtendedDirichlet(
        config=ExtendedDirichlet.Config(num_clients=100, seed=seed)
    )
    actual = partition.clients(source=source)
    assert len(actual) == 100
    assert sorted(
        sample for _, _, client in actual for sample in client["sample"]
    ) == list(range(len(labels)))
    for client_id, name, client in actual:
        assert name == f"Toy/{client_id}"
        assert sorted(client["sample"].to_list()) == sorted(expected[client_id])
        assert client["label"].n_unique() <= 2
        assert client.height >= 10


@pytest.mark.parametrize(
    "values",
    [
        {"num_clients": 0},
        {"classes_per_client": 0},
        {"alpha": 0},
        {"alpha": float("nan")},
    ],
)
def test_extended_dirichlet_rejects_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExtendedDirichlet.Config(**values)


@pytest.mark.parametrize(
    "values,message",
    [
        ({"classes_per_client": 3}, "cover all labels"),
        ({"num_clients": 1, "classes_per_client": 1}, "cover all labels"),
        ({"num_clients": 3}, "at least 10"),
    ],
)
def test_extended_dirichlet_rejects_impossible_assignments(
    values: dict[str, Any], message: str
) -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {"sample": range(20), "label": [0, 1] * 10, "step": [0] * 20}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match=message):
        ExtendedDirichlet(config=ExtendedDirichlet.Config(**values)).clients(
            source=source
        )


@pytest.mark.parametrize(
    "partition_type", [Dirichlet, ExtendedDirichlet, Pathological, ClassPerClient]
)
@pytest.mark.parametrize(
    "labels,message", [([0, 1], "exactly one label"), ([0, None], "every sample")]
)
def test_labeled_partitions_reject_conflicting_or_missing_labels(
    partition_type: Any, labels: list[int | None], message: str
) -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            ("Toy", pl.DataFrame({"sample": [0, 0], "label": labels, "step": [0, 1]}))
        ],
    )
    with pytest.raises(ValueError, match=message):
        partition_type(config=partition_type.Config()).clients(source=source)


def test_shared_labeled_code_changes_prepared_data_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    path = Path(inspect.getfile(Labeled))
    original = Path.read_bytes
    before = [
        code(types=(kind,))
        for kind in (Dirichlet, ExtendedDirichlet, Pathological, ClassPerClient)
    ]

    def changed(value: Path) -> bytes:  # positional: Path method replacement.
        content = original(value)
        return content + b"\n# changed helper\n" if value == path else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    after = [
        code(types=(kind,))
        for kind in (Dirichlet, ExtendedDirichlet, Pathological, ClassPerClient)
    ]
    assert all(first != second for first, second in zip(before, after, strict=True))


def test_extended_dirichlet_single_client_preserves_all_rows() -> None:
    frame = pl.DataFrame({"sample": range(20), "label": [0, 1] * 10, "step": [0] * 20})
    source = SimpleNamespace(name="Toy", clients=[("Toy", frame)])
    actual = ExtendedDirichlet(config=ExtendedDirichlet.Config(num_clients=1)).clients(
        source=source
    )
    assert actual[0][2].equals(frame)


def test_extended_dirichlet_refuses_empty_balanced_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = SimpleNamespace(
        choice=lambda *args, **kwargs: np.array([0, 1]),
        shuffle=lambda _values: None,
        dirichlet=lambda _alpha: np.zeros(2),
    )
    monkeypatch.setattr(np.random, "default_rng", lambda _seed: generator)
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {"sample": range(20), "label": [0, 1] * 10, "step": [0] * 20}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="proportions vanished"):
        ExtendedDirichlet(config=ExtendedDirichlet.Config()).clients(source=source)


def test_pathological_paper_dimensions_and_shard_ownership() -> None:
    labels = np.tile(np.arange(10), 6000)
    frame = pl.DataFrame(
        {
            "sample": np.arange(60000),
            "label": labels,
            "step": np.zeros(60000, dtype=int),
        }
    )
    source = SimpleNamespace(name="MNIST-shaped", clients=[("MNIST-shaped", frame)])
    partition = Pathological(config=Pathological.Config(num_clients=100, seed=3))
    first = partition.clients(source=source)
    second = partition.clients(source=source)
    sorted_ids = np.argsort(labels, kind="stable")
    shard_of = {sample: position // 300 for position, sample in enumerate(sorted_ids)}
    all_samples = []
    for (client_id, _, client), (_, _, repeated) in zip(first, second, strict=True):
        assert client.height == 600
        assert client["label"].n_unique() <= 2
        assert client.equals(repeated)
        counts: dict[int, int] = {}
        for sample in client["sample"]:
            shard = shard_of[sample]
            counts[shard] = counts.get(shard, 0) + 1
            all_samples.append(sample)
        assert len(counts) == 2
        assert set(counts.values()) == {300}
        assert client_id < 100
    assert sorted(all_samples) == list(range(60000))


@pytest.mark.parametrize("values", [{"num_clients": 0}, {"shards_per_client": 0}])
def test_pathological_rejects_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Pathological.Config(**values)


@pytest.mark.parametrize("size", [0, 3, 5])
def test_pathological_rejects_empty_or_unequal_shards(size: int) -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {
                        "sample": list(range(size)),
                        "label": [0] * size,
                        "step": [0] * size,
                    }
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="equal nonempty"):
        Pathological(config=Pathological.Config()).clients(source=source)


@pytest.mark.parametrize("balance", [False, True])
def test_class_per_client_matches_pinned_pfllib(balance: bool) -> None:
    path = Path(__file__).parents[2] / "tmp/PFLlib/dataset/utils/dataset_utils.py"
    if not path.is_file():
        pytest.skip("pinned original PFLlib checkout is absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "32a474bd9de9110f1a1bb28a6088bbf3c25d76957956ad205a02680cfa0590da"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "separate_data"
    )
    generator = np.random.default_rng(3)
    # Original randint truncates float bounds; Generator requires explicit integers.
    random = SimpleNamespace(
        randint=lambda low, high, size: generator.integers(int(low), int(high), size)
    )
    environment: dict[str, Any] = {
        "np": SimpleNamespace(**{**vars(np), "random": random}),
        "batch_size": 10,
        "train_ratio": 0.75,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        environment,
    )
    labels = np.arange(6000) % 10
    expected = environment["separate_data"](
        (np.arange(6000), labels),
        20,
        10,
        niid=True,
        balance=balance,
        partition="pat",
        class_per_client=2,
    )[0]
    frame = pl.DataFrame(
        {"sample": np.arange(6000), "label": labels, "step": np.zeros(6000, dtype=int)}
    )
    source = SimpleNamespace(name="Toy", clients=[("Toy", frame)])
    partition = ClassPerClient(
        config=ClassPerClient.Config(num_clients=20, balance=balance, seed=3)
    )
    actual = partition.clients(source=source)
    assert sorted(
        sample for _, _, client in actual for sample in client["sample"]
    ) == list(range(6000))
    for client_id, _, client in actual:
        assert sorted(client["sample"].to_list()) == sorted(expected[client_id])
        assert client["label"].n_unique() <= 2


@pytest.mark.parametrize("values", [{"num_clients": 0}, {"classes_per_client": 0}])
def test_class_per_client_rejects_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ClassPerClient.Config(**values)


def test_pfllib_legacy_random_size_bounds_truncate() -> None:
    expected = np.random.RandomState(3).randint(3, 9, 20)
    legacy: Any = np.random.RandomState(3)
    actual = legacy.randint(3.8, 9.2, 20)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "values,size,message",
    [
        ({"classes_per_client": 3}, 20, "cover all labels"),
        ({"num_clients": 1, "classes_per_client": 1}, 20, "cover all labels"),
        ({"num_clients": 20, "balance": True}, 4, "empty client"),
        ({"num_clients": 20}, 4, "random-size interval"),
    ],
)
def test_class_per_client_rejects_impossible_allocations(
    values: dict[str, Any], size: int, message: str
) -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {
                        "sample": range(size),
                        "label": [0, 1] * (size // 2),
                        "step": [0] * size,
                    }
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match=message):
        ClassPerClient(config=ClassPerClient.Config(**values)).clients(source=source)


def test_class_per_client_refuses_to_drop_labels_when_slots_run_out() -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame(
                    {"sample": range(60), "label": [0, 1, 2] * 20, "step": [0] * 60}
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="exhausted slots"):
        ClassPerClient(config=ClassPerClient.Config(balance=True)).clients(
            source=source
        )


@pytest.mark.parametrize(
    "partition_type", [Dirichlet, ExtendedDirichlet, Pathological, ClassPerClient]
)
def test_labeled_partitions_reject_null_sample_ids(partition_type: Any) -> None:
    source = SimpleNamespace(
        name="Toy",
        clients=[
            (
                "Toy",
                pl.DataFrame({"sample": [None, 1], "label": [0, 1], "step": [0, 0]}),
            )
        ],
    )
    with pytest.raises(ValueError, match="sample ID for every row"):
        partition_type(config=partition_type.Config()).clients(source=source)
