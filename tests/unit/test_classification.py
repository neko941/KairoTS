from __future__ import annotations

import polars as pl
import pytest
import torch

from kairots.data.partitions.Dirichlet import Dirichlet
from kairots.data.partitions.IID import IID
from kairots.data.splits.Predefined import Predefined
from kairots.metrics.MacroF1 import MacroF1


def test_predefined_split_is_stratified_and_keeps_test_fold() -> None:
    samples = pl.DataFrame(
        {
            "fold": ["train"] * 20 + ["test"] * 8,
            "label": [index % 4 for index in range(20)] + [0] * 8,
        }
    )
    splits = Predefined(config=Predefined.Config(validation_ratio=0.2)).split(
        samples=samples
    )
    labels = samples.get_column("label").to_list()
    assert sorted(labels[index] for index in splits["validation"]) == [0, 1, 2, 3]
    assert splits["test"] == list(range(20, 28))
    assert not set(splits["train"]) & set(splits["validation"])


def test_iid_deals_every_sample_once() -> None:
    frame = pl.DataFrame({"sample": [0, 0, 1, 1, 2, 2], "step": [0, 1] * 3})

    class Source:
        name = "Toy"
        clients = [("Toy", frame)]

    clients = IID(config=IID.Config(num_clients=2)).clients(source=Source())
    dealt = [frame.get_column("sample").unique().to_list() for _, _, frame in clients]
    assert sorted(sum(dealt, [])) == [0, 1, 2]


def test_dirichlet_deals_labeled_samples_deterministically() -> None:
    frame = pl.DataFrame(
        {
            "sample": [sample for sample in range(120) for _ in range(2)],
            "step": [0, 1] * 120,
            "label": [sample % 4 for sample in range(120) for _ in range(2)],
        }
    )

    class Source:
        name = "Toy"
        clients = [("Toy", frame)]

    def partition(*, alpha: float) -> list[tuple[int, str, pl.DataFrame]]:
        return Dirichlet(
            config=Dirichlet.Config(num_clients=4, alpha=alpha, min_samples=10, seed=3)
        ).clients(source=Source())

    skewed = partition(alpha=0.01)
    repeated = partition(alpha=0.01)
    assert [client_id for client_id, _, _ in skewed] == [0, 1, 2, 3]
    assert [name for _, name, _ in skewed] == [f"Toy/{i}" for i in range(4)]
    assert [
        client_frame.get_column("sample").unique().to_list()
        for _, _, client_frame in skewed
    ] == [
        client_frame.get_column("sample").unique().to_list()
        for _, _, client_frame in repeated
    ]
    dealt = [
        client_frame.get_column("sample").unique().to_list()
        for _, _, client_frame in skewed
    ]
    assert sorted(
        sample for client_samples in dealt for sample in client_samples
    ) == list(range(120))
    assert min(map(len, dealt)) >= 10
    for _, _, client_frame in skewed:
        counts = client_frame.select("sample", "label").unique().group_by("label").len()
        label_counts = [int(count) for count in counts.get_column("len").to_list()]
        assert max(label_counts) / sum(label_counts) >= 0.9

    balanced = partition(alpha=1000)
    for _, _, client_frame in balanced:
        assert set(client_frame.get_column("label")) == {0, 1, 2, 3}

    with pytest.raises(ValueError, match="sample"):
        Dirichlet(config=Dirichlet.Config()).clients(
            source=type(
                "Source",
                (),
                {"name": "Toy", "clients": [("Toy", pl.DataFrame({"label": [0]}))]},
            )()
        )


def test_macro_f1_merges_exactly_and_skips_absent_classes() -> None:
    # Client A sees classes 0 and 1, client B classes 1 and 2; class 3 never occurs.
    scores = torch.eye(4)
    first, second, pooled = MacroF1(), MacroF1(), MacroF1()
    first.update(prediction=scores[[0, 1, 1]], target=torch.tensor([0, 1, 0]))
    second.update(prediction=scores[[2, 1]], target=torch.tensor([2, 2]))
    pooled.update(
        prediction=scores[[0, 1, 1, 2, 1]], target=torch.tensor([0, 1, 0, 2, 2])
    )
    first.merge(other=second)
    # Per class: 0 -> tp 1, fn 1 (F1 2/3); 1 -> tp 1, fp 2 (F1 1/2); 2 -> tp 1, fn 1.
    expected = (2 / 3 + 1 / 2 + 2 / 3) / 3
    assert abs(first.compute() - expected) < 1e-12
    assert abs(pooled.compute() - expected) < 1e-12
