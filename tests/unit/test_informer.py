"""Paper operations and the native four-input forecasting adapter."""

import math
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from kairots.config.schema import Experiment
from kairots.models.Informer import Informer
from kairots.models.Informer.Attention import Attention
from kairots.models.Informer.DataEmbedding import DataEmbedding
from kairots.models.Informer.DecoderLayer import DecoderLayer
from kairots.registry import require
from kairots.tasks.Forecasting import Forecasting

CALENDAR = ("MonthOfYear", "DayOfMonth", "DayOfWeek", "HourOfDay")


def small_model() -> Informer:
    torch.manual_seed(941)
    return Informer(
        config=Informer.Config(
            input_length=12,
            output_length=4,
            label_length=4,
            input_channels=3,
            output_channels=2,
            d_model=8,
            d_ff=16,
            encoder_heads=2,
            decoder_heads=2,
            sampling_factor=1,
            dropout=0,
        )
    )


def prepared_batch() -> tuple[Forecasting, Tensor, Tensor]:
    task = Forecasting(
        config=Forecasting.Config(
            input_channels=("one", "two", "past"),
            target_channels=("one", "two"),
            calendar=CALENDAR,
        )
    )
    generator = torch.Generator().manual_seed(123)
    history = torch.randn(2, 12, 7, generator=generator)
    future = torch.randn(2, 4, 7, generator=generator)
    # Valid, distinct calendar categories; values remain unmodified.
    history[..., 3:] = -0.5
    future[..., 3:] = -0.5
    return task, history, future


def predict(*, model: Informer, packed: Tensor) -> Tensor:
    torch.manual_seed(71)
    with torch.no_grad():
        return model(batch_x=packed)


def test_informer_future_values() -> None:
    model = small_model().eval()
    task, history, future = prepared_batch()
    packed, target = task.prepare(batch_x=history, batch_y=future, output_length=4)
    poisoned = future.clone()
    poisoned[..., :3] = 100000
    packed_poisoned, poisoned_target = task.prepare(
        batch_x=history, batch_y=poisoned, output_length=4
    )
    assert packed[:, 12:, :3].isnan().all()
    assert packed_poisoned[:, 12:, :3].isnan().all()
    assert not torch.equal(target, poisoned_target)
    prediction = predict(model=model, packed=packed)
    assert prediction.shape == (2, 4, 2)
    assert prediction.isfinite().all()
    assert torch.equal(prediction, predict(model=model, packed=packed_poisoned))
    # The model itself also ignores values in horizon slots, even if supplied.
    packed_poisoned[:, 12:, :3] = -12345
    assert torch.equal(prediction, predict(model=model, packed=packed_poisoned))


def test_informer_future_calendar() -> None:
    model = small_model().eval()
    task, history, future = prepared_batch()
    packed, _ = task.prepare(batch_x=history, batch_y=future, output_length=4)
    changed = packed.clone()
    changed[:, 12:, -1] = 0.5
    assert not torch.equal(
        predict(model=model, packed=packed), predict(model=model, packed=changed)
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("batch,heads,length", [(2, 2, 3), (1, 1, 1)])
def test_informer_dense_oracle(
    *, sparse: bool, batch: int, heads: int, length: int
) -> None:
    torch.manual_seed(941)
    attention = Attention(
        d_model=8, heads=heads, sparse=sparse, sampling_factor=100, dropout=0
    ).double()
    features = torch.randn(batch, length, 8, dtype=torch.float64)
    projected = [
        projection(features).reshape(batch, length, heads, 8 // heads).transpose(1, 2)
        for projection in (
            attention.query_projection,
            attention.key_projection,
            attention.value_projection,
        )
    ]
    queries, keys, values = projected
    weights = (queries @ keys.transpose(-2, -1) / math.sqrt(8 // heads)).softmax(-1)
    expected = attention.output_projection(
        (weights @ values).transpose(1, 2).reshape(batch, length, 8)
    )
    actual = attention(queries=features, keys=features, values=features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("causal", [False, True])
def test_informer_sparse_attention(*, causal: bool) -> None:
    torch.manual_seed(13)
    queries = -torch.rand(2, 2, 8, 4, dtype=torch.float64)
    keys = torch.rand_like(queries)
    values = torch.randn_like(queries)
    attention = Attention(
        d_model=8, heads=2, sparse=True, causal=causal, sampling_factor=1, dropout=0
    ).double()
    # Independent loop oracle of Eq4's zero-filled score and Algorithm1 contexts.
    torch.manual_seed(31)
    sampled = torch.randint(8, (8, 3))
    expected = (
        values.cumsum(2)
        if causal
        else values.mean(2, keepdim=True).expand_as(values).clone()
    )
    for batch in range(2):
        for head in range(2):
            scores = (queries[batch, head, :, None] * keys[batch, head, sampled]).sum(
                -1
            )
            # All sampled dot products are negative; unsampled pairs are zero.
            selected = (0 - scores.sum(-1) / 8).topk(3, sorted=False).indices
            for query in selected.tolist():
                logits = keys[batch, head] @ queries[batch, head, query] / 2
                if causal:
                    logits[query + 1 :] = -torch.inf
                expected[batch, head, query] = logits.softmax(-1) @ values[batch, head]
    torch.manual_seed(31)
    actual = attention.sparse_context(queries=queries, keys=keys, values=values)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_informer_encoder_stacks() -> None:
    model = small_model().eval()
    captured: dict[str, tuple[int, ...]] = {}

    def record(*, name: str) -> Any:
        def hook(module: Any, args: Any, kwargs: Any) -> None:
            captured[name] = tuple(kwargs["features"].shape)

        return hook

    handles = [
        model.encoder.replica.register_forward_pre_hook(
            record(name="replica"), with_kwargs=True
        )
    ]
    handles += [
        layer.register_forward_pre_hook(record(name=f"layer{index}"), with_kwargs=True)
        for index, layer in enumerate(model.encoder.main)
    ]
    with torch.no_grad():
        encoded = model.encoder(features=torch.randn(2, 12, 8))
    for handle in handles:
        handle.remove()
    assert captured == {
        "layer0": (2, 12, 8),
        "layer1": (2, 6, 8),
        "layer2": (2, 3, 8),
        "replica": (2, 3, 8),
    }
    assert encoded.shape == (2, 6, 8)
    assert model.encoder.main[0] is not model.encoder.replica
    assert model.decoder[0] is not model.decoder[1]
    decoder_layer = model.decoder[0]
    assert isinstance(decoder_layer, DecoderLayer)
    assert not decoder_layer.cross_attention.sparse


def test_informer_embeddings() -> None:
    model = small_model().eval()
    embedding = model.encoder_embedding
    assert embedding.scalar.kernel_size == (3,)
    assert embedding.scalar.bias is not None
    assert [table.num_embeddings for table in embedding.calendar] == [13, 32, 7, 24]
    assert all(table.weight.requires_grad for table in embedding.calendar)
    positions = embedding.position(features=torch.zeros(1, 12, 3))
    for position in range(12):
        for pair in range(4):
            angle = position / 10000 ** (2 * pair / 8)
            assert positions[0, position, 2 * pair].item() == pytest.approx(
                math.sin(angle), abs=1e-6
            )
            assert positions[0, position, 2 * pair + 1].item() == pytest.approx(
                math.cos(angle), abs=1e-6
            )
    features = torch.zeros(1, 2, 3)
    marks = torch.tensor([[[-0.5] * 4, [0.5] * 4]])
    expected = embedding.scalar(features.transpose(1, 2)).transpose(1, 2)
    expected = expected + embedding.position(features=features)
    for table, categories in zip(
        embedding.calendar, ([1, 12], [1, 31], [0, 6], [0, 23]), strict=True
    ):
        expected = expected + table(torch.tensor([categories]))
    torch.testing.assert_close(embedding(features=features, marks=marks), expected)
    assert (
        model.encoder_embedding.calendar[0].weight
        is not model.decoder_embedding.calendar[0].weight
    )
    minute_embedding = DataEmbedding(
        input_channels=3, calendar_channels=5, d_model=8, max_length=4, dropout=0
    )
    minute_marks = torch.full((1, 4, 5), -0.5)
    minute_marks[0, :, 4] = torch.tensor([0, 15, 30, 45]) / 59 - 0.5
    zeros = torch.zeros(1, 4, 3)
    baseline = minute_marks.clone()
    baseline[..., 4] = -0.5
    delta = minute_embedding(features=zeros, marks=minute_marks) - minute_embedding(
        features=zeros, marks=baseline
    )
    minute_table = minute_embedding.calendar[-1]
    expected_delta = minute_table(torch.arange(4)) - minute_table(
        torch.zeros(4, dtype=torch.long)
    )
    torch.testing.assert_close(delta[0], expected_delta, atol=1e-6, rtol=1e-6)


def test_informer_gradients() -> None:
    model = small_model()
    task, history, future = prepared_batch()
    packed, target = task.prepare(batch_x=history, batch_y=future, output_length=4)
    loss = (model(batch_x=packed) - target).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name


def test_informer_paper_defaults() -> None:
    config = Informer.Config()
    assert (config.d_model, config.d_ff) == (512, 2048)
    assert (config.encoder_heads, config.decoder_heads) == (16, 8)
    assert (config.dropout, config.sampling_factor) == (0.1, 5)
    for changes in (
        {"label_length": 96},
        {"input_length": 3},
        {"d_model": 7},
        {"sampling_factor": 0},
        {"output_length": 0},
        {"calendar_channels": 0},
        {"dropout": 1},
    ):
        with pytest.raises(ValueError, match="Informer"):
            replace(config, **changes)


def test_informer_applicability() -> None:
    base: dict[str, Any] = {
        "model": {"name": "Informer"},
        "task": {"name": "Forecasting", "calendar": CALENDAR},
    }
    experiment = Experiment.from_dict(values=base)
    assert experiment.model.calendar_channels == 4
    with pytest.raises(ValueError, match="calendar"):
        Experiment.from_dict(values={"model": {"name": "Informer"}})
    with pytest.raises(ValueError, match="calendar"):
        Experiment.from_dict(
            values={**base, "task": {"name": "Forecasting", "calendar": CALENDAR[::-1]}}
        )
    with pytest.raises(ValueError, match="Informer"):
        Experiment.from_dict(values={**base, "task": {"name": "Classification"}})
    with pytest.raises(ValueError, match="Forecasting"):
        require(category="models", name="Informer", task="Classification")
