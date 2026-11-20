import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from kairots.models.TimeKAN import TimeKAN
from kairots.models.TimeKAN.ChebyshevKAN import ChebyshevKAN
from kairots.models.TimeKAN.FrequencyDecomposition import FrequencyDecomposition
from kairots.models.TimeKAN.FrequencyUpsampling import FrequencyUpsampling

REFERENCE = Path(__file__).resolve().parents[2] / "tmp" / "TimeKAN"


def _reference_model() -> type:
    if not (REFERENCE / "models" / "TimeKAN.py").is_file():
        pytest.skip("huangst21/TimeKAN reference checkout is missing")
    for name in [
        name for name in sys.modules if name.split(".")[0] in ("models", "layers")
    ]:
        del sys.modules[name]
    sys.path.insert(0, str(REFERENCE))
    importlib.invalidate_caches()
    try:
        return importlib.import_module("models.TimeKAN").Model
    finally:
        sys.path.remove(str(REFERENCE))


def _official_config(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = dict(
        task_name="long_term_forecast",
        seq_len=16,
        label_len=0,
        pred_len=6,
        down_sampling_window=2,
        down_sampling_layers=2,
        channel_independence=1,
        e_layers=2,
        moving_avg=25,
        enc_in=3,
        c_out=3,
        use_future_temporal_feature=0,
        d_model=8,
        embed="timeF",
        freq="h",
        dropout=0.0,
        use_norm=1,
        begin_order=1,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _copy_official(*, ours: TimeKAN, theirs: Any) -> None:
    """Load the official model's weights into ours (code mode has the same parts)."""
    with torch.no_grad():
        for mine, official in zip(
            ours.normalizations, theirs.normalize_layers, strict=True
        ):
            cast(Any, mine.weight).copy_(official.affine_weight)
            cast(Any, mine.bias).copy_(official.affine_bias)
        cast(Any, ours.embedding.layer).weight.copy_(
            theirs.enc_embedding.value_embedding.tokenConv.weight
        )
        for mixing, official in zip(
            cast(list[Any], ours.mixings), theirs.add_blocks, strict=True
        ):
            # Official order: front_block is the lowest level, front_blocks[i] go up.
            official_blocks = [
                *reversed(list(official.front_blocks)),
                official.front_block,
            ]
            for mine, block in zip(mixing.blocks, official_blocks, strict=True):
                mine.kan.coefficients.copy_(block.channel_mixer[0].fc1.cheby_coeffs)
                mine.convolution.weight.copy_(block.conv.conv.weight)
        ours.time_projection.load_state_dict(theirs.predict_layer.state_dict())
        ours.channel_projection.load_state_dict(theirs.projection_layer.state_dict())


@pytest.mark.parametrize("layers", [1, 2])
def test_code_mode_matches_official_forward(layers: int) -> None:
    reference_model = _reference_model()
    torch.manual_seed(0)
    theirs: Any = reference_model(_official_config(e_layers=layers)).eval()
    ours = TimeKAN(
        config=TimeKAN.Config(
            input_length=16,
            output_length=6,
            input_channels=3,
            d_model=8,
            encoder_layers=layers,
            down_sampling_layers=2,
            down_sampling_window=2,
            lowest_order=1,
            dropout=0.0,
            reference="code",
        )
    ).eval()
    _copy_official(ours=ours, theirs=theirs)
    batch_x = torch.randn(4, 16, 3) * 3 + 1
    with torch.no_grad():
        expected = theirs(batch_x, None, None, None)
        actual = ours(batch_x=batch_x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_orders_rise_with_frequency() -> None:
    model = TimeKAN(
        config=TimeKAN.Config(input_length=16, down_sampling_layers=2, lowest_order=1)
    )
    # Band 1 (highest frequency) has order b + k - 1, the lowest level order b.
    blocks = cast(Any, model.mixings[0]).blocks
    orders = [block.kan.coefficients.shape[-1] - 1 for block in blocks]
    assert orders == [3, 2, 1]


def test_paper_mode_squashes_once_and_embeds_linearly() -> None:
    paper = TimeKAN(config=TimeKAN.Config(input_length=16))
    code = TimeKAN(config=TimeKAN.Config(input_length=16, reference="code"))
    assert isinstance(paper.embedding.layer, torch.nn.Linear)
    assert isinstance(code.embedding.layer, torch.nn.Conv1d)
    layer = ChebyshevKAN(input_size=2, output_size=1, order=2, squashes=1)
    features = torch.tensor([[0.3, -1.2]])
    expected = sum(
        layer.coefficients[j, 0, i]
        * torch.cos(i * torch.acos(torch.tanh(features[0, j])))
        for j in range(2)
        for i in range(3)
    )
    torch.testing.assert_close(layer(features=features)[0, 0], expected)


def test_decomposition_bands_add_back_to_the_level() -> None:
    torch.manual_seed(0)
    high = torch.randn(2, 8, 4)
    low = torch.randn(2, 4, 4)
    bands = FrequencyDecomposition()(levels=[high, low])
    rebuilt = bands[0] + FrequencyUpsampling.apply(
        features=low, length=4, target_length=8
    )
    torch.testing.assert_close(rebuilt, high)
    assert bands[1] is low


def test_paper_mode_refuses_lengths_the_pooling_does_not_divide() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TimeKAN.Config(input_length=15, down_sampling_layers=2)
    TimeKAN.Config(input_length=15, down_sampling_layers=2, reference="code")


def test_forecast_shape_and_gradient() -> None:
    model = TimeKAN(
        config=TimeKAN.Config(input_length=32, output_length=8, input_channels=5)
    )
    batch_x = torch.randn(3, 32, 5, requires_grad=True)
    prediction = model(batch_x=batch_x)
    assert prediction.shape == (3, 8, 5)
    prediction.square().mean().backward()
    assert batch_x.grad is not None and torch.isfinite(batch_x.grad).all()
