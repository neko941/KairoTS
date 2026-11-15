import pytest
import torch

from kairots.models.GRUD import GRUD


def _panel() -> torch.Tensor:
    torch.manual_seed(7)
    batch_x = torch.randn(3, 8, 4)
    batch_x[:, :, 1:] = torch.nan
    batch_x[:, 0, 1:] = torch.randn(3, 3)
    batch_x[:, 1:, 0] = torch.arange(1, 8, dtype=batch_x.dtype)
    return batch_x


def _model(*, input_length: int = 8) -> GRUD:
    torch.manual_seed(11)
    model = GRUD(
        config=GRUD.Config(input_length=input_length, input_channels=4, hidden=5)
    )
    model.eval()
    return model


def test_padding_rows_are_inert() -> None:
    batch_x = _panel()
    model = _model()
    extended = torch.cat((batch_x, torch.full((3, 5, 4), torch.nan)), dim=1)
    assert torch.allclose(model(batch_x=batch_x), model(batch_x=extended))


def test_missing_values_produce_finite_logits_and_observations_matter() -> None:
    batch_x = _panel()
    model = _model()
    logits = model(batch_x=batch_x)
    changed = batch_x.clone()
    changed[0, 0, 1] += 2
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits[0], model(batch_x=changed)[0])


def test_decay_uses_time() -> None:
    batch_x = _panel()
    model = _model()
    spread = batch_x.clone()
    spread[:, 1:, 0] *= 10
    assert not torch.allclose(model(batch_x=batch_x), model(batch_x=spread))


def test_num_classes_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        GRUD(config=GRUD.Config(num_classes=1))


def test_irregular_source_needs_a_mask_aware_model() -> None:
    import json
    from pathlib import Path

    from kairots.config.schema import Experiment

    path = Path(__file__).parents[2] / "configs/classification/PhysioNet2012.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    Experiment.from_dict(values=values)
    values["model"] = {"name": "DLinear"}
    with pytest.raises(ValueError, match="does not read masks"):
        Experiment.from_dict(values=values)
