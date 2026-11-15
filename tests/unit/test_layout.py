"""Component layouts preserve registry identities and isolate model helpers."""

import ast
import inspect
from pathlib import Path

from kairots.data.stores.Parquet import family
from kairots.registry import get, names


def test_model_helpers_have_their_own_named_files() -> None:
    models = Path(__file__).resolve().parents[2] / "src/kairots/models"
    violations = []
    for path in models.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if path == models / "base.py":
            # Shared schema bases cannot hide runtime model helpers.
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    assert len(node.body) == 1
                    assert isinstance(node.body[0], ast.ClassDef)
                    assert node.body[0].name == "Config"
            continue
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        expected = path.parent.name if path.name == "__init__.py" else path.stem
        if classes and classes != [expected]:
            violations.append(f"{path.relative_to(models)}: {classes}")
    assert not violations, "\n".join(violations)


def test_single_dataset_packages_keep_their_families() -> None:
    for name in (
        "Electricity",
        "ExchangeRate",
        "GoogleStock",
        "JenaWeather",
        "PhysioNet2012",
        "TinyWeather5K",
        "TourismLarge",
    ):
        source = get(category="data/sources", name=name)
        assert source.__module__ == f"kairots.data.sources.{name}"
        assert Path(inspect.getfile(source)).name == "__init__.py"
        assert family(source_type=source) == name


def test_model_specific_initializer_aliases_stay_removed() -> None:
    assert "DLinearAverage" not in names(category="initializers")


def test_flat_categories_keep_components_in_the_category_directory() -> None:
    root = Path(__file__).resolve().parents[2] / "src/kairots"
    for name in ("initializers", "evaluation", "training"):
        directory = root / name
        assert all(path.parent == directory for path in directory.rglob("*.py"))
