from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_STYLE_SPEC = importlib.util.spec_from_file_location(
    "check_style", Path(__file__).parents[2] / "scripts" / "check_style.py"
)
assert _STYLE_SPEC is not None and _STYLE_SPEC.loader is not None
check_style = importlib.util.module_from_spec(_STYLE_SPEC)
_STYLE_SPEC.loader.exec_module(check_style)


def test_check_style_rejects_bad_glossary_names() -> None:
    snippet = (
        "def fit(*, seq_len: int) -> None:\n"
        "    model.pred_len = seq_len\n"
        "    train(inputs=seq_len)\n"
    )
    errors = check_style.check_source(source=snippet)
    assert any("seq_len" in error for error in errors)
    assert any("pred_len" in error for error in errors)
    assert any("inputs" in error for error in errors)


@pytest.mark.parametrize("token", sorted(check_style.ABBREVIATIONS))
@pytest.mark.parametrize("kind", ["field", "parameter", "local"])
def test_readable_names_reject_abbreviations(token: str, kind: str) -> None:
    name = f"value_{token}"
    snippets = {
        "field": f"class Method:\n    class Config:\n        {name}: int = 1\n",
        "parameter": f"def method(*, {name}: int) -> None: pass\n",
        "local": f"def method() -> None:\n    {name} = 1\n",
    }
    assert any(
        "D2" in error for error in check_style.check_source(source=snippets[kind])
    )


def test_readable_names_keep_cited_symbols_and_accepted_prefixes() -> None:
    assert all(check_style.PAPER_SYMBOLS.values())
    assert not check_style.check_source(
        source=(
            "def method(*, d_model: int, d_ff: int, k: int, "
            "num_clients: int, max_depth: int, min_windows: int) -> None:\n"
            "    values = tensor.mean(dim=0)\n"
        )
    )


@pytest.mark.parametrize("name", sorted(check_style.SHORT_BINDINGS))
def test_standalone_shorthand_bindings_need_full_words(name: str) -> None:
    assert any(
        "D2" in error
        for error in check_style.check_source(
            source=f"def method() -> None:\n    {name} = 1\n"
        )
    )


def test_config_symbol_needs_its_paper_in_the_docstring() -> None:
    assert any(
        "D3" in error
        for error in check_style.check_source(
            source="class Method:\n    class Config:\n        d_ff: int = 16\n"
        )
    )
    assert not check_style.check_source(
        source=(
            "class Method:\n    class Config:\n"
            '        """Vaswani et al.: feed-forward dimension d_ff."""\n'
            "        d_ff: int = 16\n"
        )
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "class Method:\n"
        "    class Config: pass\n"
        "    def __init__(self, *, config: Any) -> None: self.config = config\n",
        "class Client:\n    config: Any\n",
        "class Method:\n"
        "    class Config: pass\n"
        "    def __init__(self, *, config: Config) -> None:\n"
        "        self.config: Any = config\n",
    ],
)
def test_retained_component_config_is_not_any(*, snippet: str) -> None:
    assert any("A2" in error for error in check_style.check_source(source=snippet))


def test_schema_uses_component_requirements() -> None:
    for comparison in ('== "CSDI"', '!= "CSDI"', 'in ("CSDI",)', 'not in ("CSDI",)'):
        errors = check_style.check_source(
            source=f"if model.name {comparison}: pass",
            filename="src/kairots/config/schema.py",
        )
        assert any("C1" in error for error in errors)
    assert not check_style.check_source(
        source="if model.name not in allowed: pass",
        filename="src/kairots/config/schema.py",
    )
    assert not check_style.check_source(
        source='if model.name == getattr(other, "name"): pass',
        filename="src/kairots/config/schema.py",
    )


@pytest.mark.parametrize(
    "comparison",
    ['== "ZeroShot"', '!= "ZeroShot"', 'in ("ZeroShot",)', 'not in ("ZeroShot",)'],
)
def test_runtime_dispatch_uses_component_capabilities(*, comparison: str) -> None:
    assert any(
        "C1" in error
        for error in check_style.check_source(
            source=f"if experiment.strategy.name {comparison}: pass",
            filename="src/kairots/runtime/orchestrator.py",
        )
    )
    assert not check_style.check_source(
        source=f"if self.config.name {comparison}: pass",
        filename="src/kairots/strategies/ZeroShot.py",
    )
    assert any(
        "C1" in error
        for error in check_style.check_source(
            source=f"if experiment.strategy.name {comparison}: pass",
            filename="src/kairots/runtime/ZeroShot.py",
        )
    )
    assert not check_style.check_source(
        source='if path.name == "__init__.py": pass',
        filename="src/kairots/registry.py",
    )


def test_category_init_defines_only_registry_surface() -> None:
    path = check_style.ROOT / "optimizers" / "__init__.py"
    assert any(
        "C2" in error
        for error in check_style.check_file_name(path=path, source="class Adam: pass")
    )
    assert not check_style.check_file_name(
        path=path,
        source="from kairots.registry import surface\n"
        "get, names, __getattr__, __dir__ = surface(package=__name__)\n"
        '__all__ = ["get", "names"]\n',
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "server5",
        "profkang2",
        "192.168.1.5",
        "2001:db8::1",
        "C:/data",
        r"D:\data",
        "/home/user/data",
        "neko-research",
    ],
)
def test_public_text_rejects_private_environment_details(*, snippet: str) -> None:
    assert any(
        "P1" in error
        for error in check_style.check_public_text(
            path=Path("docs/example.md"), text=snippet
        )
    )


@pytest.mark.parametrize("port", ["FedProC", "FL-bench", "HtFLlib", "PFLlib"])
def test_public_text_rejects_port_citations(*, port: str) -> None:
    assert any(
        "P2" in error
        for error in check_style.check_public_text(
            path=Path("docs/example.md"), text=port
        )
    )


def test_public_text_keeps_original_partition_source_and_public_urls() -> None:
    for filename in [
        "src/kairots/data/partitions/ClassPerClient.py",
        "docs/customization/partitions.md",
        "docs/explanation/notes.md",
    ]:
        assert not check_style.check_public_text(path=Path(filename), text="PFLlib")
    assert not check_style.check_public_text(
        path=Path("docs/example.md"),
        text="https://github.com/thuml/Time-Series-Library",
    )
    assert not check_style.check_public_text(
        path=Path("src/example.py"), text="result = values[:, 1::2]"
    )
    assert any(
        "P1" in error
        for error in check_style.check_public_text(
            path=Path("src/example.py"), text='host = "2001:db8::1"'
        )
    )


def test_public_text_allows_partition_port_with_its_citation() -> None:
    path = Path("docs/reference/components/data/partitions/index.md")
    assert not check_style.check_public_text(
        path=path,
        text="PFLlib class-count label skew, Zhang et al., arXiv:2312.04992v2.",
    )
    assert any(
        "P2" in error
        for error in check_style.check_public_text(path=path, text="PFLlib")
    )


def test_directory_guard_rejects_missing_init_and_cache_only_package(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "metrics"
    missing.mkdir()
    assert any("N3" in error for error in check_style.check_directories(root=tmp_path))
    (missing / "__init__.py").write_text("")
    assert not check_style.check_directories(root=tmp_path)
    (tmp_path / "retired" / "__pycache__").mkdir(parents=True)
    assert any(
        "only a bytecode cache" in error
        for error in check_style.check_directories(root=tmp_path)
    )


def test_glossary_document_still_gets_public_detail_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src" / "kairots"
    source.mkdir(parents=True)
    document = tmp_path / "docs" / "reference" / "conventions.md"
    document.parent.mkdir(parents=True)
    document.write_text("server5")
    monkeypatch.setattr(check_style, "REPO", tmp_path)
    monkeypatch.setattr(check_style, "ROOT", source)
    with pytest.raises(SystemExit, match="P1 private environment detail"):
        check_style.main()


def test_model_imports_no_other_model_directory() -> None:
    path = check_style.ROOT / "models" / "Informer" / "DataEmbedding.py"
    source = (
        "from kairots.models.TimesNet.PositionalEmbedding import PositionalEmbedding"
    )
    assert any(
        "M1" in error
        for error in check_style.check_model_imports(path=path, source=source)
    )
    assert not check_style.check_model_imports(
        path=path,
        source="from kairots.layers.PositionalEmbedding import PositionalEmbedding\n"
        "from kairots.models.Informer.Encoder import Encoder\n"
        "from kairots.models.base import Model\n",
    )


def test_inverted_attention_variants_share_itransformer_backbone() -> None:
    path = check_style.ROOT / "models" / "iFlashformer" / "__init__.py"
    assert not check_style.check_model_imports(
        path=path,
        source=(
            "from kairots.models.iTransformer.InvertedTransformer "
            "import InvertedTransformer\n"
        ),
    )
