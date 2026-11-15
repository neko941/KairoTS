"""Sphinx configuration; `python scripts/docs.py` builds with warnings as errors."""

import importlib
import inspect
import json
from pathlib import Path

from kairots import __version__
from kairots.registry import catalog, get, module

project = "KairoTS"
release = __version__
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_title = "KairoTS"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_heading_anchors = 3

REFERENCE = Path(__file__).parent / "reference"


def _write_component(*, category: str, name: str, constants: dict[str, object]) -> str:
    """One page per component: docstring, declared constants, Config, members."""
    dotted = module(category=category, name=name)
    page = Path("components") / category / f"{name}.md"
    lines = [
        f"# {name}",
        "",
        f"Category `{category}`, module `{dotted}`. Select it with "
        f'`"name": "{name}"`.',
        "",
    ]
    if constants:
        lines += ["| Constant | Value |", "|---|---|"]
        lines += [f"| `{key}` | `{value}` |" for key, value in sorted(constants.items())]
        lines.append("")
    lines += [
        "```{eval-rst}",
        f".. automodule:: {dotted}",
        "",
        f".. autoclass:: {dotted}.{name}",
        "   :members:",
        "   :exclude-members: Config",
        "",
        f".. autoclass:: {dotted}.{name}.Config",
        "```",
        "",
    ]
    target = REFERENCE / page
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return page.as_posix()


def _write_catalog() -> None:
    """Regenerate the category overview, one table page per category and one page
    per component from the registry.
    """
    # Component pages are generated; remove pages of deleted or moved components.
    for path in (REFERENCE / "components").rglob("*.md"):
        path.unlink()
    lines = [
        "# Component catalog",
        "",
        "Every selectable component by category; a category opens its table and a",
        "name there opens the component's page with its constants and Config.",
        "Generated from the registry on every build; `python -m kairots list`",
        "prints the same list. Constants are explained in",
        "[Where a component can be used](../customization/index.md#where-a-component-can-be-used).",
        "",
        "| Category | Components |",
        "|---|---|",
    ]
    indexes: list[str] = []
    for category, components in catalog().items():
        rows = [f"# `{category}`", "", "| Name | Summary | Uses |", "|---|---|---|"]
        pages: list[str] = []
        for name, constants in components.items():
            page = _write_component(category=category, name=name, constants=constants)
            pages.append(Path(page).name)
            doc = inspect.getdoc(
                importlib.import_module(module(category=category, name=name))
            )
            summary = (doc or "").split("\n\n")[0].replace("\n", " ")
            tasks = constants.get("TASKS")
            details = ", ".join(f"`{task}`" for task in tasks) if tasks else "any task"
            if category == "strategies":
                # A method's families are its abstract parents.
                details = "families: " + ", ".join(
                    f"`{family.__name__}`"
                    for family in get(category=category, name=name).__mro__[1:-2]
                    if family.__dict__.get("ABSTRACT")
                )
            rows.append(
                f"| [{name}]({Path(page).name}) | {summary.replace('|', '/') or '-'} "
                f"| {details} |"
            )
        rows += ["", "```{toctree}", ":hidden:", "", *pages, "```", ""]
        index = f"components/{category}/index.md"
        (REFERENCE / index).write_text("\n".join(rows), encoding="utf-8")
        indexes.append(index)
        lines.append(f"| [`{category}`]({index}) | {len(components)} |")
    lines += ["", "```{toctree}", ":hidden:", "", *indexes, "```", ""]
    (REFERENCE / "catalog.md").write_text("\n".join(lines), encoding="utf-8")


def _write_reproductions() -> None:
    """List every config that declares the published setting it reproduces."""
    configs = Path(__file__).parents[1] / "configs"
    lines = [
        "# Paper settings covered",
        "",
        "Configs that reproduce a published setting exactly, from their `reproduces`",
        "block. Generated on every build; configs without the block are examples.",
        "",
        "| Config | Paper | Table | Cell | Paper value | Code |",
        "|---|---|---|---|---|---|",
    ]
    for path in sorted(configs.rglob("*.json")):
        block = json.loads(path.read_text(encoding="utf-8")).get("reproduces")
        if block is None:
            continue
        name = path.relative_to(configs).as_posix()
        for cell, value in block["cells"].items():
            lines.append(
                f"| `{name}` | {block['paper']} | {block['table']} | {cell} "
                f"| {value} | {block.get('code') or '-'} |"
            )
    lines.append("")
    (REFERENCE / "reproductions.md").write_text("\n".join(lines), encoding="utf-8")


_write_catalog()
_write_reproductions()
