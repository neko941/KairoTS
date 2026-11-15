"""Resolve Config field declaration homes through the source AST, without imports."""

import ast
from functools import cache
from pathlib import Path


def config_field_homes(*, root: Path) -> dict[str, set[str]]:
    definitions: dict[str, tuple[set[str], list[str]]] = {}
    classes: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).with_suffix("").parts
        if relative[-1] == "__init__":
            relative = relative[:-1]
        module = ".".join((root.name, *relative))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {
            alias.asname or alias.name: node.module + "." + alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and not node.level
            for alias in node.names
        }
        for owner in tree.body:
            if not isinstance(owner, ast.ClassDef):
                continue
            bases = []
            for base in owner.bases:
                parts = ast.unparse(base).split(".")
                bases.append(
                    ".".join(
                        (aliases.get(parts[0], module + "." + parts[0]), *parts[1:])
                    )
                )
            classes[module + "." + owner.name] = bases
            for config in owner.body:
                if not isinstance(config, ast.ClassDef) or config.name != "Config":
                    continue
                declared = {
                    field.target.id
                    for field in config.body
                    if isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                }
                parents = []
                for base in config.bases:
                    parts = ast.unparse(base).split(".")
                    parent = ".".join(
                        (aliases.get(parts[0], module + "." + parts[0]), *parts[1:])
                    )
                    parents.append(parent)
                definitions[module + "." + owner.name + ".Config"] = declared, parents

    @cache
    def inheritance_order(*, owner: str) -> tuple[str, ...]:
        parents = classes.get(owner, [])
        sequences = [list(inheritance_order(owner=parent)) for parent in parents] + [
            parents.copy()
        ]
        result = [owner]
        while remaining := [sequence for sequence in sequences if sequence]:
            candidate = next(
                (
                    sequence[0]
                    for sequence in remaining
                    if all(sequence[0] not in other[1:] for other in remaining)
                ),
                None,
            )
            assert candidate is not None, (owner, remaining)
            result.append(candidate)
            for sequence in remaining:
                if sequence[0] == candidate:
                    sequence.pop(0)
        return tuple(result)

    @cache
    def homes(*, owner: str, field: str) -> frozenset[str]:
        if owner not in definitions:
            owner = next(
                (
                    parent + ".Config"
                    for parent in inheritance_order(owner=owner.removesuffix(".Config"))
                    if parent + ".Config" in definitions
                ),
                owner,
            )
        assert owner in definitions, owner
        declared, parents = definitions[owner]
        inherited = frozenset().union(
            *(homes(owner=parent, field=field) for parent in parents)
        )
        return inherited or (frozenset({owner}) if field in declared else frozenset())

    result: dict[str, set[str]] = {}
    for owner, (declared, _) in definitions.items():
        for field in declared - {"name"}:
            result.setdefault(field, set()).update(homes(owner=owner, field=field))
    return result
