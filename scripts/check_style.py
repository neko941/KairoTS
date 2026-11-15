"""Enforce KairoTS names, keyword-only parameters and complete annotations.

The rules are written out in docs/reference/conventions.md.
"""

import ast
import ipaddress
import re
import tokenize
from io import StringIO
from pathlib import Path

from kairots.registry import categories, module, names

REPO = Path(__file__).parents[1]
ROOT = REPO / "src" / "kairots"
# Words with one preferred spelling in prose, comments and docstrings.
WORDS = {"folder": "directory", "folders": "directories"}
TEXT_ROOTS = ("src", "scripts", "tests", "docs", "configs")
TEXT_FILES = ("README.md", "README.ai.md")
COMPONENT_NAMES = {
    name for category in categories() for name in names(category=category)
}
COMPONENT_PATHS = {
    (name, module(category=category, name=name).replace(".", "/"))
    for category in categories()
    for name in names(category=category)
}
# Component categories: packages whose __init__ exposes a registry surface. N2 applies
# inside them; other packages (runtime/, config/) hold infrastructure modules.
_SURFACES = [
    path.parent
    for path in ROOT.rglob("__init__.py")
    if "surface(package=__name__)" in path.read_text(encoding="utf-8")
]
# Leaf surfaces only: runtime/ re-exports runtime/executors/ but is infrastructure.
CATEGORIES = [
    directory
    for directory in _SURFACES
    if not any(directory in other.parents for other in _SURFACES)
]
DENY = {
    "seq_len",
    "lookback",
    "context_length",
    "context_len",
    "pred_len",
    "horizon",
    "prediction_length",
    "start_token_len",
    "gap_len",
    "enc_in",
    "n_vars",
    "num_features",
    "c_in",
    "c_out",
    "num_targets",
    "hidden_size",
    "hidden_dim",
    "embed_dim",
    "n_clients",
    "client_num",
    "client_num_in_total",
    "cid",
    "client_idx",
    "round_idx",
    "global_round",
    "comm_round",
    "round",
    "global_rounds",
    "num_rounds",
    "n_comm_rounds",
    "num_global_iters",
    "local_epochs",
    "local_ep",
    "local_n_epochs",
    "frac",
    "sample_rate",
    "client_fraction",
    "client_num_per_round",
    "lr",
    "local_lr",
    "inputs",
    "targets",
    "x_mark",
    "y_mark",
}

ABBREVIATIONS = set(
    (
        "n e d dim len avg lr ff enc dec hid emb seq pred std eps cfg idx "
        "attn conv tmp fn op iter"
    ).split()
)
PAPER_SYMBOLS = {
    "d_model": "Vaswani et al., Attention Is All You Need: model dimension d_model",
    "d_ff": "Vaswani et al., Attention Is All You Need: feed-forward dimension d_ff",
    "k": "Wu et al., TimesNet: selected frequency count k",
}
SHORT_BINDINGS = {"x", "y", "i", "j", "ids", "cap", "var"}


def check_source(*, source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    errors: list[str] = []
    lines = source.splitlines()
    path = Path(filename.replace("\\", "/"))
    own_components = {
        name
        for name, component_path in COMPONENT_PATHS
        if path.as_posix().endswith(
            (component_path + ".py", component_path + "/__init__.py")
        )
    }
    for node in ast.walk(tree):
        binding = (
            node.arg
            if isinstance(node, ast.arg)
            else node.id
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            else None
        )
        if (
            isinstance(node, (ast.arg, ast.Name))
            and binding is not None
            and binding not in PAPER_SYMBOLS
            and (
                ABBREVIATIONS.intersection(binding.split("_"))
                or binding in SHORT_BINDINGS
            )
        ):
            errors.append(
                f"{filename}:{node.lineno}: D2 use full words for {binding!r}"
            )
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "config"
            and re.search(r"\bAny\b", ast.unparse(node.annotation))
        ):
            errors.append(
                f"{filename}:{node.lineno}: A2 retain the component Config type"
            )
        if isinstance(node, ast.ClassDef):
            if node.name == "Config":
                documentation = ast.get_docstring(node) or ""
                for field in node.body:
                    if not (
                        isinstance(field, ast.AnnAssign)
                        and isinstance(field.target, ast.Name)
                        and field.target.id in PAPER_SYMBOLS
                    ):
                        continue
                    symbol = field.target.id
                    author = PAPER_SYMBOLS[symbol].split()[0]
                    if (
                        symbol not in re.findall(r"\w+", documentation)
                        or author not in documentation
                    ):
                        errors.append(
                            f"{filename}:{field.lineno}: D3 document {symbol!r} "
                            f"and its paper ({author}) in the Config docstring"
                        )
            owns_config = any(
                isinstance(child, ast.ClassDef) and child.name == "Config"
                for child in node.body
            )
            for child in node.body:
                annotations = []
                if (
                    isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "config"
                ):
                    annotations.append(child.annotation)
                if (
                    owns_config
                    and isinstance(child, ast.FunctionDef)
                    and child.name == "__init__"
                ):
                    annotations.extend(
                        argument.annotation
                        for argument in (*child.args.args, *child.args.kwonlyargs)
                        if argument.arg == "config" and argument.annotation is not None
                    )
                if any(
                    re.search(r"\bAny\b", ast.unparse(annotation))
                    for annotation in annotations
                ):
                    errors.append(
                        f"{filename}:{child.lineno}: A2 retain the component "
                        "Config type"
                    )
        if (
            isinstance(node, ast.Compare)
            and any(
                isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn))
                for op in node.ops
            )
            and any(
                isinstance(value, ast.Attribute) and value.attr == "name"
                for value in (node.left, *node.comparators)
            )
        ):
            literals = {
                literal.value
                for value in (node.left, *node.comparators)
                for literal in (
                    value.elts
                    if isinstance(value, (ast.Tuple, ast.List, ast.Set))
                    else [value]
                )
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str)
            }
            if any(
                literal not in own_components
                and (
                    literal in COMPONENT_NAMES
                    or path.as_posix().endswith("config/schema.py")
                )
                for literal in literals
            ):
                errors.append(
                    f"{filename}:{node.lineno}: C1 declare component capabilities "
                    "or REQUIRES"
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [
                arg for arg in node.args.args if arg.arg not in {"self", "cls"}
            ]
            exception = "# positional:" in lines[node.lineno - 1]
            if positional and not exception:
                errors.append(
                    f"{filename}:{node.lineno}: K1 positional parameters need "
                    "keyword-only or # positional: reason"
                )
            arguments = node.args
            for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
                if arg.arg not in {"self", "cls"} and arg.annotation is None:
                    errors.append(
                        f"{filename}:{node.lineno}: "
                        f"A1 parameter {arg.arg} lacks annotation"
                    )
            if node.returns is None:
                errors.append(
                    f"{filename}:{node.lineno}: A1 return annotation is missing"
                )
        if isinstance(node, ast.arg) and node.arg in DENY:
            errors.append(
                f"{filename}:{node.lineno}: D1 deny-listed parameter {node.arg}"
            )
        if isinstance(node, ast.keyword) and node.arg in DENY:
            errors.append(
                f"{filename}:{node.lineno}: D1 deny-listed keyword {node.arg}"
            )
        # Only attributes we assign; reading a library's (ast.Assign.targets) is fine.
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and node.attr in DENY
        ):
            errors.append(
                f"{filename}:{node.lineno}: D1 deny-listed attribute {node.attr}"
            )
        if isinstance(node, ast.Name) and node.id in DENY:
            errors.append(
                f"{filename}:{node.lineno}: D1 deny-listed identifier {node.id}"
            )
    return errors


def check_model_imports(*, path: Path, source: str) -> list[str]:
    """M1: a model imports no other model's directory; shared blocks go in layers/."""
    models = ROOT / "models"
    if models not in path.parents or path.parent == models:
        return []
    own = path.relative_to(models).parts[0]
    errors = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            targets = [node.module or ""]
        elif isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        else:
            continue
        for target in targets:
            parts = target.split(".")
            if (
                parts[:2] == ["kairots", "models"]
                and len(parts) > 2
                and parts[2] != own
                and (models / parts[2]).is_dir()
            ):
                errors.append(
                    f"{path}:{node.lineno}: M1 import {target} through layers/, "
                    "not another model's directory"
                )
    return errors


def check_file_name(*, path: Path, source: str) -> list[str]:
    """N2: a PascalCase file (or package) defines the class it is named after.

    A lowercase file is infrastructure and must not define a class named after its
    capitalized stem (composite.py defining Composite should be Composite.py).
    """
    if path.name == "__init__.py" and path.parent in CATEGORIES:
        errors = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, ast.Assign) and (
                all(
                    isinstance(value, ast.Name) and value.id == "__all__"
                    for value in node.targets
                )
                or (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "surface"
                )
            ):
                continue
            errors.append(f"{path}:{node.lineno}: C2 category defines only its surface")
        return errors
    stem = path.parent.name if path.name == "__init__.py" else path.stem
    if stem.startswith("_") or not any(
        category in path.parents for category in CATEGORIES
    ):
        return []
    classes = {
        node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
    }
    if stem[0].isupper() and stem not in classes:
        return [f"{path}:1: N2 {stem}.py must define class {stem}"]
    if stem[0].islower() and stem[:1].upper() + stem[1:] in classes:
        name = stem[:1].upper() + stem[1:]
        return [f"{path}:1: N2 class {name} belongs in {name}.py, not {path.name}"]
    return []


def check_words(*, path: Path, text: str) -> list[str]:
    """W1: use the preferred spelling (directory, not folder)."""
    errors: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for word in re.findall(r"[A-Za-z]+", line):
            preferred = WORDS.get(word.lower())
            if preferred is not None:
                errors.append(f"{path}:{number}: W1 write {preferred!r}, not {word!r}")
    return errors


def check_public_text(*, path: Path, text: str) -> list[str]:
    """P1/P2: original citations and no private environment details in public files."""
    filename = path.as_posix()
    private = re.compile(
        r"\b(?:server\d+|profkang\d+)\b|(?<!\w)[A-Za-z]:[/\\]|/home/"
        r"|\bneko-(?:brain|code|research|library)(?:-[\w-]+)?\b",
        re.IGNORECASE,
    )
    ports = re.compile(r"\b(?:FedProC|FL-bench|HtFLlib)\b", re.IGNORECASE)
    partition_source = filename in {
        "src/kairots/data/partitions/ClassPerClient.py",
        "docs/customization/partitions.md",
        "docs/explanation/notes.md",
    }
    address_lines = text.splitlines()
    if path.suffix == ".py":
        # Python slice syntax resembles IPv6; addresses belong to strings/comments.
        address_lines = [""] * len(address_lines)
        for token in tokenize.generate_tokens(StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT}:
                for offset, line in enumerate(token.string.splitlines()):
                    address_lines[token.start[0] - 1 + offset] += line
    errors = []
    for number, line in enumerate(text.splitlines(), start=1):
        addresses = re.findall(
            r"(?<![\w:.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
            r"|(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F]{0,4}(?![\w:])",
            address_lines[number - 1],
        )
        has_address = False
        for address in addresses:
            if not address.strip(":"):
                continue
            try:
                ipaddress.ip_address(address)
            except ValueError:
                continue
            has_address = True
        if private.search(line) or has_address:
            errors.append(f"{path}:{number}: P1 private environment detail")
        if ports.search(line) or (
            re.search(r"\bPFLlib\b", line, re.IGNORECASE)
            and not partition_source
            and not re.search(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b", line, re.I)
        ):
            errors.append(f"{path}:{number}: P2 cite the original method source")
    return errors


def check_directories(*, root: Path) -> list[str]:
    """N3: source packages have an init and never consist solely of bytecode caches."""
    errors = []
    for path in root.rglob("*"):
        if not path.is_dir() or "__pycache__" in path.relative_to(root).parts:
            continue
        if {child.name for child in path.iterdir()} == {"__pycache__"}:
            errors.append(f"{path}: N3 directory holds only a bytecode cache")
        if not (path / "__init__.py").is_file():
            errors.append(f"{path}: N3 package needs __init__.py")
    return errors


def main() -> None:
    errors = check_directories(root=ROOT)
    for path in ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        errors.extend(check_source(source=source, filename=str(path)))
        errors.extend(check_file_name(path=path, source=source))
        errors.extend(check_model_imports(path=path, source=source))
    texts = [REPO / name for name in TEXT_FILES] + [
        path
        for root in TEXT_ROOTS
        for path in (REPO / root).rglob("*")
        if path.suffix in {".py", ".md", ".json", ".toml"}
        and "_build" not in path.parts
        and path.name != "check_style.py"
    ]
    for path in texts:
        if path.is_file():
            if path.name != "conventions.md":
                errors.extend(
                    check_words(path=path, text=path.read_text(encoding="utf-8"))
                )
            relative = path.relative_to(REPO)
            if (
                relative.parts[0] in {"src", "docs", "configs"}
                or path.name in TEXT_FILES
            ):
                errors.extend(
                    check_public_text(
                        path=relative, text=path.read_text(encoding="utf-8")
                    )
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print("style check passed")


if __name__ == "__main__":
    main()
