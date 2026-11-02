"""Load requested definitions from local upstream files, without running scripts.

No upstream implementation is embedded here. These helpers are not a sandbox:
only use source files obtained from the documented upstream repositories.
"""
import ast
import hashlib
import os
from pathlib import Path
from types import ModuleType


def source_path(variable, relative=None):
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"Set {variable} to the separately obtained upstream source; see docs/external_backbones.md")
    path = Path(value).expanduser().resolve()
    if relative is not None:
        path = path / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream source: {path}")
    return path


def checked_bytes(path, expected):
    # Git may check out CRLF on Windows; scientific source identity uses LF.
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"Upstream source version mismatch: {Path(path).name}; expected LF SHA256 {expected}, got {actual}")
    return raw


def definitions(path, names, namespace, *, expected):
    raw = checked_bytes(path, expected)
    tree = ast.parse(raw.decode("utf-8-sig"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise ValueError(f"Missing definitions in {path}: {sorted(missing)}")
    module = ModuleType("_dirags_external_" + hashlib.sha256(raw).hexdigest()[:16])
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    # Preserve source order, including repeated definition names.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module
