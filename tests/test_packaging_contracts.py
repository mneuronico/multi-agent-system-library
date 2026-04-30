from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    with (ROOT / "pyproject.toml").open("rb") as fp:
        return tomllib.load(fp)


def test_console_script_targets_are_included_packages():
    data = _pyproject()
    packages = set(data["tool"]["setuptools"]["packages"])
    scripts = data["project"]["scripts"]

    assert scripts["mas"] == "mas.cli.cli:main"
    assert "mas.cli" in packages
    assert "mas.lib" in packages
    assert "maws" in packages


def test_python_requires_matches_annotation_syntax():
    data = _pyproject()
    requires_python = data["project"]["requires-python"]

    uses_pep585_annotations = False
    for path in [ROOT / "mas", ROOT / "maws"]:
        for py_file in path.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    if node.value.id in {"list", "dict", "tuple", "set"}:
                        uses_pep585_annotations = True
                        break
            if uses_pep585_annotations:
                break

    assert not (requires_python == ">=3.8" and uses_pep585_annotations)
