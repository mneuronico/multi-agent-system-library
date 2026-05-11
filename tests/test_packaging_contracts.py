from __future__ import annotations

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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


def test_unused_pandas_is_not_a_core_runtime_dependency():
    data = _pyproject()
    dependencies = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    assert "pandas" not in dependencies
    assert any(dep.startswith("pandas") for dep in extras["data"])


def test_integration_dependencies_are_optional_extras():
    data = _pyproject()
    dependencies = "\n".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]

    for package_name in ["python-telegram-bot", "Flask", "boto3", "python-dotenv", "mutagen"]:
        assert package_name not in dependencies

    for extra_name in ["telegram", "whatsapp", "aws", "env", "audio", "maws", "all"]:
        assert extra_name in extras


def test_split_modules_preserve_public_facades():
    from mas.manager import AgentSystemManager as SplitManager
    from mas.mas import AgentSystemManager as FacadeManager
    from mas.components import Agent, Tool, Process, Automation
    from mas.parser import Parser
    from mas.bots import TelegramBot, WhatsappBot
    from maws.runtime import MawsRuntime as SplitRuntime, build_lambda_handler
    from maws.operations import start, update, setup
    from maws.maws import MawsRuntime as FacadeRuntime

    assert FacadeManager is SplitManager
    assert FacadeRuntime is SplitRuntime
    assert all(obj is not None for obj in [
        Agent, Tool, Process, Automation, Parser, TelegramBot, WhatsappBot,
        build_lambda_handler, start, update, setup,
    ])


def test_maws_operator_files_are_ascii_clean():
    paths = [
        ROOT / "maws" / "maws.py",
        ROOT / "maws" / "README.md",
        ROOT / "maws" / "resources" / "bootstrap.sh",
        ROOT / "maws" / "resources" / "pull_history.sh",
    ]

    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if any(ord(ch) > 127 for ch in path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_maws_shell_resources_are_lf_and_parse_with_bash():
    import shutil
    import subprocess

    paths = [
        ROOT / "maws" / "resources" / "bootstrap.sh",
        ROOT / "maws" / "resources" / "pull_history.sh",
    ]

    for path in paths:
        raw = path.read_bytes()
        assert b"\r\n" not in raw

    bash = shutil.which("bash")
    if bash:
        for path in paths:
            subprocess.check_call([bash, "-n", path.relative_to(ROOT).as_posix()], cwd=ROOT)


def test_maws_bootstrap_exposes_production_params():
    bootstrap = (ROOT / "maws" / "resources" / "bootstrap.sh").read_text(encoding="utf-8")

    for expected in [
        "BUSY_POLICY",
        "MAWS_BUSY_POLICY",
        "MAWS_QUEUE_URL",
        "PERSIST_FILES_S3",
        "FILES_S3_PREFIX",
        "HISTORY_MODE",
        "MAWS_MANAGER_KWARGS_JSON",
        "TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY",
        "WHATSAPP_VERIFY_SIGNATURE",
        "WHATSAPP_APP_SECRET_ENV_KEY",
        "CAPTURE_FAILED_EVENTS",
        "requirements_ref",
        "WorkerQueue",
    ]:
        assert expected in bootstrap
