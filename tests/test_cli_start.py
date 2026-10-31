"""`mas start` must never overwrite an existing project and must not hang without a TTY."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas import AgentSystemManager
from mas.cli import cli


DESCRIPTION = "A system that summarizes the videos the user sends"


def _no_tty(monkeypatch):
    monkeypatch.setattr(cli, "_is_tty", lambda: False)


def _tty_with_answers(monkeypatch, answers):
    """Simulate an interactive terminal; fail instead of looping forever if prompts run out."""
    prompts = []
    remaining = list(answers)

    def fake_input(prompt=""):
        prompts.append(prompt)
        if len(prompts) > 50:
            raise AssertionError("prompted too many times; probably an endless loop")
        if not remaining:
            raise EOFError
        value = remaining.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(cli, "_is_tty", lambda: True)
    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


class _FakeBootstrapManager:
    """Stands in for AgentSystemManager during `mas start -c "<description>"` (no model calls)."""

    _looks_like_config_path = staticmethod(AgentSystemManager._looks_like_config_path)
    calls = []

    def __init__(self, config=None, bootstrap_models=None, base_directory=None, verbose=False, **kwargs):
        type(self).calls.append({"config": config, "bootstrap_models": bootstrap_models})
        Path(base_directory, "config.json").write_text('{"generated": true}', encoding="utf-8")
        Path(base_directory, "fns.py").write_text("# generated\n", encoding="utf-8")


class _ForbiddenManager:
    _looks_like_config_path = staticmethod(AgentSystemManager._looks_like_config_path)

    def __init__(self, *args, **kwargs):
        raise AssertionError("no system should be generated")


def _write_existing_project(directory: Path):
    files = {
        "config.json": '{"components": [{"type": "agent", "name": "mine"}]}',
        ".env": "OPENAI_API_KEY=sk-real-key\n",
        "fns.py": "def my_tool():\n    return 42\n",
    }
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
    return files


def test_start_in_an_empty_directory_creates_the_project(workspace_tmp_path, monkeypatch):
    _no_tty(monkeypatch)

    assert cli.main(["start", "-d", str(workspace_tmp_path)]) == 0

    for name in ("config.json", ".env", "fns.py", "main.py"):
        assert (workspace_tmp_path / name).is_file()
    config = json.loads((workspace_tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["general_parameters"]["default_models"] == [
        {"provider": "google", "model": "gemini-2.5-flash"}
    ]
    assert (workspace_tmp_path / ".env").read_text(encoding="utf-8") == "# GOOGLE_API_KEY=\n"


def test_start_keeps_existing_config_env_and_fns(workspace_tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    existing = _write_existing_project(workspace_tmp_path)

    assert cli.main(["start", "-d", str(workspace_tmp_path)]) == 0

    for name, content in existing.items():
        assert (workspace_tmp_path / name).read_text(encoding="utf-8") == content
    assert (workspace_tmp_path / "main.py").is_file()

    out = capsys.readouterr().out
    assert "Kept existing files (not overwritten):" in out
    for name in existing:
        assert str(workspace_tmp_path / name) in out
    assert f"Created files:\n- {workspace_tmp_path / 'main.py'}" in out


def test_start_with_a_complete_project_changes_nothing(workspace_tmp_path, monkeypatch, capsys):
    prompts = _tty_with_answers(monkeypatch, [])
    existing = _write_existing_project(workspace_tmp_path)
    existing["main.py"] = "print('my main')\n"
    (workspace_tmp_path / "main.py").write_text(existing["main.py"], encoding="utf-8")

    assert cli.main(["start", "-d", str(workspace_tmp_path)]) == 0

    for name, content in existing.items():
        assert (workspace_tmp_path / name).read_text(encoding="utf-8") == content
    assert prompts == []
    assert "Nothing to create" in capsys.readouterr().out


def test_start_does_not_ask_for_an_api_key_when_env_exists(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / ".env").write_text("GOOGLE_API_KEY=sk-real\n", encoding="utf-8")
    prompts = _tty_with_answers(monkeypatch, ["", ""])

    assert cli.main(["start", "-d", str(workspace_tmp_path)]) == 0

    assert not any("API_KEY" in prompt for prompt in prompts)
    assert (workspace_tmp_path / ".env").read_text(encoding="utf-8") == "GOOGLE_API_KEY=sk-real\n"
    assert (workspace_tmp_path / "config.json").is_file()


def test_start_from_json_never_overwrites_config(workspace_tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    project = workspace_tmp_path / "project"
    project.mkdir()
    (project / "config.json").write_text('{"mine": true}', encoding="utf-8")
    source = workspace_tmp_path / "other.json"
    source.write_text('{"other": true}', encoding="utf-8")

    assert cli.main(["start", "-d", str(project), "-c", str(source)]) == 1

    assert (project / "config.json").read_text(encoding="utf-8") == '{"mine": true}'
    assert "kept (not overwritten)" in capsys.readouterr().out


def test_start_from_json_copies_into_an_empty_project(workspace_tmp_path, monkeypatch):
    _no_tty(monkeypatch)
    project = workspace_tmp_path / "project"
    source = workspace_tmp_path / "other.json"
    source.write_text('{"other": true}', encoding="utf-8")

    assert cli.main(["start", "-d", str(project), "-c", str(source)]) == 0

    assert (project / "config.json").read_text(encoding="utf-8") == '{"other": true}'


def test_start_from_description_without_tty_aborts_instead_of_looping(workspace_tmp_path, monkeypatch, capsys):
    _no_tty(monkeypatch)
    monkeypatch.setattr(cli, "AgentSystemManager", _ForbiddenManager)
    calls = {"n": 0}
    real_ask = cli._ask

    def counting_ask(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 50:
            raise AssertionError("_ask called too many times; probably an endless loop")
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(cli, "_ask", counting_ask)

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 1

    out = capsys.readouterr().out
    assert "Aborted" in out
    assert "GOOGLE_API_KEY" in out
    assert "not a TTY" in out
    assert list(workspace_tmp_path.iterdir()) == []


def test_start_from_description_aborts_when_the_terminal_hits_eof(workspace_tmp_path, monkeypatch, capsys):
    _tty_with_answers(monkeypatch, ["", ""])  # provider and model, then EOF at the key prompt
    monkeypatch.setattr(cli, "AgentSystemManager", _ForbiddenManager)

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 1

    assert "Aborted" in capsys.readouterr().out


def test_start_from_description_gives_up_after_repeated_empty_keys(workspace_tmp_path, monkeypatch):
    prompts = _tty_with_answers(monkeypatch, ["", ""] + [""] * 10)
    monkeypatch.setattr(cli, "AgentSystemManager", _ForbiddenManager)

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 1

    assert sum("GOOGLE_API_KEY" in prompt for prompt in prompts) == 3


def test_start_from_description_refuses_an_existing_project_before_prompting(workspace_tmp_path, monkeypatch, capsys):
    existing = _write_existing_project(workspace_tmp_path)
    monkeypatch.setattr(cli, "AgentSystemManager", _ForbiddenManager)

    def no_prompts(*args, **kwargs):
        raise AssertionError("should not prompt when the project already exists")

    monkeypatch.setattr(cli, "_ask", no_prompts)

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 1

    for name, content in existing.items():
        assert (workspace_tmp_path / name).read_text(encoding="utf-8") == content
    out = capsys.readouterr().out
    assert "kept (not overwritten)" in out
    assert str(workspace_tmp_path / "config.json") in out


def test_start_from_description_keeps_an_existing_env(workspace_tmp_path, monkeypatch, capsys):
    (workspace_tmp_path / ".env").write_text("GOOGLE_API_KEY=sk-real\n", encoding="utf-8")
    _FakeBootstrapManager.calls = []
    monkeypatch.setattr(cli, "AgentSystemManager", _FakeBootstrapManager)
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder")  # restored after the test; the CLI sets it
    # provider, model, key, final provider, final model
    _tty_with_answers(monkeypatch, ["", "", "sk-bootstrap", "", ""])

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 0

    assert len(_FakeBootstrapManager.calls) == 1
    assert (workspace_tmp_path / ".env").read_text(encoding="utf-8") == "GOOGLE_API_KEY=sk-real\n"
    assert (workspace_tmp_path / "config.json").read_text(encoding="utf-8") == '{"generated": true}'
    out = capsys.readouterr().out
    assert "Kept existing files (not overwritten):" in out
    assert str(workspace_tmp_path / ".env") in out


def test_start_from_description_writes_env_when_missing(workspace_tmp_path, monkeypatch):
    _FakeBootstrapManager.calls = []
    monkeypatch.setattr(cli, "AgentSystemManager", _FakeBootstrapManager)
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder")  # restored after the test; the CLI sets it
    _tty_with_answers(monkeypatch, ["", "", "sk-bootstrap", "", ""])

    assert cli.main(["start", "-d", str(workspace_tmp_path), "-c", DESCRIPTION]) == 0

    assert (workspace_tmp_path / ".env").read_text(encoding="utf-8") == "GOOGLE_API_KEY=sk-bootstrap\n"
