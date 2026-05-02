from __future__ import annotations

from maws import cli
from maws import operations


def test_pull_history_accepts_single_and_multiple_user_ids(monkeypatch, workspace_tmp_path):
    captured = {}

    def fake_run_bash(script, args, cwd=None, quiet=False):
        captured["script"] = script
        captured["args"] = args
        captured["cwd"] = cwd
        captured["quiet"] = quiet
        return 0

    monkeypatch.setattr(operations, "_run_bash", fake_run_bash)

    code = operations.pull_history(
        config_path="params.json",
        project_dir=workspace_tmp_path,
        user_ids=["user-one", "user-two,user-three", "user-one"],
        quiet=True,
    )

    assert code == 0
    assert captured["cwd"] == workspace_tmp_path
    assert captured["quiet"] is True
    assert captured["args"] == [
        "--config",
        "params.json",
        "--user-id",
        "user-one",
        "--user-id",
        "user-two",
        "--user-id",
        "user-three",
    ]


def test_pull_history_without_user_ids_keeps_full_sync_behavior(monkeypatch, workspace_tmp_path):
    captured = {}

    def fake_run_bash(script, args, cwd=None, quiet=False):
        captured["args"] = args
        return 0

    monkeypatch.setattr(operations, "_run_bash", fake_run_bash)

    assert operations.pull_history(project_dir=workspace_tmp_path) == 0
    assert captured["args"] == ["--config", "params.json"]


def test_pull_history_cli_forwards_user_id_filters(monkeypatch):
    captured = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_pull", fake_pull)

    code = cli.main([
        "pull-history",
        "--user-id",
        "chat-a",
        "--user-ids",
        "chat-b,chat-c",
        "--quiet",
    ])

    assert code == 0
    assert captured["quiet"] is True
    assert captured["user_ids"] == ["chat-a", "chat-b,chat-c"]
