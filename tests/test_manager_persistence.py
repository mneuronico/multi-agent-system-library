from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mas import AgentSystemManager


def test_history_export_import_round_trip(workspace_tmp_path):
    first = AgentSystemManager(base_directory=str(workspace_tmp_path / "first"))
    first.add_blocks({"response": "hello"}, user_id="user-1")

    exported = first.export_history("user-1")
    assert exported

    second = AgentSystemManager(base_directory=str(workspace_tmp_path / "second"))
    second.import_history("user-1", exported)
    messages = second.get_messages("user-1")

    assert len(messages) == 1
    assert messages[0]["source"] == "user"
    assert messages[0]["message"][0]["content"] == {"response": "hello"}


@pytest.mark.parametrize("user_id", ["../escape", "..\\escape", "nested/user"])
def test_history_paths_remain_inside_history_folder(workspace_tmp_path, user_id):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    history_root = Path(manager.history_folder).resolve()

    db_path = Path(manager._get_db_path_for_user(user_id)).resolve()

    assert db_path.is_relative_to(history_root)


def test_existing_history_database_is_migrated(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    user_id = "legacy-user"
    db_path = Path(manager._get_db_path_for_user(user_id))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE message_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_number INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO message_history (msg_number, role, content, type) VALUES (?, ?, ?, ?)",
        (1, "user", json.dumps([{"type": "text", "content": {"response": "old"}}]), "user"),
    )
    conn.commit()
    conn.close()

    manager.set_current_user(user_id)
    manager.add_blocks({"response": "new"}, user_id=user_id)

    migrated = sqlite3.connect(db_path)
    columns = {row[1] for row in migrated.execute("PRAGMA table_info(message_history)")}
    version = migrated.execute("PRAGMA user_version").fetchone()[0]
    rows = migrated.execute("SELECT msg_number, model FROM message_history ORDER BY msg_number").fetchall()
    migrated.close()

    assert "model" in columns
    assert version == 1
    assert rows == [(1, None), (2, None)]


def test_config_file_errors_fail_fast(workspace_tmp_path):
    missing = workspace_tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        AgentSystemManager(config=str(missing), base_directory=str(workspace_tmp_path))

    malformed = workspace_tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        AgentSystemManager(config=str(malformed), base_directory=str(workspace_tmp_path))

    empty = workspace_tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        AgentSystemManager(config=str(empty), base_directory=str(workspace_tmp_path))
