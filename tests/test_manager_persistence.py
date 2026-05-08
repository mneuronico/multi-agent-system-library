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


def test_default_history_mode_keeps_one_database_per_user(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    manager.add_blocks({"response": "hello alice"}, user_id="alice")
    manager.add_blocks({"response": "hello bob"}, user_id="bob")

    assert manager.history_mode == "per_user"
    assert sorted(path.name for path in Path(manager.history_folder).glob("*.sqlite")) == [
        "alice.sqlite",
        "bob.sqlite",
    ]


def test_get_messages_without_active_user_does_not_create_empty_history(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    assert manager.get_messages() == []
    assert list(Path(manager.history_folder).glob("*.sqlite")) == []


def test_nonblocking_run_uses_resolved_user_id_for_history_and_callback(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.create_process("echo", lambda messages=None: {"ok": True})
    seen = []

    manager.run(
        input="hello",
        component_name="echo",
        user_id="alice",
        blocking=False,
        on_complete=lambda messages, manager: seen.append(messages),
    )

    import time
    for _ in range(50):
        if seen:
            break
        time.sleep(0.02)

    assert seen
    assert [msg["source"] for msg in manager.get_messages("alice")] == ["user", "echo"]


def test_shared_history_stores_user_id_and_rotates_by_message_count(workspace_tmp_path):
    manager = AgentSystemManager(
        base_directory=str(workspace_tmp_path),
        history_mode="shared",
        history_max_messages=3,
    )

    manager.add_blocks({"response": "a1"}, user_id="alice")
    manager.add_blocks({"response": "b1"}, user_id="bob")
    manager.add_blocks({"response": "a2"}, user_id="alice")
    manager.add_blocks({"response": "b2"}, user_id="bob")

    paths = sorted(Path(manager.history_folder).glob("*.sqlite"))
    assert [path.name for path in paths] == [
        "shared_history_000001.sqlite",
        "shared_history_000002.sqlite",
    ]

    conn = sqlite3.connect(paths[0])
    columns = {row[1] for row in conn.execute("PRAGMA table_info(message_history)")}
    rows = conn.execute(
        "SELECT user_id, msg_number, role, type FROM message_history ORDER BY id"
    ).fetchall()
    conn.close()

    assert "user_id" in columns
    assert rows == [
        ("alice", 1, "user", "user"),
        ("bob", 1, "user", "user"),
        ("alice", 2, "user", "user"),
    ]

    alice = manager.get_messages("alice")
    bob = manager.get_messages("bob")
    assert [msg["message"][0]["content"]["response"] for msg in alice] == ["a1", "a2"]
    assert [msg["msg_number"] for msg in alice] == [1, 2]
    assert [msg["user_id"] for msg in alice] == ["alice", "alice"]
    assert [msg["message"][0]["content"]["response"] for msg in bob] == ["b1", "b2"]
    assert [msg["msg_number"] for msg in bob] == [1, 2]


def test_shared_history_can_rotate_by_time_period(workspace_tmp_path):
    manager = AgentSystemManager(
        base_directory=str(workspace_tmp_path),
        history_mode="shared",
        history_rotation="time_period",
        history_period="1 day",
    )

    manager.add_blocks({"response": "first"}, user_id="alice")
    first_path = Path(manager.history_folder) / "shared_history_000001.sqlite"
    conn = sqlite3.connect(first_path)
    conn.execute(
        """
        INSERT INTO history_metadata (key, value)
        VALUES ('created_at', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        ("2000-01-01T00:00:00+00:00",),
    )
    conn.commit()
    conn.close()

    manager.add_blocks({"response": "second"}, user_id="alice")

    assert [path.name for path in sorted(Path(manager.history_folder).glob("*.sqlite"))] == [
        "shared_history_000001.sqlite",
        "shared_history_000002.sqlite",
    ]
    assert [msg["message"][0]["content"]["response"] for msg in manager.get_messages("alice")] == [
        "first",
        "second",
    ]


def test_shared_history_clear_is_isolated_by_user(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), history_mode="shared")
    manager.define_variable("mood", type="string", default="neutral")

    manager.set_variable("mood", "happy", user_id="alice")
    manager.set_variable("mood", "focused", user_id="bob")
    manager.clear_message_history("alice")

    assert manager.get_variable("mood", user_id="alice") == "neutral"
    assert manager.get_variable("mood", user_id="bob") == "focused"


def test_shared_history_export_import_round_trip_filters_to_user(workspace_tmp_path):
    first = AgentSystemManager(
        base_directory=str(workspace_tmp_path / "first"),
        history_mode="shared",
    )
    first.add_blocks({"response": "alice-only"}, user_id="alice")
    first.add_blocks({"response": "bob-only"}, user_id="bob")

    exported = first.export_history("alice")
    second = AgentSystemManager(
        base_directory=str(workspace_tmp_path / "second"),
        history_mode="shared",
    )
    second.import_history("alice", exported)

    assert [msg["message"][0]["content"]["response"] for msg in second.get_messages("alice")] == [
        "alice-only"
    ]
    assert second.get_messages("bob") == []


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
