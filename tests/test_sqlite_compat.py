from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from mas import AgentSystemManager


# AWS Lambda's python3.9 runtime (MAWS default) ships SQLite 3.7.17. Local and CI SQLite are
# much newer, so emulate the old parser by rejecting syntax it does not know (UPSERT needs 3.24,
# RETURNING needs 3.35) with the same error Lambda raises.
_NEWER_SQLITE_SYNTAX = re.compile(r"ON\s+CONFLICT\s*(\(|DO\b)|\bDO\s+UPDATE\b|\bRETURNING\b", re.IGNORECASE)


def _reject_newer_sqlite_syntax(sql):
    if isinstance(sql, str) and _NEWER_SQLITE_SYNTAX.search(sql):
        raise sqlite3.OperationalError('near "ON": syntax error')


class _OldSQLiteCursor(sqlite3.Cursor):
    def execute(self, sql, *args):
        _reject_newer_sqlite_syntax(sql)
        return super().execute(sql, *args)


class _OldSQLiteConnection(sqlite3.Connection):
    def cursor(self, factory=_OldSQLiteCursor):
        return super().cursor(factory)

    def execute(self, sql, *args):
        _reject_newer_sqlite_syntax(sql)
        return super().execute(sql, *args)


@pytest.fixture
def lambda_py39_sqlite(monkeypatch):
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("factory", _OldSQLiteConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)


def _metadata(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute("SELECT key, value FROM history_metadata").fetchall())
    finally:
        conn.close()


def test_library_sql_avoids_syntax_missing_from_lambda_sqlite():
    # Catches new UPSERT/RETURNING statements even on code paths the tests below do not run.
    root = Path(__file__).resolve().parents[1]
    sql_keywords = re.compile(r"ON\s+CONFLICT\s*(\(|DO\b)|\bDO\s+UPDATE\b|\bRETURNING\b")
    offenders = []
    for package in ("mas", "maws"):
        for path in sorted((root / package).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for match in sql_keywords.finditer(source):
                offenders.append(f"{path.relative_to(root)}:{source.count(chr(10), 0, match.start()) + 1}")

    assert offenders == []


def test_emulated_old_sqlite_rejects_upsert(workspace_tmp_path, lambda_py39_sqlite):
    conn = sqlite3.connect(workspace_tmp_path / "probe.sqlite")
    conn.execute("CREATE TABLE t (key TEXT PRIMARY KEY, value TEXT)")
    with pytest.raises(sqlite3.OperationalError, match='near "ON"'):
        conn.cursor().execute(
            "INSERT INTO t (key, value) VALUES ('a', '1') ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
    conn.close()


def test_shared_history_rotation_works_on_old_sqlite(workspace_tmp_path, lambda_py39_sqlite):
    manager = AgentSystemManager(
        base_directory=str(workspace_tmp_path),
        history_mode="shared",
        history_max_messages=3,
    )

    for text in ("a1", "b1", "a2", "b2", "a3"):
        manager.add_blocks({"response": text}, user_id="alice" if text[0] == "a" else "bob")

    paths = sorted(Path(manager.history_folder).glob("*.sqlite"))
    assert [path.name for path in paths] == [
        "shared_history_000001.sqlite",
        "shared_history_000002.sqlite",
    ]
    assert "closed_at" in _metadata(paths[0])
    assert "closed_at" not in _metadata(paths[1])
    assert [msg["message"][0]["content"]["response"] for msg in manager.get_messages("alice")] == [
        "a1",
        "a2",
        "a3",
    ]


def test_history_metadata_value_is_replaced_on_old_sqlite(workspace_tmp_path, lambda_py39_sqlite):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), history_mode="shared")
    manager.add_blocks({"response": "hi"}, user_id="alice")
    db_path = Path(manager.history_folder) / "shared_history_000001.sqlite"
    conn = manager._connect_history_db(str(db_path), manager._shared_history_db_key(str(db_path)))

    manager._set_history_metadata(conn, "closed_at", "2026-01-01T00:00:00+00:00")
    manager._set_history_metadata(conn, "closed_at", "2026-02-01T00:00:00+00:00")

    rows = conn.execute("SELECT value FROM history_metadata WHERE key = 'closed_at'").fetchall()
    assert rows == [("2026-02-01T00:00:00+00:00",)]


def test_shared_history_clear_resets_metadata_on_old_sqlite(workspace_tmp_path, lambda_py39_sqlite):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), history_mode="shared")
    manager.add_blocks({"response": "only alice"}, user_id="alice")
    db_path = Path(manager.history_folder) / "shared_history_000001.sqlite"
    conn = manager._connect_history_db(str(db_path), manager._shared_history_db_key(str(db_path)))
    manager._set_history_metadata(conn, "created_at", "2000-01-01T00:00:00+00:00")
    manager._set_history_metadata(conn, "closed_at", "2000-01-02T00:00:00+00:00")

    manager.clear_message_history("alice")

    metadata = _metadata(db_path)
    assert "closed_at" not in metadata
    assert metadata["created_at"] != "2000-01-01T00:00:00+00:00"
    assert manager.get_messages("alice") == []


def test_clear_global_history_resets_metadata_on_old_sqlite(workspace_tmp_path, lambda_py39_sqlite):
    manager = AgentSystemManager(
        base_directory=str(workspace_tmp_path),
        history_mode="shared",
        history_max_messages=1,
    )
    manager.add_blocks({"response": "first"}, user_id="alice")
    manager.add_blocks({"response": "second"}, user_id="alice")
    paths = sorted(Path(manager.history_folder).glob("*.sqlite"))
    assert len(paths) == 2

    assert manager.clear_global_history() == 2

    for path in paths:
        metadata = _metadata(path)
        assert "closed_at" not in metadata
        assert "created_at" in metadata
