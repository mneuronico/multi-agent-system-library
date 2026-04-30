from __future__ import annotations

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
