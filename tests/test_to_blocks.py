import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mas.mas import AgentSystemManager


def create_manager(tmp_path):
    return AgentSystemManager(
        base_directory=str(tmp_path),
        history_folder=str(tmp_path / "history"),
        files_folder=str(tmp_path / "files")
    )

def roundtrip(manager, value):
    blocks = manager._to_blocks(value, user_id="u1")
    parsed = manager._blocks_as_tool_input(blocks)
    return manager._load_files_in_dict(parsed)

def test_dict_with_bytes(tmp_path):
    manager = create_manager(tmp_path)
    original = {"foo": b"bar"}
    result = roundtrip(manager, original)
    assert result == original

    text = manager._first_text_block(manager._to_blocks(original, user_id="u1"))
    data = json.loads(text)
    assert isinstance(data["foo"], str) and data["foo"].startswith("file:")
    assert os.path.exists(data["foo"][5:])

def test_list_with_bytes(tmp_path):
    manager = create_manager(tmp_path)
    original = [b"abc", {"nested": b"def"}]
    result = roundtrip(manager, original)
    assert result == original

    text = manager._first_text_block(manager._to_blocks(original, user_id="u1"))
    data = json.loads(text)
    assert data[0].startswith("file:")
    assert os.path.exists(data[0][5:])
    assert data[1]["nested"].startswith("file:")
    assert os.path.exists(data[1]["nested"][5:])
