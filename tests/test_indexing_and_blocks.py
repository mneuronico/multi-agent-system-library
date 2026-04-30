from __future__ import annotations

import types

import pytest

from mas.mas import Agent, AgentSystemManager, Process


def test_agent_handle_index_rejects_out_of_range_positive_index():
    rows = [("user", "{}", 1, "user", "2026-01-01")]
    agent = types.SimpleNamespace()

    assert Agent._handle_index(agent, 0, rows) == rows
    assert Agent._handle_index(agent, -1, rows) == rows
    assert Agent._handle_index(agent, 1, rows) == []


def test_process_handle_index_rejects_out_of_range_positive_index():
    rows = [("user", "{}", 1, "user", "2026-01-01")]
    process = types.SimpleNamespace()

    assert Process._handle_index(process, 0, rows) == rows
    assert Process._handle_index(process, -1, rows) == rows
    assert Process._handle_index(process, 1, rows) == []


@pytest.mark.parametrize(
    "data",
    [
        b"%PDF-1.7\nnot an image",
        b"ID3\x04\x00\x00audio bytes",
        b"plain bytes",
    ],
)
def test_non_image_bytes_are_not_classified_as_images(data):
    manager = types.SimpleNamespace()
    assert AgentSystemManager._is_image_bytes(manager, data) is False


def test_png_bytes_are_classified_as_image_bytes():
    manager = types.SimpleNamespace()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    assert AgentSystemManager._is_image_bytes(manager, png) is True
