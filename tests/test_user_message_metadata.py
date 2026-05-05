from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from mas.bots import Bot
from mas.manager import AgentSystemManager


class _Payload:
    def to_dict(self):
        return {
            "message_id": 123,
            "chat": {"id": "chat-1"},
            "text": "hello",
        }


class _Media:
    file_id = "file-1"
    file_unique_id = "unique-1"
    mime_type = "image/png"
    width = 640
    height = 480


class _Bot(Bot):
    async def _parse_payload(self, payload):
        return payload

    async def _send_blocks(self, user_id, blocks, original_message):
        return None

    async def _download_media_and_save(self, user_id, media_info):
        return "file:dummy"

    async def _send_log_files(self, user_id, files_to_send, original_message):
        return 0


def test_bot_constructor_does_not_require_an_asyncio_event_loop(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    def fail_lock_creation(*args, **kwargs):
        raise AssertionError("Bot construction should not create an asyncio.Lock")

    monkeypatch.setattr(asyncio, "Lock", fail_lock_creation)

    _Bot(manager=manager)


def test_bot_user_message_metadata_is_attached_and_persisted(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    bot = _Bot(manager=manager)
    timestamp = datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)

    blocks = asyncio.run(
        bot._build_mas_blocks(
            {
                "user_id": "chat-1",
                "message_type": "text",
                "text": "hello",
                "media_info": _Media(),
                "is_voice_note": False,
                "timestamp": timestamp,
                "metadata": {"transport": "webhook", "retry": 0},
                "original_payload": _Payload(),
            }
        )
    )

    metadata = blocks[0]["metadata"]["user_message"]
    assert metadata["channel"] == "_Bot"
    assert metadata["user_id"] == "chat-1"
    assert metadata["message_type"] == "text"
    assert metadata["timestamp"] == "2026-05-02T12:30:00+00:00"
    assert metadata["metadata"] == {"transport": "webhook", "retry": 0}
    assert metadata["media_info"]["file_id"] == "file-1"
    assert metadata["original_payload"]["message_id"] == 123

    manager.add_blocks(blocks, role="user", msg_type="user", user_id="chat-1")
    stored = manager.get_messages("chat-1")[0]["message"][0]["metadata"]["user_message"]

    assert stored == metadata
