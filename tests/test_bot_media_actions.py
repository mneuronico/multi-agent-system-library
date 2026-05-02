from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mas.bots import TelegramBot, WhatsappBot


class _Manager:
    on_update = None
    on_complete = None
    admin_user_id = None

    def get_key(self, name):
        keys = {
            "telegram_token": "telegram-token",
            "whatsapp_token": "whatsapp-token",
            "whatsapp_phone_number_id": "phone-id",
            "webhook_verify_token": "verify-token",
        }
        return keys.get(name)

    def _to_blocks(self, value, user_id=None):
        if isinstance(value, list):
            return value
        return [{"type": "text", "content": value}]

    def _block_to_plain_text(self, block):
        content = block.get("content")
        if isinstance(content, dict):
            return content.get("response", "")
        return str(content)

    def save_file(self, value, user_id=None):
        return "file:/tmp/saved-media.bin"

    def set_current_user(self, user_id):
        self.current_user_id = user_id


def _telegram_bot():
    return TelegramBot(manager=_Manager())


def _whatsapp_bot():
    return WhatsappBot(manager=_Manager())


def test_telegram_parse_payload_supports_video_document_and_sticker():
    bot = _telegram_bot()
    chat = SimpleNamespace(id=123)

    for attr, expected in [
        ("video", "video"),
        ("document", "document"),
        ("sticker", "sticker"),
    ]:
        media = SimpleNamespace(
            file_id=f"{attr}-file",
            file_name=f"{attr}.bin",
            mime_type="application/octet-stream",
        )
        message = SimpleNamespace(
            chat=chat,
            text=None,
            caption=f"{attr} caption",
            photo=[],
            voice=None,
            audio=None,
            video=None,
            document=None,
            sticker=None,
            date=None,
        )
        setattr(message, attr, media)
        update = SimpleNamespace(message=message)

        parsed = asyncio.run(bot._parse_payload(update))

        assert parsed["user_id"] == "123"
        assert parsed["message_type"] == expected
        assert parsed["text"] == f"{attr} caption"
        assert parsed["media_info"] is media


def test_telegram_send_blocks_supports_video_document_and_sticker(workspace_tmp_path):
    bot = _telegram_bot()
    calls = []

    class _Message:
        async def reply_video(self, media, **kwargs):
            calls.append(("video", media.read(), kwargs))

        async def reply_document(self, media, **kwargs):
            calls.append(("document", media.read(), kwargs))

        async def reply_sticker(self, media, **kwargs):
            calls.append(("sticker", media, kwargs))

    video = workspace_tmp_path / "clip.mp4"
    document = workspace_tmp_path / "report.pdf"
    video.write_bytes(b"video-bytes")
    document.write_bytes(b"document-bytes")

    blocks = [
        {"type": "video", "content": {"path": f"file:{video}", "caption": "clip"}},
        {
            "type": "document",
            "content": {
                "path": f"file:{document}",
                "caption": "report",
                "filename": "report.pdf",
            },
        },
        {"type": "sticker", "content": {"url": "https://example.com/sticker.webp"}},
    ]
    original = {"original_payload": SimpleNamespace(message=_Message())}

    asyncio.run(bot._send_blocks("123", blocks, original))

    assert calls[0] == ("video", b"video-bytes", {"caption": "clip"})
    assert calls[1] == (
        "document",
        b"document-bytes",
        {"caption": "report", "filename": "report.pdf"},
    )
    assert calls[2] == ("sticker", "https://example.com/sticker.webp", {})


def test_telegram_reaction_uses_platform_method():
    bot = _telegram_bot()
    calls = []
    reaction_emoji = "\U0001f44d"

    class _Api:
        async def set_message_reaction(self, **kwargs):
            calls.append(kwargs)
            return True

    bot.bot = _Api()

    result = asyncio.run(bot.react_to_message("123", "456", reaction_emoji))

    assert result is True
    assert calls == [{
        "chat_id": "123",
        "message_id": 456,
        "reaction": [{"type": "emoji", "emoji": reaction_emoji}],
    }]


def test_whatsapp_parse_and_build_blocks_supports_media_and_reactions(monkeypatch):
    bot = _whatsapp_bot()
    reaction_emoji = "\U0001f44d"

    async def fake_download(user_id, media_info):
        return f"file:/tmp/{media_info['id']}.bin"

    monkeypatch.setattr(bot, "_download_media_and_save", fake_download)

    document_payload = {
        "from": "54911",
        "type": "document",
        "timestamp": "1700000000",
        "document": {
            "id": "doc-id",
            "filename": "file.pdf",
            "mime_type": "application/pdf",
            "caption": "document caption",
        },
    }
    parsed = asyncio.run(bot._parse_payload(document_payload))
    blocks = asyncio.run(bot._build_mas_blocks(parsed))

    assert parsed["message_type"] == "document"
    assert parsed["media_info"]["filename"] == "file.pdf"
    assert blocks[0]["content"] == {"response": "document caption"}
    assert blocks[1]["type"] == "document"
    assert blocks[1]["content"]["filename"] == "file.pdf"
    assert blocks[0]["metadata"]["user_message"]["media_info"]["id"] == "doc-id"

    reaction_payload = {
        "from": "54911",
        "type": "reaction",
        "timestamp": "1700000001",
        "reaction": {"message_id": "wamid.1", "emoji": reaction_emoji},
    }
    reaction = asyncio.run(bot._parse_payload(reaction_payload))
    reaction_blocks = asyncio.run(bot._build_mas_blocks(reaction))

    assert reaction["message_type"] == "reaction"
    assert reaction_blocks[0]["content"]["message_id"] == "wamid.1"
    assert reaction_blocks[0]["metadata"]["user_message"]["media_info"]["emoji"] == reaction_emoji


def test_whatsapp_send_blocks_and_reactions_use_cloud_api_payloads(monkeypatch, workspace_tmp_path):
    bot = _whatsapp_bot()
    sent_payloads = []
    reaction_emoji = "\U0001f44d"

    async def fake_upload(path):
        return f"uploaded-{path.split('/')[-1].split(chr(92))[-1]}"

    class _Response:
        ok = True
        status_code = 200
        text = "{}"

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        sent_payloads.append(json)
        return _Response()

    monkeypatch.setattr(bot, "_upload_media", fake_upload)
    monkeypatch.setattr("mas.bots.requests.post", fake_post)

    video = workspace_tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    blocks = [
        {"type": "video", "content": {"path": f"file:{video}", "caption": "clip"}},
        {
            "type": "document",
            "content": {
                "url": "https://example.com/file.pdf",
                "caption": "report",
                "filename": "file.pdf",
            },
        },
        {"type": "sticker", "content": {"id": "sticker-media-id"}},
    ]

    asyncio.run(bot._send_blocks("54911", blocks, {"original_payload": {}}))
    asyncio.run(bot.react_to_message("54911", "wamid.1", reaction_emoji))

    assert sent_payloads[0]["type"] == "video"
    assert sent_payloads[0]["video"] == {"id": "uploaded-clip.mp4", "caption": "clip"}
    assert sent_payloads[1]["type"] == "document"
    assert sent_payloads[1]["document"] == {
        "link": "https://example.com/file.pdf",
        "caption": "report",
        "filename": "file.pdf",
    }
    assert sent_payloads[2]["type"] == "sticker"
    assert sent_payloads[2]["sticker"] == {"id": "sticker-media-id"}
    assert sent_payloads[3]["type"] == "reaction"
    assert sent_payloads[3]["reaction"] == {"message_id": "wamid.1", "emoji": reaction_emoji}
