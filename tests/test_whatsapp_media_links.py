"""WhatsApp media that arrives as a direct link instead of a Meta media id.

A relay in front of the webhook (for example one that translates another WhatsApp provider's
events into Meta's format) can only pass the media's URL along: there is no Meta media id to
look up. The bot downloads the link directly, over HTTPS only and without the access token.
"""
from __future__ import annotations

import asyncio
import os

import pytest

import mas.bots as bots_module
from mas.bots import MediaTooLargeError, WhatsappBot
from mas.manager import AgentSystemManager

LINK = "https://media.provider.example/files/abc.jpg?signature=xyz"
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 300


class _Stream:
    def __init__(self, chunks, headers=None):
        self.chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        yield from self.chunks


def _fake_get(monkeypatch, chunks, headers=None):
    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False, **kwargs):
        calls.append({"url": url, "headers": headers, "stream": stream})
        return _Stream(chunks, headers_for_response)

    headers_for_response = headers
    monkeypatch.setattr(bots_module.requests, "get", fake_get)
    return calls


def _bot(tmp_path, **kwargs):
    manager = AgentSystemManager(base_directory=str(tmp_path))
    return WhatsappBot(
        manager=manager, whatsapp_token="secret-token", phone_number_id="p", webhook_verify_token="v", **kwargs
    )


def _image_message(image):
    return {"from": "5491100000000", "id": "wamid.1", "timestamp": "1763474400", "type": "image", "image": image}


def test_media_with_a_link_and_no_id_is_parsed(workspace_tmp_path):
    bot = _bot(workspace_tmp_path)
    parsed = asyncio.run(bot._parse_payload(_image_message(
        {"link": LINK, "mime_type": "image/jpeg", "caption": "qué hago con esto?"}
    )))

    assert parsed["message_type"] == "image"
    assert parsed["text"] == "qué hago con esto?"
    assert parsed["media_info"] == {"link": LINK, "mime_type": "image/jpeg"}


def test_media_without_id_or_link_is_still_ignored(workspace_tmp_path):
    bot = _bot(workspace_tmp_path)
    assert asyncio.run(bot._parse_payload(_image_message({"mime_type": "image/jpeg"}))) is None


def test_a_link_is_downloaded_directly_without_the_access_token(workspace_tmp_path, monkeypatch):
    bot = _bot(workspace_tmp_path)
    calls = _fake_get(monkeypatch, [JPEG[:100], JPEG[100:]])

    ref = asyncio.run(bot._download_media_and_save("5491", {"link": LINK, "mime_type": "image/jpeg"}))

    assert calls == [{"url": LINK, "headers": None, "stream": True}]
    assert ref.startswith("file:") and ref.endswith(".jpeg")
    with open(ref[5:], "rb") as fh:
        assert fh.read() == JPEG


def test_a_media_id_still_uses_the_graph_api_even_with_a_link(workspace_tmp_path, monkeypatch):
    bot = _bot(workspace_tmp_path)
    calls = []

    class _Json:
        def raise_for_status(self):
            pass

        def json(self):
            return {"url": "https://lookaside.example/m"}

    def fake_get(url, headers=None, timeout=None, stream=False, **kwargs):
        calls.append((url, bool(headers)))
        return _Stream([JPEG]) if stream else _Json()

    monkeypatch.setattr(bots_module.requests, "get", fake_get)
    asyncio.run(bot._download_media_and_save("5491", {"id": "media-1", "link": LINK}))

    assert [url for url, _ in calls] == [f"{bot.graph_base_url}/media-1", "https://lookaside.example/m"]
    assert all(with_token for _, with_token in calls)


def test_links_that_are_not_https_are_refused(workspace_tmp_path, monkeypatch):
    bot = _bot(workspace_tmp_path)
    calls = _fake_get(monkeypatch, [JPEG])

    for link in ("http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://host/x.jpg"):
        with pytest.raises(ValueError):
            asyncio.run(bot._download_media_and_save("5491", {"link": link}))
    assert calls == []


def test_the_size_cap_applies_to_links(workspace_tmp_path, monkeypatch):
    bot = _bot(workspace_tmp_path, max_media_bytes=150)
    temp_paths = []
    original_new_temp = bot._new_temp_file
    monkeypatch.setattr(bot, "_new_temp_file", lambda: temp_paths.append(original_new_temp()) or temp_paths[-1])

    _fake_get(monkeypatch, [JPEG], headers={"Content-Length": "500"})
    with pytest.raises(MediaTooLargeError):
        asyncio.run(bot._download_media_and_save("5491", {"link": LINK}))

    _fake_get(monkeypatch, [b"x" * 100] * 5)
    with pytest.raises(MediaTooLargeError):
        asyncio.run(bot._download_media_and_save("5491", {"link": LINK}))

    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)


def test_the_link_is_not_kept_in_the_message_metadata(workspace_tmp_path):
    bot = _bot(workspace_tmp_path)
    parsed = asyncio.run(bot._parse_payload(_image_message({"link": LINK, "mime_type": "image/jpeg"})))

    metadata = bot._build_user_message_metadata(parsed)

    assert metadata["media_info"] == {"mime_type": "image/jpeg"}
    assert LINK not in repr(metadata)


def test_an_image_link_reaches_the_conversation_as_an_image_block(workspace_tmp_path, monkeypatch):
    bot = _bot(workspace_tmp_path)
    _fake_get(monkeypatch, [JPEG])
    parsed = asyncio.run(bot._parse_payload(_image_message({"link": LINK, "mime_type": "image/jpeg", "caption": "hola"})))

    blocks = asyncio.run(bot._build_mas_blocks(parsed))

    kinds = [block["type"] for block in blocks]
    assert kinds == ["text", "image"]
    image = blocks[1]["content"]
    assert image["kind"] == "file" and image["path"].startswith("file:") and image["mime_type"] == "image/jpeg"
