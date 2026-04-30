from __future__ import annotations

import types

from mas import AgentSystemManager
import mas.mas as mas_module


class _Response:
    def __init__(self, *, content=b"", data=None, status_code=200):
        self.content = content
        self._data = data or {}
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


def test_google_image_url_fetch_uses_agent_timeout(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), timeout=7)
    name = manager.create_agent("vision", models=[{"provider": "google", "model": "gemini"}])
    agent = manager.agents[name]
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["timeout"] = kwargs.get("timeout")
        return _Response(content=b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(mas_module.requests, "get", fake_get)

    agent._provider_format_messages(
        "google",
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "content": {"kind": "url", "url": "https://example.test/img.png"}}
                ],
            }
        ],
    )

    assert seen == {"url": "https://example.test/img.png", "timeout": 7}


def test_elevenlabs_voice_lookup_uses_manager_timeout(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), timeout=9)
    manager.api_keys["elevenlabs"] = "eleven-key"
    seen = {}

    def fake_get(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return _Response(data={"voices": [{"name": "Rachel", "voice_id": "voice-1"}]})

    def fake_post(url, **kwargs):
        return _Response(content=b"mp3")

    monkeypatch.setattr(mas_module.requests, "get", fake_get)
    monkeypatch.setattr(mas_module.requests, "post", fake_post)
    monkeypatch.setattr(manager, "_save_tts_file", lambda audio_bytes: "file:/tmp/audio.mp3")

    assert manager._generate_tts_elevenlabs("hello", "Rachel", "model") == "file:/tmp/audio.mp3"
    assert seen["timeout"] == 9
