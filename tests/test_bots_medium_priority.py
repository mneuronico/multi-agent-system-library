"""Bot pipeline, medium-priority fixes.

Long texts, duplicate deliveries, ordered delivery, admin resets during runs, Telegram groups,
the outbound file allowlist, large media, Telegram/WhatsApp consistency, privacy of stored
metadata, logging and the WhatsApp API version.

Everything runs offline against fakes; nothing calls Telegram, WhatsApp or a model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import telegram

import mas.bots as bots_module
from mas._shared import logger as library_logger
from mas.bots import (
    DEFAULT_MAX_MEDIA_BYTES,
    DEFAULT_WHATSAPP_API_VERSION,
    Bot,
    MediaTooLargeError,
    TelegramBot,
    WhatsappAPIError,
    WhatsappBot,
    _split_text,
    _split_text_once,
    _text_length,
)
from mas.manager import AgentSystemManager


REAL_PTB = getattr(telegram, "__file__", None) is not None
BOT_LOGGER = bots_module.logger.name
TOO_LARGE_20MB = "Sorry, that file is too large for me to process (the limit is 20 MB)."


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Manager:
    on_update = None
    on_complete = None

    def __init__(self, run=None, admin_user_id=None, files_folder=None, emit_update=False, final_message=None):
        self._run_hook = run
        self.admin_user_id = admin_user_id
        self.files_folder = files_folder
        self.emit_update = emit_update
        self.final_message = final_message
        self.runs = []
        self.events = []
        self._lock = threading.Lock()

    def get_key(self, name):
        return {
            "telegram_token": "123456:telegram-token",
            "whatsapp_token": "whatsapp-token",
            "whatsapp_phone_number_id": "phone-id",
            "webhook_verify_token": "verify-token",
        }.get(name)

    def _to_blocks(self, value, user_id=None):
        if isinstance(value, list):
            return value
        return [{"type": "text", "content": value}]

    def _block_to_plain_text(self, block):
        if block.get("type") != "text":
            return ""
        content = block.get("content")
        if isinstance(content, dict):
            return content.get("response", "")
        return str(content)

    def save_file(self, value, user_id=None):
        return "file:/tmp/saved-media.bin"

    def set_current_user(self, user_id):
        pass

    def _invoke_callback(self, callback, messages, manager, params):
        return callback(messages, manager, params)

    def event(self, *item):
        with self._lock:
            self.events.append(item)

    def run(self, input=None, user_id=None, on_update=None, on_update_params=None,
            on_complete=None, on_complete_params=None, **kwargs):
        text = input[0]["content"]["response"]
        with self._lock:
            self.runs.append((user_id, text))
        self.event("run", user_id, text)
        if self._run_hook:
            self._run_hook(user_id, text)
        if on_update and self.emit_update:
            self._invoke_callback(on_update, [{"message": "progress"}], self, on_update_params)
        if on_complete:
            message = self.final_message if self.final_message is not None else f"echo {text}"
            self._invoke_callback(on_complete, [{"message": message}], self, on_complete_params)
        self.event("run_end", user_id, text)
        return []

    def stt(self, file_path, provider=None, model=None, user_id=None):
        return "transcript"

    def clear_message_history(self, user_id=None):
        self.event("clear_user", user_id)

    def clear_global_history(self):
        self.event("clear_all")
        return 3

    def reset_system(self):
        self.event("reset")


class _RecordingBot(Bot):
    def __init__(self, manager, send_delays=None, **kwargs):
        super().__init__(manager=manager, **kwargs)
        self.sent = []
        self.send_delays = send_delays or {}

    async def _parse_payload(self, payload):
        return payload

    async def _send_blocks(self, user_id, blocks, original_message):
        for block in blocks:
            text = self.manager._block_to_plain_text(block)
            delay = self.send_delays.get(text)
            if delay:
                await asyncio.sleep(delay)
            self.sent.append((user_id, text))
            self.manager.event("sent", user_id, text)

    async def _download_media_and_save(self, user_id, media_info):
        return "file:/tmp/media.bin"

    async def _send_log_files(self, user_id, files_to_send, original_message):
        return 0


def _msg(user_id, text, *, message_id=None, sender_id=None):
    return {
        "user_id": str(user_id),
        "sender_id": None if sender_id is None else str(sender_id),
        "message_id": message_id,
        "message_type": "text",
        "text": text,
        "media_info": None,
        "is_voice_note": False,
        "timestamp": datetime.now(timezone.utc),
        "original_payload": {"text": text},
    }


async def _wait_thread_event(event, timeout=5):
    assert await asyncio.to_thread(event.wait, timeout)


def _assert_idle(bot):
    assert bot._user_locks == {}
    assert bot._send_queues == {}
    assert bot._active_turns == 0
    assert bot._maintenance_requests == 0


class _TgMessage:
    """Duck-typed telegram.Message: records every reply."""

    from_user = None

    def __init__(self, chat_id, text, calls, *, sender_id=None, message_id=None, chat_type="private",
                 caption=None, photo=None, voice=None, audio=None, video=None, document=None, sticker=None):
        self.chat = SimpleNamespace(id=chat_id, type=chat_type)
        if sender_id is not None:
            self.from_user = SimpleNamespace(id=sender_id, first_name="Ada", username="ada_lovelace")
        self.message_id = message_id
        self.text = text
        self.caption = caption
        self.photo = photo or []
        self.voice = voice
        self.audio = audio
        self.video = video
        self.document = document
        self.sticker = sticker
        self.date = datetime.now(timezone.utc)
        self._calls = calls

    def _record(self, kind, media, kwargs):
        if hasattr(media, "read"):
            media = ("file", media.read())
        self._calls.append((kind, media, kwargs))

    async def reply_text(self, text, **kwargs):
        self._calls.append(("text", text, kwargs))

    async def reply_photo(self, media, **kwargs):
        self._record("photo", media, kwargs)

    async def reply_audio(self, media, **kwargs):
        self._record("audio", media, kwargs)

    async def reply_voice(self, media, **kwargs):
        self._record("voice", media, kwargs)

    async def reply_video(self, media, **kwargs):
        self._record("video", media, kwargs)

    async def reply_document(self, media, **kwargs):
        self._record("document", media, kwargs)

    async def reply_sticker(self, media, **kwargs):
        self._record("sticker", media, kwargs)


def _tg_update(chat_id, text, calls, **kwargs):
    return SimpleNamespace(update_id=kwargs.get("message_id"), message=_TgMessage(chat_id, text, calls, **kwargs))


def _texts(calls):
    return [value for kind, value, _ in calls if kind == "text"]


class _Response:
    def __init__(self, ok=True, status_code=200, data=None):
        self.ok = ok
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


def _whatsapp_bot(manager=None, **kwargs):
    return WhatsappBot(manager=manager or _Manager(), **kwargs)


def _capture_whatsapp_posts(monkeypatch, response_for=None):
    posts = []

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        posts.append({"url": url, "json": json, **kwargs})
        if response_for:
            return response_for(url, json)
        return _Response(data={"messages": [{"id": "wamid.out"}]})

    monkeypatch.setattr(bots_module.requests, "post", fake_post)
    return posts


def _wa_message(sender, message_id, body):
    return {"from": sender, "id": message_id, "type": "text", "timestamp": str(int(time.time())),
            "text": {"body": body}}


def _wa_webhook(*messages):
    return {"entry": [{"changes": [{"value": {"messages": list(messages)}}]}]}


# ---------------------------------------------------------------------------
# 1. Long texts are split instead of lost
# ---------------------------------------------------------------------------

def test_split_prefers_paragraphs_then_lines_then_words():
    paragraph = "word " * 30  # 150 chars
    text = (paragraph.strip() + "\n\n") * 3
    chunks = _split_text(text, 200)
    assert chunks == [paragraph.strip()] * 3

    lines = ("x" * 60 + "\n") * 5
    assert _split_text(lines, 130) == ["x" * 60 + "\n" + "x" * 60] * 2 + ["x" * 60]

    words = " ".join(["abcdefghi"] * 30)
    chunks = _split_text(words, 50)
    assert all(_text_length(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == words.split()


def test_split_without_boundaries_cuts_hard_and_counts_emoji_as_two():
    assert _split_text("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]

    emoji = "\U0001F600"
    chunks = _split_text(emoji * 7, 4)
    assert chunks == [emoji * 2] * 3 + [emoji]
    assert all(_text_length(c) <= 4 for c in chunks)


def test_split_keeps_short_text_untouched_and_drops_blank_chunks():
    assert _split_text("  hello  ", 4096) == ["  hello  "]
    assert _split_text("", 10) == []
    assert _split_text("   ", 10) == []
    head, rest = _split_text_once("short", 10)
    assert (head, rest) == ("short", "")


def test_telegram_long_reply_is_sent_as_ordered_chunks():
    bot = TelegramBot(manager=_Manager())
    calls = []
    paragraphs = [f"Paragraph {i}: " + "lorem ipsum " * 150 for i in range(5)]  # ~1800 chars each
    text = "\n\n".join(paragraphs)
    update = _tg_update(42, "hi", calls)

    asyncio.run(bot._send_blocks("42", [{"type": "text", "content": {"response": text}}],
                                 {"original_payload": update}))

    sent = _texts(calls)
    assert len(sent) > 1
    assert all(_text_length(chunk) <= 4096 for chunk in sent)
    assert " ".join(sent).split() == text.split()
    assert sent[0].startswith("Paragraph 0")


def test_whatsapp_long_text_is_split_and_reply_context_goes_on_the_first_chunk(monkeypatch):
    bot = _whatsapp_bot()
    posts = _capture_whatsapp_posts(monkeypatch)
    text = "\n".join(f"line {i} " + "y" * 90 for i in range(100))  # ~10 000 chars

    asyncio.run(bot.send_text("5491", text, reply_to_message_id="wamid.in"))

    bodies = [p["json"]["text"]["body"] for p in posts]
    assert len(bodies) == 3
    assert all(_text_length(body) <= 4096 for body in bodies)
    assert "\n".join(bodies) == text
    assert posts[0]["json"]["context"] == {"message_id": "wamid.in"}
    assert all("context" not in p["json"] for p in posts[1:])


def test_long_captions_are_cut_at_1024_and_the_rest_follows_as_text(monkeypatch):
    bot = _whatsapp_bot()
    posts = _capture_whatsapp_posts(monkeypatch)

    asyncio.run(bot.send_document("5491", "https://example.com/r.pdf", caption="x" * 1500))

    assert posts[0]["json"]["type"] == "document"
    assert posts[0]["json"]["document"]["caption"] == "x" * 1024
    assert posts[1]["json"]["text"]["body"] == "x" * 476

    tg = TelegramBot(manager=_Manager())
    calls = []
    caption = "word " * 400
    block = {"type": "document", "content": {"url": "https://example.com/r.pdf", "caption": caption}}
    asyncio.run(tg._send_blocks("42", [block], {"original_payload": _tg_update(42, "hi", calls)}))

    kind, media, kwargs = calls[0]
    assert (kind, media) == ("document", "https://example.com/r.pdf")
    assert _text_length(kwargs["caption"]) <= 1024
    assert (kwargs["caption"] + " " + " ".join(_texts(calls))).split() == caption.split()


def test_send_failures_never_log_the_message_body(monkeypatch, caplog):
    bot = _whatsapp_bot()
    error = {"error": {"message": "(#131030) Recipient phone number not in allowed list", "code": 131030}}
    _capture_whatsapp_posts(monkeypatch, lambda url, payload: _Response(ok=False, status_code=400, data=error))
    secret = "my card PIN is 4321"

    with caplog.at_level(logging.DEBUG, logger=BOT_LOGGER):
        asyncio.run(bot._send_blocks("5491", [{"type": "text", "content": {"response": secret}}], {}))

    messages = [r.getMessage() for r in caplog.records]
    assert any("131030" in m and "HTTP 400" in m for m in messages)
    assert not any(secret in m for m in messages)

    tg = TelegramBot(manager=_Manager())

    class _Failing(_TgMessage):
        async def reply_text(self, text, **kwargs):
            raise RuntimeError("Message is too long")

    caplog.clear()
    update = SimpleNamespace(message=_Failing(42, "hi", []))
    with caplog.at_level(logging.DEBUG, logger=BOT_LOGGER):
        asyncio.run(tg._send_blocks("42", [{"type": "text", "content": {"response": secret}}],
                                    {"original_payload": update}))

    messages = [r.getMessage() for r in caplog.records]
    assert any("Message is too long" in m for m in messages)
    assert not any(secret in m for m in messages)


# ---------------------------------------------------------------------------
# 2. Duplicate deliveries are processed once
# ---------------------------------------------------------------------------

def _whatsapp_bot_with_fake_sends(manager, **kwargs):
    bot = WhatsappBot(manager=manager, ensure_delivery=True, **kwargs)
    sent = []

    async def fake_send_text(user_id, text, **_kwargs):
        sent.append((user_id, text))

    bot.send_text = fake_send_text
    return bot, sent


def test_whatsapp_redelivered_wamid_runs_and_replies_once():
    manager = _Manager()
    bot, sent = _whatsapp_bot_with_fake_sends(manager)
    webhook = _wa_webhook(_wa_message("100", "wamid.A", "hi"))

    asyncio.run(bot.process_webhook_update(webhook))
    asyncio.run(bot.process_webhook_update(webhook))  # Meta retries the same delivery
    asyncio.run(bot.process_webhook_update(_wa_webhook(
        _wa_message("100", "wamid.B", "again"), _wa_message("100", "wamid.B", "again"),
    )))

    assert manager.runs == [("100", "hi"), ("100", "again")]
    assert sent == [("100", "echo hi"), ("100", "echo again")]


def test_telegram_redelivered_update_runs_once_but_same_id_in_other_chat_runs():
    manager = _Manager()
    bot = TelegramBot(manager=manager, ensure_delivery=True)
    calls = []

    for chat_id in (1, 1, 2):
        asyncio.run(bot.process_payload(_tg_update(chat_id, "hello", calls, sender_id=chat_id, message_id=77)))

    assert manager.runs == [("1", "hello"), ("2", "hello")]
    assert _texts(calls) == ["echo hello", "echo hello"]


def test_dedup_cache_expires_is_bounded_and_can_be_disabled(monkeypatch):
    clock = [1000.0]
    bot = _RecordingBot(_Manager(), dedup_ttl=10)
    message = {"user_id": "u", "message_id": "m1"}

    monkeypatch.setattr(bots_module._time, "monotonic", lambda: clock[0])
    assert not bot._is_duplicate(message)
    assert bot._is_duplicate(message)
    clock[0] += 11
    assert not bot._is_duplicate(message)

    bot.DEDUP_MAX_ENTRIES = 3
    for i in range(5):
        bot._is_duplicate({"user_id": "u", "message_id": f"x{i}"})
    assert len(bot._seen_messages) == 3
    assert not bot._is_duplicate({"user_id": "u", "message_id": "x0"})  # evicted

    assert not bot._is_duplicate({"user_id": "u", "message_id": None})
    assert not bot._is_duplicate({"user_id": "u", "message_id": None})

    disabled = _RecordingBot(_Manager(), dedup_ttl=None)
    assert not disabled._is_duplicate(message)
    assert not disabled._is_duplicate(message)


# ---------------------------------------------------------------------------
# 3. Each user's replies arrive in order, and the turn waits for them
# ---------------------------------------------------------------------------

def test_slow_progress_update_arrives_before_the_final_answer():
    manager = _Manager(emit_update=True)
    bot = _RecordingBot(
        manager,
        on_update=lambda messages, mgr, params: "progress",
        send_delays={"progress": 0.3},
    )
    assert bot.ensure_delivery is False

    async def scenario():
        await bot.process_payload(_msg("u1", "hi"))
        # Everything is out before process_payload returns.
        assert bot.sent == [("u1", "progress"), ("u1", "echo hi")]

    asyncio.run(scenario())
    _assert_idle(bot)


def test_next_message_waits_until_the_previous_replies_are_sent():
    manager = _Manager()
    bot = _RecordingBot(manager, send_delays={"echo m1": 0.3})

    async def scenario():
        await asyncio.gather(bot.process_payload(_msg("u1", "m1")), bot.process_payload(_msg("u1", "m2")))

    asyncio.run(scenario())

    timeline = [e for e in manager.events if e[0] in ("run", "sent")]
    assert timeline == [
        ("run", "u1", "m1"), ("sent", "u1", "echo m1"),
        ("run", "u1", "m2"), ("sent", "u1", "echo m2"),
    ]
    _assert_idle(bot)


def test_send_queue_timeout_releases_the_turn_but_keeps_reply_order(caplog):
    manager = _Manager()
    bot = _RecordingBot(manager, send_delays={"echo m1": 0.5}, send_queue_timeout=0.1)

    async def scenario():
        await bot.process_payload(_msg("u1", "m1"))
        assert bot.sent == []  # still sending; the turn was released anyway
        await bot.process_payload(_msg("u1", "m2"))
        await asyncio.sleep(0.8)

    with caplog.at_level(logging.WARNING, logger=BOT_LOGGER):
        asyncio.run(scenario())

    assert bot.sent == [("u1", "echo m1"), ("u1", "echo m2")]
    assert any("send_queue_timeout" in r.getMessage() for r in caplog.records)
    _assert_idle(bot)


def test_error_reply_comes_after_progress_already_queued():
    def run(user_id, text):
        pass

    class _Manager2(_Manager):
        def run(self, input=None, user_id=None, on_update=None, on_update_params=None, **kwargs):
            self._invoke_callback(on_update, [{"message": "x"}], self, on_update_params)
            raise RuntimeError("tool failed")

    bot = _RecordingBot(_Manager2(), on_update=lambda m, mgr, p: "progress", send_delays={"progress": 0.2})

    asyncio.run(bot.process_payload(_msg("u1", "hi")))

    assert [t for _, t in bot.sent] == ["progress", bots_module.DEFAULT_ERROR_MSG]
    _assert_idle(bot)


def test_default_on_complete_sends_every_block_of_the_last_message():
    final = [
        {"type": "text", "content": {"response": "first"}},
        {"type": "text", "content": {"response": "second", "extra": 1}},
    ]
    bot = _RecordingBot(_Manager(final_message=final))

    asyncio.run(bot.process_payload(_msg("u1", "hi")))

    assert bot.sent == [("u1", "first"), ("u1", "second")]


# ---------------------------------------------------------------------------
# 4. Admin resets wait for in-flight runs and hold new ones
# ---------------------------------------------------------------------------

def test_clear_all_users_waits_for_running_messages_and_holds_new_ones():
    a_started = threading.Event()
    release_a = threading.Event()

    def run(user_id, text):
        if text == "slow":
            a_started.set()
            assert release_a.wait(5)

    manager = _Manager(run=run, admin_user_id="admin")
    bot = _RecordingBot(manager, maintenance_timeout=5)

    async def scenario():
        slow = asyncio.create_task(bot.process_payload(_msg("a", "slow")))
        await _wait_thread_event(a_started)
        admin = asyncio.create_task(bot.process_payload(_msg("admin", "/clear_all_users")))
        await asyncio.sleep(0.2)
        newcomer = asyncio.create_task(bot.process_payload(_msg("b", "new")))
        await asyncio.sleep(0.2)
        assert ("clear_all",) not in manager.events  # waiting for "slow"
        assert ("run", "b", "new") not in manager.events  # held until the reset is done
        release_a.set()
        await asyncio.gather(slow, admin, newcomer)

    try:
        asyncio.run(scenario())
    finally:
        release_a.set()

    order = [e for e in manager.events if e[0] in ("run_end", "clear_all", "run")]
    assert order == [("run", "a", "slow"), ("run_end", "a", "slow"), ("clear_all",), ("run", "b", "new"),
                     ("run_end", "b", "new")]
    assert ("admin", "Histories for all users cleared (3 DBs).") in bot.sent
    _assert_idle(bot)


def test_reset_system_times_out_without_resetting_and_tells_the_admin():
    a_started = threading.Event()
    release_a = threading.Event()

    def run(user_id, text):
        if text == "slow":
            a_started.set()
            assert release_a.wait(5)

    manager = _Manager(run=run, admin_user_id="admin")
    bot = _RecordingBot(manager, maintenance_timeout=0.2)

    async def scenario():
        slow = asyncio.create_task(bot.process_payload(_msg("a", "slow")))
        await _wait_thread_event(a_started)
        await bot.process_payload(_msg("admin", "/reset_system"))
        # Messages are accepted again after the timeout.
        await bot.process_payload(_msg("b", "after"))
        release_a.set()
        await slow

    try:
        asyncio.run(scenario())
    finally:
        release_a.set()

    assert ("reset",) not in manager.events
    admin_replies = [t for u, t in bot.sent if u == "admin"]
    assert len(admin_replies) == 1
    assert "Timed out after 0.2 s" in admin_replies[0]
    assert "1 message(s)" in admin_replies[0]
    assert "Nothing was reset" in admin_replies[0]
    assert ("b", "echo after") in bot.sent
    _assert_idle(bot)


def test_reset_system_runs_immediately_when_nothing_else_is_in_flight():
    manager = _Manager(admin_user_id="admin")
    bot = _RecordingBot(manager)

    asyncio.run(bot.process_payload(_msg("admin", "/reset_system")))

    assert ("reset",) in manager.events
    assert bot.sent == [("admin", "System reset: all user histories and files have been deleted.")]
    _assert_idle(bot)


# ---------------------------------------------------------------------------
# 5. Telegram groups: history per chat, permissions per sender, /cmd@BotName
# ---------------------------------------------------------------------------

def test_telegram_parse_keeps_chat_for_history_and_sender_for_permissions():
    bot = TelegramBot(manager=_Manager())
    parsed = asyncio.run(bot._parse_payload(
        _tg_update(-100123, "hi", [], sender_id=7, message_id=5, chat_type="supergroup")
    ))

    assert parsed["user_id"] == "-100123"
    assert parsed["sender_id"] == "7"
    assert parsed["message_id"] == 5
    assert parsed["chat_type"] == "supergroup"


def test_telegram_group_allow_list_and_admin_check_use_the_sender():
    manager = _Manager(admin_user_id="7")
    bot = TelegramBot(manager=manager, allowed_user_ids=[8], ensure_delivery=True)
    calls = []

    def send(sender, text, message_id):
        asyncio.run(bot.process_payload(_tg_update(
            -100, text, calls, sender_id=sender, message_id=message_id, chat_type="group",
        )))

    send(8, "hello", 1)                 # allowed member: history keyed by the group
    send(9, "let me in", 2)             # not allowed: ignored
    send(8, "/clear_all_users", 3)      # allowed, but not the admin
    send(7, "/clear_all_users", 4)      # the admin, writing in the group

    assert manager.runs == [("-100", "hello")]
    assert _texts(calls) == ["echo hello", "Unauthorized.", "Histories for all users cleared (3 DBs)."]
    assert ("clear_all",) in manager.events


def test_listing_a_group_id_still_admits_its_members():
    manager = _Manager()
    bot = TelegramBot(manager=manager, allowed_user_ids=[-100], ensure_delivery=True)

    asyncio.run(bot.process_payload(_tg_update(-100, "hi", [], sender_id=9, message_id=1, chat_type="group")))
    asyncio.run(bot.process_payload(_tg_update(-200, "hi", [], sender_id=9, message_id=1, chat_type="group")))

    assert manager.runs == [("-100", "hi")]


def test_commands_addressed_with_the_bot_username_are_recognized():
    manager = _Manager()
    bot = TelegramBot(manager=manager, ensure_delivery=True)
    bot.bot = SimpleNamespace(username="MyBot")
    calls = []

    for message_id, text in enumerate(["/clear@MyBot", "/start@mybot please", "/clear@OtherBot", "/nope@MyBot"], 1):
        asyncio.run(bot.process_payload(_tg_update(-100, text, calls, sender_id=8, message_id=message_id)))

    assert ("clear_user", "-100") in manager.events
    assert len([e for e in manager.events if e[0] == "clear_user"]) == 1  # not for /clear@OtherBot
    assert _texts(calls) == [bot.on_clear_msg, bot.on_start_msg, bot.unknown_command_msg]
    assert manager.runs == []


# ---------------------------------------------------------------------------
# 6. Outgoing files only from files_folder and media_dirs
# ---------------------------------------------------------------------------

def _media_tree(root):
    files = root / "files"
    (files / "u1").mkdir(parents=True)
    inside = files / "u1" / "photo.png"
    inside.write_bytes(b"inside")
    extra = root / "extra"
    extra.mkdir()
    extra_file = extra / "brochure.pdf"
    extra_file.write_bytes(b"extra")
    outside = root / "secret.env"
    outside.write_bytes(b"API_KEY=hunter2")
    return files, inside, extra, extra_file, outside


def test_local_paths_outside_files_folder_and_media_dirs_are_refused(workspace_tmp_path, caplog):
    files, inside, extra, extra_file, outside = _media_tree(workspace_tmp_path)
    bot = _RecordingBot(_Manager(files_folder=str(files)), media_dirs=[str(extra)])

    assert bot._local_path(f"file:{inside}") == str(inside)
    assert bot._local_path(str(inside)) == str(inside)
    assert bot._local_path(str(extra_file)) == str(extra_file)
    assert bot._local_path("https://example.com/a.png") is None
    assert bot._local_path("AgACAgIAAxkBAAIBY2Zz") is None  # platform file id

    with caplog.at_level(logging.WARNING, logger=BOT_LOGGER):
        for refused in (f"file:{outside}", str(outside), f"file:{files}{os.sep}..{os.sep}secret.env",
                        f"file:{files}", "file:/etc/passwd"):
            with pytest.raises(PermissionError):
                bot._local_path(refused)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 5
    assert any("secret.env" in w for w in warnings)


def test_whatsapp_block_pointing_outside_is_not_uploaded(workspace_tmp_path, monkeypatch):
    files, inside, _, _, outside = _media_tree(workspace_tmp_path)
    bot = _whatsapp_bot(_Manager(files_folder=str(files)))
    uploads = []

    async def fake_upload(path):
        uploads.append(path)
        return "media-id"

    monkeypatch.setattr(bot, "_upload_media", fake_upload)
    posts = _capture_whatsapp_posts(monkeypatch)

    asyncio.run(bot._send_blocks("5491", [
        {"type": "document", "content": {"path": f"file:{outside}"}},
        {"type": "image", "content": {"path": f"file:{inside}"}},
        {"type": "text", "content": {"response": "done"}},
    ], {}))

    assert uploads == [str(inside)]
    assert [p["json"]["type"] for p in posts] == ["image", "text"]


def test_telegram_block_pointing_outside_is_not_sent(workspace_tmp_path):
    files, inside, _, _, outside = _media_tree(workspace_tmp_path)
    bot = TelegramBot(manager=_Manager(files_folder=str(files)))
    calls = []

    asyncio.run(bot._send_blocks("42", [
        {"type": "document", "content": {"path": str(outside)}},
        {"type": "image", "content": {"path": f"file:{inside}"}},
    ], {"original_payload": _tg_update(42, "hi", calls)}))

    assert calls == [("photo", ("file", b"inside"), {})]


# ---------------------------------------------------------------------------
# 7. Large incoming media
# ---------------------------------------------------------------------------

def test_telegram_file_over_max_media_bytes_is_refused_before_downloading():
    manager = _Manager()
    bot = TelegramBot(manager=manager, max_media_bytes=1024 * 1024, ensure_delivery=True)
    get_file_calls = []

    async def get_file(file_id):
        get_file_calls.append(file_id)

    bot.bot = SimpleNamespace(get_file=get_file)
    calls = []
    document = SimpleNamespace(file_id="f1", file_size=5 * 1024 * 1024, file_name="big.pdf",
                               mime_type="application/pdf")

    asyncio.run(bot.process_payload(_tg_update(42, None, calls, document=document, message_id=1)))

    assert get_file_calls == []
    assert manager.runs == []
    assert _texts(calls) == ["Sorry, that file is too large for me to process (the limit is 1 MB)."]


def test_telegram_file_too_big_for_the_bot_api_tells_the_user():
    manager = _Manager()
    bot = TelegramBot(manager=manager, max_media_bytes=None, ensure_delivery=True)

    async def get_file(file_id):
        raise RuntimeError("File is too big")

    bot.bot = SimpleNamespace(get_file=get_file)
    calls = []
    video = SimpleNamespace(file_id="v1", file_size=None, mime_type="video/mp4")

    asyncio.run(bot.process_payload(_tg_update(42, None, calls, video=video, message_id=1)))

    assert manager.runs == []
    assert _texts(calls) == [TOO_LARGE_20MB]


def test_telegram_download_streams_to_disk_and_is_saved_in_files_folder(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    bot = TelegramBot(manager=manager, telegram_token="1:x")
    written = {}
    content = b"%PDF-1.4 report"

    class _TgFile:
        file_size = len(content)
        file_path = "documents/file_3.pdf"

        async def download_to_drive(self, custom_path=None):
            written["path"] = custom_path
            with open(custom_path, "wb") as fh:
                fh.write(content)

    async def get_file(file_id):
        return _TgFile()

    bot.bot = SimpleNamespace(get_file=get_file)
    media_info = SimpleNamespace(file_id="f", file_size=len(content), file_name="report.pdf",
                                 mime_type="application/pdf")

    ref = asyncio.run(bot._download_media_and_save("42", media_info))

    path = ref[5:]
    files_root = os.path.realpath(manager.files_folder)
    assert os.path.commonpath([files_root, os.path.realpath(path)]) == files_root
    assert path.endswith(".pdf")
    with open(path, "rb") as fh:
        assert fh.read() == content
    assert not os.path.exists(written["path"])
    assert not os.path.exists(written["path"] + ".pdf")
    assert manager._load_files_in_dict({"x": ref})["x"] == content


class _FakeStream:
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


class _FakeJson:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def _fake_whatsapp_get(monkeypatch, info, chunks, response_headers=None):
    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False, **kwargs):
        calls.append((url, stream))
        if stream:
            return _FakeStream(chunks, response_headers)
        return _FakeJson(info)

    monkeypatch.setattr(bots_module.requests, "get", fake_get)
    return calls


def test_whatsapp_download_is_streamed_with_the_configured_api_version(workspace_tmp_path, monkeypatch):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    bot = _whatsapp_bot(manager, whatsapp_token="t", phone_number_id="p", webhook_verify_token="v")
    voice = b"OggS" + b"\x00" * 200
    calls = _fake_whatsapp_get(monkeypatch, {"url": "https://lookaside.example/m"},
                               [voice[:100], voice[100:]])

    ref = asyncio.run(bot._download_media_and_save("5491", {"id": "media-1", "mime_type": "audio/ogg; codecs=opus"}))

    assert calls == [(f"https://graph.facebook.com/{DEFAULT_WHATSAPP_API_VERSION}/media-1", False),
                     ("https://lookaside.example/m", True)]
    assert ref.endswith(".ogg")
    with open(ref[5:], "rb") as fh:
        assert fh.read() == voice


def test_whatsapp_download_over_the_cap_is_aborted_and_cleaned_up(monkeypatch):
    manager = _Manager()
    bot = _whatsapp_bot(manager, max_media_bytes=150)
    temp_paths = []
    original_new_temp = bot._new_temp_file

    def tracking_new_temp():
        path = original_new_temp()
        temp_paths.append(path)
        return path

    monkeypatch.setattr(bot, "_new_temp_file", tracking_new_temp)
    saved = []
    monkeypatch.setattr(manager, "save_file", lambda *a, **k: saved.append(a) or "file:x")

    # Meta announces the size up front: refused before the media URL is fetched.
    calls = _fake_whatsapp_get(monkeypatch, {"url": "https://m", "file_size": 500}, [b"x" * 500])
    with pytest.raises(MediaTooLargeError):
        asyncio.run(bot._download_media_and_save("5491", {"id": "m1"}))
    assert calls == [(f"https://graph.facebook.com/{DEFAULT_WHATSAPP_API_VERSION}/m1", False)]

    # Size unknown: the stream is cut as soon as it passes the cap.
    _fake_whatsapp_get(monkeypatch, {"url": "https://m"}, [b"x" * 100] * 10)
    with pytest.raises(MediaTooLargeError):
        asyncio.run(bot._download_media_and_save("5491", {"id": "m2"}))

    assert saved == []
    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)


def test_whatsapp_large_image_gets_the_too_large_reply(monkeypatch):
    manager = _Manager()
    bot, sent = _whatsapp_bot_with_fake_sends(manager)
    _fake_whatsapp_get(monkeypatch, {"url": "https://m", "file_size": DEFAULT_MAX_MEDIA_BYTES + 1}, [])
    message = {"from": "100", "id": "wamid.img", "type": "image", "timestamp": str(int(time.time())),
               "image": {"id": "img-1", "mime_type": "image/jpeg"}}

    asyncio.run(bot.process_webhook_update(_wa_webhook(message)))

    assert manager.runs == []
    assert sent == [("100", TOO_LARGE_20MB)]


# ---------------------------------------------------------------------------
# 8. Telegram and WhatsApp behave the same
# ---------------------------------------------------------------------------

def test_whatsapp_direct_send_raises_when_the_api_rejects_it(monkeypatch):
    bot = _whatsapp_bot()
    _capture_whatsapp_posts(monkeypatch, lambda url, payload: _Response(
        ok=False, status_code=400, data={"error": {"message": "Invalid parameter", "code": 100}}))

    with pytest.raises(WhatsappAPIError) as excinfo:
        asyncio.run(bot.send_text("5491", "hello there"))

    assert excinfo.value.status_code == 400
    assert excinfo.value.error_code == 100
    assert "Invalid parameter" in str(excinfo.value)
    assert "hello there" not in str(excinfo.value)


def test_whatsapp_failed_log_upload_is_not_counted_as_sent(monkeypatch, workspace_tmp_path):
    bot = _whatsapp_bot()
    log_file = workspace_tmp_path / "usage.log"
    log_file.write_text("usage")

    def response_for(url, payload):
        if url.endswith("/media"):
            return _Response(ok=False, status_code=500, data={"error": {"message": "upload failed"}})
        return _Response()

    posts = _capture_whatsapp_posts(monkeypatch, response_for)

    count = asyncio.run(bot._send_log_files("5491", [{"path": str(log_file), "filename": "usage.log"}], {}))

    assert count == 0
    assert [p["url"].rsplit("/", 1)[-1] for p in posts] == ["media"]  # no "Sent 1 log file(s)."


@pytest.mark.parametrize("bot_class", [TelegramBot, WhatsappBot])
def test_custom_command_names_follow_one_rule_on_both_platforms(bot_class):
    for bad in ("/my-cmd", "status", "/", "/" + "a" * 33, "/café", None):
        with pytest.raises(ValueError, match="letters, digits or underscores"):
            bot_class(manager=_Manager(), custom_commands=[{"command": bad, "message": "x"}])

    bot = bot_class(manager=_Manager(), custom_commands={"command": "/My_Cmd2", "message": "ok"})
    assert "/my_cmd2" in bot.commands


def test_telegram_sends_url_media_and_audio_as_audio_unless_voice_note(workspace_tmp_path):
    files = workspace_tmp_path / "files"
    files.mkdir()
    voice = files / "voice.ogg"
    voice.write_bytes(b"ogg")
    song = files / "tts.mp3"
    song.write_bytes(b"mp3")
    bot = TelegramBot(manager=_Manager(files_folder=str(files)))
    calls = []

    asyncio.run(bot._send_blocks("42", [
        {"type": "image", "content": {"url": "https://example.com/cat.png", "caption": "cat"}},
        {"type": "audio", "content": {"url": "https://example.com/song.mp3"}},
        {"type": "audio", "content": {"kind": "file", "path": f"file:{voice}", "is_voice_note": True}},
        {"type": "audio", "content": {"kind": "file", "path": f"file:{song}"}},
        {"type": "image", "content": "AgACAgIAAxkBAAIBY2Zz"},
    ], {"original_payload": _tg_update(42, "hi", calls)}))

    assert calls == [
        ("photo", "https://example.com/cat.png", {"caption": "cat"}),
        ("audio", "https://example.com/song.mp3", {}),
        ("voice", ("file", b"ogg"), {}),
        ("audio", ("file", b"mp3"), {}),
        ("photo", "AgACAgIAAxkBAAIBY2Zz", {}),
    ]


@pytest.mark.skipif(not REAL_PTB, reason="python-telegram-bot is not installed (stubbed in conftest)")
def test_telegram_unknown_and_addressed_commands_through_ptb_handlers():
    from telegram import Update, User

    manager = _Manager()
    bot = TelegramBot(manager=manager, custom_commands={"command": "/status", "message": "all good"})
    sent = []

    async def fake_send_blocks(user_id, blocks, original_message):
        sent.extend(manager._block_to_plain_text(b) for b in blocks)

    bot._send_blocks = fake_send_blocks
    app = bot.application
    try:
        with app.bot._unfrozen():
            app.bot._bot_user = User(id=123456, is_bot=True, first_name="test", username="test_bot")
        app._initialized = True
    except AttributeError as exc:  # pragma: no cover - PTB internals changed
        pytest.skip(f"python-telegram-bot internals changed: {exc}")

    def update(update_id, text):
        command = text.split()[0]
        return Update.de_json({
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": int(time.time()),
                "chat": {"id": 5, "type": "group"},
                "from": {"id": 5, "is_bot": False, "first_name": "user"},
                "text": text,
                "entities": [{"type": "bot_command", "offset": 0, "length": len(command)}],
            },
        }, app.bot)

    async def scenario():
        for update_id, text in enumerate(["/nope", "/status@test_bot", "/status@other_bot"], 1):
            await app.process_update(update(update_id, text))

    asyncio.run(scenario())

    assert sent == [bot.unknown_command_msg, "all good"]


# ---------------------------------------------------------------------------
# 9. Privacy: the raw platform payload is not persisted by default
# ---------------------------------------------------------------------------

def test_telegram_metadata_keeps_ids_but_not_names():
    bot = TelegramBot(manager=_Manager())
    update = _tg_update(-100, "hello", [], sender_id=7, message_id=9, chat_type="group")
    update.to_dict = lambda: {"message": {"from": {"first_name": "Ada", "username": "ada_lovelace"}}}

    blocks = asyncio.run(bot._build_mas_blocks(asyncio.run(bot._parse_payload(update))))

    metadata = blocks[0]["metadata"]["user_message"]
    assert metadata["user_id"] == "-100"
    assert metadata["sender_id"] == "7"
    assert metadata["message_id"] == "9"
    assert metadata["chat_type"] == "group"
    dumped = json.dumps(metadata)
    assert "ada_lovelace" not in dumped and "Ada" not in dumped

    opted_in = TelegramBot(manager=_Manager(), store_original_payload=True)
    blocks = asyncio.run(opted_in._build_mas_blocks(asyncio.run(opted_in._parse_payload(update))))
    assert "ada_lovelace" in json.dumps(blocks[0]["metadata"]["user_message"]["original_payload"])


def test_telegram_error_handler_logs_ids_not_the_update(caplog):
    bot = TelegramBot(manager=_Manager())
    update = SimpleNamespace(update_id=31, to_dict=lambda: {"message": {"text": "private words"}})

    with caplog.at_level(logging.ERROR, logger=BOT_LOGGER):
        asyncio.run(bot._on_error(update, SimpleNamespace(error=RuntimeError("boom"))))

    messages = [r.getMessage() for r in caplog.records]
    assert any("update_id=31" in m for m in messages)
    assert not any("private words" in m for m in messages)


# ---------------------------------------------------------------------------
# 10. Logging goes through the library logger and follows verbose
# ---------------------------------------------------------------------------

def test_bot_logs_use_the_library_logger_and_verbose_controls_activity_logs(monkeypatch, caplog):
    assert bots_module.logger is library_logger
    assert any(isinstance(h, logging.StreamHandler) for h in library_logger.handlers)

    with caplog.at_level(logging.DEBUG, logger=BOT_LOGGER):
        TelegramBot(manager=_Manager(), verbose=True)
        TelegramBot(manager=_Manager())
    created = [r for r in caplog.records if r.getMessage() == "[TelegramBot] Instance of TelegramBot created."]
    assert len(created) == 1
    assert created[0].name == library_logger.name

    _capture_whatsapp_posts(monkeypatch)
    quiet = _whatsapp_bot()
    loud = _whatsapp_bot(verbose=True)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=BOT_LOGGER):
        asyncio.run(quiet.send_text("5491", "hi"))
        assert caplog.records == []
        asyncio.run(loud.send_text("5491", "hi"))
    assert any("Sent a text message" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 11. WhatsApp Graph API version
# ---------------------------------------------------------------------------

def test_whatsapp_api_version_default_parameter_and_key():
    assert _whatsapp_bot().graph_url == f"https://graph.facebook.com/{DEFAULT_WHATSAPP_API_VERSION}/phone-id"
    assert DEFAULT_WHATSAPP_API_VERSION != "v18.0"
    assert _whatsapp_bot(api_version="v24.0").graph_url == "https://graph.facebook.com/v24.0/phone-id"
    assert _whatsapp_bot(api_version="22.0").api_version == "v22.0"

    class _KeyManager(_Manager):
        def get_key(self, name):
            return "v21.0" if name == "whatsapp_api_version" else super().get_key(name)

    assert _whatsapp_bot(_KeyManager()).api_version == "v21.0"

    with pytest.raises(ValueError, match="Graph API version"):
        _whatsapp_bot(api_version="latest")
