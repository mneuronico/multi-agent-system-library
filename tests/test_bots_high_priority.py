"""Bot pipeline: per-user ordering, concurrency, message age, error replies, allow lists.

Everything runs offline against fake managers; nothing calls Telegram, WhatsApp or a model.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import telegram

import mas.bots as bots_module
from mas.bots import DEFAULT_ERROR_MSG, Bot, TelegramBot, WhatsappBot


# conftest.py replaces python-telegram-bot with a stub module (no __file__) when it is missing.
REAL_PTB = getattr(telegram, "__file__", None) is not None

# Bots log through the library logger (mas._shared), like the rest of MAS.
BOT_LOGGER = bots_module.logger.name


class _Manager:
    on_update = None
    on_complete = None

    def __init__(self, run=None, stt=None, admin_user_id=None):
        self._run_hook = run
        self._stt_hook = stt
        self.admin_user_id = admin_user_id
        self.runs = []
        self._runs_lock = threading.Lock()

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

    def run(self, input=None, user_id=None, on_complete=None, on_complete_params=None, **kwargs):
        text = input[0]["content"]["response"]
        with self._runs_lock:
            self.runs.append((user_id, text))
        if self._run_hook:
            self._run_hook(user_id, text)
        if on_complete:
            self._invoke_callback(on_complete, [{"message": f"echo {text}"}], self, on_complete_params)
        return []

    def stt(self, file_path, provider=None, model=None, user_id=None):
        return self._stt_hook(file_path, user_id) if self._stt_hook else "transcript"

    def texts(self):
        return [text for _, text in self.runs]


class _RecordingBot(Bot):
    def __init__(self, manager, **kwargs):
        kwargs.setdefault("ensure_delivery", True)
        super().__init__(manager=manager, **kwargs)
        self.sent = []

    async def _parse_payload(self, payload):
        return payload

    async def _send_blocks(self, user_id, blocks, original_message):
        for block in blocks:
            self.sent.append((user_id, self.manager._block_to_plain_text(block)))

    async def _download_media_and_save(self, user_id, media_info):
        if media_info == "broken":
            raise RuntimeError("download failed")
        return "file:/tmp/voice.ogg"

    async def _send_log_files(self, user_id, files_to_send, original_message):
        return 0


def _msg(user_id, text, *, age=0.0, message_type="text", media_info=None):
    return {
        "user_id": str(user_id),
        "message_type": message_type,
        "text": text,
        "media_info": media_info,
        "is_voice_note": message_type == "audio",
        "timestamp": datetime.now(timezone.utc) - timedelta(seconds=age),
        "original_payload": {"id": f"{user_id}:{text}"},
    }


async def _wait_thread_event(event, timeout=5):
    assert await asyncio.to_thread(event.wait, timeout)


# ---------------------------------------------------------------------------
# Per-user serialization (no more dropped messages)
# ---------------------------------------------------------------------------

def test_same_user_messages_wait_and_run_in_arrival_order():
    first_started = threading.Event()
    release_first = threading.Event()

    def run(user_id, text):
        if text == "m1":
            first_started.set()
            assert release_first.wait(5)

    manager = _Manager(run=run)
    bot = _RecordingBot(manager)

    async def scenario():
        first = asyncio.create_task(bot.process_payload(_msg("u1", "m1")))
        await _wait_thread_event(first_started)
        later = [asyncio.create_task(bot.process_payload(_msg("u1", f"m{i}"))) for i in range(2, 6)]
        await asyncio.sleep(0.1)
        # The later messages are waiting, neither dropped nor running alongside the first.
        assert manager.texts() == ["m1"]
        assert len(bot._user_locks) == 1
        release_first.set()
        await asyncio.gather(first, *later)

    asyncio.run(scenario())

    assert manager.texts() == ["m1", "m2", "m3", "m4", "m5"]
    assert bot.sent == [("u1", f"echo m{i}") for i in range(1, 6)]
    assert bot._user_locks == {}


def test_a_users_runs_never_overlap():
    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def run(user_id, text):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.02)
        with guard:
            active["now"] -= 1

    manager = _Manager(run=run)
    bot = _RecordingBot(manager)

    async def scenario():
        await asyncio.gather(*(bot.process_payload(_msg("u1", f"m{i}")) for i in range(8)))

    asyncio.run(scenario())

    assert active["max"] == 1
    assert manager.texts() == [f"m{i}" for i in range(8)]


def test_other_users_are_not_blocked_by_a_slow_run():
    slow_started = threading.Event()
    release_slow = threading.Event()

    def run(user_id, text):
        if user_id == "slow":
            slow_started.set()
            assert release_slow.wait(5)

    manager = _Manager(run=run)
    bot = _RecordingBot(manager)

    async def scenario():
        slow = asyncio.create_task(bot.process_payload(_msg("slow", "a")))
        await _wait_thread_event(slow_started)
        await asyncio.wait_for(bot.process_payload(_msg("fast", "b")), timeout=3)
        assert ("fast", "echo b") in bot.sent
        assert ("slow", "echo a") not in bot.sent
        release_slow.set()
        await slow

    try:
        asyncio.run(scenario())
    finally:
        release_slow.set()

    assert ("slow", "echo a") in bot.sent


def test_user_lock_map_is_cleaned_up_and_works_across_event_loops():
    manager = _Manager()
    bot = _RecordingBot(manager)

    for round_number in range(3):
        # Each asyncio.run uses a new event loop; locks from an old loop must not be reused.
        asyncio.run(bot.process_payload(_msg("u1", f"r{round_number}")))
        assert bot._user_locks == {}

    async def many_users():
        await asyncio.gather(*(bot.process_payload(_msg(f"user-{i}", "hi")) for i in range(20)))

    asyncio.run(many_users())

    assert bot._user_locks == {}
    assert len(manager.runs) == 23


def test_cancelled_waiting_message_releases_its_lock_entry():
    first_started = threading.Event()
    release_first = threading.Event()

    def run(user_id, text):
        if text == "m1":
            first_started.set()
            assert release_first.wait(5)

    manager = _Manager(run=run)
    bot = _RecordingBot(manager)

    async def scenario():
        first = asyncio.create_task(bot.process_payload(_msg("u1", "m1")))
        await _wait_thread_event(first_started)
        waiting = asyncio.create_task(bot.process_payload(_msg("u1", "m2")))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release_first.set()
        await first

    asyncio.run(scenario())

    assert manager.texts() == ["m1"]
    assert bot._user_locks == {}


# ---------------------------------------------------------------------------
# Message age is measured on arrival
# ---------------------------------------------------------------------------

def test_message_queued_behind_a_slow_run_is_not_dropped_as_too_old():
    first_started = threading.Event()

    def run(user_id, text):
        if text == "first":
            first_started.set()
            time.sleep(0.6)  # longer than max_allowed_message_delay below

    manager = _Manager(run=run)
    bot = _RecordingBot(manager, max_allowed_message_delay=0.3)

    async def scenario():
        first = asyncio.create_task(bot.process_payload(_msg("u1", "first")))
        await _wait_thread_event(first_started)
        await asyncio.gather(first, bot.process_payload(_msg("u1", "second")))

    asyncio.run(scenario())

    assert manager.texts() == ["first", "second"]


def test_message_already_too_old_on_arrival_is_dropped_with_a_warning(caplog):
    manager = _Manager()
    bot = _RecordingBot(manager, max_allowed_message_delay=120)

    with caplog.at_level(logging.DEBUG, logger=BOT_LOGGER):
        asyncio.run(bot.process_payload(_msg("u1", "stale", age=600)))

    assert manager.runs == []
    assert bot.sent == []
    warnings = [r for r in caplog.records if r.name == BOT_LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "u1" in warnings[0].getMessage()
    assert "max_allowed_message_delay=120" in warnings[0].getMessage()


def test_max_allowed_message_delay_none_accepts_any_age():
    manager = _Manager()
    bot = _RecordingBot(manager, max_allowed_message_delay=None)

    asyncio.run(bot.process_payload(_msg("u1", "ancient", age=10 ** 6)))

    assert manager.texts() == ["ancient"]


# ---------------------------------------------------------------------------
# Errors: logged with traceback, user gets error_msg, next messages still run
# ---------------------------------------------------------------------------

def _error_records(caplog):
    return [
        r for r in caplog.records
        if r.name == BOT_LOGGER and r.levelno == logging.ERROR and r.exc_info
    ]


def test_run_failure_is_logged_and_user_gets_error_reply(caplog):
    def run(user_id, text):
        if text == "explode":
            raise RuntimeError("model provider is down")

    manager = _Manager(run=run)
    bot = _RecordingBot(manager)

    async def scenario():
        await asyncio.gather(
            bot.process_payload(_msg("u1", "explode")),
            bot.process_payload(_msg("u1", "after")),
        )

    with caplog.at_level(logging.ERROR, logger=BOT_LOGGER):
        asyncio.run(scenario())

    assert bot.sent == [("u1", DEFAULT_ERROR_MSG), ("u1", "echo after")]
    records = _error_records(caplog)
    assert len(records) == 1
    assert "u1" in records[0].getMessage()
    assert "model provider is down" in str(records[0].exc_info[1])


def test_custom_error_msg_and_disabled_error_msg():
    def run(user_id, text):
        raise RuntimeError("boom")

    custom = _RecordingBot(_Manager(run=run), error_msg="Oops, try again.")
    asyncio.run(custom.process_payload(_msg("u1", "hi")))
    assert custom.sent == [("u1", "Oops, try again.")]

    silent = _RecordingBot(_Manager(run=run), error_msg=None)
    asyncio.run(silent.process_payload(_msg("u1", "hi")))
    assert silent.sent == []


def test_callback_error_gets_error_reply(caplog):
    def bad_on_complete(messages, manager, params):
        raise ValueError("callback bug")

    bot = _RecordingBot(_Manager(), on_complete=bad_on_complete)

    with caplog.at_level(logging.ERROR, logger=BOT_LOGGER):
        asyncio.run(bot.process_payload(_msg("u1", "hi")))

    assert bot.sent == [("u1", DEFAULT_ERROR_MSG)]
    assert _error_records(caplog)


def test_failed_media_download_gets_error_reply():
    manager = _Manager()
    bot = _RecordingBot(manager)

    asyncio.run(bot.process_payload(_msg("u1", "caption", message_type="image", media_info="broken")))

    assert manager.runs == []
    assert bot.sent == [("u1", DEFAULT_ERROR_MSG)]


def test_custom_speech_to_text_error_gets_error_reply():
    def stt(params):
        raise RuntimeError("stt crashed")

    manager = _Manager()
    bot = _RecordingBot(manager, speech_to_text=stt)

    asyncio.run(bot.process_payload(_msg("u1", None, message_type="audio", media_info="voice")))

    assert manager.runs == []
    assert bot.sent == [("u1", DEFAULT_ERROR_MSG)]


def test_failing_error_reply_is_logged_not_raised(caplog):
    def run(user_id, text):
        raise RuntimeError("boom")

    class _BrokenSendBot(_RecordingBot):
        async def _send_blocks(self, user_id, blocks, original_message):
            raise ConnectionError("platform unreachable")

    bot = _BrokenSendBot(_Manager(run=run))

    with caplog.at_level(logging.ERROR, logger=BOT_LOGGER):
        asyncio.run(bot.process_payload(_msg("u1", "hi")))

    assert len(_error_records(caplog)) == 2
    assert bot._user_locks == {}


def test_unparseable_payload_is_logged_and_skipped(caplog):
    class _BadParseBot(_RecordingBot):
        async def _parse_payload(self, payload):
            raise KeyError("chat")

    bot = _BadParseBot(_Manager())

    with caplog.at_level(logging.ERROR, logger=BOT_LOGGER):
        asyncio.run(bot.process_payload({"weird": True}))

    assert bot.sent == []
    assert _error_records(caplog)


def test_automatic_stt_runs_off_the_event_loop_thread():
    seen = {}

    def stt(file_path, user_id):
        seen["thread"] = threading.get_ident()
        seen["user_id"] = user_id
        return "hello from audio"

    manager = _Manager(stt=stt)
    bot = _RecordingBot(manager)

    async def scenario():
        seen["loop_thread"] = threading.get_ident()
        await bot.process_payload(_msg("u1", None, message_type="audio", media_info="voice"))

    asyncio.run(scenario())

    assert seen["thread"] != seen["loop_thread"]
    assert seen["user_id"] == "u1"
    assert manager.texts() == ["hello from audio"]


def test_async_custom_speech_to_text_is_awaited():
    async def stt(params):
        await asyncio.sleep(0)
        return "async transcript"

    manager = _Manager()
    bot = _RecordingBot(manager, speech_to_text=stt)

    asyncio.run(bot.process_payload(_msg("u1", None, message_type="audio", media_info="voice")))

    assert manager.texts() == ["async transcript"]


# ---------------------------------------------------------------------------
# allowed_user_ids
# ---------------------------------------------------------------------------

def test_allowed_user_ids_none_accepts_everyone():
    manager = _Manager()
    bot = _RecordingBot(manager)

    asyncio.run(bot.process_payload(_msg("anyone", "hi")))

    assert manager.texts() == ["hi"]


def test_disallowed_user_is_ignored_without_calling_the_model(caplog):
    manager = _Manager()
    bot = _RecordingBot(manager, allowed_user_ids=[111, "222"])

    with caplog.at_level(logging.INFO, logger=BOT_LOGGER):
        asyncio.run(bot.process_payload(_msg("999", "let me in")))
        asyncio.run(bot.process_payload(_msg("999", "/start")))

    assert manager.runs == []
    assert bot.sent == []
    infos = [r for r in caplog.records if r.name == BOT_LOGGER and r.levelno == logging.INFO]
    assert len(infos) == 2
    assert all("999" in r.getMessage() for r in infos)


def test_allowed_user_ids_compare_as_strings_and_admin_is_always_allowed():
    manager = _Manager(admin_user_id=555)
    bot = _RecordingBot(manager, allowed_user_ids=[111, "222"])

    for user_id in ("111", "222", "555"):
        asyncio.run(bot.process_payload(_msg(user_id, f"from {user_id}")))

    assert manager.texts() == ["from 111", "from 222", "from 555"]


def test_empty_allow_list_only_admits_admin():
    manager = _Manager(admin_user_id="555")
    bot = _RecordingBot(manager, allowed_user_ids=[])

    asyncio.run(bot.process_payload(_msg("111", "no")))
    asyncio.run(bot.process_payload(_msg("555", "yes")))

    assert manager.texts() == ["yes"]


def test_single_string_allow_list_is_one_id_not_characters():
    bot = _RecordingBot(_Manager(), allowed_user_ids="123")

    assert bot._is_user_allowed("123")
    assert not bot._is_user_allowed("1")


def test_whatsapp_allow_list_accepts_plus_prefixed_numbers():
    manager = _Manager()
    bot = WhatsappBot(manager=manager, allowed_user_ids=["+5491112345678"], ensure_delivery=True)
    sent = []

    async def fake_send_blocks(user_id, blocks, original_message):
        sent.append(user_id)

    bot._send_blocks = fake_send_blocks
    now = str(int(time.time()))
    webhook = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "5491112345678", "id": "wamid.1", "type": "text", "timestamp": now, "text": {"body": "hi"}},
        {"from": "14155550123", "id": "wamid.2", "type": "text", "timestamp": now, "text": {"body": "intruder"}},
    ]}}]}]}

    asyncio.run(bot.process_webhook_update(webhook))

    assert manager.runs == [("5491112345678", "hi")]
    assert sent == ["5491112345678"]


# ---------------------------------------------------------------------------
# WhatsApp webhook batches
# ---------------------------------------------------------------------------

def _whatsapp_bot_with_fake_sends(manager, **kwargs):
    bot = WhatsappBot(manager=manager, ensure_delivery=True, **kwargs)
    sent = []

    async def fake_send_text(user_id, text, **_kwargs):
        sent.append((user_id, text))

    bot.send_text = fake_send_text
    return bot, sent


def test_whatsapp_one_failing_message_does_not_drop_the_rest_of_the_batch():
    def run(user_id, text):
        if text == "bad":
            raise RuntimeError("tool failed")

    manager = _Manager(run=run)
    bot, sent = _whatsapp_bot_with_fake_sends(manager)
    now = str(int(time.time()))

    def wa(sender, msg_id, body):
        return {"from": sender, "id": msg_id, "type": "text", "timestamp": now, "text": {"body": body}}

    webhook = {"entry": [
        {"changes": [{"value": {"messages": [wa("100", "w1", "bad"), wa("100", "w2", "good")]}}]},
        {"changes": [{"value": {"messages": [wa("200", "w3", "other user")]}}]},
    ]}

    asyncio.run(bot.process_webhook_update(webhook))

    assert sorted(manager.runs) == [("100", "bad"), ("100", "good"), ("200", "other user")]
    # Same user's messages keep their order.
    assert [t for u, t in manager.runs if u == "100"] == ["bad", "good"]
    assert ("100", DEFAULT_ERROR_MSG) in sent
    assert ("100", "echo good") in sent
    assert ("200", "echo other user") in sent
    assert sent.index(("100", DEFAULT_ERROR_MSG)) < sent.index(("100", "echo good"))


def test_whatsapp_malformed_webhook_bodies_do_not_raise():
    manager = _Manager()
    bot, sent = _whatsapp_bot_with_fake_sends(manager)

    for body in ([], None, {"entry": None}, {"entry": [None, {"changes": [None, {"value": None}]}]},
                 {"entry": [{"changes": [{"value": {"messages": [None, "x"]}}]}]}):
        asyncio.run(bot.process_webhook_update(body))

    assert manager.runs == []
    assert sent == []


def test_whatsapp_stale_message_in_batch_is_dropped_but_fresh_one_runs(caplog):
    manager = _Manager()
    bot, sent = _whatsapp_bot_with_fake_sends(manager)
    webhook = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "100", "id": "old", "type": "text", "timestamp": str(int(time.time()) - 3600), "text": {"body": "old"}},
        {"from": "100", "id": "new", "type": "text", "timestamp": str(int(time.time())), "text": {"body": "new"}},
    ]}}]}]}

    with caplog.at_level(logging.WARNING, logger=BOT_LOGGER):
        asyncio.run(bot.process_webhook_update(webhook))

    assert manager.texts() == ["new"]
    assert any(r.levelno == logging.WARNING and "100" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _telegram_update(chat_id, text, replies):
    class _Message:
        def __init__(self):
            self.chat = SimpleNamespace(id=chat_id)
            self.text = text
            self.caption = None
            self.photo = []
            self.voice = None
            self.audio = None
            self.video = None
            self.document = None
            self.sticker = None
            self.date = datetime.now(timezone.utc)

        async def reply_text(self, value, **kwargs):
            replies.append((str(chat_id), value))

    return SimpleNamespace(message=_Message())


def test_telegram_error_reply_is_sent_as_a_reply():
    def run(user_id, text):
        raise RuntimeError("boom")

    bot = TelegramBot(manager=_Manager(run=run), ensure_delivery=True)
    replies = []

    asyncio.run(bot.process_payload(_telegram_update(42, "hello", replies)))

    assert replies == [("42", DEFAULT_ERROR_MSG)]


def test_telegram_disallowed_chat_is_ignored():
    manager = _Manager()
    bot = TelegramBot(manager=manager, allowed_user_ids=[1], ensure_delivery=True)
    replies = []

    asyncio.run(bot.process_payload(_telegram_update(42, "hello", replies)))
    asyncio.run(bot.process_payload(_telegram_update(1, "hello", replies)))

    assert manager.runs == [("1", "hello")]
    assert replies == [("1", "echo hello")]


@pytest.mark.skipif(not REAL_PTB, reason="python-telegram-bot is not installed (stubbed in conftest)")
def test_telegram_application_processes_updates_concurrently():
    assert TelegramBot(manager=_Manager()).application.concurrent_updates == 256
    assert TelegramBot(manager=_Manager(), concurrent_updates=8).application.concurrent_updates == 8
    assert TelegramBot(manager=_Manager(), concurrent_updates=False).application.concurrent_updates == 1


@pytest.mark.skipif(not REAL_PTB, reason="python-telegram-bot is not installed (stubbed in conftest)")
def test_telegram_dispatcher_serves_other_users_while_one_is_slow_and_keeps_order():
    """Feeds real telegram.Update objects through PTB's own update queue, offline."""
    from telegram import Update, User

    a1_started = threading.Event()
    release_a1 = threading.Event()
    events = []

    def run(user_id, text):
        events.append(("start", user_id, text))
        if text == "a1":
            a1_started.set()
            assert release_a1.wait(5)
        events.append(("end", user_id, text))

    bot = TelegramBot(manager=_Manager(run=run))
    sent = []

    async def fake_send_blocks(user_id, blocks, original_message):
        sent.append(user_id)

    bot._send_blocks = fake_send_blocks
    app = bot.application

    try:
        # Skip Telegram's network handshake (getMe) by providing the bot user directly.
        with app.bot._unfrozen():
            app.bot._bot_user = User(id=123456, is_bot=True, first_name="test")
        app._initialized = True
    except AttributeError as exc:  # pragma: no cover - PTB internals changed
        pytest.skip(f"python-telegram-bot internals changed: {exc}")

    def update(update_id, chat_id, text):
        return Update.de_json({
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": int(time.time()),
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "is_bot": False, "first_name": "user"},
                "text": text,
            },
        }, app.bot)

    async def wait_until(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.02)

    async def scenario():
        await app.start()
        try:
            for update_id, (chat_id, text) in enumerate([(1, "a1"), (1, "a2"), (2, "b1"), (1, "a3")], 1):
                await app.update_queue.put(update(update_id, chat_id, text))
            await _wait_thread_event(a1_started)
            await wait_until(lambda: ("end", "2", "b1") in events)
            assert ("end", "1", "a1") not in events  # b1 finished while a1 was still running
            release_a1.set()
            await wait_until(lambda: len([e for e in events if e[0] == "end"]) == 4)
        finally:
            release_a1.set()
            await app.stop()

    asyncio.run(scenario())

    user_1 = [e for e in events if e[1] == "1"]
    assert user_1 == [
        ("start", "1", "a1"), ("end", "1", "a1"),
        ("start", "1", "a2"), ("end", "1", "a2"),
        ("start", "1", "a3"), ("end", "1", "a3"),
    ]
    assert sorted(sent) == ["1", "1", "1", "2"]
