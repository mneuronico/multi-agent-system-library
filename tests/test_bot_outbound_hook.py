"""The ``outbound`` hook sees every message a bot sends and can deliver it through another channel.

CasanChef's WhatsApp number was migrated to another provider: replies built by the agents went
out through ``on_update``, but command answers (``/clear``), the error message and media notices
were still posted to Meta's API, which rejects them, so users got nothing.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from mas.bots import WhatsappBot
from mas.manager import AgentSystemManager


def _message(body, message_id="wamid.1"):
    return {"from": "5491100000000", "id": message_id, "type": "text", "timestamp": str(int(time.time())),
            "text": {"body": body}}


def _webhook(*messages):
    return {"entry": [{"changes": [{"value": {"messages": list(messages)}}]}]}


def _bot(tmp_path, outbound, **kwargs):
    manager = AgentSystemManager(base_directory=str(tmp_path))
    bot = WhatsappBot(
        manager=manager, whatsapp_token="t", phone_number_id="p", webhook_verify_token="v",
        outbound=outbound, ensure_delivery=True, **kwargs,
    )
    platform = []

    async def fake_send_blocks(user_id, blocks, original_message):
        platform.append((user_id, blocks))

    bot._send_blocks = fake_send_blocks
    return bot, platform


def _texts(blocks):
    texts = []
    for block in blocks:
        content = block.get("content")
        texts.append(content.get("response") if isinstance(content, dict) else content)
    return texts


def test_a_command_answer_goes_through_the_hook_and_not_the_platform(workspace_tmp_path):
    seen = []

    def outbound(user_id, blocks, original_message):
        seen.append((user_id, _texts(blocks), original_message.get("message_id")))
        return None  # delivered elsewhere

    bot, platform = _bot(workspace_tmp_path, outbound, on_clear_msg="Listo, borré la conversación.")
    asyncio.run(bot.process_webhook_update(_webhook(_message("/clear"))))

    assert seen == [("5491100000000", ["Listo, borré la conversación."], "wamid.1")]
    assert platform == []


def test_the_error_message_goes_through_the_hook(workspace_tmp_path):
    seen = []
    bot, platform = _bot(workspace_tmp_path, lambda u, b, m: seen.append(_texts(b)), error_msg="Ups, algo salió mal.")

    def failing_run(*args, **kwargs):
        raise RuntimeError("model down")

    bot.manager.run = failing_run
    asyncio.run(bot.process_webhook_update(_webhook(_message("hola"))))

    assert seen == [["Ups, algo salió mal."]]
    assert platform == []


def test_returning_blocks_sends_them_through_the_platform(workspace_tmp_path):
    def outbound(user_id, blocks, original_message):
        return blocks + [{"type": "text", "content": {"response": "(copia)"}}]

    bot, platform = _bot(workspace_tmp_path, outbound, on_clear_msg="Borrado.")
    asyncio.run(bot.process_webhook_update(_webhook(_message("/clear"))))

    assert [_texts(blocks) for _, blocks in platform] == [["Borrado.", "(copia)"]]


def test_returning_the_same_blocks_only_observes_them(workspace_tmp_path):
    seen = []

    def outbound(user_id, blocks, original_message):
        seen.append(_texts(blocks))
        return blocks

    bot, platform = _bot(workspace_tmp_path, outbound, on_clear_msg="Borrado.")
    asyncio.run(bot.process_webhook_update(_webhook(_message("/clear"))))

    assert seen == [["Borrado."]]
    assert [_texts(blocks) for _, blocks in platform] == [["Borrado."]]


def test_a_function_reference_is_resolved_from_the_functions_file(workspace_tmp_path):
    (workspace_tmp_path / "fns.py").write_text(
        "SENT = []\n"
        "def my_sender(user_id, blocks, original_message):\n"
        "    SENT.append((user_id, len(blocks)))\n",
        encoding="utf-8",
    )
    bot, platform = _bot(workspace_tmp_path, "fn:my_sender", on_clear_msg="Borrado.")
    asyncio.run(bot.process_webhook_update(_webhook(_message("/clear"))))

    assert platform == []
    assert bot.outbound.__globals__["SENT"] == [("5491100000000", 1)]


def test_an_unknown_reference_fails_when_the_bot_is_created(workspace_tmp_path):
    with pytest.raises(ValueError):
        _bot(workspace_tmp_path, "fn:does_not_exist")
    with pytest.raises(ValueError):
        _bot(workspace_tmp_path, 42)


def test_without_a_hook_messages_go_to_the_platform(workspace_tmp_path):
    bot, platform = _bot(workspace_tmp_path, None, on_clear_msg="Borrado.")
    asyncio.run(bot.process_webhook_update(_webhook(_message("/clear"))))

    assert [_texts(blocks) for _, blocks in platform] == [["Borrado."]]
