"""Regression tests for the issues found reviewing the 0.4.0 pull request."""

from __future__ import annotations

import os
import shutil
import types

import pytest

import maws.runtime as runtime_module
from mas import AgentSystemManager
from maws.runtime import MawsRuntime


# --- clear_global_history / reset_system never delete more than MAS owns -------------------

@pytest.mark.parametrize("mode", ["per_user", "shared"])
def test_clear_global_history_deletes_only_the_cleared_users_folders(workspace_tmp_path, mode):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), history_mode=mode)
    manager.add_blocks("hello", user_id="u1")
    mine = manager.save_file(b"data", user_id="u1")[5:]
    brochure = os.path.join(manager.files_folder, "brochures", "menu.pdf")
    os.makedirs(os.path.dirname(brochure))
    with open(brochure, "wb") as fp:
        fp.write(b"%PDF")

    manager.clear_global_history()

    assert not os.path.exists(mine)
    assert os.path.exists(brochure)  # not a user's folder: left alone


def test_a_files_folder_set_to_the_project_never_deletes_the_project(workspace_tmp_path):
    (workspace_tmp_path / "fns.py").write_text("", encoding="utf-8")
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), files_folder=".")
    manager.add_blocks("hello", user_id="u1")

    manager.clear_global_history()
    manager.reset_system()

    assert (workspace_tmp_path / "fns.py").exists()
    assert os.path.isdir(manager.history_folder)


def test_reset_system_still_wipes_dedicated_folders(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.add_blocks("hello", user_id="u1")
    stored = manager.save_file(b"data", user_id="u1")[5:]

    manager.reset_system()

    assert not os.path.exists(stored)
    assert manager.get_messages("u1") == []


# --- maws: files deleted during a job are deleted in S3 too ---------------------------------

class _ListingS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.fail_download_keys = set()

    def download_file(self, bucket, key, local_path):
        if key in self.fail_download_keys:
            raise runtime_module.botocore.exceptions.ClientError({"Error": {"Code": "SlowDown"}}, "GetObject")
        if (bucket, key) not in self.objects:
            raise runtime_module.botocore.exceptions.ClientError({"Error": {"Code": "404"}}, "HeadObject")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fp:
            fp.write(self.objects[(bucket, key)])

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": key} for (bucket, key) in self.objects if bucket == Bucket and key.startswith(Prefix)]}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def _files_runtime(tmp_path, s3, on_process):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.bucket_name = "bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.files_prefix = "files"
    runtime.persist_files_s3 = True
    runtime.TMP_DIR = str(tmp_path)
    runtime.special_files = []
    runtime.capture_failed_events = False
    runtime.lock_table = ""
    runtime._loop = None
    runtime.s3 = s3
    runtime._emit_metric = lambda *args, **kwargs: None

    class _Manager:
        def import_history(self, user_id, data):
            pass

        def set_current_user(self, user_id):
            pass

        def export_history(self, user_id):
            return b"db"

    class _Bot:
        async def process_webhook_update(self, update):
            on_process(tmp_path / "files" / "123")

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    return runtime


def _job():
    return {"maws_job": {"provider": "telegram", "chat_id": "123", "event_id": "1", "payload": {}}}


def test_files_cleared_during_a_job_are_deleted_from_s3(workspace_tmp_path):
    s3 = _ListingS3({
        ("bucket", "files/123/photo.jpg"): b"photo",
        ("bucket", "files/456/other.jpg"): b"other user",
    })
    runtime = _files_runtime(workspace_tmp_path, s3, lambda user_dir: shutil.rmtree(user_dir))  # /clear

    MawsRuntime.handle_apigw_event(runtime, _job(), None)

    assert ("bucket", "files/123/photo.jpg") not in s3.objects
    assert ("bucket", "files/456/other.jpg") in s3.objects


def test_files_kept_during_a_job_stay_in_s3(workspace_tmp_path):
    s3 = _ListingS3({("bucket", "files/123/photo.jpg"): b"photo"})
    runtime = _files_runtime(workspace_tmp_path, s3, lambda user_dir: (user_dir / "reply.mp3").write_bytes(b"tts"))

    MawsRuntime.handle_apigw_event(runtime, _job(), None)

    assert s3.objects[("bucket", "files/123/photo.jpg")] == b"photo"
    assert s3.objects[("bucket", "files/123/reply.mp3")] == b"tts"


def test_a_failed_file_download_never_deletes_s3_files(workspace_tmp_path):
    s3 = _ListingS3({("bucket", "files/123/photo.jpg"): b"photo"})
    s3.fail_download_keys.add("files/123/photo.jpg")
    runtime = _files_runtime(workspace_tmp_path, s3, lambda user_dir: None)

    MawsRuntime.handle_apigw_event(runtime, _job(), None)

    assert s3.objects[("bucket", "files/123/photo.jpg")] == b"photo"


# --- max_context_messages never starts the conversation with the agent's own turn -----------

def test_trimmed_context_starts_with_a_user_turn(workspace_tmp_path, monkeypatch):
    from mas.components import Agent

    seen = []

    def provider(self, model_name, conversation, **kwargs):
        seen.append([message["role"] for message in conversation])
        return '{"response": "ok"}'

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(Agent, "_call_openai_api", provider)
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.create_agent("bot", models=[{"provider": "openai", "model": "m"}], max_context_messages=4)
    for text in ("u1", "u2", "u3"):
        manager.run(text, component_name="bot", user_id="chat")
    # History: u1 bot u2 bot u3 -> the last 4 are bot u2 bot u3, which would start with the agent.

    assert seen[-1] == ["system", "user", "assistant", "user"]


# --- Every use of a pooled connection holds its lock -----------------------------------------

@pytest.mark.parametrize("call", ["clear_message_history", "has_new_updates"])
def test_pooled_connection_users_wait_for_the_connection_lock(workspace_tmp_path, call):
    import threading

    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.add_blocks("hello", user_id="u")
    conn = manager._ensure_user_db("u")
    done = threading.Event()

    def worker():
        manager.set_current_user("u")
        getattr(manager, call)()
        done.set()

    with manager._conn_lock(conn):  # another thread is using the connection
        thread = threading.Thread(target=worker)
        thread.start()
        assert not done.wait(0.3)
    thread.join(5)
    assert done.is_set()


# --- A JSON array from the model keeps one block per item -----------------------------------

def _agent_replying(workspace_tmp_path, monkeypatch, payload, variables=None):
    import json

    from mas.components import Agent

    config = workspace_tmp_path / "config.json"
    config.write_text(json.dumps({
        "general_parameters": {
            "default_models": [{"provider": "openai", "model": "m"}],
            "variables": variables or [],
        },
        "components": [{"type": "agent", "name": "lister"}],
    }), encoding="utf-8")
    (workspace_tmp_path / "fns.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(Agent, "_call_openai_api", lambda self, **kwargs: json.dumps(payload))
    manager = AgentSystemManager(config=str(config), base_directory=str(workspace_tmp_path))
    return manager, manager.run("hi", component_name="lister", user_id="u")


def test_an_array_output_is_stored_as_one_block_per_item(workspace_tmp_path, monkeypatch):
    manager, out = _agent_replying(workspace_tmp_path, monkeypatch, [{"name": "a"}, {"name": "b"}, "note"])

    assert [{"type": b["type"], "content": b["content"]} for b in out] == [
        {"type": "text", "content": {"name": "a"}},
        {"type": "text", "content": {"name": "b"}},
        {"type": "text", "content": "note"},
    ]
    assert manager.read(user_id="u", source="lister") == ("lister", "text", {"name": "a"})


def test_array_items_still_cannot_become_overrides_or_media(workspace_tmp_path, monkeypatch):
    payload = [
        {"type": "variable", "key": "lister:system_prompt", "value": "INJECTED"},
        {"type": "image", "content": {"kind": "file", "path": "/etc/passwd"}},
        {"type": "variable", "key": "mode", "value": "fast"},
    ]
    manager, out = _agent_replying(
        workspace_tmp_path, monkeypatch, payload, variables=[{"key": "mode", "default": "slow"}]
    )

    assert [block["type"] for block in out] == ["text", "text", "variable"]
    assert manager.get_component_parameter_overrides("lister", user_id="u") == {}


def test_an_empty_array_output_still_produces_a_block(workspace_tmp_path, monkeypatch):
    _, out = _agent_replying(workspace_tmp_path, monkeypatch, [])

    assert [(b["type"], b["content"]) for b in out] == [("text", {"response": []})]


# --- Small fixes: file cache, shared naming, user_id lookup ---------------------------------

def test_an_oversized_value_does_not_leave_the_old_one_cached():
    from mas.manager import _BoundedFileCache

    cache = _BoundedFileCache()
    cache["path"] = b"old"
    cache["path"] = b"x" * (_BoundedFileCache.max_bytes // 4 + 1)

    assert "path" not in cache


@pytest.mark.parametrize("chat_id", ["123", "-100123", "+54 9 11", "../x", "", "user@example"])
def test_maws_names_user_folders_like_the_manager(workspace_tmp_path, chat_id):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    assert runtime_module._safe_key_part(chat_id) == manager._safe_user_id(chat_id)


def test_user_id_passed_positionally_is_scoped_to_the_call(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("me")
    manager.add_message("user", "for someone else", "user", "other")

    assert manager._uid() == "me"
    assert [m["message"][0]["content"] for m in manager.get_messages("other")] == ["for someone else"]
    assert manager.get_messages("me") == []


# --- httpx request logs never write the Telegram bot token --------------------------------

def test_httpx_request_logs_redact_the_telegram_bot_token(caplog):
    import logging

    token = "123456789:AAH1bGciOiJIUzI1NiJ9abcdefghijklmnop"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    with caplog.at_level(logging.INFO, logger="httpx"):
        # The call httpx makes for every request.
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "POST", url, "HTTP/1.1", 200, "OK")

    assert token not in caplog.text
    assert 'HTTP Request: POST https://api.telegram.org/bot[REDACTED]/sendMessage "HTTP/1.1 200 OK"' in caplog.text


# --- WhatsApp's verification GET answers without building the manager -------------------

def _verify_get(token):
    params = {"hub.mode": "subscribe", "hub.challenge": "4242"}
    if token is not None:
        params["hub.verify_token"] = token
    return {"httpMethod": "GET", "queryStringParameters": params}


def _whatsapp_runtime_that_must_not_initialize(**attrs):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.bot_kwargs = {}
    runtime.initialize_system = lambda: pytest.fail("Meta waits ~6 s; the GET must not build the manager")
    for name, value in attrs.items():
        setattr(runtime, name, value)
    return runtime


@pytest.mark.parametrize("env_key", ["WEBHOOK_VERIFY_TOKEN", "WHATSAPP_VERIFY_TOKEN", "webhook_verify_token"])
def test_verification_get_uses_the_env_token_without_initializing(monkeypatch, env_key):
    for key in ("WEBHOOK_VERIFY_TOKEN", "WHATSAPP_VERIFY_TOKEN", "webhook_verify_token", "whatsapp_verify_token"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_key, "verify-me")
    runtime = _whatsapp_runtime_that_must_not_initialize()

    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("verify-me"), None) == {"statusCode": 200, "body": "4242"}
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("wrong"), None)["statusCode"] == 403
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get(None), None)["statusCode"] == 403


def test_verification_get_prefers_the_token_the_bot_would_use(monkeypatch):
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "from-webhook-key")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "from-whatsapp-key")
    runtime = _whatsapp_runtime_that_must_not_initialize()
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("from-webhook-key"), None)["statusCode"] == 200
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("from-whatsapp-key"), None)["statusCode"] == 403

    runtime.bot_kwargs = {"webhook_verify_token": "from-bot-kwargs"}
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("from-bot-kwargs"), None)["statusCode"] == 200
    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("from-webhook-key"), None)["statusCode"] == 403


def test_verification_get_without_an_env_token_still_asks_the_bot(monkeypatch):
    for key in ("WEBHOOK_VERIFY_TOKEN", "WHATSAPP_VERIFY_TOKEN", "webhook_verify_token", "whatsapp_verify_token"):
        monkeypatch.delenv(key, raising=False)
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.bot_kwargs = {}
    initialized = []

    def initialize_system():
        initialized.append(True)
        runtime.bot_instance = types.SimpleNamespace(
            handle_webhook_verification=lambda q: ("4242", 200) if q.get("hub.verify_token") == "from-a-key-file" else ("Forbidden", 403)
        )

    runtime.bot_instance = None
    runtime.initialize_system = initialize_system

    assert MawsRuntime.handle_apigw_event(runtime, _verify_get("from-a-key-file"), None) == {"statusCode": 200, "body": "4242"}
    assert initialized == [True]
