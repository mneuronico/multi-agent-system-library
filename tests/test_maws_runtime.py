from __future__ import annotations

import json
import hmac
import hashlib
import time
import types

import pytest

import maws.runtime as runtime_module
from maws.maws import MawsRuntime


class _Context:
    invoked_function_arn = "arn:aws:lambda:region:acct:function:test"


class _FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def list_objects_v2(self, Bucket, Prefix):
        contents = [
            {"Key": key}
            for (bucket, key) in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents}

    def download_file(self, bucket, key, local_path):
        data = self.objects[(bucket, key)]
        import os

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fp:
            fp.write(data)

    def put_object(self, Bucket, Key, Body):
        self.puts[(Bucket, Key)] = Body


def test_apigw_post_acknowledges_without_initializing_system():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: {"StatusCode": 202}
    )

    def initialize_system():
        raise AssertionError("initial POST should not initialize MAS")

    runtime.initialize_system = initialize_system

    event = {
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"entry": []}),
    }

    response = MawsRuntime.handle_apigw_event(runtime, event, _Context())

    assert response["statusCode"] == 200


def test_whatsapp_post_splits_batches_into_per_message_jobs():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.busy_policy = "drop"
    invocations = []
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(kwargs) or {"StatusCode": 202}
    )

    body = {
        "entry": [{
            "id": "entry-1",
            "changes": [{
                "value": {
                    "metadata": {"phone_number_id": "phone"},
                    "messages": [
                        {"id": "wamid.1", "from": "user-a", "type": "text", "text": {"body": "one"}},
                        {"id": "wamid.2", "from": "user-b", "type": "text", "text": {"body": "two"}},
                    ],
                }
            }],
        }]
    }

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"requestContext": {"http": {"method": "POST"}}, "body": json.dumps(body)},
        _Context(),
    )

    assert response["statusCode"] == 200
    assert len(invocations) == 2
    jobs = [json.loads(call["Payload"])["maws_job"] for call in invocations]
    assert [job["chat_id"] for job in jobs] == ["user-a", "user-b"]
    assert [job["event_id"] for job in jobs] == ["wamid.1", "wamid.2"]
    assert all(len(job["payload"]["entry"][0]["changes"][0]["value"]["messages"]) == 1 for job in jobs)


def test_fifo_policy_enqueues_jobs_with_stable_group_and_dedupe():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.busy_policy = "fifo"
    runtime.queue_url = "https://sqs.test/queue.fifo"
    runtime.history_mode = "per_user"
    sent = []
    runtime.sqs = types.SimpleNamespace(
        send_message=lambda **kwargs: sent.append(kwargs) or {"MessageId": "1"}
    )

    event = {
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"update_id": 99, "message": {"message_id": 3, "chat": {"id": 123}, "text": "hi"}}),
    }

    response = MawsRuntime.handle_apigw_event(runtime, event, _Context())

    assert response["statusCode"] == 200
    assert len(sent) == 1
    assert sent[0]["QueueUrl"] == runtime.queue_url
    assert sent[0]["MessageGroupId"] == "123"
    assert len(sent[0]["MessageDeduplicationId"]) == 64
    assert json.loads(sent[0]["MessageBody"])["maws_job"]["event_id"] == "99"


def test_telegram_secret_token_is_enforced_only_when_configured():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.telegram_secret_token = "expected"

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "headers": {"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            "body": json.dumps({"message": {"chat": {"id": 1}}}),
        },
        _Context(),
    )

    assert response["statusCode"] == 403


def test_whatsapp_signature_verification_uses_raw_body():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.whatsapp_verify_signature = True
    runtime.whatsapp_app_secret = "app-secret"
    runtime.busy_policy = "drop"
    invocations = []
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(kwargs) or {"StatusCode": 202}
    )
    body = json.dumps({
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"id": "wamid.1", "from": "user-a", "type": "text", "text": {"body": "hi"}}]
                }
            }]
        }]
    }, separators=(",", ":"))
    signature = "sha256=" + hmac.new(b"app-secret", body.encode("utf-8"), hashlib.sha256).hexdigest()

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "headers": {"x-hub-signature-256": signature},
            "body": body,
        },
        _Context(),
    )

    assert response["statusCode"] == 200
    assert len(invocations) == 1


def test_whatsapp_bad_signature_is_rejected_before_invocation():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.whatsapp_verify_signature = True
    runtime.whatsapp_app_secret = "app-secret"
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not invoke"))
    )

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "headers": {"x-hub-signature-256": "sha256=bad"},
            "body": json.dumps({"entry": []}),
        },
        _Context(),
    )

    assert response["statusCode"] == 403


def test_history_append_requires_valid_hmac_and_queues_a_worker_job():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = True
    runtime.history_write_secret = "history-secret"
    runtime.busy_policy = "drop"
    invocations = []
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(kwargs) or {"StatusCode": 202}
    )
    body = json.dumps({
        "chat_id": "5491100000000",
        "event_id": "wamid.outbound-1",
        "content": "Hola desde el bulk",
    }, separators=(",", ":"))
    timestamp = str(int(time.time()))
    signature = "sha256=" + hmac.new(
        b"history-secret",
        timestamp.encode("ascii") + b"." + body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/Prod/history/outgoing",
            "headers": {
                "X-MAWS-Timestamp": timestamp,
                "X-MAWS-Signature": signature,
            },
            "body": body,
        },
        _Context(),
    )

    assert response["statusCode"] == 202
    assert len(invocations) == 1
    job = json.loads(invocations[0]["Payload"])["maws_job"]
    assert job["operation"] == "append_history"
    assert job["chat_id"] == "5491100000000"
    assert job["event_id"] == "wamid.outbound-1"
    assert job["payload"]["content"] == "Hola desde el bulk"


def test_history_append_rejects_invalid_signature():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = True
    runtime.history_write_secret = "history-secret"
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not invoke"))
    )
    body = json.dumps({
        "chat_id": "user-1",
        "event_id": "evt-1",
        "content": "sent",
    })

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/Prod/history/outgoing",
            "headers": {
                "X-MAWS-Timestamp": str(int(time.time())),
                "X-MAWS-Signature": "sha256=bad",
            },
            "body": body,
        },
        _Context(),
    )

    assert response["statusCode"] == 403


def test_history_append_requires_configured_secret():
    runtime = object.__new__(MawsRuntime)
    runtime.history_append_enabled = True
    body = json.dumps({
        "chat_id": "user-1",
        "event_id": "evt-1",
        "content": "sent",
    })
    timestamp = str(int(time.time()))

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/Prod/history/outgoing",
            "headers": {
                "X-MAWS-Timestamp": timestamp,
                "X-MAWS-Signature": "sha256=unused",
            },
            "body": body,
        },
        _Context(),
    )

    assert response["statusCode"] == 503


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"message": {"chat": {"id": 123}}}, "123"),
        ({"edited_message": {"chat": {"id": 456}}}, ""),
    ],
)
def test_extract_telegram_chat_id_contract(event, expected):
    assert MawsRuntime._extract_telegram_chat_id(event) == expected


def test_s3_sqlite_key_sanitizes_chat_id():
    runtime = object.__new__(MawsRuntime)
    key = MawsRuntime.s3_sqlite_key(runtime, "../escape")

    assert key.startswith("history/")
    assert ".." not in key
    assert key.count("/") == 1


def test_worker_event_without_chat_id_returns_explicit_response():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"

    def initialize_system():
        raise AssertionError("no-chat worker should not initialize MAS")

    runtime.initialize_system = initialize_system

    response = MawsRuntime.handle_apigw_event(runtime, {"update": {"edited_message": {}}}, _Context())

    assert response["statusCode"] == 200
    assert "No chat_id" in json.loads(response["body"])


def test_worker_event_requires_bucket_before_initializing_system():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.bucket_name = ""

    def initialize_system():
        raise AssertionError("missing-bucket worker should fail before MAS init")

    runtime.initialize_system = initialize_system

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"update": {"message": {"chat": {"id": 123}}}},
        _Context(),
    )

    assert response["statusCode"] == 500
    assert "BUCKET_NAME" in json.loads(response["body"])


def test_worker_lock_denied_returns_explicit_response():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.bucket_name = "history-bucket"
    runtime.TMP_DIR = "/tmp"
    runtime.bot_instance = object()

    def initialize_system():
        runtime.manager = object()

    runtime.initialize_system = initialize_system
    runtime.try_acquire_user_lock = lambda chat_id: False

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"update": {"message": {"chat": {"id": 123}}}},
        _Context(),
    )

    assert response["statusCode"] == 200
    assert "Already processing" in json.loads(response["body"])


def test_worker_imports_processes_and_exports_per_user_history(workspace_tmp_path):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime._loop = None
    runtime.s3 = _FakeS3({
        ("history-bucket", "history/123.sqlite"): b"old-history",
    })
    calls = []

    class _Manager:
        def import_history(self, user_id, sqlite_bytes):
            calls.append(("import", user_id, sqlite_bytes))

        def set_current_user(self, user_id):
            calls.append(("current", user_id))

        def export_history(self, user_id):
            calls.append(("export", user_id))
            return b"new-history"

    class _Bot:
        async def process_webhook_update(self, update):
            calls.append(("process", update))

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    runtime.try_acquire_user_lock = lambda chat_id: True
    runtime.release_user_lock = lambda chat_id: calls.append(("release", chat_id))

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"maws_job": {
            "provider": "telegram",
            "bot_type": "telegram",
            "chat_id": "123",
            "event_id": "evt-1",
            "payload": {"message": {"chat": {"id": 123}}},
        }},
        _Context(),
    )

    assert response["statusCode"] == 200
    assert ("import", "123", b"old-history") in calls
    assert ("process", {"message": {"chat": {"id": 123}}}) in calls
    assert ("export", "123") in calls
    assert ("release", "123") in calls
    assert runtime.s3.puts[("history-bucket", "history/123.sqlite")] == b"new-history"


def test_worker_appends_outgoing_history_without_processing_webhook(workspace_tmp_path):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = True
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime._loop = None
    runtime.s3 = _FakeS3({
        ("history-bucket", "history/5491100000000.sqlite"): b"old-history",
    })
    calls = []

    class _Manager:
        agents = {"chef": object()}

        def import_history(self, user_id, sqlite_bytes):
            calls.append(("import", user_id, sqlite_bytes))

        def set_current_user(self, user_id):
            calls.append(("current", user_id))

        def get_messages(self, user_id):
            return []

        def add_blocks(self, content, **kwargs):
            calls.append(("add_blocks", content, kwargs))

        def export_history(self, user_id):
            calls.append(("export", user_id))
            return b"history-with-outgoing"

    class _Bot:
        async def process_webhook_update(self, update):
            raise AssertionError("outgoing history must not process a webhook")

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    runtime.try_acquire_user_lock = lambda chat_id: True
    runtime.release_user_lock = lambda chat_id: calls.append(("release", chat_id))

    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"maws_job": {
            "provider": "history",
            "chat_id": "5491100000000",
            "event_id": "wamid.outbound-1",
            "operation": "append_history",
            "payload": {"content": "Hola desde el bulk"},
        }},
        _Context(),
    )

    assert response["statusCode"] == 200
    assert ("add_blocks", [{
        "type": "text",
        "content": {"response": "Hola desde el bulk"},
        "metadata": {"maws_event_id": "wamid.outbound-1"},
    }], {
        "role": "chef",
        "msg_type": "agent",
        "user_id": "5491100000000",
    }) in calls
    assert ("export", "5491100000000") in calls
    assert ("release", "5491100000000") in calls
    assert runtime.s3.puts[("history-bucket", "history/5491100000000.sqlite")] == b"history-with-outgoing"


def test_history_append_deduplicates_existing_event_id():
    runtime = object.__new__(MawsRuntime)
    runtime.manager = types.SimpleNamespace(get_messages=lambda user_id: [{
        "message": [{"type": "text", "content": {"response": "sent"},
                     "metadata": {"maws_event_id": "evt-1"}}]
    }])

    assert MawsRuntime._history_contains_external_event(runtime, "user-1", "evt-1")
    assert not MawsRuntime._history_contains_external_event(runtime, "user-1", "evt-2")


def test_optional_user_file_s3_persistence_round_trips(workspace_tmp_path):
    runtime = object.__new__(MawsRuntime)
    runtime.bucket_name = "history-bucket"
    runtime.persist_files_s3 = True
    runtime.files_prefix = "files"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.s3 = _FakeS3({
        ("history-bucket", "files/user-a/inbound.txt"): b"inbound",
    })

    MawsRuntime._sync_user_files_from_s3(runtime, "user-a")
    local_dir = workspace_tmp_path / "files" / "user-a"
    assert (local_dir / "inbound.txt").read_bytes() == b"inbound"

    (local_dir / "outbound.txt").write_bytes(b"outbound")
    MawsRuntime._sync_user_files_to_s3(runtime, "user-a")

    assert runtime.s3.puts[("history-bucket", "files/user-a/inbound.txt")] == b"inbound"
    assert runtime.s3.puts[("history-bucket", "files/user-a/outbound.txt")] == b"outbound"


def test_shared_history_sync_uses_shared_prefix_and_global_group(workspace_tmp_path):
    runtime = object.__new__(MawsRuntime)
    runtime.bucket_name = "history-bucket"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.shared_history_prefix = "history/shared"
    runtime.history_mode = "shared"
    runtime.s3 = _FakeS3({
        ("history-bucket", "history/shared/shared_history_000001.sqlite"): b"db-one",
    })

    MawsRuntime._sync_shared_history_from_s3(runtime)
    downloaded = workspace_tmp_path / "history" / "shared_history_000001.sqlite"
    assert downloaded.read_bytes() == b"db-one"
    assert MawsRuntime._queue_group_id_for(runtime, {"chat_id": "user-a"}) == "maws-shared-history"

    downloaded.write_bytes(b"db-two")
    runtime.manager = types.SimpleNamespace(
        _db_pool={},
        _list_shared_history_db_paths=lambda: [str(downloaded)],
        _shared_history_db_key=lambda path: f"shared:{path}",
    )
    MawsRuntime._upload_shared_history_to_s3(runtime)
    assert runtime.s3.puts[("history-bucket", "history/shared/shared_history_000001.sqlite")] == b"db-two"


def test_sqs_worker_raises_processing_errors_for_retry():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"

    def fail_process_jobs(jobs, context=None, raise_on_error=False):
        assert raise_on_error is True
        raise RuntimeError("retry me")

    runtime._process_jobs = fail_process_jobs
    event = {
        "Records": [{
            "eventSource": "aws:sqs",
            "body": json.dumps({
                "maws_job": {
                    "provider": "telegram",
                    "bot_type": "telegram",
                    "chat_id": "123",
                    "event_id": "99",
                    "payload": {"message": {"chat": {"id": 123}}},
                }
            }),
        }]
    }

    with pytest.raises(RuntimeError, match="retry me"):
        MawsRuntime.handle_apigw_event(runtime, event, _Context())


def test_failed_self_invoke_can_be_captured_to_s3():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "telegram"
    runtime.busy_policy = "drop"
    runtime.capture_failed_events = True
    runtime.bucket_name = "history-bucket"
    runtime.failed_events_prefix = "failed-events"
    runtime.s3 = _FakeS3()

    def fail_invoke(**kwargs):
        raise RuntimeError("invoke failed")

    runtime.lambda_client = types.SimpleNamespace(invoke=fail_invoke)
    event = {
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"update_id": 42, "message": {"chat": {"id": 123}, "text": "hi"}}),
    }

    response = MawsRuntime.handle_apigw_event(runtime, event, _Context())

    assert response["statusCode"] == 200
    assert "warning" in json.loads(response["body"])
    [(bucket, key)] = list(runtime.s3.puts)
    assert bucket == "history-bucket"
    assert key.startswith("failed-events/")
    assert key.endswith("/telegram/42.json")


def test_initialize_system_applies_runtime_manager_and_bot_kwargs(monkeypatch, workspace_tmp_path):
    captured = {}

    class _App:
        async def initialize(self):
            return None

        async def start(self):
            return None

    class _Bot:
        application = _App()

    class _Manager:
        def __init__(self, **kwargs):
            captured["manager_kwargs"] = kwargs

        def start_telegram_bot(self, start_polling=False, **kwargs):
            captured["start_polling"] = start_polling
            captured["bot_kwargs"] = kwargs
            return _Bot()

    monkeypatch.setattr(runtime_module, "AgentSystemManager", _Manager)

    runtime = object.__new__(MawsRuntime)
    runtime.special_files = []
    runtime.config_path = "custom.json"
    runtime.CODE_ROOT = str(workspace_tmp_path)
    runtime.TMP_DIR = str(workspace_tmp_path / "tmp")
    runtime.verbose = False
    runtime.history_mode = "shared"
    runtime.history_rotation = "time_period"
    runtime.history_max_messages = 10
    runtime.history_period = "1d"
    runtime.manager_kwargs = {"admin_user_id": "admin", "usage_logging": True}
    runtime.bot_kwargs = {"unknown_command_msg": None}
    runtime.ensure_delivery = True
    runtime.delivery_timeout = 12.5
    runtime.max_allowed_message_delay = 600.0
    runtime.bot_type = "telegram"
    runtime.manager = None
    runtime.bot_instance = None
    runtime._loop = None

    MawsRuntime.initialize_system(runtime)

    assert captured["manager_kwargs"]["admin_user_id"] == "admin"
    assert captured["manager_kwargs"]["usage_logging"] is True
    assert captured["manager_kwargs"]["history_mode"] == "shared"
    assert captured["manager_kwargs"]["history_rotation"] == "time_period"
    assert captured["manager_kwargs"]["config"] == "custom.json"
    assert captured["start_polling"] is False
    assert captured["bot_kwargs"]["unknown_command_msg"] is None
    assert captured["bot_kwargs"]["ensure_delivery"] is True
    assert captured["bot_kwargs"]["delivery_timeout"] == 12.5
    assert captured["bot_kwargs"]["max_allowed_message_delay"] == 600.0


# -------------------------
# History append: opt-in isolation and hardening
# -------------------------
_HISTORY_BODY = json.dumps({"chat_id": "user-1", "event_id": "evt-1", "content": "sent"})
_HISTORY_JOB = {
    "provider": "history",
    "chat_id": "user-1",
    "event_id": "evt-1",
    "operation": "append_history",
    "payload": {"content": "sent"},
}


def _signed_history_event(body, secret="history-secret", timestamp=None):
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    signature = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "requestContext": {"http": {"method": "POST"}},
        "rawPath": "/Prod/history/outgoing",
        "headers": {"X-MAWS-Timestamp": timestamp, "X-MAWS-Signature": signature},
        "body": body,
    }


def _history_worker_runtime(workspace_tmp_path, *, enabled=True, lock_results=(True,), lock_wait=5.0):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = enabled
    runtime.history_append_lock_wait = lock_wait
    runtime.history_append_lock_poll = 0
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime._loop = None
    runtime.s3 = _FakeS3({("history-bucket", "history/user-1.sqlite"): b"old-history"})
    calls = []

    class _Manager:
        agents = {"chef": object()}

        def import_history(self, user_id, sqlite_bytes):
            calls.append(("import", user_id))

        def set_current_user(self, user_id):
            calls.append(("current", user_id))

        def get_messages(self, user_id):
            return []

        def add_blocks(self, content, **kwargs):
            calls.append(("add_blocks", content, kwargs))

        def export_history(self, user_id):
            calls.append(("export", user_id))
            return b"new-history"

    class _Bot:
        component_name = None

        async def process_webhook_update(self, update):
            calls.append(("process", update))

    pending_locks = list(lock_results)

    def try_acquire_user_lock(chat_id):
        calls.append(("lock", chat_id))
        return pending_locks.pop(0) if pending_locks else False

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    runtime.try_acquire_user_lock = try_acquire_user_lock
    runtime.release_user_lock = lambda chat_id: calls.append(("release", chat_id))
    return runtime, calls


def test_history_append_disabled_leaves_post_on_webhook_path():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_write_secret = "history-secret"
    calls = []
    runtime._handle_http_post = lambda event, context: calls.append("webhook") or {"statusCode": 200}

    def fail_history_append(event, context):
        raise AssertionError("disabled endpoint must not be routed")

    runtime._handle_history_append = fail_history_append

    response = MawsRuntime.handle_apigw_event(runtime, _signed_history_event(_HISTORY_BODY), _Context())

    assert response["statusCode"] == 200
    assert calls == ["webhook"]


def test_history_append_disabled_worker_job_keeps_webhook_processing(workspace_tmp_path):
    runtime, calls = _history_worker_runtime(workspace_tmp_path, enabled=False)

    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": dict(_HISTORY_JOB)}, _Context())

    assert response["statusCode"] == 200
    assert ("process", {"content": "sent"}) in calls
    assert not [call for call in calls if call[0] == "add_blocks"]


def test_history_append_rejects_stale_timestamp():
    runtime = object.__new__(MawsRuntime)
    runtime.history_append_enabled = True
    runtime.history_write_secret = "history-secret"
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not invoke"))
    )

    event = _signed_history_event(_HISTORY_BODY, timestamp=int(time.time()) - 301)
    response = MawsRuntime.handle_apigw_event(runtime, event, _Context())

    assert response["statusCode"] == 403


def test_history_append_fifo_enqueues_in_user_group():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = True
    runtime.history_write_secret = "history-secret"
    runtime.busy_policy = "fifo"
    runtime.queue_url = "https://sqs.test/queue.fifo"
    runtime.history_mode = "per_user"
    sent = []
    runtime.sqs = types.SimpleNamespace(send_message=lambda **kwargs: sent.append(kwargs) or {"MessageId": "1"})

    response = MawsRuntime.handle_apigw_event(runtime, _signed_history_event(_HISTORY_BODY), _Context())

    assert response["statusCode"] == 202
    assert len(sent) == 1
    assert sent[0]["MessageGroupId"] == "user-1"
    assert json.loads(sent[0]["MessageBody"])["maws_job"]["operation"] == "append_history"


def test_history_append_fifo_enqueue_failure_returns_502():
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.history_append_enabled = True
    runtime.history_write_secret = "history-secret"
    runtime.busy_policy = "fifo"
    runtime.queue_url = "https://sqs.test/queue.fifo"
    runtime.history_mode = "per_user"
    runtime.capture_failed_events = False

    def fail_send(**kwargs):
        raise RuntimeError("sqs down")

    runtime.sqs = types.SimpleNamespace(send_message=fail_send)

    response = MawsRuntime.handle_apigw_event(runtime, _signed_history_event(_HISTORY_BODY), _Context())

    assert response["statusCode"] == 502


def test_history_append_waits_for_busy_user_lock(workspace_tmp_path):
    runtime, calls = _history_worker_runtime(workspace_tmp_path, lock_results=(False, False, True))

    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": dict(_HISTORY_JOB)}, _Context())

    assert response["statusCode"] == 200
    assert [call for call in calls if call[0] == "lock"] == [("lock", "user-1")] * 3
    assert [call[0] for call in calls if call[0] == "add_blocks"] == ["add_blocks"]
    assert ("release", "user-1") in calls
    assert runtime.s3.puts[("history-bucket", "history/user-1.sqlite")] == b"new-history"


def test_history_append_lock_timeout_raises_for_retry_even_in_drop_mode(workspace_tmp_path):
    runtime, calls = _history_worker_runtime(workspace_tmp_path, lock_results=(False,), lock_wait=0)

    with pytest.raises(runtime_module.HistoryAppendLockBusy):
        MawsRuntime.handle_apigw_event(runtime, {"maws_job": dict(_HISTORY_JOB)}, _Context())

    assert not [call for call in calls if call[0] in ("import", "add_blocks", "release")]
    assert runtime.s3.puts == {}


def test_webhook_jobs_still_drop_on_busy_lock_when_append_enabled(workspace_tmp_path):
    runtime, calls = _history_worker_runtime(workspace_tmp_path, lock_results=(False,))
    job = {"provider": "whatsapp", "chat_id": "user-1", "event_id": "wamid.1", "payload": {"entry": []}}

    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": job}, _Context())

    assert response["statusCode"] == 200
    assert "Already processing" in json.loads(response["body"])
    assert [call for call in calls if call[0] == "lock"] == [("lock", "user-1")]
    assert not [call for call in calls if call[0] == "process"]


@pytest.mark.parametrize(
    ("configured", "component", "agents", "expected"),
    [
        ("chef", None, ["chef", "router"], ("chef", "agent")),
        ("", "chef", ["chef", "router"], ("chef", "agent")),
        ("", None, ["chef"], ("chef", "agent")),
        ("", None, ["chef", "router"], ("assistant", "assistant")),
        ("", "flow", ["chef", "router"], ("assistant", "assistant")),
        ("ghost", None, ["chef"], ("ghost", "assistant")),
    ],
)
def test_history_append_author_resolution(configured, component, agents, expected):
    runtime = object.__new__(MawsRuntime)
    runtime.history_append_role = configured
    runtime.manager = types.SimpleNamespace(agents={name: object() for name in agents})
    runtime.bot_instance = types.SimpleNamespace(component_name=component)

    assert MawsRuntime._history_append_author(runtime) == expected


def test_appended_message_is_the_agents_own_turn_in_real_manager(workspace_tmp_path):
    from mas import AgentSystemManager

    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.create_agent(name="chef", system="system", positive_filter=["user", "chef"])
    manager.create_agent(name="router", system="system", positive_filter=["user", "agent"])
    manager.add_blocks("hola", role="user", msg_type="user", user_id="user-1")

    runtime = object.__new__(MawsRuntime)
    runtime.manager = manager
    runtime.bot_instance = None
    runtime.history_append_role = "chef"
    MawsRuntime._append_external_history(runtime, "user-1", dict(_HISTORY_JOB))
    MawsRuntime._append_external_history(runtime, "user-1", dict(_HISTORY_JOB))  # retry is deduplicated

    assert len(manager.get_messages("user-1")) == 2
    rows = manager._get_all_messages(manager._get_user_db())
    chef = manager.agents["chef"]
    chef_view = chef._transform_to_conversation(chef._apply_filters(rows))
    assert chef_view[-1] == {"role": "assistant", "content": [{"type": "text", "content": {"response": "sent"}}]}
    router = manager.agents["router"]
    router_view = router._transform_to_conversation(router._apply_filters(rows))
    assert router_view[-1]["content"][-1] == {"type": "text", "content": {"response": "sent"}}
