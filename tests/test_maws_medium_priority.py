"""Regression tests for the MAWS medium-priority production fixes (Python side).

Covers: busy_policy=drop message loss (per-chat job groups, bounded lock wait, drop logging and EMF
metric), 5xx answers when a webhook update cannot be handed off, SQS worker errors that must not be
acknowledged, `maws remove` safety, stale local history, token-file uploads, redaction of
failed-event captures, the pinned SSM parameter version, params.json parsing, `--set` coercion,
`maws start` re-runs, opt-in installs and new-project bucket names.
bootstrap.sh changes are covered in tests/test_maws_bootstrap_script.py (needs bash + jq).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import types

import pytest

import maws.runtime as runtime_module
from maws import cli, operations
from maws.operations import MawsConfigError
from maws.runtime import MawsRuntime, MawsWorkerError


class _Context:
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:bot"

    def __init__(self, remaining_ms=None):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        if self.remaining_ms is None:
            raise RuntimeError("no deadline")
        return self.remaining_ms


def _client_error(code, message="error"):
    return runtime_module.botocore.exceptions.ClientError({"Error": {"Code": code, "Message": message}}, "op")


# -------------------------
# Fakes
# -------------------------
class _ConditionalCheckFailed(Exception):
    pass


class _FakeDynamo:
    def __init__(self):
        self.items = {}
        self.puts = 0
        self.exceptions = types.SimpleNamespace(ConditionalCheckFailedException=_ConditionalCheckFailed)

    def put_item(self, TableName, Item, ConditionExpression, ExpressionAttributeValues):
        self.puts += 1
        key = Item["user_id"]["S"]
        current = self.items.get(key)
        if ConditionExpression == "lockToken = :token":
            if current is None or current["lockToken"]["S"] != ExpressionAttributeValues[":token"]["S"]:
                raise _ConditionalCheckFailed()
        else:
            now = int(ExpressionAttributeValues[":now"]["N"])
            if current is not None and not int(current["expiresAt"]["N"]) < now:
                raise _ConditionalCheckFailed()
        self.items[key] = Item

    def delete_item(self, TableName, Key, ConditionExpression=None, ExpressionAttributeValues=None):
        key = Key["user_id"]["S"]
        current = self.items.get(key)
        if ConditionExpression and (current is None or current["lockToken"]["S"] != ExpressionAttributeValues[":token"]["S"]):
            raise _ConditionalCheckFailed()
        self.items.pop(key, None)


class _FakeS3:
    """S3 with ETags and conditional PutObject (If-Match / If-None-Match)."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []
        self.version = 0

    def _etag(self, body):
        return '"' + hashlib.md5(body).hexdigest() + '"'

    def download_file(self, bucket, key, local_path):
        if (bucket, key) not in self.objects:
            raise _client_error("404", "Not Found")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fp:
            fp.write(self.objects[(bucket, key)])

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", "The specified key does not exist.")
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ETag": self._etag(body)}

    def put_object(self, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None):
        current = self.objects.get((Bucket, Key))
        if IfNoneMatch == "*" and current is not None:
            raise _client_error("PreconditionFailed", "At least one of the pre-conditions you specified did not hold")
        if IfMatch is not None and (current is None or self._etag(current) != IfMatch):
            raise _client_error("PreconditionFailed", "At least one of the pre-conditions you specified did not hold")
        body = Body if isinstance(Body, bytes) else bytes(Body)
        self.objects[(Bucket, Key)] = body
        self.puts.append({"Key": Key, "IfMatch": IfMatch, "IfNoneMatch": IfNoneMatch, "Body": body})
        return {"ETag": self._etag(body)}


def _worker(workspace_tmp_path, *, lock_table=True, s3_objects=None):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.busy_policy = "drop"
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime.busy_wait_poll = 0
    runtime._loop = None
    runtime.lock_table = "locks" if lock_table else ""
    runtime.lock_ttl = 135
    runtime.dynamodb = _FakeDynamo()
    runtime.s3 = _FakeS3(s3_objects)
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
            calls.append(("process", update["text"]))

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    return runtime, calls


def _wa_job(chat_id, n):
    return {"provider": "whatsapp", "chat_id": chat_id, "event_id": f"wamid.{n}", "payload": {"text": f"m{n}"}}


def _whatsapp_batch(*messages):
    return {
        "entry": [{
            "id": "entry-1",
            "changes": [{"value": {
                "metadata": {"phone_number_id": "phone"},
                "messages": [
                    {"id": f"wamid.{n}", "from": sender, "type": "text", "text": {"body": f"m{n}"}}
                    for n, sender in messages
                ],
            }}],
        }]
    }


def _post_runtime(bot="whatsapp", busy_policy="drop"):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = bot
    runtime.allow_unauthenticated_webhooks = True  # routing tests; auth is covered elsewhere
    runtime.busy_policy = busy_policy
    runtime.history_mode = "per_user"
    runtime.capture_failed_events = False
    return runtime


def _post(runtime, body):
    return MawsRuntime.handle_apigw_event(
        runtime, {"requestContext": {"http": {"method": "POST"}}, "body": json.dumps(body)}, _Context()
    )


# -------------------------
# 1. busy_policy=drop message loss
# -------------------------
def test_whatsapp_batch_is_handed_off_once_per_chat_in_order():
    runtime = _post_runtime()
    invocations = []
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(json.loads(kwargs["Payload"])) or {"StatusCode": 202}
    )

    response = _post(runtime, _whatsapp_batch((1, "user-a"), (2, "user-b"), (3, "user-a"), (4, "user-a")))

    assert response["statusCode"] == 200
    assert len(invocations) == 2
    assert [job["event_id"] for job in invocations[0]["maws_jobs"]] == ["wamid.1", "wamid.3", "wamid.4"]
    assert invocations[1]["maws_job"]["event_id"] == "wamid.2"  # single jobs keep the old payload shape


def test_shared_history_batch_is_one_group():
    runtime = _post_runtime()
    runtime.history_mode = "shared"
    invocations = []
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(json.loads(kwargs["Payload"])) or {"StatusCode": 202}
    )

    _post(runtime, _whatsapp_batch((1, "user-a"), (2, "user-b")))

    assert [[job["chat_id"] for job in inv["maws_jobs"]] for inv in invocations] == [["user-a", "user-b"]]


def test_worker_processes_a_chat_group_in_order_under_one_lock(workspace_tmp_path):
    runtime, calls = _worker(workspace_tmp_path, s3_objects={("history-bucket", "history/user-a.sqlite"): b"old"})

    response = MawsRuntime.handle_apigw_event(
        runtime, {"maws_jobs": [_wa_job("user-a", 1), _wa_job("user-a", 2), _wa_job("user-a", 3)]}, _Context(60_000)
    )

    assert response["statusCode"] == 200
    assert [call[1] for call in calls if call[0] == "process"] == ["m1", "m2", "m3"]
    # One lock acquisition for the whole group (the other writes are processed-message records),
    # released afterwards.
    assert runtime.dynamodb.puts - 2 * 3 == 1
    assert sorted(runtime.dynamodb.items) == [f"maws-event#whatsapp#user-a#wamid.{n}" for n in (1, 2, 3)]
    assert runtime.s3.objects[("history-bucket", "history/user-a.sqlite")] == b"new-history"


def test_drop_mode_waits_for_a_busy_lock_and_then_processes(workspace_tmp_path):
    runtime, calls = _worker(workspace_tmp_path)
    attempts = iter([False, False, True])
    runtime.try_acquire_user_lock = lambda chat_id, context=None: next(attempts)
    runtime.release_user_lock = lambda chat_id: calls.append(("release", chat_id))

    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("user-a", 1)}, _Context(60_000))

    assert response["statusCode"] == 200
    assert ("process", "m1") in calls
    assert ("release", "user-a") in calls


@pytest.mark.parametrize(
    ("cap", "remaining_ms", "expected"),
    [
        (None, 120_000, 30.0),   # default cap
        (None, 20_000, 10.0),    # never more than half of the remaining time
        (5, 120_000, 5.0),       # MAWS_BUSY_WAIT_SECONDS
        (0, 120_000, 0.0),       # 0 = drop immediately (old behaviour)
        (None, None, 30.0),      # no Lambda deadline available
        (float("nan"), 120_000, 30.0),
    ],
)
def test_busy_wait_budget(cap, remaining_ms, expected):
    runtime = object.__new__(MawsRuntime)
    if cap is not None:
        runtime.busy_wait_seconds = cap
    context = _Context(remaining_ms) if remaining_ms is not None else object()

    assert MawsRuntime._busy_wait_budget(runtime, context) == pytest.approx(expected)


def test_runtime_reads_busy_wait_seconds_from_env(monkeypatch, workspace_tmp_path):
    monkeypatch.delenv("ENV_PARAMETER_NAME", raising=False)
    monkeypatch.setenv("MAWS_TMP_DIR", str(workspace_tmp_path))
    monkeypatch.setenv("MAWS_BUSY_WAIT_SECONDS", "12.5")

    assert MawsRuntime().busy_wait_seconds == 12.5


def test_dropped_updates_are_logged_at_warning_and_counted(workspace_tmp_path, caplog, capsys, monkeypatch):
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "demo-fn")
    runtime, calls = _worker(workspace_tmp_path)
    runtime.busy_wait_seconds = 0.05
    runtime.busy_wait_poll = 0.01
    runtime.dynamodb.items["user-a"] = {"user_id": {"S": "user-a"}, "expiresAt": {"N": str(10**12)}, "lockToken": {"S": "x"}}
    caplog.set_level(logging.WARNING, logger="maws")

    response = MawsRuntime.handle_apigw_event(
        runtime, {"maws_jobs": [_wa_job("user-a", 1), _wa_job("user-a", 2)]}, _Context(60_000)
    )

    assert response["statusCode"] == 200
    assert "Already processing" in json.loads(response["body"])
    assert not [call for call in calls if call[0] in ("import", "process")]
    assert runtime.dynamodb.puts >= 2  # it polled instead of giving up at once
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "[maws][drop]" in r.getMessage()]
    assert [("wamid.1" in r.getMessage(), "wamid.2" in r.getMessage()) for r in warnings] == [(True, False), (False, True)]
    emf = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    [metric] = [m for m in emf if "DroppedMessages" in m]
    assert metric["DroppedMessages"] == 2
    assert metric["FunctionName"] == "demo-fn"
    assert metric["event_ids"] == ["wamid.1", "wamid.2"]
    directive = metric["_aws"]["CloudWatchMetrics"][0]
    assert directive["Namespace"] == "MAWS"
    assert directive["Dimensions"] == [["FunctionName"]]
    assert directive["Metrics"] == [{"Name": "DroppedMessages", "Unit": "Count"}]
    assert isinstance(metric["_aws"]["Timestamp"], int)


def test_processing_error_in_drop_mode_is_counted(workspace_tmp_path, capsys):
    runtime, calls = _worker(workspace_tmp_path)

    class _FailingBot:
        async def process_webhook_update(self, update):
            raise RuntimeError("model down")

    runtime.bot_instance = _FailingBot()

    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("user-a", 1)}, _Context(60_000))

    assert response["statusCode"] == 200
    out = capsys.readouterr().out
    assert any('"FailedMessages": 1' in line for line in out.splitlines())


# -------------------------
# 2. Hand-off failures answer 5xx; SQS jobs are never acknowledged by mistake
# -------------------------
def test_fifo_enqueue_failure_returns_5xx_and_retry_is_deduplicated():
    runtime = _post_runtime(busy_policy="fifo")
    runtime.queue_url = "https://sqs.test/q.fifo"
    sent, failures = [], iter([None, RuntimeError("sqs down")])

    def send_message(**kwargs):
        error = next(failures, None)
        if error:
            raise error
        sent.append(kwargs)

    runtime.sqs = types.SimpleNamespace(send_message=send_message)
    body = _whatsapp_batch((1, "user-a"), (2, "user-b"))

    response = _post(runtime, body)

    assert response["statusCode"] == 503
    retry_sent = []
    runtime.sqs = types.SimpleNamespace(send_message=lambda **kwargs: retry_sent.append(kwargs))
    assert _post(runtime, body)["statusCode"] == 200
    # The job that was enqueued before the failure carries the same deduplication id on the retry.
    assert sent[0]["MessageDeduplicationId"] == retry_sent[0]["MessageDeduplicationId"]


@pytest.mark.parametrize("invoke_result", [RuntimeError("throttled"), {"StatusCode": 500}])
def test_failed_self_invoke_returns_5xx(invoke_result):
    runtime = _post_runtime(bot="telegram")

    def invoke(**kwargs):
        if isinstance(invoke_result, Exception):
            raise invoke_result
        return invoke_result

    runtime.lambda_client = types.SimpleNamespace(invoke=invoke)

    response = _post(runtime, {"update_id": 5, "message": {"chat": {"id": 1}, "text": "hi"}})

    assert response["statusCode"] == 502


def test_unparseable_webhook_body_is_still_acknowledged():
    runtime = _post_runtime(bot="telegram")

    response = MawsRuntime.handle_apigw_event(
        runtime, {"requestContext": {"http": {"method": "POST"}}, "body": "{not json"}, _Context()
    )

    assert response["statusCode"] == 200


def _sqs_event(job):
    return {"Records": [{"eventSource": "aws:sqs", "body": json.dumps({"maws_job": job})}]}


def test_sqs_worker_without_bucket_raises_instead_of_acknowledging(workspace_tmp_path):
    runtime, _ = _worker(workspace_tmp_path)
    runtime.bucket_name = ""

    with pytest.raises(MawsWorkerError, match="BUCKET_NAME"):
        MawsRuntime.handle_apigw_event(runtime, _sqs_event(_wa_job("user-a", 1)), _Context(60_000))


def test_sqs_worker_with_failed_init_raises_instead_of_acknowledging(workspace_tmp_path):
    runtime, _ = _worker(workspace_tmp_path)
    runtime.bot_instance = None

    with pytest.raises(MawsWorkerError, match="initialization"):
        MawsRuntime.handle_apigw_event(runtime, _sqs_event(_wa_job("user-a", 1)), _Context(60_000))


def test_direct_worker_keeps_returning_for_missing_bucket_and_failed_init(workspace_tmp_path):
    runtime, _ = _worker(workspace_tmp_path)
    runtime.bucket_name = ""
    assert MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("u", 1)}, _Context())["statusCode"] == 500

    runtime.bucket_name = "history-bucket"
    runtime.bot_instance = None
    response = MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("u", 1)}, _Context())
    assert "Initialization failed" in json.loads(response["body"])


def test_history_append_without_bucket_raises_for_a_retry(workspace_tmp_path):
    runtime, _ = _worker(workspace_tmp_path)
    runtime.bucket_name = ""
    runtime.history_append_enabled = True
    job = {"provider": "history", "chat_id": "u", "event_id": "e", "operation": "append_history", "payload": {"content": "x"}}

    with pytest.raises(MawsWorkerError):
        MawsRuntime.handle_apigw_event(runtime, {"maws_job": job}, _Context())


# -------------------------
# 5. Stale local history, token uploads, capture redaction, SSM version
# -------------------------
def test_missing_s3_history_is_not_resurrected_from_a_warm_container(workspace_tmp_path):
    runtime, calls = _worker(workspace_tmp_path)  # S3 has no history for user-a
    history = workspace_tmp_path / "history"
    history.mkdir()
    for suffix in ("", "-wal", "-shm"):
        (history / f"user-a.sqlite{suffix}").write_bytes(b"stale")

    MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("user-a", 1)}, _Context(60_000))

    assert ("import", "user-a", b"") in calls
    assert not (history / "user-a.sqlite-wal").exists()
    assert not (history / "user-a.sqlite-shm").exists()


def test_missing_s3_history_closes_pooled_connections_before_deleting(workspace_tmp_path):
    runtime, calls = _worker(workspace_tmp_path)
    closed = []
    runtime.manager._db_pool = {"user-a": types.SimpleNamespace(close=lambda: closed.append("user-a"))}

    MawsRuntime._discard_local_history(runtime, str(workspace_tmp_path / "history" / "user-a.sqlite"))

    assert closed == ["user-a"]
    assert runtime.manager._db_pool == {}


def _token_runtime(workspace_tmp_path, s3_objects=None, package_files=None):
    runtime = object.__new__(MawsRuntime)
    runtime.TMP_DIR = str(workspace_tmp_path / "tmp")
    runtime.CODE_ROOT = str(workspace_tmp_path / "code")
    os.makedirs(runtime.TMP_DIR, exist_ok=True)
    os.makedirs(runtime.CODE_ROOT, exist_ok=True)
    for name, body in (package_files or {}).items():
        with open(os.path.join(runtime.CODE_ROOT, name), "wb") as fp:
            fp.write(body)
    runtime.sync_tokens_s3 = True
    runtime.bucket_name = "history-bucket"
    runtime.tokens_prefix = "secrets"
    runtime.s3 = _FakeS3(s3_objects)
    return runtime


def _write_local(runtime, name, body):
    with open(os.path.join(runtime.TMP_DIR, name), "wb") as fp:
        fp.write(body)


def test_unchanged_token_is_not_uploaded_and_changed_token_uses_if_match(workspace_tmp_path):
    runtime = _token_runtime(workspace_tmp_path, {("history-bucket", "secrets/token.json"): b"v1"})
    MawsRuntime.ensure_token_file(runtime, "token.json")

    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")
    assert runtime.s3.puts == []

    _write_local(runtime, "token.json", b"v2")
    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")
    [put] = runtime.s3.puts
    assert put["IfMatch"] == '"' + hashlib.md5(b"v1").hexdigest() + '"'
    assert runtime.s3.objects[("history-bucket", "secrets/token.json")] == b"v2"

    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")  # unchanged since the upload
    assert len(runtime.s3.puts) == 1


def test_token_changed_by_another_container_is_not_overwritten(workspace_tmp_path):
    runtime = _token_runtime(workspace_tmp_path, {("history-bucket", "secrets/token.json"): b"v1"})
    MawsRuntime.ensure_token_file(runtime, "token.json")
    runtime.s3.objects[("history-bucket", "secrets/token.json")] = b"other-container"
    _write_local(runtime, "token.json", b"mine")

    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")

    assert runtime.s3.objects[("history-bucket", "secrets/token.json")] == b"other-container"
    assert not os.path.exists(os.path.join(runtime.TMP_DIR, "token.json"))
    # The next job fetches the S3 version again.
    MawsRuntime.ensure_token_file(runtime, "token.json")
    with open(os.path.join(runtime.TMP_DIR, "token.json"), "rb") as fp:
        assert fp.read() == b"other-container"


def test_token_missing_in_s3_is_created_only_if_still_absent(workspace_tmp_path):
    runtime = _token_runtime(workspace_tmp_path, package_files={"token.json": b"pkg"})
    MawsRuntime.ensure_token_file(runtime, "token.json")

    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")
    assert runtime.s3.puts == []  # the package copy is unchanged: nothing to upload

    _write_local(runtime, "token.json", b"refreshed")
    MawsRuntime.upload_token_back_if_exists(runtime, "token.json")
    [put] = runtime.s3.puts
    assert put["IfNoneMatch"] == "*"


def test_worker_refetches_a_discarded_token_before_the_next_job(workspace_tmp_path):
    runtime, calls = _worker(workspace_tmp_path)
    runtime.special_files = ["token.json"]
    runtime.sync_tokens_s3 = True
    runtime.tokens_prefix = "secrets"
    runtime.CODE_ROOT = str(workspace_tmp_path / "code")
    runtime.s3.objects[("history-bucket", "secrets/token.json")] = b"from-s3"

    MawsRuntime.handle_apigw_event(runtime, {"maws_job": _wa_job("user-a", 1)}, _Context(60_000))

    with open(os.path.join(runtime.TMP_DIR, "token.json"), "rb") as fp:
        assert fp.read() == b"from-s3"
    assert not [p for p in runtime.s3.puts if p["Key"] == "secrets/token.json"]


def test_failed_event_capture_redacts_webhook_secrets(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "_LOADED_SECRET_VALUES", {"sk-live-openai-key-123456"})
    runtime = object.__new__(MawsRuntime)
    runtime.capture_failed_events = True
    runtime.bucket_name = "history-bucket"
    runtime.failed_events_prefix = "failed-events"
    runtime.telegram_secret_token = "tg-webhook-secret-value"
    runtime.whatsapp_app_secret = "meta-app-secret-value"
    runtime.history_write_secret = ""
    runtime.s3 = _FakeS3()
    bot_token = "123456789:AAH1bGciOiJIUzI1NiJ9abcdefghijklmnop"
    event = {
        "headers": {
            "X-Telegram-Bot-Api-Secret-Token": "tg-webhook-secret-value",
            "X-Hub-Signature-256": "sha256=deadbeef",
            "Authorization": "Bearer abc.def",
            "Content-Type": "application/json",
        },
        "multiValueHeaders": {"x-telegram-bot-api-secret-token": ["tg-webhook-secret-value"]},
        "queryStringParameters": {"hub.verify_token": "verify-me"},
        "rawQueryString": "hub.mode=subscribe&hub.verify_token=verify-me",
        "requestContext": {"identity": {"apiKey": "k", "sourceIp": "1.2.3.4"}},
        "body": json.dumps({"message": {"text": "my key is sk-live-openai-key-123456"}}),
    }

    MawsRuntime._capture_failed_event(
        runtime, event, "post_failed", RuntimeError(f"POST https://api.telegram.org/bot{bot_token}/sendMessage failed")
    )

    [put] = runtime.s3.puts
    stored = put["Body"].decode("utf-8")
    for secret in ("tg-webhook-secret-value", "deadbeef", "abc.def", "verify-me", bot_token,
                   "sk-live-openai-key-123456", "meta-app-secret-value"):
        assert secret not in stored, secret
    captured = json.loads(stored)
    assert captured["event"]["headers"]["Content-Type"] == "application/json"
    assert captured["event"]["requestContext"]["identity"]["sourceIp"] == "1.2.3.4"
    assert captured["event"]["headers"]["X-Hub-Signature-256"] == "[REDACTED]"
    assert captured["event"]["multiValueHeaders"]["x-telegram-bot-api-secret-token"] == ["[REDACTED]"]
    assert "api.telegram.org/bot[REDACTED]/sendMessage" in captured["error"]


def test_capture_failure_does_not_mask_the_original_error(workspace_tmp_path):
    runtime = object.__new__(MawsRuntime)
    runtime.capture_failed_events = True
    runtime.bucket_name = "history-bucket"
    runtime.s3 = types.SimpleNamespace(put_object=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("s3 down")))

    MawsRuntime._capture_failed_event(runtime, {"chat_id": "u"}, "worker_processing_failed", RuntimeError("x"))


def _ssm_runtime(monkeypatch, responses):
    requested = []

    def get_parameter(Name, WithDecryption):
        requested.append(Name)
        result = responses[Name]
        if isinstance(result, Exception):
            raise result
        return {"Parameter": {"Value": result}}

    ssm = types.SimpleNamespace(get_parameter=get_parameter)
    monkeypatch.setattr(runtime_module, "_require_boto3", lambda: types.SimpleNamespace(client=lambda name: ssm))
    monkeypatch.setenv("ENV_PARAMETER_NAME", "/p/prod/env")
    monkeypatch.delenv("_ENV_LOADED_FROM_SSM", raising=False)
    monkeypatch.delenv("MAWS_TEST_SECRET", raising=False)
    return requested


def test_runtime_reads_the_pinned_ssm_parameter_version(monkeypatch):
    requested = _ssm_runtime(monkeypatch, {"/p/prod/env:7": "MAWS_TEST_SECRET=v7"})
    monkeypatch.setenv("MAWS_ENV_PARAMETER_VERSION", "7")

    runtime_module._load_env_from_ssm_if_needed()

    assert requested == ["/p/prod/env:7"]
    assert os.environ["MAWS_TEST_SECRET"] == "v7"


def test_runtime_falls_back_to_latest_when_the_pinned_version_is_gone(monkeypatch):
    requested = _ssm_runtime(monkeypatch, {
        "/p/prod/env:3": _client_error("ParameterVersionNotFound"),
        "/p/prod/env": "MAWS_TEST_SECRET=latest",
    })
    monkeypatch.setenv("MAWS_ENV_PARAMETER_VERSION", "3")

    runtime_module._load_env_from_ssm_if_needed()

    assert requested == ["/p/prod/env:3", "/p/prod/env"]
    assert os.environ["MAWS_TEST_SECRET"] == "latest"


def test_runtime_without_a_pinned_version_reads_latest(monkeypatch):
    requested = _ssm_runtime(monkeypatch, {"/p/prod/env": "MAWS_TEST_SECRET=x"})
    monkeypatch.delenv("MAWS_ENV_PARAMETER_VERSION", raising=False)

    runtime_module._load_env_from_ssm_if_needed()

    assert requested == ["/p/prod/env"]


# -------------------------
# 6. params.json, --set, start, setup, installs
# -------------------------
def test_params_json_syntax_error_fails_hard(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / "params.json").write_text('{"project": "p",\n "region": }', encoding="utf-8")
    monkeypatch.setattr(operations, "_run_bash", lambda *a, **k: pytest.fail("bootstrap must not run"))

    with pytest.raises(MawsConfigError, match=r"not valid JSON: .*line 2"):
        operations.update(project_dir=workspace_tmp_path, allow_windows=True)
    for call in (operations.setup, operations.describe, operations.remove_project):
        with pytest.raises(MawsConfigError):
            call(project_dir=workspace_tmp_path)


def test_cli_reports_config_errors_with_exit_code_2(workspace_tmp_path, capsys):
    (workspace_tmp_path / "params.json").write_text("{oops", encoding="utf-8")

    code = cli.main(["describe", "--dir", str(workspace_tmp_path), "--no-aws"])

    assert code == 2
    assert "is not valid JSON" in capsys.readouterr().err


def test_missing_params_json_is_still_allowed_with_overrides(workspace_tmp_path, monkeypatch):
    captured = {}

    def fake_run_bash(script, args, cwd=None, quiet=False):
        captured.update(json.loads((cwd / args[1]).read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(operations, "_run_bash", fake_run_bash)

    assert operations.update(project_dir=workspace_tmp_path, project="p", region="r", bot="telegram", allow_windows=True) == 0
    assert captured == {"project": "p", "region": "r", "bot": "telegram"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.10", "1.10"),
        ("0123456", "0123456"),
        ("nan", "nan"),
        ("1e3", "1e3"),
        ("-0", "-0"),
        ("True", "True"),
        ("v0.3.4", "v0.3.4"),
        ("", ""),
        ("true", True),
        ("false", False),
        ("null", None),
        ("120", 120),
        ("1.5", 1.5),
        ('"quoted text"', "quoted text"),
        ('{"a": [1, 2]}', {"a": [1, 2]}),
        ('["x"]', ["x"]),
    ],
)
def test_set_values_stay_strings_unless_exact_json_literals(raw, expected):
    out = operations._apply_overrides({}, None, None, None, [f"key={raw}"])

    assert out["key"] == expected
    assert type(out["key"]) is type(expected)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", '{"a": NaN}'])
def test_set_rejects_non_finite_numbers(raw):
    with pytest.raises(MawsConfigError, match="NaN/Infinity"):
        operations._apply_overrides({}, None, None, None, [f"key={raw}"])


def test_set_dotted_keys_are_nested_paths_and_do_not_mutate_the_base():
    base = {"infra": {"runtime": "python3.11", "memory_size": 512}, "runtime": "not-an-object"}

    out = operations._apply_overrides(base, None, None, None, [
        "infra.tracing=false", "failure_handling.worker_dlq=true", "webhook_security.allow_unauthenticated=false",
    ])

    assert out["infra"] == {"runtime": "python3.11", "memory_size": 512, "tracing": False}
    assert out["failure_handling"] == {"worker_dlq": True}
    assert "infra.tracing" not in out
    assert base["infra"] == {"runtime": "python3.11", "memory_size": 512}
    with pytest.raises(MawsConfigError, match="runtime is 'not-an-object'"):
        operations._apply_overrides(base, None, None, None, ["runtime.ensure_delivery=false"])
    for bad in ("no-equals-sign", "infra..tracing=1", "=1"):
        with pytest.raises(MawsConfigError):
            operations._apply_overrides(base, None, None, None, [bad])


def test_update_writes_valid_json_for_bootstrap(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / "params.json").write_text(json.dumps({"project": "p", "infra": {"memory_size": 512}}), encoding="utf-8")
    seen = {}

    def fake_run_bash(script, args, cwd=None, quiet=False):
        seen["args"] = args
        seen["params"] = json.loads((cwd / args[1]).read_text(encoding="utf-8"))
        return 0

    monkeypatch.setattr(operations, "_run_bash", fake_run_bash)

    operations.update(project_dir=workspace_tmp_path, set_kv=["infra.tracing=false", "requirements_ref=1.10"], allow_windows=True)

    assert seen["args"] == ["--config", "params.tmp.json"]
    assert seen["params"]["infra"] == {"memory_size": 512, "tracing": False}
    assert seen["params"]["requirements_ref"] == "1.10"
    assert not (workspace_tmp_path / "params.tmp.json").exists()
    with pytest.raises(MawsConfigError):
        operations.update(params={"x": float("nan")}, project_dir=workspace_tmp_path, allow_windows=True)


def _start(workspace_tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(operations._shutil, "which", lambda cmd: None)
    kwargs.setdefault("allow_windows", True)
    return operations.start(project_dir=workspace_tmp_path, **kwargs)


def test_start_rerun_keeps_existing_params(workspace_tmp_path, monkeypatch, capsys):
    existing = {
        "project": "alpha", "region": "sa-east-1", "bot": "whatsapp",
        "history_bucket": "alpha-history", "stack_name": "alpha-stack",
    }
    (workspace_tmp_path / "params.json").write_text(json.dumps(existing), encoding="utf-8")

    _start(workspace_tmp_path, monkeypatch)
    assert json.loads((workspace_tmp_path / "params.json").read_text(encoding="utf-8")) == existing

    _start(workspace_tmp_path, monkeypatch, project="beta", region="us-east-1", bot="telegram")
    assert json.loads((workspace_tmp_path / "params.json").read_text(encoding="utf-8")) == existing
    out = capsys.readouterr().out
    assert "params.json already sets project='alpha'; keeping it" in out
    assert "--overwrite" in out


def test_start_overwrite_replaces_only_project_region_bot(workspace_tmp_path, monkeypatch):
    existing = {"project": "alpha", "region": "sa-east-1", "bot": "whatsapp", "history_bucket": "alpha-history", "verbose": True}
    (workspace_tmp_path / "params.json").write_text(json.dumps(existing), encoding="utf-8")

    _start(workspace_tmp_path, monkeypatch, region="us-east-1", bot="telegram", overwrite=True)

    data = json.loads((workspace_tmp_path / "params.json").read_text(encoding="utf-8"))
    assert data == {**existing, "region": "us-east-1", "bot": "telegram"}


def test_start_fills_missing_keys_of_an_existing_params_json(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / "params.json").write_text(json.dumps({"project": "alpha"}), encoding="utf-8")

    _start(workspace_tmp_path, monkeypatch, region="eu-west-1")

    data = json.loads((workspace_tmp_path / "params.json").read_text(encoding="utf-8"))
    assert data == {"project": "alpha", "region": "eu-west-1", "bot": "telegram"}
    assert "history_bucket" not in data  # may already be deployed with the legacy default


def test_start_rejects_a_broken_params_json(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / "params.json").write_text("{", encoding="utf-8")

    with pytest.raises(MawsConfigError):
        _start(workspace_tmp_path, monkeypatch)
    assert (workspace_tmp_path / "params.json").read_text(encoding="utf-8") == "{"


BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


def test_new_project_gets_unique_bucket_names(workspace_tmp_path, monkeypatch):
    first = workspace_tmp_path / "one"
    second = workspace_tmp_path / "two"
    _start(first, monkeypatch, project="My_Bot", region="us-east-1", bot="telegram")
    _start(second, monkeypatch, project="My_Bot", region="us-east-1", bot="telegram")

    one = json.loads((first / "params.json").read_text(encoding="utf-8"))
    two = json.loads((second / "params.json").read_text(encoding="utf-8"))
    assert one["project"] == "My_Bot"
    for data in (one, two):
        assert re.fullmatch(r"my-bot-history-[0-9a-f]{8}", data["history_bucket"])
        assert re.fullmatch(r"my-bot-deploy-[0-9a-f]{8}", data["deployment_bucket"])
    assert one["history_bucket"] != two["history_bucket"]


@pytest.mark.parametrize("project", ["x" * 80, "xn--weird", "--", "Proj.With.Dots"])
def test_unique_bucket_names_are_valid_s3_names(project):
    for kind in ("history", "deploy"):
        name = operations._unique_bucket_name(project, kind)
        assert BUCKET_RE.fullmatch(name), name
        assert not name.startswith(("xn--", "sthree-"))


def test_setup_uses_unique_names_only_for_a_new_params_json(workspace_tmp_path):
    new_dir = workspace_tmp_path / "new"
    new_dir.mkdir()
    operations.setup(project_dir=new_dir)
    created = json.loads((new_dir / "params.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"my-bot-history-[0-9a-f]{8}", created["history_bucket"])

    old_dir = workspace_tmp_path / "old"
    old_dir.mkdir()
    (old_dir / "params.json").write_text(json.dumps({"project": "legacy"}), encoding="utf-8")
    operations.setup(project_dir=old_dir)
    kept = json.loads((old_dir / "params.json").read_text(encoding="utf-8"))
    assert kept["history_bucket"] == "legacy-history-bucket"
    assert kept["deployment_bucket"] == "legacy-deployment-bucket"


def test_cli_start_does_not_install_or_configure_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_start", lambda **kwargs: captured.update(kwargs))

    assert cli.main(["start", "--project", "p", "--region", "r", "--bot", "telegram"]) == 0
    assert captured["install_deps"] is False
    assert captured["run_config"] is False

    cli.main(["start", "--install-deps", "--run-config"])
    assert captured["install_deps"] is True
    assert captured["run_config"] is True


def test_aws_configure_is_skipped_when_credentials_work(workspace_tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(operations._shutil, "which", lambda cmd: "/usr/bin/aws" if cmd == "aws" else None)
    monkeypatch.setattr(operations._subprocess, "check_output", lambda *a, **k: '{"Account": "123456789012"}')
    called = []
    monkeypatch.setattr(operations._subprocess, "call", lambda cmd, *a, **k: called.append(cmd))

    operations.start(project="p", region="r", bot="telegram", project_dir=workspace_tmp_path, run_config=True, allow_windows=True)

    assert called == []
    assert "not running 'aws configure'" in capsys.readouterr().out


def test_aws_configure_runs_when_asked_and_credentials_fail(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(operations._shutil, "which", lambda cmd: "/usr/bin/aws" if cmd == "aws" else None)

    def no_credentials(*args, **kwargs):
        raise operations._subprocess.CalledProcessError(255, args[0])

    monkeypatch.setattr(operations._subprocess, "check_output", no_credentials)
    called = []
    monkeypatch.setattr(operations._subprocess, "call", lambda cmd, *a, **k: called.append(cmd))

    operations.start(project="p", region="r", bot="telegram", project_dir=workspace_tmp_path, run_config=True, allow_windows=True)

    assert called == [["aws", "configure"]]


# -------------------------
# 3. maws remove
# -------------------------
class _StackMissing(Exception):
    def __init__(self, stack):
        super().__init__(f"An error occurred (ValidationError) when calling the DescribeStacks operation: Stack with id {stack} does not exist")
        self.response = {"Error": {"Code": "ValidationError", "Message": f"Stack with id {stack} does not exist"}}


class _RemoveCloudFormation:
    def __init__(self, *, exists=True, buckets=("alpha-history",), delete_error=None, wait_error=None):
        self.exists = exists
        self.buckets = list(buckets)
        self.delete_error = delete_error
        self.wait_error = wait_error
        self.calls = []

    def describe_stacks(self, StackName):
        self.calls.append(("describe", StackName))
        if not self.exists:
            raise _StackMissing(StackName)
        return {"Stacks": [{"StackName": StackName, "StackStatus": "UPDATE_COMPLETE"}]}

    def describe_stack_resources(self, StackName):
        resources = [{"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": b} for b in self.buckets]
        resources.append({"ResourceType": "AWS::Lambda::Function", "PhysicalResourceId": "alpha-stack-AlphaFunction-XYZ"})
        return {"StackResources": resources}

    def delete_stack(self, StackName):
        self.calls.append(("delete", StackName))
        if self.delete_error:
            raise self.delete_error

    def get_waiter(self, name):
        cf = self

        class _Waiter:
            def wait(self, StackName):
                cf.calls.append(("wait", StackName))
                if cf.wait_error:
                    raise cf.wait_error

        return _Waiter()


class _RemoveS3:
    """Versioned buckets: {bucket: [(key, version_id)]}."""

    def __init__(self, buckets):
        self.buckets = {name: list(objects) for name, objects in buckets.items()}
        self.calls = []
        self.exceptions = types.SimpleNamespace(NoSuchBucket=type("NoSuchBucket", (Exception,), {}))

    def _objects(self, Bucket, Prefix=""):
        if Bucket not in self.buckets:
            raise self.exceptions.NoSuchBucket(Bucket)
        return [(k, v) for k, v in self.buckets[Bucket] if k.startswith(Prefix)]

    def get_paginator(self, name):
        s3 = self

        class _Paginator:
            def paginate(self, Bucket, Prefix="", ExpectedBucketOwner=None):
                s3.calls.append((name, Bucket, Prefix, ExpectedBucketOwner))
                objects = s3._objects(Bucket, Prefix)
                if name == "list_object_versions":
                    return [{"Versions": [{"Key": k, "VersionId": v} for k, v in objects]}]
                return [{"Contents": []}]

        return _Paginator()

    def delete_objects(self, Bucket, Delete, ExpectedBucketOwner=None):
        self.calls.append(("delete_objects", Bucket, ExpectedBucketOwner))
        doomed = {(o["Key"], o.get("VersionId")) for o in Delete["Objects"]}
        self.buckets[Bucket] = [obj for obj in self.buckets[Bucket] if obj not in doomed]
        return {}

    def list_object_versions(self, Bucket, MaxKeys, ExpectedBucketOwner=None):
        return {"Versions": [{"Key": k} for k, _ in self._objects(Bucket)[:MaxKeys]]}

    def list_objects_v2(self, Bucket, MaxKeys, ExpectedBucketOwner=None):
        return {"Contents": [{"Key": k} for k, _ in self._objects(Bucket)[:MaxKeys]]}

    def delete_bucket(self, Bucket, ExpectedBucketOwner=None):
        self.calls.append(("delete_bucket", Bucket, ExpectedBucketOwner))
        assert not self.buckets[Bucket]
        del self.buckets[Bucket]


class _SSM:
    def __init__(self, error=None):
        self.deleted = []
        self.error = error

    def delete_parameter(self, Name):
        if self.error:
            raise self.error
        self.deleted.append(Name)


ALPHA_PARAMS = {
    "project": "alpha",
    "region": "sa-east-1",
    "stack_name": "alpha-stack",
    "history_bucket": "alpha-history",
    "deployment_bucket": "shared-deploy",
    "env_param_name": "/alpha/prod/env",
}


def _remove(workspace_tmp_path, monkeypatch, *, cf=None, s3=None, ssm=None, answer=None, **kwargs):
    (workspace_tmp_path / "params.json").write_text(json.dumps(ALPHA_PARAMS), encoding="utf-8")
    cf = cf or _RemoveCloudFormation()
    s3 = s3 or _RemoveS3({
        "alpha-history": [("history/u1.sqlite", "v1"), ("secrets/token.json", "v1")],
        "shared-deploy": [("alpha-stack/abc.zip", "null"), ("alpha-stack/abc.template", "null"), ("other-stack/x.zip", "null")],
    })
    ssm = ssm or _SSM()
    monkeypatch.setattr(operations, "_boto", lambda region=None: (cf, s3, None))
    monkeypatch.setattr(operations, "_ssm_client", lambda region=None: ssm)
    monkeypatch.setattr(operations, "_account_id", lambda region=None: "123456789012")
    if answer is not None:
        monkeypatch.setattr(operations, "_ask_bool", lambda prompt, default=True: answer)
    kwargs.setdefault("yes", answer is None)
    code = operations.remove_project(project_dir=workspace_tmp_path, **kwargs)
    return code, cf, s3, ssm


def test_remove_checks_the_stack_region_before_deleting_anything(workspace_tmp_path, monkeypatch, capsys):
    code, cf, s3, ssm = _remove(workspace_tmp_path, monkeypatch, cf=_RemoveCloudFormation(exists=False), region="us-west-2")

    assert code == 1
    assert cf.calls == [("describe", "alpha-stack")]
    assert s3.calls == [] and ssm.deleted == []
    out = capsys.readouterr().out
    assert "does not exist in region us-west-2. Nothing was deleted." in out
    assert "[OK]" not in out


def test_remove_deletes_stack_ssm_and_only_this_stacks_deploy_prefix(workspace_tmp_path, monkeypatch, capsys):
    code, cf, s3, ssm = _remove(workspace_tmp_path, monkeypatch, wait=True)

    assert code == 0
    out = capsys.readouterr().out
    assert "Delete SSM:       /alpha/prod/env" in out
    assert "[OK] Done." in out
    assert cf.calls == [("describe", "alpha-stack"), ("delete", "alpha-stack"), ("wait", "alpha-stack")]
    assert s3.buckets["alpha-history"] == []
    assert s3.buckets["shared-deploy"] == [("other-stack/x.zip", "null")]  # shared bucket kept
    assert ("delete_bucket", "shared-deploy", "123456789012") not in s3.calls
    assert "Kept deployment bucket shared-deploy" in out
    assert ("list_object_versions", "shared-deploy", "alpha-stack/", "123456789012") in s3.calls
    assert ssm.deleted == ["/alpha/prod/env"]


def test_remove_deletes_the_deploy_bucket_when_it_is_left_empty(workspace_tmp_path, monkeypatch):
    s3 = _RemoveS3({"alpha-history": [], "shared-deploy": [("alpha-stack/abc.zip", "null")]})

    code, _, s3, _ = _remove(workspace_tmp_path, monkeypatch, s3=s3)

    assert code == 0
    assert "shared-deploy" not in s3.buckets
    assert ("delete_bucket", "shared-deploy", "123456789012") in s3.calls


@pytest.mark.parametrize(
    ("cf", "message"),
    [
        (_RemoveCloudFormation(delete_error=RuntimeError("AccessDenied")), "Could not delete stack"),
        (_RemoveCloudFormation(wait_error=RuntimeError("Waiter StackDeleteComplete failed")), "was not deleted"),
    ],
)
def test_remove_fails_loudly_and_keeps_secrets_when_the_stack_is_not_deleted(workspace_tmp_path, monkeypatch, capsys, cf, message):
    code, _, s3, ssm = _remove(workspace_tmp_path, monkeypatch, cf=cf, wait=True)

    assert code == 1
    out = capsys.readouterr().out
    assert message in out
    assert "[OK] Done" not in out
    assert ssm.deleted == []
    assert ("alpha-stack/abc.zip", "null") in s3.buckets["shared-deploy"]


def test_remove_does_not_empty_a_history_bucket_outside_the_stack(workspace_tmp_path, monkeypatch, capsys):
    code, _, s3, _ = _remove(workspace_tmp_path, monkeypatch, cf=_RemoveCloudFormation(buckets=()))

    assert code == 0
    assert s3.buckets["alpha-history"] == [("history/u1.sqlite", "v1"), ("secrets/token.json", "v1")]
    assert "alpha-history is not a resource of alpha-stack: it is left untouched" in capsys.readouterr().out


def test_remove_reports_ssm_failures(workspace_tmp_path, monkeypatch, capsys):
    code, *_ = _remove(workspace_tmp_path, monkeypatch, ssm=_SSM(error=RuntimeError("AccessDenied")))

    assert code == 1
    out = capsys.readouterr().out
    assert "SSM parameter /alpha/prod/env: AccessDenied" in out
    assert "[OK] Done" not in out


def test_remove_accepts_an_already_deleted_ssm_parameter(workspace_tmp_path, monkeypatch):
    error = _client_error("ParameterNotFound")

    code, *_ = _remove(workspace_tmp_path, monkeypatch, ssm=_SSM(error=error))

    assert code == 0


def test_remove_keep_deploy_bucket_and_confirmation(workspace_tmp_path, monkeypatch):
    code, cf, s3, ssm = _remove(workspace_tmp_path, monkeypatch, answer=False)
    assert code == 1
    assert cf.calls == [("describe", "alpha-stack")]  # read-only checks only
    assert s3.calls == [] and ssm.deleted == []

    code, _, s3, _ = _remove(workspace_tmp_path, monkeypatch, keep_deploy_bucket=True)
    assert code == 0
    assert len(s3.buckets["shared-deploy"]) == 3
