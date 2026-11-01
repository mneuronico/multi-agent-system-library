"""Processed-message records: a message delivered to the maws worker more than once is processed once.

Repeated deliveries come from the provider retrying a whole webhook after one chat's hand-off failed,
from Lambda delivering an async invocation twice, and from SQS redelivering a message whose worker
failed after processing it. The records live in the locks table (keys "maws-event#...").
"""

from __future__ import annotations

import json
import time
import types

import pytest

import maws.runtime as runtime_module
from maws.runtime import MawsRuntime, UserLockBusy

NOW = 1_800_000_000
EIGHT_DAYS = 8 * 86400


class _Context:
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:bot"

    def get_remaining_time_in_millis(self):
        return 60_000


class _ConditionalCheckFailed(Exception):
    pass


class _FakeDynamo:
    """Locks table that evaluates the condition expressions MAWS sends."""

    def __init__(self):
        self.items = {}
        self.fail_put_for = None
        self.exceptions = types.SimpleNamespace(ConditionalCheckFailedException=_ConditionalCheckFailed)

    def put_item(self, TableName, Item, ConditionExpression, ExpressionAttributeValues):
        key = Item["user_id"]["S"]
        if self.fail_put_for and key.endswith(self.fail_put_for):
            raise RuntimeError("ProvisionedThroughputExceededException")
        current = self.items.get(key)
        if ConditionExpression == "lockToken = :token":
            if current is None or current["lockToken"]["S"] != ExpressionAttributeValues[":token"]["S"]:
                raise _ConditionalCheckFailed()
        else:
            assert ConditionExpression == "attribute_not_exists(user_id) OR expiresAt < :now"
            now = int(ExpressionAttributeValues[":now"]["N"])
            if current is not None and not int(current["expiresAt"]["N"]) < now:
                raise _ConditionalCheckFailed()
        self.items[key] = Item

    def delete_item(self, TableName, Key, ConditionExpression=None, ExpressionAttributeValues=None):
        key = Key["user_id"]["S"]
        if ConditionExpression is not None:
            assert ConditionExpression == "lockToken = :token"
            current = self.items.get(key)
            if current is None or current["lockToken"]["S"] != ExpressionAttributeValues[":token"]["S"]:
                raise _ConditionalCheckFailed()
        self.items.pop(key, None)

    def hold(self, key, expires_at, status=None):
        item = {"user_id": {"S": key}, "expiresAt": {"N": str(expires_at)}, "lockToken": {"S": "someone-else"}}
        if status:
            item["status"] = {"S": status}
        self.items[key] = item


class _FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_downloads = 0

    def download_file(self, bucket, key, local_path):
        if self.fail_downloads:
            self.fail_downloads -= 1
            raise runtime_module.botocore.exceptions.ClientError(
                {"Error": {"Code": "SlowDown", "Message": "Please reduce your request rate."}}, "GetObject"
            )
        raise runtime_module.botocore.exceptions.ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body


def _runtime(workspace_tmp_path, *, busy_policy="drop"):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = "whatsapp"
    runtime.allow_unauthenticated_webhooks = True  # routing only; authentication is tested elsewhere
    runtime.busy_policy = busy_policy
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.history_mode = "per_user"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime.busy_wait_seconds = 0
    runtime.busy_wait_poll = 0
    runtime._loop = None
    runtime.lock_table = "locks"
    runtime.lock_ttl = 135
    runtime.dynamodb = _FakeDynamo()
    runtime.s3 = _FakeS3()
    runtime._lock_now_epoch = lambda: NOW
    processed = []
    metrics = []

    class _Manager:
        def import_history(self, user_id, sqlite_bytes):
            pass

        def set_current_user(self, user_id):
            pass

        def export_history(self, user_id):
            return b"history"

    class _Bot:
        fail = False

        async def process_webhook_update(self, update):
            message = update["entry"][0]["changes"][0]["value"]["messages"][0]
            processed.append(message["text"]["body"])
            if self.fail:
                raise RuntimeError("the model provider timed out")

    runtime.manager = _Manager()
    runtime.bot_instance = _Bot()
    runtime.initialize_system = lambda: None
    runtime._emit_metric = lambda name, value=1, unit="Count", **properties: metrics.append((name, value))
    return runtime, processed, metrics


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


def _jobs(runtime, *messages):
    return runtime._normalize_whatsapp_jobs(_whatsapp_batch(*messages))


def _post(runtime, body):
    return MawsRuntime.handle_apigw_event(
        runtime, {"requestContext": {"http": {"method": "POST"}}, "body": json.dumps(body)}, _Context()
    )


def _work(runtime, event):
    return MawsRuntime.handle_apigw_event(runtime, event, _Context())


def _sqs(job):
    return {"Records": [{"eventSource": "aws:sqs", "body": json.dumps({"maws_job": job})}]}


def _record(runtime, chat_id, n):
    return runtime.dynamodb.items.get(f"maws-event#whatsapp#{chat_id}#wamid.{n}")


# -------------------------
# The reported case: a webhook retried after a partial hand-off
# -------------------------
def test_a_webhook_retried_after_a_partial_hand_off_processes_each_chat_once(workspace_tmp_path):
    runtime, processed, metrics = _runtime(workspace_tmp_path)
    invocations = []

    def invoke(**kwargs):
        invocations.append(json.loads(kwargs["Payload"]))
        if len(invocations) == 2:
            raise RuntimeError("Rate exceeded")  # user-b's hand-off fails; user-a's already went out
        return {"StatusCode": 202}

    runtime.lambda_client = types.SimpleNamespace(invoke=invoke)
    batch = _whatsapp_batch((1, "user-a"), (2, "user-b"))

    assert _post(runtime, batch)["statusCode"] == 502
    assert _post(runtime, batch)["statusCode"] == 200  # WhatsApp retries the whole webhook

    accepted = [invocations[0], invocations[2], invocations[3]]  # user-a, user-a again, user-b
    for event in accepted:
        assert _work(runtime, event)["statusCode"] == 200

    assert processed == ["m1", "m2"]
    assert metrics == [("DuplicateMessages", 1)]
    assert _record(runtime, "user-a", 1)["status"]["S"] == "processed"
    assert _record(runtime, "user-b", 2)["status"]["S"] == "processed"


def test_a_message_is_recorded_as_processed_for_eight_days_by_default(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)

    _work(runtime, {"maws_job": _jobs(runtime, (1, "user-a"))[0]})

    record = _record(runtime, "user-a", 1)
    assert record["status"]["S"] == "processed"
    assert int(record["expiresAt"]["N"]) == NOW + EIGHT_DAYS
    assert "user-a" not in runtime.dynamodb.items  # the user lock itself is released


def test_the_retention_period_is_configurable(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    runtime.dedup_retention_days = 1.5

    _work(runtime, {"maws_job": _jobs(runtime, (1, "user-a"))[0]})

    assert int(_record(runtime, "user-a", 1)["expiresAt"]["N"]) == NOW + int(1.5 * 86400)


def test_a_record_past_its_retention_no_longer_blocks_the_message(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    job = _jobs(runtime, (1, "user-a"))[0]
    _work(runtime, {"maws_job": job})

    runtime._lock_now_epoch = lambda: NOW + EIGHT_DAYS + 1
    _work(runtime, {"maws_job": job})

    assert processed == ["m1", "m1"]


def test_only_the_new_messages_of_a_group_are_processed(workspace_tmp_path):
    runtime, processed, metrics = _runtime(workspace_tmp_path)
    first, second = _jobs(runtime, (1, "user-a"), (2, "user-a"))
    _work(runtime, {"maws_job": first})

    _work(runtime, {"maws_jobs": [first, second]})

    assert processed == ["m1", "m2"]
    assert metrics == [("DuplicateMessages", 1)]


def test_a_message_another_worker_is_processing_is_skipped_without_waiting_for_the_lock(workspace_tmp_path):
    runtime, processed, metrics = _runtime(workspace_tmp_path)
    runtime.busy_wait_seconds = 30
    job = _jobs(runtime, (1, "user-a"))[0]
    # The first delivery's worker holds the chat lock and its claim on the message.
    runtime.dynamodb.hold("user-a", NOW + 100)
    runtime.dynamodb.hold("maws-event#whatsapp#user-a#wamid.1", NOW + 100, status="processing")

    started = time.monotonic()
    response = _work(runtime, {"maws_job": job})

    assert time.monotonic() - started < 5
    assert json.loads(response["body"]) == "Duplicate update ignored."
    assert processed == []
    assert metrics == [("DuplicateMessages", 1)]  # a repeat, not a dropped message


def test_the_claim_of_a_worker_that_crashed_expires_so_the_retry_processes_the_message(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    runtime.dynamodb.hold("maws-event#whatsapp#user-a#wamid.1", NOW - 1, status="processing")

    _work(runtime, {"maws_job": _jobs(runtime, (1, "user-a"))[0]})

    assert processed == ["m1"]


# -------------------------
# Messages that were not processed stay processable
# -------------------------
def test_a_message_dropped_because_the_user_was_busy_can_still_be_processed_later(workspace_tmp_path):
    runtime, processed, metrics = _runtime(workspace_tmp_path)
    job = _jobs(runtime, (1, "user-a"))[0]
    runtime.dynamodb.hold("user-a", NOW + 100)

    _work(runtime, {"maws_job": job})

    assert processed == []
    assert metrics == [("DroppedMessages", 1)]
    assert _record(runtime, "user-a", 1) is None  # claim released
    del runtime.dynamodb.items["user-a"]
    _work(runtime, {"maws_job": job})
    assert processed == ["m1"]


def test_fifo_releases_the_claim_when_the_user_is_busy_so_the_redelivery_processes_it(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path, busy_policy="fifo")
    job = _jobs(runtime, (1, "user-a"))[0]
    runtime.dynamodb.hold("user-a", NOW + 100)

    with pytest.raises(UserLockBusy):
        _work(runtime, _sqs(job))

    assert _record(runtime, "user-a", 1) is None
    del runtime.dynamodb.items["user-a"]
    _work(runtime, _sqs(job))
    assert processed == ["m1"]


def test_a_message_whose_history_could_not_be_loaded_is_processed_by_the_next_delivery(workspace_tmp_path):
    runtime, processed, metrics = _runtime(workspace_tmp_path)
    job = _jobs(runtime, (1, "user-a"))[0]
    runtime.s3.fail_downloads = 1

    _work(runtime, {"maws_job": job})

    assert processed == []
    assert ("FailedMessages", 1) in metrics
    assert _record(runtime, "user-a", 1) is None
    _work(runtime, {"maws_job": job})
    assert processed == ["m1"]


def test_fifo_retries_a_message_whose_processing_raised(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path, busy_policy="fifo")
    job = _jobs(runtime, (1, "user-a"))[0]
    runtime.bot_instance.fail = True

    with pytest.raises(RuntimeError, match="timed out"):
        _work(runtime, _sqs(job))

    runtime.bot_instance.fail = False
    _work(runtime, _sqs(job))
    assert processed == ["m1", "m1"]
    assert _record(runtime, "user-a", 1)["status"]["S"] == "processed"


def test_a_failed_claim_releases_the_claims_already_taken_and_raises(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    first, second = _jobs(runtime, (1, "user-a"), (2, "user-a"))
    runtime.dynamodb.fail_put_for = "wamid.2"

    with pytest.raises(RuntimeError, match="ProvisionedThroughput"):
        _work(runtime, {"maws_jobs": [first, second]})

    # Lambda retries the invocation: neither message is left claimed, so both are processed.
    assert runtime.dynamodb.items == {}
    runtime.dynamodb.fail_put_for = None
    _work(runtime, {"maws_jobs": [first, second]})
    assert processed == ["m1", "m2"]


# -------------------------
# Opt-out and scope
# -------------------------
def test_dedup_retention_days_zero_turns_the_records_off(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    runtime.dedup_retention_days = 0
    job = _jobs(runtime, (1, "user-a"))[0]

    _work(runtime, {"maws_job": job})
    _work(runtime, {"maws_job": job})

    assert processed == ["m1", "m1"]
    assert runtime.dynamodb.items == {}


def test_without_a_locks_table_messages_are_processed_without_records(workspace_tmp_path):
    runtime, processed, _ = _runtime(workspace_tmp_path)
    table = runtime.dynamodb
    runtime.lock_table = ""
    job = _jobs(runtime, (1, "user-a"))[0]

    _work(runtime, {"maws_job": job})

    assert processed == ["m1"]
    assert table.items == {}


def test_history_append_jobs_are_not_claimed(workspace_tmp_path):
    runtime, _, _ = _runtime(workspace_tmp_path)
    runtime.history_append_enabled = True
    append_job = {"operation": "append_history", "provider": "whatsapp", "chat_id": "user-a", "event_id": "ext-1"}
    message_job = _jobs(runtime, (1, "user-a"))[0]

    claimed = runtime._claim_jobs([append_job, message_job], _Context())

    assert claimed == [append_job, message_job]
    assert list(runtime.dynamodb.items) == ["maws-event#whatsapp#user-a#wamid.1"]


def test_the_runtime_keeps_records_for_eight_days_unless_configured(monkeypatch, workspace_tmp_path):
    for name in ("ENV_PARAMETER_NAME", "AWS_LAMBDA_FUNCTION_NAME", "TOKEN_ENV_MAP_JSON", "MAWS_DEDUP_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAWS_TMP_DIR", str(workspace_tmp_path))
    monkeypatch.setenv("MAWS_CONFIG_PATH", str(workspace_tmp_path / "missing-config.json"))

    assert MawsRuntime()._dedup_retention_seconds() == EIGHT_DAYS
    monkeypatch.setenv("MAWS_DEDUP_RETENTION_DAYS", "0")
    assert MawsRuntime()._dedup_retention_seconds() == 0
