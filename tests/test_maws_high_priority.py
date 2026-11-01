"""Regression tests for the MAWS high-priority production fixes.

Covers: mandatory webhook authentication (runtime + scaffold), FIFO/SQS busy-lock retries,
lock expiry/ownership, lock release when saving fails, `maws remove --project` resource names,
the python3.12 default and the parts of bootstrap.sh that can be checked without jq.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import maws.runtime as runtime_module
from maws import operations
from maws.runtime import LOCK_EXPIRY_MARGIN_SECONDS, MawsRuntime, UserLockBusy


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "maws" / "resources" / "bootstrap.sh"
TELEGRAM_SECRET_CHARS = re.compile(r"[A-Za-z0-9_-]{1,256}")


class _Context:
    invoked_function_arn = "arn:aws:lambda:region:acct:function:test"


class _LambdaContext(_Context):
    def __init__(self, remaining_ms):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


# -------------------------
# Fakes
# -------------------------
class _ConditionalCheckFailed(Exception):
    pass


class _FakeDynamo:
    """In-memory lock table that evaluates the condition expressions MAWS sends."""

    def __init__(self):
        self.items = {}
        self.exceptions = types.SimpleNamespace(ConditionalCheckFailedException=_ConditionalCheckFailed)

    def put_item(self, TableName, Item, ConditionExpression, ExpressionAttributeValues):
        key = Item["user_id"]["S"]
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


class _FakeS3:
    def __init__(self, objects=None, fail_put=False):
        self.objects = dict(objects or {})
        self.puts = {}
        self.fail_put = fail_put

    def download_file(self, bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fp:
            fp.write(self.objects[(bucket, key)])

    def put_object(self, Bucket, Key, Body):
        if self.fail_put:
            raise RuntimeError("s3 put failed")
        self.puts[(Bucket, Key)] = Body


def _lock_runtime(dynamo, now=1_000_000, lock_ttl=135):
    runtime = object.__new__(MawsRuntime)
    runtime.lock_table = "locks"
    runtime.lock_ttl = lock_ttl
    runtime.manager = None
    runtime.history_mode = "per_user"
    runtime.dynamodb = dynamo
    runtime._lock_now_epoch = lambda: now
    return runtime


def _worker_runtime(workspace_tmp_path, dynamo, *, fail_put=False):
    runtime = _lock_runtime(dynamo)
    runtime.bot_type = "telegram"
    runtime.busy_policy = "fifo"
    runtime.bucket_name = "history-bucket"
    runtime.history_prefix = "history"
    runtime.TMP_DIR = str(workspace_tmp_path)
    runtime.special_files = []
    runtime.persist_files_s3 = False
    runtime.capture_failed_events = False
    runtime.busy_wait_seconds = 0  # busy_policy=drop jobs drop without waiting in these tests
    runtime._loop = None
    runtime.s3 = _FakeS3({("history-bucket", "history/123.sqlite"): b"old"}, fail_put=fail_put)
    calls = []

    class _Manager:
        def import_history(self, user_id, sqlite_bytes):
            calls.append(("import", user_id))

        def set_current_user(self, user_id):
            calls.append(("current", user_id))

        def export_history(self, user_id):
            calls.append(("export", user_id))
            return b"new"

    class _Bot:
        async def process_webhook_update(self, update):
            calls.append(("process", update))

    runtime.bot_instance = _Bot()

    def initialize_system():
        runtime.manager = _Manager()

    runtime.initialize_system = initialize_system
    return runtime, calls


def _sqs_event(chat_id="123"):
    job = {
        "provider": "telegram",
        "bot_type": "telegram",
        "chat_id": chat_id,
        "event_id": "99",
        "payload": {"message": {"chat": {"id": int(chat_id)}}},
    }
    return {"Records": [{"eventSource": "aws:sqs", "body": json.dumps({"maws_job": job})}]}


# -------------------------
# 1. Webhook authentication is mandatory
# -------------------------
def test_runtime_defaults_require_webhook_authentication(monkeypatch, workspace_tmp_path):
    for name in (
        "ENV_PARAMETER_NAME",
        "AWS_LAMBDA_FUNCTION_NAME",
        "WHATSAPP_VERIFY_SIGNATURE",
        "MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS",
        "TELEGRAM_WEBHOOK_SECRET_TOKEN",
        "WHATSAPP_APP_SECRET",
        "TOKEN_ENV_MAP_JSON",
        "LOCK_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAWS_TMP_DIR", str(workspace_tmp_path))
    monkeypatch.setenv("MAWS_CONFIG_PATH", str(workspace_tmp_path / "missing-config.json"))

    runtime = MawsRuntime()

    assert runtime.whatsapp_verify_signature is True
    assert runtime.allow_unauthenticated_webhooks is False
    assert runtime.lock_ttl == 180


def _post(runtime, headers=None, body=None):
    invocations = []
    runtime.busy_policy = "drop"
    runtime.lambda_client = types.SimpleNamespace(
        invoke=lambda **kwargs: invocations.append(kwargs) or {"StatusCode": 202}
    )
    if body is None:
        if runtime.bot_type == "telegram":
            body = {"update_id": 1, "message": {"chat": {"id": 7}, "text": "hi"}}
        else:
            body = {"entry": [{"changes": [{"value": {"messages": [{"id": "wamid.1", "from": "u1"}]}}]}]}
    response = MawsRuntime.handle_apigw_event(
        runtime,
        {"requestContext": {"http": {"method": "POST"}}, "headers": headers or {}, "body": json.dumps(body)},
        _Context(),
    )
    return response, invocations


@pytest.mark.parametrize(
    ("bot", "attrs", "headers", "expected_status", "processed"),
    [
        # Missing secret: refused unless the deploy opted out.
        ("telegram", {}, {}, 500, False),
        ("telegram", {"allow_unauthenticated_webhooks": True}, {}, 200, True),
        ("whatsapp", {}, {}, 500, False),
        ("whatsapp", {"whatsapp_verify_signature": True}, {}, 500, False),
        # whatsapp_verify_signature=false alone is not an opt-out.
        ("whatsapp", {"whatsapp_verify_signature": False}, {}, 500, False),
        ("whatsapp", {"whatsapp_verify_signature": False, "allow_unauthenticated_webhooks": True}, {}, 200, True),
        # A configured secret is enforced even with the opt-out.
        ("telegram", {"telegram_secret_token": "s3cret", "allow_unauthenticated_webhooks": True},
         {"X-Telegram-Bot-Api-Secret-Token": "wrong"}, 403, False),
        ("telegram", {"telegram_secret_token": "s3cret"}, {"x-telegram-bot-api-secret-token": "s3cret"}, 200, True),
        ("whatsapp", {"whatsapp_app_secret": "app", "allow_unauthenticated_webhooks": True},
         {"X-Hub-Signature-256": "sha256=bad"}, 403, False),
    ],
)
def test_webhook_post_authentication_matrix(bot, attrs, headers, expected_status, processed):
    runtime = object.__new__(MawsRuntime)
    runtime.bot_type = bot
    for name, value in attrs.items():
        setattr(runtime, name, value)

    response, invocations = _post(runtime, headers=headers)

    assert response["statusCode"] == expected_status
    assert bool(invocations) is processed


def _scaffold(workspace_tmp_path, monkeypatch, bot, name="proj"):
    monkeypatch.setattr(operations._shutil, "which", lambda cmd: None)
    project_dir = workspace_tmp_path / name
    operations.start(project=name, region="us-east-1", bot=bot, project_dir=project_dir, allow_windows=True)
    return (project_dir / ".env.prod").read_text(encoding="utf-8").splitlines()


def test_start_generates_an_uncommented_telegram_webhook_secret(workspace_tmp_path, monkeypatch):
    lines = _scaffold(workspace_tmp_path, monkeypatch, "telegram", "tg1")
    other = _scaffold(workspace_tmp_path, monkeypatch, "telegram", "tg2")

    [secret_line] = [line for line in lines if line.startswith("TELEGRAM_WEBHOOK_SECRET_TOKEN=")]
    secret = secret_line.split("=", 1)[1]
    assert TELEGRAM_SECRET_CHARS.fullmatch(secret)
    assert len(secret) >= 40  # token_urlsafe(32) -> 43 chars
    assert secret not in "\n".join(other)
    assert not any(line.startswith("#") and "TELEGRAM_WEBHOOK_SECRET_TOKEN=" in line for line in lines)


def test_scaffolded_secret_reaches_the_runtime_through_the_ssm_env_loader(workspace_tmp_path, monkeypatch):
    lines = _scaffold(workspace_tmp_path, monkeypatch, "telegram")
    secret = next(line.split("=", 1)[1] for line in lines if line.startswith("TELEGRAM_WEBHOOK_SECRET_TOKEN="))
    raw = "\r\n".join(lines) + "\r\n"  # as written by maws start on Windows
    ssm = types.SimpleNamespace(
        get_parameter=lambda Name, WithDecryption: {"Parameter": {"Value": raw}}
    )
    monkeypatch.setattr(runtime_module, "_require_boto3", lambda: types.SimpleNamespace(client=lambda name: ssm))
    monkeypatch.setenv("ENV_PARAMETER_NAME", "/proj/prod/env")
    monkeypatch.delenv("_ENV_LOADED_FROM_SSM", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET_TOKEN", raising=False)

    runtime_module._load_env_from_ssm_if_needed()

    assert os.environ["TELEGRAM_WEBHOOK_SECRET_TOKEN"] == secret


def test_start_writes_an_explicit_whatsapp_app_secret_line(workspace_tmp_path, monkeypatch):
    lines = _scaffold(workspace_tmp_path, monkeypatch, "whatsapp")

    index = lines.index("WHATSAPP_APP_SECRET=")
    assert "App settings > Basic" in lines[index - 1] or "App settings > Basic" in lines[index - 2]
    assert lines[index - 1].startswith("#") and lines[index - 2].startswith("#")
    assert not any(line.startswith("TELEGRAM_WEBHOOK_SECRET_TOKEN=") for line in lines)


# -------------------------
# 3. FIFO/SQS busy locks, lock expiry and release
# -------------------------
def test_lock_lifetime_follows_the_invocation_deadline():
    dynamo = _FakeDynamo()
    runtime = _lock_runtime(dynamo, now=1_000_000, lock_ttl=999)
    lambda_timeout = 120

    assert MawsRuntime.try_acquire_user_lock(runtime, "u1", context=_LambdaContext(119_001))

    expires_at = int(dynamo.items["u1"]["expiresAt"]["N"])
    assert expires_at == 1_000_000 + 120 + LOCK_EXPIRY_MARGIN_SECONDS
    # Outlives the invocation, but expires before SQS redelivers (visibility = timeout + 30).
    assert 1_000_000 + lambda_timeout < expires_at < 1_000_000 + lambda_timeout + 30


def test_lock_lifetime_falls_back_to_lock_ttl_seconds_without_context():
    dynamo = _FakeDynamo()
    runtime = _lock_runtime(dynamo, now=500, lock_ttl=135)

    assert MawsRuntime.try_acquire_user_lock(runtime, "u1")

    assert int(dynamo.items["u1"]["expiresAt"]["N"]) == 635


def test_expired_lock_is_taken_over_even_before_dynamodb_ttl_deletes_it():
    dynamo = _FakeDynamo()
    dynamo.items["u1"] = {"user_id": {"S": "u1"}, "expiresAt": {"N": "999"}, "lockToken": {"S": "old"}}
    dynamo.items["u2"] = {"user_id": {"S": "u2"}, "expiresAt": {"N": "1001"}, "lockToken": {"S": "live"}}
    runtime = _lock_runtime(dynamo, now=1000)

    assert MawsRuntime.try_acquire_user_lock(runtime, "u1") is True
    assert dynamo.items["u1"]["lockToken"]["S"] != "old"
    assert MawsRuntime.try_acquire_user_lock(runtime, "u2") is False
    assert dynamo.items["u2"]["lockToken"]["S"] == "live"


def test_release_only_deletes_the_workers_own_lock():
    dynamo = _FakeDynamo()
    worker_a = _lock_runtime(dynamo, now=1000, lock_ttl=10)
    assert MawsRuntime.try_acquire_user_lock(worker_a, "u1")

    # A's lock expired; B takes it over before A gets to release.
    worker_b = _lock_runtime(dynamo, now=1020, lock_ttl=10)
    assert MawsRuntime.try_acquire_user_lock(worker_b, "u1")
    b_token = dynamo.items["u1"]["lockToken"]["S"]

    MawsRuntime.release_user_lock(worker_a, "u1")
    assert dynamo.items["u1"]["lockToken"]["S"] == b_token

    MawsRuntime.release_user_lock(worker_b, "u1")
    assert "u1" not in dynamo.items


def test_sqs_job_for_a_busy_user_raises_so_sqs_retries_it(workspace_tmp_path):
    dynamo = _FakeDynamo()
    dynamo.items["123"] = {"user_id": {"S": "123"}, "expiresAt": {"N": str(10**12)}, "lockToken": {"S": "other"}}
    runtime, calls = _worker_runtime(workspace_tmp_path, dynamo)

    with pytest.raises(UserLockBusy):
        MawsRuntime.handle_apigw_event(runtime, _sqs_event(), _LambdaContext(100_000))

    assert not [call for call in calls if call[0] in ("import", "process", "export")]
    assert dynamo.items["123"]["lockToken"]["S"] == "other"  # never released someone else's lock
    assert runtime.s3.puts == {}


def test_direct_job_for_a_busy_user_keeps_drop_behaviour(workspace_tmp_path):
    dynamo = _FakeDynamo()
    dynamo.items["123"] = {"user_id": {"S": "123"}, "expiresAt": {"N": str(10**12)}, "lockToken": {"S": "other"}}
    runtime, calls = _worker_runtime(workspace_tmp_path, dynamo)
    runtime.busy_policy = "drop"
    job = json.loads(_sqs_event()["Records"][0]["body"])

    response = MawsRuntime.handle_apigw_event(runtime, job, _LambdaContext(100_000))

    assert response["statusCode"] == 200
    assert "Already processing" in json.loads(response["body"])
    assert not [call for call in calls if call[0] == "process"]


def test_sqs_job_processes_and_releases_when_the_lock_is_free(workspace_tmp_path):
    dynamo = _FakeDynamo()
    runtime, calls = _worker_runtime(workspace_tmp_path, dynamo)

    response = MawsRuntime.handle_apigw_event(runtime, _sqs_event(), _LambdaContext(100_000))

    assert response["statusCode"] == 200
    assert ("process", {"message": {"chat": {"id": 123}}}) in calls
    assert runtime.s3.puts[("history-bucket", "history/123.sqlite")] == b"new"
    # The user lock is released; only the record that this message was processed stays.
    assert [item["status"]["S"] for item in dynamo.items.values()] == ["processed"]
    assert "123" not in dynamo.items


def test_lock_is_released_when_saving_history_fails_on_the_sqs_path(workspace_tmp_path):
    dynamo = _FakeDynamo()
    runtime, calls = _worker_runtime(workspace_tmp_path, dynamo, fail_put=True)

    with pytest.raises(RuntimeError, match="s3 put failed"):
        MawsRuntime.handle_apigw_event(runtime, _sqs_event(), _LambdaContext(100_000))

    assert ("process", {"message": {"chat": {"id": 123}}}) in calls
    assert "123" not in dynamo.items  # the retry will not find the user busy
    # The replies were already sent, so the SQS redelivery skips the message instead of answering twice.
    calls.clear()
    MawsRuntime.handle_apigw_event(runtime, _sqs_event(), _LambdaContext(100_000))
    assert not [call for call in calls if call[0] == "process"]


# -------------------------
# 4. maws remove --project
# -------------------------
class _FakeCloudFormation:
    def __init__(self, buckets=()):
        self.deleted = []
        self.buckets = list(buckets)

    def describe_stacks(self, StackName):
        return {"Stacks": [{"StackName": StackName, "StackStatus": "UPDATE_COMPLETE"}]}

    def describe_stack_resources(self, StackName):
        return {"StackResources": [
            {"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": bucket} for bucket in self.buckets
        ]}

    def delete_stack(self, StackName):
        self.deleted.append(StackName)

    def get_waiter(self, name):
        return types.SimpleNamespace(wait=lambda StackName: None)


class _FakeBucketS3:
    def __init__(self):
        self.listed = []
        self.deleted_buckets = []
        self.exceptions = types.SimpleNamespace(NoSuchBucket=type("NoSuchBucket", (Exception,), {}))

    def get_paginator(self, name):
        s3 = self

        class _Paginator:
            def paginate(self, Bucket, **kwargs):
                s3.listed.append(Bucket)
                return [{}]

        return _Paginator()

    def list_object_versions(self, Bucket, **kwargs):
        return {}

    def list_objects_v2(self, Bucket, **kwargs):
        return {}

    def delete_objects(self, Bucket, Delete, **kwargs):
        raise AssertionError("no objects expected")

    def delete_bucket(self, Bucket, **kwargs):
        self.deleted_buckets.append(Bucket)


def _run_remove(workspace_tmp_path, monkeypatch, capsys, **kwargs):
    (workspace_tmp_path / "params.json").write_text(json.dumps({
        "project": "alpha",
        "region": "sa-east-1",
        "stack_name": "alpha-custom-stack",
        "history_bucket": "alpha-history",
        "deployment_bucket": "alpha-deploy",
    }), encoding="utf-8")
    cf, s3 = _FakeCloudFormation(buckets=["alpha-history", "beta-history-bucket"]), _FakeBucketS3()
    regions = []
    monkeypatch.setattr(operations, "_boto", lambda region=None: regions.append(region) or (cf, s3, None))
    monkeypatch.setattr(operations, "_ssm_client", lambda region=None: types.SimpleNamespace(delete_parameter=lambda Name: None))
    monkeypatch.setattr(operations, "_account_id", lambda region=None: "123456789012")

    code = operations.remove_project(yes=True, project_dir=workspace_tmp_path, **kwargs)

    assert code == 0
    return cf, s3, regions, capsys.readouterr().out


def test_remove_other_project_ignores_local_params_resource_names(workspace_tmp_path, monkeypatch, capsys):
    cf, s3, regions, out = _run_remove(workspace_tmp_path, monkeypatch, capsys, project="beta")

    assert cf.deleted == ["beta-stack"]
    assert set(s3.listed) == {"beta-history-bucket", "beta-deployment-bucket"}
    assert s3.deleted_buckets == ["beta-deployment-bucket"]
    assert not [name for name in cf.deleted + s3.listed + s3.deleted_buckets if "alpha" in name]
    assert "--project beta differs from params.json (project 'alpha')" in out
    assert "ignoring its stack/bucket/SSM names" in out
    assert regions == ["sa-east-1"]
    assert "pass --region" in out


@pytest.mark.parametrize("project", [None, "alpha"])
def test_remove_own_project_uses_params_resource_names(workspace_tmp_path, monkeypatch, capsys, project):
    cf, s3, _, out = _run_remove(workspace_tmp_path, monkeypatch, capsys, project=project)

    assert cf.deleted == ["alpha-custom-stack"]
    assert set(s3.listed) == {"alpha-history", "alpha-deploy"}
    assert "differs from" not in out


# -------------------------
# 5. Lambda runtime default and setup defaults
# -------------------------
def test_setup_defaults_to_python312_and_whatsapp_signature_verification(workspace_tmp_path):
    conf = workspace_tmp_path / "params.json"
    conf.write_text(json.dumps({"project": "p", "bot": "whatsapp"}), encoding="utf-8")

    operations.setup(project_dir=workspace_tmp_path)

    data = json.loads(conf.read_text(encoding="utf-8"))
    assert data["infra"]["runtime"] == "python3.12" == operations.DEFAULT_LAMBDA_RUNTIME
    assert data["webhook_security"]["whatsapp_verify_signature"] is True


def test_setup_keeps_explicit_runtime_and_security_choices(workspace_tmp_path):
    conf = workspace_tmp_path / "params.json"
    conf.write_text(json.dumps({
        "project": "p",
        "bot": "whatsapp",
        "infra": {"runtime": "python3.11", "memory_size": 512},
        "webhook_security": {"whatsapp_verify_signature": False, "allow_unauthenticated": True},
    }), encoding="utf-8")

    operations.setup(project_dir=workspace_tmp_path)

    data = json.loads(conf.read_text(encoding="utf-8"))
    assert data["infra"] == {"runtime": "python3.11", "memory_size": 512}
    assert data["webhook_security"] == {"whatsapp_verify_signature": False, "allow_unauthenticated": True}


# -------------------------
# bootstrap.sh: static contract (no bash/jq needed)
# -------------------------
def _bootstrap_text():
    return BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_defaults_to_python312_and_tolerates_pip_upgrade_failure():
    bootstrap = _bootstrap_text()

    assert "'.infra.runtime // \"python3.12\"'" in bootstrap
    assert "python3.9" not in bootstrap
    assert 'if ! python3 -m pip install --quiet --upgrade pip "setuptools>=77" wheel; then' in bootstrap
    assert "\npython3 -m pip install" not in bootstrap


def test_bootstrap_lock_ttl_is_derived_from_lambda_timeout():
    bootstrap = _bootstrap_text()
    visibility = int(re.search(r"VISIBILITY_TIMEOUT=\$\(\(LAMBDA_TIMEOUT \+ (\d+)\)\)", bootstrap).group(1))
    lock = int(re.search(r"LOCK_TTL_SECONDS=\$\(\(LAMBDA_TIMEOUT \+ (\d+)\)\)", bootstrap).group(1))

    assert 0 < lock < visibility
    assert 0 < LOCK_EXPIRY_MARGIN_SECONDS < visibility
    assert 'LOCK_TTL_SECONDS: "${LOCK_TTL_SECONDS}"' in bootstrap
    assert 'LOCK_TTL_SECONDS: "180"' not in bootstrap


def test_bootstrap_validates_webhook_auth_before_touching_aws():
    bootstrap = _bootstrap_text()
    check = bootstrap.index(">> Checking webhook authentication")

    for later in ("aws sts get-caller-identity", "aws ssm put-parameter", "sam build\n", "sam deploy"):
        assert check < bootstrap.index(later)
    # Booleans never use `// default` (an explicit false would become the default).
    assert "whatsapp_verify_signature //" not in bootstrap
    assert "allow_unauthenticated //" not in bootstrap
    assert 'MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "${WEBHOOK_ALLOW_UNAUTHENTICATED}"' in bootstrap


def test_bootstrap_set_webhook_is_checked_and_never_cached():
    bootstrap = _bootstrap_text()

    assert '--data-urlencode "url=${WEBHOOK_URL}"' in bootstrap
    assert '--data-urlencode "secret_token=${TELEGRAM_SECRET_VALUE}"' in bootstrap
    assert '-d "url=' not in bootstrap
    assert ">/dev/null\n      tmp=" not in bootstrap
    assert "STATE_FILE" not in bootstrap
    assert "Telegram rejected setWebhook: ${TG_DESCRIPTION}" in bootstrap


def _bash():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")
    return bash


def _bootstrap_function(name):
    match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", _bootstrap_text(), re.S | re.M)
    assert match, f"{name} not found in bootstrap.sh"
    return match.group(0)


def test_bootstrap_env_reader_matches_the_runtime_ssm_parser(workspace_tmp_path, monkeypatch):
    env_text = (
        "# comment\r\n"
        "\r\n"
        "TELEGRAM_TOKEN=123:abc\r\n"
        "  SPACED = padded value \r\n"
        "QUOTED=\"q\"\r\n"
        "export EXPORTED=1\r\n"
        "DUP=first\r\n"
        "DUP=second\r\n"
        "EQUALS=a=b=c\r\n"
        "LAST=no-newline"
    )
    (workspace_tmp_path / ".env.prod").write_bytes(env_text.encode("utf-8"))
    keys = ["TELEGRAM_TOKEN", "SPACED", "QUOTED", "EXPORTED", "export EXPORTED", "DUP", "EQUALS", "LAST", "MISSING"]
    script = workspace_tmp_path / "read_env.sh"
    script.write_bytes((
        "set -euo pipefail\n"
        + _bootstrap_function("env_file_value")
        + 'for k in "$@"; do printf "%s=<%s>\\n" "$k" "$(env_file_value "$k")"; done\n'
    ).encode("utf-8"))

    out = subprocess.run(
        [_bash(), "read_env.sh", *keys], cwd=workspace_tmp_path, capture_output=True, check=True
    ).stdout.decode("utf-8")
    from_bash = dict(re.findall(r"^(.*?)=<(.*)>$", out, re.M))

    ssm = types.SimpleNamespace(get_parameter=lambda Name, WithDecryption: {"Parameter": {"Value": env_text}})
    monkeypatch.setattr(runtime_module, "_require_boto3", lambda: types.SimpleNamespace(client=lambda name: ssm))
    monkeypatch.setenv("ENV_PARAMETER_NAME", "/p/prod/env")
    monkeypatch.delenv("_ENV_LOADED_FROM_SSM", raising=False)
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    runtime_module._load_env_from_ssm_if_needed()
    from_runtime = {key: os.environ.get(key, "") for key in keys}

    assert from_bash == from_runtime
    assert from_bash["TELEGRAM_TOKEN"] == "123:abc"  # CR stripped
    assert from_bash["QUOTED"] == '"q"'  # kept, so the secret check rejects it like Telegram would
    assert from_bash["DUP"] == "first"


def test_bootstrap_telegram_secret_check_matches_telegram_rules(workspace_tmp_path):
    regex = re.search(r"^TELEGRAM_SECRET_RE='(.*)'$", _bootstrap_text(), re.M).group(1)
    values = {
        "abc_DEF-123": True,
        operations._new_telegram_webhook_secret(): True,
        "x" * 256: True,
        "x" * 257: False,
        '"quoted"': False,
        "with space": False,
        "dot.ted": False,
        "café": False,
    }
    script = workspace_tmp_path / "check.sh"
    script.write_bytes((
        f"TELEGRAM_SECRET_RE='{regex}'\n"
        'while IFS= read -r v; do\n'
        '  if [[ "$v" =~ ${TELEGRAM_SECRET_RE} && ${#v} -le 256 ]]; then echo valid; else echo invalid; fi\n'
        "done\n"
    ).encode("utf-8"))

    out = subprocess.run(
        [_bash(), "check.sh"],
        cwd=workspace_tmp_path,
        input="\n".join(values).encode("utf-8") + b"\n",
        capture_output=True,
        check=True,
        env={**os.environ, "LC_ALL": "C.UTF-8"},
    ).stdout.decode("utf-8").split()

    assert out == ["valid" if ok else "invalid" for ok in values.values()]
    assert re.search(
        r'\[\[ "\$\{TELEGRAM_SECRET_VALUE\}" =~ \$\{TELEGRAM_SECRET_RE\} && \$\{#TELEGRAM_SECRET_VALUE\} -le 256 \]\]',
        _bootstrap_text(),
    )
