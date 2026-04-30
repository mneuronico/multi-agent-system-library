from __future__ import annotations

import json
import types

import pytest

from maws.maws import MawsRuntime


class _Context:
    invoked_function_arn = "arn:aws:lambda:region:acct:function:test"


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
