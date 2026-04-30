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
