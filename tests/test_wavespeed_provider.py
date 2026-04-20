"""Unit tests for the Wavespeed provider added to mas.Agent._call_wavespeed_api.

The method is exercised via a lightweight stand-in that provides just the
attributes Agent uses (name, model_params, timeout, required_outputs,
_build_json_schema). This avoids spinning up AgentSystemManager / SQLite for
what is a pure HTTP wrapper.

Run with:  python -m pytest tests/test_wavespeed_provider.py -v
An opt-in live test is also available when WAVESPEED_API_KEY_LIVE is set.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mas.mas import Agent  # noqa: E402


def _make_stub(required_outputs=None, model_params=None):
    stub = types.SimpleNamespace()
    stub.name = "stub_agent"
    stub.timeout = 30
    stub.model_params = model_params or {}
    stub.required_outputs = required_outputs or {"response": "The answer."}

    stub._build_json_schema = types.MethodType(Agent._build_json_schema, stub)
    stub._call_wavespeed_api = types.MethodType(Agent._call_wavespeed_api, stub)
    return stub


def _fake_ok(content="Hello", prompt_tokens=7, completion_tokens=3):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return m


def _fake_http_error(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = f"fake {status_code} body"
    err = requests.exceptions.HTTPError(f"{status_code} Error", response=resp)
    m = MagicMock()
    m.raise_for_status.side_effect = err
    m.status_code = status_code
    m.text = resp.text
    return m


def test_primary_endpoint_headers_and_body():
    stub = _make_stub(
        required_outputs={"response": "A short greeting."},
        model_params={"temperature": 0.5, "max_tokens": 100},
    )
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
    ]

    with patch("mas.mas.requests.post", return_value=_fake_ok("ok")) as m_post:
        result = stub._call_wavespeed_api(
            model_name="moonshotai/kimi-k2.5",
            conversation=conversation,
            api_key="sk-test",
            verbose=False,
            return_token_count=True,
        )

    assert m_post.call_count == 1
    args, kwargs = m_post.call_args
    assert args[0] == "https://llm.wavespeed.ai/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    body = kwargs["json"]
    assert body["model"] == "moonshotai/kimi-k2.5"
    assert body["messages"] == conversation
    assert body["response_format"]["type"] == "json_schema"
    assert "schema" in body["response_format"]["json_schema"]
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 100

    content, in_tok, out_tok = result
    assert content == "ok"
    assert in_tok == 7
    assert out_tok == 3


def test_returns_plain_content_when_token_count_disabled():
    stub = _make_stub()
    with patch("mas.mas.requests.post", return_value=_fake_ok("just text")):
        out = stub._call_wavespeed_api(
            model_name="minimax/minimax-m2.5",
            conversation=[{"role": "user", "content": "hi"}],
            api_key="k",
            verbose=False,
            return_token_count=False,
        )
    assert out == "just text"


def test_401_triggers_tropical_fallback():
    stub = _make_stub()

    bad = _fake_http_error(401)
    good = _fake_ok("after fallback")

    with patch("mas.mas.requests.post", side_effect=[bad, good]) as m_post:
        out = stub._call_wavespeed_api(
            model_name="moonshotai/kimi-k2.5",
            conversation=[{"role": "user", "content": "hi"}],
            api_key="stale-key",
            verbose=False,
            return_token_count=False,
        )

    assert m_post.call_count == 2
    primary_url = m_post.call_args_list[0].args[0]
    fallback_url = m_post.call_args_list[1].args[0]
    assert primary_url == "https://llm.wavespeed.ai/v1/chat/completions"
    assert fallback_url == "https://tropical-llm.wavespeed.ai/v1/chat/completions"
    assert out == "after fallback"


def test_non_401_error_is_raised_without_fallback():
    stub = _make_stub()

    bad = _fake_http_error(500)

    with patch("mas.mas.requests.post", side_effect=[bad]) as m_post:
        with pytest.raises(requests.exceptions.HTTPError):
            stub._call_wavespeed_api(
                model_name="moonshotai/kimi-k2.5",
                conversation=[{"role": "user", "content": "hi"}],
                api_key="k",
                verbose=False,
                return_token_count=False,
            )

    assert m_post.call_count == 1


@pytest.mark.skipif(
    not os.environ.get("WAVESPEED_API_KEY_LIVE"),
    reason="Set WAVESPEED_API_KEY_LIVE to hit the real Wavespeed gateway.",
)
def test_live_minimal_call():
    stub = _make_stub(required_outputs={"response": "The short reply."})
    out = stub._call_wavespeed_api(
        model_name="minimax/minimax-m2.5",
        conversation=[
            {"role": "system", "content": "Reply with a single word."},
            {"role": "user", "content": "Say the word READY."},
        ],
        api_key=os.environ["WAVESPEED_API_KEY_LIVE"],
        verbose=False,
        return_token_count=True,
    )
    content, in_tok, out_tok = out
    assert isinstance(content, str) and content.strip()
    assert in_tok > 0
    assert out_tok > 0
