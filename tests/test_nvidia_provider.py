from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from mas.mas import Agent


def _make_stub(required_outputs=None, model_params=None):
    stub = types.SimpleNamespace()
    stub.name = "nvidia_stub"
    stub.timeout = 30
    stub.model_params = model_params or {}
    stub.required_outputs = required_outputs or {"response": "The answer."}
    stub._call_nvidia_api = types.MethodType(Agent._call_nvidia_api, stub)
    return stub


def _fake_ok(content='{"response":"ok"}', prompt_tokens=11, completion_tokens=5):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return response


def test_nvidia_provider_uses_integrate_chat_completions_endpoint():
    stub = _make_stub(model_params={"temperature": 0, "max_tokens": 64})
    conversation = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
    ]

    with patch("mas.components.requests.post", return_value=_fake_ok()) as post:
        content, in_tokens, out_tokens = stub._call_nvidia_api(
            model_name="nvidia/llama-3.1-nemotron-nano-8b-v1",
            conversation=conversation,
            api_key="nvapi-test",
            verbose=False,
            return_token_count=True,
        )

    assert post.call_count == 1
    args, kwargs = post.call_args
    assert args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer nvapi-test"
    assert kwargs["headers"]["Accept"] == "application/json"
    body = kwargs["json"]
    assert body["model"] == "nvidia/llama-3.1-nemotron-nano-8b-v1"
    assert body["messages"] == conversation
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0
    assert body["max_tokens"] == 64
    assert kwargs["timeout"] == 30
    assert content == '{"response":"ok"}'
    assert in_tokens == 11
    assert out_tokens == 5


def test_nvidia_provider_allows_base_url_override():
    stub = _make_stub()

    with patch("mas.components.requests.post", return_value=_fake_ok()) as post:
        stub._call_nvidia_api(
            model_name="local/model",
            conversation=[{"role": "user", "content": "hi"}],
            api_key="key",
            verbose=False,
            base_url="http://localhost:8000/v1/",
        )

    assert post.call_args.args[0] == "http://localhost:8000/v1/chat/completions"


def test_nvidia_provider_handles_reasoning_only_response_shape():
    stub = _make_stub()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": None, "reasoning_content": "thinking"}}],
        "usage": {},
    }

    with patch("mas.components.requests.post", return_value=response):
        content = stub._call_nvidia_api(
            model_name="openai/gpt-oss-20b",
            conversation=[{"role": "user", "content": "hi"}],
            api_key="key",
            verbose=False,
        )

    assert content == "thinking"


def test_nvidia_provider_formats_messages_as_text_strings():
    agent = Agent(
        name="nvidia_formatter",
        system_prompt="Return JSON.",
        system_prompt_original="Return JSON.",
        required_outputs={"response": "The response."},
        models=[],
    )
    agent.manager = types.SimpleNamespace(
        _to_blocks=lambda value: [{"type": "text", "content": value}]
    )

    formatted = agent._provider_format_messages(
        "nvidia",
        [
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": {"source": "make_loop_items"}},
                    {
                        "type": "text",
                        "content": {
                            "total": 10,
                            "items": ["alpha", "beta"],
                        },
                    },
                ],
            },
        ],
    )

    assert formatted[0] == {"role": "system", "content": "Return JSON."}
    assert isinstance(formatted[1]["content"], str)
    assert '"total": 10' in formatted[1]["content"]
    assert "alpha" in formatted[1]["content"]
