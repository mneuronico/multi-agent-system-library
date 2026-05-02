from __future__ import annotations

from unittest.mock import MagicMock, patch

from mas.mas import Agent, AgentSystemManager


class _Manager:
    def _get_user_db(self):
        return object()

    def _get_all_messages(self, db_conn):
        return []

    def get_key(self, provider):
        return "test-key"

    def _to_blocks(self, value, user_id=None):
        return [{"type": "text", "content": value}]

    def _active_user_id(self):
        return "user-1"

    def cost_model_call(self, provider, model, input_tokens, output_tokens):
        return 0.0


def test_agent_falls_back_when_provider_response_shape_is_invalid(monkeypatch):
    agent = Agent(
        name="agent",
        system_prompt="system",
        system_prompt_original="system",
        required_outputs={"response": "text"},
        models=[
            {"provider": "openai", "model": "bad-response"},
            {"provider": "groq", "model": "fallback"},
        ],
    )
    agent.manager = _Manager()
    agent._build_default_conversation = lambda messages, verbose: [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    def bad_provider(**kwargs):
        raise KeyError("choices")

    def good_provider(**kwargs):
        return '{"response":"fallback ok"}'

    monkeypatch.setattr(agent, "_call_openai_api", bad_provider)
    monkeypatch.setattr(agent, "_call_groq_api", good_provider)

    result = agent.run()

    assert result[0]["content"] == {"response": "fallback ok"}


def test_agent_metadata_keeps_failed_and_successful_provider_attempts(monkeypatch):
    agent = Agent(
        name="agent",
        system_prompt="system",
        system_prompt_original="system",
        required_outputs={"response": "text"},
        models=[
            {"provider": "openai", "model": "bad-response"},
            {"provider": "groq", "model": "fallback"},
        ],
    )
    agent.manager = _Manager()
    agent._build_default_conversation = lambda messages, verbose: [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    def bad_provider(**kwargs):
        raise KeyError("choices")

    def good_provider(**kwargs):
        content = '{"response":"fallback ok"}'
        agent._last_provider_response_metadata = agent._provider_success_metadata(
            "groq",
            "fallback",
            raw_response={
                "id": "chatcmpl-test",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            content=content,
            usage={"prompt_tokens": 3, "completion_tokens": 2},
        )
        return content

    monkeypatch.setattr(agent, "_call_openai_api", bad_provider)
    monkeypatch.setattr(agent, "_call_groq_api", good_provider)

    result = agent.run(return_token_count=False)
    metadata = result[0]["metadata"]

    assert result[0]["content"] == {"response": "fallback ok"}
    assert metadata["provider_response"]["provider"] == "groq"
    assert metadata["provider_response"]["raw_response"]["id"] == "chatcmpl-test"
    assert metadata["provider_response"]["usage"]["input_tokens"] == 3
    assert [attempt["provider"] for attempt in metadata["provider_attempts"]] == ["openai", "groq"]
    assert metadata["provider_attempts"][0]["ok"] is False
    assert metadata["provider_attempts"][0]["errors"][0]["type"] == "KeyError"
    assert metadata["provider_errors"][0]["type"] == "KeyError"


def test_agent_default_output_keeps_provider_errors_when_all_attempts_fail(monkeypatch):
    agent = Agent(
        name="agent",
        system_prompt="system",
        system_prompt_original="system",
        required_outputs={"response": "text"},
        models=[{"provider": "openai", "model": "bad-response"}],
        default_output={"response": "default"},
    )
    agent.manager = _Manager()
    agent._build_default_conversation = lambda messages, verbose: [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    def bad_provider(**kwargs):
        raise ValueError("provider failed")

    monkeypatch.setattr(agent, "_call_openai_api", bad_provider)

    result = agent.run()
    metadata = result[0]["metadata"]

    assert result[0]["content"] == {"response": "default"}
    assert metadata["provider_attempts"][0]["provider"] == "openai"
    assert metadata["provider_attempts"][0]["ok"] is False
    assert metadata["provider_errors"][0]["message"] == "provider failed"


def test_provider_metadata_is_persisted_in_message_history(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.get_key = lambda provider: "test-key"
    manager.create_agent(
        name="agent",
        system="system",
        required_outputs={"response": "text"},
        models=[{"provider": "groq", "model": "model"}],
    )
    agent = manager.agents["agent"]

    def provider(**kwargs):
        content = '{"response":"ok"}'
        agent._last_provider_response_metadata = agent._provider_success_metadata(
            "groq",
            "model",
            raw_response={
                "id": "chatcmpl-history",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
            content=content,
            usage={"prompt_tokens": 4, "completion_tokens": 1},
        )
        return content

    monkeypatch.setattr(agent, "_call_groq_api", provider)

    manager.run(input="hello", component_name="agent", user_id="history-user")

    agent_message = manager.get_messages("history-user")[-1]
    metadata = agent_message["message"][0]["metadata"]
    assert metadata["provider_response"]["raw_response"]["id"] == "chatcmpl-history"
    assert metadata["provider_attempts"][0]["provider"] == "groq"
    assert metadata["provider_errors"] == []


def test_invalid_success_shape_keeps_raw_provider_response():
    agent = Agent(
        name="agent",
        system_prompt="system",
        system_prompt_original="system",
        required_outputs={"response": "text"},
        models=[{"provider": "nvidia", "model": "bad-shape"}],
        default_output={"response": "default"},
    )
    agent.manager = _Manager()
    agent._build_default_conversation = lambda messages, verbose: [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    response = MagicMock()
    response.status_code = 200
    response.headers = {"x-request-id": "req-shape"}
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "unexpected": True,
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    }

    with patch("mas.components.requests.post", return_value=response):
        result = agent.run()

    metadata = result[0]["metadata"]
    attempt = metadata["provider_attempts"][0]
    assert result[0]["content"] == {"response": "default"}
    assert attempt["ok"] is False
    assert attempt["raw_response"]["unexpected"] is True
    assert attempt["request_id"] == "req-shape"
    assert attempt["usage"]["input_tokens"] == 1
    assert metadata["provider_errors"][0]["type"] == "KeyError"
