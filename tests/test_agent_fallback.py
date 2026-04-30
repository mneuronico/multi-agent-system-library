from __future__ import annotations

from mas.mas import Agent


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
