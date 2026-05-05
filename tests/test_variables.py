from __future__ import annotations

import json

import pytest

from mas.manager import AgentSystemManager


def _write_config(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_agent_resolves_variable_defaults_and_history_overrides(monkeypatch, workspace_tmp_path):
    config = _write_config(
        workspace_tmp_path / "config.json",
        {
            "general_parameters": {
                "variables": [
                    {"key": "mood", "type": ["formal", "casual"], "default": "formal"},
                    {"key": "temp", "type": "number", "default": 0.1},
                    {"key": "model_name", "type": "string", "default": "default-model"},
                ]
            },
            "components": [
                {
                    "type": "agent",
                    "name": "writer",
                    "system": "Use a $mood$ tone.",
                    "required_outputs": {"response": "Answer text."},
                    "models": [{"provider": "openai", "model": "$model_name$"}],
                    "model_params": {"temperature": "$temp$"},
                }
            ],
        },
    )
    manager = AgentSystemManager(config=str(config), base_directory=str(workspace_tmp_path))
    manager.get_key = lambda provider: "test-key"
    calls = []
    agent = manager.agents["writer"]

    def provider(**kwargs):
        calls.append(
            {
                "model": kwargs["model_name"],
                "params": agent._runtime_model_params(),
                "system": kwargs["conversation"][0]["content"],
            }
        )
        return '{"response":"ok"}'

    monkeypatch.setattr(agent, "_call_openai_api", provider)

    manager.run(input="hello", component_name="writer", user_id="user-a")
    assert calls[-1]["model"] == "default-model"
    assert calls[-1]["params"] == {"temperature": 0.1}
    assert "Use a formal tone." in calls[-1]["system"]
    assert '"response": "Answer text."' in calls[-1]["system"]

    manager.set_variable("mood", "casual", user_id="user-a")
    manager.set_variable("temp", 0.8, user_id="user-a")
    manager.set_variable("model_name", "runtime-model", user_id="user-a")

    manager.run(input="again", component_name="writer", user_id="user-a")
    assert calls[-1]["model"] == "runtime-model"
    assert calls[-1]["params"] == {"temperature": 0.8}
    assert "Use a casual tone." in calls[-1]["system"]


def test_component_parameter_overrides_can_be_written_by_history(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.define_variable("tone", type="string", default="calm")
    manager.get_key = lambda provider: "test-key"

    def set_runtime_values(manager):
        return [
            {"type": "variable", "key": "tone", "value": "warm"},
            {"type": "variable", "key": "assistant:provider", "value": "groq"},
            {"type": "variable", "key": "assistant:model", "value": "runtime-model"},
            {"type": "variable", "key": "assistant:temperature", "value": 0.2},
        ]

    manager.create_process("set_runtime_values", set_runtime_values)
    manager.create_agent(
        name="assistant",
        system="Tone: $tone$.",
        required_outputs={"response": "Text."},
        models=[{"provider": "openai", "model": "base-model"}],
        model_params={"temperature": 0.9},
    )
    manager.create_automation("flow", ["set_runtime_values", "assistant"])
    agent = manager.agents["assistant"]
    calls = []

    def provider(**kwargs):
        calls.append(
            {
                "model": kwargs["model_name"],
                "params": agent._runtime_model_params(),
                "system": kwargs["conversation"][0]["content"],
            }
        )
        return '{"response":"ok"}'

    monkeypatch.setattr(agent, "_call_groq_api", provider)

    result = manager.run(component_name="flow", user_id="runtime-user")

    assert result[0]["content"] == {"response": "ok"}
    assert calls[-1]["model"] == "runtime-model"
    assert calls[-1]["params"]["temperature"] == 0.2
    assert "Tone: warm." in calls[-1]["system"]
    assert manager.get_variable("tone", user_id="runtime-user") == "warm"


def test_agent_can_emit_variable_blocks(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.define_variable("mode", type=["draft", "final"], default="draft")
    manager.get_key = lambda provider: "test-key"
    manager.create_agent(
        name="router",
        system="Set mode.",
        models=[{"provider": "groq", "model": "model"}],
    )
    agent = manager.agents["router"]

    monkeypatch.setattr(
        agent,
        "_call_groq_api",
        lambda **kwargs: '{"type":"variable","key":"mode","value":"final"}',
    )

    blocks = manager.run(input="finish", component_name="router", user_id="emit-user")

    assert blocks[0]["type"] == "variable"
    assert blocks[0]["key"] == "mode"
    assert blocks[0]["value"] == "final"
    assert manager.get_variable("mode", user_id="emit-user") == "final"


def test_variable_type_and_enum_validation(workspace_tmp_path):
    missing_default = _write_config(
        workspace_tmp_path / "missing_default.json",
        {
            "general_parameters": {
                "variables": [{"key": "missing_default", "type": "string"}]
            },
            "components": [],
        },
    )
    with pytest.raises(ValueError, match="must include a 'default'"):
        AgentSystemManager(config=str(missing_default), base_directory=str(workspace_tmp_path))

    bad_config = _write_config(
        workspace_tmp_path / "bad_config.json",
        {
            "general_parameters": {
                "variables": [{"key": "level", "type": [1, 2, 3], "default": 4}]
            },
            "components": [],
        },
    )
    with pytest.raises(ValueError, match="must be one of"):
        AgentSystemManager(config=str(bad_config), base_directory=str(workspace_tmp_path))

    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.define_variable("level", type=[1, 2, 3], default=1)
    manager.define_variable("enabled", type="boolean", default=False)

    with pytest.raises(ValueError, match="must be one of"):
        manager.add_blocks({"type": "variable", "key": "level", "value": 4}, user_id="typed-user")

    with pytest.raises(ValueError, match="must be of type"):
        manager.add_blocks({"type": "variable", "key": "enabled", "value": "true"}, user_id="typed-user")

    with pytest.raises(ValueError, match="Unknown variable"):
        manager.add_blocks({"type": "variable", "key": "unknown", "value": "x"}, user_id="typed-user")

    msg_number = manager.add_blocks(
        {"type": "variable", "key": "agent:temperature", "value": 0.3},
        user_id="typed-user",
    )
    assert isinstance(msg_number, int)
    assert manager.get_variable("agent:temperature", user_id="typed-user") == 0.3


def test_variables_are_isolated_per_user_and_clear_to_default(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.define_variable("mood", type="string", default="neutral")

    manager.set_variable("mood", "happy", user_id="alice")
    manager.set_variable("mood", "focused", user_id="bob")

    assert manager.get_variable("mood", user_id="alice") == "happy"
    assert manager.get_variable("mood", user_id="bob") == "focused"

    manager.clear_message_history("alice")

    assert manager.get_variable("mood", user_id="alice") == "neutral"
    assert manager.get_variable("mood", user_id="bob") == "focused"
