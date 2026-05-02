from __future__ import annotations

from mas.manager import AgentSystemManager


def _noop_tool(query=None):
    return {"result": query}


def _true_path():
    return {"path": "true"}


def _false_path():
    return {"path": "false"}


def test_default_models_are_not_shared_between_manager_instances(workspace_tmp_path):
    first_dir = workspace_tmp_path / "first"
    second_dir = workspace_tmp_path / "second"

    first = AgentSystemManager(base_directory=str(first_dir))
    first.default_models.append({"provider": "openai", "model": "example"})

    second = AgentSystemManager(base_directory=str(second_dir))

    assert second.default_models == [{"provider": "groq", "model": "llama-3.1-8b-instant"}]


def test_agent_default_required_outputs_are_isolated_after_linking(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.create_tool(
        name="search_tool",
        inputs={"query": "Search query"},
        outputs={"result": "Search result"},
        function=_noop_tool,
    )

    manager.create_agent(name="agent-one")
    manager.link_tool_to_agent_as_output("search_tool", "agent-one")
    manager.create_agent(name="agent-two")

    assert manager.agents["agent-one"].required_outputs == {"query": "Search query"}
    assert manager.agents["agent-two"].required_outputs == {"response": "Text to send to user."}


def test_auto_names_ignore_non_numeric_prefixed_names(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    manager.create_agent(name="agent-custom")
    manager.create_automation(name="automation-custom", sequence=[])

    assert manager.create_agent() == "agent-1"
    assert manager.create_automation(sequence=[]) == "automation-1"


def test_automation_condition_out_of_range_positive_index_is_false(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("condition-user")
    manager.add_message("marker", {"ok": True}, msg_type="process")
    manager.create_process("true_path", _true_path)
    manager.create_process("false_path", _false_path)
    manager.create_automation(
        "flow",
        [
            {
                "control_flow_type": "branch",
                "condition": ":marker?1?[ok]",
                "if_true": ["true_path"],
                "if_false": ["false_path"],
            }
        ],
    )

    result = manager.run(component_name="flow", user_id="condition-user")

    assert result == [{"type": "text", "content": {"path": "false"}}]
