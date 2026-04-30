from __future__ import annotations

from mas.mas import AgentSystemManager, Parser


def test_parser_handles_multiple_sources_with_fields_and_indexes():
    parsed = Parser().parse_input_string("agent_b:(agent_a?-1[answer], tool_x?~[items])")

    assert parsed["component_or_param"] == "agent_b"
    assert parsed["single_source"] is None
    assert parsed["multiple_sources"] == [
        {"component": "agent_a", "index": -1, "fields": ["answer"]},
        {"component": "tool_x", "index": "~", "fields": ["items"]},
    ]


def test_tool_invokes_manager_aware_function_with_keyword_inputs(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    def combine(manager, left, right="default"):
        return {"combined": f"{left}-{right}"}

    manager.create_tool(
        name="combiner",
        inputs={"left": "Left side", "right": "Right side"},
        outputs={"combined": "Combined value"},
        function=combine,
    )

    out = manager.tools["combiner"].run(input_data={"left": "a", "right": "b"})

    assert out["combined"] == "a-b"
    assert out["metadata"]["usd_cost"] == 0.0


def test_process_accepts_manager_and_messages_in_any_order(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.add_blocks({"response": "hello"}, user_id="user-1")

    def summarize(messages, manager):
        return {"count": len(messages), "base": bool(manager.base_directory)}

    manager.create_process("summarizer", summarize)
    out = manager.processes["summarizer"].run()

    assert out == {"count": 1, "base": True}
