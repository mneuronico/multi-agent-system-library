from __future__ import annotations

import pytest

from mas.mas import AgentSystemManager, Parser


def _add_history(manager, entries):
    for role, payload in entries:
        manager.add_blocks(payload, role=role, msg_type="component")


def _message_payload(manager, message):
    return manager._blocks_as_tool_input(message["message"])


def _conversation_entries(manager, conversation):
    entries = []
    for message in conversation:
        if message["role"] != "user":
            continue
        blocks = message["content"]
        source = blocks[0]["content"]["source"]
        payload = manager._blocks_as_tool_input(blocks[1:])
        entries.append((source, payload))
    return entries


def test_parser_handles_multiple_sources_with_fields_and_indexes():
    parsed = Parser().parse_input_string("agent_b:(agent_a?-1[answer], tool_x?~[items])")

    assert parsed["component_or_param"] == "agent_b"
    assert parsed["single_source"] is None
    assert parsed["multiple_sources"] == [
        {"component": "agent_a", "index": -1, "fields": ["answer"]},
        {"component": "tool_x", "index": "~", "fields": ["items"]},
    ]
    assert parsed["selection"]["selector"]["type"] == "include"


def test_parser_handles_timeline_wildcards_exclusions_and_anchors():
    last_five = Parser().parse_input_string("agent:*?-5~")
    assert last_five["component_or_param"] == "agent"
    assert last_five["selection"] == {
        "mode": "timeline",
        "selector": {"type": "all"},
        "global_index": (-5, None),
    }

    anchored = Parser().parse_input_string("agent:(research, critic)?planner?-2~")
    assert anchored["selection"] == {
        "mode": "timeline",
        "selector": {
            "type": "include",
            "sources": [
                {"component": "research", "index": None, "fields": None},
                {"component": "critic", "index": None, "fields": None},
            ],
        },
        "global_index": (
            {"type": "anchor", "component": "planner", "index": -2},
            None,
        ),
    }

    excluded = Parser().parse_input_string("agent:*!(debug, internal)?-20~")
    assert excluded["selection"] == {
        "mode": "timeline",
        "selector": {"type": "exclude", "components": ["debug", "internal"]},
        "global_index": (-20, None),
    }


def test_bare_component_range_remains_local_but_parenthesized_singleton_is_global(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("research", {"value": "r1"}),
            ("critic", {"value": "c1"}),
            ("research", {"value": "r2"}),
            ("critic", {"value": "c2"}),
            ("research", {"value": "r3"}),
        ],
    )

    def collect(messages, manager):
        return {
            "sources": [m["source"] for m in messages],
            "values": [_message_payload(manager, m)["value"] for m in messages],
        }

    manager.create_process("collector", collect)

    local = manager.processes["collector"].run(target_input=":research?-2~")
    global_first = manager.processes["collector"].run(target_input=":(research)?-2~")

    assert local == {"sources": ["research", "research"], "values": ["r2", "r3"]}
    assert global_first == {"sources": ["research"], "values": ["r3"]}


def test_process_applies_global_anchor_before_per_source_filters(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("planner", {"value": "p-old"}),
            ("research", {"label": "before", "noise": 0}),
            ("critic", {"summary": "before", "other": "x"}),
            ("planner", {"value": "p-anchor"}),
            ("research", {"label": "after-1", "noise": 1}),
            ("critic", {"summary": "after", "other": "y"}),
            ("planner", {"value": "p-last"}),
            ("research", {"label": "after-2", "noise": 2}),
        ],
    )

    def collect(messages, manager):
        return {
            "sources": [m["source"] for m in messages],
            "payloads": [_message_payload(manager, m) for m in messages],
        }

    manager.create_process("collector", collect)

    out = manager.processes["collector"].run(
        target_input=":(research?-1[label], critic?[summary])?planner?-2~"
    )

    assert out == {
        "sources": ["critic", "research"],
        "payloads": [{"summary": "after"}, {"label": "after-2"}],
    }


def test_tool_negative_selector_applies_after_global_window(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("kept", {"right": "outside-window"}),
            ("debug", {"left": "debug-old", "right": "debug-old"}),
            ("debug", {"right": "debug-in-window"}),
            ("kept", {"left": "inside-window"}),
            ("debug", {"left": "debug-new"}),
        ],
    )

    def echo(left=None, right=None):
        return {"left": left, "right": right}

    manager.create_tool(
        name="echo",
        inputs={"left": "Left", "right": "Right"},
        outputs={"left": "Left", "right": "Right"},
        function=echo,
    )

    out = manager.tools["echo"].run(target_input=":*!(debug)?-3~")

    assert out["left"] == "inside-window"
    assert out["right"] is None


def test_agent_timeline_selection_uses_global_last_messages(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("alpha", {"value": "a1"}),
            ("beta", {"value": "b1"}),
            ("gamma", {"value": "g1"}),
        ],
    )
    manager.create_agent(
        name="reader",
        models=[{"provider": "groq", "model": "test-model"}],
        include_timestamp=False,
    )

    parsed = manager.parser.parse_input_string("reader:*?-2~")
    conversation = manager.agents["reader"]._build_conversation_from_parser_result(
        parsed,
        manager._get_user_db(),
        verbose=False,
    )

    assert _conversation_entries(manager, conversation) == [
        ("beta", {"value": "b1"}),
        ("gamma", {"value": "g1"}),
    ]


def test_automation_string_steps_preserve_timeline_selection_suffix(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("one", {"value": "outside"}),
            ("two", {"value": "inside-1"}),
            ("three", {"value": "inside-2"}),
        ],
    )

    def collect(messages, manager):
        return {
            "sources": [m["source"] for m in messages],
            "values": [_message_payload(manager, m)["value"] for m in messages],
        }

    manager.create_process("collector", collect)
    manager.create_automation("flow", ["collector:*?-2~"])

    out = manager.automations["flow"].run()

    assert out == {"sources": ["two", "three"], "values": ["inside-1", "inside-2"]}


def test_automation_conditions_can_use_timeline_selectors(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("state", {"ok": False}),
            ("debug", {"ok": False}),
            ("state", {"ok": True}),
        ],
    )

    manager.create_process("truth", lambda: {"branch": "true"})
    manager.create_process("falsity", lambda: {"branch": "false"})
    manager.create_automation(
        "brancher",
        [
            {
                "control_flow_type": "branch",
                "condition": {"input": ":*!(debug)?-2~", "value": {"ok": True}},
                "if_true": ["truth"],
                "if_false": ["falsity"],
            }
        ],
    )

    out = manager.automations["brancher"].run()

    assert out == {"branch": "true"}


def test_missing_global_anchor_selects_no_messages(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("research", {"value": "r1"}),
            ("critic", {"value": "c1"}),
        ],
    )

    def collect(messages):
        return {"count": len(messages)}

    manager.create_process("collector", collect)

    out = manager.processes["collector"].run(target_input=":(research)?planner?-1~")

    assert out == {"count": 0}


@pytest.mark.parametrize(
    ("target_input", "expected_sources", "expected_values"),
    [
        (
            ":*",
            ["planner", "research", "debug", "critic", "planner", "research", "critic", "debug", "research"],
            ["p1", "r1", "d1", "c1", "p2", "r2", "c2", "d2", "r3"],
        ),
        (
            ":*?-3~",
            ["critic", "debug", "research"],
            ["c2", "d2", "r3"],
        ),
        (
            ":*!(debug)?-5~",
            ["planner", "research", "critic", "research"],
            ["p2", "r2", "c2", "r3"],
        ),
        (
            ":(research, critic)?planner?-1~",
            ["research", "critic", "research"],
            ["r2", "c2", "r3"],
        ),
        (
            ":(research?-1[answer], critic?~[summary])?planner?-1~",
            ["critic", "research"],
            ["c2", "r3"],
        ),
        (
            ":*?planner?-2~planner?-1",
            ["planner", "research", "debug", "critic"],
            ["p1", "r1", "d1", "c1"],
        ),
    ],
)
def test_process_timeline_selector_matrix(workspace_tmp_path, target_input, expected_sources, expected_values):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("planner", {"value": "p1"}),
            ("research", {"value": "r1", "answer": "r1"}),
            ("debug", {"value": "d1"}),
            ("critic", {"value": "c1", "summary": "c1"}),
            ("planner", {"value": "p2"}),
            ("research", {"value": "r2", "answer": "r2"}),
            ("critic", {"value": "c2", "summary": "c2"}),
            ("debug", {"value": "d2"}),
            ("research", {"value": "r3", "answer": "r3"}),
        ],
    )

    def collect(messages, manager):
        payloads = [_message_payload(manager, m) for m in messages]
        return {
            "sources": [m["source"] for m in messages],
            "values": [p.get("value") or p.get("answer") or p.get("summary") for p in payloads],
        }

    manager.create_process("collector", collect)

    out = manager.processes["collector"].run(target_input=target_input)

    assert out == {"sources": expected_sources, "values": expected_values}


@pytest.mark.parametrize(
    ("target_input", "expected"),
    [
        (":(a,b)", {"x": "a2", "y": "b2", "shared": "b2"}),
        (":*?-2~", {"x": None, "y": "b2", "shared": "b2"}),
        (":*!(debug)?-3~", {"x": "a2", "y": "b2", "shared": "b2"}),
        (":(a?~[x], b?-1[y])", {"x": "a2", "y": "b2", "shared": None}),
    ],
)
def test_tool_timeline_selector_matrix(workspace_tmp_path, target_input, expected):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("a", {"x": "a1", "shared": "a1"}),
            ("b", {"y": "b1", "shared": "b1"}),
            ("a", {"x": "a2", "shared": "a2"}),
            ("debug", {"shared": "debug"}),
            ("b", {"y": "b2", "shared": "b2"}),
        ],
    )

    def echo(x=None, y=None, shared=None):
        return {"x": x, "y": y, "shared": shared}

    manager.create_tool(
        name="echo",
        inputs={"x": "X", "y": "Y", "shared": "Shared"},
        outputs={"x": "X", "y": "Y", "shared": "Shared"},
        function=echo,
    )

    out = manager.tools["echo"].run(target_input=target_input)

    assert {k: out[k] for k in expected} == expected


def test_legacy_group_automation_matches_direct_custom_process_input(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("research", {"value": "r1"}),
            ("critic", {"value": "c1"}),
            ("research", {"value": "r2"}),
        ],
    )

    def collect(messages, manager):
        return {
            "sources": [m["source"] for m in messages],
            "values": [_message_payload(manager, m)["value"] for m in messages],
        }

    manager.create_process("collector", collect)

    old_style = manager.processes["collector"].run(
        target_custom=[
            {"component": "research", "index": None, "fields": None},
            {"component": "critic", "index": None, "fields": None},
        ]
    )
    new_step_style = manager.processes["collector"].run(target_input=":(research, critic)")

    assert new_step_style == old_style


def test_agent_filters_apply_before_timeline_selection(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    _add_history(
        manager,
        [
            ("alpha", {"value": "a1"}),
            ("debug", {"value": "d1"}),
            ("beta", {"value": "b1"}),
            ("gamma", {"value": "g1"}),
        ],
    )
    manager.create_agent(
        name="reader",
        models=[{"provider": "groq", "model": "test-model"}],
        negative_filter=["debug"],
        include_timestamp=False,
    )

    parsed = manager.parser.parse_input_string("reader:*?-3~")
    conversation = manager.agents["reader"]._build_conversation_from_parser_result(
        parsed,
        manager._get_user_db(),
        verbose=False,
    )

    assert _conversation_entries(manager, conversation) == [
        ("alpha", {"value": "a1"}),
        ("beta", {"value": "b1"}),
        ("gamma", {"value": "g1"}),
    ]


def test_parser_rejects_invalid_timeline_selector_shapes():
    parser = Parser()

    with pytest.raises(ValueError):
        parser.parse_input_string("agent:*research")

    with pytest.raises(ValueError):
        parser.parse_input_string("agent:(research)?planner?[field]~")


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


def test_tool_rejects_non_dict_and_missing_output_returns(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    manager.create_tool(
        name="bad_type",
        inputs={},
        outputs={"value": "Value"},
        function=lambda: "not a dict",
        default_output={"value": "default"},
    )
    assert manager.tools["bad_type"].run(input_data={}) == {"value": "default"}

    manager.create_tool(
        name="missing_key",
        inputs={},
        outputs={"value": "Value"},
        function=lambda: {"other": 1},
        default_output={"value": "default"},
    )
    assert manager.tools["missing_key"].run(input_data={}) == {"value": "default"}


def test_process_accepts_manager_and_messages_in_any_order(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.add_blocks({"response": "hello"}, user_id="user-1")

    def summarize(messages, manager):
        return {"count": len(messages), "base": bool(manager.base_directory)}

    manager.create_process("summarizer", summarize)
    out = manager.processes["summarizer"].run()

    assert out == {"count": 1, "base": True}


def test_run_callbacks_accept_two_or_three_arguments(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    calls = []

    def generate(manager):
        return {"response": "done"}

    def on_update(messages, manager):
        calls.append(("update", len(messages), bool(manager.base_directory)))

    def on_complete(messages, manager, params):
        calls.append(("complete", len(messages), params["marker"]))

    manager.create_process("generate", generate)
    manager.run(
        component_name="generate",
        user_id="callback-user",
        on_update=on_update,
        on_complete=on_complete,
        on_complete_params={"marker": "ok"},
    )

    assert calls == [
        ("update", 0, True),
        ("complete", 1, "ok"),
    ]
