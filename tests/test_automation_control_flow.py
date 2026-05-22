from __future__ import annotations

import json
import time

from mas.manager import AgentSystemManager


def test_json_while_with_end_condition_only_builds_and_runs(workspace_tmp_path):
    fns = workspace_tmp_path / "fns.py"
    fns.write_text(
        "def mark(messages, manager):\n"
        "    return {'ran': True}\n",
        encoding="utf-8",
    )
    config = workspace_tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "general_parameters": {"functions": "fns.py"},
                "components": [
                    {"type": "process", "name": "mark", "function": "fn:mark"},
                    {
                        "type": "automation",
                        "name": "flow",
                        "sequence": [
                            {
                                "control_flow_type": "while",
                                "run_first_pass": True,
                                "end_condition": True,
                                "body": ["mark"],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manager = AgentSystemManager(config=str(config), base_directory=str(workspace_tmp_path))
    result = manager.run(component_name="flow", user_id="while-only")

    assert result[0]["content"] == {"ran": True}


def test_branch_condition_treats_string_false_as_false(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("branch-string-bool")
    manager.add_message("router", {"vinculado_a_comida": "false"}, msg_type="agent")
    manager.create_process("image_path", lambda: {"path": "image"})
    manager.create_process("skip_path", lambda: {"path": "skip"})
    manager.create_automation(
        "flow",
        [
            {
                "control_flow_type": "branch",
                "condition": ":router?-1[vinculado_a_comida]",
                "if_true": ["image_path"],
                "if_false": ["skip_path"],
            }
        ],
    )

    result = manager.run(component_name="flow", user_id="branch-string-bool")

    assert result == [{"type": "text", "content": {"path": "skip"}}]


def test_branch_condition_treats_string_true_as_true(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("branch-string-bool-true")
    manager.add_message("router", {"vinculado_a_comida": "true"}, msg_type="agent")
    manager.create_process("image_path", lambda: {"path": "image"})
    manager.create_process("skip_path", lambda: {"path": "skip"})
    manager.create_automation(
        "flow",
        [
            {
                "control_flow_type": "branch",
                "condition": ":router?-1[vinculado_a_comida]",
                "if_true": ["image_path"],
                "if_false": ["skip_path"],
            }
        ],
    )

    result = manager.run(component_name="flow", user_id="branch-string-bool-true")

    assert result == [{"type": "text", "content": {"path": "image"}}]


def test_parallel_automation_preserves_history_order_and_speeds_dependency_dag(workspace_tmp_path):
    sequence = [
        "a",
        "b:a",
        "c:a",
        "d:(a,b)",
        "e:(a,e,f)",
        "f:f",
        "c:(a,c)",
    ]
    delay = 0.08

    def build_manager(name, parallel):
        manager = AgentSystemManager(base_directory=str(workspace_tmp_path / name))

        def make_step(step_name):
            def step(messages=None):
                time.sleep(delay)
                return {
                    "step": step_name,
                    "sources": [message["source"] for message in (messages or [])],
                }
            return step

        for step_name in ["a", "b", "c", "d", "e", "f"]:
            manager.create_process(step_name, make_step(step_name))
        manager.create_automation("flow", sequence, parallel=parallel)
        return manager

    serial = build_manager("serial", parallel=False)
    start = time.perf_counter()
    serial.run(component_name="flow", user_id="dag-user")
    serial_elapsed = time.perf_counter() - start

    parallel = build_manager("parallel", parallel=True)
    start = time.perf_counter()
    parallel.run(component_name="flow", user_id="dag-user")
    parallel_elapsed = time.perf_counter() - start

    def compact_history(manager):
        return [
            (message["source"], manager._blocks_as_tool_input(message["message"]))
            for message in manager.get_messages("dag-user")
        ]

    assert compact_history(parallel) == compact_history(serial)
    assert [source for source, _payload in compact_history(parallel)] == [
        "a", "b", "c", "d", "e", "f", "c"
    ]
    assert parallel_elapsed < serial_elapsed * 0.75


def test_parallel_automation_on_update_follows_logical_commit_order(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    def slow(messages=None):
        time.sleep(0.12)
        return {"step": "slow"}

    def fast(messages=None):
        time.sleep(0.01)
        return {"step": "fast"}

    manager.create_process("slow", slow)
    manager.create_process("fast", fast)
    manager.create_automation("flow", ["slow", "fast:fast"])

    updates = []

    def on_update(messages, manager):
        updates.append([message["source"] for message in messages])

    manager.run(component_name="flow", user_id="callbacks", on_update=on_update)

    assert updates == [["slow"], ["slow", "fast"]]


def test_json_automation_parallel_flag_can_disable_default(workspace_tmp_path):
    fns = workspace_tmp_path / "fns.py"
    fns.write_text(
        "def mark(messages):\n"
        "    return {'ran': True}\n",
        encoding="utf-8",
    )
    config = workspace_tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "general_parameters": {"functions": "fns.py"},
                "components": [
                    {"type": "process", "name": "mark", "function": "fn:mark"},
                    {
                        "type": "automation",
                        "name": "flow",
                        "parallel": False,
                        "sequence": ["mark"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manager = AgentSystemManager(config=str(config), base_directory=str(workspace_tmp_path))

    assert manager.automations["flow"].parallel is False
