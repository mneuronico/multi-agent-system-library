from __future__ import annotations

import json

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
