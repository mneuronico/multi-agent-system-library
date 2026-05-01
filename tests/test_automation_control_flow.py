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
