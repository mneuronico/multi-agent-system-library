import json
import sqlite3

from mas import AgentSystemManager
from mas.dashboard import build_dashboard_state


def _write_dashboard_project(root):
    (root / "fns.py").write_text(
        "\n".join(
            [
                "def get_weather(messages=None, manager=None):",
                "    return {'weather_summary': 'Sunny, 22 C', 'location': 'Buenos Aires'}",
                "",
                "def format_weather(manager, weather_summary, response=None):",
                "    return {'weather_answer': f'{weather_summary} for {response or \"the request\"}'}",
                "",
                "def save_note(messages=None, manager=None):",
                "    return {'note_status': 'saved'}",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "general_parameters": {
            "api_keys_path": ".env",
            "functions": "fns.py",
            "history_folder": "histories",
            "variables": [
                {"key": "tone", "type": ["warm", "brief"], "default": "warm"}
            ],
        },
        "components": [
            {
                "type": "process",
                "name": "weather_context",
                "description": "Collects today's weather context.",
                "function": "fn:get_weather",
            },
            {
                "type": "agent",
                "name": "router",
                "description": "Chooses actions.",
                "system": "Return the actions to perform.",
                "required_outputs": {"actions": "List of action strings."},
                "default_output": {"actions": ["default"]},
            },
            {
                "type": "tool",
                "name": "weather_formatter",
                "description": "Formats a weather answer.",
                "function": "fn:format_weather",
                "inputs": {
                    "weather_summary": "Weather summary.",
                    "response": "Original user text.",
                },
                "outputs": {"weather_answer": "Answer text."},
            },
            {
                "type": "process",
                "name": "note_saver",
                "description": "Persists a note.",
                "function": "fn:save_note",
            },
            {
                "type": "agent",
                "name": "fallback_agent",
                "system": "Handle unrecognized actions.",
                "required_outputs": {"fallback": "Fallback result."},
            },
            {
                "type": "agent",
                "name": "final_responder",
                "system": "Write the final response in a $tone$ tone.",
                "required_outputs": {"response": "Final user-facing reply."},
            },
            {
                "type": "automation",
                "name": "main_flow",
                "description": "Preprocess, route, act, and respond.",
                "sequence": [
                    "weather_context:user?-1",
                    "router:*?-6~",
                    {
                        "control_flow_type": "for",
                        "items": ":router?-1[actions]",
                        "body": [
                            {
                                "control_flow_type": "switch",
                                "value": ":iterator?-1[item]",
                                "cases": [
                                    {
                                        "case": "answer_weather",
                                        "body": [
                                            "weather_formatter:(weather_context?-1[weather_summary], user?-1[response])"
                                        ],
                                    },
                                    {
                                        "case": "save_note",
                                        "body": ["note_saver:user?-1"],
                                    },
                                    {
                                        "case": "default",
                                        "body": ["fallback_agent:*?router?-1~"],
                                    },
                                ],
                            }
                        ],
                    },
                    "final_responder:*?router?-1~",
                ],
            },
        ],
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_dashboard_state_loads_config_automations_and_histories(workspace_tmp_path):
    config_path = _write_dashboard_project(workspace_tmp_path)
    manager = AgentSystemManager(config=str(config_path), base_directory=str(workspace_tmp_path))
    manager.add_blocks(
        {"response": "Need weather and save this as a note."},
        role="user",
        msg_type="user",
        user_id="user-1",
    )
    manager.add_blocks(
        {"actions": ["answer_weather", "save_note"]},
        role="router",
        msg_type="agent",
        user_id="user-1",
    )
    manager.set_variable("tone", "brief", user_id="user-1")

    state = build_dashboard_state(workspace_tmp_path, history_limit=20)

    assert state["has_config"] is True
    assert state["manager_loaded"] is True
    assert state["load_error"] is None
    assert [component["name"] for component in state["components"]] == [
        "weather_context",
        "router",
        "weather_formatter",
        "note_saver",
        "fallback_agent",
        "final_responder",
        "main_flow",
    ]

    annotated = state["annotated_automations"][0]["annotated_sequence"]
    assert annotated[1]["raw"] == "router:*?-6~"
    assert annotated[1]["parsed_input"]["selection"]["selector"]["type"] == "all"
    assert annotated[2]["kind"] == "for"
    assert annotated[2]["body"][0]["kind"] == "switch"
    assert annotated[-1]["raw"] == "final_responder:*?router?-1~"

    history = next(item for item in state["histories"] if item["user_id"] == "user-1")
    assert [message["role"] for message in history["messages"]] == [
        "user",
        "router",
        "variable",
    ]
    assert history["messages"][0]["blocks"][0]["content"]["response"].startswith("Need weather")
    assert history["messages"][-1]["blocks"][0]["type"] == "variable"
    assert history["messages"][-1]["blocks"][0]["value"] == "brief"


def test_dashboard_groups_shared_history_files_by_user(workspace_tmp_path):
    config = {
        "general_parameters": {
            "history_storage": {
                "mode": "shared",
                "max_messages": 10,
            },
        },
        "components": [],
    }
    config_path = workspace_tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manager = AgentSystemManager(config=str(config_path), base_directory=str(workspace_tmp_path))
    manager.add_blocks({"response": "alice"}, user_id="alice")
    manager.add_blocks({"response": "bob"}, user_id="bob")

    state = build_dashboard_state(workspace_tmp_path, history_limit=20)

    histories = {item["user_id"]: item for item in state["histories"]}
    assert set(histories) == {"alice", "bob"}
    assert histories["alice"]["messages"][0]["blocks"][0]["content"]["response"] == "alice"
    assert histories["bob"]["messages"][0]["blocks"][0]["content"]["response"] == "bob"


def test_dashboard_state_reports_missing_config(workspace_tmp_path):
    state = build_dashboard_state(workspace_tmp_path)

    assert state["has_config"] is False
    assert state["manager_loaded"] is False
    assert "No config.json" in state["load_error"]


def test_dashboard_reads_legacy_history_without_model_column(workspace_tmp_path):
    config_path = workspace_tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "general_parameters": {"history_folder": "history"},
                "components": [
                    {
                        "type": "agent",
                        "name": "hello",
                        "system": "Say hello.",
                        "required_outputs": {"response": "Reply."},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    history_dir = workspace_tmp_path / "history"
    history_dir.mkdir()
    db_path = history_dir / "legacy-user.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE message_history ("
            "msg_number INTEGER PRIMARY KEY, role TEXT, content TEXT, type TEXT, timestamp TEXT)"
        )
        conn.execute(
            "INSERT INTO message_history (msg_number, role, content, type, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                1,
                "hello",
                json.dumps([{"type": "text", "content": {"response": "legacy"}}]),
                "agent",
                "2026-05-05T12:00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    state = build_dashboard_state(workspace_tmp_path)

    history = state["histories"][0]
    assert history["user_id"] == "legacy-user"
    assert history["messages"][0]["model"] is None
    assert history["messages"][0]["blocks"][0]["content"]["response"] == "legacy"


def test_cli_dashboard_command_dispatches(monkeypatch, workspace_tmp_path):
    import mas.dashboard
    from mas.cli import cli

    called = {}

    def fake_dashboard_main(args):
        called["args"] = args
        return 42

    monkeypatch.setattr(mas.dashboard, "dashboard_main", fake_dashboard_main)

    result = cli.main(
        [
            "dashboard",
            "--directory",
            str(workspace_tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            "9876",
            "--auto-port",
            "--no-browser",
            "--history-limit",
            "5",
        ]
    )

    assert result == 42
    assert called["args"].directory == str(workspace_tmp_path)
    assert called["args"].port == 9876
    assert called["args"].auto_port is True
    assert called["args"].no_browser is True
    assert called["args"].history_limit == 5
