import http.client
import json
import os
import socket
import sqlite3
import threading

import pytest

from mas import AgentSystemManager, dashboard
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
    assert state["components"] == []
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


# --- Read-only state: no project code, no .env, no folders ------------------------

def test_dashboard_state_does_not_run_project_code_or_create_folders(workspace_tmp_path, monkeypatch):
    import mas.manager

    def forbidden(*args, **kwargs):
        raise AssertionError("the dashboard must not build an AgentSystemManager")

    monkeypatch.setattr(mas.manager.AgentSystemManager, "__init__", forbidden)
    marker = workspace_tmp_path / "fns_was_imported.txt"
    (workspace_tmp_path / "fns.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n\ndef collect(messages):\n    return {{}}\n",
        encoding="utf-8",
    )
    (workspace_tmp_path / ".env").write_text("MAS_DASHBOARD_TEST_SECRET=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("MAS_DASHBOARD_TEST_SECRET", raising=False)
    (workspace_tmp_path / "config.json").write_text(
        json.dumps({
            "general_parameters": {"imports": "std->weather_search"},
            "components": [
                {"type": "process", "name": "collect", "function": "fn:collect"},
                {"type": "agent", "name": "answer"},
                {"type": "automation", "name": "flow", "sequence": ["collect", "answer", "weather_search"]},
            ],
        }),
        encoding="utf-8",
    )
    before = sorted(path.name for path in workspace_tmp_path.iterdir())

    state = build_dashboard_state(workspace_tmp_path)

    assert not marker.exists()
    assert "MAS_DASHBOARD_TEST_SECRET" not in os.environ
    assert sorted(path.name for path in workspace_tmp_path.iterdir()) == before  # no history/, files/
    assert state["load_error"] is None
    components = {c["name"]: c for c in state["components"]}
    assert list(components) == ["collect", "answer", "flow", "weather_search"]
    assert components["collect"]["function"] == "collect"
    assert components["answer"]["required_outputs"] == {"response": "Text to send to user."}
    assert components["weather_search"]["function"] == "weather_query"
    assert components["weather_search"]["imported_from"].replace("\\", "/").endswith("std.json")
    assert [a["name"] for a in state["automations"]] == ["flow"]


def test_dashboard_reports_imports_it_cannot_read(workspace_tmp_path):
    (workspace_tmp_path / "config.json").write_text(
        json.dumps({
            "general_parameters": {"imports": ["missing_library", "std?[current_date]"]},
            "components": [{"type": "agent", "name": "answer"}],
        }),
        encoding="utf-8",
    )

    state = build_dashboard_state(workspace_tmp_path)

    assert "missing_library" in state["load_error"]
    assert [c["name"] for c in state["components"]] == ["answer", "current_date"]


# --- HTTP server: Host header, access token, port clashes -------------------------

def _write_minimal_project(root):
    (root / "config.json").write_text(
        json.dumps({"components": [{"type": "agent", "name": "secret_agent", "system": "Top secret."}]}),
        encoding="utf-8",
    )


class _RunningServer:
    def __init__(self, project_dir, **kwargs):
        self.httpd = dashboard.create_dashboard_server(project_dir, "127.0.0.1", 0, **kwargs)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)

    def get(self, path, host=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.putrequest("GET", path, skip_host=True)
            if host is not False:
                conn.putheader("Host", host or f"127.0.0.1:{self.port}")
            for key, value in (headers or {}).items():
                conn.putheader(key, value)
            conn.endheaders()
            response = conn.getresponse()
            return response.status, response.read().decode("utf-8"), dict(response.getheaders())
        finally:
            conn.close()


def test_dashboard_rejects_foreign_host_headers(workspace_tmp_path):
    _write_minimal_project(workspace_tmp_path)

    with _RunningServer(workspace_tmp_path) as server:
        for host in ("attacker.example", f"attacker.example:{server.port}", f"127.0.0.1:{server.port + 1}"):
            status, body, _ = server.get("/api/state", host=host)
            assert status == 403, host
            assert "secret_agent" not in body
        assert server.get("/", host="attacker.example")[0] == 403
        assert server.get("/api/state", host=False)[0] == 403  # no Host header at all

        for host in (f"127.0.0.1:{server.port}", f"localhost:{server.port}", f"[::1]:{server.port}"):
            status, body, headers = server.get("/api/state", host=host)
            assert status == 200, host
            assert "secret_agent" in body
            assert headers["Cache-Control"] == "no-store"
        status, body, headers = server.get("/")
        assert status == 200
        assert "MAS Dashboard" in body
        assert headers["Referrer-Policy"] == "no-referrer"


def test_dashboard_token_protects_the_html_and_the_api(workspace_tmp_path):
    _write_minimal_project(workspace_tmp_path)

    with _RunningServer(workspace_tmp_path, token="s3cret-token") as server:
        status, body, _ = server.get("/api/state")
        assert status == 401
        assert "secret_agent" not in body
        assert server.get("/api/state?token=wrong")[0] == 401
        assert server.get("/api/state", headers={dashboard.TOKEN_HEADER: "wrong"})[0] == 401
        status, body, _ = server.get("/")
        assert status == 401
        assert "?token=" in body

        status, body, _ = server.get("/api/state", headers={dashboard.TOKEN_HEADER: "s3cret-token"})
        assert status == 200
        assert "secret_agent" in body
        assert server.get("/api/state?token=s3cret-token")[0] == 200
        status, body, _ = server.get("/?token=s3cret-token")
        assert status == 200
        assert dashboard.TOKEN_HEADER in body  # the page sends the token back on API calls


def _no_bind(monkeypatch):
    monkeypatch.setattr(dashboard._DashboardServer, "server_bind", lambda self: None)
    monkeypatch.setattr(dashboard._DashboardServer, "server_activate", lambda self: None)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20"])
def test_non_loopback_hosts_always_require_a_token(workspace_tmp_path, monkeypatch, host):
    _no_bind(monkeypatch)  # no real socket on a network interface during tests

    httpd = dashboard.create_dashboard_server(workspace_tmp_path, host, 8765)
    try:
        assert httpd.dashboard_token and len(httpd.dashboard_token) >= 20
        url = dashboard.dashboard_url(httpd)
        assert url.endswith(f"/?token={httpd.dashboard_token}")
        if host == "0.0.0.0":
            assert httpd.allowed_hosts is None  # any name reaches it; the token is what protects it
            assert url.startswith("http://127.0.0.1:8765/")
        else:
            assert httpd.allowed_hosts == {"192.168.1.20:8765"}
            assert url.startswith("http://192.168.1.20:8765/")
    finally:
        httpd.server_close()

    given = dashboard.create_dashboard_server(workspace_tmp_path, host, 8765, token="mine")
    try:
        assert given.dashboard_token == "mine"
    finally:
        given.server_close()


def test_loopback_hosts_accept_only_local_names():
    assert dashboard.allowed_host_headers("127.0.0.1", 8765) == {
        "127.0.0.1:8765", "localhost:8765", "[::1]:8765"
    }
    assert "localhost" in dashboard.allowed_host_headers("localhost", 80)
    assert dashboard.is_loopback_host("::1")
    assert dashboard.is_loopback_host("[::1]")
    assert not dashboard.is_loopback_host("0.0.0.0")
    assert not dashboard.is_loopback_host("10.0.0.5")


def test_a_second_dashboard_on_the_same_port_fails_clearly(workspace_tmp_path, capsys):
    first = dashboard.create_dashboard_server(workspace_tmp_path, "127.0.0.1", 0)
    port = first.server_address[1]
    try:
        with pytest.raises(OSError):
            dashboard.create_dashboard_server(workspace_tmp_path, "127.0.0.1", port)

        assert dashboard.serve_dashboard(workspace_tmp_path, "127.0.0.1", port, open_browser=False) == 1
        assert "Could not start dashboard" in capsys.readouterr().out

        if os.name == "nt":
            # Another program asking for SO_REUSEADDR cannot take the port over either.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
                other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with pytest.raises(OSError):
                    other.bind(("127.0.0.1", port))

        assert dashboard.find_free_port("127.0.0.1", port) != port
    finally:
        first.server_close()


def test_dashboard_main_passes_token_and_auto_port(monkeypatch, workspace_tmp_path):
    from mas.cli import cli

    seen = {}
    monkeypatch.setattr(dashboard, "serve_dashboard", lambda *args, **kwargs: seen.update(kwargs) or 0)
    monkeypatch.setenv(dashboard.TOKEN_ENV_VAR, "from-env")

    assert cli.main(["dashboard", "-d", str(workspace_tmp_path), "--auto-port", "--no-browser"]) == 0
    assert seen["token"] == "from-env"
    assert seen["auto_port"] is True

    assert cli.main(["dashboard", "-d", str(workspace_tmp_path), "--token", "from-flag"]) == 0
    assert seen["token"] == "from-flag"
    assert seen["auto_port"] is False
