from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import mas.manager as manager_module
from mas import AgentSystemManager
from mas.components import Agent


MODELS = [{"provider": "openai", "model": "test-model"}]


def _project(path, components, general=None, fns=""):
    config = path / "config.json"
    config.write_text(json.dumps({
        "general_parameters": {"default_models": MODELS, **(general or {})},
        "components": components,
    }), encoding="utf-8")
    (path / "fns.py").write_text(fns, encoding="utf-8")
    return config


def _manager(path, components, general=None, fns="", **kwargs):
    config = _project(path, components, general, fns)
    return AgentSystemManager(config=str(config), base_directory=str(path), **kwargs)


# --- Connections after the pool is reset ------------------------------------------

def test_other_threads_reconnect_after_import_history_and_reset(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    worker = ThreadPoolExecutor(max_workers=1)
    worker.submit(lambda: (manager.set_current_user("u1"), manager.add_message("user", "hello"))).result()

    manager.import_history("u1", manager.export_history("u1"))
    worker.submit(lambda: manager.add_message("user", "after import")).result()

    manager.reset_system()
    worker.submit(lambda: manager.add_message("user", "after reset")).result()

    assert [m["message"][0]["content"] for m in manager.get_messages("u1")] == ["after reset"]


def test_import_history_discards_a_stale_write_ahead_log(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.add_blocks("hello", user_id="u1")
    data = manager.export_history("u1")
    db_path = manager._get_db_path_for_user("u1")
    manager._close_all_db_conns()
    with open(db_path + "-wal", "wb") as fp:
        fp.write(b"stale wal from the previous database")

    manager.import_history("u1", data)

    assert not os.path.exists(db_path + "-wal")
    assert len(manager.get_messages("u1")) == 1


# --- Clearing history removes the user's files -------------------------------------

@pytest.mark.parametrize("mode", ["per_user", "shared"])
def test_clear_message_history_deletes_the_users_files(workspace_tmp_path, mode):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path), history_mode=mode)
    mine = manager.save_file(b"mine", user_id="u1")[5:]
    other = manager.save_file(b"other", user_id="u2")[5:]
    manager.set_current_user("u1")
    tts = manager._save_tts_file(b"audio")

    manager.clear_message_history("u1")

    assert not os.path.exists(mine) and not os.path.exists(tts)
    assert os.path.exists(other)
    assert all(not path.startswith(os.path.dirname(mine)) for path in manager._file_cache)


def test_tts_files_live_in_the_users_folder(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("../escape")

    path = manager._save_tts_file(b"audio")

    assert os.path.realpath(path).startswith(os.path.realpath(manager.files_folder) + os.sep)
    assert manager._safe_user_id("../escape") in path


def test_clear_global_history_deletes_all_files(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    stored = manager.save_file(b"data", user_id="u1")[5:]

    manager.clear_global_history()

    assert not os.path.exists(stored)
    assert os.path.isdir(manager.files_folder)


def test_file_cache_is_bounded(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module._BoundedFileCache, "max_items", 3)
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    refs = [manager.save_file(f"file {i}".encode(), user_id="u1") for i in range(10)]

    assert len(manager._file_cache) == 3
    assert manager._load_files_in_dict({"x": refs[0]})["x"] == b"file 0"


# --- Usage logging ---------------------------------------------------------------

def _provider_with_usage(self, model_name, conversation, **kwargs):
    content = json.dumps({"response": "ok"})
    Agent._remember_provider_response(self, Agent._provider_success_metadata(
        self, "openai", model_name, content=content,
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
    ))
    return content


def test_usage_is_logged_without_return_token_count(workspace_tmp_path, monkeypatch):
    (workspace_tmp_path / "costs.json").write_text(json.dumps({
        "models": {"openai": {"test-model": {"input_per_1m": 1.0, "output_per_1m": 2.0}}}
    }), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(Agent, "_call_openai_api", _provider_with_usage)
    manager = _manager(workspace_tmp_path, [{"type": "agent", "name": "a"}], usage_logging=True)

    manager.run("hi", component_name="a", user_id="u")

    entry = json.loads(open(manager._usage_log_path, encoding="utf-8").read().splitlines()[-1])
    assert (entry["in_toks"], entry["out_toks"]) == (1000, 500)
    assert entry["cost"] == pytest.approx(0.002)
    stats = manager.get_usage_stats("u")["models"]["openai:test-model"]
    assert (stats["input_tokens"], stats["output_tokens"]) == (1000, 500)
    assert manager.get_cost_report()["overall_cost"] == pytest.approx(0.002)


def test_cost_report_without_usage_logging_explains_itself(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    with pytest.raises(RuntimeError, match="usage_logging=True"):
        manager.get_cost_report()


# --- Provider parameters and truncation -------------------------------------------

def _openai_response(content, finish_reason="stop"):
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    return response


def test_openai_uses_max_completion_tokens_and_unsupported_params_are_reported(workspace_tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    manager = _manager(workspace_tmp_path, [{
        "type": "agent", "name": "a", "model_params": {"max_tokens": 64, "seed": 7, "reasoning_effort": "low"},
    }])

    with patch("mas.components.requests.post", return_value=_openai_response('{"response": "ok"}')) as post, \
            caplog.at_level(logging.WARNING):
        manager.run("hi", component_name="a", user_id="u")
        manager.run("again", component_name="a", user_id="u")

    body = post.call_args.kwargs["json"]
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
    assert caplog.text.count("are not sent to any provider") == 1
    assert "reasoning_effort" in caplog.text and "seed" in caplog.text


def test_truncated_responses_fall_back_to_the_next_model(workspace_tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    manager = _manager(workspace_tmp_path, [{
        "type": "agent", "name": "a",
        "models": [{"provider": "openai", "model": "small"}, {"provider": "groq", "model": "big"}],
    }])
    responses = [_openai_response('{"response": "a partial ans', "length"), _openai_response('{"response": "full"}')]

    with patch("mas.components.requests.post", side_effect=responses):
        out = manager.run("hi", component_name="a", user_id="u")

    assert out[0]["content"] == {"response": "full"}
    errors = out[0]["metadata"]["provider_errors"]
    assert "truncated" in errors[0]["message"]


# --- Context window ----------------------------------------------------------------

def test_max_context_messages_limits_what_the_agent_reads(workspace_tmp_path, monkeypatch):
    seen = []

    def provider(self, model_name, conversation, **kwargs):
        seen.append(conversation)
        return json.dumps({"response": "ok"})

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(Agent, "_call_openai_api", provider)
    manager = _manager(
        workspace_tmp_path,
        [{"type": "agent", "name": "short"}, {"type": "agent", "name": "long", "max_context_messages": 4}],
        general={"max_context_messages": 2},
    )
    for i in range(5):
        manager.add_blocks(f"m{i}", user_id="u")

    manager.run(component_name="short", user_id="u")
    manager.run(component_name="long", user_id="u")

    assert len(seen[0]) == 1 + 2   # system prompt + last 2 messages
    assert len(seen[1]) == 1 + 4
    with pytest.raises(ValueError, match="max_context_messages"):
        AgentSystemManager(base_directory=str(workspace_tmp_path), max_context_messages=0)


# --- Control flow ----------------------------------------------------------------

TICK_FNS = """COUNT = {"n": 0}

def tick(messages):
    COUNT["n"] += 1
    return {"keep_going": COUNT["n"] < 3, "query": "q%d" % COUNT["n"]}

def check(data):
    CHECKED.append(data)
    return data.get("query") == "q2"

CHECKED = []
"""


def _ticks(manager, user):
    return [m for m in manager.get_messages(user) if m["source"] == "tick"]


def test_while_condition_loops_while_it_holds(workspace_tmp_path):
    manager = _manager(workspace_tmp_path, [
        {"type": "process", "name": "tick", "function": "fn:tick"},
        {"type": "automation", "name": "main", "sequence": [
            "tick",
            {"control_flow_type": "while", "condition": ":tick?[keep_going]", "body": ["tick"]},
        ]},
    ], fns=TICK_FNS)

    manager.run("go", user_id="w")

    assert len(_ticks(manager, "w")) == 3


def test_readme_function_condition_with_input_works(workspace_tmp_path):
    manager = _manager(workspace_tmp_path, [
        {"type": "process", "name": "tick", "function": "fn:tick"},
        {"type": "automation", "name": "main", "sequence": [
            {"control_flow_type": "while", "start_condition": True,
             "end_condition": "fn:check:tick?[query]", "body": ["tick"]},
        ]},
    ], fns=TICK_FNS)

    manager.run("go", user_id="w")

    assert len(_ticks(manager, "w")) == 2
    module = next(iter(manager._function_cache.values()))
    assert module.CHECKED == [{"query": "q1"}, {"query": "q2"}]


def test_function_condition_without_input_gets_the_latest_message(workspace_tmp_path):
    fns = TICK_FNS + "\ndef is_q1(data):\n    return data.get('query') == 'q1'\n"
    manager = _manager(workspace_tmp_path, [
        {"type": "process", "name": "tick", "function": "fn:tick"},
        {"type": "process", "name": "yes", "function": "fn:tick"},
        {"type": "automation", "name": "main", "sequence": [
            "tick",
            {"control_flow_type": "branch", "condition": "fn:is_q1", "if_true": ["yes"], "if_false": []},
        ]},
    ], fns=fns)

    manager.run("go", user_id="w")

    assert [m["source"] for m in manager.get_messages("w")][-1] == "yes"


def test_missing_condition_function_fails_when_the_config_loads(workspace_tmp_path):
    with pytest.raises(ValueError, match="nope"):
        _manager(workspace_tmp_path, [
            {"type": "process", "name": "tick", "function": "fn:tick"},
            {"type": "automation", "name": "main", "sequence": [
                {"control_flow_type": "branch", "condition": "fn:nope", "if_true": ["tick"], "if_false": []},
            ]},
        ], fns=TICK_FNS)


def test_while_loops_stop_at_max_iterations(workspace_tmp_path):
    manager = _manager(workspace_tmp_path, [
        {"type": "process", "name": "tick", "function": "fn:tick"},
        {"type": "automation", "name": "main", "sequence": [
            {"control_flow_type": "while", "condition": True, "max_iterations": 5, "body": ["tick"]},
        ]},
    ], fns=TICK_FNS)

    with pytest.raises(RuntimeError, match="max_iterations=5"):
        manager.run("go", user_id="w")
    assert len(_ticks(manager, "w")) == 5


# --- Tool input coercion -----------------------------------------------------------

def test_tool_coercion_respects_str_annotations(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))

    def tool(name: str, flag, note: str = ""):
        return {"seen": [name, flag, note]}

    manager.create_tool("t", {"name": "", "flag": "", "note": ""}, {"seen": ""}, tool)

    out = manager.tools["t"].run(input_data={"name": "None", "flag": "false", "note": ""})

    assert out["seen"] == ["None", False, ""]


def test_add_message_accepts_a_user_id_for_that_call_only(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.set_current_user("alice")

    manager.add_message("internal", "note for bob", user_id="bob")

    assert manager._uid() == "alice"
    assert len(manager.get_messages("bob")) == 1


def test_saved_file_paths_load_as_the_file_contents(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    source = workspace_tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4 data")

    ref = manager.save_file(str(source), user_id="u1")

    assert manager._load_files_in_dict({"x": ref})["x"] == b"%PDF-1.4 data"


def test_concurrent_use_of_one_users_history_returns_correct_rows(workspace_tmp_path):
    # Threads serving the same user share one pooled connection. Without serializing it, a read
    # running while another thread writes got rows from the other thread's query (1-column
    # tuples) or too few rows; a parallel automation step then saw no input at all.
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    for i in range(60):
        manager.add_blocks(f"m{i}", user_id="u")
    conn = manager._ensure_user_db("u")

    def write():
        manager.set_current_user("u")
        for n in range(200):
            manager.add_message("w", f"x{n}")

    def read(_):
        problems = []
        for _ in range(150):
            rows = manager._get_all_messages(conn)
            numbers = [row[2] for row in rows if len(row) == 5]
            if len(rows) < 60 or len(numbers) != len(rows) or numbers != list(range(1, len(rows) + 1)):
                problems.append(len(rows))
        return problems

    writer = threading.Thread(target=write)
    writer.start()
    with ThreadPoolExecutor(max_workers=6) as pool:
        problems = [p for result in pool.map(read, range(6)) for p in result]
    writer.join()

    assert problems == []
