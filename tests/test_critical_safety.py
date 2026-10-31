from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import mas.manager as manager_module
from mas import AgentSystemManager
from mas.bots import WhatsappBot
from mas.components import Agent


ROOT = Path(__file__).resolve().parents[1]


def _write_project(workspace_tmp_path, components, fns=""):
    config_path = workspace_tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "general_parameters": {"default_models": [{"provider": "openai", "model": "test-model"}]},
        "components": components,
    }), encoding="utf-8")
    (workspace_tmp_path / "fns.py").write_text(fns, encoding="utf-8")
    return config_path


def _forbid_bootstrap(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("bootstrap must not run")

    monkeypatch.setattr(manager_module, "get_readme", fail)


# --- Description-mode bootstrap -------------------------------------------------

@pytest.mark.parametrize("config", ["configs/prod_config", "config", "missing.json", "./my configs/prod"])
def test_missing_config_path_raises_instead_of_bootstrapping(workspace_tmp_path, monkeypatch, config):
    _forbid_bootstrap(monkeypatch)
    monkeypatch.chdir(workspace_tmp_path)

    with pytest.raises(FileNotFoundError, match="config file not found"):
        AgentSystemManager(config=config, base_directory=str(workspace_tmp_path))


def test_bootstrap_refuses_to_overwrite_existing_project(workspace_tmp_path, monkeypatch):
    _forbid_bootstrap(monkeypatch)
    (workspace_tmp_path / "config.json").write_text("{}", encoding="utf-8")
    history = workspace_tmp_path / "history"
    history.mkdir()
    (history / "customer.sqlite").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="would overwrite"):
        AgentSystemManager(config="a bot that answers questions", base_directory=str(workspace_tmp_path))

    assert (history / "customer.sqlite").read_text(encoding="utf-8") == "keep me"


def test_bootstrap_keeps_existing_history_and_files(workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "get_readme", lambda *args, **kwargs: "readme")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = {
        "general_parameters": {
            "base_directory": "/somewhere/else",
            "default_models": [{"provider": "openai", "model": "test-model"}],
        },
        "components": [{"type": "agent", "name": "generated"}],
        "fns": ["def helper():\n    return 1\n"],
    }
    monkeypatch.setattr(Agent, "_call_openai_api", lambda self, **kwargs: json.dumps(payload))
    for folder in ("history", "files"):
        (workspace_tmp_path / folder).mkdir()
        (workspace_tmp_path / folder / "keep.txt").write_text("keep me", encoding="utf-8")

    manager = AgentSystemManager(config="a bot that answers questions", base_directory=str(workspace_tmp_path))

    assert "generated" in manager.agents
    assert manager.base_directory == str(workspace_tmp_path)
    assert (workspace_tmp_path / "history" / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert (workspace_tmp_path / "files" / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert "def helper" in (workspace_tmp_path / "fns.py").read_text(encoding="utf-8")
    assert not (workspace_tmp_path / "_bootstrap_config.json").exists()


def test_existing_config_file_with_uppercase_extension_is_loaded(workspace_tmp_path):
    config_path = _write_project(workspace_tmp_path, [{"type": "agent", "name": "a"}])
    upper = workspace_tmp_path / "CONFIG.JSON"
    upper.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    manager = AgentSystemManager(config=str(upper), base_directory=str(workspace_tmp_path))

    assert list(manager.agents) == ["a"]


# --- file: references ----------------------------------------------------------

ECHO_TOOL = {
    "type": "tool",
    "name": "echo",
    "inputs": {"response": "value"},
    "outputs": {"seen": "repr of the value"},
    "function": "fn:echo",
}
ECHO_FNS = "def echo(response):\n    return {'seen': repr(response)}\n"


class _Payload:
    def __reduce__(self):
        return (exec, ("import os; os.environ['MAS_PICKLE_PAYLOAD_RAN'] = '1'",))


def test_user_text_with_file_prefix_is_not_read_from_disk(workspace_tmp_path, monkeypatch):
    monkeypatch.delenv("MAS_PICKLE_PAYLOAD_RAN", raising=False)
    config_path = _write_project(workspace_tmp_path, [ECHO_TOOL], ECHO_FNS)
    secret = workspace_tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    evil = workspace_tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps(_Payload()))
    manager = AgentSystemManager(config=str(config_path), base_directory=str(workspace_tmp_path))

    out = manager.run(f"file:{secret}", component_name="echo", user_id="u1")
    assert out[0]["content"]["seen"] == repr(f"file:{secret}")

    manager.run(f"file:{evil}", component_name="echo", user_id="u1")
    assert "MAS_PICKLE_PAYLOAD_RAN" not in os.environ


def test_file_references_inside_files_folder_still_load(workspace_tmp_path):
    config_path = _write_project(workspace_tmp_path, [ECHO_TOOL], ECHO_FNS)
    manager = AgentSystemManager(config=str(config_path), base_directory=str(workspace_tmp_path))
    ref = manager.save_file(b"stored bytes", user_id="u1")

    out = manager.run({"response": ref}, component_name="echo", user_id="u1")

    assert out[0]["content"]["seen"] == repr(b"stored bytes")


def test_copied_pickle_files_are_never_stored_as_pickles(workspace_tmp_path):
    config_path = _write_project(workspace_tmp_path, [])
    manager = AgentSystemManager(config=str(config_path), base_directory=str(workspace_tmp_path))
    source = workspace_tmp_path / "upload.pkl"
    source.write_bytes(pickle.dumps({"a": 1}))

    ref = manager.save_file(str(source), user_id="u1")

    assert ref.endswith(".bin")
    manager.clear_file_cache()
    assert manager._load_files_in_dict({"x": ref})["x"] == source.read_bytes()


# --- WhatsApp webhook signature ------------------------------------------------

class _BotManager:
    on_update = None
    on_complete = None
    admin_user_id = None

    def __init__(self, keys):
        self.keys = keys

    def get_key(self, name):
        return self.keys.get(name)


BOT_KEYS = {
    "whatsapp_token": "token",
    "whatsapp_phone_number_id": "phone-id",
    "webhook_verify_token": "verify",
}


def _sign(secret, body):
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_whatsapp_signature_verification():
    bot = WhatsappBot(manager=_BotManager({**BOT_KEYS, "whatsapp_app_secret": "app-secret"}))
    body = b'{"entry": []}'

    assert bot.verify_webhook_signature(body, _sign("app-secret", body))
    assert not bot.verify_webhook_signature(body, _sign("other-secret", body))
    assert not bot.verify_webhook_signature(body + b" ", _sign("app-secret", body))
    assert not bot.verify_webhook_signature(body, None)
    assert not bot.verify_webhook_signature(body, "sha256=é")


def test_whatsapp_server_refuses_to_start_without_app_secret():
    bot = WhatsappBot(manager=_BotManager(dict(BOT_KEYS)))

    with pytest.raises(ValueError, match="app secret"):
        bot.create_flask_app()


def test_whatsapp_server_rejects_unsigned_posts(monkeypatch):
    flask = pytest.importorskip("flask")
    if not hasattr(flask.Flask, "test_client"):
        pytest.skip("Flask is stubbed in this environment")

    bot = WhatsappBot(manager=_BotManager({**BOT_KEYS, "whatsapp_app_secret": "app-secret"}))
    scheduled = []
    bot.process_webhook_update = lambda data: scheduled.append(data)
    import mas.bots as bots_module
    monkeypatch.setattr(bots_module.asyncio, "run_coroutine_threadsafe", lambda coro, loop: None)
    client = bot.create_flask_app("/webhook").test_client()
    body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
        {"from": "admin", "type": "text", "text": {"body": "/reset_system"}}
    ]}}]}]}).encode()

    assert client.post("/webhook", data=body).status_code == 403
    assert client.post("/webhook", data=body,
                       headers={"X-Hub-Signature-256": _sign("wrong", body)}).status_code == 403
    assert scheduled == []

    ok = client.post("/webhook", data=body, headers={"X-Hub-Signature-256": _sign("app-secret", body)})
    assert ok.status_code == 200
    assert len(scheduled) == 1


# --- MAWS Lambda package -------------------------------------------------------

def _staging_script() -> str:
    bootstrap = (ROOT / "maws" / "resources" / "bootstrap.sh").read_text(encoding="utf-8")
    match = re.search(r"python3 - \"\$\{PACKAGE_DIR\}\" <<'PY'\n(.*?)\nPY\n", bootstrap, re.S)
    assert match, "package staging step not found in bootstrap.sh"
    return match.group(1)


def test_maws_template_deploys_only_the_staged_package():
    bootstrap = (ROOT / "maws" / "resources" / "bootstrap.sh").read_text(encoding="utf-8")

    assert "CodeUri: ${PACKAGE_DIR}" in bootstrap
    assert "CodeUri: .\n" not in bootstrap
    assert bootstrap.index("Staging Lambda package") < bootstrap.index("sam build\n")


def test_maws_package_excludes_secrets_and_histories(workspace_tmp_path):
    project = workspace_tmp_path / "bot"
    files = {
        "config.json": "{}",
        "fns.py": "def f():\n    pass\n",
        "lambda_function.py": "handler = None\n",
        "requirements.txt": "mas\n",
        "token.json": "{}",
        "helpers/util.py": "X = 1\n",
        ".env.prod": "OPENAI_API_KEY=sk-secret\n",
        ".env": "OPENAI_API_KEY=sk-secret\n",
        "helpers/.env": "OPENAI_API_KEY=sk-secret\n",
        "api_keys.json": '{"openai": "sk-secret"}',
        "history/u1.sqlite": "chat",
        "files/u1/a.png": "img",
        "data/cache.sqlite": "db",
        "params.json": "{}",
        "samconfig.toml": "",
        "template.yaml": "",
        ".aws-sam/build/x": "",
        "__pycache__/fns.cpython-312.pyc": "",
    }
    for rel, content in files.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    script = workspace_tmp_path / "stage.py"
    script.write_text(_staging_script(), encoding="utf-8")

    subprocess.check_call([sys.executable, str(script), ".maws_package"], cwd=project)

    package = project / ".maws_package"
    shipped = sorted(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file())
    assert shipped == [
        "config.json",
        "fns.py",
        "helpers/util.py",
        "lambda_function.py",
        "requirements.txt",
        "token.json",
    ]

    # Like bootstrap.sh: wipe and re-stage. copytree creates the destination before walking the
    # project, so the destination itself must be skipped or it would be copied into itself.
    shutil.rmtree(package)
    subprocess.check_call([sys.executable, str(script), ".maws_package"], cwd=project)
    assert not (package / ".maws_package").exists()
    assert (package / "fns.py").exists()


def test_maws_start_writes_a_gitignore_that_covers_secrets(workspace_tmp_path, monkeypatch):
    from maws import operations

    monkeypatch.setattr(operations._shutil, "which", lambda name: None)
    operations.start(project="p", region="us-east-1", bot="telegram", project_dir=workspace_tmp_path, allow_windows=True)

    gitignore = (workspace_tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".env.*", "history/", "files/", ".maws_package/"):
        assert entry in gitignore
    assert not (workspace_tmp_path / ".samignore").exists()
