"""Run maws/resources/bootstrap.sh end to end with stubbed aws, sam, curl and pip.

Nothing here touches the network: aws/sam/curl and `python3 -m pip` are replaced by stubs on PATH
(checked before every run). Needs bash and jq, so it is skipped where jq is missing (e.g. Git Bash
on Windows) and runs on Linux/WSL/CI. This module deliberately imports nothing from mas/maws.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "maws" / "resources" / "bootstrap.sh"
API_URL = "https://abc123.execute-api.us-east-1.amazonaws.com/Prod"
SECRET = "Abc_123-xyz" * 4
TELEGRAM_OK = '{"ok":true,"result":true,"description":"Webhook was set"}'

_STUBS = {
    "aws": r"""#!/usr/bin/env bash
printf 'aws %s\n' "$*" >> "${MAWS_TEST_LOG}"
case "$1 $2" in
  "sts get-caller-identity") echo "123456789012" ;;
  "cloudformation describe-stacks") echo "${MAWS_TEST_API_URL}" ;;
esac
exit 0
""",
    "sam": r"""#!/usr/bin/env bash
printf 'sam %s\n' "$*" >> "${MAWS_TEST_LOG}"
exit 0
""",
    "curl": r"""#!/usr/bin/env bash
printf 'curl\n' >> "${MAWS_TEST_LOG}"
{ echo "CALL"; for a in "$@"; do printf 'ARG:%s\n' "$a"; done; } >> "${MAWS_TEST_CURL_LOG}"
printf '%s' "${MAWS_TEST_CURL_RESPONSE}"
exit "${MAWS_TEST_CURL_EXIT:-0}"
""",
    "python3": r"""#!/usr/bin/env bash
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  printf 'pip %s\n' "$*" >> "${MAWS_TEST_LOG}"
  exit "${MAWS_TEST_PIP_EXIT:-0}"
fi
exec "${MAWS_TEST_REAL_PYTHON}" "$@"
""",
}


def _bash_with_jq():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "-c", "command -v jq"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("jq is not available to bash")
    return bash


class _Bootstrap:
    def __init__(self, root: Path, bash: str):
        self.bash = bash
        self.root = root
        self.stubs = root / "maws_stub_bin"
        self.stubs.mkdir()
        for name, body in _STUBS.items():
            path = self.stubs / name
            path.write_bytes(body.encode("utf-8"))
            path.chmod(0o755)
        self.log = root / "commands.log"
        self.curl_log = root / "curl.log"
        self.project = root / "bot"
        self.project.mkdir()
        (self.project / "config.json").write_text("{}", encoding="utf-8")
        (self.project / "fns.py").write_text("# fns\n", encoding="utf-8")

    def env(self, **extra):
        env = dict(os.environ)
        env["PATH"] = str(self.stubs) + os.pathsep + env.get("PATH", "")
        env.update(
            MAWS_TEST_LOG=self.log.as_posix(),
            MAWS_TEST_CURL_LOG=self.curl_log.as_posix(),
            MAWS_TEST_API_URL=API_URL,
            MAWS_TEST_REAL_PYTHON=Path(sys.executable).as_posix(),
            MAWS_TEST_CURL_RESPONSE=TELEGRAM_OK,
        )
        env.update(extra)
        return env

    def run(self, params: dict, env_prod: str, **extra):
        (self.project / "params.json").write_text(json.dumps(params), encoding="utf-8")
        (self.project / ".env.prod").write_bytes(env_prod.encode("utf-8"))
        env = self.env(**extra)

        # Never run the script unless every external command resolves to a stub.
        resolved = subprocess.run(
            [self.bash, "-c", "for c in aws sam curl python3; do command -v \"$c\"; done"],
            cwd=self.project, env=env, capture_output=True, text=True,
        ).stdout.split()
        if len(resolved) != 4 or not all("maws_stub_bin" in path for path in resolved):
            pytest.fail(f"stubs do not shadow the real commands: {resolved}")

        script = Path(os.path.relpath(BOOTSTRAP, self.project)).as_posix()
        proc = subprocess.run(
            [self.bash, script, "--config", "params.json"],
            cwd=self.project, env=env, capture_output=True, timeout=300,
        )
        return proc.returncode, proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")

    def commands(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def curl_calls(self):
        if not self.curl_log.exists():
            return []
        calls = []
        for line in self.curl_log.read_text(encoding="utf-8").splitlines():
            if line == "CALL":
                calls.append([])
            elif line.startswith("ARG:"):
                calls[-1].append(line[4:])
        return calls

    def template(self):
        return (self.project / "template.yaml").read_text(encoding="utf-8")


@pytest.fixture
def bootstrap(workspace_tmp_path):
    return _Bootstrap(workspace_tmp_path, _bash_with_jq())


def _telegram(**extra):
    return {"project": "demo-bot", "region": "us-east-1", "bot": "telegram", **extra}


def _whatsapp(**extra):
    return {"project": "demo-bot", "region": "us-east-1", "bot": "whatsapp", **extra}


# -------------------------
# Validation fails before anything is uploaded or deployed
# -------------------------
def _assert_failed_before_aws(bootstrap, code, out):
    assert code != 0
    assert bootstrap.commands() == []  # no aws/sam/curl/pip at all
    assert not (bootstrap.project / "template.yaml").exists()


def test_telegram_without_secret_fails_before_any_aws_call(bootstrap):
    code, out = bootstrap.run(_telegram(), "TELEGRAM_TOKEN=123:tok\n")

    _assert_failed_before_aws(bootstrap, code, out)
    assert ".env.prod has no TELEGRAM_WEBHOOK_SECRET_TOKEN" in out
    assert "allow_unauthenticated" in out
    assert "secrets.token_urlsafe(32)" in out


@pytest.mark.parametrize("secret", ['"quoted-secret"', "has space", "semi;colon", "x" * 257])
def test_telegram_secret_with_invalid_characters_fails(bootstrap, secret):
    code, out = bootstrap.run(_telegram(), f"TELEGRAM_TOKEN=123:tok\nTELEGRAM_WEBHOOK_SECRET_TOKEN={secret}\n")

    _assert_failed_before_aws(bootstrap, code, out)
    assert "is not a valid Telegram secret token" in out


def test_telegram_without_bot_token_fails(bootstrap):
    code, out = bootstrap.run(_telegram(), f"TELEGRAM_WEBHOOK_SECRET_TOKEN={SECRET}\n")

    _assert_failed_before_aws(bootstrap, code, out)
    assert ".env.prod has no TELEGRAM_TOKEN" in out


def test_whatsapp_without_app_secret_fails(bootstrap):
    code, out = bootstrap.run(_whatsapp(), "WHATSAPP_TOKEN=t\nWHATSAPP_VERIFY_TOKEN=v\n# WHATSAPP_APP_SECRET=x\n")

    _assert_failed_before_aws(bootstrap, code, out)
    assert ".env.prod has no WHATSAPP_APP_SECRET" in out
    assert "App settings > Basic" in out


def test_whatsapp_verify_false_is_not_an_opt_out(bootstrap):
    params = _whatsapp(webhook_security={"whatsapp_verify_signature": False, "allow_unauthenticated": False})

    code, out = bootstrap.run(params, "WHATSAPP_TOKEN=t\nWHATSAPP_APP_SECRET=app\n")

    _assert_failed_before_aws(bootstrap, code, out)
    assert "whatsapp_verify_signature=false" in out


# -------------------------
# Successful deploys
# -------------------------
def test_telegram_deploy_with_crlf_env_sets_a_checked_webhook(bootstrap):
    env_prod = (
        "# Windows editor\r\n"
        "\r\n"
        "TELEGRAM_TOKEN=123:tok\r\n"
        f"TELEGRAM_WEBHOOK_SECRET_TOKEN={SECRET}\r\n"
    )

    code, out = bootstrap.run(_telegram(), env_prod, MAWS_TEST_PIP_EXIT="1")  # PEP 668 pip

    assert code == 0, out
    assert "[WARN] Could not upgrade pip/setuptools/wheel" in out
    assert f"[OK] Telegram webhook set to: {API_URL}/webhook (secret token enabled)" in out
    assert [call for call in bootstrap.curl_calls()] == [[
        "-sS", "--max-time", "30",
        "https://api.telegram.org/bot123:tok/setWebhook",
        "--data-urlencode", f"url={API_URL}/webhook",
        "--data-urlencode", f"secret_token={SECRET}",
    ]]
    commands = bootstrap.commands()
    names = [line.split(" ", 2)[0] + " " + line.split(" ", 2)[1] for line in commands if " " in line]
    assert names.index("aws ssm") < names.index("sam deploy") < names.index("aws cloudformation")
    assert commands[-1] == "curl"

    template = bootstrap.template()
    assert "Runtime: python3.12" in template
    assert 'LOCK_TTL_SECONDS: "135"' in template
    assert 'MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "false"' in template
    assert not (bootstrap.project / ".bootstrap_state.json").exists()

    # setWebhook is not cached: a second deploy calls it again.
    code, out = bootstrap.run(_telegram(), env_prod)
    assert code == 0, out
    assert len(bootstrap.curl_calls()) == 2


def test_telegram_custom_secret_env_key(bootstrap):
    params = _telegram(webhook_security={"telegram_secret_token_env_key": "MY_TG_SECRET"})

    code, out = bootstrap.run(params, f"TELEGRAM_TOKEN=123:tok\nMY_TG_SECRET={SECRET}\n")

    assert code == 0, out
    assert f"secret_token={SECRET}" in bootstrap.curl_calls()[0]
    assert 'TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY: "MY_TG_SECRET"' in bootstrap.template()


@pytest.mark.parametrize(
    ("curl_extra", "message"),
    [
        ({"MAWS_TEST_CURL_RESPONSE": '{"ok":false,"error_code":401,"description":"Unauthorized"}'},
         "Telegram rejected setWebhook: Unauthorized"),
        ({"MAWS_TEST_CURL_RESPONSE": "<html>Bad Gateway</html>"},
         "Telegram rejected setWebhook: unexpected"),
        ({"MAWS_TEST_CURL_RESPONSE": "", "MAWS_TEST_CURL_EXIT": "6"},
         "Could not reach the Telegram API"),
    ],
)
def test_telegram_set_webhook_failures_exit_non_zero(bootstrap, curl_extra, message):
    code, out = bootstrap.run(_telegram(), f"TELEGRAM_TOKEN=123:tok\nTELEGRAM_WEBHOOK_SECRET_TOKEN={SECRET}\n", **curl_extra)

    assert code != 0
    assert message in out
    assert "[OK] Telegram webhook set" not in out
    assert "[OK] Done." not in out


def test_telegram_explicit_opt_out_deploys_with_a_loud_warning(bootstrap):
    code, out = bootstrap.run(_telegram(webhook_security={"allow_unauthenticated": True}), "TELEGRAM_TOKEN=123:tok\n")

    assert code == 0, out
    assert "DEPLOYING AN UNAUTHENTICATED WEBHOOK" in out
    assert "(UNAUTHENTICATED: no secret token)" in out
    assert not [arg for arg in bootstrap.curl_calls()[0] if arg.startswith("secret_token=")]
    assert 'MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "true"' in bootstrap.template()


def test_whatsapp_deploy_verifies_signatures_by_default(bootstrap):
    code, out = bootstrap.run(_whatsapp(), "WHATSAPP_TOKEN=t\nWHATSAPP_VERIFY_TOKEN=v\nWHATSAPP_APP_SECRET=app\r\n")

    assert code == 0, out
    template = bootstrap.template()
    assert 'WHATSAPP_VERIFY_SIGNATURE: "true"' in template
    assert 'MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "false"' in template
    assert bootstrap.curl_calls() == []


def test_whatsapp_explicit_opt_out_deploys_without_verification(bootstrap):
    code, out = bootstrap.run(_whatsapp(webhook_security={"allow_unauthenticated": True}), "WHATSAPP_TOKEN=t\n")

    assert code == 0, out
    assert "DEPLOYING AN UNAUTHENTICATED WEBHOOK" in out
    template = bootstrap.template()
    assert 'WHATSAPP_VERIFY_SIGNATURE: "false"' in template
    assert 'MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS: "true"' in template


def test_fifo_lock_ttl_expires_before_sqs_redelivery(bootstrap):
    params = _telegram(busy_policy="fifo", lambda_timeout=300, infra={"runtime": "python3.11"})

    code, out = bootstrap.run(params, f"TELEGRAM_TOKEN=123:tok\nTELEGRAM_WEBHOOK_SECRET_TOKEN={SECRET}\n")

    assert code == 0, out
    template = bootstrap.template()
    assert "Timeout: 300" in template
    assert "VisibilityTimeout: 330" in template
    assert 'LOCK_TTL_SECONDS: "315"' in template
    assert "Runtime: python3.11" in template
