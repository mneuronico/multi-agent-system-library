"""Run maws/resources/bootstrap.sh (and pull_history.sh) end to end with stubbed aws, sam, curl and pip.

Nothing here touches the network: aws/sam/curl and `python3 -m pip` are replaced by stubs on PATH
(checked before every run). Needs bash and jq, so it is skipped where jq is missing (e.g. Git Bash
on Windows) and runs on Linux/WSL/CI. This module deliberately imports nothing from mas/maws.

The aws stub simulates: `ssm put-parameter` returning a version (MAWS_TEST_SSM_VERSION), the legacy
log group lookup (MAWS_TEST_LEGACY_LOG_GROUP), and `s3api head-bucket` for buckets owned by another
account (MAWS_TEST_FOREIGN_BUCKETS, 403) or missing until `s3 mb` creates them
(MAWS_TEST_MISSING_BUCKETS, 404).
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
arg_after() { local want="$1"; shift; while [[ $# -gt 0 ]]; do [[ "$1" == "$want" ]] && { printf '%s' "$2"; return; }; shift; done; }
case "$1 $2" in
  "sts get-caller-identity")
    printf '%s' "${AWS_PROFILE-<unset>}" > "${MAWS_TEST_LOG}.profile"
    echo "123456789012" ;;
  "cloudformation describe-stacks") echo "${MAWS_TEST_API_URL}" ;;
  "cloudformation describe-stack-resource")
    if [[ -n "${MAWS_TEST_LEGACY_LOG_GROUP:-}" ]]; then echo "${MAWS_TEST_LEGACY_LOG_GROUP}"
    else echo "An error occurred (ValidationError): Resource does not exist" >&2; exit 254; fi ;;
  "ssm put-parameter") echo "${MAWS_TEST_SSM_VERSION-7}" ;;
  "s3api head-bucket")
    bucket="$(arg_after --bucket "$@")"
    [[ "$(arg_after --expected-bucket-owner "$@")" == "123456789012" ]] || { echo "missing owner check" >&2; exit 99; }
    if [[ " ${MAWS_TEST_FOREIGN_BUCKETS:-} " == *" ${bucket} "* ]]; then
      echo "An error occurred (403) when calling the HeadBucket operation: Forbidden" >&2; exit 254
    fi
    if [[ " ${MAWS_TEST_MISSING_BUCKETS:-} " == *" ${bucket} "* ]] && ! grep -qx "${bucket}" "${MAWS_TEST_LOG}.created" 2>/dev/null; then
      echo "An error occurred (404) when calling the HeadBucket operation: Not Found" >&2; exit 254
    fi ;;
  "s3 mb") printf '%s\n' "${3#s3://}" >> "${MAWS_TEST_LOG}.created" ;;
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
        env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
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
        return (self.project / ".maws_build" / "template.yaml").read_text(encoding="utf-8")

    def package_file(self, name):
        return (self.project / ".maws_package" / name).read_text(encoding="utf-8")

    def command_index(self, prefix):
        return next(i for i, line in enumerate(self.commands()) if line.startswith(prefix))

    def commands_starting(self, prefix):
        return [line for line in self.commands() if line.startswith(prefix)]


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
    assert not (bootstrap.project / ".maws_build" / "template.yaml").exists()
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
    assert (
        bootstrap.command_index("sam build")
        < bootstrap.command_index("aws ssm put-parameter")
        < bootstrap.command_index("sam deploy")
        < bootstrap.command_index("aws cloudformation describe-stacks")
    )
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


# -------------------------
# Medium-priority fixes
# -------------------------
TG_ENV = f"TELEGRAM_TOKEN=123:tok\nTELEGRAM_WEBHOOK_SECRET_TOKEN={SECRET}\n"
WA_ENV = "WHATSAPP_TOKEN=t\nWHATSAPP_VERIFY_TOKEN=v\nWHATSAPP_APP_SECRET=app\n"


def test_explicit_false_booleans_are_honoured(bootstrap):
    params = _telegram(
        sync_tokens_s3=False,
        runtime={"ensure_delivery": False},
        infra={"tracing": False},
        failure_handling={"async_failure_queue": False},
    )

    code, out = bootstrap.run(params, TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert 'SYNC_TOKENS_S3: "false"' in template
    assert 'ENSURE_DELIVERY: "false"' in template
    assert "Tracing: PassThrough" in template
    assert "TracingEnabled: false" in template
    assert "AsyncFailureQueue" not in template
    assert "EventInvokeConfig" not in template


def test_boolean_defaults_when_absent(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert 'SYNC_TOKENS_S3: "true"' in template
    assert 'ENSURE_DELIVERY: "true"' in template
    assert "Tracing: Active" in template
    assert 'VERBOSE: "false"' in template


def test_bootstrap_has_no_jq_default_operator_on_booleans():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    for key in ("sync_tokens_s3", "ensure_delivery", "tracing", "verbose", "persist_files_s3",
                "capture_failed_events", "worker_dlq", "enabled", "s3_versioning"):
        assert f".{key} // true" not in text and f".{key} // false" not in text, key


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (_telegram(verbose="maybe"), "verbose in params.json must be true or false"),
        (_telegram(infra={"log_retention_days": 10}), "infra.log_retention_days"),
        (_telegram(busy_policy="queue"), "busy_policy"),
        ({"region": "us-east-1", "bot": "telegram"}, 'has no "project"'),
        (_telegram(failure_handling={"max_receive_count": 0}), "max_receive_count"),
        (_telegram(runtime={"dedup_retention_days": "a week"}), "runtime.dedup_retention_days"),
        (_telegram(runtime={"dedup_retention_days": -1}), "runtime.dedup_retention_days"),
    ],
)
def test_invalid_params_fail_before_any_aws_call(bootstrap, params, message):
    code, out = bootstrap.run(params, TG_ENV)

    _assert_failed_before_aws(bootstrap, code, out)
    assert message in out


@pytest.mark.parametrize(
    ("params", "statement"),
    [
        (_telegram(persist_files_s3=True), 'Resource: "arn:aws:s3:::demo-bot-history-bucket/files/*"'),
        (_telegram(persist_files_s3=True, files_s3_prefix="/media/"), 'Resource: "arn:aws:s3:::demo-bot-history-bucket/media/*"'),
        (_telegram(), None),
    ],
)
def test_delete_permission_is_limited_to_persisted_user_files(bootstrap, params, statement):
    code, out = bootstrap.run(params, TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    if statement is None:
        assert "s3:DeleteObject" not in template
    else:
        block = template[template.index('Action: [ "s3:DeleteObject" ]'):]
        assert block.splitlines()[1].strip() == statement


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ({}, None),  # the runtime default (8 days) applies
        ({"dedup_retention_days": 14}, 'MAWS_DEDUP_RETENTION_DAYS: "14"'),
        ({"dedup_retention_days": 0}, 'MAWS_DEDUP_RETENTION_DAYS: "0"'),
    ],
)
def test_dedup_retention_days_reaches_the_function(bootstrap, runtime, expected):
    code, out = bootstrap.run(_telegram(runtime=runtime), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    if expected is None:
        assert "MAWS_DEDUP_RETENTION_DAYS" not in template
    else:
        assert expected in template


def test_self_invoke_permission_is_scoped_to_this_function(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert 'Resource: "*"' not in template
    policy = template[template.index("DemoBotSelfInvokePolicy:"):]
    assert "Type: AWS::IAM::Policy" in policy
    assert "- !Ref DemoBotFunctionRole" in policy
    assert "- !GetAtt DemoBotFunction.Arn" in policy
    assert '- !Sub "${DemoBotFunction.Arn}:*"' in policy
    function = template[template.index("  DemoBotFunction:"):template.index("DemoBotSelfInvokePolicy:")]
    assert "lambda:InvokeFunction" not in function  # in the role's own policies it would be a cycle


def test_fifo_has_no_self_invoke_permission_and_a_dlq_by_default(bootstrap):
    code, out = bootstrap.run(_telegram(busy_policy="fifo"), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert "lambda:InvokeFunction" not in template
    assert "DemoBotWorkerDlq:" in template
    assert "deadLetterTargetArn: !GetAtt DemoBotWorkerDlq.Arn" in template
    assert "maxReceiveCount: 3" in template
    assert "WorkerDeadLetterQueueUrl" in template
    assert "AsyncFailureQueue" not in template  # nothing is invoked asynchronously in FIFO mode


def test_fifo_dlq_opt_out_and_max_receive_count(bootstrap):
    code, out = bootstrap.run(_telegram(busy_policy="fifo", failure_handling={"worker_dlq": False}), TG_ENV)
    assert code == 0, out
    assert "WorkerDlq" not in bootstrap.template()
    assert "failure_handling.worker_dlq=false" in out

    code, out = bootstrap.run(_telegram(busy_policy="fifo", failure_handling={"max_receive_count": 5}), TG_ENV)
    assert code == 0, out
    assert "maxReceiveCount: 5" in bootstrap.template()


def test_drop_mode_async_invocations_have_an_on_failure_queue(bootstrap):
    code, out = bootstrap.run(_whatsapp(), WA_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert "DemoBotAsyncFailureQueue:" in template
    assert 'QueueName: !Sub "${AWS::StackName}-async-failures"' in template
    assert "MaximumRetryAttempts: 2" in template
    assert "Destination: !GetAtt DemoBotAsyncFailureQueue.Arn" in template
    send = template.index("Resource: !GetAtt DemoBotAsyncFailureQueue.Arn")
    assert template.rfind("sqs:SendMessage", 0, send) > template.index("  DemoBotFunction:")
    assert "demo-bot-stack-async-failures" in out


def test_buckets_are_checked_with_the_expected_owner(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    heads = bootstrap.commands_starting("aws s3api head-bucket")
    assert heads == [
        "aws s3api head-bucket --bucket demo-bot-deployment-bucket --expected-bucket-owner 123456789012",
        "aws s3api head-bucket --bucket demo-bot-history-bucket --expected-bucket-owner 123456789012",
    ]
    assert not bootstrap.commands_starting("aws s3 mb")


def test_missing_deployment_bucket_is_created_and_verified(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV, MAWS_TEST_MISSING_BUCKETS="demo-bot-deployment-bucket demo-bot-history-bucket")

    assert code == 0, out
    assert bootstrap.commands_starting("aws s3 mb") == ["aws s3 mb s3://demo-bot-deployment-bucket --region us-east-1"]
    assert len(bootstrap.commands_starting("aws s3api head-bucket --bucket demo-bot-deployment-bucket")) == 2
    assert "does not exist yet; the stack creates it" in out


@pytest.mark.parametrize("bucket", ["demo-bot-deployment-bucket", "demo-bot-history-bucket"])
def test_bucket_owned_by_another_account_stops_the_deploy(bootstrap, bucket):
    code, out = bootstrap.run(_telegram(), TG_ENV, MAWS_TEST_FOREIGN_BUCKETS=bucket)

    assert code != 0
    assert f"{bucket} exists but is not owned by account 123456789012" in out
    assert not bootstrap.commands_starting("aws s3 mb")
    assert not bootstrap.commands_starting("sam build")
    assert not bootstrap.commands_starting("aws ssm")


def test_user_files_in_the_project_are_never_overwritten(bootstrap):
    custom = {
        "requirements.txt": "requests\nboto3\nmas[maws,telegram,whatsapp,env] @ git+https://example.com/fork.git\npandas\n",
        "lambda_function.py": "from maws import build_lambda_handler\nlambda_handler = build_lambda_handler('telegram')  # mine\n",
        "template.yaml": "# my own template\n",
        "samconfig.toml": "# my own samconfig\n",
    }
    for name, content in custom.items():
        (bootstrap.project / name).write_text(content, encoding="utf-8")

    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    for name, content in custom.items():
        assert (bootstrap.project / name).read_text(encoding="utf-8") == content, name
    assert bootstrap.package_file("requirements.txt") == custom["requirements.txt"]
    assert bootstrap.package_file("lambda_function.py") == custom["lambda_function.py"]
    assert "Using your lambda_function.py" in out
    assert "Using your requirements.txt" in out
    assert "are not used" in out
    assert "# my own" not in bootstrap.template()
    assert not (bootstrap.project / ".maws_package" / "template.yaml").exists()
    assert not (bootstrap.project / ".maws_package" / "samconfig.toml").exists()
    [build] = bootstrap.commands_starting("sam build")
    assert build == "sam build --template-file .maws_build/template.yaml --base-dir . --build-dir .aws-sam/build"


def test_new_project_gets_generated_files_in_the_package_only(bootstrap):
    params = _telegram(requirements_ref="v0.3.4", extra_requirements=["pandas"])

    code, out = bootstrap.run(params, TG_ENV)

    assert code == 0, out
    for name in ("requirements.txt", "lambda_function.py", "template.yaml", "samconfig.toml"):
        assert not (bootstrap.project / name).exists(), name
    requirements = bootstrap.package_file("requirements.txt").splitlines()
    assert requirements[0].startswith("# Generated by maws update")
    assert "mas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git@v0.3.4" in requirements
    assert "pandas" in requirements
    assert "build_lambda_handler(BOT_TYPE)" in bootstrap.package_file("lambda_function.py")
    assert "requirements_ref is not set" not in out


def test_files_written_by_older_maws_are_replaced_in_the_package_but_left_in_place(bootstrap):
    legacy_handler = (
        'import os\nfrom maws import build_lambda_handler\n\n'
        'BOT_TYPE = os.environ.get("BOT_TYPE", "whatsapp")\nlambda_handler = build_lambda_handler(BOT_TYPE)\n'
    )
    legacy_requirements = (
        "boto3\nmas[maws,telegram,whatsapp,env] @ git+https://github.com/mneuronico/multi-agent-system-library.git\nrequests\n"
    )
    (bootstrap.project / "lambda_function.py").write_text(legacy_handler, encoding="utf-8")
    (bootstrap.project / "requirements.txt").write_text(legacy_requirements, encoding="utf-8")

    code, out = bootstrap.run(_telegram(requirements_ref="abc1234"), TG_ENV)

    assert code == 0, out
    assert (bootstrap.project / "requirements.txt").read_text(encoding="utf-8") == legacy_requirements
    assert (bootstrap.project / "lambda_function.py").read_text(encoding="utf-8") == legacy_handler
    assert "@abc1234" in bootstrap.package_file("requirements.txt")  # params.json still applies
    assert bootstrap.package_file("lambda_function.py").startswith("# Generated by maws update")


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("./vendor/mas-0.4.0-py3-none-any.whl", "./vendor/mas-0.4.0-py3-none-any.whl[maws,telegram,whatsapp,env]"),
        ("../multi-agent-system-library", "../multi-agent-system-library[maws,telegram,whatsapp,env]"),
        ("mas-0.4.0.tar.gz", "mas-0.4.0.tar.gz[maws,telegram,whatsapp,env]"),
        ("./vendor/mas-0.4.0-py3-none-any.whl[telegram]", "./vendor/mas-0.4.0-py3-none-any.whl[telegram]"),
        ("mas[telegram] @ file:///opt/mas", "mas[telegram] @ file:///opt/mas"),
    ],
)
def test_local_package_paths_get_the_lambda_extras(bootstrap, ref, expected):
    code, out = bootstrap.run(_telegram(requirements_source="local", requirements_ref=ref), TG_ENV)

    assert code == 0, out
    assert expected in bootstrap.package_file("requirements.txt").splitlines()


def test_requirements_written_by_older_maws_for_a_local_wheel_are_still_generated(bootstrap):
    wheel = "./vendor/mas-0.4.0-py3-none-any.whl"
    (bootstrap.project / "requirements.txt").write_text(f"{wheel}\nboto3\nrequests\n", encoding="utf-8")

    code, out = bootstrap.run(_telegram(requirements_source="local", requirements_ref=wheel), TG_ENV)

    assert code == 0, out
    assert "Using your requirements.txt" not in out
    assert f"{wheel}[maws,telegram,whatsapp,env]" in bootstrap.package_file("requirements.txt").splitlines()


def test_env_prod_is_read_not_sourced(bootstrap):
    marker = bootstrap.project / "pwned"
    env_prod = (
        TG_ENV
        + "WEIRD=a b; touch pwned\n"
        + "SUBST=$(touch pwned)\n"
        + "QUOTE=it's\n"
        + "AWS_PROFILE=\"deploy-profile\"\n"
        + "MAWS_HISTORY_WRITE_SECRET=x\n"
    )

    code, out = bootstrap.run(_telegram(history_append={"enabled": True}), env_prod)

    assert code == 0, out
    assert not marker.exists()
    assert (bootstrap.log.parent / "commands.log.profile").read_text(encoding="utf-8") == "deploy-profile"
    assert "Using AWS_PROFILE from .env.prod" in out
    assert "MAWS_HISTORY_WRITE_SECRET; the endpoint will answer 503" not in out


def test_history_append_secret_warning_uses_the_env_reader(bootstrap):
    code, out = bootstrap.run(_telegram(history_append={"enabled": True}), TG_ENV)

    assert code == 0, out
    assert "no MAWS_HISTORY_WRITE_SECRET; the endpoint will answer 503" in out
    assert (bootstrap.log.parent / "commands.log.profile").read_text(encoding="utf-8") == "<unset>"


def test_function_reads_the_ssm_version_it_was_deployed_with(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV, MAWS_TEST_SSM_VERSION="42")

    assert code == 0, out
    template = bootstrap.template()
    assert "MAWS_ENV_PARAMETER_VERSION: !Ref EnvParamVersion" in template
    [ssm] = bootstrap.commands_starting("aws ssm put-parameter")
    assert "--query Version --output text" in ssm
    [deploy] = bootstrap.commands_starting("sam deploy")
    assert "--parameter-overrides EnvParamVersion=42" in deploy
    assert "--no-fail-on-empty-changeset" in deploy
    assert "--stack-name demo-bot-stack --s3-bucket demo-bot-deployment-bucket --s3-prefix demo-bot-stack" in deploy
    assert bootstrap.command_index("aws ssm put-parameter") < bootstrap.command_index("sam deploy")


def test_missing_ssm_version_stops_before_deploying(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV, MAWS_TEST_SSM_VERSION="")

    assert code != 0
    assert "Could not read the new version of SSM parameter" in out
    assert not bootstrap.commands_starting("sam deploy")


def test_requirement_source_warnings(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    assert "requirements_ref is not set" in out


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("pypi", "the PyPI package named 'mas' is a different"),
        ("gti", 'must be "git" or "local" (got: gti)'),
    ],
)
def test_requirement_source_other_than_git_or_local_stops_before_touching_aws(bootstrap, source, error):
    code, out = bootstrap.run(_telegram(requirements_source=source), TG_ENV)

    assert code != 0
    assert error in out
    assert not bootstrap.commands_starting("aws")
    assert not bootstrap.commands_starting("sam")


def test_default_log_group_has_retention_and_receives_the_logs(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    group = template[template.index("  DemoBotLogs:"):]
    assert 'LogGroupName: !Sub "/aws/lambda/${AWS::StackName}-maws"' in group
    assert "RetentionInDays: 30" in group
    assert "LogGroup: !Ref DemoBotLogs" in template
    assert "LogFormat: JSON" in template
    assert "/aws/lambda/demo-bot-stack-maws (retention: 30 days)" in out


def test_stack_with_the_legacy_log_group_keeps_it(bootstrap):
    code, out = bootstrap.run(
        _telegram(infra={"log_retention_days": 14}),
        TG_ENV,
        MAWS_TEST_LEGACY_LOG_GROUP="/aws/lambda/demo-bot-stack-DemoBotFunction-AbC",
    )

    assert code == 0, out
    template = bootstrap.template()
    assert "  DemoBotFunctionLogGroup:" in template
    assert 'LogGroupName: !Sub "/aws/lambda/${DemoBotFunction}"' in template
    assert "RetentionInDays: 14" in template
    assert "LogGroup: !Ref" not in template  # the default destination is that group
    assert "DemoBotLogs" not in template


def test_log_retention_zero_keeps_logs_forever(bootstrap):
    code, out = bootstrap.run(_telegram(infra={"log_retention_days": 0}), TG_ENV)

    assert code == 0, out
    assert "RetentionInDays" not in bootstrap.template()


def test_lifecycle_only_expires_user_data_and_noncurrent_versions(bootstrap):
    params = _telegram(infra={"s3_lifecycle_days": 90, "s3_versioning": True}, files_s3_prefix="media")

    code, out = bootstrap.run(params, TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    rules = template[template.index("LifecycleConfiguration:"):template.index("  DemoBotLogs:")]
    assert "Prefix: history/" in rules
    assert "Prefix: media/" in rules
    assert "Prefix: failed-events/" in rules
    assert "secrets" not in rules
    assert rules.count("ExpirationInDays: 90") == 3
    assert "NoncurrentDays: 30" in rules
    assert "ExpiredObjectDeleteMarker: true" in rules
    assert "ExpireMawsObjects" not in template


def test_versioning_without_lifecycle_still_expires_noncurrent_versions(bootstrap):
    code, out = bootstrap.run(_telegram(infra={"s3_versioning": True, "s3_noncurrent_version_days": 7}), TG_ENV)

    assert code == 0, out
    template = bootstrap.template()
    assert "NoncurrentDays: 7" in template
    assert "ExpirationInDays" not in template


def test_no_lifecycle_by_default(bootstrap):
    code, out = bootstrap.run(_telegram(), TG_ENV)

    assert code == 0, out
    assert "LifecycleConfiguration" not in bootstrap.template()


# -------------------------
# pull_history.sh
# -------------------------
PULL_HISTORY = ROOT / "maws" / "resources" / "pull_history.sh"


def _pull(bootstrap, **extra):
    (bootstrap.project / "params.json").write_text(json.dumps(_telegram()), encoding="utf-8")
    script = Path(os.path.relpath(PULL_HISTORY, bootstrap.project)).as_posix()
    proc = subprocess.run(
        [bootstrap.bash, script, "--config", "params.json"],
        cwd=bootstrap.project, env=bootstrap.env(**extra), capture_output=True, timeout=60,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")


def test_pull_history_checks_the_bucket_owner(bootstrap):
    code, out = _pull(bootstrap)

    assert code == 0, out
    assert bootstrap.commands_starting("aws s3api head-bucket") == [
        "aws s3api head-bucket --bucket demo-bot-history-bucket --expected-bucket-owner 123456789012"
    ]
    assert bootstrap.commands_starting("aws s3 sync")


def test_pull_history_refuses_a_bucket_of_another_account(bootstrap):
    code, out = _pull(bootstrap, MAWS_TEST_FOREIGN_BUCKETS="demo-bot-history-bucket")

    assert code != 0
    assert "not owned by account 123456789012" in out
    assert not bootstrap.commands_starting("aws s3 sync")
