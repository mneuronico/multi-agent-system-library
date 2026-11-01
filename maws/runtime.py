from __future__ import annotations

import asyncio
import base64
import copy
import datetime as _dt
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shutil
import time
import traceback
import types as _types
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import boto3, botocore
except ImportError:
    boto3 = None

    class _MissingClientError(Exception):
        def __init__(self, error_response=None, operation_name=None):
            super().__init__(str(error_response or {}))
            self.response = error_response or {"Error": {"Code": "MissingBoto3"}}
            self.operation_name = operation_name

    botocore = _types.SimpleNamespace(
        exceptions=_types.SimpleNamespace(ClientError=_MissingClientError)
    )

# MAS
from mas import AgentSystemManager, TelegramBot, WhatsappBot


# Warnings go through `logging` so they carry level WARNING in Lambda's JSON logs.
logger = logging.getLogger("maws")

# CloudWatch namespace of the metrics MAWS emits as Embedded Metric Format log lines.
METRICS_NAMESPACE = "MAWS"

# busy_policy=drop: how long a worker waits for a busy user lock before dropping its update. The
# wait is also limited to BUSY_WAIT_SHARE_OF_REMAINING of the invocation's remaining time, so the
# update can still be processed once the lock is free.
DEFAULT_BUSY_WAIT_SECONDS = 30.0
BUSY_WAIT_SHARE_OF_REMAINING = 0.5
BUSY_WAIT_POLL_SECONDS = 0.5

# Processed-message records: how long a message stays marked as processed, so that a second delivery
# of it is skipped. WhatsApp retries an unacknowledged webhook for up to 7 days.
DEFAULT_DEDUP_RETENTION_DAYS = 8.0
# The records share the locks table with the user locks; their keys start with this prefix.
DEDUP_KEY_PREFIX = "maws-event#"


# -------------------------
# Utilities and configuration
# -------------------------
def _as_bool(v, default=False):
    if v is None:
        return default
    value = str(v).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v, default=None):
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _json_env(name: str, default):
    try:
        value = json.loads(os.environ.get(name, ""))
    except Exception:
        return copy.deepcopy(default)
    if isinstance(default, dict) and not isinstance(value, dict):
        return copy.deepcopy(default)
    if isinstance(default, list) and not isinstance(value, list):
        return copy.deepcopy(default)
    return value


def _safe_key_part(value: str) -> str:
    raw = str(value)
    if (
        raw
        and raw not in {".", ".."}
        and ".." not in raw
        and re.fullmatch(r"[A-Za-z0-9_.-]+", raw)
    ):
        return raw
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    return f"u_{encoded or 'empty'}"


def _require_boto3():
    if boto3 is None:
        raise ImportError("boto3 and botocore are required for maws AWS operations: pip install boto3")
    return boto3


# Values of secret-looking keys loaded from SSM; failed-event captures are scrubbed of them.
_LOADED_SECRET_VALUES: set = set()
_SECRET_ENV_KEY_RE = re.compile(
    r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|AUTH|CREDENTIAL|SIGNATURE|PRIVATE|SESSION|COOKIE)", re.I
)
_MIN_SCRUBBED_SECRET_LENGTH = 6


def _ssm_error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return type(error).__name__


def _load_env_from_ssm_if_needed():
    """
    Load a .env from SSM (SecureString) and merge it into ``os.environ``
    without overriding existing keys. Runs only once per process.

    When MAWS_ENV_PARAMETER_VERSION is set (maws deploys set it), that exact parameter version is
    read, so a function keeps the secrets it was deployed with even after a newer deploy wrote a
    new version (for example when that deploy then failed and CloudFormation rolled back).
    """
    param_name = os.environ.get("ENV_PARAMETER_NAME")
    if not param_name:
        return
    if os.environ.get("_ENV_LOADED_FROM_SSM") == "1":
        return
    ssm = _require_boto3().client("ssm")
    version = os.environ.get("MAWS_ENV_PARAMETER_VERSION", "").strip()
    name = f"{param_name}:{version}" if version else param_name
    try:
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
    except Exception as e:
        if not version or _ssm_error_code(e) != "ParameterVersionNotFound":
            raise
        print(
            f"[maws][WARN] SSM parameter {param_name} has no version {version} any more; "
            "reading the latest version instead. Redeploy with maws update."
        )
        name = param_name
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
    raw = resp["Parameter"]["Value"]
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and _SECRET_ENV_KEY_RE.search(k) and len(v) >= _MIN_SCRUBBED_SECRET_LENGTH:
            _LOADED_SECRET_VALUES.add(v)
        if k and k not in os.environ:
            os.environ[k] = v
    os.environ["_ENV_LOADED_FROM_SSM"] = "1"
    print(f"[maws] Loaded env from SSM parameter: {name}")


# -------------------------
# Redaction of failed-event captures
# -------------------------
_REDACTED = "[REDACTED]"
# Header / query / field names whose values are secrets (Telegram secret token, WhatsApp
# X-Hub-Signature-256, Authorization, cookies, API keys, X-MAWS-Signature, hub.verify_token, ...).
_SENSITIVE_NAME_RE = re.compile(
    r"(authorization|cookie|secret|signature|token|api[-_]?key|access[-_]?key|password|passwd|"
    r"credential|x-amz-security)",
    re.I,
)
# A Telegram bot token, e.g. inside an API URL in an exception message.
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"(?<![0-9])[0-9]{5,}:[A-Za-z0-9_-]{30,}")
# key=value pairs with a secret-looking key in query strings / free text.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"((?:token|secret|signature|password|api[-_]?key)[A-Za-z0-9_.-]*=)[^&\s\"']*", re.I
)


def _scrub_text(text: str, secret_values: Iterable[str]) -> str:
    for secret in sorted(secret_values, key=len, reverse=True):
        if secret and secret in text:
            text = text.replace(secret, _REDACTED)
    text = _TELEGRAM_BOT_TOKEN_RE.sub(_REDACTED, text)
    return _SECRET_ASSIGNMENT_RE.sub(lambda m: m.group(1) + _REDACTED, text)


def _redact_for_capture(value: Any, secret_values: Iterable[str]) -> Any:
    """Copy of ``value`` with secret-named fields (headers, query parameters) and known secret values removed."""
    secret_values = list(secret_values)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_NAME_RE.search(key):
                out[key] = [_REDACTED for _ in item] if isinstance(item, (list, tuple)) else _REDACTED
            else:
                out[key] = _redact_for_capture(item, secret_values)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_for_capture(item, secret_values) for item in value]
    if isinstance(value, str):
        return _scrub_text(value, secret_values)
    return value


def _event_body_bytes(event: dict) -> bytes:
    body = event.get("body", "")
    if body is None:
        body = ""
    if isinstance(body, bytes):
        raw = body
    else:
        raw = str(body).encode("utf-8")
    if event.get("isBase64Encoded"):
        return base64.b64decode(raw)
    return raw


def _event_body_text(event: dict) -> str:
    return _event_body_bytes(event).decode("utf-8")


def _header_value(headers: Optional[dict], name: str) -> str:
    if not headers:
        return ""
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return "" if value is None else str(value)
    return ""


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class HistoryAppendLockBusy(RuntimeError):
    """Raised so Lambda/SQS retries a history append instead of dropping it."""


class UserLockBusy(RuntimeError):
    """
    Raised on the SQS (busy_policy=fifo) path when another worker holds the user lock. SQS deletes
    a message when the invocation succeeds, so the invocation must fail for the message to be retried.
    """


class MawsWorkerError(RuntimeError):
    """
    Raised by a worker that cannot run a job it must not acknowledge (SQS messages, history
    appends): a misconfigured BUCKET_NAME or a failed MAS initialization. The failed invocation
    makes SQS (or Lambda async retries) try again, and finally moves the job to the DLQ.
    """


class _TokenWriteConflict(RuntimeError):
    """Another container changed a token file in S3 since this container read it."""


# A lock taken inside Lambda expires this long after the invocation's own deadline. It must stay
# below the 30 s that bootstrap.sh adds to lambda_timeout for the SQS visibility timeout, so a
# crashed worker's lock is gone before SQS makes its message visible again; the margin covers clock
# skew between Lambda hosts.
LOCK_EXPIRY_MARGIN_SECONDS = 15


# -------------------------
# Main runtime
# -------------------------
class MawsRuntime:
    def __init__(self):
        _load_env_from_ssm_if_needed()

        # Paths first because config probing can need them.
        self.CODE_ROOT = "/var/task" if "AWS_LAMBDA_FUNCTION_NAME" in os.environ else os.getcwd()
        self.TMP_DIR = os.environ.get("MAWS_TMP_DIR", "/tmp")
        os.makedirs(f"{self.TMP_DIR}/history", exist_ok=True)
        os.makedirs(f"{self.TMP_DIR}/files", exist_ok=True)

        # General config
        self.bucket_name = os.environ.get("BUCKET_NAME", "").strip()
        self.bot_type = os.environ.get("BOT_TYPE", "whatsapp").strip().lower()
        self.verbose = _as_bool(os.environ.get("VERBOSE", "false"), default=False)
        self.config_path = os.environ.get("MAWS_CONFIG_PATH", "config.json")

        # Runtime behavior. Defaults intentionally preserve the original MAWS path.
        self.busy_policy = os.environ.get("MAWS_BUSY_POLICY", os.environ.get("BUSY_POLICY", "drop")).strip().lower()
        if self.busy_policy not in {"drop", "fifo"}:
            print(f"[maws][WARN] Unknown busy policy {self.busy_policy!r}; using 'drop'.")
            self.busy_policy = "drop"
        self.queue_url = os.environ.get("MAWS_QUEUE_URL", os.environ.get("QUEUE_URL", "")).strip()
        # busy_policy=drop: upper bound for waiting on a busy user lock before dropping an update.
        self.busy_wait_seconds = _as_float(os.environ.get("MAWS_BUSY_WAIT_SECONDS"), DEFAULT_BUSY_WAIT_SECONDS)

        self.history_prefix = os.environ.get("HISTORY_S3_PREFIX", "history").strip().strip("/") or "history"
        self.shared_history_prefix = os.environ.get(
            "SHARED_HISTORY_S3_PREFIX",
            f"{self.history_prefix}/shared",
        ).strip().strip("/")
        self.history_mode = (
            os.environ.get("HISTORY_MODE", "").strip().lower()
            or self._history_mode_from_config_file()
            or "per_user"
        )
        self.history_rotation = os.environ.get("HISTORY_ROTATION", "message_count")
        self.history_max_messages = _as_int(os.environ.get("HISTORY_MAX_MESSAGES"), 1000)
        self.history_period = os.environ.get("HISTORY_PERIOD", "1w")

        # Optional S3-backed files. Off by default to preserve current /tmp-only behavior.
        self.persist_files_s3 = _as_bool(os.environ.get("PERSIST_FILES_S3"), default=False)
        self.files_prefix = os.environ.get("FILES_S3_PREFIX", "files").strip().strip("/") or "files"

        # DynamoDB locks (optional). LOCK_TTL_SECONDS is only used when no Lambda context is
        # available; inside Lambda the lock lifetime follows the invocation deadline.
        self.lock_table = os.environ.get("LOCKS_TABLE_NAME") or ""
        self.lock_ttl = _as_int(os.environ.get("LOCK_TTL_SECONDS"), 180)
        self._lock_tokens: Dict[str, str] = {}
        # Processed-message records, stored in the locks table so that a message delivered twice is
        # processed once. MAWS_DEDUP_RETENTION_DAYS=0 turns them off.
        self.dedup_retention_days = _as_float(
            os.environ.get("MAWS_DEDUP_RETENTION_DAYS"), DEFAULT_DEDUP_RETENTION_DAYS
        )
        self._claim_tokens: Dict[str, str] = {}

        # Provider webhook authentication is mandatory. MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true
        # (params.json webhook_security.allow_unauthenticated) is the explicit opt-out.
        self.allow_unauthenticated_webhooks = _as_bool(
            os.environ.get("MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS"), default=False
        )
        if self.allow_unauthenticated_webhooks:
            print(
                "[maws][security][WARN] MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true: webhook POSTs are "
                "accepted without a provider secret/signature when none is configured."
            )
        self.telegram_secret_token_env_key = os.environ.get(
            "TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY",
            "TELEGRAM_WEBHOOK_SECRET_TOKEN",
        )
        self.telegram_secret_token = (
            os.environ.get(self.telegram_secret_token_env_key, "")
            or os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
        )
        self.whatsapp_verify_signature = _as_bool(os.environ.get("WHATSAPP_VERIFY_SIGNATURE"), default=True)
        self.whatsapp_app_secret_env_key = os.environ.get(
            "WHATSAPP_APP_SECRET_ENV_KEY",
            "WHATSAPP_APP_SECRET",
        )
        self.whatsapp_app_secret = (
            os.environ.get(self.whatsapp_app_secret_env_key, "")
            or os.environ.get("WHATSAPP_APP_SECRET", "")
        )

        # External history append endpoint (/history/outgoing). Off unless explicitly enabled,
        # so bots that do not opt in keep the exact previous request and worker paths.
        self.history_append_enabled = _as_bool(os.environ.get("MAWS_HISTORY_APPEND_ENABLED"), default=False)
        self.history_write_secret = os.environ.get("MAWS_HISTORY_WRITE_SECRET", "")
        self.history_append_role = os.environ.get("MAWS_HISTORY_APPEND_ROLE", "").strip()
        self.history_append_lock_wait = _as_float(os.environ.get("MAWS_HISTORY_APPEND_LOCK_WAIT_SECONDS"), 60.0)

        # Failure capture. Off by default; FIFO mode can rely on SQS retries/DLQs.
        self.capture_failed_events = _as_bool(os.environ.get("CAPTURE_FAILED_EVENTS"), default=False)
        self.failed_events_prefix = os.environ.get("FAILED_EVENTS_S3_PREFIX", "failed-events").strip().strip("/") or "failed-events"

        # Optional runtime extension points.
        self.manager_kwargs = _json_env("MAWS_MANAGER_KWARGS_JSON", {})
        self.bot_kwargs = _json_env("MAWS_BOT_KWARGS_JSON", {})
        self.ensure_delivery = _as_bool(os.environ.get("ENSURE_DELIVERY"), default=True)
        self.delivery_timeout = _as_float(os.environ.get("DELIVERY_TIMEOUT"), 60.0)
        self.max_allowed_message_delay = _as_float(os.environ.get("MAX_ALLOWED_MESSAGE_DELAY"), None)

        # Special tokens (optional)
        self.sync_tokens_s3 = _as_bool(os.environ.get("SYNC_TOKENS_S3", "1"), default=True)
        self.tokens_prefix = os.environ.get("TOKENS_S3_PREFIX", "secrets")
        self.special_files = []
        try:
            self.special_files = json.loads(os.environ.get("SPECIAL_TOKEN_FILES_JSON", "[]"))
            if not isinstance(self.special_files, list):
                self.special_files = []
        except Exception:
            self.special_files = []

        self.token_env_map = {}
        try:
            self.token_env_map = json.loads(os.environ.get("TOKEN_ENV_MAP_JSON", "{}"))
            if not isinstance(self.token_env_map, dict):
                self.token_env_map = {}
        except Exception:
            self.token_env_map = {}
        # filename -> {"sha256", "etag", "s3"}: what this container last read from / wrote to S3.
        self._token_states: Dict[str, dict] = {}

        # AWS clients are created lazily so config-only paths and tests do not
        # contact AWS before the handler knows which path it is serving.
        self._s3 = None
        self._lambda_client = None
        self._dynamodb = None
        self._sqs = None

        # Warm state
        self.manager = None
        self.bot_instance = None
        self._loop = None

        # If files were mapped to ENV variables, publish paths in ENV
        for fname, envkey in self.token_env_map.items():
            os.environ.setdefault(envkey, f"{self.TMP_DIR}/{fname}")

    @property
    def s3(self):
        if self._s3 is None:
            self._s3 = _require_boto3().client("s3")
        return self._s3

    @s3.setter
    def s3(self, client):
        self._s3 = client

    @property
    def lambda_client(self):
        if self._lambda_client is None:
            self._lambda_client = _require_boto3().client("lambda")
        return self._lambda_client

    @lambda_client.setter
    def lambda_client(self, client):
        self._lambda_client = client

    @property
    def sqs(self):
        if self._sqs is None:
            self._sqs = _require_boto3().client("sqs")
        return self._sqs

    @sqs.setter
    def sqs(self, client):
        self._sqs = client

    @property
    def dynamodb(self):
        if not self.lock_table:
            return None
        if self._dynamodb is None:
            self._dynamodb = _require_boto3().client("dynamodb")
        return self._dynamodb

    @dynamodb.setter
    def dynamodb(self, client):
        self._dynamodb = client

    def _history_mode_from_config_file(self) -> str:
        candidates = []
        if os.path.isabs(getattr(self, "config_path", "")):
            candidates.append(Path(self.config_path))
        else:
            candidates.append(Path(getattr(self, "CODE_ROOT", os.getcwd())) / getattr(self, "config_path", "config.json"))
            candidates.append(Path.cwd() / getattr(self, "config_path", "config.json"))
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            general = data.get("general_parameters", {}) if isinstance(data, dict) else {}
            storage = general.get("history_storage", {}) if isinstance(general, dict) else {}
            if isinstance(storage, dict) and storage.get("mode"):
                return str(storage.get("mode")).strip().lower()
            if isinstance(general, dict) and general.get("history_mode"):
                return str(general.get("history_mode")).strip().lower()
        return ""

    def _missing_bucket_response(self):
        return {
            "statusCode": 500,
            "body": json.dumps("BUCKET_NAME is required for MAWS worker history persistence."),
        }

    def _log(self, event_type: str, **fields):
        payload = {"maws_event_type": event_type, **fields}
        print("[maws] " + json.dumps(payload, sort_keys=True, default=str))

    def _emit_metric(self, name: str, value: float = 1, unit: str = "Count", **properties):
        """
        Emit a CloudWatch metric as an Embedded Metric Format (EMF) log line: CloudWatch Logs turns
        it into metric ``MAWS/<name>`` (dimension FunctionName) without any extra IAM permission.
        Extra ``properties`` are searchable in the log line but are not dimensions.
        """
        payload = {
            **{k: v for k, v in properties.items() if k not in ("_aws", "FunctionName", name)},
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": METRICS_NAMESPACE,
                    "Dimensions": [["FunctionName"]],
                    "Metrics": [{"Name": name, "Unit": unit}],
                }],
            },
            "FunctionName": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "local"),
            name: value,
        }
        # One JSON object per line, nothing else on it: that is what CloudWatch parses as EMF.
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)

    # -------------------------
    # Helpers event loop / S3 / paths
    # -------------------------
    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _uses_shared_history(self) -> bool:
        manager = getattr(self, "manager", None)
        if manager is not None and hasattr(manager, "_history_mode_is_shared"):
            try:
                return bool(manager._history_mode_is_shared())
            except Exception:
                pass
        return getattr(self, "history_mode", "per_user") == "shared"

    def s3_sqlite_key(self, chat_id: str) -> str:
        return f"{getattr(self, 'history_prefix', 'history')}/{_safe_key_part(chat_id)}.sqlite"

    def _s3_key_for(self, filename: str) -> str:
        prefix = (getattr(self, "tokens_prefix", "secrets") or "").strip().strip("/")
        return f"{prefix}/{filename}" if prefix else filename

    def _download_s3_if_exists(self, bucket: str, key: str, local_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(bucket, key, local_path)
        print(f"[maws] s3://{bucket}/{key} -> {local_path}")
        return True

    def _copy_code_to_tmp_if_exists(self, filename: str, dst_path: str) -> bool:
        src_path = os.path.join(getattr(self, "CODE_ROOT", "/var/task"), filename)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            print(f"[maws] {src_path} -> {dst_path}")
            return True
        return False

    # -------------------------
    # Token files (special_token_files)
    # -------------------------
    # Each container remembers the hash and S3 ETag of every token file it read. After a job a token
    # is uploaded only when its content changed, with a conditional write (If-Match the ETag it read,
    # or If-None-Match when S3 had no copy). If another container changed the token in between, the
    # write is refused: this container keeps the S3 version and re-reads it, instead of the old
    # last-writer-wins upload of every token after every job.
    def _token_state_map(self) -> Dict[str, dict]:
        states = getattr(self, "_token_states", None)
        if states is None:
            states = self._token_states = {}
        return states

    def _remember_token(self, filename: str, body: Optional[bytes], etag: Optional[str], s3: str):
        """s3: "present" (read from S3 with ``etag``), "absent" (S3 had no copy) or "unknown"."""
        self._token_state_map()[filename] = {
            "sha256": hashlib.sha256(body).hexdigest() if body is not None else None,
            "etag": etag,
            "s3": s3,
        }

    def _fetch_token_from_s3(self, key: str) -> Optional[Tuple[bytes, Optional[str]]]:
        """Return (content, ETag) of s3://bucket/key in one request, or None when it does not exist."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return obj["Body"].read(), obj.get("ETag")

    def ensure_token_file(self, filename: str) -> str:
        """Ensure that /tmp/<filename> exists (prefers S3 over the package)."""
        local_tmp = os.path.join(getattr(self, "TMP_DIR", "/tmp"), filename)
        if os.path.exists(local_tmp):
            if filename not in self._token_state_map():
                with open(local_tmp, "rb") as fp:
                    self._remember_token(filename, fp.read(), None, "unknown")
            return local_tmp
        s3_checked = False
        if getattr(self, "sync_tokens_s3", True) and getattr(self, "bucket_name", ""):
            key = self._s3_key_for(filename)
            fetched = self._fetch_token_from_s3(key)
            s3_checked = True
            if fetched is not None:
                body, etag = fetched
                os.makedirs(os.path.dirname(local_tmp) or ".", exist_ok=True)
                with open(local_tmp, "wb") as fp:
                    fp.write(body)
                self._remember_token(filename, body, etag, "present")
                print(f"[maws] s3://{self.bucket_name}/{key} -> {local_tmp}")
                return local_tmp
        elif getattr(self, "sync_tokens_s3", True):
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token sync.")
        s3_state = "absent" if s3_checked else "unknown"
        if self._copy_code_to_tmp_if_exists(filename, local_tmp):
            with open(local_tmp, "rb") as fp:
                self._remember_token(filename, fp.read(), None, s3_state)
            return local_tmp
        self._remember_token(filename, None, None, s3_state)
        print(f"[maws][WARN] {filename} not found in S3 or the package.")
        return local_tmp

    def _put_token_object(self, key: str, body: bytes, state: Optional[dict]):
        conditions = {}
        if state and state.get("s3") == "present" and state.get("etag"):
            conditions["IfMatch"] = state["etag"]
        elif state and state.get("s3") == "absent":
            conditions["IfNoneMatch"] = "*"
        try:
            return self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=body, **conditions)
        except botocore.exceptions.ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if conditions and code in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise _TokenWriteConflict(code) from e
            raise
        except Exception as e:
            if conditions and type(e).__name__ == "ParamValidationError":
                # botocore older than S3 conditional writes (late 2024): plain write.
                print("[maws][WARN] This boto3 does not support S3 conditional writes; uploading the token unconditionally.")
                return self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=body)
            raise

    def upload_token_back_if_exists(self, filename: str):
        if not getattr(self, "sync_tokens_s3", True):
            return
        if not getattr(self, "bucket_name", ""):
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token upload.")
            return
        path = os.path.join(getattr(self, "TMP_DIR", "/tmp"), filename)
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            body = f.read()
        digest = hashlib.sha256(body).hexdigest()
        state = self._token_state_map().get(filename)
        if state is not None and state.get("sha256") == digest:
            return  # unchanged since it was read or last uploaded
        key = self._s3_key_for(filename)
        try:
            response = self._put_token_object(key, body, state)
        except _TokenWriteConflict:
            logger.warning(
                "[maws][token] s3://%s/%s changed in another container since this one read it; keeping "
                "the S3 version and discarding this container's copy of %s.",
                self.bucket_name, key, filename,
            )
            try:
                os.remove(path)
            except OSError:
                pass
            self._token_state_map().pop(filename, None)
            return
        etag = response.get("ETag") if isinstance(response, dict) else None
        self._remember_token(filename, body, etag, "present" if etag else "unknown")
        print(f"[maws] uploaded {path} -> s3://{self.bucket_name}/{key}")

    # -------------------------
    # Shared history and file persistence
    # -------------------------
    def _close_manager_history_connections(self):
        manager = getattr(self, "manager", None)
        pool = getattr(manager, "_db_pool", None)
        if not isinstance(pool, dict):
            return
        for key, conn in list(pool.items()):
            try:
                conn.close()
            except Exception:
                pass
            pool.pop(key, None)
        if hasattr(manager, "_tls"):
            manager._tls.db_conn = None

    def _iter_s3_keys(self, prefix: str) -> List[str]:
        keys: List[str] = []
        if hasattr(self.s3, "get_paginator"):
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return keys
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def _sync_shared_history_from_s3(self):
        history_dir = Path(getattr(self, "TMP_DIR", "/tmp")) / "history"
        self._close_manager_history_connections()
        for path in history_dir.glob("shared_history_*.sqlite*"):
            try:
                path.unlink()
            except OSError:
                pass

        prefix = f"{getattr(self, 'shared_history_prefix', 'history/shared').strip('/')}/"
        try:
            keys = [k for k in self._iter_s3_keys(prefix) if k.endswith(".sqlite")]
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchBucket", "NoSuchKey", "NotFound"):
                keys = []
            else:
                raise

        for key in keys:
            local_path = history_dir / os.path.basename(key)
            self.s3.download_file(self.bucket_name, key, str(local_path))
            print(f"[maws] s3://{self.bucket_name}/{key} -> {local_path}")

    def _upload_shared_history_to_s3(self):
        manager = getattr(self, "manager", None)
        if manager is None:
            return
        for db_path in manager._list_shared_history_db_paths():
            pool_key = manager._shared_history_db_key(db_path)
            conn = getattr(manager, "_db_pool", {}).get(pool_key)
            if conn is not None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception as e:
                    print(f"[maws][WARN] Could not checkpoint shared history before upload: {e}")
            if not os.path.exists(db_path):
                continue
            key = f"{getattr(self, 'shared_history_prefix', 'history/shared').strip('/')}/{os.path.basename(db_path)}"
            with open(db_path, "rb") as fp:
                self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=fp.read())
            print(f"[maws] uploaded {db_path} -> s3://{self.bucket_name}/{key}")

    def _local_user_files_dir(self, chat_id: str) -> Path:
        return Path(getattr(self, "TMP_DIR", "/tmp")) / "files" / _safe_key_part(chat_id)

    def _files_s3_prefix_for(self, chat_id: str) -> str:
        return f"{getattr(self, 'files_prefix', 'files').strip('/')}/{_safe_key_part(chat_id)}"

    def _sync_user_files_from_s3(self, chat_id: str):
        if not getattr(self, "persist_files_s3", False):
            return
        local_dir = self._local_user_files_dir(chat_id)
        file_cache = getattr(getattr(self, "manager", None), "_file_cache", None)
        if isinstance(file_cache, dict):
            local_prefix = str(local_dir)
            for cached_path in list(file_cache):
                if str(cached_path).startswith(local_prefix):
                    file_cache.pop(cached_path, None)
        shutil.rmtree(local_dir, ignore_errors=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._files_s3_prefix_for(chat_id) + "/"
        for key in self._iter_s3_keys(prefix):
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            if not rel or rel.startswith("../") or "/../" in rel:
                continue
            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket_name, key, str(dst))
            print(f"[maws] s3://{self.bucket_name}/{key} -> {dst}")

    def _sync_user_files_to_s3(self, chat_id: str):
        if not getattr(self, "persist_files_s3", False):
            return
        local_dir = self._local_user_files_dir(chat_id)
        if not local_dir.exists():
            return
        prefix = self._files_s3_prefix_for(chat_id)
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}"
            with open(path, "rb") as fp:
                self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=fp.read())
            print(f"[maws] uploaded {path} -> s3://{self.bucket_name}/{key}")

    def _known_secret_values(self) -> List[str]:
        values = set(_LOADED_SECRET_VALUES)
        for attr in ("telegram_secret_token", "whatsapp_app_secret", "history_write_secret"):
            value = getattr(self, attr, "") or ""
            if len(value) >= 4:
                values.add(value)
        return sorted(values)

    def _capture_failed_event(self, job_or_event: Any, reason: str, error: Optional[BaseException] = None):
        if not getattr(self, "capture_failed_events", False):
            return
        if not getattr(self, "bucket_name", ""):
            return
        now = _dt.datetime.now(_dt.timezone.utc)
        provider = "unknown"
        event_id = ""
        if isinstance(job_or_event, dict):
            provider = str(job_or_event.get("provider") or job_or_event.get("bot_type") or provider)
            event_id = str(job_or_event.get("event_id") or "")
        if not event_id:
            event_id = _stable_hash(_redact_for_capture(job_or_event, ()))[:24]
        key = (
            f"{getattr(self, 'failed_events_prefix', 'failed-events')}/"
            f"{now:%Y/%m/%d}/{_safe_key_part(provider)}/{_safe_key_part(event_id)}.json"
        )
        # Never store webhook secrets: the raw API Gateway event carries the Telegram secret token,
        # X-Hub-Signature-256, Authorization and similar headers, and errors can quote bot tokens.
        secrets = self._known_secret_values()
        payload = {
            "reason": reason,
            "error": _scrub_text(repr(error), secrets) if error else None,
            "captured_at": now.isoformat(),
            "event": _redact_for_capture(job_or_event, secrets),
        }
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
            )
        except Exception as e:
            # Best effort: a failed capture must not replace the original error.
            print(f"[maws][capture][WARN] Could not capture the failed event: {e}")
            return
        print(f"[maws] captured failed event -> s3://{self.bucket_name}/{key}")

    # -------------------------
    # Locks (optional)
    # -------------------------
    def _lock_now_epoch(self):
        return int(time.time())

    def _lock_id_for(self, user_id: str) -> str:
        if self._uses_shared_history():
            return "__maws_shared_history__"
        return str(user_id)

    def _lock_ttl_seconds(self, context=None) -> int:
        """
        Lifetime of a lock taken now. Inside Lambda the lock lives until this invocation's hard
        deadline plus LOCK_EXPIRY_MARGIN_SECONDS: it outlives any running invocation, and a crashed
        worker's lock expires before SQS makes the message visible again (visibility timeout is
        lambda_timeout + 30 s and the deadline is at most lambda_timeout after the message was
        received). Without a Lambda context, LOCK_TTL_SECONDS is used.
        """
        get_remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
        if callable(get_remaining_ms):
            try:
                remaining = int(math.ceil(float(get_remaining_ms()) / 1000.0))
                return max(1, remaining) + LOCK_EXPIRY_MARGIN_SECONDS
            except Exception:
                pass
        return max(1, int(getattr(self, "lock_ttl", 180) or 180))

    def _held_lock_tokens(self) -> Dict[str, str]:
        tokens = getattr(self, "_lock_tokens", None)
        if tokens is None:
            tokens = self._lock_tokens = {}
        return tokens

    def try_acquire_user_lock(self, user_id: str, context=None) -> bool:
        """Return True if the lock is acquired; False if another worker is processing the user."""
        if not (getattr(self, "lock_table", "") and self.dynamodb):
            return True
        lock_id = self._lock_id_for(user_id)
        now = self._lock_now_epoch()
        expires_at = now + self._lock_ttl_seconds(context)
        token = uuid.uuid4().hex
        try:
            self.dynamodb.put_item(
                TableName=self.lock_table,
                Item={
                    "user_id": {"S": str(lock_id)},
                    "expiresAt": {"N": str(expires_at)},
                    "lockToken": {"S": token},
                },
                # Expiry is enforced by this condition, not by DynamoDB TTL: TTL deletion is lazy
                # (it can lag for days), so an expired item still exists and is overwritten here.
                ConditionExpression="attribute_not_exists(user_id) OR expiresAt < :now",
                ExpressionAttributeValues={":now": {"N": str(now)}},
            )
        except self.dynamodb.exceptions.ConditionalCheckFailedException:
            return False
        self._held_lock_tokens()[str(lock_id)] = token
        return True

    def release_user_lock(self, user_id: str):
        if not (getattr(self, "lock_table", "") and self.dynamodb):
            return
        lock_id = self._lock_id_for(user_id)
        token = self._held_lock_tokens().pop(str(lock_id), None)
        request = {
            "TableName": self.lock_table,
            "Key": {"user_id": {"S": str(lock_id)}},
        }
        if token:
            # Delete only our own lock: if it expired and another worker took it over, keep theirs.
            request["ConditionExpression"] = "lockToken = :token"
            request["ExpressionAttributeValues"] = {":token": {"S": token}}
        try:
            self.dynamodb.delete_item(**request)
        except Exception as e:
            conditional_failed = getattr(
                getattr(self.dynamodb, "exceptions", None), "ConditionalCheckFailedException", None
            )
            if isinstance(conditional_failed, type) and isinstance(e, conditional_failed):
                print(f"[maws][unlock][WARN] Lock for {lock_id} was taken over by another worker; leaving it.")

    @staticmethod
    def _remaining_seconds(context=None) -> Optional[float]:
        get_remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
        if not callable(get_remaining_ms):
            return None
        try:
            return float(get_remaining_ms()) / 1000.0
        except Exception:
            return None

    def _busy_wait_budget(self, context=None) -> float:
        """
        busy_policy=drop: seconds to wait for a busy user lock. MAWS_BUSY_WAIT_SECONDS (default 30)
        caps it, and it never exceeds half of the invocation's remaining time, so the update can
        still be processed (import, model calls, export) once the lock is free.
        """
        cap = _as_float(getattr(self, "busy_wait_seconds", DEFAULT_BUSY_WAIT_SECONDS), DEFAULT_BUSY_WAIT_SECONDS)
        cap = max(0.0, cap if math.isfinite(cap) else DEFAULT_BUSY_WAIT_SECONDS)
        remaining = self._remaining_seconds(context)
        if remaining is not None:
            cap = min(cap, max(0.0, remaining * BUSY_WAIT_SHARE_OF_REMAINING))
        return cap

    def _acquire_user_lock_with_wait(self, user_id: str, context=None) -> Tuple[bool, float]:
        """Poll the user lock for up to _busy_wait_budget() seconds. Returns (acquired, waited_seconds)."""
        budget = self._busy_wait_budget(context)
        poll = max(0.0, float(getattr(self, "busy_wait_poll", BUSY_WAIT_POLL_SECONDS) or 0.0))
        started = time.monotonic()
        deadline = started + budget
        while True:
            if self.try_acquire_user_lock(user_id, context=context):
                waited = time.monotonic() - started
                if waited >= 0.5:
                    print(f"[maws] User lock for {user_id} acquired after waiting {waited:.1f}s.")
                return True, waited
            now = time.monotonic()
            if now >= deadline:
                return False, now - started
            time.sleep(min(poll, deadline - now))

    def _report_dropped_jobs(self, jobs: List[dict], reason: str, waited: float):
        """Every dropped update is logged at WARNING and counted in the MAWS/DroppedMessages metric."""
        for job in jobs:
            logger.warning(
                "[maws][drop] Dropped %s update %s for chat %s (%s): the user lock was still held "
                "after waiting %.1fs (busy_policy=drop). Use busy_policy=fifo to queue updates instead.",
                job.get("provider") or getattr(self, "bot_type", ""),
                job.get("event_id") or "?",
                job.get("chat_id") or "?",
                reason,
                waited,
            )
        self._emit_metric(
            "DroppedMessages",
            len(jobs),
            reason=reason,
            busy_policy=getattr(self, "busy_policy", "drop"),
            waited_seconds=round(waited, 1),
            event_ids=[str(job.get("event_id") or "") for job in jobs],
        )

    # -------------------------
    # Processed-message records
    # -------------------------
    # The same message can reach a worker more than once: the provider retries a whole webhook when
    # one chat's hand-off failed after another chat's succeeded, Lambda can deliver an async
    # invocation twice, and SQS redelivers a message whose worker failed after processing it. Each
    # message is claimed in the locks table before it is processed (the same conditional write as
    # the user lock) and recorded as processed afterwards, so a second delivery is skipped.
    def _dedup_retention_seconds(self) -> int:
        days = _as_float(
            getattr(self, "dedup_retention_days", DEFAULT_DEDUP_RETENTION_DAYS), DEFAULT_DEDUP_RETENTION_DAYS
        )
        if days is None or not math.isfinite(days) or days <= 0:
            return 0
        return int(days * 86400)

    def _dedup_enabled(self) -> bool:
        return bool(getattr(self, "lock_table", "") and self._dedup_retention_seconds() > 0 and self.dynamodb)

    def _dedup_key_for(self, job: dict) -> str:
        provider = str(job.get("provider") or getattr(self, "bot_type", "whatsapp"))
        event_id = str(job.get("event_id") or _stable_hash(job.get("payload"))[:24])
        return f"{DEDUP_KEY_PREFIX}{provider}#{job.get('chat_id') or ''}#{event_id}"

    def _held_claim_tokens(self) -> Dict[str, str]:
        tokens = getattr(self, "_claim_tokens", None)
        if tokens is None:
            tokens = self._claim_tokens = {}
        return tokens

    def _try_claim_job(self, job: dict, context=None) -> bool:
        """Claim a message until this invocation's deadline. False if it was or is being processed."""
        key = self._dedup_key_for(job)
        now = self._lock_now_epoch()
        token = uuid.uuid4().hex
        try:
            self.dynamodb.put_item(
                TableName=self.lock_table,
                Item={
                    "user_id": {"S": key},
                    "expiresAt": {"N": str(now + self._lock_ttl_seconds(context))},
                    "lockToken": {"S": token},
                    "status": {"S": "processing"},
                },
                # A crashed worker's claim expires like its lock, so a retry can process the message;
                # a processed record expires after the retention period.
                ConditionExpression="attribute_not_exists(user_id) OR expiresAt < :now",
                ExpressionAttributeValues={":now": {"N": str(now)}},
            )
        except self.dynamodb.exceptions.ConditionalCheckFailedException:
            return False
        self._held_claim_tokens()[key] = token
        return True

    def _claim_jobs(self, jobs: List[dict], context=None) -> List[dict]:
        """Return the jobs this worker should process, skipping (and counting) repeated deliveries."""
        if not self._dedup_enabled():
            return list(jobs)
        claimed: List[dict] = []
        duplicates: List[dict] = []
        try:
            for job in jobs:
                # History appends are deduplicated by their event id inside the history itself.
                if self._is_history_append_job(job) or self._try_claim_job(job, context):
                    claimed.append(job)
                else:
                    duplicates.append(job)
        except BaseException:
            self._release_job_claims(claimed)
            raise
        if duplicates:
            self._report_duplicate_jobs(duplicates)
        return claimed

    def _release_job_claims(self, jobs: List[dict]):
        """Give up the claims of jobs that were not processed, so a retry or redelivery processes them."""
        for job in jobs:
            key = self._dedup_key_for(job)
            token = self._held_claim_tokens().pop(key, None)
            if token is None:
                continue
            try:
                self.dynamodb.delete_item(
                    TableName=self.lock_table,
                    Key={"user_id": {"S": key}},
                    ConditionExpression="lockToken = :token",
                    ExpressionAttributeValues={":token": {"S": token}},
                )
            except Exception as e:
                # The claim still expires with this invocation's deadline.
                print(f"[maws][dedup][WARN] Could not release the claim on {key}: {e}")

    def _mark_job_processed(self, job: dict):
        key = self._dedup_key_for(job)
        token = self._held_claim_tokens().pop(key, None)
        if token is None:
            return
        now = self._lock_now_epoch()
        try:
            self.dynamodb.put_item(
                TableName=self.lock_table,
                Item={
                    "user_id": {"S": key},
                    "expiresAt": {"N": str(now + self._dedup_retention_seconds())},
                    "lockToken": {"S": token},
                    "status": {"S": "processed"},
                },
                ConditionExpression="lockToken = :token",
                ExpressionAttributeValues={":token": {"S": token}},
            )
        except Exception as e:
            # The claim stays until this invocation's deadline; after that a redelivery of this
            # message would be processed again.
            logger.warning("[maws][dedup] Could not record %s as processed: %s", key, e)

    def _report_duplicate_jobs(self, jobs: List[dict]):
        """Skipped repeated deliveries are logged and counted in the MAWS/DuplicateMessages metric."""
        for job in jobs:
            print(
                f"[maws][dedup] Skipping {job.get('provider') or getattr(self, 'bot_type', '')} update "
                f"{job.get('event_id') or '?'} for chat {job.get('chat_id') or '?'}: it was already "
                "processed, or another worker is processing it."
            )
        self._emit_metric(
            "DuplicateMessages",
            len(jobs),
            busy_policy=getattr(self, "busy_policy", "drop"),
            event_ids=[str(job.get("event_id") or "") for job in jobs],
        )

    # -------------------------
    # MAS initialization + bots
    # -------------------------
    def initialize_system(self):
        if self.manager and self.bot_instance:
            print("[maws] Warm start; system already initialized.")
            return

        # Bring tokens to /tmp if configured
        for f in getattr(self, "special_files", []):
            self.ensure_token_file(f)

        manager_kwargs = dict(getattr(self, "manager_kwargs", {}) or {})
        manager_kwargs.update({
            "config": getattr(self, "config_path", "config.json"),
            "base_directory": getattr(self, "CODE_ROOT", os.getcwd()),
            "history_folder": f"{getattr(self, 'TMP_DIR', '/tmp')}/history",
            "files_folder": f"{getattr(self, 'TMP_DIR', '/tmp')}/files",
            "verbose": getattr(self, "verbose", False),
            "history_mode": getattr(self, "history_mode", "per_user"),
            "history_rotation": getattr(self, "history_rotation", "message_count"),
            "history_max_messages": getattr(self, "history_max_messages", 1000),
            "history_period": getattr(self, "history_period", "1w"),
        })
        self.manager = AgentSystemManager(**manager_kwargs)
        print("[maws] Manager created.")

        bot_kwargs = dict(getattr(self, "bot_kwargs", {}) or {})
        bot_kwargs.setdefault("ensure_delivery", getattr(self, "ensure_delivery", True))
        bot_kwargs.setdefault("delivery_timeout", getattr(self, "delivery_timeout", 60.0))
        bot_kwargs.setdefault("verbose", getattr(self, "verbose", False))
        if getattr(self, "max_allowed_message_delay", None) is not None:
            bot_kwargs.setdefault("max_allowed_message_delay", self.max_allowed_message_delay)

        if getattr(self, "bot_type", "whatsapp") == "telegram":
            self.bot_instance = self.manager.start_telegram_bot(
                start_polling=False,
                **bot_kwargs,
            )

            async def _ensure_ptb_initialized():
                app = self.bot_instance.application
                await app.initialize()
                await app.start()
                print("[maws] Telegram Application initialized for webhook.")

            self.get_event_loop().run_until_complete(_ensure_ptb_initialized())
        else:
            self.bot_instance = self.manager.start_whatsapp_bot(
                run_server=False,
                **bot_kwargs,
            )

        print("[maws] Initialization completed.")

    # -------------------------
    # chat_id extraction + event normalization
    # -------------------------
    @staticmethod
    def _extract_telegram_chat_id(update: dict) -> str:
        try:
            return str(update["message"]["chat"]["id"])
        except Exception:
            return ""

    @staticmethod
    def _extract_whatsapp_chat_id(update: dict) -> str:
        try:
            return (
                update.get("entry", [{}])[0]
                .get("changes", [{}])[0]
                .get("value", {})
                .get("messages", [{}])[0]
                .get("from")
            )
        except Exception:
            return ""

    def _telegram_event_id(self, update: dict) -> str:
        if update.get("update_id") is not None:
            return str(update["update_id"])
        message = update.get("message") or {}
        if message.get("message_id") is not None:
            return str(message["message_id"])
        return _stable_hash(update)[:24]

    def _whatsapp_message_event_id(self, msg: dict) -> str:
        return str(msg.get("id") or _stable_hash(msg)[:24])

    def _job(self, *, provider: str, chat_id: str, event_id: str, payload: dict) -> dict:
        return {
            "maws_job_version": 1,
            "provider": provider,
            "bot_type": getattr(self, "bot_type", provider),
            "chat_id": str(chat_id or ""),
            "event_id": str(event_id or _stable_hash(payload)[:24]),
            "payload": payload,
        }

    def _normalize_telegram_jobs(self, update: dict) -> List[dict]:
        chat_id = self._extract_telegram_chat_id(update)
        if not chat_id:
            return []
        return [self._job(
            provider="telegram",
            chat_id=chat_id,
            event_id=self._telegram_event_id(update),
            payload=update,
        )]

    def _normalize_whatsapp_jobs(self, update: dict) -> List[dict]:
        jobs: List[dict] = []
        for entry in update.get("entry", []) or []:
            changes = entry.get("changes", []) or []
            for change in changes:
                value = change.get("value", {}) or {}
                messages = value.get("messages", []) or []
                for msg in messages:
                    chat_id = msg.get("from")
                    if not chat_id:
                        continue
                    value_copy = copy.deepcopy(value)
                    value_copy["messages"] = [copy.deepcopy(msg)]
                    change_copy = copy.deepcopy(change)
                    change_copy["value"] = value_copy
                    entry_copy = copy.deepcopy(entry)
                    entry_copy["changes"] = [change_copy]
                    payload = {"entry": [entry_copy]}
                    jobs.append(self._job(
                        provider="whatsapp",
                        chat_id=chat_id,
                        event_id=self._whatsapp_message_event_id(msg),
                        payload=payload,
                    ))
        return jobs

    def _jobs_from_provider_payload(self, payload: dict) -> List[dict]:
        if getattr(self, "bot_type", "whatsapp") == "telegram":
            return self._normalize_telegram_jobs(payload)
        return self._normalize_whatsapp_jobs(payload)

    def _jobs_from_worker_event(self, event: dict) -> List[dict]:
        if isinstance(event.get("maws_jobs"), list):
            return [job for job in event["maws_jobs"] if isinstance(job, dict)]
        if isinstance(event.get("maws_job"), dict):
            return [event["maws_job"]]
        if "update" in event:
            jobs = self._normalize_telegram_jobs(event["update"])
            return jobs or [self._job(provider="telegram", chat_id="", event_id="", payload=event["update"])]
        if "whatsapp_update" in event:
            jobs = self._normalize_whatsapp_jobs(event["whatsapp_update"])
            return jobs or [self._job(provider="whatsapp", chat_id="", event_id="", payload=event["whatsapp_update"])]
        return []

    def _group_jobs_by_lock(self, jobs: List[dict]) -> List[List[dict]]:
        """
        Split jobs into groups that need the same lock (one chat, or everything in shared-history
        mode), keeping the provider's order inside each group. Jobs without a chat id are skipped.
        """
        groups: Dict[str, List[dict]] = {}
        for job in jobs:
            chat_id = str(job.get("chat_id") or "")
            if not chat_id:
                print(f"[maws] Skipping a {job.get('provider') or 'provider'} update without chat_id.")
                continue
            groups.setdefault(self._lock_id_for(chat_id), []).append(job)
        return list(groups.values())

    def _queue_group_id_for(self, job: dict) -> str:
        if getattr(self, "history_mode", "per_user") == "shared":
            return "maws-shared-history"
        return _safe_key_part(job.get("chat_id", "") or "unknown")

    def _dedupe_id_for(self, job: dict) -> str:
        return _stable_hash({
            "provider": job.get("provider"),
            "chat_id": job.get("chat_id"),
            "event_id": job.get("event_id"),
            "payload": job.get("payload"),
        })

    # -------------------------
    # Webhook security
    # -------------------------
    def _verify_post_security(self, event: dict, raw_body: bytes) -> Optional[dict]:
        """
        Authenticate a provider webhook POST. Authentication is mandatory: without the provider
        secret the POST is refused with 500 (a misconfiguration; providers retry once it is fixed)
        unless MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true opted out explicitly.
        """
        headers = event.get("headers") or {}
        allow_unauthenticated = bool(getattr(self, "allow_unauthenticated_webhooks", False))
        misconfigured = {"statusCode": 500, "body": json.dumps("Webhook authentication is misconfigured.")}
        if getattr(self, "bot_type", "whatsapp") == "telegram":
            expected = getattr(self, "telegram_secret_token", "")
            if not expected:
                if allow_unauthenticated:
                    return None
                print(
                    "[maws][security][ERR] TELEGRAM_WEBHOOK_SECRET_TOKEN is empty; refusing an "
                    "unauthenticated Telegram webhook. Set it in .env.prod and redeploy, or set "
                    "MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true."
                )
                return misconfigured
            provided = _header_value(headers, "X-Telegram-Bot-Api-Secret-Token")
            if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
                print("[maws][security] Invalid Telegram webhook secret token.")
                return {"statusCode": 403, "body": json.dumps("Forbidden")}
            return None

        verify = bool(getattr(self, "whatsapp_verify_signature", True))
        secret = getattr(self, "whatsapp_app_secret", "")
        if not (verify and secret):
            if allow_unauthenticated:
                return None
            if not verify:
                print(
                    "[maws][security][ERR] WHATSAPP_VERIFY_SIGNATURE=false requires "
                    "MAWS_ALLOW_UNAUTHENTICATED_WEBHOOKS=true; refusing an unverified WhatsApp webhook."
                )
            else:
                print(
                    "[maws][security][ERR] WHATSAPP_APP_SECRET is empty; refusing an unverified "
                    "WhatsApp webhook. Set it in .env.prod and redeploy."
                )
            return misconfigured
        signature = _header_value(headers, "X-Hub-Signature-256")
        if not signature.startswith("sha256="):
            print("[maws][security] Missing WhatsApp X-Hub-Signature-256 header.")
            return {"statusCode": 403, "body": json.dumps("Forbidden")}
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature.encode("utf-8"), expected.encode("utf-8")):
            print("[maws][security] Invalid WhatsApp webhook signature.")
            return {"statusCode": 403, "body": json.dumps("Forbidden")}
        return None

    # -------------------------
    # Main handler
    # -------------------------
    def handle_apigw_event(self, event, context):
        if self._is_sqs_event(event):
            return self._handle_sqs_event(event, context)

        http_method = (
            event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod")
        )

        if http_method == "POST":
            if self._history_append_enabled() and self._is_history_append_path(event):
                return self._handle_history_append(event, context)
            return self._handle_http_post(event, context)

        # GET verification (WhatsApp)
        if http_method == "GET" and getattr(self, "bot_type", "whatsapp") == "whatsapp":
            self.initialize_system()
            if not self.bot_instance:
                return {"statusCode": 200, "body": json.dumps("Error: Initialization failed.")}
            print("[maws] WhatsApp verification (GET).")
            try:
                query_params = event.get("queryStringParameters", {}) or {}
                response_body, status_code = self.bot_instance.handle_webhook_verification(query_params)
                return {"statusCode": status_code, "body": response_body}
            except Exception as e:
                print(f"[maws][verify][ERR] {e}")
                return {"statusCode": 200, "body": "Verification (warning)."}

        jobs = self._jobs_from_worker_event(event)
        if not jobs:
            return {"statusCode": 400, "body": json.dumps("Invocation not recognized.")}
        return self._process_jobs(jobs, context=context, raise_on_error=False)

    # -------------------------
    # External history append (/history/outgoing)
    # -------------------------
    def _history_append_enabled(self) -> bool:
        return bool(getattr(self, "history_append_enabled", False))

    @staticmethod
    def _is_history_append_path(event: dict) -> bool:
        http_path = event.get("rawPath") or event.get("path") or ""
        return http_path.rstrip("/").endswith("/history/outgoing")

    def _is_history_append_job(self, job: dict) -> bool:
        return self._history_append_enabled() and job.get("operation") == "append_history"

    def _verify_history_write_signature(self, event: dict, raw_body: bytes) -> Optional[dict]:
        secret = getattr(self, "history_write_secret", "")
        if not secret:
            return {
                "statusCode": 503,
                "body": json.dumps("MAWS_HISTORY_WRITE_SECRET is not configured."),
            }

        headers = event.get("headers") or {}
        timestamp = _header_value(headers, "x-maws-timestamp")
        signature = _header_value(headers, "x-maws-signature")
        if (
            not timestamp.isascii()
            or not timestamp.isdigit()
            or not signature.isascii()
        ):
            return {"statusCode": 403, "body": json.dumps("Forbidden")}
        try:
            timestamp_seconds = int(timestamp)
        except (TypeError, ValueError):
            return {"statusCode": 403, "body": json.dumps("Forbidden")}

        if abs(time.time() - timestamp_seconds) > 300:
            return {"statusCode": 403, "body": json.dumps("Forbidden")}

        signed_body = timestamp.encode("ascii") + b"." + raw_body
        expected = "sha256=" + hmac.new(
            secret.encode("utf-8"), signed_body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return {"statusCode": 403, "body": json.dumps("Forbidden")}
        return None

    def _handle_history_append(self, event: dict, context):
        raw_body = _event_body_bytes(event)
        security_response = self._verify_history_write_signature(event, raw_body)
        if security_response is not None:
            return security_response

        try:
            data = json.loads(raw_body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"statusCode": 400, "body": json.dumps("Invalid JSON body.")}
        if not isinstance(data, dict):
            return {"statusCode": 400, "body": json.dumps("JSON body must be an object.")}

        chat_id = data.get("chat_id")
        event_id = data.get("event_id")
        content = data.get("content")
        if (
            not isinstance(chat_id, str)
            or not chat_id.strip()
            or len(chat_id) > 256
            or not isinstance(event_id, str)
            or not event_id.strip()
            or len(event_id) > 256
            or not isinstance(content, str)
            or not content.strip()
            or len(content) > 65536
        ):
            return {
                "statusCode": 400,
                "body": json.dumps("chat_id, event_id and content are required."),
            }

        job = {
            "maws_job_version": 1,
            "provider": "history",
            "bot_type": getattr(self, "bot_type", "whatsapp"),
            "chat_id": chat_id.strip(),
            "event_id": event_id.strip(),
            "operation": "append_history",
            "payload": {"content": content},
        }

        if getattr(self, "busy_policy", "drop") == "fifo":
            try:
                queued = self._enqueue_jobs([job])
            except Exception as e:
                print(f"[maws][history][ERR] Could not enqueue history append: {e}")
                self._capture_failed_event(job, "enqueue_failed", e)
                return {"statusCode": 502, "body": json.dumps("History append could not be queued.")}
            if queued["statusCode"] >= 300:
                return queued
        else:
            if getattr(self, "history_mode", "per_user") == "shared":
                print(
                    "[maws][history][WARN] Shared history with busy_policy=drop serializes every append "
                    "behind one global lock; use busy_policy=fifo for bulk sends."
                )
            try:
                invocation = self.lambda_client.invoke(
                    FunctionName=context.invoked_function_arn,
                    InvocationType="Event",
                    Payload=json.dumps({"maws_job": job}),
                )
                if invocation.get("StatusCode") != 202:
                    return {"statusCode": 502, "body": json.dumps("History append could not be queued.")}
            except Exception as e:
                self._capture_failed_event(job, "self_invoke_failed", e)
                return {"statusCode": 502, "body": json.dumps("History append could not be queued.")}

        return {
            "statusCode": 202,
            "body": json.dumps({"status": "accepted", "event_id": event_id.strip()}),
        }

    def _history_contains_external_event(self, chat_id: str, event_id: str) -> bool:
        for message in self.manager.get_messages(chat_id):
            blocks = message.get("message", [])
            if isinstance(blocks, dict):
                blocks = [blocks]
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                metadata = block.get("metadata")
                if isinstance(metadata, dict) and metadata.get("maws_event_id") == event_id:
                    return True
        return False

    def _wait_for_history_append_lock(self, chat_id: str, context=None):
        """
        Unlike webhook jobs, history appends must not be dropped when the user is busy:
        wait for the lock, then raise so Lambda async retries (or SQS) try again later.
        """
        max_wait = getattr(self, "history_append_lock_wait", 60.0) or 0.0
        get_remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
        if callable(get_remaining_ms):
            try:
                # Keep ~20s for S3 import/export once the lock is acquired.
                max_wait = min(max_wait, get_remaining_ms() / 1000.0 - 20.0)
            except Exception:
                pass
        deadline = time.monotonic() + max(0.0, max_wait)
        while not self.try_acquire_user_lock(chat_id, context=context):
            if time.monotonic() >= deadline:
                raise HistoryAppendLockBusy(
                    f"History append for {chat_id} is still waiting for the user lock; retrying later."
                )
            time.sleep(getattr(self, "history_append_lock_poll", 1.0))

    def _history_append_author(self) -> Tuple[str, str]:
        """
        Return (role, msg_type) for an appended message. MAS shows a message to an agent as its
        own turn only when role == agent name, so resolve the agent that answers this bot.
        """
        agents = getattr(getattr(self, "manager", None), "agents", None)
        agents = agents if isinstance(agents, dict) else {}

        role = getattr(self, "history_append_role", "")
        if not role:
            component = getattr(getattr(self, "bot_instance", None), "component_name", None)
            if component in agents:
                role = component
            elif len(agents) == 1:
                role = next(iter(agents))

        if not role:
            print(
                "[maws][history][WARN] Could not infer which agent answers this bot; storing the "
                "message as 'assistant'. Set MAWS_HISTORY_APPEND_ROLE to the agent name."
            )
            return "assistant", "assistant"
        if role not in agents:
            print(f"[maws][history][WARN] MAWS_HISTORY_APPEND_ROLE={role!r} is not an agent in this system.")
            return role, "assistant"
        return role, "agent"

    def _append_external_history(self, chat_id: str, job: dict):
        event_id = str(job.get("event_id") or "")
        if self._history_contains_external_event(chat_id, event_id):
            print(f"[maws][history] Event {event_id} already in history for {chat_id}; skipping.")
            return
        role, msg_type = self._history_append_author()
        self.manager.add_blocks(
            [{
                "type": "text",
                "content": {"response": (job.get("payload") or {}).get("content", "")},
                "metadata": {"maws_event_id": event_id},
            }],
            role=role,
            msg_type=msg_type,
            user_id=chat_id,
        )

    def _handle_http_post(self, event: dict, context):
        print("[maws] Received webhook POST.")
        try:
            raw_body = _event_body_bytes(event)
            security_response = self._verify_post_security(event, raw_body)
            if security_response is not None:
                return security_response
            data = json.loads(raw_body.decode("utf-8") or "{}")
            jobs = self._jobs_from_provider_payload(data)
        except Exception as e:
            # An unreadable body would fail the same way on every provider retry: acknowledge it.
            print(f"[maws][POST][ERR] {e}")
            self._capture_failed_event(event, "post_failed", e)
            return {"statusCode": 200, "body": json.dumps("Webhook received (warning).")}
        if not jobs:
            print("[maws] No processable messages found in webhook POST.")
            return {"statusCode": 200, "body": json.dumps("Webhook received.")}

        # From here on a failure answers 5xx, so Telegram/WhatsApp deliver the update again.
        if getattr(self, "busy_policy", "drop") == "fifo":
            try:
                return self._enqueue_jobs(jobs)
            except Exception as e:
                # Jobs already enqueued are deduplicated by SQS when the retry enqueues them again
                # (content-based MessageDeduplicationId, 5-minute window).
                print(f"[maws][POST][ERR] Could not enqueue the webhook update(s): {e}")
                for job in jobs:
                    self._capture_failed_event(job, "enqueue_failed", e)
                return {"statusCode": 503, "body": json.dumps("Webhook could not be queued; retry later.")}
        return self._invoke_job_groups(jobs, context)

    def _invoke_job_groups(self, jobs: List[dict], context) -> dict:
        """
        busy_policy=drop: hand jobs to async self-invocations, one invocation per lock group, so the
        messages of one chat in a webhook batch are processed in order by a single worker instead of
        racing each other for the user lock.
        """
        failed = False
        for group in self._group_jobs_by_lock(jobs):
            payload = {"maws_job": group[0]} if len(group) == 1 else {"maws_jobs": group}
            try:
                response = self.lambda_client.invoke(
                    FunctionName=context.invoked_function_arn,
                    InvocationType="Event",
                    Payload=json.dumps(payload),
                )
                status = response.get("StatusCode") if isinstance(response, dict) else None
                if status is not None and status != 202:
                    raise RuntimeError(f"async invoke returned StatusCode {status}")
            except Exception as e:
                failed = True
                print(f"[maws][POST][ERR] Could not hand the update(s) to a worker: {e}")
                for job in group:
                    self._capture_failed_event(job, "self_invoke_failed", e)
        if failed:
            # The provider retries the whole webhook. Groups that were handed off before the failure
            # are handed off again, and their workers skip the messages already processed.
            return {"statusCode": 502, "body": json.dumps("Webhook could not be processed; retry later.")}
        return {"statusCode": 200, "body": json.dumps("Webhook received.")}

    def _enqueue_jobs(self, jobs: List[dict]) -> dict:
        queue_url = getattr(self, "queue_url", "")
        if not queue_url:
            print("[maws][config][ERR] MAWS_QUEUE_URL is required when MAWS_BUSY_POLICY=fifo.")
            return {"statusCode": 500, "body": json.dumps("Queue is not configured.")}
        for job in jobs:
            self.sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps({"maws_job": job}),
                MessageGroupId=self._queue_group_id_for(job),
                MessageDeduplicationId=self._dedupe_id_for(job),
            )
        return {"statusCode": 200, "body": json.dumps("Webhook queued.")}

    def _is_sqs_event(self, event: dict) -> bool:
        records = event.get("Records")
        if not isinstance(records, list) or not records:
            return False
        source = records[0].get("eventSource") or records[0].get("EventSource")
        return source == "aws:sqs"

    def _handle_sqs_event(self, event: dict, context):
        jobs: List[dict] = []
        for record in event.get("Records", []):
            try:
                body = json.loads(record.get("body") or "{}")
                jobs.extend(self._jobs_from_worker_event(body))
            except Exception as e:
                print(f"[maws][sqs][ERR] Could not parse record: {e}")
                raise
        # Any exception makes SQS redeliver the message after the visibility timeout; returning
        # normally deletes it. A busy user must therefore raise here instead of being ignored.
        return self._process_jobs(jobs, context=context, raise_on_error=True, retry_when_busy=True)

    def _process_jobs(
        self,
        jobs: List[dict],
        context=None,
        raise_on_error: bool = False,
        retry_when_busy: bool = False,
    ):
        if not jobs:
            return {"statusCode": 200, "body": json.dumps("No jobs found.")}
        groups = self._group_jobs_by_lock(jobs)
        if not groups:
            print("[maws] Could not extract chat_id; exiting.")
            return {"statusCode": 200, "body": json.dumps("No chat_id found; update ignored.")}
        last_result = {"statusCode": 200, "body": json.dumps("Processed.")}
        for group in groups:
            last_result = self._process_job_group(
                group,
                context=context,
                raise_on_error=raise_on_error,
                retry_when_busy=retry_when_busy,
            )
        return last_result

    def _process_job(
        self,
        job: dict,
        context=None,
        raise_on_error: bool = False,
        retry_when_busy: bool = False,
    ):
        return self._process_jobs(
            [job], context=context, raise_on_error=raise_on_error, retry_when_busy=retry_when_busy
        )

    def _process_job_group(
        self,
        jobs: List[dict],
        context=None,
        raise_on_error: bool = False,
        retry_when_busy: bool = False,
    ):
        """
        Process jobs that share one lock (same chat, or shared history) in order, under a single
        acquisition of that lock.

        Busy lock: history appends wait and then raise for a retry; SQS jobs (busy_policy=fifo)
        raise UserLockBusy so SQS redelivers them; webhook jobs in busy_policy=drop wait up to
        _busy_wait_budget() and are then dropped (WARNING log + MAWS/DroppedMessages metric).
        """
        first = jobs[0]
        chat_id = str(first.get("chat_id") or "")
        is_history_append = any(self._is_history_append_job(job) for job in jobs)
        if is_history_append:
            # The caller already got a 202, so failures must surface for Lambda/SQS retries.
            raise_on_error = True

        if not getattr(self, "bucket_name", ""):
            print("[maws][config][ERR] BUCKET_NAME is required for worker events.")
            if raise_on_error:
                # Returning would make SQS delete the message: fail so it is retried / reaches the DLQ.
                raise MawsWorkerError("BUCKET_NAME is not configured; the job was not processed.")
            return self._missing_bucket_response()

        self.initialize_system()
        if not self.bot_instance:
            print("[maws][init][ERR] MAS initialization did not create a bot.")
            if raise_on_error:
                raise MawsWorkerError("MAS initialization failed; the job was not processed.")
            return {"statusCode": 200, "body": json.dumps("Error: Initialization failed.")}

        start = time.time()
        # Repeated deliveries are skipped before waiting for the lock: the worker that has the
        # message may be the one holding it.
        claimed = self._claim_jobs(jobs, context)
        if len(claimed) < len(jobs):
            self._log_jobs_finished(
                [job for job in jobs if job not in claimed], "Duplicate; already processed.", start
            )
        if not claimed:
            return {"statusCode": 200, "body": json.dumps("Duplicate update ignored.")}
        jobs = claimed

        # Lock. In shared history mode this becomes a global lock/group.
        try:
            if is_history_append:
                self._wait_for_history_append_lock(chat_id, context)
            elif retry_when_busy:
                if not self.try_acquire_user_lock(chat_id, context=context):
                    # SQS (busy_policy=fifo): fail the invocation so the message is redelivered after
                    # the visibility timeout instead of being deleted.
                    print(f"[maws] Another worker is processing {chat_id}. Leaving the SQS message for retry.")
                    self._log_jobs_finished(jobs, "Already processing; will retry.", start)
                    raise UserLockBusy(f"User lock for {chat_id} is held by another worker; SQS will retry.")
            else:
                acquired, waited = self._acquire_user_lock_with_wait(chat_id, context)
                if not acquired:
                    # busy_policy=drop: acknowledge and drop, loudly.
                    self._release_job_claims(jobs)
                    self._report_dropped_jobs(jobs, "user_busy", waited)
                    result = {"statusCode": 200, "body": json.dumps("Already processing; update ignored.")}
                    self._log_jobs_finished(jobs, "Already processing; update ignored.", start)
                    return result
        except BaseException:
            # Not processed: the retry or redelivery must be able to claim these messages.
            self._release_job_claims(jobs)
            raise

        result = {"statusCode": 200, "body": json.dumps("Processed.")}
        try:
            for job in jobs:
                result = self._run_locked_job(job, context=context, raise_on_error=raise_on_error)
        finally:
            # Jobs after one that raised were never run.
            self._release_job_claims(jobs)
            # Always release, even when a job failed and the error is re-raised for a retry;
            # otherwise the retry would find the user busy until the lock expires.
            try:
                self.release_user_lock(chat_id)
            except Exception as e:
                print(f"[maws][unlock][WARN] {e}")
        return result

    def _log_jobs_finished(self, jobs: List[dict], status: str, start: float):
        for job in jobs:
            self._log(
                "worker_job_finished",
                provider=str(job.get("provider") or getattr(self, "bot_type", "whatsapp")),
                chat_id=str(job.get("chat_id") or ""),
                event_id=str(job.get("event_id") or ""),
                busy_policy=getattr(self, "busy_policy", "drop"),
                history_mode="shared" if self._uses_shared_history() else "per_user",
                status=status,
                duration_ms=int((time.time() - start) * 1000),
            )

    def _discard_local_history(self, local_db: str):
        """
        Remove this container's copy of a user's SQLite history (and its -wal/-shm/-journal files)
        before importing from S3. Otherwise a warm container whose S3 object was deleted would
        re-import its stale copy and upload it again, resurrecting deleted history.
        """
        self._close_manager_history_connections()
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.remove(local_db + suffix)
            except FileNotFoundError:
                pass

    def _import_history_for(self, chat_id: str, s3_key: str, local_db: str):
        if self._uses_shared_history():
            self._sync_shared_history_from_s3()
            return
        self._discard_local_history(local_db)
        try:
            self.s3.download_file(self.bucket_name, s3_key, local_db)
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
        if os.path.exists(local_db):
            with open(local_db, "rb") as f:
                self.manager.import_history(chat_id, f.read())
        else:
            self.manager.import_history(chat_id, b"")

    def _run_locked_job(self, job: dict, context=None, raise_on_error: bool = False):
        """Import history, process one job and export history. The caller holds the lock."""
        chat_id = str(job.get("chat_id") or "")
        provider = str(job.get("provider") or getattr(self, "bot_type", "whatsapp"))
        event_id = str(job.get("event_id") or "")
        start = time.time()
        imported_history = False
        result = {"statusCode": 200, "body": json.dumps("Processed.")}
        s3_key = self.s3_sqlite_key(chat_id)
        local_db = os.path.join(getattr(self, "TMP_DIR", "/tmp"), "history", f"{_safe_key_part(chat_id)}.sqlite")
        is_history_append = self._is_history_append_job(job)

        try:
            # A token file discarded after a write conflict is fetched again here.
            for f in getattr(self, "special_files", []):
                self.ensure_token_file(f)

            self._import_history_for(chat_id, s3_key, local_db)
            imported_history = True

            self._sync_user_files_from_s3(chat_id)
            self.manager.set_current_user(chat_id)

            if is_history_append:
                self._append_external_history(chat_id, job)
            else:
                # Process
                loop = self.get_event_loop()
                loop.run_until_complete(self.bot_instance.process_webhook_update(job.get("payload") or {}))
            # Recorded before the history is saved: the replies have been sent, so a redelivery
            # must not process this message again, even if saving fails below.
            self._mark_job_processed(job)
        except Exception as e:
            # Not processed (or failed half-way): let a retry or redelivery process it.
            self._release_job_claims([job])
            print(f"[maws][process][ERR] {e}")
            traceback.print_exc()
            self._capture_failed_event(job, "worker_processing_failed", e)
            result = {"statusCode": 200, "body": json.dumps("Processed with warning.")}
            if raise_on_error:
                raise
            # busy_policy=drop does not retry a failed update: count it.
            self._emit_metric("FailedMessages", 1, reason="processing_failed", provider=provider, event_ids=[event_id])
        finally:
            try:
                # Persistence only if we imported/synced history
                if imported_history:
                    for f in getattr(self, "special_files", []):
                        self.upload_token_back_if_exists(f)
                    try:
                        if self._uses_shared_history():
                            self._upload_shared_history_to_s3()
                        else:
                            sqlite_bytes = self.manager.export_history(chat_id)
                            if sqlite_bytes:
                                self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=sqlite_bytes)
                        self._sync_user_files_to_s3(chat_id)
                    except Exception as e:
                        print(f"[maws][save][ERR] {e}")
                        self._capture_failed_event(job, "worker_save_failed", e)
                        result = {"statusCode": 200, "body": json.dumps("Processed with warning.")}
                        if raise_on_error:
                            raise
            finally:
                self._log(
                    "worker_job_finished",
                    provider=provider,
                    chat_id=chat_id,
                    event_id=event_id,
                    busy_policy=getattr(self, "busy_policy", "drop"),
                    history_mode="shared" if self._uses_shared_history() else "per_user",
                    status=json.loads(result["body"]),
                    duration_ms=int((time.time() - start) * 1000),
                )

        return result


# -------------------------
# Lambda handler builder
# -------------------------
def build_lambda_handler(bot_type: str):
    runtime = MawsRuntime()
    # Respect BOT_TYPE passed via env if it differs
    if bot_type:
        runtime.bot_type = bot_type.strip().lower()

    def _handler(event, context):
        return runtime.handle_apigw_event(event, context)

    return _handler
