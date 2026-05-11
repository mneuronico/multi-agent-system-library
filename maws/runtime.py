from __future__ import annotations

import asyncio
import base64
import copy
import datetime as _dt
import hashlib
import hmac
import json
import os
import re
import shutil
import time
import traceback
import types as _types
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        raise ImportError("boto3 and botocore are required for maws AWS operations. Install mas[aws] or mas[maws].")
    return boto3


def _load_env_from_ssm_if_needed():
    """
    Load a .env from SSM (SecureString) and merge it into ``os.environ``
    without overriding existing keys. Runs only once per process.
    """
    param_name = os.environ.get("ENV_PARAMETER_NAME")
    if not param_name:
        return
    if os.environ.get("_ENV_LOADED_FROM_SSM") == "1":
        return
    ssm = _require_boto3().client("ssm")
    resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
    raw = resp["Parameter"]["Value"]
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v
    os.environ["_ENV_LOADED_FROM_SSM"] = "1"
    print(f"[maws] Loaded env from SSM parameter: {param_name}")


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

        # DynamoDB locks (optional)
        self.lock_table = os.environ.get("LOCKS_TABLE_NAME") or ""
        self.lock_ttl = int(os.environ.get("LOCK_TTL_SECONDS", "180"))

        # Provider-compatible webhook authenticity checks. Opt-in for compatibility.
        self.telegram_secret_token_env_key = os.environ.get(
            "TELEGRAM_WEBHOOK_SECRET_TOKEN_ENV_KEY",
            "TELEGRAM_WEBHOOK_SECRET_TOKEN",
        )
        self.telegram_secret_token = (
            os.environ.get(self.telegram_secret_token_env_key, "")
            or os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN", "")
        )
        self.whatsapp_verify_signature = _as_bool(os.environ.get("WHATSAPP_VERIFY_SIGNATURE"), default=False)
        self.whatsapp_app_secret_env_key = os.environ.get(
            "WHATSAPP_APP_SECRET_ENV_KEY",
            "WHATSAPP_APP_SECRET",
        )
        self.whatsapp_app_secret = (
            os.environ.get(self.whatsapp_app_secret_env_key, "")
            or os.environ.get("WHATSAPP_APP_SECRET", "")
        )

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

    # -------------------------
    # Helpers event loop / S3 / paths
    # -------------------------
    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _uses_shared_history(self) -> bool:
        if self.manager is not None and hasattr(self.manager, "_history_mode_is_shared"):
            try:
                return bool(self.manager._history_mode_is_shared())
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

    def ensure_token_file(self, filename: str) -> str:
        """Ensure that /tmp/<filename> exists (prefers S3 over the package)."""
        local_tmp = os.path.join(getattr(self, "TMP_DIR", "/tmp"), filename)
        if os.path.exists(local_tmp):
            return local_tmp
        if getattr(self, "sync_tokens_s3", True) and getattr(self, "bucket_name", ""):
            key = self._s3_key_for(filename)
            if self._download_s3_if_exists(self.bucket_name, key, local_tmp):
                return local_tmp
        elif getattr(self, "sync_tokens_s3", True):
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token sync.")
        if self._copy_code_to_tmp_if_exists(filename, local_tmp):
            return local_tmp
        print(f"[maws][WARN] {filename} not found in S3 or the package.")
        return local_tmp

    def upload_token_back_if_exists(self, filename: str):
        if not getattr(self, "sync_tokens_s3", True):
            return
        if not getattr(self, "bucket_name", ""):
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token upload.")
            return
        path = os.path.join(getattr(self, "TMP_DIR", "/tmp"), filename)
        if os.path.exists(path):
            key = self._s3_key_for(filename)
            with open(path, "rb") as f:
                self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=f.read())
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
            event_id = _stable_hash(job_or_event)[:24]
        key = (
            f"{getattr(self, 'failed_events_prefix', 'failed-events')}/"
            f"{now:%Y/%m/%d}/{_safe_key_part(provider)}/{_safe_key_part(event_id)}.json"
        )
        payload = {
            "reason": reason,
            "error": repr(error) if error else None,
            "captured_at": now.isoformat(),
            "event": job_or_event,
        }
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        )
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

    def try_acquire_user_lock(self, user_id: str) -> bool:
        """Return True if the lock is acquired; False if another worker is processing the user."""
        if not (getattr(self, "lock_table", "") and self.dynamodb):
            return True
        lock_id = self._lock_id_for(user_id)
        now = self._lock_now_epoch()
        expires_at = now + getattr(self, "lock_ttl", 180)
        try:
            self.dynamodb.put_item(
                TableName=self.lock_table,
                Item={"user_id": {"S": str(lock_id)}, "expiresAt": {"N": str(expires_at)}},
                ConditionExpression="attribute_not_exists(user_id) OR expiresAt < :now",
                ExpressionAttributeValues={":now": {"N": str(now)}},
            )
            return True
        except self.dynamodb.exceptions.ConditionalCheckFailedException:
            return False

    def release_user_lock(self, user_id: str):
        if not (getattr(self, "lock_table", "") and self.dynamodb):
            return
        lock_id = self._lock_id_for(user_id)
        try:
            self.dynamodb.delete_item(
                TableName=self.lock_table,
                Key={"user_id": {"S": str(lock_id)}},
            )
        except Exception:
            pass

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
        if isinstance(event.get("maws_job"), dict):
            return [event["maws_job"]]
        if "update" in event:
            jobs = self._normalize_telegram_jobs(event["update"])
            return jobs or [self._job(provider="telegram", chat_id="", event_id="", payload=event["update"])]
        if "whatsapp_update" in event:
            jobs = self._normalize_whatsapp_jobs(event["whatsapp_update"])
            return jobs or [self._job(provider="whatsapp", chat_id="", event_id="", payload=event["whatsapp_update"])]
        return []

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
        headers = event.get("headers") or {}
        if getattr(self, "bot_type", "whatsapp") == "telegram":
            expected = getattr(self, "telegram_secret_token", "")
            if not expected:
                return None
            provided = _header_value(headers, "X-Telegram-Bot-Api-Secret-Token")
            if not hmac.compare_digest(provided, expected):
                print("[maws][security] Invalid Telegram webhook secret token.")
                return {"statusCode": 403, "body": json.dumps("Forbidden")}
            return None

        if not getattr(self, "whatsapp_verify_signature", False):
            return None
        secret = getattr(self, "whatsapp_app_secret", "")
        if not secret:
            print("[maws][security][ERR] WHATSAPP_VERIFY_SIGNATURE=true but WHATSAPP_APP_SECRET is empty.")
            return {"statusCode": 500, "body": json.dumps("Webhook signature verification is misconfigured.")}
        signature = _header_value(headers, "X-Hub-Signature-256")
        if not signature.startswith("sha256="):
            print("[maws][security] Missing WhatsApp X-Hub-Signature-256 header.")
            return {"statusCode": 403, "body": json.dumps("Forbidden")}
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
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

    def _handle_http_post(self, event: dict, context):
        print("[maws] Received webhook POST.")
        try:
            raw_body = _event_body_bytes(event)
            security_response = self._verify_post_security(event, raw_body)
            if security_response is not None:
                return security_response
            data = json.loads(raw_body.decode("utf-8") or "{}")
            jobs = self._jobs_from_provider_payload(data)
            if not jobs:
                print("[maws] No processable messages found in webhook POST.")
                return {"statusCode": 200, "body": json.dumps("Webhook received.")}

            if getattr(self, "busy_policy", "drop") == "fifo":
                return self._enqueue_jobs(jobs)

            warning = False
            for job in jobs:
                try:
                    self.lambda_client.invoke(
                        FunctionName=context.invoked_function_arn,
                        InvocationType="Event",
                        Payload=json.dumps({"maws_job": job}),
                    )
                except Exception as e:
                    warning = True
                    print(f"[maws][POST][ERR] {e}")
                    self._capture_failed_event(job, "self_invoke_failed", e)
            if warning:
                return {"statusCode": 200, "body": json.dumps("Webhook received (warning).")}
            return {"statusCode": 200, "body": json.dumps("Webhook received.")}
        except Exception as e:
            print(f"[maws][POST][ERR] {e}")
            self._capture_failed_event(event, "post_failed", e)
            return {"statusCode": 200, "body": json.dumps("Webhook received (warning).")}

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
        return self._process_jobs(jobs, context=context, raise_on_error=True)

    def _process_jobs(self, jobs: List[dict], context=None, raise_on_error: bool = False):
        if not jobs:
            return {"statusCode": 200, "body": json.dumps("No jobs found.")}
        last_result = {"statusCode": 200, "body": json.dumps("Processed.")}
        for job in jobs:
            last_result = self._process_job(job, context=context, raise_on_error=raise_on_error)
        return last_result

    def _process_job(self, job: dict, context=None, raise_on_error: bool = False):
        chat_id = str(job.get("chat_id") or "")
        provider = str(job.get("provider") or getattr(self, "bot_type", "whatsapp"))
        event_id = str(job.get("event_id") or "")
        start = time.time()
        if not chat_id:
            print("[maws] Could not extract chat_id; exiting.")
            return {"statusCode": 200, "body": json.dumps("No chat_id found; update ignored.")}

        if not getattr(self, "bucket_name", ""):
            print("[maws][config][ERR] BUCKET_NAME is required for worker events.")
            return self._missing_bucket_response()

        self.initialize_system()
        if not self.bot_instance:
            return {"statusCode": 200, "body": json.dumps("Error: Initialization failed.")}

        acquired_lock = False
        imported_history = False
        result = {"statusCode": 200, "body": json.dumps("Processed.")}
        s3_key = self.s3_sqlite_key(chat_id)
        local_db = os.path.join(getattr(self, "TMP_DIR", "/tmp"), "history", f"{_safe_key_part(chat_id)}.sqlite")

        try:
            # Lock. In shared history mode this becomes a global lock/group.
            if not self.try_acquire_user_lock(chat_id):
                print(f"[maws] Another worker is processing {chat_id}. Ignoring update.")
                result = {"statusCode": 200, "body": json.dumps("Already processing; update ignored.")}
                return {"statusCode": 200, "body": json.dumps("Already processing; update ignored.")}
            acquired_lock = True

            if self._uses_shared_history():
                self._sync_shared_history_from_s3()
            else:
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
            imported_history = True

            self._sync_user_files_from_s3(chat_id)
            self.manager.set_current_user(chat_id)

            # Process
            loop = self.get_event_loop()
            loop.run_until_complete(self.bot_instance.process_webhook_update(job.get("payload") or {}))
        except Exception as e:
            print(f"[maws][process][ERR] {e}")
            traceback.print_exc()
            self._capture_failed_event(job, "worker_processing_failed", e)
            result = {"statusCode": 200, "body": json.dumps("Processed with warning.")}
            if raise_on_error:
                raise
        else:
            result = {"statusCode": 200, "body": json.dumps("Processed.")}
        finally:
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

            if acquired_lock:
                try:
                    self.release_user_lock(chat_id)
                except Exception as e:
                    print(f"[maws][unlock][WARN] {e}")

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
