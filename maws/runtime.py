from __future__ import annotations
# maws.py
import os, json, asyncio, shutil, traceback, base64, re, types as _types
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

from typing import Optional, Union, List, Dict
import datetime as _dt

# -------------------------
# Utilities and configuration
# -------------------------
def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

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

# -------------------------
# Main runtime
# -------------------------
class MawsRuntime:
    def __init__(self):
        _load_env_from_ssm_if_needed()

        # General config
        self.bucket_name = os.environ.get("BUCKET_NAME", "").strip()
        self.sync_tokens_s3 = _as_bool(os.environ.get("SYNC_TOKENS_S3", "1"), default=True)
        self.tokens_prefix = os.environ.get("TOKENS_S3_PREFIX", "secrets")
        self.bot_type = os.environ.get("BOT_TYPE", "whatsapp").strip().lower()
        self.verbose = _as_bool(os.environ.get("VERBOSE", "false"), default=False)

        # DynamoDB locks (optional)
        self.lock_table = os.environ.get("LOCKS_TABLE_NAME") or ""
        self.lock_ttl = int(os.environ.get("LOCK_TTL_SECONDS", "180"))

        # Special tokens (optional)
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

        # Paths
        self.CODE_ROOT = "/var/task"  # package (read-only)
        self.TMP_DIR = "/tmp"
        os.makedirs(f"{self.TMP_DIR}/history", exist_ok=True)
        os.makedirs(f"{self.TMP_DIR}/files", exist_ok=True)

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
    def dynamodb(self):
        if not self.lock_table:
            return None
        if self._dynamodb is None:
            self._dynamodb = _require_boto3().client("dynamodb")
        return self._dynamodb

    @dynamodb.setter
    def dynamodb(self, client):
        self._dynamodb = client

    def _missing_bucket_response(self):
        return {
            "statusCode": 500,
            "body": json.dumps("BUCKET_NAME is required for MAWS worker history persistence."),
        }

    # -------------------------
    # Helpers event loop / S3 / paths
    # -------------------------
    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def s3_sqlite_key(self, chat_id: str) -> str:
        return f"history/{_safe_key_part(chat_id)}.sqlite"

    def _s3_key_for(self, filename: str) -> str:
        prefix = (self.tokens_prefix or "").strip().strip("/")
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
        src_path = os.path.join(self.CODE_ROOT, filename)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            print(f"[maws] {src_path} -> {dst_path}")
            return True
        return False

    def ensure_token_file(self, filename: str) -> str:
        """Ensure that /tmp/<filename> exists (prefers S3 over the package)."""
        local_tmp = os.path.join(self.TMP_DIR, filename)
        if os.path.exists(local_tmp):
            return local_tmp
        if self.sync_tokens_s3 and self.bucket_name:
            key = self._s3_key_for(filename)
            if self._download_s3_if_exists(self.bucket_name, key, local_tmp):
                return local_tmp
        elif self.sync_tokens_s3:
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token sync.")
        if self._copy_code_to_tmp_if_exists(filename, local_tmp):
            return local_tmp
        print(f"[maws][WARN] {filename} not found in S3 or the package.")
        return local_tmp

    def upload_token_back_if_exists(self, filename: str):
        if not self.sync_tokens_s3:
            return
        if not self.bucket_name:
            print("[maws][WARN] BUCKET_NAME is not set; skipping S3 token upload.")
            return
        path = os.path.join(self.TMP_DIR, filename)
        if os.path.exists(path):
            key = self._s3_key_for(filename)
            with open(path, "rb") as f:
                self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=f.read())
            print(f"[maws] uploaded {path} -> s3://{self.bucket_name}/{key}")

    # -------------------------
    # Locks (optional)
    # -------------------------
    def _lock_now_epoch(self):
        from time import time
        return int(time())

    def try_acquire_user_lock(self, user_id: str) -> bool:
        """Return True if the lock is acquired; False if another worker is processing the user."""
        if not (self.lock_table and self.dynamodb):
            return True
        now = self._lock_now_epoch()
        expires_at = now + self.lock_ttl
        try:
            self.dynamodb.put_item(
                TableName=self.lock_table,
                Item={"user_id": {"S": str(user_id)}, "expiresAt": {"N": str(expires_at)}},
                ConditionExpression="attribute_not_exists(user_id) OR expiresAt < :now",
                ExpressionAttributeValues={":now": {"N": str(now)}},
            )
            return True
        except self.dynamodb.exceptions.ConditionalCheckFailedException:
            return False

    def release_user_lock(self, user_id: str):
        if not (self.lock_table and self.dynamodb):
            return
        try:
            self.dynamodb.delete_item(
                TableName=self.lock_table,
                Key={"user_id": {"S": str(user_id)}},
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
        for f in self.special_files:
            self.ensure_token_file(f)

        base_dir = "/var/task" if "AWS_LAMBDA_FUNCTION_NAME" in os.environ else os.getcwd()
        self.manager = AgentSystemManager(
            config="config.json",
            base_directory=base_dir,
            history_folder=f"{self.TMP_DIR}/history",
            files_folder=f"{self.TMP_DIR}/files",
            verbose=self.verbose,
        )
        print("[maws] Manager created.")

        if self.bot_type == "telegram":
            self.bot_instance = self.manager.start_telegram_bot(
                start_polling=False,
                ensure_delivery=True,
                delivery_timeout=60.0,
                verbose=self.verbose,
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
                ensure_delivery=True,
                delivery_timeout=60.0,
                verbose=self.verbose,
            )

        print("[maws] Initialization completed.")

    # -------------------------
    # chat_id extraction
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

    # -------------------------
    # Main handler
    # -------------------------
    def handle_apigw_event(self, event, context):
        http_method = (
            event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod")
        )

        # POST fan-out (both)
        if http_method == "POST":
            print("[maws] Received webhook POST.")
            try:
                body = event.get("body", "{}")
                data = json.loads(body)
                payload_key = "update" if self.bot_type == "telegram" else "whatsapp_update"
                self.lambda_client.invoke(
                    FunctionName=context.invoked_function_arn,
                    InvocationType="Event",
                    Payload=json.dumps({payload_key: data}),
                )
                return {"statusCode": 200, "body": json.dumps("Webhook received.")}
            except Exception as e:
                print(f"[maws][POST][ERR] {e}")
                return {"statusCode": 200, "body": json.dumps("Webhook received (warning).")}

        # GET verification (WhatsApp)
        if http_method == "GET" and self.bot_type == "whatsapp":
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

        # Background: procesamiento real
        is_tg = "update" in event
        is_wa = "whatsapp_update" in event
        if not (is_tg or is_wa):
            return {"statusCode": 400, "body": json.dumps("Invocation not recognized.")}

        update = event["update"] if is_tg else event["whatsapp_update"]
        chat_id = (
            self._extract_telegram_chat_id(update)
            if is_tg
            else self._extract_whatsapp_chat_id(update)
        )
        if not chat_id:
            print("[maws] Could not extract chat_id; exiting.")
            return {"statusCode": 200, "body": json.dumps("No chat_id found; update ignored.")}

        if not self.bucket_name:
            print("[maws][config][ERR] BUCKET_NAME is required for worker events.")
            return self._missing_bucket_response()

        self.initialize_system()
        if not self.bot_instance:
            return {"statusCode": 200, "body": json.dumps("Error: Initialization failed.")}

        acquired_lock = False
        imported_history = False
        s3_key = self.s3_sqlite_key(chat_id)
        local_db = os.path.join(self.TMP_DIR, "history", f"{_safe_key_part(chat_id)}.sqlite")

        try:
            # Lock
            if not self.try_acquire_user_lock(chat_id):
                print(f"[maws] Another worker is processing {chat_id}. Ignoring update.")
                return {"statusCode": 200, "body": json.dumps("Already processing; update ignored.")}
            acquired_lock = True

            # Fetch history
            try:
                self.s3.download_file(self.bucket_name, s3_key, local_db)
            except botocore.exceptions.ClientError as e:
                if e.response.get("Error", {}).get("Code") != "404":
                    raise

            if os.path.exists(local_db):
                with open(local_db, "rb") as f:
                    self.manager.import_history(chat_id, f.read())
            else:
                self.manager.import_history(chat_id, b"")
            imported_history = True

            self.manager.set_current_user(chat_id)

            # Process
            loop = self.get_event_loop()
            loop.run_until_complete(self.bot_instance.process_webhook_update(update))
        except Exception as e:
            print(f"[maws][process][ERR] {e}")
            traceback.print_exc()
            result = {"statusCode": 200, "body": json.dumps("Processed with warning.")}
        else:
            result = {"statusCode": 200, "body": json.dumps("Processed.")}
        finally:
            # Persistence only if we imported the history
            if imported_history:
                # Upload tokens if they changed
                for f in self.special_files:
                    self.upload_token_back_if_exists(f)
                # Save DB
                try:
                    sqlite_bytes = self.manager.export_history(chat_id)
                    if sqlite_bytes:
                        self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=sqlite_bytes)
                except Exception as e:
                    print(f"[maws][save][ERR] {e}")

            # Release lock if acquired
            if acquired_lock:
                try:
                    self.release_user_lock(chat_id)
                except Exception as e:
                    print(f"[maws][unlock][WARN] {e}")

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
