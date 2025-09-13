# maws.py
import os, json, asyncio, shutil, traceback
import boto3, botocore

# MAS
from mas import AgentSystemManager, TelegramBot, WhatsappBot

# -------------------------
# Utilidades y configuración
# -------------------------
def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _load_env_from_ssm_if_needed():
    """
    Carga un .env desde SSM (SecureString) y lo vuelca a os.environ,
    sin sobreescribir claves ya presentes. Lo hace una sola vez.
    """
    param_name = os.environ.get("ENV_PARAMETER_NAME")
    if not param_name:
        return
    if os.environ.get("_ENV_LOADED_FROM_SSM") == "1":
        return
    ssm = boto3.client("ssm")
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
# Runtime principal
# -------------------------
class MawsRuntime:
    def __init__(self):
        _load_env_from_ssm_if_needed()

        # Config general
        self.bucket_name = os.environ.get("BUCKET_NAME", "MISSING_BUCKET")
        self.sync_tokens_s3 = _as_bool(os.environ.get("SYNC_TOKENS_S3", "1"), default=True)
        self.tokens_prefix = os.environ.get("TOKENS_S3_PREFIX", "secrets")
        self.bot_type = os.environ.get("BOT_TYPE", "whatsapp").strip().lower()
        self.verbose = _as_bool(os.environ.get("VERBOSE", "false"), default=False)

        # Locks DynamoDB (opcional)
        self.lock_table = os.environ.get("LOCKS_TABLE_NAME") or ""
        self.lock_ttl = int(os.environ.get("LOCK_TTL_SECONDS", "180"))

        # Tokens especiales (opcionales)
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

        # AWS clients
        self.s3 = boto3.client("s3")
        self.lambda_client = boto3.client("lambda")
        self.dynamodb = boto3.client("dynamodb") if self.lock_table else None

        # Rutas
        self.CODE_ROOT = "/var/task"  # paquete (read-only)
        self.TMP_DIR = "/tmp"
        os.makedirs(f"{self.TMP_DIR}/history", exist_ok=True)
        os.makedirs(f"{self.TMP_DIR}/files", exist_ok=True)

        # Estado "warm"
        self.manager = None
        self.bot_instance = None
        self._loop = None

        # Si mapeaste archivos->ENV, publica paths en ENV
        for fname, envkey in self.token_env_map.items():
            os.environ.setdefault(envkey, f"{self.TMP_DIR}/{fname}")

    # -------------------------
    # Helpers event loop / S3 / paths
    # -------------------------
    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def s3_sqlite_key(self, chat_id: str) -> str:
        return f"history/{chat_id}.sqlite"

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
        """Garantiza que /tmp/<filename> exista (S3 > paquete)."""
        local_tmp = os.path.join(self.TMP_DIR, filename)
        if os.path.exists(local_tmp):
            return local_tmp
        if self.sync_tokens_s3:
            key = self._s3_key_for(filename)
            if self._download_s3_if_exists(self.bucket_name, key, local_tmp):
                return local_tmp
        if self._copy_code_to_tmp_if_exists(filename, local_tmp):
            return local_tmp
        print(f"[maws][WARN] No existe {filename} ni en S3 ni en el paquete.")
        return local_tmp

    def upload_token_back_if_exists(self, filename: str):
        if not self.sync_tokens_s3:
            return
        path = os.path.join(self.TMP_DIR, filename)
        if os.path.exists(path):
            key = self._s3_key_for(filename)
            with open(path, "rb") as f:
                self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=f.read())
            print(f"[maws] subido {path} -> s3://{self.bucket_name}/{key}")

    # -------------------------
    # Locks (opcionales)
    # -------------------------
    def _lock_now_epoch(self):
        from time import time
        return int(time())

    def try_acquire_user_lock(self, user_id: str) -> bool:
        """True si toma el lock; False si ya hay otro worker procesando ese user_id."""
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
    # Inicialización MAS + Bots
    # -------------------------
    def initialize_system(self):
        if self.manager and self.bot_instance:
            print("[maws] Warm start; sistema ya inicializado.")
            return

        # Traer tokens a /tmp si están configurados
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
        print("[maws] Manager creado.")

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

        print("[maws] Inicialización completada.")

    # -------------------------
    # Extracción de chat_id
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
    # Handler principal
    # -------------------------
    def handle_apigw_event(self, event, context):
        self.initialize_system()
        if not self.bot_instance:
            return {"statusCode": 200, "body": json.dumps("Error: Fallo en la inicialización.")}

        http_method = (
            event.get("requestContext", {}).get("http", {}).get("method")
            or event.get("httpMethod")
        )

        # GET verification (WhatsApp)
        if http_method == "GET" and self.bot_type == "whatsapp":
            print("[maws] Verificación (GET) WhatsApp.")
            try:
                query_params = event.get("queryStringParameters", {}) or {}
                response_body, status_code = self.bot_instance.handle_webhook_verification(query_params)
                return {"statusCode": status_code, "body": response_body}
            except Exception as e:
                print(f"[maws][verify][ERR] {e}")
                return {"statusCode": 200, "body": "Verification (warning)."}

        # POST fan-out (ambos)
        if http_method == "POST":
            print("[maws] Webhook POST recibido.")
            try:
                body = event.get("body", "{}")
                data = json.loads(body)
                payload_key = "update" if self.bot_type == "telegram" else "whatsapp_update"
                self.lambda_client.invoke(
                    FunctionName=context.invoked_function_arn,
                    InvocationType="Event",
                    Payload=json.dumps({payload_key: data}),
                )
                return {"statusCode": 200, "body": json.dumps("Webhook recibido.")}
            except Exception as e:
                print(f"[maws][POST][ERR] {e}")
                return {"statusCode": 200, "body": json.dumps("Webhook recibido (warning).")}

        # Background: procesamiento real
        is_tg = "update" in event
        is_wa = "whatsapp_update" in event
        if not (is_tg or is_wa):
            return {"statusCode": 400, "body": json.dumps("Invocación no reconocida.")}

        update = event["update"] if is_tg else event["whatsapp_update"]
        chat_id = (
            self._extract_telegram_chat_id(update)
            if is_tg
            else self._extract_whatsapp_chat_id(update)
        )
        if not chat_id:
            print("[maws] No pude extraer chat_id; saliendo.")
            return

        acquired_lock = False
        imported_history = False
        s3_key = self.s3_sqlite_key(chat_id)
        local_db = f"{self.TMP_DIR}/history/{chat_id}.sqlite"

        try:
            # Lock
            if not self.try_acquire_user_lock(chat_id):
                print(f"[maws] Otro worker procesa {chat_id}. Ignoro update.")
                return
            acquired_lock = True

            # Traer historial
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

            # Procesar
            loop = self.get_event_loop()
            loop.run_until_complete(self.bot_instance.process_webhook_update(update))
        except Exception as e:
            print(f"[maws][process][ERR] {e}")
            traceback.print_exc()
        finally:
            # Persistencia sólo si importamos el historial
            if imported_history:
                # Subir tokens si cambiaron
                for f in self.special_files:
                    self.upload_token_back_if_exists(f)
                # Guardar DB
                try:
                    sqlite_bytes = self.manager.export_history(chat_id)
                    if sqlite_bytes:
                        self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=sqlite_bytes)
                except Exception as e:
                    print(f"[maws][save][ERR] {e}")

            # Liberar lock si lo tomamos
            if acquired_lock:
                try:
                    self.release_user_lock(chat_id)
                except Exception as e:
                    print(f"[maws][unlock][WARN] {e}")

        return

# -------------------------
# Creador de handler Lambda
# -------------------------
def build_lambda_handler(bot_type: str):
    runtime = MawsRuntime()
    # Honrar BOT_TYPE pasado por env si no coincide
    if bot_type:
        runtime.bot_type = bot_type.strip().lower()

    def _handler(event, context):
        return runtime.handle_apigw_event(event, context)

    return _handler
