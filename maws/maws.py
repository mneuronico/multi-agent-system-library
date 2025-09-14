# maws.py
import os, json, asyncio, shutil, traceback
import boto3, botocore

# MAS
from mas import AgentSystemManager, TelegramBot, WhatsappBot

from typing import Optional, Union, List, Dict

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


import sys
import json as _json
import tempfile as _tempfile
import subprocess as _subprocess
import shutil as _shutil
import platform as _platform
from pathlib import Path
import textwrap

try:
    from importlib.resources import files as _res_files
except Exception:  # pragma: no cover
    import importlib_resources  # type: ignore
    _res_files = importlib_resources.files  # type: ignore

def _resource_path(name: str) -> Path:
    return Path(_res_files("maws.resources") / name)

def _ensure_script_in_cwd(name: str, dest_dir: Path, force: bool = False) -> Path:
    """
    Devuelve una ruta al script. Por defecto usa el **embebido** dentro del paquete.
    Si 'force' es True o MAWS_COPY_SCRIPTS=1, **copia** el script al proyecto y devuelve esa ruta.
    """
    use_copy = force or os.environ.get("MAWS_COPY_SCRIPTS") == "1"
    if not use_copy:
        return _resource_path(name)

    dest = dest_dir / name
    src = _resource_path(name)
    if not dest.exists() or force:
        dest.write_bytes(src.read_bytes())
        dest.chmod(0o755)
    return dest

def _run_bash(
    script: Path,
    args: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> int:
    bash = _shutil.which("bash")
    if bash is None and _platform.system().lower().startswith("win"):
        if _shutil.which("wsl"):
            cmd = ["wsl", "bash", str(script), *args]
        else:
            raise RuntimeError(
                "No encontré 'bash'. En Windows instalá WSL o Git Bash.\n"
                "WSL: https://learn.microsoft.com/windows/wsl/install"
            )
    else:
        cmd = [bash or "bash", str(script), *args]

    proc = _subprocess.Popen(
        cmd, cwd=str(cwd or Path.cwd()), env=env or os.environ.copy(),
        stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    last_lines: list[str] = []
    for line in proc.stdout:
        if not quiet:
            sys.stdout.write(line)
        last_lines.append(line)
        if len(last_lines) > 50:
            last_lines.pop(0)
    proc.wait()
    if proc.returncode != 0 and quiet:
        sys.stderr.write("---- bootstrap output (últimas líneas) ----\n")
        sys.stderr.writelines(last_lines[-30:])
        sys.stderr.write("\n------------------------------------------\n")
    return proc.returncode

def _apply_overrides(
    base: Dict,
    project: Optional[str],
    region: Optional[str],
    bot: Optional[str],
    kv: Optional[List[str]],
) -> Dict:
    out = dict(base)
    if project: out["project"] = project
    if region:  out["region"] = region
    if bot:     out["bot"] = bot
    for item in (kv or []):
        if "=" not in item:
            print(f"[maws] Ignoro override inválido: {item}")
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # coerción simple: true/false/int/float/JSON; de lo contrario string
        vl = v.lower()
        if v.startswith("{") or v.startswith("["):
            try:
                out[k] = _json.loads(v)
                continue
            except Exception:
                pass
        if vl in ("true","false"):
            out[k] = (vl == "true")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _is_windows() -> bool:
    return _platform.system().lower().startswith("win")

def _windows_guard(allow: bool, action_desc: str):
    """
    Bloquea en Windows (no WSL) salvo que:
      - allow=True, o
      - MAWS_ALLOW_WINDOWS=1, o
      - el usuario confirme interactivo 'y'.
    """
    if not _is_windows():
        return

    if allow or os.environ.get("MAWS_ALLOW_WINDOWS") == "1":
        print(f"⚠️  Ejecutando en Windows (no recomendado) para: {action_desc}. Continuo por override.")
        return

    msg = (
        f"⚠️  Estás ejecutando en Windows. '{action_desc}' suele fallar fuera de WSL.\n"
        "   Recomendado: usar WSL (Ubuntu) o Linux/macOS.\n"
        "   Pasos rápidos WSL:  wsl --install -d Ubuntu   (reiniciar), luego abrir 'Ubuntu' y ejecutar allí.\n"
        "¿Continuar de todos modos? [y/N]: "
    )
    if sys.stdin and sys.stdin.isatty():
        ans = input(msg).strip().lower()
        if ans == "y":
            return

    raise RuntimeError(
        "Abortado para evitar fallos en Windows. Ejecutá en WSL/Linux/macOS, "
        "o forzá con --allow-windows / allow_windows=True / MAWS_ALLOW_WINDOWS=1."
    )


def update(
    params: Optional[Dict] = None,
    config_path: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    project: Optional[str] = None,
    region: Optional[str] = None,
    bot: Optional[str] = None,
    set_kv: Optional[List[str]] = None,
    force_copy_script: bool = False,
    quiet: bool = False,
    allow_windows: bool = False,
) -> int:
    """
    Ejecuta el bootstrap:
      - Si 'params' viene, se usa tal cual (escrito a params.tmp.json).
      - Si no, se lee config_path (o 'params.json' si no se brinda),
        y se aplican overrides (project/region/bot/--set clave=valor).
    """
    _windows_guard(allow_windows, "build/deploy (SAM)")
    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = _ensure_script_in_cwd("bootstrap.sh", workdir, force=force_copy_script)

    if params is not None and config_path is not None:
        raise ValueError("Pasá 'params' (dict) o 'config_path', pero no ambos.")

    tmp_conf_path = None
    if params is None:
        conf_path = workdir / (config_path or "params.json")
        base = {}
        if conf_path.exists():
            try:
                base = _json.loads(conf_path.read_text(encoding="utf-8"))
            except Exception:
                print(f"[maws] No pude leer {conf_path}, uso base vacía.")
        merged = _apply_overrides(base, project, region, bot, set_kv)
        tmp_conf_path = workdir / "params.tmp.json"
        tmp_conf_path.write_text(_json.dumps(merged, indent=2), encoding="utf-8")
        conf = str(tmp_conf_path)
    else:
        merged = _apply_overrides(params, project, region, bot, set_kv)
        tmp_conf_path = workdir / "params.tmp.json"
        tmp_conf_path.write_text(_json.dumps(merged, indent=2), encoding="utf-8")
        conf = str(tmp_conf_path)

    try:
        return _run_bash(script, ["--config", conf], cwd=workdir, quiet=quiet)
    finally:
        if tmp_conf_path and tmp_conf_path.exists():
            try: tmp_conf_path.unlink()
            except Exception: pass

def pull_history(
    config_path: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    force_copy_script: bool = False,
    quiet: bool = False,
) -> int:
    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = _ensure_script_in_cwd("pull_history.sh", workdir, force=force_copy_script)
    conf = config_path or "params.json"
    return _run_bash(script, [conf], cwd=workdir, quiet=quiet)

def _best_effort_install():
    sys_os = _platform.system().lower()
    def _run(cmd: list[str]):
        try:
            _subprocess.check_call(cmd)
            return True
        except Exception:
            return False

    ok = True
    if sys_os == "darwin" and _shutil.which("brew"):
        ok &= _run(["brew", "install", "awscli", "aws-sam-cli", "jq"])
    elif sys_os.startswith("win"):
        if _shutil.which("choco"):
            ok &= _run(["choco", "install", "-y", "awscli"])
            ok &= _run(["choco", "install", "-y", "aws-sam-cli"])
            ok &= _run(["choco", "install", "-y", "jq"])
        else:
            print("⚠️  En Windows instalá WSL o Git Bash y usa los instaladores oficiales:")
            print("    AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
            print("    SAM CLI: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html")
            ok = False
    else:
        # Linux / WSL
        if _shutil.which("apt"):
            ok &= _run(["sudo", "apt-get", "update"])
            ok &= _run(["sudo", "apt-get", "install", "-y", "awscli", "jq"])
        # SAM CLI vía pipx si se puede
        if _shutil.which("pipx"):
            ok &= _run(["pipx", "install", "aws-sam-cli"]) or _run(["pipx", "upgrade", "aws-sam-cli"])
        else:
            print("⚠️  Instalá SAM CLI manualmente si no quedó instalado:")
            print("    https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html")
    return ok

def start(
    project: Optional[str] = None,
    region: Optional[str] = None,
    bot: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    install_deps: bool = False,
    run_config: bool = False,
    allow_windows: bool = False,
) -> Path:
    _windows_guard(allow_windows, "scaffold + instalación de deps")

    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)

    project = project or "my-bot"
    region = region or "us-east-1"
    bot = (bot or "whatsapp").lower()
    if bot not in ("whatsapp", "telegram"):
        raise ValueError("bot debe ser 'whatsapp' o 'telegram'.")

    params_path = workdir / "params.json"

    # Si NO existe, crear con plantilla mínima:
    if not params_path.exists() or overwrite:
        base = {
            "project": project or "my-bot",
            "region": region or "us-east-1",
            "bot": (bot or "whatsapp").lower(),
        }
        params_path.write_text(_json.dumps(base, indent=2), encoding="utf-8")
    else:
        # Existe: si pasaron overrides, mergearlos sin pisar otras claves
        if project or region or bot:
            try:
                data = _json.loads(params_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            if project is not None: data["project"] = project
            if region  is not None: data["region"]  = region
            if bot     is not None: data["bot"]     = (bot or "").lower() or data.get("bot", "whatsapp")
            params_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")

    config_path = workdir / "config.json"
    fns_path = workdir / "fns.py"
    env_path = workdir / ".env.prod"
    gi_path = workdir / ".gitignore"

    def _write(p: Path, content: str):
        if p.exists() and not overwrite:
            return
        p.write_text(content, encoding="utf-8")

    # Scaffold mínimo
    _write(params_path, _json.dumps({"project": project, "region": region, "bot": bot}, indent=2))
    _write(config_path, _json.dumps({
        "agents": [{"id":"assistant","role":"assistant","model":"gpt-4o-mini","instructions":"Sos un asistente útil."}],
        "automations": [{"id":"default","type":"chat","agent":"assistant"}]
    }, indent=2))
    _write(fns_path, "# define tus funciones aquí\n")
    _write(env_path, "# TELEGRAM_TOKEN=xxx\n# WHATSAPP_VERIFY_TOKEN=xxx\n")
    _write(gi_path, ".aws-sam/\n__pycache__/\nhistory/\nfiles/\n.env.prod\n.bootstrap_state.json\n")

    # Copiar scripts si no están
    #_ensure_script_in_cwd("bootstrap.sh", workdir, force=False)
    #_ensure_script_in_cwd("pull_history.sh", workdir, force=False)

    print("\n✅ Estructura creada en:", workdir)
    print("  - params.json")
    print("  - config.json")
    print("  - fns.py")
    print("  - .env.prod (vacío)")
    print("  - .gitignore")

    if install_deps:
        print("\n🔧 Instalando dependencias del sistema (best-effort)…")
        ok = _best_effort_install()
        if not ok:
            print("ℹ️  Seguí las instrucciones mostradas arriba para completar la instalación si quedó algo pendiente.")

    # Probar credenciales
    if _shutil.which("aws"):
        try:
            out = _subprocess.check_output(["aws", "sts", "get-caller-identity", "--output", "json"], text=True)
            acct = _json.loads(out).get("Account")
            print(f"🔐 AWS listo. Account: {acct}")
        except Exception:
            print("🔐 Configurá credenciales:  aws configure  (o usá AWS_PROFILE)")

        if run_config:
            print("▶️  Ejecutando 'aws configure'…")
            try:
                _subprocess.call(["aws", "configure"])
            except Exception:
                print("No pude correr 'aws configure' automáticamente. Corrélo manualmente cuando puedas.")

    else:
        print("\n⚠️  No encontré 'aws'. Instalá AWS CLI y configuralo con 'aws configure'.")

        # Avisos no intrusivos si faltan herramientas clave
    if not _shutil.which("sam"):
        print("⚠️  No encontré 'sam' (AWS SAM CLI). Podés instalarlo o correr 'maws start --install-deps'.")

    if not _shutil.which("jq"):
        print("⚠️  No encontré 'jq'. Es útil para el bootstrap; instalalo si ves errores relacionados.")


    print("\n👉 Próximos pasos:")
    print("   1) Completá .env.prod con tus tokens.")
    print("   2) (Opcional) Ajustá params.json, config.json y fns.py a gusto.")
    print("   3) Deploy:   maws update   (o desde Python: maws.update())")

    return workdir
