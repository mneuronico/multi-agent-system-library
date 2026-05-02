from __future__ import annotations
import sys
import os
import json
import json as _json
import tempfile as _tempfile
import subprocess as _subprocess
import shutil as _shutil
import platform as _platform
import datetime as _dt
from pathlib import Path
from typing import Optional, Union, List, Dict

from .runtime import _require_boto3

try:
    from importlib.resources import files as _res_files
except Exception:  # pragma: no cover
    import importlib_resources  # type: ignore
    _res_files = importlib_resources.files  # type: ignore

def _resource_path(name: str) -> Path:
    return Path(_res_files("maws.resources") / name)

def _ensure_script_in_cwd(name: str, dest_dir: Path, force: bool = False) -> Path:
    """
    Return a path to the script. By default it uses the embedded copy inside the package.
    If ``force`` is True or MAWS_COPY_SCRIPTS=1, copy the script into the project and return that path.
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
                "Could not find 'bash'. On Windows install WSL or Git Bash.\n"
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
        sys.stderr.write("---- bootstrap output (last lines) ----\n")
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
            print(f"[maws] Ignoring invalid override: {item}")
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # simple coercion: true/false/int/float/JSON; otherwise string
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
    Block execution on Windows (outside WSL) unless one of the following is true:
      - ``allow`` is True,
      - MAWS_ALLOW_WINDOWS=1 is set,
      - the user confirms interactively with 'y'.
    """
    if not _is_windows():
        return

    if allow or os.environ.get("MAWS_ALLOW_WINDOWS") == "1":
        print(f"[WARN] Running on Windows (not recommended) for: {action_desc}. Proceeding due to override.")
        return

    msg = (
        f"[WARN] You are running on Windows. '{action_desc}' often fails outside WSL.\n"
        "   Recommended: use WSL (Ubuntu) or Linux/macOS.\n"
        "   Quick WSL steps:  wsl --install -d Ubuntu   (reboot), then open 'Ubuntu' and run it there.\n"
        "Continue anyway? [y/N]: "
    )
    if sys.stdin and sys.stdin.isatty():
        ans = input(msg).strip().lower()
        if ans == "y":
            return

    raise RuntimeError(
        "Aborted to avoid failures on Windows. Use WSL/Linux/macOS, "
        "or force with --allow-windows / allow_windows=True / MAWS_ALLOW_WINDOWS=1."
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
    Run the bootstrap:
      - If ``params`` is provided, use it as-is (written to params.tmp.json).
      - Otherwise read config_path (or 'params.json' by default) and apply overrides
        (project/region/bot/--set key=value).
    """
    _windows_guard(allow_windows, "build/deploy (SAM)")
    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = _ensure_script_in_cwd("bootstrap.sh", workdir, force=force_copy_script)

    if params is not None and config_path is not None:
        raise ValueError("Pass either 'params' (dict) or 'config_path', but not both.")

    tmp_conf_path = None
    if params is None:
        conf_path = workdir / (config_path or "params.json")
        base = {}
        if conf_path.exists():
            try:
                base = _json.loads(conf_path.read_text(encoding="utf-8"))
            except Exception:
                print(f"[maws] Could not read {conf_path}, using empty base.")
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
    user_ids: Optional[Union[str, List[str]]] = None,
) -> int:
    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = _ensure_script_in_cwd("pull_history.sh", workdir, force=force_copy_script)
    conf = config_path or "params.json"
    args = ["--config", conf]
    for user_id in _normalize_history_user_ids(user_ids):
        args.extend(["--user-id", user_id])
    return _run_bash(script, args, cwd=workdir, quiet=quiet)

def _normalize_history_user_ids(user_ids: Optional[Union[str, List[str]]]) -> List[str]:
    if user_ids is None:
        return []

    raw_items = user_ids if isinstance(user_ids, list) else [user_ids]
    normalized = []
    seen = set()
    for item in raw_items:
        if item is None:
            continue
        for part in str(item).split(","):
            user_id = part.strip()
            if not user_id or user_id in seen:
                continue
            normalized.append(user_id)
            seen.add(user_id)
    return normalized

def _best_effort_install():
    """
    Linux/WSL only:
      - APT for curl, unzip, jq
      - AWS CLI v2 via official installer (zip)
      - SAM CLI via official installer (zip)
    Returns True if everything succeeded or was already installed.
    """
    import tempfile
    sys_os = _platform.system().lower()
    if sys_os != "linux":
        print("[WARN] Assisted installation supported only on Linux/WSL. On other OSes, install manually.")
        return False

    def _run(cmd):
        try:
            _subprocess.check_call(cmd)
            return True
        except Exception:
            return False

    ok = True

    # Base tools
    if _shutil.which("apt"):
        _run(["sudo", "apt-get", "update"])
        if not _shutil.which("curl"):
            ok &= _run(["sudo", "apt-get", "install", "-y", "curl"])
        if not _shutil.which("unzip"):
            ok &= _run(["sudo", "apt-get", "install", "-y", "unzip"])
        if not _shutil.which("jq"):
            ok &= _run(["sudo", "apt-get", "install", "-y", "jq"])
    else:
        print("[WARN] APT not found. Install curl/unzip/jq manually.")
        ok = False

    # AWS CLI v2 (official installer)
    if not _shutil.which("aws"):
        tmp = tempfile.mkdtemp()
        arch = _platform.machine().lower()
        if "aarch64" in arch or "arm64" in arch:
            aws_url = "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
        else:
            aws_url = "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
        ok &= _run(["curl", "-sSL", aws_url, "-o", f"{tmp}/awscliv2.zip"])
        ok &= _run(["unzip", "-q", f"{tmp}/awscliv2.zip", "-d", tmp])
        ok &= _run(["sudo", f"{tmp}/aws/install", "--update"])
        if _shutil.which("aws"):
            try:
                out = _subprocess.check_output(["aws", "--version"], text=True)
                print(f"[OK] AWS CLI ready: {out.strip()}")
            except Exception:
                pass
        else:
            print("[ERR] Could not install AWS CLI. Install manually: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
            ok = False

    if not _shutil.which("sam"):
        tmp = tempfile.mkdtemp()
        arch = _platform.machine().lower()
        if "aarch64" in arch or "arm64" in arch:
            sam_url = "https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-aarch64.zip"
        else:
            sam_url = "https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip"
        ok &= _run(["curl", "-sSL", sam_url, "-o", f"{tmp}/aws-sam-cli.zip"])
        ok &= _run(["unzip", "-q", f"{tmp}/aws-sam-cli.zip", "-d", f"{tmp}/sam-installation"])
        # key detail: use --update for pre-existing installations
        ok &= _run(["sudo", f"{tmp}/sam-installation/install", "--update"])

        # If the installer detected a previous installation but did not leave the symlink:
        sam_bin = _shutil.which("sam")
        if not sam_bin and os.path.exists("/usr/local/aws-sam-cli/current/bin/sam"):
            _run(["sudo", "ln", "-sf", "/usr/local/aws-sam-cli/current/bin/sam", "/usr/local/bin/sam"])
            _subprocess.call(["hash", "-r"])  # does not fail if the shell is not bash
            sam_bin = _shutil.which("sam")

        if sam_bin:
            try:
                out = _subprocess.check_output(["sam", "--version"], text=True)
                print(f"[OK] SAM CLI ready: {out.strip()}")
            except Exception:
                pass
        else:
            print("[ERR] Could not install SAM CLI. Install manually: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html")
            ok = False


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
    _windows_guard(allow_windows, "scaffold + dependency installation")

    workdir = Path(project_dir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)

    # Prompt helpers (safe when there is no TTY)
    def _is_tty():
        return bool(getattr(sys, "stdin", None) and sys.stdin.isatty())

    def _ask(prompt, default=None):
        if not _is_tty():
            return default or ""
        sfx = f" [{default}]" if default else ""
        resp = input(f"{prompt}{sfx}: ").strip()
        return resp or (default or "")

    def _ask_choice(prompt, choices, default):
        choices_str = "/".join(choices)
        val = _ask(f"{prompt} ({choices_str})", default).lower()
        if val not in choices:
            print(f"[maws] Invalid option '{val}', using '{default}'.")
            val = default
        return val

    # Ask only when arguments were not provided; provide friendly defaults
    if project is None:
        project = _ask("Project name", "my-bot")
    if region is None:
        region = _ask("AWS region", "us-east-1")
    if bot is None:
        bot = _ask_choice("Bot type", ["telegram", "whatsapp"], "telegram")

    bot = bot.lower()
    if bot not in ("whatsapp", "telegram"):
        raise ValueError("bot must be 'whatsapp' or 'telegram'.")

    params_path = workdir / "params.json"

    # If it does NOT exist, create with a minimal template:
    if not params_path.exists() or overwrite:
        base = {
            "project": project or "my-bot",
            "region": region or "us-east-1",
            "bot": (bot or "whatsapp").lower(),
        }
        params_path.write_text(_json.dumps(base, indent=2), encoding="utf-8")
    else:
        # Exists: if overrides were provided, merge them without clobbering other keys
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
    gi_path = workdir / ".samignore"

    def _write(p: Path, content: str):
        if p.exists() and not overwrite:
            return
        p.write_text(content, encoding="utf-8")

    # Minimal scaffold

    def _is_tty() -> bool:
        return bool(getattr(sys, "stdin", None) and sys.stdin.isatty())

    def _ask(prompt: str, default: Optional[str] = None) -> str:
        if not _is_tty():
            return default or ""
        sfx = f" [{default}]" if default else ""
        resp = input(f"{prompt}{sfx}: ").strip()
        return resp or (default or "")

    # 2) Choose provider/model (interactive when running on a TTY), defaults: google / gemini-2.5-flash
    _default_models_map = {
        "google": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-5-nano",
        "groq":   "llama-3.1-70b-versatile",
    }
    provider = _ask("Default provider (google/openai/openrouter/groq)", "google").lower()
    if provider not in _default_models_map:
        print(f"[maws] Unknown provider '{provider}', using 'google'.")
        provider = "google"
    model = _ask("Default model", _default_models_map[provider]) or _default_models_map[provider]

    # 3) config.json with default_models
    config_template = {
        "general_parameters": {
            "general_system_description": "A simple query system.",
            "default_models": [
                {"provider": provider, "model": model}
            ]
        },
        "components": [
            {
                "type": "agent",
                "name": "simple_agent",
                "system": "You are a basic assistant for answering questions.",
                "required_outputs": {
                    "response": "A text response to be sent to the user."
                }
            }
        ]
    }
    _write(config_path, _json.dumps(config_template, indent=2))

    # 4) Empty fns.py (if it does not exist or overwrite=True)
    _write(fns_path, "# define your functions here\n")

    entered = {
        "google": "",
        "openai": "",
        "openrouter": "",
        "groq": "",
        "telegram_token": "",
        "whatsapp_token": "",
        "whatsapp_phone_id": "",
        "whatsapp_verify": "",
    }

    # Ask for the selected provider key (optional)
    provider_key_env = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq":   "GROQ_API_KEY",
    }[provider]
    maybe_key = _ask(f"Paste {provider_key_env} (press Enter to skip)")
    if maybe_key:
        entered[provider] = maybe_key

    # Ask for bot tokens
    if bot == "telegram":
        tk = _ask("Paste TELEGRAM_TOKEN (press Enter to skip)")
        if tk:
            entered["telegram_token"] = tk
    else:
        wtk = _ask("Paste WHATSAPP_TOKEN (press Enter to skip)")
        if wtk:
            entered["whatsapp_token"] = wtk
        wid = _ask("Paste WHATSAPP_PHONE_NUMBER_ID (press Enter to skip)")
        if wid:
            entered["whatsapp_phone_id"] = wid
        wv  = _ask("Paste WHATSAPP_VERIFY_TOKEN (press Enter to skip)")
        if wv:
            entered["whatsapp_verify"] = wv

    # Build .env.prod lines
    def _line(envname: str, value: str, commented_if_empty: bool = True) -> str:
        if value:
            return f"{envname}={value}"
        return f"# {envname}=" if commented_if_empty else f"{envname}="

    env_lines: list[str] = []
    # Always present (commented if empty). The selected one is uncommented
    env_lines.append(_line("GOOGLE_API_KEY", entered["google"]))
    env_lines.append(_line("OPENAI_API_KEY", entered["openai"]))
    env_lines.append(_line("OPENROUTER_API_KEY", entered["openrouter"]))
    env_lines.append(_line("GROQ_API_KEY",   entered["groq"]))

    if bot == "telegram":
        env_lines.append(_line("TELEGRAM_TOKEN", entered["telegram_token"]))
        # If you want to keep this generic, leave it commented:
        env_lines.append("# WEBHOOK_VERIFY_TOKEN=")
        env_lines.append("# WHATSAPP_TOKEN=")
        env_lines.append("# WHATSAPP_PHONE_NUMBER_ID=")
        env_lines.append("# WHATSAPP_VERIFY_TOKEN=")
    else:
        env_lines.append(_line("WHATSAPP_TOKEN",           entered["whatsapp_token"]))
        env_lines.append(_line("WHATSAPP_PHONE_NUMBER_ID", entered["whatsapp_phone_id"]))
        env_lines.append(_line("WHATSAPP_VERIFY_TOKEN",    entered["whatsapp_verify"]))
        env_lines.append("# TELEGRAM_TOKEN=")
        env_lines.append("# WEBHOOK_VERIFY_TOKEN=")

    _write(env_path, "\n".join(env_lines) + "\n")

    # 5) .gitignore
    _write(gi_path, ".aws-sam/\n__pycache__/\nhistory/\nfiles/\n.env.prod\n.bootstrap_state.json\n")

    # Copy scripts if they are missing
    #_ensure_script_in_cwd("bootstrap.sh", workdir, force=False)
    #_ensure_script_in_cwd("pull_history.sh", workdir, force=False)

    print("\n[OK] Structure created at:", workdir)
    print("  - params.json")
    print("  - config.json")
    print("  - fns.py")
    print("  - .env.prod")
    print("  - .samignore")

    if install_deps:
        print("\n[INFO] Installing system dependencies (best effort)...")
        ok = _best_effort_install()
        if not ok:
            print("[INFO] Follow the instructions shown above to finish any pending installation steps.")

    # Probar credenciales
    if _shutil.which("aws"):
        try:
            out = _subprocess.check_output(["aws", "sts", "get-caller-identity", "--output", "json"], text=True)
            acct = _json.loads(out).get("Account")
            print(f"[OK] AWS ready. Account: {acct}")
        except Exception:
            print("[INFO] Configure credentials:  aws configure  (or use AWS_PROFILE)")

        if run_config:
            print("[INFO] Running 'aws configure'...")
            try:
                _subprocess.call(["aws", "configure"])
            except Exception:
                print("Could not run 'aws configure' automatically. Please run it manually when you can.")

    else:
        print("\n[WARN] 'aws' not found. Install AWS CLI and configure it with 'aws configure'.")

        # Avisos no intrusivos si faltan herramientas clave
    if not _shutil.which("sam"):
        print("[WARN] 'sam' (AWS SAM CLI) not found. Install it or run 'maws start --install-deps'.")

    if not _shutil.which("jq"):
        print("[WARN] 'jq' not found. It is useful for the bootstrap; install it if you encounter related errors.")


    print("\nNext steps:")
    print("   1) Complete .env.prod with your tokens.")
    print("   2) (Optional) Adjust params.json, config.json, and fns.py as needed.")
    print("   3) Deploy:   maws update   (or from Python: maws.update())")

    return workdir


# --- NUEVO: helpers comunes para prompts/AWS/params ---
def _is_tty() -> bool:
    return bool(getattr(sys, "stdin", None) and sys.stdin.isatty())

def _ask(prompt: str, default: Optional[str] = None) -> str:
    if not _is_tty():
        return default or ""
    sfx = f" [{default}]" if default else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return val or (default or "")

def _ask_bool(prompt: str, default: bool = True) -> bool:
    if not _is_tty():
        return default
    sfx = "Y/n" if default else "y/N"
    val = input(f"{prompt} ({sfx}): ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true")

def _effective_region(explicit: Optional[str], params: Dict) -> str:
    return (
        explicit
        or params.get("region")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

def _load_params(conf_path: Path) -> Dict:
    try:
        if conf_path.exists():
            return json.loads(conf_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_params(conf_path: Path, data: Dict):
    conf_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _boto(region: Optional[str] = None):
    """Return boto3 clients forcing the region when provided."""
    kw = {"region_name": region} if region else {}
    boto = _require_boto3()
    return (
        boto.client("cloudformation", **kw),
        boto.client("s3", **kw),
        boto.client("lambda", **kw),  # kept in case it is needed later
    )

def setup(
    config_path: Optional[Union[str, Path]] = None,
    project_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Interactive guide to complete/update params.json with safe defaults.
    Leaves other files untouched.
    """
    workdir = Path(project_dir or Path.cwd())
    conf = workdir / (config_path or "params.json")
    data = _load_params(conf)

    # current defaults (if present) or reasonable fallbacks
    d_project = data.get("project", "my-bot")
    d_region  = data.get("region", "us-east-1")
    d_bot     = (data.get("bot") or "telegram").lower()
    if d_bot not in ("telegram", "whatsapp"):
        d_bot = "telegram"

    # prompts
    project = _ask("Project name", d_project) or d_project
    region  = _ask("AWS region", d_region) or d_region
    bot     = _ask("Bot type (telegram/whatsapp)", d_bot) or d_bot
    bot     = bot.lower() if bot in ("telegram", "whatsapp") else d_bot

    stack_name = _ask("CloudFormation stack_name", data.get("stack_name") or f"{project}-stack")
    history_bucket = _ask("S3 history_bucket", data.get("history_bucket") or f"{project}-history-bucket")
    deploy_bucket  = _ask("S3 deployment_bucket", data.get("deployment_bucket") or f"{project}-deployment-bucket")
    env_param_name = _ask("SSM env_param_name", data.get("env_param_name") or f"/{project}/prod/env")
    api_path       = _ask("API path", data.get("api_path") or "/webhook")

    lambda_timeout = _ask("Lambda function timeout (in seconds)", data.get("lambda_timeout") or "120")

    sync_tokens_s3 = _ask_bool("Sync tokens from S3 (SYNC_TOKENS_S3)", bool(data.get("sync_tokens_s3", True)))
    tokens_prefix  = _ask("S3 prefix for tokens (TOKENS_S3_PREFIX)",
                          data.get("tokens_s3_prefix") or "secrets")
    verbose        = _ask_bool("Verbose mode in Lambda (VERBOSE)", bool(data.get("verbose", False)))

    # update minimal structure
    data.update({
        "project": project,
        "region": region,
        "bot": bot,
        "stack_name": stack_name,
        "history_bucket": history_bucket,
        "deployment_bucket": deploy_bucket,
        "env_param_name": env_param_name,
        "api_path": api_path,
        "lambda_timeout": int(lambda_timeout),
        "sync_tokens_s3": bool(sync_tokens_s3),
        "tokens_s3_prefix": tokens_prefix,
        "verbose": bool(verbose),
    })

    _save_params(conf, data)
    print(f"[OK] Params written to {conf}")
    return conf

# --- NUEVO: maws describe ---
def describe(
    config_path: Optional[Union[str, Path]] = None,
    project_dir: Optional[Union[str, Path]] = None,
    region: Optional[str] = None,
    no_aws: bool = False,
) -> int:
    """
    Describe the current project:
      - Show local files
      - Read params.json
      - If credentials are available, query CloudFormation for outputs (ApiUrl)
    """
    workdir = Path(project_dir or Path.cwd())
    conf = workdir / (config_path or "params.json")
    data = _load_params(conf)

    print(f"\nProject at: {workdir}")
    print(f"   - params.json: {'OK' if conf.exists() else 'NO'}")
    print(f"   - config.json: {'OK' if (workdir/'config.json').exists() else 'NO'}")
    print(f"   - fns.py     : {'OK' if (workdir/'fns.py').exists() else 'NO'}")
    print(f"   - .env.prod  : {'OK' if (workdir/'.env.prod').exists() else 'NO'}")

    if not data:
        print("\n[WARN] Could not find a readable params.json. Run 'maws setup' or 'maws start'.")
        return 1

    project = data.get("project", "my-bot")
    reg = _effective_region(region, data)
    stack = data.get("stack_name") or f"{project}-stack"
    history_bucket = data.get("history_bucket") or f"{project}-history-bucket"
    deploy_bucket  = data.get("deployment_bucket") or f"{project}-deployment-bucket"

    print("\nparams.json")
    for k in ("project","region","bot","stack_name","history_bucket","deployment_bucket","env_param_name","api_path","sync_tokens_s3","tokens_s3_prefix","verbose"):
        if k in data:
            print(f"   - {k}: {data[k]}")

    if no_aws:
        return 0

    # AWS info (best-effort)
    try:
        cf, s3, _ = _boto(reg)
        ds = cf.describe_stacks(StackName=stack)
        st = ds["Stacks"][0]
        status = st.get("StackStatus")
        updated = st.get("LastUpdatedTime") or st.get("CreationTime")
        updated_s = updated.strftime("%Y-%m-%d %H:%M:%S") if isinstance(updated, _dt.datetime) else str(updated)
        outputs = {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}
        api_url = outputs.get("ApiUrl")

        print(f"\nCloudFormation [{reg}]")
        print(f"   - Stack: {stack}  ({status}, {updated_s})")
        if api_url:
            print(f"   - ApiUrl: {api_url}")

        # bucket existence
        def _bucket_exists(name: str) -> bool:
            try:
                s3.head_bucket(Bucket=name)
                return True
            except Exception:
                return False

        print("\nBuckets")
        print(f"   - history_bucket    : {history_bucket}  ({'exists' if _bucket_exists(history_bucket) else 'missing'})")
        print(f"   - deployment_bucket : {deploy_bucket}  ({'exists' if _bucket_exists(deploy_bucket) else 'missing'})")
    except Exception as e:
        print(f"\n[INFO] AWS unavailable or missing permissions: {e}")

    return 0

# --- NUEVO: maws list (stacks) ---
def list_projects(region: Optional[str] = None) -> List[Dict]:
    """
    List CloudFormation stacks in the region (filtering those that expose the 'ApiUrl' output).
    Returns a list of dicts {stack, status, updated, api_url, region} and prints a simple table.
    """
    reg = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    cf, *_ = _boto(reg)

    wanted = [
        "CREATE_COMPLETE","UPDATE_COMPLETE","UPDATE_ROLLBACK_COMPLETE",
        "ROLLBACK_COMPLETE","IMPORT_COMPLETE"
    ]
    out_rows: List[Dict] = []

    try:
        paginator = cf.get_paginator("list_stacks")
        for page in paginator.paginate(StackStatusFilter=wanted):
            for s in page.get("StackSummaries", []):
                name = s.get("StackName")
                try:
                    ds = cf.describe_stacks(StackName=name)
                    st = ds["Stacks"][0]
                    outputs = {o["OutputKey"]: o["OutputValue"] for o in st.get("Outputs", [])}
                    api_url = outputs.get("ApiUrl")
                    if not api_url:
                        continue  # likely not a MAWS stack
                    updated = st.get("LastUpdatedTime") or st.get("CreationTime")
                    out_rows.append({
                        "stack": name,
                        "status": st.get("StackStatus"),
                        "updated": updated.isoformat() if isinstance(updated, _dt.datetime) else str(updated),
                        "api_url": api_url,
                        "region": reg,
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"[INFO] Could not list stacks in {reg}: {e}")
        return []

    # print simple table
    if out_rows:
        print("\nMAWS stacks in", reg)
        print("".ljust(80, "-"))
        print(f"{'STACK':35}  {'STATUS':22}  {'UPDATED':19}  {'API URL'}")
        print("".ljust(80, "-"))
        for r in out_rows:
            print(f"{r['stack'][:35]:35}  {r['status'][:22]:22}  {r['updated'][:19]:19}  {r['api_url']}")
        print("".ljust(80, "-"))
    else:
        print(f"\n(no MAWS stacks detected in {reg})")

    return out_rows

# --- NUEVO: remove_project ---
def _empty_bucket(s3, bucket: str):
    """Delete objects and versions (if versioning is enabled) to allow bucket/stack removal."""
    try:
        # versiones y delete-markers
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            to_del = []
            for v in page.get("Versions", []):
                to_del.append({"Key": v["Key"], "VersionId": v["VersionId"]})
            for m in page.get("DeleteMarkers", []):
                to_del.append({"Key": m["Key"], "VersionId": m["VersionId"]})
            if to_del:
                for chunk in [to_del[i:i+1000] for i in range(0, len(to_del), 1000)]:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
        # objetos actuales (por si no hay versioning)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            keys = [{"Key": it["Key"]} for it in page.get("Contents", [])]
            if keys:
                for chunk in [keys[i:i+1000] for i in range(0, len(keys), 1000)]:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
    except s3.exceptions.NoSuchBucket:
        return
    except Exception as e:
        print(f"[WARN] Could not empty bucket {bucket}: {e}")

def remove_project(
    project: Optional[str] = None,
    region: Optional[str] = None,
    yes: bool = False,
    keep_deploy_bucket: bool = False,
    config_path: Optional[Union[str, Path]] = None,
    project_dir: Optional[Union[str, Path]] = None,
    wait: bool = False,
) -> int:
    """
    Delete AWS resources for the project:
      - Empty the history bucket
      - Remove the CloudFormation stack (API/Lambda/DDB, etc.)
      - Optionally delete the deployment bucket (if keep_deploy_bucket is False)
    Pulls names from params.json when not provided explicitly.
    """
    workdir = Path(project_dir or Path.cwd())
    conf = workdir / (config_path or "params.json")
    data = _load_params(conf)

    proj = project or data.get("project")
    if not proj:
        print("[ERR] Could not determine 'project'. Pass it with --project or set it in params.json.")
        return 2

    reg = _effective_region(region, data)
    stack = data.get("stack_name") or f"{proj}-stack"
    history_bucket = data.get("history_bucket") or f"{proj}-history-bucket"
    deploy_bucket  = data.get("deployment_bucket") or f"{proj}-deployment-bucket"

    print("\nDeletion plan")
    print(f"   Region:           {reg}")
    print(f"   Stack to delete:  {stack}")
    print(f"   Empty bucket:     {history_bucket}")
    if not keep_deploy_bucket:
        print(f"   Delete bucket:    {deploy_bucket} (deployment)")
    else:
        print(f"   Keep bucket:      {deploy_bucket} (deployment)")

    if not yes and not _ask_bool("Continue?", False):
        print("Aborted.")
        return 1

    cf, s3, _ = _boto(reg)

    # Empty history bucket (so CloudFormation can delete it)
    print(f"[INFO] Emptying s3://{history_bucket} ...")
    _empty_bucket(s3, history_bucket)

    # Delete stack
    print(f"[INFO] Deleting stack {stack} ...")
    try:
        cf.delete_stack(StackName=stack)
        if wait:
            waiter = cf.get_waiter("stack_delete_complete")
            print("[INFO] Waiting for stack deletion to finish...")
            waiter.wait(StackName=stack)
            print("[OK] Stack deleted.")
        else:
            print("[INFO] Deletion started (asynchronous).")
    except Exception as e:
        print(f"[WARN] Error deleting stack: {e}")

    # Deployment bucket (outside the stack) - delete if requested
    if not keep_deploy_bucket:
        print(f"[INFO] Emptying s3://{deploy_bucket} ...")
        _empty_bucket(s3, deploy_bucket)
        try:
            s3.delete_bucket(Bucket=deploy_bucket)
            print("[OK] Deployment bucket deleted.")
        except Exception as e:
            print(f"[WARN] Could not delete deployment bucket: {e}")

    print("[OK] Done. Some resources may take a few minutes to disappear completely.")
    return 0
