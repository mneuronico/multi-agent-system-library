from __future__ import annotations
import sys
import os
import copy as _copy
import json
import json as _json
import re as _re
import secrets as _secrets
import tempfile as _tempfile
import subprocess as _subprocess
import shutil as _shutil
import platform as _platform
import datetime as _dt
from pathlib import Path
from typing import Any, Optional, Union, List, Dict

from .runtime import _require_boto3

try:
    from importlib.resources import files as _res_files
except Exception:  # pragma: no cover
    import importlib_resources  # type: ignore
    _res_files = importlib_resources.files  # type: ignore

# Default Lambda runtime; keep in sync with infra.runtime's default in resources/bootstrap.sh.
DEFAULT_LAMBDA_RUNTIME = "python3.12"


class MawsConfigError(ValueError):
    """params.json or a --set override is invalid. The CLI prints it and exits with code 2."""


def _new_telegram_webhook_secret() -> str:
    """Random secret_token for Telegram setWebhook (Telegram accepts 1-256 chars of A-Za-z0-9_-)."""
    return _secrets.token_urlsafe(32)


def _legacy_bucket_names(project: str) -> Dict[str, str]:
    """
    Bucket names bootstrap.sh derives when params.json does not set them. Existing deployments use
    these names, so they must never change (a different history bucket means losing the history).
    """
    return {
        "history_bucket": f"{project}-history-bucket",
        "deployment_bucket": f"{project}-deployment-bucket",
    }


def _unique_bucket_name(project: str, kind: str) -> str:
    """
    A hard-to-guess S3 bucket name for a new project, e.g. ``my-bot-history-3f9a1c2e``. S3 names
    are global, so predictable names can be claimed by another account first.
    """
    suffix = _secrets.token_hex(4)
    base = _re.sub(r"[^a-z0-9-]+", "-", str(project).lower()).strip("-") or "maws"
    if base.startswith(("xn--", "sthree-")):
        base = "maws-" + base
    room = 63 - len(kind) - len(suffix) - 2
    base = base[:room].rstrip("-") or "maws"
    return f"{base}-{kind}-{suffix}"


def _new_project_bucket_names(project: str) -> Dict[str, str]:
    return {
        "history_bucket": _unique_bucket_name(project, "history"),
        "deployment_bucket": _unique_bucket_name(project, "deploy"),
    }


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

class _NonFiniteJsonNumber(ValueError):
    pass


def _reject_json_constant(name: str):
    raise _NonFiniteJsonNumber(name)


def _parse_override_value(key: str, raw: str) -> Any:
    """
    Parse the value of ``--set key=value``. It stays a string unless it is a JSON literal that
    round-trips exactly: ``true``/``false``/``null``, ``120``, ``1.5``, ``"quoted text"``, or a JSON
    object/array. So ``1.10``, ``0123456``, ``1e3`` and ``nan`` stay strings. NaN/Infinity would
    produce invalid JSON for bootstrap.sh and are refused.
    """
    try:
        parsed = _json.loads(raw, parse_constant=_reject_json_constant)
    except _NonFiniteJsonNumber:
        raise MawsConfigError(
            f"--set {key}={raw}: NaN/Infinity are not valid JSON numbers. "
            f"Quote the value to pass it as text: --set '{key}=\"{raw}\"'."
        ) from None
    except ValueError:
        return raw
    if isinstance(parsed, (dict, list, str)):
        return parsed
    if _json.dumps(parsed) == raw:
        return parsed
    return raw


def _set_override(out: Dict, key: str, value: Any):
    """``infra.tracing`` sets out["infra"]["tracing"], creating intermediate objects."""
    parts = [part.strip() for part in key.split(".")]
    if not all(parts):
        raise MawsConfigError(f"--set {key}=...: invalid key.")
    target = out
    for depth, part in enumerate(parts[:-1]):
        current = target.get(part)
        if current is None:
            current = target[part] = {}
        elif not isinstance(current, dict):
            path = ".".join(parts[: depth + 1])
            raise MawsConfigError(f"--set {key}=...: {path} is {current!r}, not an object.")
        target = current
    target[parts[-1]] = value


def _apply_overrides(
    base: Dict,
    project: Optional[str],
    region: Optional[str],
    bot: Optional[str],
    kv: Optional[List[str]],
) -> Dict:
    out = _copy.deepcopy(base)
    if project: out["project"] = project
    if region:  out["region"] = region
    if bot:     out["bot"] = bot
    for item in (kv or []):
        if "=" not in item:
            raise MawsConfigError(f"Invalid override {item!r}: use --set key=value (dotted keys for nested params, e.g. infra.tracing=false).")
        k, v = item.split("=", 1)
        _set_override(out, k.strip(), _parse_override_value(k.strip(), v.strip()))
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

    if params is None:
        # A params.json that does not parse is an error, never an empty config: deploying {} would
        # derive new default names and could create an empty history bucket.
        base = _load_params(workdir / (config_path or "params.json"))
    else:
        base = params
    merged = _apply_overrides(base, project, region, bot, set_kv)
    try:
        rendered = _json.dumps(merged, indent=2, allow_nan=False)
    except ValueError as e:
        raise MawsConfigError(f"params contain a value that is not valid JSON (NaN/Infinity): {e}") from None
    tmp_conf_path = workdir / "params.tmp.json"
    tmp_conf_path.write_text(rendered, encoding="utf-8")
    conf = tmp_conf_path.name

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

    params_path = workdir / "params.json"
    # An existing params.json is the source of truth (and must parse: never treat it as {}).
    existing_params = _load_params(params_path) if params_path.exists() else None
    keep_existing = existing_params is not None and not overwrite
    current = existing_params or {}
    fallback = {"project": "my-bot", "region": "us-east-1", "bot": "telegram"}

    def _choose(key: str, explicit: Optional[str], ask) -> str:
        value_now = current.get(key)
        if keep_existing and value_now not in (None, ""):
            if explicit is not None and str(explicit).strip().lower() != str(value_now).strip().lower():
                print(
                    f"[WARN] params.json already sets {key}={value_now!r}; keeping it. "
                    f"Pass --overwrite to replace it with {explicit!r}."
                )
            return str(value_now)
        if explicit is not None:
            return explicit
        # Re-running with --overwrite offers the current values as defaults.
        return ask(str(value_now or fallback[key]))

    project = _choose("project", project, lambda d: _ask("Project name", d))
    region = _choose("region", region, lambda d: _ask("AWS region", d))
    bot = _choose(
        "bot", bot,
        lambda d: _ask_choice("Bot type", ["telegram", "whatsapp"], d if d in ("telegram", "whatsapp") else "telegram"),
    )

    bot = bot.lower()
    if bot not in ("whatsapp", "telegram"):
        raise ValueError("bot must be 'whatsapp' or 'telegram'.")

    if existing_params is None:
        # New project: explicit, hard-to-guess bucket names instead of the predictable defaults.
        base = {"project": project, "region": region, "bot": bot, **_new_project_bucket_names(project)}
        params_path.write_text(_json.dumps(base, indent=2), encoding="utf-8")
    else:
        # Existing project: keep every other key (bucket names, stack name, ...). Without
        # --overwrite only keys that are missing are filled in.
        data = dict(existing_params)
        for key, value in (("project", project), ("region", region), ("bot", bot)):
            data[key] = value
        if data != existing_params:
            params_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        if overwrite and existing_params.get("project") not in (None, project):
            print(
                f"[WARN] project changed from {existing_params.get('project')!r} to {project!r}. Stack, bucket and SSM "
                "names that params.json does not set explicitly are derived from the project name, so the next "
                "maws update deploys a separate stack (with empty histories)."
            )

    config_path = workdir / "config.json"
    fns_path = workdir / "fns.py"
    env_path = workdir / ".env.prod"
    gi_path = workdir / ".gitignore"

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
    # Only ask what will be written: an existing config.json / .env.prod is kept unless --overwrite.
    config_will_be_written = overwrite or not config_path.exists()
    env_will_be_written = overwrite or not env_path.exists()
    provider = "google"
    if config_will_be_written or env_will_be_written:
        provider = _ask("Default provider (google/openai/openrouter/groq)", "google").lower()
    if provider not in _default_models_map:
        print(f"[maws] Unknown provider '{provider}', using 'google'.")
        provider = "google"
    model = _default_models_map[provider]
    if config_will_be_written:
        model = _ask("Default model", model) or model

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
        "whatsapp_app_secret": "",
    }

    # Ask for the selected provider key (optional)
    provider_key_env = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq":   "GROQ_API_KEY",
    }[provider]
    maybe_key = _ask(f"Paste {provider_key_env} (press Enter to skip)") if env_will_be_written else ""
    if maybe_key:
        entered[provider] = maybe_key

    # Ask for bot tokens (only when .env.prod will be written)
    if not env_will_be_written:
        pass
    elif bot == "telegram":
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
        was = _ask("Paste WHATSAPP_APP_SECRET (Meta App Dashboard > App settings > Basic; press Enter to skip)")
        if was:
            entered["whatsapp_app_secret"] = was

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
        env_lines.append("# Webhook secret (generated by maws start). Telegram sends it with every update;")
        env_lines.append("# MAWS rejects webhook POSTs without it. Letters, digits, _ and - only.")
        env_lines.append(f"TELEGRAM_WEBHOOK_SECRET_TOKEN={_new_telegram_webhook_secret()}")
        # If you want to keep this generic, leave it commented:
        env_lines.append("# WEBHOOK_VERIFY_TOKEN=")
        env_lines.append("# WHATSAPP_TOKEN=")
        env_lines.append("# WHATSAPP_PHONE_NUMBER_ID=")
        env_lines.append("# WHATSAPP_VERIFY_TOKEN=")
        env_lines.append("# WHATSAPP_APP_SECRET=")
    else:
        env_lines.append(_line("WHATSAPP_TOKEN",           entered["whatsapp_token"]))
        env_lines.append(_line("WHATSAPP_PHONE_NUMBER_ID", entered["whatsapp_phone_id"]))
        env_lines.append(_line("WHATSAPP_VERIFY_TOKEN",    entered["whatsapp_verify"]))
        env_lines.append("# Required: Meta App Dashboard > App settings > Basic > App secret.")
        env_lines.append("# MAWS uses it to verify the X-Hub-Signature-256 of every webhook POST.")
        env_lines.append(_line("WHATSAPP_APP_SECRET", entered["whatsapp_app_secret"], commented_if_empty=False))
        env_lines.append("# TELEGRAM_TOKEN=")
        env_lines.append("# TELEGRAM_WEBHOOK_SECRET_TOKEN=")
        env_lines.append("# WEBHOOK_VERIFY_TOKEN=")

    env_written = env_will_be_written
    _write(env_path, "\n".join(env_lines) + "\n")

    # 5) .gitignore
    _write(gi_path, ".aws-sam/\n.maws_build/\n.maws_package/\nparams.tmp.json\n__pycache__/\nhistory/\nfiles/\n.env\n.env.*\n.bootstrap_state.json\n")

    # Copy scripts if they are missing
    #_ensure_script_in_cwd("bootstrap.sh", workdir, force=False)
    #_ensure_script_in_cwd("pull_history.sh", workdir, force=False)

    print("\n[OK] Structure created at:", workdir)
    print("  - params.json")
    print("  - config.json")
    print("  - fns.py")
    print("  - .env.prod")
    print("  - .gitignore")

    if install_deps:
        print("\n[INFO] Installing system dependencies (best effort, requested with --install-deps)...")
        print(
            "[INFO] This uses sudo: apt-get for curl/unzip/jq, and the latest AWS CLI / SAM CLI installers\n"
            "       downloaded from awscli.amazonaws.com and github.com/aws/aws-sam-cli. Their checksums and\n"
            "       signatures are NOT verified; prefer your package manager or the official instructions."
        )
        ok = _best_effort_install()
        if not ok:
            print("[INFO] Follow the instructions shown above to finish any pending installation steps.")

    # Check credentials; 'aws configure' only runs when asked for and credentials do not work.
    if _shutil.which("aws"):
        credentials_work = False
        try:
            out = _subprocess.check_output(
                ["aws", "sts", "get-caller-identity", "--output", "json"], text=True, stderr=_subprocess.DEVNULL
            )
            acct = _json.loads(out).get("Account")
            credentials_work = True
            print(f"[OK] AWS ready. Account: {acct}")
        except Exception:
            print("[INFO] Configure credentials:  aws configure  (or use AWS_PROFILE)")

        if run_config and credentials_work:
            print("[INFO] AWS credentials already work; not running 'aws configure'.")
        elif run_config:
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
    if bot == "telegram":
        if env_written:
            print("      TELEGRAM_WEBHOOK_SECRET_TOKEN was generated for you; keep it secret.")
        else:
            print("      .env.prod already existed: make sure it has TELEGRAM_WEBHOOK_SECRET_TOKEN (required).")
    else:
        print("      WHATSAPP_APP_SECRET is required (Meta App Dashboard > App settings > Basic).")
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
    """
    Read params.json. A missing file is ``{}``; a file that cannot be parsed raises
    MawsConfigError with the parser's message instead of silently becoming ``{}``.
    """
    conf_path = Path(conf_path)
    if not conf_path.exists():
        return {}
    try:
        data = json.loads(conf_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise MawsConfigError(
            f"{conf_path} is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno}). Fix it and re-run."
        ) from None
    except (OSError, UnicodeDecodeError) as e:
        raise MawsConfigError(f"Could not read {conf_path}: {e}") from None
    if not isinstance(data, dict):
        raise MawsConfigError(f"{conf_path} must contain a JSON object, not {type(data).__name__}.")
    return data

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
    new_project = not conf.exists()
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
    # A new project gets unguessable bucket names. An existing params.json without bucket names
    # may already be deployed with the legacy defaults: keep those, or the history would be lost.
    default_buckets = _new_project_bucket_names(project) if new_project else _legacy_bucket_names(project)
    history_bucket = _ask("S3 history_bucket", data.get("history_bucket") or default_buckets["history_bucket"])
    deploy_bucket  = _ask("S3 deployment_bucket", data.get("deployment_bucket") or default_buckets["deployment_bucket"])
    env_param_name = _ask("SSM env_param_name", data.get("env_param_name") or f"/{project}/prod/env")
    api_path       = _ask("API path", data.get("api_path") or "/webhook")

    lambda_timeout = _ask("Lambda function timeout (in seconds)", data.get("lambda_timeout") or "120")

    busy_policy = _ask("Busy message policy (drop/fifo)", data.get("busy_policy") or "drop").lower()
    if busy_policy not in ("drop", "fifo"):
        print(f"[maws] Invalid busy policy '{busy_policy}', using 'drop'.")
        busy_policy = "drop"

    sync_tokens_s3 = _ask_bool("Sync tokens from S3 (SYNC_TOKENS_S3)", bool(data.get("sync_tokens_s3", True)))
    tokens_prefix  = _ask("S3 prefix for tokens (TOKENS_S3_PREFIX)",
                          data.get("tokens_s3_prefix") or "secrets")
    persist_files_s3 = _ask_bool("Persist user files/media to S3", bool(data.get("persist_files_s3", False)))
    files_prefix = _ask("S3 prefix for persisted files",
                        data.get("files_s3_prefix") or "files")
    history_mode = _ask("History mode (per_user/shared)", data.get("history_mode") or "per_user").lower()
    if history_mode not in ("per_user", "shared"):
        print(f"[maws] Invalid history mode '{history_mode}', using 'per_user'.")
        history_mode = "per_user"
    history_rotation = _ask("Shared history rotation (message_count/time_period/both)",
                            data.get("history_rotation") or "message_count")
    history_max_messages = _ask("Shared history max messages",
                                data.get("history_max_messages") or "1000")
    history_period = _ask("Shared history period",
                          data.get("history_period") or "1w")
    webhook_security = dict(data.get("webhook_security")) if isinstance(data.get("webhook_security"), dict) else {}
    if bot == "whatsapp":
        # Verification is on by default and maws update refuses to deploy without it unless
        # webhook_security.allow_unauthenticated=true.
        webhook_security["whatsapp_verify_signature"] = _ask_bool(
            "Verify WhatsApp POST signatures (needs WHATSAPP_APP_SECRET in .env.prod)",
            bool(webhook_security.get("whatsapp_verify_signature", True)),
        )
        if not webhook_security["whatsapp_verify_signature"] and webhook_security.get("allow_unauthenticated") is not True:
            print(
                "[WARN] maws update will refuse to deploy without WhatsApp signature verification unless "
                "params.json sets webhook_security.allow_unauthenticated=true (NOT recommended)."
            )
    infra = dict(data.get("infra")) if isinstance(data.get("infra"), dict) else {}
    lambda_runtime = _ask(
        "Lambda Python runtime (sam build needs the same Python version locally)",
        infra.get("runtime") or DEFAULT_LAMBDA_RUNTIME,
    ) or DEFAULT_LAMBDA_RUNTIME
    infra["runtime"] = lambda_runtime
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
        "busy_policy": busy_policy,
        "sync_tokens_s3": bool(sync_tokens_s3),
        "tokens_s3_prefix": tokens_prefix,
        "persist_files_s3": bool(persist_files_s3),
        "files_s3_prefix": files_prefix,
        "history_mode": history_mode,
        "history_rotation": history_rotation,
        "history_max_messages": int(history_max_messages),
        "history_period": history_period,
        "webhook_security": webhook_security,
        "infra": infra,
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
    for k in (
        "project","region","bot","stack_name","history_bucket","deployment_bucket",
        "env_param_name","api_path","lambda_timeout","busy_policy",
        "sync_tokens_s3","tokens_s3_prefix","persist_files_s3","files_s3_prefix",
        "history_mode","history_rotation","history_max_messages","history_period",
        "verbose","requirements_source","requirements_ref",
    ):
        if k in data:
            print(f"   - {k}: {data[k]}")
    for nested in ("webhook_security", "runtime", "failure_handling", "infra", "history_append"):
        if nested in data:
            print(f"   - {nested}: {json.dumps(data[nested], sort_keys=True)}")

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

# --- remove_project ---
def _client_error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


def _is_missing_stack_error(error: BaseException) -> bool:
    return "does not exist" in str(error) and _client_error_code(error) in ("ValidationError", "")


def _ssm_client(region: Optional[str] = None):
    kw = {"region_name": region} if region else {}
    return _require_boto3().client("ssm", **kw)


def _account_id(region: Optional[str] = None) -> Optional[str]:
    """The caller's AWS account id, or None when it cannot be determined."""
    try:
        kw = {"region_name": region} if region else {}
        return str(_require_boto3().client("sts", **kw).get_caller_identity()["Account"])
    except Exception:
        return None


def _is_no_such_bucket(s3, error: BaseException) -> bool:
    no_such_bucket = getattr(getattr(s3, "exceptions", None), "NoSuchBucket", None)
    if isinstance(no_such_bucket, type) and isinstance(error, no_such_bucket):
        return True
    return _client_error_code(error) == "NoSuchBucket"


def _delete_object_batch(s3, bucket: str, objects: List[Dict], owner: Dict):
    for start in range(0, len(objects), 1000):
        response = s3.delete_objects(
            Bucket=bucket, Delete={"Objects": objects[start:start + 1000], "Quiet": True}, **owner
        )
        errors = response.get("Errors") if isinstance(response, dict) else None
        if errors:
            raise RuntimeError(f"{len(errors)} object(s) could not be deleted, e.g. {errors[0]}")


def _empty_bucket(s3, bucket: str, prefix: str = "", expected_owner: Optional[str] = None) -> bool:
    """
    Delete every object, version and delete marker under ``prefix`` ("" = the whole bucket).
    Returns True when done (or the bucket does not exist), False after printing the error.
    """
    owner = {"ExpectedBucketOwner": expected_owner} if expected_owner else {}
    listing = {"Bucket": bucket, **owner}
    if prefix:
        listing["Prefix"] = prefix
    try:
        for page in s3.get_paginator("list_object_versions").paginate(**listing):
            objects = [
                {"Key": item["Key"], "VersionId": item["VersionId"]}
                for item in page.get("Versions", []) + page.get("DeleteMarkers", [])
            ]
            _delete_object_batch(s3, bucket, objects, owner)
        # Objects of a bucket that never had versioning (normally already covered above).
        for page in s3.get_paginator("list_objects_v2").paginate(**listing):
            _delete_object_batch(s3, bucket, [{"Key": item["Key"]} for item in page.get("Contents", [])], owner)
    except Exception as e:
        if _is_no_such_bucket(s3, e):
            return True
        target = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
        print(f"[ERR] Could not empty {target}: {e}")
        return False
    return True


def _bucket_is_empty(s3, bucket: str, expected_owner: Optional[str] = None) -> bool:
    owner = {"ExpectedBucketOwner": expected_owner} if expected_owner else {}
    versions = s3.list_object_versions(Bucket=bucket, MaxKeys=1, **owner)
    if versions.get("Versions") or versions.get("DeleteMarkers"):
        return False
    return not s3.list_objects_v2(Bucket=bucket, MaxKeys=1, **owner).get("Contents")


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
      - Check (read-only) that the stack exists in the region; otherwise delete nothing
      - Empty the history bucket when it is a resource of that stack
      - Remove the CloudFormation stack (API/Lambda/DDB, etc.)
      - Delete the SSM SecureString holding .env.prod
      - Delete the stack's deployment artifacts (s3://<deployment_bucket>/<stack>/) and the
        deployment bucket only if nothing else is left in it (unless keep_deploy_bucket)
    Returns 0 only when every step succeeded. Pulls names from params.json when not provided
    explicitly. When ``project`` differs from the params.json project, params.json describes
    another project: its stack/bucket/SSM names are ignored and the default names are derived from
    ``project``.
    """
    workdir = Path(project_dir or Path.cwd())
    conf = workdir / (config_path or "params.json")
    data = _load_params(conf)

    params_project = data.get("project")
    proj = project or params_project
    if not proj:
        print("[ERR] Could not determine 'project'. Pass it with --project or set it in params.json.")
        return 2

    other_project = bool(project) and project != params_project
    names = {} if other_project else data

    reg = _effective_region(region, data)
    stack = names.get("stack_name") or f"{proj}-stack"
    legacy = _legacy_bucket_names(proj)
    history_bucket = names.get("history_bucket") or legacy["history_bucket"]
    deploy_bucket  = names.get("deployment_bucket") or legacy["deployment_bucket"]
    env_param = names.get("env_param_name") or f"/{proj}/prod/env"
    # samconfig/sam deploy upload this stack's artifacts under s3://<deployment bucket>/<stack>/;
    # the bucket itself may be shared with other stacks.
    deploy_prefix = f"{stack}/"

    cf, s3, _ = _boto(reg)

    # Read-only checks first. Bucket names are global: emptying them for a stack that is not in
    # this region would destroy the data of a deployment that keeps running elsewhere.
    try:
        stack_info = cf.describe_stacks(StackName=stack)["Stacks"][0]
    except Exception as e:
        if _is_missing_stack_error(e):
            print(f"[ERR] Stack {stack} does not exist in region {reg}. Nothing was deleted.")
            print("      Pass --region with the stack's region (or --project / -c for another project).")
        else:
            print(f"[ERR] Could not read stack {stack} in region {reg}: {e}. Nothing was deleted.")
        return 1
    try:
        resources = cf.describe_stack_resources(StackName=stack).get("StackResources", [])
    except Exception as e:
        print(f"[ERR] Could not list the resources of stack {stack}: {e}. Nothing was deleted.")
        return 1
    stack_buckets = {r.get("PhysicalResourceId") for r in resources if r.get("ResourceType") == "AWS::S3::Bucket"}
    functions = [r.get("PhysicalResourceId") for r in resources if r.get("ResourceType") == "AWS::Lambda::Function"]
    history_in_stack = history_bucket in stack_buckets

    print("\nDeletion plan")
    if other_project and params_project:
        print(f"   [NOTE] --project {proj} differs from {conf.name} (project '{params_project}'):")
        print(f"          ignoring its stack/bucket/SSM names; using the default names for '{proj}'.")
        if not region and data.get("region"):
            print(f"          The region below still comes from {conf.name}; pass --region if '{proj}' lives elsewhere.")
    print(f"   Project:          {proj}")
    print(f"   Region:           {reg}")
    print(f"   Stack to delete:  {stack} ({stack_info.get('StackStatus', 'unknown status')})")
    if history_in_stack:
        print(f"   Empty bucket:     {history_bucket} (all histories, files and tokens; deleted with the stack)")
    else:
        print(f"   [WARN] {history_bucket} is not a resource of {stack}: it is left untouched.")
    print(f"   Delete SSM:       {env_param} (SecureString with the .env.prod secrets)")
    if not keep_deploy_bucket:
        print(f"   Delete files:     s3://{deploy_bucket}/{deploy_prefix} (deployment artifacts of this stack)")
        print(f"                     then the bucket {deploy_bucket}, only if nothing else is left in it")
    else:
        print(f"   Keep bucket:      {deploy_bucket} (deployment)")

    if not yes and not _ask_bool("Continue?", False):
        print("Aborted.")
        return 1

    failures: List[str] = []

    # Empty the history bucket (so CloudFormation can delete it).
    if history_in_stack:
        print(f"[INFO] Emptying s3://{history_bucket} ...")
        if not _empty_bucket(s3, history_bucket):
            print("[ERR] The stack was not deleted: CloudFormation cannot delete a bucket that is not empty.")
            return 1

    print(f"[INFO] Deleting stack {stack} ...")
    try:
        cf.delete_stack(StackName=stack)
    except Exception as e:
        print(f"[ERR] Could not delete stack {stack}: {e}")
        print("      The SSM parameter and the deployment files were kept. Fix the problem and re-run maws remove.")
        return 1
    if wait:
        print("[INFO] Waiting for stack deletion to finish...")
        try:
            cf.get_waiter("stack_delete_complete").wait(StackName=stack)
        except Exception as e:
            print(f"[ERR] Stack {stack} was not deleted: {e}")
            print("      The SSM parameter and the deployment files were kept. Check the stack events in the")
            print("      CloudFormation console, fix the problem and re-run maws remove.")
            return 1
        print("[OK] Stack deleted.")
    else:
        print("[INFO] Stack deletion started (asynchronous; use --wait to wait for it). Check it with:")
        print(f"       aws cloudformation describe-stacks --stack-name {stack} --region {reg}")

    # SSM SecureString with every secret from .env.prod (not part of the stack).
    print(f"[INFO] Deleting SSM parameter {env_param} ...")
    try:
        _ssm_client(reg).delete_parameter(Name=env_param)
        print("[OK] SSM parameter deleted.")
    except Exception as e:
        if _client_error_code(e) == "ParameterNotFound" or type(e).__name__ == "ParameterNotFound":
            print("[INFO] SSM parameter not found (already deleted).")
        else:
            failures.append(f"SSM parameter {env_param}: {e}")

    # Deployment artifacts (outside the stack; the bucket may be shared).
    if not keep_deploy_bucket:
        owner = _account_id(reg)
        print(f"[INFO] Deleting s3://{deploy_bucket}/{deploy_prefix} ...")
        if not _empty_bucket(s3, deploy_bucket, prefix=deploy_prefix, expected_owner=owner):
            failures.append(f"deployment files s3://{deploy_bucket}/{deploy_prefix}")
        else:
            owner_kw = {"ExpectedBucketOwner": owner} if owner else {}
            try:
                if _bucket_is_empty(s3, deploy_bucket, expected_owner=owner):
                    s3.delete_bucket(Bucket=deploy_bucket, **owner_kw)
                    print(f"[OK] Deployment bucket {deploy_bucket} deleted.")
                else:
                    print(f"[INFO] Kept deployment bucket {deploy_bucket}: it still holds other objects.")
            except Exception as e:
                if _is_no_such_bucket(s3, e):
                    print(f"[INFO] Deployment bucket {deploy_bucket} does not exist.")
                else:
                    failures.append(f"deployment bucket {deploy_bucket}: {e}")

    for function_name in functions:
        print(
            f"[INFO] If the log group /aws/lambda/{function_name} exists (Lambda created it before MAWS managed "
            "the log group), it is not part of the stack: delete it in CloudWatch Logs if you no longer need it."
        )

    if failures:
        print("[ERR] Not everything could be deleted:")
        for failure in failures:
            print(f"      - {failure}")
        return 1
    print("[OK] Done." + ("" if wait else " The stack is still being deleted in the background."))
    return 0
