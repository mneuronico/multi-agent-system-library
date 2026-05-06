from __future__ import annotations

from ._shared import *

import argparse
import socket
import socketserver
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .manager import AgentSystemManager
from .parser import Parser


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    raw = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _coerce_blocks(content: Any) -> List[dict]:
    content = _safe_json_loads(content)
    if isinstance(content, list):
        return [b if isinstance(b, dict) else {"type": "text", "content": b} for b in content]
    if isinstance(content, dict) and "type" in content:
        return [content]
    return [{"type": "text", "content": content}]


def _component_kind(component: Any) -> str:
    cls = component.__class__.__name__.lower()
    if cls.endswith("bot"):
        return "bot"
    return cls


def _component_to_summary(name: str, component: Any) -> dict:
    kind = _component_kind(component)
    summary = {
        "name": name,
        "type": kind,
        "description": getattr(component, "description", None),
    }
    if kind == "agent":
        summary.update({
            "system": getattr(component, "system_prompt_original", None),
            "required_outputs": getattr(component, "required_outputs", None),
            "default_output": getattr(component, "default_output", None),
            "models": getattr(component, "models", None),
            "positive_filter": getattr(component, "positive_filter", None),
            "negative_filter": getattr(component, "negative_filter", None),
            "model_params": getattr(component, "model_params", None),
            "include_timestamp": getattr(component, "include_timestamp", None),
        })
    elif kind == "tool":
        summary.update({
            "inputs": getattr(component, "inputs", None),
            "outputs": getattr(component, "outputs", None),
            "default_output": getattr(component, "default_output", None),
            "function": getattr(getattr(component, "function", None), "__name__", None),
        })
    elif kind == "process":
        summary.update({
            "function": getattr(getattr(component, "function", None), "__name__", None),
            "expected_params": getattr(component, "expected_params", None),
        })
    elif kind == "automation":
        summary["sequence"] = getattr(component, "sequence", None)
    return summary


def _raw_components(config: dict) -> List[dict]:
    components = config.get("components", [])
    return components if isinstance(components, list) else []


def _automation_sequences(config: dict, manager: Optional[AgentSystemManager]) -> List[dict]:
    out = []
    if manager is not None:
        for name, automation in manager.automations.items():
            out.append({
                "name": name,
                "description": getattr(automation, "description", None),
                "sequence": getattr(automation, "sequence", []),
            })
        return out

    for component in _raw_components(config):
        if component.get("type") == "automation":
            out.append({
                "name": component.get("name") or "automation",
                "description": component.get("description"),
                "sequence": component.get("sequence", []),
            })
    return out


def _parse_input_spec(step: Any) -> Optional[dict]:
    if not isinstance(step, str):
        return None
    try:
        parsed = Parser().parse_input_string(step)
        return parsed
    except Exception as exc:
        return {"error": str(exc)}


def _annotate_sequence(sequence: Any) -> Any:
    if isinstance(sequence, list):
        return [_annotate_sequence(item) for item in sequence]
    if isinstance(sequence, str):
        return {
            "kind": "component_step",
            "raw": sequence,
            "parsed_input": _parse_input_spec(sequence),
        }
    if isinstance(sequence, dict):
        kind = sequence.get("control_flow_type", "dict")
        annotated = {"kind": kind, "raw": sequence}
        for key in ("body", "if_true", "if_false"):
            if key in sequence:
                annotated[key] = _annotate_sequence(sequence[key])
        if "cases" in sequence and isinstance(sequence["cases"], list):
            annotated["cases"] = [
                {
                    "case": case.get("case"),
                    "body": _annotate_sequence(case.get("body", [])),
                }
                for case in sequence["cases"]
                if isinstance(case, dict)
            ]
        if kind == "for" and isinstance(sequence.get("items"), str):
            annotated["parsed_items"] = _parse_input_spec(sequence["items"])
        if kind == "switch" and isinstance(sequence.get("value"), str):
            annotated["parsed_value"] = _parse_input_spec(sequence["value"])
        return annotated
    return sequence


def _read_history_file(path: Path, limit: int) -> dict:
    rows = []
    error = None
    try:
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT role, content, msg_number, type, model, timestamp "
                    "FROM message_history ORDER BY msg_number DESC LIMIT ?",
                    (limit,),
                )
                fetched = cur.fetchall()
            except sqlite3.OperationalError:
                cur.execute(
                    "SELECT role, content, msg_number, type, timestamp "
                    "FROM message_history ORDER BY msg_number DESC LIMIT ?",
                    (limit,),
                )
                fetched = [
                    (role, content, msg_number, msg_type, None, timestamp)
                    for role, content, msg_number, msg_type, timestamp in cur.fetchall()
                ]
        finally:
            conn.close()
        for role, content, msg_number, msg_type, model, timestamp in reversed(fetched):
            blocks = _coerce_blocks(content)
            rows.append({
                "role": role,
                "type": msg_type,
                "model": model,
                "timestamp": timestamp,
                "msg_number": msg_number,
                "blocks": blocks,
            })
    except Exception as exc:
        error = str(exc)

    return {
        "user_id": path.stem,
        "path": str(path),
        "messages": rows,
        "error": error,
    }


def _history_folder(project_dir: Path, config: dict, manager: Optional[AgentSystemManager] = None) -> Path:
    if manager is not None and getattr(manager, "history_folder", None):
        return Path(manager.history_folder)
    general = config.get("general_parameters", {}) if isinstance(config, dict) else {}
    configured = general.get("history_folder")
    if configured:
        folder = Path(configured)
        return folder if folder.is_absolute() else project_dir / folder
    return project_dir / "history"


def build_dashboard_state(project_dir: Path, history_limit: int = 200) -> dict:
    project_dir = project_dir.resolve()
    config_path = project_dir / "config.json"
    state = {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "config_path": str(config_path),
        "has_config": config_path.is_file(),
        "load_error": None,
        "manager_loaded": False,
        "config": {},
        "general_parameters": {},
        "components": [],
        "automations": [],
        "annotated_automations": [],
        "histories": [],
    }

    if not config_path.is_file():
        state["load_error"] = "No config.json was found in this directory."
        return state

    try:
        state["config"] = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        state["load_error"] = f"Could not read config.json: {exc}"
        return state

    config = state["config"]
    state["general_parameters"] = config.get("general_parameters", {}) if isinstance(config, dict) else {}

    manager = None
    try:
        manager = AgentSystemManager(config=str(config_path), base_directory=str(project_dir))
        state["manager_loaded"] = True
        ordered_names = getattr(manager, "_component_order", [])
        components = []
        for name in ordered_names:
            component = manager._get_component(name)
            if component is not None:
                components.append(_component_to_summary(name, component))
        state["components"] = components
    except Exception as exc:
        state["load_error"] = f"Loaded raw config, but manager initialization failed: {exc}"
        state["components"] = _raw_components(config)

    state["automations"] = _automation_sequences(config, manager)
    state["annotated_automations"] = [
        {
            **automation,
            "annotated_sequence": _annotate_sequence(automation.get("sequence", [])),
        }
        for automation in state["automations"]
    ]

    folder = _history_folder(project_dir, config, manager)
    if folder.is_dir():
        state["histories"] = [
            _read_history_file(path, history_limit)
            for path in sorted(folder.glob("*.sqlite"))
        ]
    return state


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MAS Dashboard</title>
  <style>
    :root {
      --bg: #f6f1e6;
      --bg-2: #fbf7ec;
      --panel: #ffffff;
      --ink: #1a201d;
      --muted: #6b7672;
      --line: #e1d8c5;
      --line-strong: #c9bfa8;
      --shadow: 0 14px 40px rgba(39, 31, 18, 0.08);
      --shadow-soft: 0 4px 14px rgba(39, 31, 18, 0.06);

      --agent: #e56b5d;
      --agent-bg: #fbe6e1;
      --agent-ink: #a23e32;

      --tool: #0f9a8a;
      --tool-bg: #d8efea;
      --tool-ink: #066559;

      --process: #5b8b59;
      --process-bg: #e1ecdf;
      --process-ink: #345e35;

      --automation: #5362b7;
      --automation-bg: #e3e6f5;
      --automation-ink: #2c3784;

      --user-c: #2e6cb6;
      --user-bg: #e2edf9;
      --user-ink: #1d4677;

      --switch: #d99a22;
      --switch-bg: #fbeccd;
      --switch-ink: #8a5a00;

      --branch: #b15ec4;
      --branch-bg: #f3e0f6;
      --branch-ink: #6f2c80;

      --loop: #5362b7;
      --loop-bg: #e8eafa;
      --loop-ink: #2c3784;

      --iterator: #8d6cd6;
      --iterator-bg: #ece3fa;
      --iterator-ink: #4a328a;

      --variable: #b58522;
      --variable-bg: #f5e6c2;
      --variable-ink: #6a4a0d;

      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--ink);
      font-size: 14px;
      line-height: 1.5;
    }
    button, select, input { font: inherit; color: inherit; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }

    .app { display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }
    .sidebar {
      background: #efe8d6;
      border-right: 1px solid var(--line);
      padding: 18px 12px;
      position: sticky; top: 0; height: 100vh; overflow: auto;
    }
    .brand {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 6px 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 12px;
    }
    .brand .mark {
      width: 34px; height: 34px; border-radius: 9px;
      display: grid; place-items: center; color: white;
      background: linear-gradient(135deg, var(--agent), var(--tool));
      font-weight: 800; font-size: 16px;
    }
    .brand .name { font-weight: 800; font-size: 15px; }
    .brand .sub { color: var(--muted); font-size: 12px; }

    .project-pick {
      display: flex; align-items: center; gap: 8px;
      padding: 10px 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      margin-bottom: 14px;
    }
    .project-pick .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .project-pick .value { font-weight: 700; word-break: break-all; font-size: 13px; }

    .nav { display: grid; gap: 2px; margin-bottom: 18px; }
    .nav button {
      width: 100%; border: 0; background: transparent; color: var(--muted);
      padding: 9px 10px; border-radius: 8px; cursor: pointer;
      display: flex; align-items: center; gap: 10px; text-align: left;
      font-weight: 600;
    }
    .nav button:hover { background: rgba(255,255,255,0.6); color: var(--ink); }
    .nav button.active {
      background: var(--panel); color: var(--ink);
      box-shadow: inset 0 0 0 1px var(--line);
    }
    .nav .ic { width: 18px; height: 18px; flex: 0 0 18px; }
    .nav .count {
      margin-left: auto; min-width: 22px; height: 20px;
      border-radius: 999px; padding: 0 7px;
      background: rgba(0,0,0,0.06); color: var(--ink);
      font-size: 11px; font-weight: 700;
      display: inline-grid; place-items: center;
    }
    .nav button.active .count { background: rgba(15,154,138,0.16); color: var(--tool-ink); }

    .legend { padding: 10px 6px; border-top: 1px solid var(--line); margin-top: 8px; }
    .legend h4 { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 6px 0 8px; }
    .legend-row { display: flex; align-items: center; gap: 10px; padding: 4px 0; font-size: 12px; }
    .legend-dot { width: 10px; height: 10px; border-radius: 999px; }

    .main { min-width: 0; padding: 22px 28px 60px; }
    .topbar {
      display: flex; align-items: center; justify-content: space-between; gap: 16px;
      padding-bottom: 18px; border-bottom: 1px solid var(--line);
      margin-bottom: 20px;
    }
    .topbar h1 { margin: 0; font-size: 22px; }
    .topbar .crumbs { color: var(--muted); font-size: 13px; word-break: break-all; }
    .topbar .right { display: flex; gap: 8px; }
    .btn {
      border: 1px solid var(--line);
      background: var(--panel); color: var(--ink);
      padding: 8px 12px; border-radius: 8px;
      cursor: pointer; font-weight: 700;
      box-shadow: var(--shadow-soft);
      display: inline-flex; align-items: center; gap: 6px;
    }
    .btn:hover { border-color: var(--line-strong); }
    .btn.primary { background: var(--ink); color: white; border-color: var(--ink); }
    .btn.ghost { background: transparent; box-shadow: none; }
    .btn.small { padding: 4px 8px; font-size: 12px; }

    .section { display: none; }
    .section.active { display: block; }

    /* === OVERVIEW === */
    .stat-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px; margin-bottom: 22px;
    }
    .stat {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 14px 16px; box-shadow: var(--shadow-soft);
    }
    .stat .v { font-size: 26px; font-weight: 800; }
    .stat .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-top: 2px; }

    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 18px; box-shadow: var(--shadow-soft);
      margin-bottom: 18px;
    }
    .panel h2 { margin: 0 0 12px; font-size: 16px; display: flex; align-items: center; gap: 8px; }
    .panel h3 { margin: 0 0 8px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

    .two-col { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 18px; }
    @media (max-width: 1100px) { .two-col { grid-template-columns: 1fr; } }

    .kv { display: grid; grid-template-columns: 160px 1fr; gap: 8px 14px; align-items: start; }
    .kv .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .kv .v { word-break: break-word; }
    .kv-row { display: contents; }

    .mini-graph {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
    }
    .mini-graph .mini-card {
      border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
      background: var(--bg-2); display: flex; align-items: center; gap: 10px;
      cursor: pointer;
    }
    .mini-graph .mini-card:hover { box-shadow: var(--shadow-soft); }
    .mini-graph .mc-icon {
      width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center;
      color: white; flex: 0 0 28px;
    }
    .mini-graph .mc-name { font-weight: 700; font-size: 13px; }
    .mini-graph .mc-type { font-size: 11px; color: var(--muted); }

    /* === TYPE COLOR HELPERS === */
    .t-agent .icon-bg, .icon-bg.t-agent { background: var(--agent); }
    .t-tool .icon-bg, .icon-bg.t-tool { background: var(--tool); }
    .t-process .icon-bg, .icon-bg.t-process { background: var(--process); }
    .t-automation .icon-bg, .icon-bg.t-automation { background: var(--automation); }
    .t-user .icon-bg, .icon-bg.t-user { background: var(--user-c); }
    .t-iterator .icon-bg, .icon-bg.t-iterator { background: var(--iterator); }
    .t-variable .icon-bg, .icon-bg.t-variable { background: var(--variable); }

    .t-agent { --acc: var(--agent); --acc-bg: var(--agent-bg); --acc-ink: var(--agent-ink); }
    .t-tool { --acc: var(--tool); --acc-bg: var(--tool-bg); --acc-ink: var(--tool-ink); }
    .t-process { --acc: var(--process); --acc-bg: var(--process-bg); --acc-ink: var(--process-ink); }
    .t-automation { --acc: var(--automation); --acc-bg: var(--automation-bg); --acc-ink: var(--automation-ink); }
    .t-user { --acc: var(--user-c); --acc-bg: var(--user-bg); --acc-ink: var(--user-ink); }
    .t-iterator { --acc: var(--iterator); --acc-bg: var(--iterator-bg); --acc-ink: var(--iterator-ink); }
    .t-variable { --acc: var(--variable); --acc-bg: var(--variable-bg); --acc-ink: var(--variable-ink); }

    .badge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 700;
      background: rgba(0,0,0,0.06); color: var(--ink);
      text-transform: uppercase; letter-spacing: .03em;
    }
    .badge.t-agent { background: var(--agent-bg); color: var(--agent-ink); }
    .badge.t-tool { background: var(--tool-bg); color: var(--tool-ink); }
    .badge.t-process { background: var(--process-bg); color: var(--process-ink); }
    .badge.t-automation { background: var(--automation-bg); color: var(--automation-ink); }
    .badge.t-user { background: var(--user-bg); color: var(--user-ink); }
    .badge.t-iterator { background: var(--iterator-bg); color: var(--iterator-ink); }
    .badge.t-variable { background: var(--variable-bg); color: var(--variable-ink); }
    .badge.subtle { background: rgba(0,0,0,0.04); color: var(--muted); }

    /* === FLOW VISUALIZATION === */
    .flow-toolbar {
      display: flex; gap: 10px; align-items: center; margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .flow-toolbar .auto-pick {
      display: flex; gap: 4px; padding: 4px;
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 10px;
    }
    .flow-toolbar .auto-pick button {
      border: 0; background: transparent;
      padding: 6px 12px; border-radius: 7px;
      cursor: pointer; font-weight: 700; color: var(--muted);
    }
    .flow-toolbar .auto-pick button.active { background: var(--ink); color: white; }

    .flow-canvas {
      background:
        radial-gradient(circle at 1px 1px, rgba(0,0,0,0.06) 1px, transparent 1px) 0 0 / 22px 22px,
        var(--bg-2);
      border: 1px solid var(--line); border-radius: 16px;
      padding: 28px 18px; box-shadow: var(--shadow-soft);
      min-width: 0; overflow-x: hidden;
    }
    .flow-vertical {
      display: flex; flex-direction: column; align-items: center;
      gap: 0; min-width: 0;
    }
    .fc-arrow {
      flex: 0 0 auto;
      position: relative;
      width: 2px; min-height: 22px;
      background: var(--line-strong);
      margin: 2px 0;
    }
    .fc-arrow::after {
      content: ''; position: absolute; left: 50%; bottom: -1px;
      width: 0; height: 0;
      border-left: 5px solid transparent;
      border-right: 5px solid transparent;
      border-top: 7px solid var(--line-strong);
      transform: translateX(-50%);
    }

    .fc-node {
      background: var(--panel); border: 1.5px solid var(--line);
      border-radius: 12px; padding: 12px 14px;
      box-shadow: var(--shadow-soft);
      width: 100%;
      max-width: 380px;
      min-width: 0;
      cursor: pointer; transition: transform 0.12s, box-shadow 0.12s;
      border-left: 4px solid var(--acc, var(--line-strong));
      position: relative;
    }
    .fc-node:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
    .fc-node.selected { box-shadow: 0 0 0 2px var(--acc, var(--ink)), var(--shadow); }
    .fc-node.highlight { box-shadow: 0 0 0 2px var(--acc), 0 8px 22px rgba(39, 31, 18, 0.14); }
    .fc-node .node-head {
      display: grid; grid-template-columns: auto 1fr auto; gap: 10px;
      align-items: center;
    }
    .fc-node .nm-icon {
      width: 32px; height: 32px; border-radius: 8px;
      background: var(--acc-bg, rgba(0,0,0,0.05)); color: var(--acc-ink, var(--ink));
      display: grid; place-items: center;
    }
    .fc-node .nm-name { font-weight: 800; font-size: 14px; word-break: break-word; }
    .fc-node .nm-meta { color: var(--muted); font-size: 11.5px; margin-top: 2px; word-break: break-word; }
    .fc-node .io-row {
      display: grid; grid-template-columns: auto 1fr; gap: 6px 8px;
      margin-top: 8px; padding-top: 8px;
      border-top: 1px dashed var(--line);
      align-items: center;
    }
    .fc-node .io-row + .io-row { margin-top: 4px; padding-top: 4px; border-top: 0; }
    .fc-node .io-label {
      font-size: 10px; font-weight: 800;
      color: var(--muted); text-transform: uppercase; letter-spacing: .05em;
    }
    .fc-node .io-pills {
      display: flex; flex-wrap: wrap; gap: 4px; min-width: 0;
    }

    .fc-container {
      border: 1.5px dashed var(--acc, var(--line-strong));
      background: rgba(255,255,255,0.55);
      border-radius: 14px;
      padding: 30px 16px 18px;
      position: relative;
      width: 100%;
      max-width: 100%;
      min-width: 0;
    }
    .fc-container > .fc-container-label {
      position: absolute; top: -13px; left: 18px;
      background: var(--bg-2); border: 1px solid var(--acc, var(--line-strong));
      color: var(--acc-ink, var(--ink));
      padding: 3px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em;
      display: inline-flex; align-items: center; gap: 6px;
      max-width: calc(100% - 36px);
    }
    .fc-container .fc-container-sub {
      text-align: center; color: var(--muted); margin: -8px 0 14px;
      font-size: 12px;
      display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; align-items: center;
    }

    .fc-switch-cases {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      align-items: start;
      min-width: 0;
    }
    .fc-case {
      display: flex; flex-direction: column; align-items: stretch;
      min-width: 0;
    }
    .fc-case > * { width: 100%; }
    .fc-case .fc-arrow { align-self: center; }
    .fc-case-label {
      align-self: center;
      background: var(--switch-bg); color: var(--switch-ink);
      border: 1px solid var(--switch);
      padding: 4px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 800; text-transform: uppercase;
      margin-bottom: 8px;
      max-width: 100%;
      overflow: hidden; text-overflow: ellipsis;
      white-space: nowrap;
    }
    .fc-case-label.true { background: var(--process-bg); color: var(--process-ink); border-color: var(--process); }
    .fc-case-label.false { background: var(--agent-bg); color: var(--agent-ink); border-color: var(--agent); }
    .fc-case-empty {
      padding: 10px 14px; border: 1px dashed var(--line);
      border-radius: 10px; color: var(--muted); font-size: 12px;
      background: rgba(255,255,255,0.5); text-align: center;
    }

    /* Reference and output pills */
    .ref {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 2px 8px 2px 4px;
      border-radius: 999px;
      background: var(--acc-bg);
      color: var(--acc-ink);
      border: 1px solid var(--acc);
      font-size: 11px; font-weight: 700;
      cursor: pointer; vertical-align: middle;
      transition: box-shadow 0.12s, transform 0.12s;
      max-width: 100%;
    }
    .ref .ref-field {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: rgba(255,255,255,0.85);
      color: var(--acc-ink);
      border-radius: 999px;
      padding: 1px 7px;
      font-size: 10.5px; font-weight: 700;
      max-width: 140px; overflow: hidden; text-overflow: ellipsis;
    }
    .ref .ref-arrow { opacity: 0.55; font-size: 10px; }
    .ref .ref-comp { font-weight: 800; }
    .ref .ref-idx { opacity: 0.7; font-weight: 600; font-size: 10px; }
    .ref.highlight, .ref:hover { box-shadow: 0 0 0 2px var(--acc); transform: translateY(-1px); }

    .out-pill {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 1px 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.6);
      color: var(--acc-ink);
      border: 1px dashed var(--acc);
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 10.5px; font-weight: 700;
      max-width: 100%; cursor: pointer;
      transition: box-shadow 0.12s, transform 0.12s;
    }
    .out-pill .out-tip { opacity: 0.55; font-size: 9.5px; font-family: inherit; }
    .out-pill.highlight, .out-pill:hover {
      background: var(--acc-bg);
      box-shadow: 0 0 0 2px var(--acc);
      transform: translateY(-1px);
    }

    .hl-context {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 2px 8px; border-radius: 999px;
      background: rgba(0,0,0,0.04); border: 1px dashed var(--line-strong);
      color: var(--muted); font-size: 11px; font-weight: 600;
    }

    /* Detail panel for selected node in flow */
    .flow-with-detail { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; }
    @media (max-width: 1280px) { .flow-with-detail { grid-template-columns: 1fr; } }
    .detail-card {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 16px;
      box-shadow: var(--shadow-soft);
      position: sticky; top: 14px;
      max-height: calc(100vh - 30px);
      overflow: auto;
    }
    .detail-card h3 { margin: 6px 0 12px; font-size: 16px; }
    .detail-card .group { margin-bottom: 14px; }
    .detail-card .group h4 {
      margin: 0 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
      color: var(--muted);
    }
    .detail-card pre {
      background: var(--bg-2); padding: 10px; border-radius: 8px;
      border: 1px solid var(--line); font-size: 12px;
      white-space: pre-wrap; word-break: break-word; margin: 0;
    }
    .detail-card .system-prompt {
      background: var(--bg-2); border: 1px solid var(--line);
      border-radius: 8px; padding: 10px; font-size: 12.5px;
      white-space: pre-wrap; line-height: 1.5;
    }

    /* === COMPONENT CARDS === */
    .components-toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; align-items: center; }
    .filter-chip {
      padding: 6px 12px; border-radius: 999px; cursor: pointer;
      border: 1px solid var(--line); background: var(--panel);
      font-size: 12px; font-weight: 700;
    }
    .filter-chip.active { background: var(--ink); color: white; border-color: var(--ink); }
    .components-grid {
      display: grid; gap: 14px;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    }
    .comp-card {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 16px; box-shadow: var(--shadow-soft);
      border-left: 4px solid var(--acc, var(--line-strong));
      display: flex; flex-direction: column; gap: 10px;
    }
    .comp-head { display: flex; align-items: center; gap: 10px; }
    .comp-head .ic {
      width: 36px; height: 36px; border-radius: 9px; flex: 0 0 36px;
      background: var(--acc-bg); color: var(--acc-ink);
      display: grid; place-items: center;
    }
    .comp-head .nm { font-weight: 800; font-size: 15px; word-break: break-word; }
    .comp-desc { color: var(--muted); font-size: 13px; }
    .comp-card .system-prompt {
      background: var(--bg-2); border: 1px dashed var(--line);
      border-radius: 8px; padding: 8px 10px;
      font-size: 12px; line-height: 1.5;
      max-height: 110px; overflow: auto;
    }
    .io-list { display: grid; gap: 4px; font-size: 12.5px; }
    .io-list .io {
      display: grid; grid-template-columns: max-content 1fr; gap: 8px;
      padding: 4px 0;
    }
    .io-list .io .nm {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-weight: 700; color: var(--acc-ink, var(--ink));
    }
    .io-list .io .ds { color: var(--muted); }

    /* === HISTORY === */
    .history-layout { display: grid; grid-template-columns: 240px 1fr; gap: 18px; }
    @media (max-width: 1000px) { .history-layout { grid-template-columns: 1fr; } }
    .user-list { display: grid; gap: 6px; align-content: start; }
    .user-btn {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 12px; border-radius: 10px;
      background: var(--panel); border: 1px solid var(--line);
      cursor: pointer; box-shadow: var(--shadow-soft);
    }
    .user-btn:hover { border-color: var(--line-strong); }
    .user-btn.active { box-shadow: 0 0 0 2px var(--ink), var(--shadow-soft); }
    .user-btn .avatar {
      width: 32px; height: 32px; border-radius: 999px;
      display: grid; place-items: center; flex: 0 0 32px;
      background: var(--user-bg); color: var(--user-ink);
      font-weight: 800;
    }
    .user-btn .meta { display: grid; min-width: 0; }
    .user-btn .uid { font-weight: 700; font-size: 13px; word-break: break-all; }
    .user-btn .cnt { color: var(--muted); font-size: 11px; }

    .turn {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 14px; padding: 18px; margin-bottom: 16px;
      box-shadow: var(--shadow-soft);
    }
    .turn-head {
      display: flex; align-items: center; gap: 10px;
      padding-bottom: 12px; border-bottom: 1px dashed var(--line);
      margin-bottom: 14px;
    }
    .turn-head .turn-no {
      background: var(--bg-2); color: var(--muted);
      padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 700;
    }
    .turn-user-text {
      font-weight: 600; flex: 1; min-width: 0;
      max-width: 100%;
    }
    .turn-user-text .preview { display: block; word-break: break-word; }

    .step-list { display: grid; gap: 10px; }
    .step {
      display: grid; grid-template-columns: 36px 1fr; gap: 12px;
      padding: 8px 0;
    }
    .step .step-rail {
      position: relative;
      display: flex; flex-direction: column; align-items: center;
    }
    .step .step-rail .icon-circle {
      width: 32px; height: 32px; border-radius: 999px;
      background: var(--acc-bg); color: var(--acc-ink);
      display: grid; place-items: center;
      flex: 0 0 32px;
      box-shadow: 0 0 0 2px var(--panel), 0 0 0 3px var(--acc);
      z-index: 1;
    }
    .step .step-rail::after {
      content: ''; position: absolute; top: 32px; bottom: -10px;
      width: 2px; background: var(--line); left: 50%; transform: translateX(-50%);
    }
    .step:last-child .step-rail::after { display: none; }
    .step .step-body {
      background: var(--bg-2); border: 1px solid var(--line);
      border-left: 3px solid var(--acc);
      border-radius: 10px; padding: 10px 14px;
      min-width: 0;
    }
    .step .step-name {
      font-weight: 800; font-size: 13.5px; display: flex; align-items: center; gap: 8px;
    }
    .step .step-name .ts { color: var(--muted); font-weight: 500; font-size: 11.5px; margin-left: auto; }
    .step .step-content { margin-top: 6px; font-size: 13px; }

    .field-grid { display: grid; gap: 6px; }
    .field {
      display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 10px;
      padding: 4px 0; border-bottom: 1px dotted var(--line);
    }
    .field:last-child { border-bottom: 0; }
    .field .fk {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 11.5px; color: var(--muted); padding-top: 2px;
      text-transform: lowercase;
    }
    .field .fv { word-break: break-word; }
    .field .fv code { font-size: 12px; padding: 1px 4px; background: rgba(0,0,0,0.05); border-radius: 4px; }
    .field .fv pre {
      margin: 4px 0 0; background: rgba(0,0,0,0.04);
      padding: 6px 8px; border-radius: 6px;
      font-size: 11.5px; max-height: 220px; overflow: auto;
      white-space: pre-wrap; word-break: break-word;
    }
    .field .pill-list { display: flex; gap: 4px; flex-wrap: wrap; }
    .field .pill-list span {
      padding: 1px 7px; border-radius: 999px;
      background: var(--acc-bg, rgba(0,0,0,0.06));
      color: var(--acc-ink, var(--ink));
      font-size: 11px; font-weight: 700;
    }
    .text-bubble { white-space: pre-wrap; word-break: break-word; }

    .meta-toggle {
      margin-top: 6px;
      display: inline-flex; align-items: center; gap: 6px;
      background: transparent; border: 0; cursor: pointer;
      color: var(--muted); font-size: 11.5px; font-weight: 700;
      padding: 2px 0;
    }
    .meta-toggle:hover { color: var(--ink); }
    .meta-content { display: none; margin-top: 6px; }
    .meta-content.open { display: block; }
    .meta-stats { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
    .meta-stat {
      display: inline-flex; gap: 6px; align-items: center;
      background: var(--bg-2); border: 1px solid var(--line);
      padding: 3px 8px; border-radius: 999px; font-size: 11px;
    }
    .meta-stat .k { color: var(--muted); font-weight: 600; }
    .meta-stat .v { font-weight: 700; }

    /* === RAW === */
    .raw-wrap {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 12px; padding: 14px; box-shadow: var(--shadow-soft);
      max-height: 80vh; overflow: auto;
    }
    .raw-wrap pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.5; }

    .empty {
      padding: 24px; border: 1.5px dashed var(--line);
      border-radius: 12px; color: var(--muted);
      background: rgba(255,255,255,0.55); text-align: center;
    }
    .status-banner {
      padding: 10px 14px; border-radius: 10px; margin-bottom: 14px;
      background: var(--tool-bg); color: var(--tool-ink); font-weight: 700;
    }
    .status-banner.error { background: var(--agent-bg); color: var(--agent-ink); }

    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; height: auto; }
      .topbar { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="mark">M</div>
      <div>
        <div class="name">MAS Dashboard</div>
        <div class="sub">Multi-agent system explorer</div>
      </div>
    </div>

    <div class="project-pick">
      <div style="min-width: 0;">
        <div class="label">Project</div>
        <div class="value" id="projectName">—</div>
      </div>
    </div>

    <nav class="nav">
      <button data-view="overview" class="active">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
        Overview
      </button>
      <button data-view="flow">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><path d="M6 8.5v3a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-3"/><path d="M12 13.5v2"/></svg>
        Flow
        <span class="count" id="navFlowCount">0</span>
      </button>
      <button data-view="components">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>
        Components
        <span class="count" id="navCompCount">0</span>
      </button>
      <button data-view="history">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v5l3 2"/></svg>
        History
        <span class="count" id="navHistCount">0</span>
      </button>
      <button data-view="raw">
        <svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 4l-4 8 4 8M16 4l4 8-4 8"/></svg>
        Raw
      </button>
    </nav>

    <div class="legend">
      <h4>Legend</h4>
      <div class="legend-row"><span class="legend-dot" style="background: var(--agent)"></span>Agent (LLM call)</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--tool)"></span>Tool (Python fn)</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--process)"></span>Process</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--automation)"></span>Automation</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--user-c)"></span>User input</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--iterator)"></span>Iterator</div>
      <div class="legend-row"><span class="legend-dot" style="background: var(--variable)"></span>Variable</div>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div>
        <h1 id="mainTitle">Overview</h1>
        <div class="crumbs" id="projectPath"></div>
      </div>
      <div class="right">
        <button class="btn" id="refreshBtn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/></svg>
          Refresh
        </button>
        <button class="btn primary" id="openFlowBtn">Open Flow</button>
      </div>
    </div>

    <section id="overview" class="section active">
      <div id="status"></div>
      <div class="stat-grid" id="stats"></div>
      <div class="two-col">
        <div class="panel">
          <h2>Components</h2>
          <div id="overviewMini" class="mini-graph"></div>
        </div>
        <div>
          <div class="panel">
            <h2>System description</h2>
            <div id="overviewSysDesc" class="text-bubble"></div>
          </div>
          <div class="panel">
            <h2>General parameters</h2>
            <div id="overviewParams"></div>
          </div>
        </div>
      </div>
    </section>

    <section id="flow" class="section">
      <div class="flow-toolbar">
        <div class="auto-pick" id="autoPick"></div>
        <div id="flowDescription" style="color: var(--muted); font-size: 13px;"></div>
      </div>
      <div class="flow-with-detail">
        <div class="flow-canvas" id="flowCanvas"></div>
        <div class="detail-card" id="flowDetail">
          <div class="empty">Click any node to inspect it.</div>
        </div>
      </div>
    </section>

    <section id="components" class="section">
      <div class="components-toolbar" id="compFilters"></div>
      <div class="components-grid" id="componentsGrid"></div>
    </section>

    <section id="history" class="section">
      <div class="history-layout">
        <div class="user-list" id="userList"></div>
        <div id="historyMessages"></div>
      </div>
    </section>

    <section id="raw" class="section">
      <div class="raw-wrap"><pre id="rawState"></pre></div>
    </section>
  </main>
</div>
<script>
let STATE = null;
let selectedUser = null;
let selectedAutomation = 0;
let selectedComponentName = null;
let compFilter = 'all';
let openMeta = new Set();

const VIEW_TITLES = {
  overview: 'Overview',
  flow: 'Flow',
  components: 'Components',
  history: 'History',
  raw: 'Raw state'
};

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({
  '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
}[ch]));
const fmtJSON = (value) => esc(JSON.stringify(value, null, 2));

const TYPE_ICONS = {
  agent: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="6" width="16" height="13" rx="3"/><circle cx="9" cy="12.5" r="1.2" fill="currentColor"/><circle cx="15" cy="12.5" r="1.2" fill="currentColor"/><path d="M12 6V3M9.5 3h5"/></svg>',
  tool: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.1-2.1z"/></svg>',
  process: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a7.7 7.7 0 0 0 0-6l2.1-1.6-2-3.5-2.5.9a7.7 7.7 0 0 0-5.2-3l-.4-2.6h-4l-.4 2.6a7.7 7.7 0 0 0-5.2 3l-2.5-.9-2 3.5L4.6 9a7.7 7.7 0 0 0 0 6l-2.1 1.6 2 3.5 2.5-.9a7.7 7.7 0 0 0 5.2 3l.4 2.6h4l.4-2.6a7.7 7.7 0 0 0 5.2-3l2.5.9 2-3.5z" stroke-linejoin="round"/></svg>',
  automation: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M3 12h12M3 18h18"/><circle cx="18" cy="12" r="2"/></svg>',
  user: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
  iterator: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 1 9 9"/><path d="M12 21l-3-3 3-3"/></svg>',
  variable: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 5l-3 7 3 7M16 5l3 7-3 7"/><path d="M11 9l2 6"/></svg>',
  switch: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l4 4-4 4"/><path d="M16 7H6v10"/><path d="M12 21l-4-4 4-4"/><path d="M8 17h10V7"/></svg>',
  branch: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="5" r="2"/><circle cx="18" cy="5" r="2"/><circle cx="12" cy="19" r="2"/><path d="M6 7v3a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7"/><path d="M12 12v5"/></svg>',
  for: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/></svg>',
  while: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 0 0 18 0"/><path d="M3 12a9 9 0 0 1 18 0"/></svg>',
  unknown: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 1 1 4 3v1"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/></svg>'
};
function typeIcon(type) { return TYPE_ICONS[type] || TYPE_ICONS.unknown; }
function typeClass(type) {
  return ['agent','tool','process','automation','user','iterator','variable'].includes(type)
    ? 't-' + type
    : '';
}

function setView(id) {
  document.querySelectorAll('.section').forEach(el => el.classList.toggle('active', el.id === id));
  document.querySelectorAll('.nav button').forEach(el => el.classList.toggle('active', el.dataset.view === id));
  document.getElementById('mainTitle').textContent = VIEW_TITLES[id] || id;
}

document.querySelectorAll('.nav button').forEach(btn => btn.addEventListener('click', () => setView(btn.dataset.view)));
document.getElementById('refreshBtn').addEventListener('click', loadState);
document.getElementById('openFlowBtn').addEventListener('click', () => setView('flow'));

async function loadState() {
  const res = await fetch('/api/state');
  STATE = await res.json();
  if (selectedUser === null && STATE.histories[0]) selectedUser = STATE.histories[0].user_id;
  if (selectedComponentName === null && STATE.components[0]) selectedComponentName = STATE.components[0].name;
  render();
}

function render() {
  const auto = STATE.annotated_automations || [];
  document.getElementById('projectName').textContent = STATE.project_name || STATE.project_dir || '';
  document.getElementById('projectPath').textContent = STATE.project_dir || '';
  document.getElementById('navFlowCount').textContent = auto.length;
  document.getElementById('navCompCount').textContent = (STATE.components || []).filter(c => c.type !== 'automation').length;
  document.getElementById('navHistCount').textContent = (STATE.histories || []).length;
  renderStatus();
  renderOverview();
  renderFlow();
  renderComponents();
  renderHistory();
  document.getElementById('rawState').textContent = JSON.stringify(STATE, null, 2);
}

function renderStatus() {
  const el = document.getElementById('status');
  if (!STATE.has_config) {
    el.innerHTML = `<div class="status-banner error">${esc(STATE.load_error || 'No config.json found')}</div>`;
  } else if (STATE.load_error) {
    el.innerHTML = `<div class="status-banner error">${esc(STATE.load_error)}</div>`;
  } else {
    el.innerHTML = '';
  }
}

/* ============ OVERVIEW ============ */

function renderOverview() {
  const components = STATE.components || [];
  const counts = { agent: 0, tool: 0, process: 0, automation: 0 };
  components.forEach(c => { if (counts[c.type] !== undefined) counts[c.type]++; });
  const totalMsgs = (STATE.histories || []).reduce((acc, h) => acc + (h.messages?.length || 0), 0);
  const stats = [
    { k: 'Components', v: components.length },
    { k: 'Agents', v: counts.agent, type: 'agent' },
    { k: 'Tools', v: counts.tool, type: 'tool' },
    { k: 'Processes', v: counts.process, type: 'process' },
    { k: 'Automations', v: counts.automation, type: 'automation' },
    { k: 'Conversations', v: (STATE.histories || []).length, type: 'user' },
    { k: 'History messages', v: totalMsgs }
  ];
  document.getElementById('stats').innerHTML = stats.map(s => `
    <div class="stat ${typeClass(s.type)}">
      <div class="v" style="color: var(--acc, var(--ink))">${esc(s.v)}</div>
      <div class="k">${esc(s.k)}</div>
    </div>`).join('');

  document.getElementById('overviewMini').innerHTML = components.map(c => `
    <div class="mini-card ${typeClass(c.type)}" data-comp="${esc(c.name)}">
      <div class="mc-icon icon-bg ${typeClass(c.type)}">${typeIcon(c.type)}</div>
      <div style="min-width:0">
        <div class="mc-name">${esc(c.name)}</div>
        <div class="mc-type">${esc(c.type || 'component')}</div>
      </div>
    </div>
  `).join('') || `<div class="empty">No components defined.</div>`;
  document.getElementById('overviewMini').querySelectorAll('.mini-card').forEach(el => {
    el.addEventListener('click', () => {
      selectedComponentName = el.dataset.comp;
      setView('components');
      compFilter = 'all';
      renderComponents();
      const target = document.querySelector(`.comp-card[data-comp="${CSS.escape(el.dataset.comp)}"]`);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  const gp = STATE.general_parameters || {};
  document.getElementById('overviewSysDesc').textContent = gp.general_system_description || '(no system description)';
  document.getElementById('overviewParams').innerHTML = renderGeneralParams(gp);
}

function renderGeneralParams(gp) {
  const rows = [];
  if (gp.timezone) rows.push(['Timezone', esc(gp.timezone)]);
  if (gp.history_folder) rows.push(['History folder', `<code>${esc(gp.history_folder)}</code>`]);
  if (gp.api_keys_path) rows.push(['API keys file', `<code>${esc(gp.api_keys_path)}</code>`]);
  if (gp.functions) rows.push(['Functions module', `<code>${esc(gp.functions)}</code>`]);
  if (Array.isArray(gp.default_models) && gp.default_models.length) {
    rows.push(['Default models',
      gp.default_models.map(m => `<span class="badge t-agent">${esc(m.provider)}: ${esc(m.model)}</span>`).join(' ')]);
  }
  if (Array.isArray(gp.variables) && gp.variables.length) {
    rows.push(['Variables',
      `<div class="field-grid">${gp.variables.map(v => `
        <div class="field">
          <div class="fk">${esc(v.key)}</div>
          <div class="fv">
            <span class="badge subtle">${esc(Array.isArray(v.type) ? v.type.join(' | ') : (v.type || 'any'))}</span>
            ${v.default !== undefined ? `<span class="badge t-variable">default: ${esc(v.default)}</span>` : ''}
          </div>
        </div>`).join('')}</div>`]);
  }
  if (!rows.length) return `<div class="empty">No general parameters set.</div>`;
  return `<div class="kv">${rows.map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${v}</div>`).join('')}</div>`;
}

/* ============ FLOW ============ */

function findComponent(name) {
  return (STATE.components || []).find(c => c.name === name);
}

const VIRTUAL_COMPONENTS = {
  user: { type: 'user', outputs_keys: ['response'], description: 'Latest user message.' },
  iterator: { type: 'iterator', outputs_keys: ['item', 'item_number'], description: 'Current iteration.' }
};

function componentTypeOf(name) {
  if (!name) return null;
  if (VIRTUAL_COMPONENTS[name]) return VIRTUAL_COMPONENTS[name].type;
  const comp = findComponent(name);
  return comp ? comp.type : null;
}

function fieldsOf(comp, name) {
  if (VIRTUAL_COMPONENTS[name]) return VIRTUAL_COMPONENTS[name].outputs_keys;
  if (!comp) return [];
  if (comp.type === 'agent' && comp.required_outputs) return Object.keys(comp.required_outputs);
  if (comp.type === 'tool' && comp.outputs) return Object.keys(comp.outputs);
  return [];
}

function indexLabel(idx) {
  if (idx === undefined || idx === null) return '';
  if (idx === -1) return 'latest';
  if (typeof idx === 'number' && idx < 0) return `${Math.abs(idx)} back`;
  return `#${idx}`;
}

function renderRefPill(source, opts) {
  if (!source || !source.component) return '';
  const t = componentTypeOf(source.component) || 'unknown';
  const cls = typeClass(t) || '';
  const fields = (source.fields || []).filter(Boolean);
  const idx = source.index;
  const idxText = indexLabel(idx);
  const dataField = fields.length ? ` data-field="${esc(fields[0])}"` : '';
  const hideIdx = (opts && opts.hideIndex) || idx === -1 || idx === undefined || idx === null;
  return `<span class="ref ${cls}" data-comp="${esc(source.component)}"${dataField} title="${esc(source.component)}${fields.length ? '.' + fields.join('.') : ''} (${idxText || '#?'})">
    ${fields.map(f => `<span class="ref-field">${esc(f)}</span>`).join('')}
    ${fields.length ? '<span class="ref-arrow">↩</span>' : ''}
    <span class="ref-comp">${esc(source.component)}</span>
    ${hideIdx ? '' : `<span class="ref-idx">${esc(idxText)}</span>`}
  </span>`;
}

function renderRefsFromParsed(parsed) {
  if (!parsed) return '';
  if (parsed.error) return `<span class="hl-context">parse error: ${esc(parsed.error)}</span>`;
  if (parsed.multiple_sources && parsed.multiple_sources.length) {
    return parsed.multiple_sources.map(s => renderRefPill(s)).join(' ');
  }
  if (parsed.single_source) {
    return renderRefPill(parsed.single_source);
  }
  if (parsed.selection) {
    const sel = parsed.selection.selector || {};
    const gi = parsed.selection.global_index;
    if (sel.type === 'all' && Array.isArray(gi)) {
      const anchor = gi[0];
      if (anchor && typeof anchor === 'object' && anchor.component) {
        return `<span class="hl-context">context · since</span> ${renderRefPill({ component: anchor.component, index: anchor.index, fields: [] }, { hideIndex: true })}`;
      }
      if (typeof anchor === 'number') {
        return `<span class="hl-context">context · last ${Math.abs(anchor)} msgs</span>`;
      }
      return `<span class="hl-context">all messages</span>`;
    }
    if (sel.sources && sel.sources.length) {
      return sel.sources.map(s => renderRefPill(s)).join(' ');
    }
    if (sel.type === 'all') return `<span class="hl-context">all messages</span>`;
  }
  return '';
}

function renderOutputPill(component, field, description) {
  const t = componentTypeOf(component) || 'unknown';
  const cls = typeClass(t);
  const tip = description ? ` title="${esc(description)}"` : '';
  return `<span class="out-pill ${cls}" data-out-comp="${esc(component)}" data-out-field="${esc(field)}"${tip}>${esc(field)}</span>`;
}

function renderOutputsRow(name, comp) {
  const keys = fieldsOf(comp, name);
  if (!keys.length) return '';
  const descMap = comp && comp.required_outputs ? comp.required_outputs
                : comp && comp.outputs ? comp.outputs
                : {};
  return `<div class="io-row">
    <span class="io-label">outputs</span>
    <div class="io-pills">${keys.map(k => renderOutputPill(name, k, descMap[k])).join('')}</div>
  </div>`;
}

function renderInputsRow(parsed) {
  const html = renderRefsFromParsed(parsed);
  if (!html) return '';
  return `<div class="io-row">
    <span class="io-label">inputs</span>
    <div class="io-pills">${html}</div>
  </div>`;
}

function renderFlow() {
  const auto = STATE.annotated_automations || [];
  const picker = document.getElementById('autoPick');
  if (!auto.length) {
    picker.innerHTML = '';
    document.getElementById('flowCanvas').innerHTML = `<div class="empty">No automations defined.</div>`;
    document.getElementById('flowDescription').textContent = '';
    document.getElementById('flowDetail').innerHTML = `<div class="empty">No automations defined.</div>`;
    return;
  }
  picker.innerHTML = auto.map((a, i) => `<button data-idx="${i}" class="${i === selectedAutomation ? 'active' : ''}">${esc(a.name)}</button>`).join('');
  picker.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
    selectedAutomation = parseInt(btn.dataset.idx, 10);
    renderFlow();
  }));

  const a = auto[selectedAutomation] || auto[0];
  document.getElementById('flowDescription').textContent = a.description || '';

  const userOutputs = renderOutputsRow('user', null);
  const canvas = document.getElementById('flowCanvas');
  canvas.innerHTML = `
    <div class="flow-vertical">
      <div class="fc-node t-user" data-step-component="user" data-step-payload='${esc(JSON.stringify({ kind: 'entry', automation_name: a.name }))}'>
        <div class="node-head">
          <div class="nm-icon icon-bg t-user">${typeIcon('user')}</div>
          <div>
            <div class="nm-name">User input</div>
            <div class="nm-meta">Trigger of automation <code>${esc(a.name)}</code></div>
          </div>
          <span class="badge t-user">user</span>
        </div>
        ${userOutputs}
      </div>
      ${renderSequence(a.annotated_sequence, 'root')}
    </div>
  `;
  attachFlowInteractions(canvas);
  renderFlowDetail({ kind: 'automation_root', automation: a });
}

function attachFlowInteractions(canvas) {
  canvas.querySelectorAll('[data-step-payload]').forEach(el => {
    el.addEventListener('click', (e) => {
      // Only trigger when clicking a node directly, not when bubbling up from a ref pill click
      if (e.target.closest('.ref') || e.target.closest('.out-pill')) return;
      e.stopPropagation();
      try {
        const payload = JSON.parse(el.dataset.stepPayload);
        renderFlowDetail(payload);
        canvas.querySelectorAll('.fc-node.selected, .fc-container.selected').forEach(n => n.classList.remove('selected'));
        el.classList.add('selected');
      } catch (err) { console.error(err); }
    });
  });

  const highlightFor = (comp, field) => {
    if (!comp) return;
    canvas.querySelectorAll(`.ref[data-comp="${CSS.escape(comp)}"]`).forEach(n => {
      if (!field || !n.dataset.field || n.dataset.field === field) n.classList.add('highlight');
    });
    canvas.querySelectorAll(`.out-pill[data-out-comp="${CSS.escape(comp)}"]`).forEach(n => {
      if (!field || n.dataset.outField === field) n.classList.add('highlight');
    });
    canvas.querySelectorAll(`.fc-node[data-step-component="${CSS.escape(comp)}"]`).forEach(n => n.classList.add('highlight'));
  };
  const clearHighlights = () => {
    canvas.querySelectorAll('.highlight').forEach(n => n.classList.remove('highlight'));
  };

  canvas.addEventListener('mouseover', (e) => {
    const ref = e.target.closest('.ref');
    if (ref) {
      clearHighlights();
      highlightFor(ref.dataset.comp, ref.dataset.field);
      return;
    }
    const out = e.target.closest('.out-pill');
    if (out) {
      clearHighlights();
      highlightFor(out.dataset.outComp, out.dataset.outField);
      return;
    }
    const node = e.target.closest('.fc-node[data-step-component]');
    if (node) {
      clearHighlights();
      highlightFor(node.dataset.stepComponent, null);
    }
  });
  canvas.addEventListener('mouseout', (e) => {
    const stillIn = e.relatedTarget && (e.relatedTarget.closest('.ref') || e.relatedTarget.closest('.out-pill') || e.relatedTarget.closest('.fc-node[data-step-component]'));
    if (!stillIn) clearHighlights();
  });
}

function renderSequence(seq, idPrefix) {
  if (!Array.isArray(seq) || !seq.length) {
    return '';
  }
  const parts = [];
  seq.forEach((step, i) => {
    parts.push(`<div class="fc-arrow"></div>`);
    parts.push(renderStep(step, `${idPrefix}-${i}`));
  });
  return parts.join('');
}

function renderStep(step, id) {
  if (!step || typeof step !== 'object') {
    return `<div class="fc-node"><div class="node-head"><div class="nm-icon"></div><div><div class="nm-name">${esc(step)}</div></div></div></div>`;
  }
  if (step.kind === 'component_step') return renderComponentStep(step, id);
  if (step.kind === 'for') return renderForStep(step, id);
  if (step.kind === 'while') return renderWhileStep(step, id);
  if (step.kind === 'switch') return renderSwitchStep(step, id);
  if (step.kind === 'branch') return renderBranchStep(step, id);
  return `<div class="fc-node t-automation" data-step-payload='${esc(JSON.stringify({ kind: 'unknown', step }))}'>
    <div class="node-head">
      <div class="nm-icon icon-bg t-automation">${typeIcon('automation')}</div>
      <div>
        <div class="nm-name">${esc(step.kind || 'node')}</div>
        <div class="nm-meta">unknown step</div>
      </div>
    </div>
  </div>`;
}

function renderComponentStep(step, id) {
  const parsed = step.parsed_input || {};
  const name = parsed.component_or_param || step.raw;
  const comp = findComponent(name);
  const t = comp ? comp.type : (componentTypeOf(name) || 'unknown');
  const desc = comp && comp.description ? comp.description : '';
  const inputsHtml = renderInputsRow(parsed);
  const outputsHtml = renderOutputsRow(name, comp);
  const payload = { kind: 'component_step', step, comp };
  return `<div class="fc-node ${typeClass(t)}" data-step-component="${esc(name)}" data-step-payload='${esc(JSON.stringify(payload))}'>
    <div class="node-head">
      <div class="nm-icon icon-bg ${typeClass(t)}">${typeIcon(t)}</div>
      <div>
        <div class="nm-name">${esc(name)}</div>
        ${desc ? `<div class="nm-meta">${esc(desc)}</div>` : ''}
      </div>
      <span class="badge ${typeClass(t)}">${esc(t)}</span>
    </div>
    ${inputsHtml}
    ${outputsHtml}
  </div>`;
}

function renderForStep(step, id) {
  const refHtml = renderRefsFromParsed(step.parsed_items);
  const itemsRaw = step.raw && step.raw.items ? step.raw.items : '';
  const fallbackText = refHtml ? '' : `<code style="font-size:11px">${esc(itemsRaw)}</code>`;
  return `<div class="fc-container t-iterator" style="--acc: var(--iterator); --acc-bg: var(--iterator-bg); --acc-ink: var(--iterator-ink)" data-step-component="iterator" data-step-payload='${esc(JSON.stringify({ kind: 'for', step }))}'>
    <div class="fc-container-label">${typeIcon('for')} For each item in</div>
    <div class="fc-container-sub">${refHtml || fallbackText}</div>
    ${renderSequence(step.body || [], id + '-body')}
  </div>`;
}

function renderWhileStep(step, id) {
  const cond = step.raw && (step.raw.end_condition || step.raw.condition) || '';
  return `<div class="fc-container t-iterator" style="--acc: var(--iterator); --acc-bg: var(--iterator-bg); --acc-ink: var(--iterator-ink)" data-step-payload='${esc(JSON.stringify({ kind: 'while', step }))}'>
    <div class="fc-container-label">${typeIcon('while')} While</div>
    <div class="fc-container-sub"><code style="font-size:11px">${esc(cond)}</code></div>
    ${renderSequence(step.body || [], id + '-body')}
  </div>`;
}

function renderSwitchStep(step, id) {
  const refHtml = renderRefsFromParsed(step.parsed_value);
  const valueRaw = step.raw && step.raw.value || '';
  const fallbackText = refHtml ? '' : `<code style="font-size:11px">${esc(valueRaw)}</code>`;
  const cases = step.cases || [];
  return `<div class="fc-container" style="--acc: var(--switch); --acc-bg: var(--switch-bg); --acc-ink: var(--switch-ink)" data-step-payload='${esc(JSON.stringify({ kind: 'switch', step }))}'>
    <div class="fc-container-label">${typeIcon('switch')} Switch on</div>
    <div class="fc-container-sub">${refHtml || fallbackText}</div>
    <div class="fc-switch-cases">
      ${cases.map((c, i) => `
        <div class="fc-case">
          <div class="fc-case-label" title="${esc(c.case)}">${esc(c.case)}</div>
          ${renderCaseBody(c.body || [], `${id}-c${i}`)}
        </div>
      `).join('')}
    </div>
  </div>`;
}

function renderBranchStep(step, id) {
  const cond = step.raw && step.raw.condition || '';
  return `<div class="fc-container" style="--acc: var(--branch); --acc-bg: var(--branch-bg); --acc-ink: var(--branch-ink)" data-step-payload='${esc(JSON.stringify({ kind: 'branch', step }))}'>
    <div class="fc-container-label">${typeIcon('branch')} Branch</div>
    <div class="fc-container-sub"><span class="hl-context">if</span> <code style="font-size:11px">${esc(cond)}</code></div>
    <div class="fc-switch-cases">
      <div class="fc-case">
        <div class="fc-case-label true">true</div>
        ${renderCaseBody(step.if_true || [], id + '-true')}
      </div>
      <div class="fc-case">
        <div class="fc-case-label false">false</div>
        ${renderCaseBody(step.if_false || [], id + '-false')}
      </div>
    </div>
  </div>`;
}

function renderCaseBody(body, id) {
  if (!body || !body.length) return `<div class="fc-case-empty">no-op</div>`;
  return body.map((step, i) => {
    const html = renderStep(step, `${id}-${i}`);
    return i === 0 ? html : `<div class="fc-arrow"></div>${html}`;
  }).join('');
}

function renderFlowDetail(payload) {
  const root = document.getElementById('flowDetail');
  if (!payload) {
    root.innerHTML = `<div class="empty">Click any node to inspect it.</div>`;
    return;
  }
  if (payload.kind === 'automation_root') {
    const a = payload.automation;
    root.innerHTML = `
      <h3>${esc(a.name)} <span class="badge t-automation">automation</span></h3>
      ${a.description ? `<p style="color: var(--muted); font-size: 13px; margin: 0 0 14px">${esc(a.description)}</p>` : ''}
      <div class="group">
        <h4>Steps</h4>
        <pre>${fmtJSON(a.sequence)}</pre>
      </div>`;
    return;
  }
  if (payload.kind === 'component_step') {
    const comp = payload.comp;
    const step = payload.step;
    const parsed = step.parsed_input || {};
    const t = comp ? comp.type : 'unknown';
    let body = `<h3>${esc(parsed.component_or_param || step.raw)} <span class="badge ${typeClass(t)}">${esc(t || 'step')}</span></h3>`;
    if (comp) {
      body += renderComponentDetail(comp, true);
    } else {
      body += `<p style="color: var(--muted); font-size:13px">No component named <code>${esc(parsed.component_or_param || '')}</code>.</p>`;
    }
    body += `<div class="group"><h4>Step expression</h4><pre>${esc(step.raw)}</pre></div>`;
    body += `<div class="group"><h4>Resolved inputs</h4><pre>${fmtJSON(parsed)}</pre></div>`;
    root.innerHTML = body;
    return;
  }
  if (payload.kind === 'for' || payload.kind === 'while') {
    const step = payload.step;
    root.innerHTML = `
      <h3>${esc(payload.kind === 'for' ? 'For loop' : 'While loop')} <span class="badge t-automation">control flow</span></h3>
      <div class="group"><h4>Definition</h4><pre>${fmtJSON(step.raw)}</pre></div>`;
    return;
  }
  if (payload.kind === 'switch') {
    const step = payload.step;
    root.innerHTML = `
      <h3>Switch <span class="badge" style="background: var(--switch-bg); color: var(--switch-ink)">control flow</span></h3>
      <div class="group"><h4>Switch on</h4><pre>${esc(step.raw && step.raw.value || '')}</pre></div>
      <div class="group"><h4>Cases</h4>${(step.cases || []).map(c => `<div><span class="badge" style="background: var(--switch-bg); color: var(--switch-ink)">${esc(c.case)}</span> &rarr; ${(c.body || []).map(b => esc(b.raw && (typeof b.raw === 'string' ? b.raw : (b.raw.control_flow_type || JSON.stringify(b.raw))) || b.kind)).join(', ') || '<em>no-op</em>'}</div>`).join('')}</div>`;
    return;
  }
  if (payload.kind === 'branch') {
    const step = payload.step;
    root.innerHTML = `
      <h3>Branch <span class="badge" style="background: var(--branch-bg); color: var(--branch-ink)">control flow</span></h3>
      <div class="group"><h4>Condition</h4><pre>${esc(step.raw && step.raw.condition || '')}</pre></div>`;
    return;
  }
  root.innerHTML = `<pre>${fmtJSON(payload)}</pre>`;
}

function renderComponentDetail(comp, compact) {
  const items = [];
  if (comp.description) items.push(['Description', esc(comp.description)]);
  if (comp.type === 'agent') {
    if (comp.system) items.push(['System prompt', `<div class="system-prompt">${esc(comp.system)}</div>`]);
    if (comp.required_outputs) items.push(['Required outputs', renderIOList(comp.required_outputs)]);
    if (comp.default_output) items.push(['Default output', `<pre>${fmtJSON(comp.default_output)}</pre>`]);
    if (Array.isArray(comp.models) && comp.models.length) {
      items.push(['Models', comp.models.map(m => `<span class="badge t-agent">${esc(m.provider)}: ${esc(m.model)}</span>`).join(' ')]);
    }
    if (comp.model_params) items.push(['Model params', `<pre>${fmtJSON(comp.model_params)}</pre>`]);
  }
  if (comp.type === 'tool') {
    if (comp.function) items.push(['Function', `<code>${esc(comp.function)}</code>`]);
    if (comp.inputs) items.push(['Inputs', renderIOList(comp.inputs)]);
    if (comp.outputs) items.push(['Outputs', renderIOList(comp.outputs)]);
    if (comp.default_output) items.push(['Default output', `<pre>${fmtJSON(comp.default_output)}</pre>`]);
  }
  if (comp.type === 'process') {
    if (comp.function) items.push(['Function', `<code>${esc(comp.function)}</code>`]);
    if (comp.expected_params) items.push(['Expected params', (comp.expected_params || []).map(p => `<code>${esc(p)}</code>`).join(' ')]);
  }
  if (comp.type === 'automation') {
    if (Array.isArray(comp.sequence)) items.push(['Sequence', `<pre>${fmtJSON(comp.sequence)}</pre>`]);
  }
  if (!items.length) return '';
  return items.map(([k, v]) => `<div class="group"><h4>${esc(k)}</h4>${v}</div>`).join('');
}

function renderIOList(map) {
  if (!map || !Object.keys(map).length) return `<em style="color: var(--muted)">none</em>`;
  return `<div class="io-list">${Object.keys(map).map(k => `
    <div class="io">
      <div class="nm">${esc(k)}</div>
      <div class="ds">${esc(map[k])}</div>
    </div>
  `).join('')}</div>`;
}

/* ============ COMPONENTS ============ */

function renderComponents() {
  const components = STATE.components || [];
  const types = ['all', ...new Set(components.map(c => c.type).filter(Boolean))];
  const tools = document.getElementById('compFilters');
  tools.innerHTML = types.map(t => `
    <button class="filter-chip ${compFilter === t ? 'active' : ''}" data-filter="${esc(t)}">${esc(t === 'all' ? 'All' : t)}</button>
  `).join('') + (components.length ? '' : '');
  tools.querySelectorAll('.filter-chip').forEach(b => b.addEventListener('click', () => {
    compFilter = b.dataset.filter;
    renderComponents();
  }));
  const visible = compFilter === 'all' ? components : components.filter(c => c.type === compFilter);
  const grid = document.getElementById('componentsGrid');
  if (!visible.length) {
    grid.innerHTML = `<div class="empty">No components match this filter.</div>`;
    return;
  }
  grid.innerHTML = visible.map(c => `
    <article class="comp-card ${typeClass(c.type)}" data-comp="${esc(c.name)}">
      <div class="comp-head">
        <div class="ic">${typeIcon(c.type)}</div>
        <div style="min-width:0; flex:1">
          <div class="nm">${esc(c.name)}</div>
          <div><span class="badge ${typeClass(c.type)}">${esc(c.type || 'component')}</span></div>
        </div>
      </div>
      ${c.description ? `<div class="comp-desc">${esc(c.description)}</div>` : ''}
      ${renderComponentDetail(c, false)}
    </article>
  `).join('');
}

/* ============ HISTORY ============ */

function renderHistory() {
  const histories = STATE.histories || [];
  const userList = document.getElementById('userList');
  const messages = document.getElementById('historyMessages');
  if (!histories.length) {
    userList.innerHTML = `<div class="empty">No history databases found.</div>`;
    messages.innerHTML = '';
    return;
  }
  userList.innerHTML = histories.map(h => {
    const initials = (h.user_id || '?').slice(0, 2).toUpperCase();
    return `
      <button class="user-btn ${h.user_id === selectedUser ? 'active' : ''}" data-user="${esc(h.user_id)}">
        <div class="avatar">${esc(initials)}</div>
        <div class="meta">
          <div class="uid">${esc(h.user_id)}</div>
          <div class="cnt">${(h.messages || []).length} messages</div>
        </div>
      </button>`;
  }).join('');
  userList.querySelectorAll('.user-btn').forEach(btn => btn.addEventListener('click', () => {
    selectedUser = btn.dataset.user;
    renderHistory();
  }));
  const history = histories.find(h => h.user_id === selectedUser) || histories[0];
  if (history.error) {
    messages.innerHTML = `<div class="status-banner error">${esc(history.error)}</div>`;
    return;
  }
  if (!history.messages || !history.messages.length) {
    messages.innerHTML = `<div class="empty">No messages in this history.</div>`;
    return;
  }
  // Group messages into "turns" - a turn starts at a user message
  const turns = [];
  let current = null;
  history.messages.forEach(m => {
    if (m.type === 'user' || m.role === 'user') {
      if (current) turns.push(current);
      current = { user: m, steps: [] };
    } else if (current) {
      current.steps.push(m);
    } else {
      current = { user: null, steps: [m] };
    }
  });
  if (current) turns.push(current);
  messages.innerHTML = turns.map((t, i) => renderTurn(t, i)).join('');
  messages.querySelectorAll('.meta-toggle').forEach(btn => btn.addEventListener('click', () => {
    const key = btn.dataset.key;
    if (openMeta.has(key)) openMeta.delete(key); else openMeta.add(key);
    btn.parentElement.querySelector('.meta-content').classList.toggle('open');
    btn.querySelector('.lbl').textContent = openMeta.has(key) ? 'Hide details' : 'Show details';
  }));
}

function renderTurn(turn, idx) {
  const userPreview = turn.user ? extractText(turn.user) : '(no user message)';
  const turnNo = turn.user ? `#${turn.user.msg_number}` : `#${idx + 1}`;
  const ts = turn.user && turn.user.timestamp ? formatTimestamp(turn.user.timestamp) : '';
  return `<div class="turn">
    <div class="turn-head">
      <div class="badge t-user">${typeIcon('user')} user</div>
      <div class="turn-no">${esc(turnNo)}</div>
      <div class="turn-user-text"><span class="preview">${esc(userPreview)}</span></div>
      <div style="color: var(--muted); font-size: 12px">${esc(ts)}</div>
    </div>
    <div class="step-list">
      ${turn.steps.map(s => renderStepRow(s)).join('') || '<div class="empty">No steps recorded.</div>'}
    </div>
  </div>`;
}

function renderStepRow(message) {
  const t = (message.type || message.role || '').toLowerCase();
  const tcls = ['agent','tool','process','automation','user','iterator','variable'].includes(t) ? 't-' + t : 't-automation';
  const ts = formatTimestamp(message.timestamp);
  return `<div class="step ${tcls}">
    <div class="step-rail"><div class="icon-circle">${typeIcon(t || 'unknown')}</div></div>
    <div class="step-body">
      <div class="step-name">
        <span>${esc(message.role || message.type || 'step')}</span>
        <span class="badge ${tcls}">${esc(message.type || message.role || '')}</span>
        ${message.model ? `<span class="badge subtle">${esc(message.model)}</span>` : ''}
        <span class="ts">${esc(ts)}</span>
      </div>
      <div class="step-content">${(message.blocks || []).map((b, i) => renderHistoryBlock(b, message, i)).join('')}</div>
    </div>
  </div>`;
}

function renderHistoryBlock(block, message, idx) {
  const type = block.type || 'text';
  const metaKey = `${message.msg_number}-${idx}`;
  let body = '';
  if (type === 'variable') {
    body = `<div class="field-grid">
      <div class="field"><div class="fk">key</div><div class="fv"><code>${esc(block.key || '')}</code></div></div>
      <div class="field"><div class="fk">value</div><div class="fv"><code>${esc(typeof block.value === 'object' ? JSON.stringify(block.value) : (block.value ?? ''))}</code></div></div>
    </div>`;
  } else if (type === 'image' || type === 'file') {
    body = renderFields({ ...block });
  } else {
    body = renderContent(block.content !== undefined ? block.content : block);
  }
  const meta = block.metadata ? renderBlockMetadata(block.metadata, metaKey) : '';
  return body + meta;
}

function renderContent(content) {
  if (content === null || content === undefined) return '<em style="color: var(--muted)">empty</em>';
  if (typeof content === 'string') return `<div class="text-bubble">${esc(content)}</div>`;
  if (typeof content === 'number' || typeof content === 'boolean') return `<code>${esc(String(content))}</code>`;
  if (Array.isArray(content)) {
    if (!content.length) return '<em style="color: var(--muted)">empty list</em>';
    if (content.every(v => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')) {
      return `<div class="pill-list">${content.map(v => `<span>${esc(String(v))}</span>`).join('')}</div>`;
    }
    return `<div class="field-grid">${content.map((v, i) => `<div class="field"><div class="fk">[${i}]</div><div class="fv">${renderContent(v)}</div></div>`).join('')}</div>`;
  }
  if (typeof content === 'object') {
    const keys = Object.keys(content);
    if (!keys.length) return '<em style="color: var(--muted)">empty object</em>';
    return renderFields(content);
  }
  return esc(String(content));
}

function renderFields(obj) {
  return `<div class="field-grid">${Object.keys(obj).map(k => {
    const v = obj[k];
    let inner;
    if (v === null || v === undefined) inner = '<em style="color: var(--muted)">null</em>';
    else if (typeof v === 'string') inner = `<div class="text-bubble">${esc(v)}</div>`;
    else if (typeof v === 'number' || typeof v === 'boolean') inner = `<code>${esc(String(v))}</code>`;
    else if (Array.isArray(v) && v.every(x => typeof x === 'string' || typeof x === 'number' || typeof x === 'boolean')) {
      inner = v.length ? `<div class="pill-list">${v.map(x => `<span>${esc(String(x))}</span>`).join('')}</div>` : '<em style="color: var(--muted)">empty list</em>';
    }
    else inner = `<pre>${fmtJSON(v)}</pre>`;
    return `<div class="field"><div class="fk">${esc(k)}</div><div class="fv">${inner}</div></div>`;
  }).join('')}</div>`;
}

function renderBlockMetadata(metadata, key) {
  const stats = [];
  const provider = metadata && metadata.provider_response;
  if (provider) {
    if (provider.provider) stats.push(['provider', provider.provider]);
    if (provider.model) stats.push(['model', provider.model]);
    const usage = provider.usage || {};
    if (usage.input_tokens !== undefined) stats.push(['input', usage.input_tokens + ' tok']);
    if (usage.output_tokens !== undefined) stats.push(['output', usage.output_tokens + ' tok']);
    if (usage.total_tokens !== undefined) stats.push(['total', usage.total_tokens + ' tok']);
    const total_time = provider.usage && provider.usage.raw && provider.usage.raw.total_time;
    if (total_time !== undefined) stats.push(['latency', (typeof total_time === 'number' ? total_time.toFixed(2) : total_time) + 's']);
  }
  const isOpen = openMeta.has(key);
  return `<div>
    <div class="meta-stats">${stats.map(([k, v]) => `<span class="meta-stat"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></span>`).join('')}</div>
    <button class="meta-toggle" data-key="${esc(key)}">
      <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 4l3 3 3-3" stroke="currentColor" fill="none" stroke-width="1.5"/></svg>
      <span class="lbl">${isOpen ? 'Hide details' : 'Show details'}</span>
    </button>
    <div class="meta-content ${isOpen ? 'open' : ''}"><pre>${fmtJSON(metadata)}</pre></div>
  </div>`;
}

function extractText(message) {
  for (const block of (message.blocks || [])) {
    const c = block && block.content !== undefined ? block.content : block;
    if (typeof c === 'string') return c;
    if (c && typeof c === 'object') {
      if (typeof c.response === 'string') return c.response;
      if (typeof c.text === 'string') return c.text;
      if (typeof c.content === 'string') return c.content;
    }
  }
  return '(no text content)';
}

function formatTimestamp(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

loadState();
</script>
</body>
</html>
"""


def serve_dashboard(
    project_dir: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    history_limit: int = 200,
) -> int:
    project_path = Path(project_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("[dashboard] " + format, *args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                return _html_response(self, HTML)
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                limit = history_limit
                if "history_limit" in query:
                    try:
                        limit = max(1, int(query["history_limit"][0]))
                    except Exception:
                        pass
                return _json_response(self, build_dashboard_state(project_path, history_limit=limit))
            return _json_response(self, {"error": "not found"}, status=404)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    try:
        with Server((host, port), Handler) as httpd:
            actual_host, actual_port = httpd.server_address
            url = f"http://{actual_host}:{actual_port}"
            print(f"MAS dashboard running at {url}")
            print(f"Project: {project_path}")
            if not (project_path / "config.json").is_file():
                print("Warning: no config.json found in this directory.")
            if open_browser:
                webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0
    except OSError as exc:
        print(f"Could not start dashboard on {host}:{port}: {exc}")
        return 1


def find_free_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred))
            return sock.getsockname()[1]
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def dashboard_main(args: argparse.Namespace) -> int:
    port = args.port
    if args.auto_port:
        port = find_free_port(args.host, args.port)
    return serve_dashboard(
        args.directory,
        host=args.host,
        port=port,
        open_browser=not args.no_browser,
        history_limit=args.history_limit,
    )
