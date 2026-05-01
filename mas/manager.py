from __future__ import annotations

from ._shared import *
from .components import Agent, Automation, Component, Process, Tool
from .parser import Parser
from .bots import TelegramBot, WhatsappBot

class AgentSystemManager:

    def __init__(
        self,
        config: str = None,
        imports: List[str] = None,
        base_directory: str = os.getcwd(),
        verbose: bool = False,
        bootstrap_models: Optional[List[Dict[str, str]]] = None,
        api_keys_path: Optional[str] = None,
        costs_path: str = None,
        history_folder: Optional[str] = None,
        files_folder: Optional[str] = None,
        general_system_description: str = "This is a multi agent system.",
        functions: Union[str, List[str]] = "fns.py",
        default_models: List[Dict[str, str]] = [{"provider": "groq", "model": "llama-3.1-8b-instant"}],
        on_update: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        include_timestamp: bool = False,
        timezone: str = "UTC",
        log_level=None,
        admin_user_id: Optional[str] = None,
        usage_logging: bool = False,
        timeout: Optional[int] = None
    ):

        self._tls = threading.local()
        self._db_pool = {}
        self._db_locks = {}
        self._db_write_lock = threading.RLock()
        self.base_directory = os.path.abspath(base_directory)
        self._usage_logging_enabled = bool(usage_logging)
        self.general_system_description = general_system_description
        self.on_update = self._resolve_callable(on_update)
        self.on_complete = self._resolve_callable(on_complete)
        self.api_keys_path = None

        self.timeout = timeout
        
        description_mode = (
            isinstance(config, str)
            and not config.lower().endswith(".json")
            and not os.path.exists(config)
        )

        if description_mode:
            self._resolve_api_keys_path(api_keys_path)
            self._load_api_keys()

            config = self._bootstrap_from_description(
                description       = config,
                base_directory    = base_directory,
                default_models    = bootstrap_models,
                verbose           = verbose,
                api_keys_path     = api_keys_path
            )
        
        if log_level is not None:
            logger.setLevel(log_level)

        self._file_cache = {}
        self._component_order: List[str] = []

        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.processes: Dict[str, Process] = {}
        self.automations: Dict[str, Automation] = {}

        self.parser = Parser()

        self.default_models = default_models

        if not imports:
            self.imports = []
        elif isinstance(imports, str):
            self.imports = [imports]
        elif isinstance(imports, list):
            self.imports = imports
        else:
            raise ValueError(f"Imports must be list or string")

        self._processed_imports = set()

        self.admin_user_id = str(admin_user_id) if admin_user_id is not None else None

        self.include_timestamp = include_timestamp
        self.timezone = timezone

        self._function_cache = {}
        self.functions = functions

        self.costs_path = costs_path
        self.costs = {}

        if isinstance(self.functions, str):
            self.functions = [self.functions]
        else:
            self.functions = self.functions
        
        if config and config.endswith(".json"):
            try:
                self.build_from_json(config)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.exception(f"System build failed: {e}")
                raise
        
        if not self.api_keys_path:
            self._resolve_api_keys_path(api_keys_path)

        try:
            self.timezone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as e:
            logger.warning(f"Invalid timezone {self.timezone}, defaulting to UTC. Error: {e}")
            self.timezone = ZoneInfo("UTC")

        self.api_keys: Dict[str, str] = {}
        self._load_api_keys()
        self._load_costs()

        if self.timeout is None:
            self.timeout = 120

        self._process_imports()

        if hasattr(self, 'pending_links'):
            for agent_name, tool_name in self.pending_links:
                try:
                    self.link(agent_name, tool_name)
                except ValueError as e:
                    logger.error(f"Warning: Could not link agent '{agent_name}' to tool '{tool_name}'. Error: {e}")

        self._last_known_update = None

        if history_folder is not None:
            if os.path.isabs(history_folder):
                self.history_folder = history_folder
            else:
                self.history_folder = os.path.join(self.base_directory, history_folder)
        else:
            if getattr(self, "history_folder", None) is None:
                self.history_folder = os.path.join(self.base_directory, "history")
        os.makedirs(self.history_folder, exist_ok=True)

        if files_folder is not None:
            if os.path.isabs(files_folder):
                self.files_folder = files_folder
            else:
                self.files_folder = os.path.join(self.base_directory, files_folder)
        else:
            if getattr(self, "files_folder", None) is None:
                self.files_folder = os.path.join(self.base_directory, "files")
        os.makedirs(self.files_folder, exist_ok=True)

        if self._usage_logging_enabled:
            self.logs_folder = os.path.join(self.base_directory, "logs")
            os.makedirs(self.logs_folder, exist_ok=True)
            self._usage_log_path   = os.path.join(self.logs_folder, "usage.log")
            self._summary_log_path = os.path.join(self.logs_folder, "summary.log")

    def _recursively_parse_json_strings(self, data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                new_dict[key] = self._recursively_parse_json_strings(value)
            return new_dict
        elif isinstance(data, list):
            return [self._recursively_parse_json_strings(item) for item in data]
        elif isinstance(data, str):
            try:
                s = data.strip()
                if (s.startswith('{') and s.endswith('}')) or \
                   (s.startswith('[') and s.endswith(']')):
                    return json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return data
        return data


    def _bootstrap_from_description(self, *, description: str,
                                    base_directory: str,
                                    default_models: Optional[List[Dict[str, str]]],
                                    verbose: bool = False,
                                    api_keys_path: Optional[str] = None) -> str:

        readme_text = get_readme("mneuronico", "multi-agent-system-library")

        combined_input = textwrap.dedent(f"""\
            ## MAS Library README
            {readme_text}

            ## USER REQUIREMENT
            {description}

            """)
        
        if default_models is None:
            default_models = [
                {"provider": "openai", "model": "gpt-5"},
                {"provider": "google", "model": "gemini-2.5-pro"},
                {"provider": "anthropic", "model": "claude-sonnet-4"},
            ]
        elif isinstance(default_models, dict):
            default_models = [default_models]

        
        recommended_models = {
            "openai": "gpt-5-nano",
            "openrouter": "openai/gpt-5-nano",
            "google": "gemini-2.5-flash",
            "anthropic": "claude-sonnet-4",
            "deepseek": "deepseek-chat",
            "groq": "openai/gpt-oss-120b",
            "wavespeed": "moonshotai/kimi-k2.5",
            "nvidia": "nvidia/llama-3.1-nemotron-nano-8b-v1"
        }
        potential_providers = ["openai", "openrouter", "google", "anthropic", "deepseek", "groq", "wavespeed", "nvidia"]

        available_providers = [p for p in potential_providers if self.get_key(p)]
        available_providers_str = ", ".join(available_providers) or "None"

        context_prompt = textwrap.dedent(f"""
        ## System Context & Constraints
        You must generate a `config.json` file. Pay close attention to these rules:

        1.  **Available API Providers:** The user has API keys for the following providers ONLY: **{available_providers_str}**. You MUST NOT include any other providers in the `default_models` list. If the list of available providers is empty, create an empty `default_models` list.
        2.  **Recommended Models:** When choosing models for the `default_models` list, you MUST use these preferred models. Select only from your list of available providers:
            - openai: {recommended_models['openai']}
            - openrouter: {recommended_models['openrouter']}
            - google: {recommended_models['google']}
            - anthropic: {recommended_models['anthropic']}
            - deepseek: {recommended_models['deepseek']}
            - groq: {recommended_models['groq']}
            - wavespeed: {recommended_models['wavespeed']}
            - nvidia: {recommended_models['nvidia']}
        3.  **Functions File:** If your configuration requires custom functions, include them in the `fns` list as full Python source code strings. Ensure that any function you reference in the components is defined here. If those functions rely on external libraries, you must include the necessary import statements at the top of the `fns.py` file, and attempt to install them via pip if the import fails.
        4.  **Processes and Tools:** Remember that ALL Tool components require that a previous component's outputs have keys that match the inputs required by that tool. Sometimes this means requiring outputs for an agent that produces those tool inputs. Sometimes that means building a Process that converts something (for example, the user message) into the correct tool input format. Inside process or tool functions, you can use the manager passed as an argument, and in particular manager.read() might be useful to get info out of messages and blocks (check the docs).
        """)

        original_system_prompt = (
            "You are `system_writer`, an expert MAS engineer.\n"
            "Your job is to read the user's requirement and the MAS library README, then use the provided context and constraints to create a valid system configuration.\n"
            "You must OUTPUT **ONE** JSON object with exactly these keys:\n"
            "  general_parameters -> A JSON object for the config.\n"
            "  components         -> A list of MAS component definitions.\n"
            "  fns                -> A list of strings, where each string is the full Python source for a function.\n"
            "Return NOTHING ELSE. No markdown, no explanations, just the raw JSON object."
        )

        bootstrap_cfg = {
            "general_parameters": {
                "base_directory": base_directory,
                "default_models": default_models
            },
            "components": [
                {
                    "type": "agent",
                    "name": "system_writer",
                    "system": f"{original_system_prompt}\n\n{context_prompt}",
                    "required_outputs": {
                        "general_parameters": "Block for config.json, respecting the provided constraints.",
                        "components": "Component list for the user's requirement.",
                        "fns": "Python functions needed by the components."
                    }
                }
            ]
        }

        sub_mgr = None
        bootstrap_cfg_path = os.path.join(base_directory,
                                          "_bootstrap_config.json")
        

        try:
            with open(bootstrap_cfg_path, "w", encoding="utf-8") as fp:
                json.dump(bootstrap_cfg, fp, indent=2)

            sub_mgr = AgentSystemManager(config=bootstrap_cfg_path,
                                        base_directory=base_directory)

            blocks = sub_mgr.run(
                input=combined_input,
                component_name="system_writer",
                verbose = verbose
            )

            agent_payload = None
            raw_payload_excerpt = None
            for blk in blocks:
                if blk.get("type") == "text":
                    raw = blk.get("content")
                    if isinstance(raw, dict):
                        agent_payload = raw
                        try:
                            raw_payload_excerpt = json.dumps(raw, ensure_ascii=False)
                        except TypeError:
                            raw_payload_excerpt = str(raw)
                    else:
                        raw_payload_excerpt = str(raw)
                        try:
                            agent_payload = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            agent_payload = None
                    break

            if raw_payload_excerpt and len(raw_payload_excerpt) > 500:
                raw_payload_excerpt = raw_payload_excerpt[:497] + '...'

            if not isinstance(agent_payload, dict):
                detail = f" Received payload: {raw_payload_excerpt}" if raw_payload_excerpt else ""
                raise RuntimeError("system_writer did not return a valid JSON object with the expected keys." + detail)

            cleaned_payload = self._recursively_parse_json_strings(agent_payload)

            required_keys = ("general_parameters", "components", "fns")
            missing = [key for key in required_keys if key not in cleaned_payload]
            if missing:
                available = ", ".join(map(str, cleaned_payload.keys())) or 'none'
                raise RuntimeError(
                    f"system_writer output is missing required sections after parsing: "
                    f"{', '.join(missing)}. Available keys: {available}."
                )

            gp       = cleaned_payload["general_parameters"]
            comps    = cleaned_payload["components"]
            fns_list = cleaned_payload["fns"]

            final_cfg_path = os.path.join(base_directory, "config.json")
            with open(final_cfg_path, "w", encoding="utf-8") as fp:
                json.dump({"general_parameters": gp, "components": comps}, fp, indent=2)

            final_fns_path = os.path.join(base_directory, "fns.py")
            if isinstance(fns_list, list):
                with open(final_fns_path, "w", encoding="utf-8") as fp:
                    fp.write("\n\n".join(map(str, fns_list)))

            return final_cfg_path

        finally:
            if sub_mgr:
                try:
                    sub_mgr._close_all_db_conns()
                except Exception:
                    pass

            # Borrar las carpetas temporales creadas por el sub_mgr
            for folder in ("history", "files"):
                path = os.path.join(base_directory, folder)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)

            # Borrar el archivo de config temporal
            if os.path.exists(bootstrap_cfg_path):
                try:
                    os.remove(bootstrap_cfg_path)
                except OSError:
                    pass

    def _resolve_api_keys_path(self, api_keys_path):
        if api_keys_path is None:
            env_path = os.path.join(self.base_directory, ".env")
            json_path = os.path.join(self.base_directory, "api_keys.json")
            
            if os.path.exists(env_path):
                self.api_keys_path = env_path
            elif os.path.exists(json_path):
                self.api_keys_path = json_path
            else:
                self.api_keys_path = None
        else:
            if os.path.isabs(api_keys_path):
                self.api_keys_path = api_keys_path
            else:
                self.api_keys_path = os.path.join(self.base_directory, api_keys_path)

    def _process_imports(self):
        for import_str in self.imports:
            self._process_single_import(import_str)

    def _uid(self):
        return getattr(self._tls, "current_user_id", None)
    
    def _process_single_import(self, import_str: str):
        if import_str in self._processed_imports:
            return
        self._processed_imports.add(import_str)

        file_part = import_str.strip()
        components = None

        if "->" in import_str:
            file_part, single_comp = import_str.split("->", 1)
            file_part = file_part.strip()
            single_comp = single_comp.strip()
            components = [single_comp]
        
        elif "?" in import_str:
            file_part, comps_part = import_str.split("?", 1)
            file_part = file_part.strip()
            comps_part = comps_part.strip()
            if comps_part.startswith("[") and comps_part.endswith("]"):
                inside = comps_part[1:-1].strip()
                if inside:
                    components = [c.strip() for c in inside.split(",") if c.strip()]
                else:
                    components = None

        base, ext = os.path.splitext(file_part)
        if not ext:
            ext = ".json"
            file_part = base + ext
        
        resolved_json_path = None
        if not os.path.isabs(file_part):
            candidate_local = os.path.join(self.base_directory, file_part)
            candidate_local = os.path.normpath(candidate_local)
            if os.path.exists(candidate_local):
                resolved_json_path = candidate_local
            else:
                short_name = os.path.splitext(os.path.basename(file_part))[0]
                resolved_json_path = self._load_local_json_if_exists(short_name)
        else:
            if os.path.exists(file_part):
                resolved_json_path = file_part

        if not resolved_json_path:
            raise FileNotFoundError(f"Could not locate JSON file for '{file_part}' "
                                    f"locally or in GitHub 'lib' folder.")

        temp_manager = AgentSystemManager(config=resolved_json_path)
        self._merge_imported_components(temp_manager, components)

    def _load_local_json_if_exists(self, filename_no_ext: str) -> str:
        json_file_name = f"{filename_no_ext}.json"

        try:
            file_text = resources.read_text("mas.lib", json_file_name)
        except FileNotFoundError:
            return None
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_file.write(file_text.encode("utf-8"))
        tmp_file.flush()
        tmp_file.close()
        return tmp_file.name
    
    def _merge_imported_components(self, other_mgr, only_names):

        for name, agent_obj in other_mgr.agents.items():
            if only_names and name not in only_names:
                continue
            if name not in self.agents:
                agent_obj.manager = self
                self.agents[name] = agent_obj
                self._component_order.append(name)

        for name, tool_obj in other_mgr.tools.items():
            if only_names and name not in only_names:
                continue
            if name not in self.tools:
                tool_obj.manager = self
                self.tools[name] = tool_obj
                self._component_order.append(name)

        for name, proc_obj in other_mgr.processes.items():
            if only_names and name not in only_names:
                continue
            if name not in self.processes:
                proc_obj.manager = self
                self.processes[name] = proc_obj
                self._component_order.append(name)

        for name, auto_obj in other_mgr.automations.items():
            if only_names and name not in only_names:
                continue
            if name not in self.automations:
                auto_obj.manager = self
                self.automations[name] = auto_obj
                self._component_order.append(name)

    def _component_exists(self, name: str) -> bool:
        return any(
            name in getattr(self, registry)
            for registry in ['agents', 'tools', 'processes', 'automations']
        )

    def _resolve_callable(self, func):
        if isinstance(func, str) and ":" in func:
            return self._get_function_from_string(func)
        elif callable(func):
            return func
        return None

    def _invoke_callback(
        self,
        callback: Optional[Callable],
        messages,
        manager=None,
        params: Optional[Dict] = None,
    ):
        if not callback:
            return None

        manager = manager or self
        try:
            sig = inspect.signature(callback)
        except (TypeError, ValueError):
            if params is not None:
                return callback(messages, manager, params)
            return callback(messages, manager)

        parameters = list(sig.parameters.values())
        accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
        positional = [
            p for p in parameters
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if accepts_varargs or len(positional) >= 3:
            if params is not None:
                return callback(messages, manager, params)
            return callback(messages, manager)
        if len(positional) == 2:
            return callback(messages, manager)
        if len(positional) == 1:
            return callback(messages)
        return callback()
        
    def clear_file_cache(self):
        self._file_cache = {}
        
    def to_string(self) -> str:
        components_details = []

        for category, components in [
            ("Agents", self.agents),
            ("Tools", self.tools),
            ("Processes", self.processes),
            ("Automations", self.automations),
        ]:
            if components:
                components_details.append(f"{category}:\n")
                for name, component in components.items():
                    components_details.append(f"  {component.to_string()}\n")

        components_summary = "".join(components_details) if components_details else "No components registered."

        return (
            f"AgentSystemManager(\n"
            f"  Base Directory: {self.base_directory}\n"
            f"  Current User ID: {self._uid()}\n"
            f"  General System Description: {self.general_system_description}\n\n"
            f"Components:\n"
            f"{components_summary}\n"
            f")"
        )

    def _load_api_keys(self):
        self.api_keys = {}

        if self.api_keys_path is None or not os.path.exists(self.api_keys_path):
            return
        
        if self.api_keys_path.endswith('.json'):
            try:
                with open(self.api_keys_path, "r", encoding="utf-8") as f:
                    self.api_keys = json.load(f)
            except OSError as e:
                logger.error(f"Error opening API keys file: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {self.api_keys_path}: {e}")

        elif '.env' in self.api_keys_path:
            load_dotenv(self.api_keys_path, override=True)
        else:
            raise ValueError(f"Unsupported API keys format: {self.api_keys_path}")
        
    def _load_costs(self):
        if self.costs_path and os.path.exists(self.costs_path):
            try:
                with open(self.costs_path, "r", encoding="utf-8") as f:
                    self.costs = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read costs file: {e}")
                self.costs = {}
        else:
            auto = os.path.join(self.base_directory, "costs.json")
            if os.path.exists(auto):
                self.costs_path = auto
                self._load_costs()
            else:
                self.costs = {}

    def _price_for_model(self, provider: str, model: str):
        prov = self.costs.get("models", {}).get(provider.lower(), {})
        mdl  = prov.get(model, {})
        in_p  = mdl.get("input_per_1m", 0.0)
        out_p = mdl.get("output_per_1m", 0.0)
        return in_p / 1_000_000.0, out_p / 1_000_000.0

    def _price_for_tool(self, tool_name: str):
        return self.costs.get("tools", {}).get(tool_name, {}).get("per_call", 0.0)

    def _price_for_stt(self, provider: str, model: str):
        return (
            self.costs.get("stt", {})
                    .get(provider.lower(), {})
                    .get(model, {})
                    .get("per_minute", 0.0)
        )
    
    def _log_stt_call(self,
                      provider: str,
                      model: str,
                      seconds: float,
                      transcription: str):
        usd = round(self._price_for_stt(provider, model) *
                    (seconds / 60.0), 10)

        blocks = self._to_blocks({"transcription": transcription},
                                 user_id=self._active_user_id())
        blocks[0].setdefault("metadata", {})
        md = blocks[0]["metadata"]
        md["usd_cost"] = usd
        md["seconds"]  = seconds

        self._save_message(
            self._get_user_db(),
            role   = f"stt_{provider.lower()}",
            content= blocks,
            type   = "tool",
            model  = f"stt:{provider}:{model}"
        )

    def cost_model_call(self, provider, model, in_tokens, out_tokens):
        in_pp, out_pp = self._price_for_model(provider, model)
        return round(in_tokens * in_pp + out_tokens * out_pp, 10)

    def cost_tool_call(self, tool_name):
        return self._price_for_tool(tool_name)
    
    def _now_iso(self):
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _write_usage_entry(self, entry: dict, force_summary: bool = False):
        entry["ts"] = self._now_iso()
        with open(self._usage_log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

        lines_since = getattr(self, "_lines_since_refresh", 0) + 1
        self._lines_since_refresh = lines_since

        if force_summary or lines_since >= 100:
            self._refresh_cost_summary()
            self._lines_since_refresh = 0

    def _iter_usage_rows(self):
        if not os.path.isfile(self._usage_log_path):
            return
        with open(self._usage_log_path, encoding="utf-8") as fp:
            for line in fp:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def _bucket_for(self, ts_iso, span):
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - ts
        return {
            "minute":  delta.total_seconds() <= 60,
            "hour":    delta.total_seconds() <= 3600,
            "day":     delta.total_seconds() <= 86400,
            "week":    delta.total_seconds() <= 604800,
            "month":   delta.total_seconds() <= 2_592_000,   # 30d
            "year":    delta.total_seconds() <= 31_536_000   # 365d
        }[span]

    def _refresh_cost_summary(self):
        spans = ["minute", "hour", "day", "week", "month", "year", "lifetime"]
        agg   = {s: collections.defaultdict(lambda: {
                    "calls":0, "input_tokens":0, "output_tokens":0,
                    "seconds":0, "usd_cost":0.0}) for s in spans}

        for row in self._iter_usage_rows():
            for span in spans:
                if span == "lifetime" or self._bucket_for(row["ts"], span):
                    key = row["id"]
                    bucket = agg[span][key]
                    bucket["calls"]         += 1
                    bucket["usd_cost"]      += row.get("cost", 0.0)
                    bucket["input_tokens"]  += row.get("in_toks", 0)
                    bucket["output_tokens"] += row.get("out_toks", 0)
                    bucket["seconds"]       += row.get("seconds", 0)

        summary = {}
        for span, data in agg.items():
            total_cost = sum(v["usd_cost"] for v in data.values())
            users      = {row["user"] for row in self._iter_usage_rows()
                        if span == "lifetime" or self._bucket_for(row["ts"], span)}
            avg_per_user = total_cost / max(len(users), 1)

            summary[span] = {
                "overall_cost": round(total_cost, 6),
                "avg_cost_per_user": round(avg_per_user, 6),
                "by_id": {k: {**v, "usd_cost": round(v["usd_cost"], 6)} for k, v in data.items()}
            }

        with open(self._summary_log_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)

    def _log_if_enabled(self, entry: dict):
        if self._usage_logging_enabled:
            self._write_usage_entry(entry)

    def get_cost_report(self, span="lifetime", user=None, model_or_tool=None):
        if not os.path.isfile(self._summary_log_path):
            self._refresh_cost_summary()
        with open(self._summary_log_path, encoding="utf-8") as fp:
            data = json.load(fp)
        span_data = data.get(span, {})
        if user or model_or_tool:
            # build filtered ad-hoc from raw lines
            result = {"overall_cost": 0, "avg_cost_per_user": 0, "by_id": {}}
            users_set = set()
            for row in self._iter_usage_rows():
                if span != "lifetime" and not self._bucket_for(row["ts"], span):
                    continue
                if user and row["user"] != str(user):
                    continue
                if model_or_tool and row["id"] != model_or_tool:
                    continue
                bucket = result["by_id"].setdefault(row["id"], {
                    "calls":0,"input_tokens":0,"output_tokens":0,"seconds":0,"usd_cost":0.0})
                bucket["calls"]        += 1
                bucket["usd_cost"]     += row.get("cost", 0.0)
                bucket["input_tokens"] += row.get("in_toks", 0)
                bucket["output_tokens"]+= row.get("out_toks",0)
                bucket["seconds"]      += row.get("seconds",0)
                result["overall_cost"] += row.get("cost", 0.0)
                users_set.add(row["user"])
            if users_set:
                result["avg_cost_per_user"] = round(result["overall_cost"]/len(users_set),6)
            result["overall_cost"] = round(result["overall_cost"],6)
            for v in result["by_id"].values():
                v["usd_cost"] = round(v["usd_cost"],6)
            return result
        return span_data

    def _get_db_path_for_user(self, user_id: str) -> str:
        return os.path.join(self.history_folder, f"{self._safe_user_id(user_id)}.sqlite")

    def _safe_user_id(self, user_id: Any) -> str:
        raw = str(user_id)
        if (
            raw
            and raw not in {".", ".."}
            and ".." not in raw
            and re.fullmatch(r"[A-Za-z0-9_.-]+", raw)
        ):
            return raw
        encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
        return f"u_{encoded or 'empty'}"

    def _ensure_user_db(self, user_id: str) -> sqlite3.Connection:
        if not hasattr(self, "_db_pool"):
            self._db_pool = {}
        if not hasattr(self, "_db_locks"):
            self._db_locks = {}
        user_id = str(user_id)
        if user_id in self._db_pool:
            return self._db_pool[user_id]
        db_path = self._get_db_path_for_user(user_id)

        conn = sqlite3.connect(db_path, check_same_thread=False)

        self._migrate_history_schema(conn)

        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        self._db_pool[user_id] = conn
        self._db_locks.setdefault(user_id, threading.RLock())
        return conn
    
    def _active_user_id(self) -> Optional[str]:
        if hasattr(self, "_tls") and getattr(self._tls, "current_user_id", None):
            return self._tls.current_user_id
        return None


    def export_history(self, user_id: str) -> bytes:
        db_path = self._get_db_path_for_user(user_id)

        if hasattr(self, "_db_pool") and user_id in self._db_pool:
            conn = self._db_pool[user_id]
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception as e:
                logger.warning(f"Could not checkpoint database for user {user_id} before export: {e}")

            with open(db_path, "rb") as f:
                return f.read()
        else:
            return b""
    
    def import_history(self, user_id: str, sqlite_bytes: bytes):
        if hasattr(self, "_db_pool") and user_id in self._db_pool:
            try:
                self._db_pool[user_id].close()
            except Exception as e:
                logger.warning(f"Error closing existing connection for user {user_id} before import: {e}")
            del self._db_pool[user_id]

        if hasattr(self, "_tls") and getattr(self._tls, "current_user_id", None) == user_id:
            if hasattr(self._tls, "db_conn") and self._tls.db_conn is not None:
                try:
                    self._tls.db_conn.close()
                except Exception: pass
            self._tls.db_conn = None
                    
        db_path = self._get_db_path_for_user(user_id)
        #if os.path.exists(db_path):
        #    logger.warning(f"Overwriting existing history for user {user_id}")
        with open(db_path, "wb") as f:
            f.write(sqlite_bytes)


    def _create_table(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS message_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                msg_number INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT
            )
        """)
        conn.commit()

    def _migrate_history_schema(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='message_history';")
        if not cur.fetchone():
            self._create_table(conn)

        cur.execute("PRAGMA table_info(message_history)")
        columns = {row[1] for row in cur.fetchall()}
        migrations = []
        if "msg_number" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN msg_number INTEGER")
        if "role" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if "content" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN content TEXT NOT NULL DEFAULT ''")
        if "type" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN type TEXT NOT NULL DEFAULT 'user'")
        if "timestamp" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN timestamp DATETIME")
        if "model" not in columns:
            migrations.append("ALTER TABLE message_history ADD COLUMN model TEXT")

        for statement in migrations:
            cur.execute(statement)

        if "msg_number" not in columns:
            cur.execute("UPDATE message_history SET msg_number = id WHERE msg_number IS NULL")
        if "timestamp" not in columns:
            cur.execute(
                "UPDATE message_history SET timestamp = ? WHERE timestamp IS NULL",
                (datetime.now(self.timezone).isoformat(),)
            )

        cur.execute("PRAGMA user_version = 1")
        conn.commit()

    def has_new_updates(self) -> bool:
        db_conn = self._get_user_db()
        cur = db_conn.cursor()
        cur.execute("SELECT MAX(timestamp) FROM message_history")
        latest_timestamp = cur.fetchone()[0]

        if latest_timestamp != self._last_known_update:
            self._last_known_update = latest_timestamp
            return True
        return False
    

    def set_current_user(self, user_id: str):
        user_id = str(user_id)

        if not hasattr(self, "_tls"):
            self._tls = threading.local()

        self._tls.current_user_id = user_id
        self._tls.db_conn         = self._ensure_user_db(user_id)


    def ensure_user(self, user_id: Optional[str] = None) -> None:
        if user_id:
            self.set_current_user(user_id)
            return
        if not hasattr(self, "_tls"):
            self._tls = threading.local()
        if getattr(self._tls, "current_user_id", None) is None:
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)


    def _get_user_db(self) -> sqlite3.Connection:
        if not hasattr(self, "_tls"):
            self._tls = threading.local()
        conn = getattr(self._tls, "db_conn", None)
        if conn is not None:
            return conn

        self.ensure_user()
        return self._tls.db_conn


    def _get_next_msg_number(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(msg_number), 0) FROM message_history")
        max_num = cur.fetchone()[0]
        return max_num + 1

    def _save_message(self, conn: sqlite3.Connection, role: str, content: Union[str, dict, list], type="user", model: Optional[str] = None):
        timestamp = datetime.now(self.timezone).isoformat()

        content = self._to_blocks(content, user_id=self._uid())
        if isinstance(content, (dict, list)):
            content_str = json.dumps(
                self._persist_non_json_values(content, self._uid()),
                indent=2, ensure_ascii=False
            )
        else:
            content_str = str(content)

        uid = self._uid()
        lock = self._db_locks.get(str(uid)) if uid is not None and hasattr(self, "_db_locks") else None
        lock = lock or getattr(self, "_db_write_lock", threading.RLock())

        with lock:
            cur = conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                cur.execute("SELECT COALESCE(MAX(msg_number), 0) + 1 FROM message_history")
                next_num = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO message_history (msg_number, role, content, type, model, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (next_num, role, content_str, type, model, timestamp)
                )
                conn.commit()
                return next_num
            except Exception:
                conn.rollback()
                raise


    def add_message(self, role: str, content: Union[str, dict], msg_type: str = "developer") -> None:
        db_conn = self._get_user_db()
        self._save_message(db_conn, role, content, msg_type)

    def _save_component_output(
        self,
        db_conn,
        component,
        output_value,
        verbose=False
    ):
        if isinstance(component, Automation):
            return

        model_info = getattr(component, "_last_used_model", None)
        if model_info:
            del component._last_used_model
            model_str = f"{model_info['provider']}:{model_info['model']}"
        else:
            model_str = None

        if output_value is None:
            return

        save_role      = component.name
        component_type = (
            "agent"      if isinstance(component, Agent)      else
            "tool"       if isinstance(component, Tool)       else
            "process"    if isinstance(component, Process)    else
            "automation"
        )

        self._save_message(
            db_conn,
            save_role,
            output_value,
            component_type,
            model_str
        )

        meta = self._extract_metadata_from_blocks(output_value)
        if not meta:
            return

        try:
            entry = {
                "user":   self._active_user_id(),
                "type":   component_type,
                "id":     model_str or save_role,
                "in_toks":  meta.get("input_tokens", 0),
                "out_toks": meta.get("output_tokens", 0),
                "seconds": meta.get("seconds", 0),
                "cost":    meta.get("usd_cost", 0.0)
            }
            self._log_if_enabled(entry)
        except Exception as e:
            logger.warning(f"Could not log usage entry: {e}")
        
    def _dict_to_json_with_file_persistence(self, data: dict, user_id: str) -> str:
        data_copy = self._persist_non_json_values(data, user_id)
        return json.dumps(data_copy, indent=2, ensure_ascii=False)

    def _persist_non_json_values(self, value: Any, user_id: str) -> Any:
        if isinstance(value, dict):
            return {k: self._persist_non_json_values(v, user_id) for k, v in value.items()}

        if isinstance(value, list):
            return [self._persist_non_json_values(item, user_id) for item in value]

        if isinstance(value, (bytes, bytearray)):
            return self._store_file(value, user_id)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return self._store_file(value, user_id)
    
    

    def _detect_extension(self, data: bytes, fallback=".bin") -> str:
        ext = what(None, h=data)
        if ext:
            return f".{ext}"
        if data[:3] == b"ID3":
            return ".mp3"
        if data[:4] == b"OggS":
            return ".ogg"
        return fallback

    def _store_file(self, obj: Any, user_id: str) -> str:
        user_id = str(user_id)
        files_dir = os.path.join(self.files_folder, self._safe_user_id(user_id))
        os.makedirs(files_dir, exist_ok=True)

        if isinstance(obj, str) and os.path.isfile(obj):
            ext = os.path.splitext(obj)[1] or ".bin"
            new_path = os.path.join(files_dir, f"{uuid.uuid4()}{ext}")
            shutil.copy2(obj, new_path)
            self._file_cache[new_path] = obj
            return f"file:{new_path}"

        if isinstance(obj, (bytes, bytearray)):
            ext = self._detect_extension(obj)
            new_path = os.path.join(files_dir, f"{uuid.uuid4()}{ext}")
            with open(new_path, "wb") as f:
                f.write(obj)
            self._file_cache[new_path] = bytes(obj)
            return f"file:{new_path}"

        new_path = os.path.join(files_dir, f"{uuid.uuid4()}.pkl")
        with open(new_path, "wb") as f:
            pickle.dump(obj, f)
        self._file_cache[new_path] = obj
        return f"file:{new_path}"

    def _load_files_in_dict(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

        if isinstance(value, dict):
            return {k: self._load_files_in_dict(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._load_files_in_dict(item) for item in value]

        if isinstance(value, str) and value.startswith("file:"):
            return self._load_file(value[5:])

        return value
       
    def _load_file(self, file_path: str) -> Any:
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pkl":
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
            else:
                with open(file_path, "rb") as f:
                    obj = f.read()

            self._file_cache[file_path] = obj
            return obj
        except:
            #logger.warning(f"[Manager] File not found while loading from history: {file_path}")
            return f"Error: Archivo '{os.path.basename(file_path)}' no encontrado en el historial."
        

    def add_blocks(
        self,
        content,
        *,
        role: str = "user",
        msg_type: str = "user",
        user_id: Union[str, None] = None
    ) -> int:
        self.ensure_user(user_id)
        conn = self._get_user_db()
        msg_number = self._save_message(conn, role, content, msg_type)
        return msg_number

    
    def _filter_blocks_by_fields(self, blocks, fields):
        if not fields:
            return blocks

        new_blocks = copy.deepcopy(blocks)
        for b in new_blocks:
            if b.get("type") == "text":
                raw = b["content"]

                if isinstance(raw, dict):
                    sub = {k: raw[k] for k in fields if k in raw}
                    b["content"] = sub
                    continue

                if isinstance(raw, str):
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        sub = {k: obj[k] for k in fields if k in obj}
                        b["content"] = sub
        return new_blocks
    
    def _get_all_messages(self, conn: sqlite3.Connection, include_model = False) -> List[tuple]:
        cur = conn.cursor()

        if include_model:
            cur.execute("SELECT role, content, msg_number, type, model, timestamp FROM message_history ORDER BY msg_number ASC")
        else:
            cur.execute("SELECT role, content, msg_number, type, timestamp FROM message_history ORDER BY msg_number ASC")
        return cur.fetchall()

    def show_history(self, user_id: Optional[str] = None, message_char_limit: int = 2000):
        if user_id is not None:
            self.set_current_user(user_id)

        conn = self._get_user_db()
        rows = self._get_all_messages(conn, include_model = True)
        logger.info(f"=== Message History for user [{self._uid()}] ===")
        for i, (role, content, msg_number, msg_type, model, timestamp) in enumerate(rows, start=1):
            model_str = f" ({model})" if model else ""
            role_str = f"{msg_number}. {msg_type} - {role}{model_str}" if role != "user" else f"{msg_number}. {role}{model_str}"
            if len(content) > message_char_limit:
                content = content[:message_char_limit] + "...\nMessage too long to show in history."
            logger.info(f"{role_str}: {content}")
        logger.info("============================================\n")

    def get_messages(self, user_id: Optional[str] = None) -> List[Dict[str, str]]:
        if user_id is not None:
            self.set_current_user(user_id)

        conn = self._get_user_db()
        rows = self._get_all_messages(conn, include_model=True)

        messages = []
        for role, content, msg_number, msg_type, model, timestamp in rows:
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                except json.JSONDecodeError:
                    content_data = content
            else:
                content_data = content

            messages.append({
                "source": role,
                "message": content_data,
                "msg_number": msg_number,
                "type": msg_type,
                "model": model,
                "timestamp": timestamp
            })

        return messages
    
    def _extract_metadata_from_blocks(self, content):
        if isinstance(content, dict):
            return content.get("metadata", {})

        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if "metadata" in blk:
                    return blk["metadata"]

                if blk.get("type") == "text":
                    raw = blk.get("content")
                    if isinstance(raw, dict) and "metadata" in raw:
                        return raw["metadata"]
                    if isinstance(raw, str):
                        try:
                            obj = json.loads(raw)
                            if isinstance(obj, dict) and "metadata" in obj:
                                return obj["metadata"]
                        except json.JSONDecodeError:
                            pass
        return {}
    
    def get_usage_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        messages = self.get_messages(user_id)
        models = {}
        tools  = {}

        for msg in messages:
            mtype = msg["type"]
            src   = msg["source"]
            content = msg["message"]
            metadata = self._extract_metadata_from_blocks(content)
            cost     = metadata.get("usd_cost", 0.0)

            if mtype == "agent":
                key = msg.get("model") or "unknown"
                m = models.setdefault(key, {"input_tokens": 0, "output_tokens": 0, "usd_cost": 0.0})
                m["input_tokens"]  += metadata.get("input_tokens", 0)
                m["output_tokens"] += metadata.get("output_tokens", 0)
                m["usd_cost"]      += cost

            elif mtype == "tool":
                t = tools.setdefault(src, {"calls": 0, "usd_cost": 0.0})
                t["calls"]     += 1
                t["usd_cost"]  += cost

        models_overall = {"input_tokens": 0, "output_tokens": 0, "usd_cost": 0.0}
        for v in models.values():
            models_overall["input_tokens"]  += v["input_tokens"]
            models_overall["output_tokens"] += v["output_tokens"]
            models_overall["usd_cost"]      += v["usd_cost"]

        tools_overall = {"calls": 0, "usd_cost": 0.0}
        for v in tools.values():
            tools_overall["calls"]    += v["calls"]
            tools_overall["usd_cost"] += v["usd_cost"]

        grand_total = models_overall["usd_cost"] + tools_overall["usd_cost"]

        models["overall"] = models_overall
        tools["overall"] = tools_overall

        return {
            "models": models,
            "tools":  tools,
            "overall": {"usd_cost": grand_total}
        }

    def _build_agent_prompt(self,
        general_desc: str,
        name: str,
        specific_desc: str, 
        required_outputs: dict
    ) -> str:

        prompt = f"System general description: {general_desc.strip()}\n"
    
        lines = []
        lines.append(f"You are {name}, an agent who is part of this system.")
        lines.append(f"Your task has been defined as follows:{specific_desc.strip()}\n")
        lines.append("You must always and only answer in JSON format.")
        lines.append("Below is the JSON structure you must produce:")
        lines.append("")
        lines.append("{")
    
        for field_name, info in required_outputs.items():
            if isinstance(info, dict):
                field_type = info.get("type", "string")
                field_desc = info.get("description", str(info))
                line = f'  "{field_name}" ({field_type}): "{field_desc}",'
            else:
                field_desc = info
                line = f'  "{field_name}": "{field_desc}",'
            lines.append(line)
    
        if len(lines) > 3:
            last_line = lines[-1]
            if last_line.endswith(","):
                lines[-1] = last_line[:-1]
    
        lines.append("}")
    
        prompt += "\n".join(lines)

        prompt += (
            "In this system, messages are exchanged between different components (users, agents, tools, etc.).\n"
            "To help you understand the origin of each message, they include a 'source', which indicates the originator of the message (e.g., 'user', 'my_agent', 'my_tool').\n"
            "When you receive a message, remember that the 'source' tells you who sent it.\n"
            "You do not need to include the 'source' in your output, since they will be added by the system. Focus solely on producing the JSON structure with the required fields defined above.\n"
            "You never need to include things like '```json' or anything else. Just the bare object: {'required_output_name': 'required_output_value', ...}, with one field per required output, is enough for the system to parse it correctly. If you do something differently, the system will fail.\n"
        )

        return prompt
    
    def create_agent(
        self,
        name: Optional[str] = None,
        system: str = "You are a helpful assistant",
        required_outputs: Union[Dict[str, Any], str] = {"response": "Text to send to user."},
        models: List[Dict[str, str]] = None,
        default_output: Optional[Dict[str, Any]] = {"response": "No valid response."},
        positive_filter: Optional[List[str]] = None,
        negative_filter: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        include_timestamp: Optional[bool] = None,
        description: str = None,
        timeout: Optional[int] = None
    ):
        if name is None:
            name = self._generate_agent_name()
    
        if name in self.agents:
            raise ValueError(f"[Manager] Agent '{name}' already exists.")

        if models is None:
            models = self.default_models.copy()
    
        if isinstance(required_outputs, str):
            required_outputs = {"response": required_outputs}
    
        final_prompt = self._build_agent_prompt(
            self.general_system_description,
            name,
            system,
            required_outputs
        )

        final_timeout = timeout if timeout is not None else self.timeout
    
        agent = Agent(
            name=name,
            system_prompt=final_prompt,
            system_prompt_original=system,
            required_outputs=required_outputs,
            models=models,
            default_output=default_output or {"response": "No valid response."},
            positive_filter=positive_filter,
            negative_filter=negative_filter,
            general_system_description=self.general_system_description,
            model_params=model_params,
            include_timestamp=include_timestamp if include_timestamp is not None else self.include_timestamp,
            description=description,
            timeout=final_timeout
        )

        agent.manager = self
        self.agents[name] = agent
        self._component_order.append(name)

        return name

    def _generate_agent_name(self) -> str:
        existing_names = [name for name in self.agents.keys() if name.startswith("agent-")]
        if not existing_names:
            return "agent-1"
    
        max_number = max(int(name.split("-")[1]) for name in existing_names if name.split("-")[1].isdigit())
        return f"agent-{max_number + 1}"

    def create_tool(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        function: Callable,
        default_output: Optional[Dict[str, Any]] = None,
        description: str = None
    ):
        if name in self.tools:
            raise ValueError(f"[Manager] Tool '{name}' already exists.")
        t = Tool(name, inputs, outputs, function, default_output, description)
        t.manager = self
        self.tools[name] = t
        self._component_order.append(name)

        return name

    def create_process(self, name: str, function: Callable, description: str = None):
        if name in self.processes:
            raise ValueError(f"[Manager] Process '{name}' already exists.")
        p = Process(name, function, description)
        p.manager = self
        self.processes[name] = p
        self._component_order.append(name)
        return name

    def create_automation(self, name: Optional[str] = None, sequence: List[Union[str, dict]] = None, description: str = None):
        if name is None:
            name = self._generate_automation_name()

        if name in self.automations:
            raise ValueError(f"Automation '{name}' already exists.")

        automation = Automation(name=name, sequence=sequence, description=description)
        automation.manager = self
        self.automations[name] = automation
        self._component_order.append(name)

        return name

    def _generate_automation_name(self) -> str:
        existing_names = [name for name in self.automations.keys() if name.startswith("automation-")]
        if not existing_names:
            return "automation-1"
    
        max_number = max(int(name.split("-")[1]) for name in existing_names if name.split("-")[1].isdigit())
        return f"automation-{max_number + 1}"
    
    def link_tool_to_agent_as_output(self, tool_name: str, agent_name: str):
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found.")
        tool = self.tools[tool_name]
        agent = self.agents[agent_name]

        if len(agent.required_outputs) == 1 and "response" in agent.required_outputs:
            del agent.required_outputs["response"]

        changed = False
        for inp_name, inp_desc in tool.inputs.items():
            if inp_name not in agent.required_outputs:
                agent.required_outputs[inp_name] = inp_desc
                changed = True

        if changed:
            new_prompt = self._build_agent_prompt(
                self.general_system_description,
                agent_name,
                agent.system_prompt_original,
                agent.required_outputs
            )
            agent.system_prompt = new_prompt

    def link_tool_to_agent_as_input(self, tool_name: str, agent_name: str):
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found.")
        tool = self.tools[tool_name]
        agent = self.agents[agent_name]

        if agent.negative_filter and f"{tool_name}" in agent.negative_filter:
            agent.negative_filter.remove(f"{tool_name}")

        if agent.positive_filter is not None and f"{tool_name}" not in agent.positive_filter:
            agent.positive_filter.append(f"{tool_name}")

        extra = "\n\nPay attention to the following tool output:\n"
        for out_name, out_desc in tool.outputs.items():
            extra += f"'{out_name}': {out_desc}\n"
        agent.system_prompt += extra

    def link(self, name1: str, name2: str):
        is_tool_1 = name1 in self.tools
        is_tool_2 = name2 in self.tools
        is_agent_1 = name1 in self.agents
        is_agent_2 = name2 in self.agents

        if is_tool_1 and is_agent_2:
            self.link_tool_to_agent_as_input(name1, name2)
        elif is_agent_1 and is_tool_2:
            self.link_tool_to_agent_as_output(name2, name1)
        else:
            raise ValueError(
                f"Invalid linkage: Ensure one of the components is a tool and the other is an agent.\n"
                f"Tools: {list(self.tools.keys())}, Agents: {list(self.agents.keys())}"
            )

    def _get_component(self, name: str) -> Optional[Component]:
        if name in self.agents:
            return self.agents[name]
        if name in self.tools:
            return self.tools[name]
        if name in self.processes:
            return self.processes[name]
        if name in self.automations:
            return self.automations[name]
        return None


    def _run_internal(
            self,
            component_name: Optional[str],
            input: Optional[Any],
            user_id: Optional[str],
            role: Optional[str],
            verbose: bool,
            target_input: Optional[str],
            target_index: Optional[int],
            target_custom: Optional[list],
            on_update: Optional[Callable],
            on_update_params: Optional[Dict] = None,
            return_token_count = False
        ) -> Dict:

        if user_id:
            self.set_current_user(user_id)
        elif not self._uid():
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)

        db_conn = self._get_user_db()

        if input is not None:
            store_role = role or "user"

            if store_role == "user":
                processed_input = input
                if isinstance(input, str):
                    processed_input = {"response": input}
                self.add_blocks(processed_input, role="user", msg_type="user")
                if verbose:
                    logger.debug("[Manager] Saved user input")
            else:
                self._save_message(db_conn, store_role, input, store_role)
                logger.debug("[Manager] Saved developer input")

        if component_name is None:
            if not self.automations or len(self.automations) < 1:
                if verbose:
                    logger.info("[Manager] Using default automation.")
                default_automation_name = "default_automation"
                default_automation_sequence = list(self._component_order)
                if default_automation_name not in self.automations:
                    self.create_automation(name=default_automation_name, sequence=default_automation_sequence)
                comp = self._get_component(default_automation_name)
            elif len(self.automations) == 1:
                automation_name = list(self.automations.keys())[0]
                if verbose:
                    logger.info(f"[Manager] Using single existing automation: {automation_name}")
                comp = self._get_component(automation_name)
            else:
                latest_automation_name = list(self.automations.keys())[-1]
                if verbose:
                    logger.info(f"[Manager] Using the latest automation: {latest_automation_name}")
                comp = self._get_component(latest_automation_name)
        else:
            comp = self._get_component(component_name)
            if not comp:
                raise ValueError(f"[Manager] Component '{component_name}' not found.")

        if isinstance(comp, Automation):
            if verbose:
                logger.debug(f"[Manager] Running Automation: {component_name or comp.name}")
            
            if on_update_params:
                output_dict = comp.run(
                    verbose=verbose,
                    on_update_params = on_update_params,
                    on_update=lambda messages, manager, on_update_params: self._invoke_callback(on_update, messages, manager, on_update_params),
                    return_token_count=return_token_count
                )
            else:
                output_dict = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager: self._invoke_callback(on_update, messages, manager),
                    return_token_count=return_token_count
                )
        else:
            if verbose:
                logger.debug(f"[Manager] Running {component_name or comp.name} with target_input={target_input}, "
                    f"target_index={target_index}, target_custom={target_custom}")
                
            if isinstance(comp, Agent):
                output_dict = comp.run(
                    verbose=verbose,
                    target_input=target_input,
                    target_index=target_index,
                    target_custom=target_custom,
                    return_token_count=return_token_count
                )
            else:
                output_dict = comp.run(
                    verbose=verbose,
                    target_input=target_input,
                    target_index=target_index,
                    target_custom=target_custom
                )
            if on_update:
                self._invoke_callback(on_update, self.get_messages(user_id), self, on_update_params)

        self._save_component_output(db_conn, comp, output_dict, verbose=verbose)

        if self._usage_logging_enabled:
            self._refresh_cost_summary()

        return output_dict


    def run(
        self,
        input: Optional[Any] = None,
        component_name: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        verbose: bool = False,
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_custom: Optional[list] = None,
        blocking: bool = True,
        on_update: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_update_params: Optional[Dict] = None,
        on_complete_params: Optional[Dict] = None,
        return_token_count = False
    ) -> Dict:
        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        run_user_id = user_id or self._uid()
        if not run_user_id:
            run_user_id = str(uuid.uuid4())
            self.set_current_user(run_user_id)

        if return_token_count and blocking:
            _uid = user_id or self._uid()
            prev_usage = self.get_usage_stats(_uid)
        else:
            prev_usage = None

        def task():
            self._run_internal(
                component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params, return_token_count
            )
            if on_complete:
                self._invoke_callback(on_complete, self.get_messages(user_id), self, on_complete_params)

        if blocking:
            self._run_internal(
                component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params, return_token_count
            )

            db_conn = self._get_user_db()
            cur = db_conn.cursor()
            cur.execute("SELECT content FROM message_history ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            
            last_content_str = row[0]
            final_blocks = json.loads(last_content_str)

            if return_token_count and prev_usage is not None:
                _uid = user_id or self._uid()
                post_usage = self.get_usage_stats(_uid)

                delta_in   = post_usage["models"]["overall"]["input_tokens"]   - prev_usage["models"]["overall"]["input_tokens"]
                delta_out  = post_usage["models"]["overall"]["output_tokens"]  - prev_usage["models"]["overall"]["output_tokens"]
                delta_mcost = post_usage["models"]["overall"]["usd_cost"]      - prev_usage["models"]["overall"]["usd_cost"]
                delta_tcost = post_usage["tools"]["overall"]["usd_cost"]       - prev_usage["tools"]["overall"]["usd_cost"]
                delta_total = post_usage["overall"]["usd_cost"]                - prev_usage["overall"]["usd_cost"]

                usage_summary = {
                    "input_tokens":  delta_in,
                    "output_tokens": delta_out,
                    "models_cost":   round(delta_mcost, 6),
                    "tools_cost":    round(delta_tcost, 6),
                    "total_cost":    round(delta_total, 6)
                }

                if verbose:
                    logger.info(
                        f"[Usage] +{delta_in} in / +{delta_out} out tokens -> "
                        f"${usage_summary['total_cost']:.6f} "
                        f"(models: ${usage_summary['models_cost']:.6f}, "
                        f"tools: ${usage_summary['tools_cost']:.6f})"
                    )

                if final_blocks and isinstance(final_blocks, list):
                    final_blocks[0].setdefault("metadata", {})
                    final_blocks[0]["metadata"]["usage_summary"] = usage_summary

            if on_complete:
                self._invoke_callback(on_complete, self.get_messages(user_id), self, on_complete_params)
            return final_blocks
        else:
            thread = threading.Thread(target=task)
            thread.daemon = True
            thread.start()
            return None

    def save_file(self, obj: Any, user_id = None) -> str:
        uid = str(user_id) if user_id is not None else str(self._active_user_id() or "unknown")
        self.ensure_user(uid)
        return self._store_file(obj, uid)


    def build_from_json(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                system_definition = json.load(f)
        except OSError as e:
            logger.error(f"Error opening JSON file at {json_path}: {e}")
            raise FileNotFoundError(f"Cannot open MAS config JSON at {json_path}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file at {json_path}: {e}")
            raise

        if not isinstance(system_definition, dict):
            raise ValueError(f"MAS config JSON at {json_path} must contain an object.")
        if not system_definition:
            raise ValueError(f"MAS config JSON at {json_path} is empty.")

        general_params = system_definition.get("general_parameters", {})

        self.base_directory = general_params.get("base_directory", self.base_directory)
        self.general_system_description = general_params.get("general_system_description", "This is a multi-agent system.")
        self.functions = general_params.get("functions", "fns.py")

        if isinstance(self.functions, str):
            self.functions = [self.functions]
        elif isinstance(self.functions, list):
            self.functions = self.functions
        else:
            self.functions = []

        self.default_models = general_params.get("default_models", self.default_models)
        self.on_update = self._resolve_callable(general_params.get("on_update"))
        self.on_complete = self._resolve_callable(general_params.get("on_complete"))
        self.include_timestamp = general_params.get("include_timestamp", False)
        self._usage_logging_enabled  = general_params.get("usage_logging", self._usage_logging_enabled)
        self.timezone = general_params.get("timezone", 'UTC')

        admin_from_cfg = general_params.get("admin_user_id", None)
        self.admin_user_id = str(admin_from_cfg) if admin_from_cfg is not None else getattr(self, "admin_user_id", None)

        history_folder = general_params.get("history_folder")
        if history_folder:
            if os.path.isabs(history_folder):
                self.history_folder = history_folder
            else:
                self.history_folder = os.path.join(self.base_directory, history_folder)
        else:
            self.history_folder = os.path.join(self.base_directory, "history")

        files_folder = general_params.get("files_folder")
        if files_folder:
            if os.path.isabs(files_folder):
                self.files_folder = files_folder
            else:
                self.files_folder = os.path.join(self.base_directory, files_folder)
        else:
            self.files_folder = os.path.join(self.base_directory, "files")

        
        imports = general_params.get("imports")

        if not imports:
            self.imports = []
        elif isinstance(imports, str):
            self.imports = [imports]
        elif isinstance(imports, list):
            self.imports = imports
        else:
            raise ValueError(f"Imports must be list or string")
        
        self._resolve_api_keys_path(general_params.get("api_keys_path"))

        self.costs_path = general_params.get("costs_path")
        self._load_costs()

        if self.timeout is None:
            self.timeout = general_params.get("timeout", 120)

        components = system_definition.get("components", [])
        for component in components:
            self._create_component_from_json(component)

        links = system_definition.get("links", {})
        for input_component, output_component in links.items():
            try:
                self.link(input_component, output_component)
            except ValueError as e:
                logger.error(f"Warning: Could not create link from '{input_component}' to '{output_component}'. Error: {e}")

    def _create_component_from_json(self, component: dict):
        component_type = component.get("type")
        name = component.get("name")
        description = component.get("description")

        if not component_type:
            raise ValueError("Component must have a 'type'.")
        
        if not name:
            if component_type == "agent":
                name = self._generate_agent_name()
            elif component_type == "automation":
                name = self._generate_automation_name()
            else:
                raise ValueError(
                    f"Component type '{component_type}' must have a 'name'."
                )

        if component_type == "agent":
            agent_name = self.create_agent(
                name=name,
                system=component.get("system", "You are a helpful assistant."),
                required_outputs=component.get("required_outputs", {"response": "Text to send to user."}),
                models=component.get("models"),
                default_output=component.get("default_output", {"response": "No valid response."}),
                positive_filter=component.get("positive_filter"),
                negative_filter=component.get("negative_filter"),
                include_timestamp=component.get("include_timestamp"),
                model_params=component.get("model_params"),
                description=description,
                timeout=component.get("timeout")
            )

            uses_tool = component.get("uses_tool")
            if uses_tool:
                if not hasattr(self, 'pending_links'):
                    self.pending_links = []
                self.pending_links.append((agent_name, uses_tool))

        elif component_type == "tool":
            fn = component.get("function")
            if not fn:
                raise ValueError("Tool must have a 'function'.")
          
            self.create_tool(
                name=name,
                inputs=component.get("inputs", {}),
                outputs=component.get("outputs", {}),
                function=self._get_function_from_string(fn),
                default_output=component.get("default_output", None),
                description=description
            )

        elif component_type == "process":
            fn = component.get("function")
            if not fn:
                raise ValueError("Process must have a 'function'.")
            self.create_process(
                name=name,
                function=self._get_function_from_string(fn),
                description=description
            )

        elif component_type == "automation":
            self.create_automation(
                name=name,
                sequence=self._resolve_automation_sequence(component.get("sequence", [])),
                description=description
            )

        else:
            raise ValueError(f"Unsupported component type: {component_type}")

    def _get_function_from_string(self, ref: str) -> Callable:
        ref = ref.strip()
        if not ref:
            raise ValueError(f"Empty function reference.")

        if ":" in ref and not ref.startswith("fn:"):
            file_part, func_name = ref.rsplit(":", 1)
            file_path = file_part.strip()
            func_name = func_name.strip()
            return self._load_function_from_file(file_path, func_name)

        if ref.startswith("fn:"):
            func_name = ref[3:].strip()
            for candidate_path in self.functions:
                try:
                    return self._load_function_from_file(candidate_path, func_name)
                except AttributeError:
                    pass

            raise ValueError(f"Function '{func_name}' not found in any file specified by 'self.functions'.")

        raise ValueError(f"Invalid function reference '{ref}'. Must be 'file.py:func' or 'fn:func'.")

    def _load_function_from_file(self, file_path: str, function_name: str) -> Callable:
        if file_path == "":
            raise FileNotFoundError(f"Empty file path found when trying to search for function: {function_name}")

        base_name, ext = os.path.splitext(file_path)
        if not ext:
            ext = ".py"
            file_path = base_name + ext

        resolved_path = None
        if not os.path.isabs(file_path):
            candidate_local = os.path.join(self.base_directory, file_path)
            candidate_local = os.path.normpath(candidate_local)
            if os.path.exists(candidate_local):
                resolved_path = candidate_local
            else:
                short_name = os.path.splitext(os.path.basename(file_path))[0]
                fetched = self._load_local_py_if_exists(short_name)
                if fetched:
                    resolved_path = fetched
        else:
            if os.path.exists(file_path):
                resolved_path = file_path

        if not resolved_path:
            raise FileNotFoundError(f"Cannot find Python file for '{file_path}' locally or on GitHub.")


        if resolved_path not in self._function_cache:
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"Cannot find function file: {resolved_path}")
            spec = importlib.util.spec_from_file_location(f"dynamic_{id(self)}", resolved_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._function_cache[resolved_path] = mod

        module = self._function_cache[resolved_path]
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in '{resolved_path}'.")
        return getattr(module, function_name)
    
    def _load_local_py_if_exists(self, filename_no_ext: str) -> str:
        py_file_name = f"{filename_no_ext}.py"

        try:
            file_text = resources.read_text("mas.lib", py_file_name)
        except FileNotFoundError:
            return None

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        tmp_file.write(file_text.encode("utf-8"))
        tmp_file.flush()
        tmp_file.close()
        return tmp_file.name
        
    def _resolve_automation_sequence(self, sequence):
        resolved_sequence = []

        for step in sequence:
            if isinstance(step, str) and "->" in step:
                left_part, right_part = step.rsplit("->", 1)
                left_part = left_part.strip()
                right_part = right_part.strip()

                if ":" in right_part:
                    component_name, target_params = right_part.split(":", 1)
                    component_name = component_name.strip()
                    target_params = target_params.strip()
                else:
                    component_name = right_part
                    target_params = ""

                import_ref = f"{left_part}->{component_name}"

                self._process_single_import(import_ref)

                resolved_step = f"{component_name}:{target_params}" if target_params else component_name
                resolved_sequence.append(resolved_step)

            elif isinstance(step, str):
                resolved_sequence.append(step)
            elif isinstance(step, dict):
                control_flow_type = step.get("control_flow_type")
                if control_flow_type == "branch":
                    if "condition" not in step or "if_true" not in step or "if_false" not in step:
                        raise ValueError("Branch must have 'condition', 'if_true', and 'if_false'.")
                    step["condition"] = self._resolve_condition(step["condition"])
                    step["if_true"] = self._resolve_automation_sequence(step.get("if_true", []))
                    step["if_false"] = self._resolve_automation_sequence(step.get("if_false", []))
                elif control_flow_type == "while":
                    if "body" not in step:
                        raise ValueError("While must have 'body'.")

                    if "condition" not in step and "end_condition" not in step:
                        raise ValueError("While must have either 'condition' or 'end_condition'.")

                    if "condition" in step:
                        step["condition"] = self._resolve_condition(step["condition"])
                    step["run_first_pass"] = step.get("run_first_pass", True)
                    step["start_condition"] = self._resolve_condition(step.get("start_condition", step.get("condition", step["run_first_pass"])))
                    step["end_condition"] = self._resolve_condition(step.get("end_condition", step.get("condition")))
                    step["body"] = self._resolve_automation_sequence(step.get("body", []))
                    
                elif control_flow_type == "for":
                    if "items" not in step or "body" not in step:
                        raise ValueError("For must have 'items' and 'body'.")
                    
                    step["items"] = step["items"]
                    step["body"] = self._resolve_automation_sequence(step.get("body", []))

                elif control_flow_type == "switch":
                    if "value" not in step or "cases" not in step:
                        raise ValueError("Switch must have 'value' and 'cases'")
                    
                    for case in step["cases"]:
                        if "case" not in case or "body" not in case:
                            raise ValueError("Each switch case must have 'case' and 'body'")
                    
                    step["value"] = step["value"]
                    step["cases"] = [{
                        "case": case["case"],
                        "body": self._resolve_automation_sequence(case.get("body", []))
                    } for case in step["cases"]]

                else:
                    raise ValueError(f"Unsupported control flow type: {control_flow_type}")
                resolved_sequence.append(step)
            else:
                raise ValueError(f"Unsupported step in sequence: {step}")

        return resolved_sequence

    def _resolve_condition(self, condition):
        if isinstance(condition, bool):
            return condition
        elif isinstance(condition, str):
            if ":" in condition and condition[0] != ":":
                return self._get_function_from_string(condition)
            return condition
        elif isinstance(condition, dict):
            return condition
        elif callable(condition):
            return condition
        else:
            raise ValueError(f"Unsupported condition type: {condition}")
    
    def clear_message_history(self, user_id: Optional[str] = None):
        if user_id is not None:
            self.set_current_user(user_id)

        if not self._uid():
            return

        conn = self._get_user_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM message_history")
        conn.commit()

    def clear_global_history(self) -> int:
        self._close_all_db_conns()
        os.makedirs(self.history_folder, exist_ok=True)

        cleared = 0
        for fname in os.listdir(self.history_folder):
            if not fname.endswith(".sqlite"):
                continue
            db_path = os.path.join(self.history_folder, fname)
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS message_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        msg_number INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        model TEXT
                    )
                """)
                cur.execute("DELETE FROM message_history")
                conn.commit()
                conn.close()
                cleared += 1
            except Exception:
                logger.exception(f"Error clearing history for DB: {db_path}")
        #logger.info(f"[Manager] Cleared history in {cleared} sqlite DB(s).")
        return cleared

    def reset_system(self) -> None:
        self._close_all_db_conns()
        try:
            shutil.rmtree(self.history_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove history folder.")
        try:
            shutil.rmtree(self.files_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove files folder.")
        try:
            if hasattr(self, "logs_folder") and self.logs_folder:
                shutil.rmtree(self.logs_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove logs folder.")

        os.makedirs(self.history_folder, exist_ok=True)
        os.makedirs(self.files_folder, exist_ok=True)
        if hasattr(self, "logs_folder") and self.logs_folder:
            os.makedirs(self.logs_folder, exist_ok=True)
            self._usage_log_path   = os.path.join(self.logs_folder, "usage.log")
            self._summary_log_path = os.path.join(self.logs_folder, "summary.log")
            if hasattr(self, "_lines_since_refresh"):
                self._lines_since_refresh = 0
        if hasattr(self, "_db_pool"):
            self._db_pool.clear()
        self._file_cache = {}
        logger.info("[Manager] System reset completed (history/, files/, logs/ wiped).")


    def escape_markdown(self, text: str) -> str:
        pattern = r'([_*\[\]\(\)~`>#\+\-=|{}\.!\\])'
        return re.sub(pattern, r'\\\1', text)
    
    def _is_block(self, obj):
        return isinstance(obj, dict) and "type" in obj and "content" in obj
    
    def _is_image_bytes(self, val):
        return isinstance(val, (bytes, bytearray)) and what(None, h=bytes(val)) in {
            "jpeg", "png", "gif", "bmp", "tiff", "webp", "heic", "avif"
        }

    def _is_valid_image_path(self, p):
        if not isinstance(p, str):
            return False
        mime, _ = mimetypes.guess_type(p)
        return mime and mime.startswith("image/") and os.path.isfile(p)

    def list_agents(self) -> list[str]:
        return sorted(self.agents.keys())

    def list_tools(self) -> list[str]:
        return sorted(self.tools.keys())

    def list_processes(self) -> list[str]:
        return sorted(self.processes.keys())

    def list_automations(self) -> list[str]:
        return sorted(self.automations.keys())

    def list_components(
        self, *, types: Optional[List[str]] = None,
        name_contains: Optional[str] = None,
        regex: Optional[str] = None
    ) -> list[str]:
        registries = []
        if not types:
            registries = [
                ("agent", self.agents),
                ("tool", self.tools),
                ("process", self.processes),
                ("automation", self.automations),
            ]
        else:
            m = {
                "agent": self.agents, "tool": self.tools,
                "process": self.processes, "automation": self.automations
            }
            registries = [(t, m[t]) for t in types if t in m]

        out = []
        for _t, reg in registries:
            for name in reg.keys():
                if name_contains and name_contains not in name:
                    continue
                if regex and not re.search(regex, name):
                    continue
                out.append(name)
        return sorted(out)

    def read(
        self,
        messages: Optional[Union[Dict, List[Dict]]] = None,
        user_id: Optional[str] = None,
        *,
        source: Optional[Union[str, List[str]]] = None,
        index: Optional[Union[int, tuple]] = -1,
        get_full_dict: bool = False,
        block_type: Optional[str] = None,
        block_index: Optional[Union[int, None]] = None
    ) -> Any:

        if source is not None and not isinstance(source, (str, list, tuple)):
            raise ValueError("source must be None, str or list/tuple of str.")
        if isinstance(source, (list, tuple)) and not all(isinstance(s, str) for s in source):
            raise ValueError("All elements of source must be str.")
        if block_type not in (None, "text", "image", "audio"):
            raise ValueError("block_type must be None, 'text', 'image' or 'audio'.")
        if block_index is not None and not isinstance(block_index, int):
            raise ValueError("block_index must be None or int.")
        if messages is not None and not isinstance(messages, (dict, list)):
            raise ValueError("messages must be None, dict or list[dict].")

        source_messages = []
        if messages is not None:
            source_messages = [messages] if isinstance(messages, dict) else list(messages)
        else:
            active_user = user_id or self._uid()
            if active_user:
                source_messages = self.get_messages(active_user)
        
        if not source_messages:
            return None

        source_messages = copy.deepcopy(source_messages)

        if source:
            sources_to_match = [source] if isinstance(source, str) else source
            source_messages = [msg for msg in source_messages if msg.get("source") in sources_to_match]

        selected_messages = []
        if index is None:
            selected_messages = source_messages
        elif isinstance(index, int):
            try:
                selected_messages = [source_messages[index]]
            except IndexError:
                return None
        elif isinstance(index, tuple) and len(index) == 2:
            start, end = index
            selected_messages = source_messages[start:end]
        
        if not selected_messages:
            return None

        if block_index is None and isinstance(index, int) and not get_full_dict:
            block_index = 0

        if block_type or block_index is not None:
            processed_messages = []
            for msg in selected_messages:
                original_blocks = msg.get("message", [])
                
                filtered_blocks = [b for b in original_blocks if b.get("type") == block_type] if block_type else original_blocks
                
                if block_index is not None and isinstance(block_index, int):
                    try:
                        final_blocks = [filtered_blocks[block_index]]
                    except IndexError:
                        final_blocks = []
                else:
                    final_blocks = filtered_blocks
                
                if final_blocks:
                    msg["message"] = final_blocks
                    processed_messages.append(msg)
            
            selected_messages = processed_messages

        if not selected_messages:
            return None

        if len(selected_messages) > 1:
            return selected_messages
        
        single_message = selected_messages[0]
        
        if get_full_dict:
            return single_message
        
        source = single_message.get("source")
        blocks = single_message.get("message", [])

        if not blocks:
            return source, [], []

        if block_index is not None and isinstance(block_index, int):
            target_block = blocks[0]
            block_content = target_block.get("content")
            
            if target_block.get("type") == "text" and isinstance(block_content, str):
                try:
                    block_content = json.loads(block_content)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            return source, target_block.get("type"), block_content
        else:
            return source, blocks

    def _to_blocks(self, value, user_id=None, detail="auto"):
        self.ensure_user(user_id)
        user_id = user_id or self._uid()

        if isinstance(value, list) and value and all(self._is_block(b) for b in value):
            return value
        
        if isinstance(value, list):
            merged = []
            for item in value:
                merged.extend(self._to_blocks(item, user_id=user_id, detail=detail))
            return merged

        if self._is_block(value):
            return [value]

        if self._is_image_bytes(value):
            file_ref = self.save_file(bytes(value), user_id)
            return [{
                "type": "image",
                "content": {"kind": "file", "path": file_ref, "detail": detail}
            }]
        if self._is_valid_image_path(value):
            with open(value, "rb") as f:
                raw = f.read()
            file_ref = self.save_file(raw, user_id)
            return [{
                "type": "image",
                "content": {"kind": "file", "path": file_ref, "detail": detail}
            }]

        if isinstance(value, dict):
            data_copy = self._persist_non_json_values(value, user_id)
            return [{
                "type": "text",
                "content": data_copy
            }]

        return [{
            "type": "text",
            "content": str(value)
        }]
    
    def _first_text_block(self, blocks):
        for b in blocks:
            if b.get("type") == "text":
                return b["content"]
        return None

    def _blocks_as_tool_input(self, blocks):

        if isinstance(blocks, str):
            try:
                maybe = json.loads(blocks)
                if isinstance(maybe, list) and all(isinstance(b, dict) for b in maybe):
                    blocks = maybe
                else:
                    blocks = [{"type": "text", "content": blocks}]
            except json.JSONDecodeError:
                blocks = [{"type": "text", "content": blocks}]

        if not isinstance(blocks, list):
            raise ValueError("Tool input must be a list of blocks or JSON-encoded list")
        
        raw = self._first_text_block(blocks)
        if raw is None:
            raise ValueError("Input must contain at least one text block")
        
        if isinstance(raw, dict):
            return raw
        
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            return {"text_content": raw}

        return {"text_content": str(raw)}
    
    def get_key(self, name: str) -> Optional[str]:
        name = name.lower()

        variations = [
            name,
            f"{name}_api_key",
            f"{name}-api-key",
            f"{name}_key",
            f"{name}-key",
            f"{name}key",
        ]
        
        for var in variations:
            var_lower = var.lower()
            for key_in_file in self.api_keys:
                if key_in_file.lower() == var_lower:
                    return self.api_keys[key_in_file]

        for var in variations:
            value = os.environ.get(var)

            if value:
                return value
            
            env_var_name = var.upper().replace('-', '_')
            value = os.environ.get(env_var_name)
            if value:
                return value
        
        return None
    
    def _json_unwrap(self, value, max_depth: int = 3):
        cur = value
        for _ in range(max_depth):
            if not isinstance(cur, str):
                break
            s = cur.strip()
            if not s:
                break
            try:
                cur = json.loads(s)
            except Exception:
                break
        return cur
    
    def _block_to_plain_text(self, blk) -> str:
        if blk.get("type") != "text":
            return ""
        raw = blk.get("content")

        if isinstance(raw, dict):
            if "response" in raw and isinstance(raw["response"], str):
                return raw["response"]
            return json.dumps(raw, ensure_ascii=False, indent=2)
        if isinstance(raw, list):
            return json.dumps(raw, ensure_ascii=False, indent=2)

        val = self._json_unwrap(raw)
        if isinstance(val, dict):
            if "response" in val and isinstance(val["response"], str):
                return val["response"]
            return json.dumps(val, ensure_ascii=False, indent=2)
        if isinstance(val, list):
            return json.dumps(val, ensure_ascii=False, indent=2)
        return str(val)

    def start_telegram_bot(
        self, 
        telegram_token: str = None, 
        start_polling: bool = True,
        **kwargs
    ):
        bot_instance = TelegramBot(
            manager=self,
            telegram_token=telegram_token,
            **kwargs
        )

        if start_polling:
            bot_instance.start_polling()
            return None
        else:
            return bot_instance

    def start_whatsapp_bot(
        self,
        *,
        run_server: bool = True,
        host: str = "0.0.0.0",
        port: int = 5000,
        base_path: str = "/webhook",
        **kwargs
    ):
        bot_instance = WhatsappBot(
            manager=self,
            **kwargs
        )
        if run_server:
            bot_instance.run_server(host=host, port=port, base_path=base_path)
            return None
        else:
            return bot_instance

    def stt(self, file_path, provider=None, model=None, user_id=None):
        if provider:
            provider = provider.lower()
        else:
            provider = ("groq" if self.get_key("groq")
                        else "openai" if self.get_key("openai")
                        else None)
        if not provider:
            raise ValueError("No supported STT provider API key found")

        if not model:
            model = {
                "groq":  "whisper-large-v3-turbo",
                "openai": "whisper-1"
            }[provider]

        if isinstance(file_path, str) and file_path.startswith("file:"):
            file_path = file_path[5:]

        def _audio_duration_sec(path: str):
            if path.startswith("file:"):
                path = path[5:]

            if not os.path.isfile(path):
                return None

            ext = os.path.splitext(path)[1].lower()
            if ext in (".wav", ".wave"):
                
                with wave.open(path, "rb") as w:
                    return w.getnframes() / w.getframerate()

            try:
                ffprobe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "format=duration", "-of", "json", path],
                    capture_output=True, text=True, timeout=5)
                if ffprobe.returncode == 0:
                    j = json.loads(ffprobe.stdout)
                    dur = float(j["format"]["duration"])
                    if dur > 0:
                        return dur
            except Exception:
                pass

            try:
                from mutagen import File as mutagen_file
                m = mutagen_file(path)
                if m is not None and m.info.length:
                    return float(m.info.length)
            except Exception:
                pass

            return None


        seconds = _audio_duration_sec(file_path) or 60.0

        with open(file_path, "rb") as audio_file:
            if provider == "groq":
                transcript = self._transcribe_groq(audio_file, model)
            elif provider == "openai":
                transcript = self._transcribe_openai(audio_file, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        self._log_stt_call(provider, model, seconds, transcript)

        try:
            rate_pm = self._price_for_stt(provider, model)
            cost    = round(rate_pm * (seconds / 60.0), 6)
        except Exception:
            cost = 0.0

        try:
            self._log_if_enabled({
                "user":  user_id or self._uid(),
                "type":  "stt",
                "id":    f"{provider}:{model}",
                "seconds": seconds,
                "in_toks": 0,
                "out_toks": 0,
                "cost":  cost
            })
        except Exception as e:
            logger.warning(f"Could not log STT usage: {e}")

        return transcript

    def _transcribe_groq(self, audio_file, model):
        api_key = self.get_key("groq")
        if not api_key:
            raise ValueError("Groq API key not found")

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        files = {
            "file": audio_file,
            "model": (None, model)#,
            #"temperature": (None, "0"),
            #"response_format": (None, "json"),
            #"language": (None, "en")
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files, timeout=self.timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API Error: {response.text}")
            response.raise_for_status()
            
        return response.json().get("text", "")

    def _transcribe_openai(self, audio_file, model):
        api_key = self.get_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found")

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        files = {
            "file": audio_file,
            "model": (None, model)
        }

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files, timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"OpenAI API error: {response.text}")
        
        return response.json().get("text", "")

    def tts(
        self,
        text: str,
        voice: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:

        provider = (provider or "openai").lower()
        if provider == "openai":
            file_path = self._generate_tts_openai(text, voice, model, instruction)
        elif provider == "elevenlabs":
            file_path = self._generate_tts_elevenlabs(text, voice, model)
        else:
            raise ValueError(f"Unsupported TTS provider: {provider}")
        return file_path

    def _generate_tts_openai(
        self,
        text: str,
        voice: str,
        model: Optional[str],
        instruction: Optional[str]
    ) -> str:
        api_key = self.get_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not found.")
        if not model:
            model = "gpt-4o-mini-tts"
        
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "input": text,
            "voice": voice.lower()
        }
        if instruction:
            data["instructions"] = instruction

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        audio_bytes = response.content
        return self._save_tts_file(audio_bytes)

    def _generate_tts_elevenlabs(
        self,
        text: str,
        voice: str,
        model: Optional[str]
    ) -> str:
        api_key = self.get_key("elevenlabs")
        if not api_key:
            raise ValueError("ElevenLabs API key not found.")
        if not model:
            model = "eleven_multilingual_v2"
        
        voices_url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": api_key}
        voices_response = requests.get(voices_url, headers=headers, timeout=self.timeout)
        voices_response.raise_for_status()
        voices_data = voices_response.json()
        voice_id = None
        for v in voices_data.get("voices", []):
            if v.get("name", "").lower() == voice.lower():
                voice_id = v.get("voice_id")
                break
        if not voice_id:
            raise ValueError(f"Voice '{voice}' not found in ElevenLabs voices.")
        
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
        payload = {
            "text": text,
            "model_id": model
        }
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(tts_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        audio_bytes = response.content
        return self._save_tts_file(audio_bytes)

    def _save_tts_file(self, audio_bytes: bytes) -> str:
        tts_folder = os.path.join(self.files_folder, "tts")
        os.makedirs(tts_folder, exist_ok=True)
        uid = str(self._active_user_id() or self._uid() or "unknown")
        filename = f"{uid}_{uuid.uuid4().hex}.mp3"
        file_path = os.path.join(tts_folder, filename)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        return file_path
    
    def _close_all_db_conns(self):
        if hasattr(self, "_db_pool"):
            for _uid, conn in list(self._db_pool.items()):
                try:
                    conn.close()
                except Exception:
                    pass
            self._db_pool.clear()


