from __future__ import annotations

from ._shared import *

class Component:
    def __init__(self, name: str):
        self.name = name 
        self.manager = None

    def to_string(self) -> str:
        return f"Component(Name: {self.name})"


    def run(self, input_data: Any = None,
        verbose: bool = False) -> Dict:
        raise NotImplementedError("Subclass must implement .run().")

    def _has_timeline_selection(self, parsed: dict) -> bool:
        selection = parsed.get("selection") if isinstance(parsed, dict) else None
        return isinstance(selection, dict) and selection.get("mode") == "timeline"

    def _message_sort_key(self, row: tuple) -> Any:
        return row[2]

    def _select_by_index(self, index, subset: List[tuple], default: str = "all") -> List[tuple]:
        if not subset:
            return []

        if index is None:
            return subset[-1:] if default == "latest" else subset[:]

        if index == "~":
            return subset[:]

        if isinstance(index, int):
            if -len(subset) <= index < len(subset):
                return [subset[index]]
            return []

        if isinstance(index, tuple):
            start, end = index
            start = 0 if start is None else int(start)
            end = len(subset) if end is None else int(end)
            return subset[start:end]

        return subset[-1:] if default == "latest" else subset[:]

    def _normalize_global_position(self, pos: int, length: int) -> int:
        return length + pos if pos < 0 else pos

    def _resolve_global_endpoint(self, endpoint, messages: List[tuple]) -> Optional[int]:
        if endpoint is None:
            return None

        if isinstance(endpoint, int):
            return self._normalize_global_position(endpoint, len(messages))

        if isinstance(endpoint, dict) and endpoint.get("type") == "anchor":
            component = endpoint.get("component")
            index = endpoint.get("index", -1)
            subset = [row for row in messages if row[0] == component]
            chosen = self._select_by_index(index, subset, default="latest")
            if len(chosen) != 1:
                return None
            target_msg_number = chosen[0][2]
            for pos, row in enumerate(messages):
                if row[2] == target_msg_number:
                    return pos
            return None

        return None

    def _apply_global_index(self, messages: List[tuple], global_index) -> List[tuple]:
        if global_index is None or global_index == "~":
            return messages[:]

        if isinstance(global_index, int):
            if -len(messages) <= global_index < len(messages):
                return [messages[global_index]]
            return []

        if isinstance(global_index, dict):
            pos = self._resolve_global_endpoint(global_index, messages)
            if pos is None or not (0 <= pos < len(messages)):
                return []
            return [messages[pos]]

        if isinstance(global_index, tuple):
            start_endpoint, end_endpoint = global_index
            start = self._resolve_global_endpoint(start_endpoint, messages) if start_endpoint is not None else 0
            end = self._resolve_global_endpoint(end_endpoint, messages) if end_endpoint is not None else len(messages)
            if start is None or end is None:
                return []
            start = max(0, min(start, len(messages)))
            end = max(0, min(end, len(messages)))
            if end < start:
                return []
            return messages[start:end]

        return messages[:]

    def _content_as_blocks_for_fields(self, content: Any) -> List[dict]:
        data = content
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(data, list) and all(isinstance(b, dict) and "type" in b for b in data):
            return data
        return self.manager._to_blocks(data)

    def _with_filtered_fields(self, msg_tuples: List[tuple], fields: Optional[list]) -> List[tuple]:
        if not fields:
            return msg_tuples
        out = []
        for role, content, msg_number, msg_type, timestamp in msg_tuples:
            blocks = self._content_as_blocks_for_fields(content)
            filtered = self.manager._filter_blocks_by_fields(blocks, fields)
            out.append((role, filtered, msg_number, msg_type, timestamp))
        return out

    def _apply_timeline_selection(
        self,
        selection: dict,
        messages: List[tuple],
        *,
        per_source_default: str = "all"
    ) -> List[tuple]:
        window = self._apply_global_index(messages, selection.get("global_index"))
        selector = selection.get("selector", {}) or {}
        selector_type = selector.get("type")

        if selector_type == "all":
            return window[:]

        if selector_type == "exclude":
            excluded = set(selector.get("components", []) or [])
            return [row for row in window if row[0] not in excluded]

        if selector_type == "include":
            selected = []
            for source_item in selector.get("sources", []) or []:
                comp_name = source_item.get("component")
                fields = source_item.get("fields")
                subset = [row for row in window if row[0] == comp_name] if comp_name else window[:]
                chosen = self._select_by_index(
                    source_item.get("index"),
                    subset,
                    default=per_source_default
                )
                selected.extend(self._with_filtered_fields(chosen, fields))
            return sorted(selected, key=self._message_sort_key)

        return []

    def _messages_to_input_dict(self, msg_tuples: List[tuple]) -> dict:
        final_input = {}
        for role, content, msg_number, msg_type, timestamp in sorted(msg_tuples, key=self._message_sort_key):
            data = self.manager._blocks_as_tool_input(content)
            data = self.manager._load_files_in_dict(data)
            if isinstance(data, dict):
                final_input.update(data)
        return final_input


class Agent(Component):

    def __init__(
        self,
        name: str,
        system_prompt: str,
        system_prompt_original: str,
        required_outputs: Union[Dict[str, Any], str],
        models: List[Dict[str, str]],
        default_output: Optional[Dict[str, Any]] = None,
        positive_filter: Optional[List[str]] = None,
        negative_filter: Optional[List[str]] = None,
        general_system_description: str = "",
        model_params: Optional[Dict[str, Any]] = None,
        include_timestamp: bool = False,
        description: str = None,
        timeout: int = 120
    ):
        super().__init__(name)
        self.system_prompt = system_prompt
        self.system_prompt_original = system_prompt_original
        if isinstance(required_outputs, str):
            self.required_outputs = {"response": required_outputs}
        else:
            self.required_outputs = copy.deepcopy(required_outputs)

        self.models = copy.deepcopy(models)
        self.default_output = copy.deepcopy(default_output) if default_output is not None else {"response": "No valid response."}
        self.positive_filter = copy.deepcopy(positive_filter)
        self.negative_filter = copy.deepcopy(negative_filter)
        self.general_system_description = general_system_description
        self.model_params = copy.deepcopy(model_params) if model_params is not None else {}
        self.include_timestamp = include_timestamp
        self.description = description if description else "Agent"
        self.timeout = timeout

    def to_string(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"System Prompt: {self.system_prompt}\n"
            f"Models: {self.models}\n"
            f"Default Output: {self.default_output}\n"
            f"Positive Filter: {self.positive_filter}\n"
            f"Negative Filter: {self.negative_filter}\n"
            f"Description: {self.description}"
        )

    def _runtime_config(self) -> Dict[str, Any]:
        getter = getattr(self.manager, "_get_runtime_component_config", None)
        if callable(getter):
            cfg = getter(self.name)
            if isinstance(cfg, dict):
                return cfg
        return {}

    def _runtime_attr(self, name: str, default: Any) -> Any:
        cfg = self._runtime_config()
        if name in cfg:
            return cfg[name]
        resolver = getattr(self.manager, "resolve_runtime_value", None)
        if callable(resolver):
            return resolver(default, component_name=self.name)
        return default

    def _runtime_system_prompt(self) -> str:
        return self._runtime_attr("system_prompt", self.system_prompt)

    def _runtime_models(self) -> List[Dict[str, Any]]:
        models = self._runtime_attr("models", self.models)
        if isinstance(models, dict):
            models = [models]
        return copy.deepcopy(models or [])

    def _runtime_model_params(self) -> Dict[str, Any]:
        params = self._runtime_attr("model_params", self.model_params)
        return copy.deepcopy(params) if isinstance(params, dict) else {}

    def _runtime_required_outputs(self) -> Dict[str, Any]:
        outputs = self._runtime_attr("required_outputs", self.required_outputs)
        if isinstance(outputs, str):
            return {"response": outputs}
        return copy.deepcopy(outputs) if isinstance(outputs, dict) else {}

    def _runtime_default_output(self) -> Dict[str, Any]:
        return copy.deepcopy(self._runtime_attr("default_output", self.default_output))

    def _runtime_timeout(self) -> int:
        return self._runtime_attr("timeout", self.timeout)

    def _runtime_positive_filter(self):
        return self._runtime_attr("positive_filter", self.positive_filter)

    def _runtime_negative_filter(self):
        return self._runtime_attr("negative_filter", self.negative_filter)

    def _runtime_include_timestamp(self) -> bool:
        return bool(self._runtime_attr("include_timestamp", self.include_timestamp))

    @staticmethod
    def _runtime_model_params_for_call(instance) -> Dict[str, Any]:
        getter = getattr(instance, "_runtime_model_params", None)
        if callable(getter):
            return getter()
        params = getattr(instance, "model_params", {}) or {}
        return copy.deepcopy(params) if isinstance(params, dict) else {}

    @staticmethod
    def _runtime_required_outputs_for_call(instance) -> Dict[str, Any]:
        getter = getattr(instance, "_runtime_required_outputs", None)
        if callable(getter):
            return getter()
        outputs = getattr(instance, "required_outputs", {}) or {}
        if isinstance(outputs, str):
            return {"response": outputs}
        return copy.deepcopy(outputs) if isinstance(outputs, dict) else {}

    @staticmethod
    def _runtime_timeout_for_call(instance) -> int:
        getter = getattr(instance, "_runtime_timeout", None)
        if callable(getter):
            return getter()
        return getattr(instance, "timeout", 120)

    def _push_runtime_config(self) -> bool:
        resolver = getattr(self.manager, "resolve_component_runtime_config", None)
        pusher = getattr(self.manager, "_push_runtime_component_config", None)
        if not callable(resolver) or not callable(pusher):
            return False
        pusher(self.name, resolver(self))
        return True

    def _pop_runtime_config(self) -> None:
        popper = getattr(self.manager, "_pop_runtime_component_config", None)
        if callable(popper):
            popper(self.name)

    def run(
        self,
        input_data: Any = None,
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> dict:
        runtime_pushed = self._push_runtime_config()
        try:
            return self._run_with_runtime_config(
                input_data=input_data,
                target_input=target_input,
                target_index=target_index,
                target_fields=target_fields,
                target_custom=target_custom,
                verbose=verbose,
                return_token_count=return_token_count
            )
        finally:
            if runtime_pushed:
                self._pop_runtime_config()

    def _run_with_runtime_config(
        self,
        input_data: Any = None,
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> dict:
        db_conn = self.manager._get_user_db()

        conversation = None
        if target_input and any(x in target_input for x in [":", "fn?", "fn:", "?"]):
            parsed = self.manager.parser.parse_input_string(target_input)
            conversation = self._build_conversation_from_parser_result(parsed, db_conn, verbose)

        elif target_custom:
            all_msgs = self.manager._get_all_messages(db_conn)
            filtered_msgs = self._apply_filters(all_msgs)
            conversation = self._build_conversation_from_custom(filtered_msgs, target_custom, verbose)

        elif target_input or target_index or target_fields:
            all_msgs = self.manager._get_all_messages(db_conn)
            filtered_msgs = self._apply_filters(all_msgs)
            conversation = self._build_conversation_from_target(filtered_msgs, target_input, target_index, target_fields, verbose)
        else:
            all_msgs = self.manager._get_all_messages(db_conn)
            filtered_msgs = self._apply_filters(all_msgs)
            conversation = self._build_default_conversation(filtered_msgs, verbose)

        if verbose:
            logger.debug(f"[Agent:{self.name}] Final conversation for model input:\n\n{json.dumps(conversation, indent=2, ensure_ascii=False)}\n")

        provider_attempts = []
        runtime_models = self._runtime_models()
        if hasattr(self.manager, "prepare_model_attempts"):
            model_attempts = self.manager.prepare_model_attempts(runtime_models)
        else:
            model_attempts = [
                {
                    "model_info": model_info,
                    "skip": False,
                    "skip_reason": None,
                    "status": {},
                }
                for model_info in runtime_models
            ]

        for model_attempt in model_attempts:
            model_info = model_attempt["model_info"]
            provider = model_info["provider"].lower()
            model_name = model_info["model"]

            if model_attempt.get("skip"):
                provider_attempts.append(
                    self._provider_skip_metadata(
                        provider,
                        model_name,
                        model_attempt.get("skip_reason") or "recent_failure_cooldown",
                        model_attempt.get("status") or {}
                    )
                )
                if verbose:
                    logger.warning(
                        f"[Agent:{self.name}] Model {provider}/{model_name} is temporarily suppressed "
                        "after recent failures. Trying the next available model."
                    )
                continue

            api_key = self.manager.get_key(provider)

            if not api_key:
                failure_metadata = self._provider_error_metadata(
                    provider,
                    model_name,
                    error_type="missing_api_key",
                    message=f"No API key for provider '{provider}'."
                )
                provider_attempts.append(failure_metadata)
                self._record_manager_model_failure(provider, model_name, model_info, failure_metadata)
                if verbose:
                    logger.warning(f"[Agent:{self.name}] No API key for '{provider}'. Skipping.")
                continue
            
            formatted_conversation = self._provider_format_messages(provider, conversation)

            try:
                response = None
                self._last_provider_response_metadata = None
                
                if provider == "openai":
                    response = self._call_openai_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "openrouter":
                    response = self._call_openrouter_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "lmstudio":
                    base_url = model_info.get("base_url")
                    response = self._call_lmstudio_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count,
                        base_url=base_url
                    )
                elif provider == "deepseek":
                    response = self._call_deepseek_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "google":
                    response = self._call_google_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "groq":
                    response = self._call_groq_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "anthropic":
                    response = self._call_anthropic_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "wavespeed":
                    response = self._call_wavespeed_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count
                    )
                elif provider == "nvidia":
                    response = self._call_nvidia_api(
                        model_name=model_name,
                        conversation=formatted_conversation,
                        api_key=api_key,
                        verbose=verbose,
                        return_token_count=return_token_count,
                        base_url=model_info.get("base_url")
                    )
                else:
                    raise ValueError(f"[Agent:{self.name}] Unknown provider '{provider}'")

                if return_token_count:
                    response_str, input_tokens, output_tokens = response
                else:
                    response_str = response

                provider_metadata = getattr(self, "_last_provider_response_metadata", None)
                if provider_metadata is None:
                    provider_metadata = self._provider_success_metadata(
                        provider,
                        model_name,
                        content=response_str
                    )
                provider_metadata["ok"] = True
                provider_attempts.append(provider_metadata)
                self._record_manager_model_success(provider, model_name, model_info, provider_metadata)

                response_dict = self._extract_and_parse_json(response_str)

                if response_dict is None:
                    response_dict = {"response": response_str}

                response_blocks = self.manager._to_blocks(
                    response_dict, self.manager._active_user_id()
                )

                self._last_used_model = {"provider": provider, "model": model_name}
                if verbose:
                    logger.debug(f"[Agent:{self.name}] => success from provider={provider}\n{response_dict}")
                
                if return_token_count:
                    response_blocks[0].setdefault("metadata", {})
                    md = response_blocks[0]["metadata"]
                    md["input_tokens"]  = input_tokens
                    md["output_tokens"] = output_tokens
                    md["usd_cost"]      = self.manager.cost_model_call(
                        provider, model_name, input_tokens, output_tokens
                    )
                self._attach_provider_metadata(response_blocks, provider_metadata, provider_attempts)
                return response_blocks
                
            except requests.exceptions.RequestException as req_err:
                failure_metadata = self._provider_exception_metadata(provider, model_name, req_err)
                provider_attempts.append(failure_metadata)
                self._record_manager_model_failure(provider, model_name, model_info, failure_metadata)
                if verbose:
                    logger.error(f"[Agent:{self.name}] HTTP error with provider={provider}, model={model_name}: {req_err}")
            except json.JSONDecodeError as json_err:
                failure_metadata = self._provider_exception_metadata(provider, model_name, json_err)
                provider_attempts.append(failure_metadata)
                self._record_manager_model_failure(provider, model_name, model_info, failure_metadata)
                if verbose:
                    logger.error(f"[Agent:{self.name}] JSON decoding error with provider={provider}, model={model_name}: {json_err}")
            except ValueError as val_err:
                failure_metadata = self._provider_exception_metadata(provider, model_name, val_err)
                provider_attempts.append(failure_metadata)
                self._record_manager_model_failure(provider, model_name, model_info, failure_metadata)
                if verbose:
                    logger.error(f"[Agent:{self.name}] Value error with provider={provider}, model={model_name}: {val_err}")
            except (KeyError, IndexError, TypeError) as parse_err:
                failure_metadata = self._provider_exception_metadata(provider, model_name, parse_err)
                provider_attempts.append(failure_metadata)
                self._record_manager_model_failure(provider, model_name, model_info, failure_metadata)
                if verbose:
                    logger.error(
                        f"[Agent:{self.name}] Invalid response shape with provider={provider}, "
                        f"model={model_name}: {parse_err}"
                    )

        if verbose:
            logger.warning(f"[Agent:{self.name}] => All providers failed. Returning default:\n{self._runtime_default_output()}")
        default_blocks = self.manager._to_blocks(
            self._runtime_default_output(),
            self.manager._active_user_id()
        )
        self._attach_provider_metadata(default_blocks, None, provider_attempts)
        return default_blocks

    def _as_blocks(self, value):
        is_block = getattr(self.manager, "_is_block", None)
        if isinstance(value, list) and value and all(
            is_block(b) if callable(is_block) else isinstance(b, dict) and "type" in b and "content" in b
            for b in value
        ):
            return value

        return self.manager._to_blocks(value)

    def _extract_request_id(self, response: Any) -> Optional[str]:
        headers = getattr(response, "headers", None) or {}
        for key in (
            "x-request-id",
            "x-request-id".title(),
            "x-goog-request-id",
            "request-id",
            "cf-ray",
        ):
            try:
                value = headers.get(key)
            except AttributeError:
                value = None
            if value:
                return value
        return None

    def _response_body_for_metadata(self, response: Any) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except Exception:
            return getattr(response, "text", None)

    def _normalize_usage_metadata(self, usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        usage = usage or {}
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidatesTokenCount")
            or 0
        )
        total_tokens = (
            usage.get("total_tokens")
            or usage.get("totalTokenCount")
            or ((input_tokens or 0) + (output_tokens or 0))
        )
        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens or 0,
            "raw": usage,
        }

    def _provider_success_metadata(
        self,
        provider: str,
        model_name: str,
        *,
        response: Any = None,
        raw_response: Any = None,
        content: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if raw_response is None:
            raw_response = Agent._response_body_for_metadata(self, response)
        return {
            "provider": provider,
            "model": model_name,
            "ok": True,
            "status_code": getattr(response, "status_code", None),
            "request_id": Agent._extract_request_id(self, response),
            "content": content,
            "usage": Agent._normalize_usage_metadata(self, usage),
            "error": None,
            "errors": [],
            "raw_response": raw_response,
        }

    def _provider_error_metadata(
        self,
        provider: str,
        model_name: str,
        *,
        error_type: str,
        message: str,
        response: Any = None,
        raw_response: Any = None
    ) -> Dict[str, Any]:
        if raw_response is None:
            raw_response = Agent._response_body_for_metadata(self, response)
        error = {
            "type": error_type,
            "message": message,
        }
        return {
            "provider": provider,
            "model": model_name,
            "ok": False,
            "status_code": getattr(response, "status_code", None),
            "request_id": Agent._extract_request_id(self, response),
            "content": None,
            "usage": Agent._normalize_usage_metadata(self, None),
            "error": error,
            "errors": [error],
            "raw_response": raw_response,
        }

    def _provider_skip_metadata(
        self,
        provider: str,
        model_name: str,
        reason: str,
        health_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        remaining = health_status.get("cooldown_seconds_remaining")
        if remaining is None:
            message = f"Model {provider}/{model_name} is temporarily suppressed after recent failures."
        else:
            message = (
                f"Model {provider}/{model_name} is temporarily suppressed after recent failures "
                f"for another {round(float(remaining), 3)} seconds."
            )
        metadata = Agent._provider_error_metadata(
            self,
            provider,
            model_name,
            error_type="model_temporarily_suppressed",
            message=message,
        )
        metadata["skipped"] = True
        metadata["skip_reason"] = reason
        metadata["model_health"] = copy.deepcopy(health_status)
        return metadata

    def _provider_exception_metadata(
        self,
        provider: str,
        model_name: str,
        exc: Exception
    ) -> Dict[str, Any]:
        response = getattr(exc, "response", None)
        base = getattr(self, "_last_provider_response_metadata", None)
        if isinstance(base, dict) and base.get("provider") == provider and base.get("model") == model_name:
            metadata = copy.deepcopy(base)
            metadata["ok"] = False
            metadata["status_code"] = metadata.get("status_code") or getattr(response, "status_code", None)
            metadata["request_id"] = metadata.get("request_id") or Agent._extract_request_id(self, response)
            if metadata.get("raw_response") is None:
                metadata["raw_response"] = Agent._response_body_for_metadata(self, response)
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            metadata["error"] = error
            metadata["errors"] = [error]
            return metadata

        return Agent._provider_error_metadata(
            self,
            provider,
            model_name,
            error_type=exc.__class__.__name__,
            message=str(exc),
            response=response,
        )

    def _remember_provider_response(self, metadata: Dict[str, Any]) -> None:
        self._last_provider_response_metadata = metadata

    def _record_manager_model_failure(
        self,
        provider: str,
        model_name: str,
        model_info: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        recorder = getattr(self.manager, "record_model_failure", None)
        if callable(recorder):
            recorder(provider, model_name, model_info, metadata)

    def _record_manager_model_success(
        self,
        provider: str,
        model_name: str,
        model_info: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        recorder = getattr(self.manager, "record_model_success", None)
        if callable(recorder):
            recorder(provider, model_name, model_info, metadata)

    def _attach_provider_metadata(
        self,
        response_blocks: Any,
        provider_metadata: Optional[Dict[str, Any]],
        provider_attempts: List[Dict[str, Any]]
    ) -> None:
        if not isinstance(response_blocks, list) or not response_blocks:
            return
        first = response_blocks[0]
        if not isinstance(first, dict):
            return
        first.setdefault("metadata", {})
        md = first["metadata"]
        if provider_metadata is not None:
            md["provider_response"] = copy.deepcopy(provider_metadata)
        md["provider_attempts"] = copy.deepcopy(provider_attempts)
        md["provider_errors"] = [
            error
            for attempt in provider_attempts
            for error in attempt.get("errors", [])
        ]

    def _provider_format_messages(self, provider: str, conversation: list) -> list:
        provider = provider.lower()
        out = []

        def _clean_path(path_str):
            if isinstance(path_str, str) and path_str.startswith("file:"):
                return path_str[5:]
            return path_str
        
        def _image_to_datauri(path):
            path = _clean_path(path)
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
        
        def _text_as_string(content):
            if isinstance(content, (dict, list)):
                return json.dumps(content, ensure_ascii=False)
            return str(content)

        for msg in conversation:
            role   = msg["role"]
            blocks = self._as_blocks(msg["content"])

            if provider == "nvidia":
                text = "\n".join(
                    _text_as_string(blk["content"]) if blk["type"] == "text"
                    else "[image omitted]"
                    for blk in blocks
                )
                out.append({"role": role, "content": text})

            elif provider in {"openai", "openrouter", "groq", "deepseek", "lmstudio", "wavespeed"}:
                if role == "user":
                    parts = []
                    for blk in blocks:
                        if blk["type"] == "text":
                            parts.append({"type": "text", "text": _text_as_string(blk["content"])})
                        elif blk["type"] == "image":
                            try:
                                src = blk["content"]
                                if src["kind"] == "file":
                                    url = _image_to_datauri(src["path"])
                                elif src["kind"] == "url":
                                    url = src["url"]
                                else:
                                    url = f"data:image/*;base64,{src['b64']}"
                                parts.append({"type": "image_url",
                                            "image_url": {"url": url,
                                                            "detail": src.get("detail","auto")}})
                            except:
                                logger.warning(f"File not found in history, skipping image: {src.get('path')}")
                                parts.append({"type": "text", "text": "[image omitted]"})
                                continue
                    out.append({"role": role, "content": parts})
                else:
                    text = "\n".join(
                        _text_as_string(blk["content"]) if blk["type"] == "text"
                        else "[image omitted]"
                        for blk in blocks
                    )
                    out.append({"role": role, "content": text})

            elif provider == "google":
                gem_parts = []
                for blk in blocks:
                    if blk["type"] == "text":
                        gem_parts.append({"text": _text_as_string(blk["content"])})
                    elif blk["type"] == "image":
                        try:
                            src = blk["content"]
                            if src["kind"] == "file":
                                data = open(_clean_path(src["path"]),"rb").read()
                            elif src["kind"] == "url":
                                data = requests.get(src["url"], timeout=Agent._runtime_timeout_for_call(self)).content
                            else:
                                data = base64.b64decode(src["b64"])
                            gem_parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(data).decode()
                                }
                            })
                        except:
                            logger.warning(f"File not found in history, skipping image: {src.get('path')}")
                            gem_parts.append({"text": "[image omitted]"})
                            continue
                if role == "system":
                    out.append({"role": role, "parts": gem_parts})
                else:
                    out.append({"role": "user" if role=="user" else "model",
                                "parts": gem_parts})

            elif provider == "anthropic":
                c_blocks = []
                for blk in blocks:
                    if blk["type"] == "text":
                        c_blocks.append({"type":"text", "text": _text_as_string(blk["content"])})
                    elif blk["type"] == "image":
                        try:
                            src = blk["content"]
                            if src["kind"] == "file":
                                data = open(_clean_path(src["path"]),"rb").read()
                                mime = mimetypes.guess_type(src["path"])[0] or "image/jpeg"
                                b64  = base64.b64encode(data).decode()
                                c_blocks.append({"type":"image",
                                                "source":{"type":"base64",
                                                        "media_type":mime,
                                                        "data": b64}})
                            elif src["kind"] == "url":
                                c_blocks.append({"type":"image",
                                                "source":{"type":"url",
                                                        "media_type":"image/jpeg",
                                                        "url": src["url"]}})
                        except:
                            logger.warning(f"File not found in history, skipping image: {src.get('path')}")
                            c_blocks.append({"type":"text", "text": "[image omitted]"})
                            continue
                out.append({"role": role, "content": c_blocks})

            else:
                out.append({"role": role, "content": blocks})

        return out


    def _build_conversation_from_parser_result(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> List[Dict[str, Any]]:
        if verbose:
            logger.debug(f"[Agent:{self.name}] _build_conversation_from_parser_result => parsed={parsed}")

        all_msgs = self.manager._get_all_messages(db_conn)
        filtered_msgs = self._apply_filters(all_msgs)

        if self._has_timeline_selection(parsed):
            selected = self._apply_timeline_selection(
                parsed["selection"],
                filtered_msgs,
                per_source_default="all"
            )
            conversation = self._transform_to_conversation(selected)
        elif parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp_name = parsed["component_or_param"]
                chosen = [
                    (r, c, n, t, ts) 
                    for (r, c, n, t, ts) in filtered_msgs 
                    if r == comp_name
                ]
                conversation = self._transform_to_conversation(chosen)
            else:
                conversation = self._transform_to_conversation(filtered_msgs)
        elif parsed["multiple_sources"]:
            combined = []
            for source_item in parsed["multiple_sources"]:
                partial = self._collect_msg_snippets_for_agent(source_item, filtered_msgs)
                combined.extend(partial)
            combined.sort(key=lambda x: x[2])
            conversation = self._transform_to_conversation(combined)
        else:
            partial = self._collect_msg_snippets_for_agent(parsed["single_source"], filtered_msgs)
            conversation = self._transform_to_conversation(partial)

        conversation = [{"role": "system", "content": self._runtime_system_prompt()}] + conversation
        return conversation


    def _collect_msg_snippets_for_agent(self, source_item: dict, filtered_msgs: List[tuple]) -> List[tuple]:
        comp_name = source_item["component"]
        index = source_item["index"]

        if comp_name:

            subset = [
                (r, c, n, t, ts) 
                for (r, c, n, t, ts) in filtered_msgs 
                if r == comp_name
            ]
        else:
            subset = None

        if not subset:
            return []

        chosen = self._handle_index(index, subset)

        return chosen


    def _handle_index(self, index, subset):
        if index is None or index == "~":
            chosen = subset[:]
        elif isinstance(index, int):
            if -len(subset) <= index < len(subset):
                chosen = [subset[index]]
            else:
                return []
        elif isinstance(index, tuple):
            start, end = index

            if start is None:
                start = 0
            else:
                start = int(start)

            if end is None:
                end = len(subset)
            else:
                end = int(end)

            chosen = subset[start:end]
        else:
            chosen = subset[:]

        return chosen
    
    def _strip_metadata(self, blocks: list) -> list:
        cleaned = []
        for blk in blocks:
            if isinstance(blk, dict):
                if blk.get("type") == "variable":
                    continue
                blk = blk.copy()
                blk.pop("metadata", None)
            cleaned.append(blk)
        return cleaned

    def _transform_to_conversation(self, msg_tuples: List[tuple], fields: Optional[list] = None, include_message_number: Optional[bool] = False) -> List[Dict[str, str]]:
        conversation = []
        for (role, content, msg_number, msg_type, timestamp) in msg_tuples:
            data = json.loads(content) if isinstance(content, str) else content

            if isinstance(data, list) and all(isinstance(b, dict) and "type" in b for b in data):
                blocks = [b.copy() for b in data]
            else:
                blocks = self._as_blocks(data)

            blocks = self._strip_metadata(blocks)

            if fields:
                blocks = self.manager._filter_blocks_by_fields(
                    blocks, fields
                )

            final_blocks = []

            if role != self.name:
                source_id = f"{role}"
                source_obj = {"source": source_id}

                if self._runtime_include_timestamp():
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        formatted_ts = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        formatted_ts = timestamp

                    source_obj["timestamp"] = formatted_ts

                source_block = {
                    "type": "text",
                    "content": source_obj
                }
                final_blocks.append(source_block)

                final_blocks.extend(blocks)
            else:
                final_blocks = blocks

            conversation.append({
                "role": "assistant" if role == self.name else "user",
                "content": final_blocks,
                "msg_number": msg_number
            })

        conversation.sort(key=lambda e: e["msg_number"])

        if not include_message_number:
            conversation = [{
                "role": e["role"],
                "content": e["content"]
            } for e in conversation]
        
        return conversation



    def _build_conversation_from_custom(
            self,
            filtered_msgs: List[tuple],
            target_custom: list,
            verbose: bool
        ) -> List[Dict[str, Any]]:
            
            conversation = []
            for item in target_custom:
                comp_name = item.get("component")
                index = item.get("index", None)
                fields = item.get("fields", None)

                subset = [
                    (r, c, n, t, ts) 
                    for (r, c, n, t, ts) in filtered_msgs 
                    if r == comp_name
                ]

                if not subset:
                    if verbose:
                        logger.debug(f"[Agent:{self.name}] No messages found for component/user '{comp_name}'. Skipping.")
                    continue

                subset = self._handle_index(index, subset)
                subset = self._handle_fields(subset, fields)
                
                conversation.extend(subset)

            conversation = self._transform_to_conversation(conversation)
            
            conversation = [{"role": "system", "content": self._runtime_system_prompt()}] + conversation
            return conversation
    
    def _handle_fields(self, msg_tuples, fields):
        processed_tuples = []
        for (role, content, msg_number, msg_type, timestamp) in msg_tuples:
            try:
                blocks = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                blocks = [{"type": "text", "content": str(content)}]

            if fields:
                blocks = self.manager._filter_blocks_by_fields(blocks, fields)
            
            processed_tuples.append((role, blocks, msg_number, msg_type, timestamp))
            
        return processed_tuples


    def _build_conversation_from_target(
        self,
        filtered_msgs: List[tuple],
        target_input: Optional[str],
        target_index: Optional[int],
        target_fields: Optional[list],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        
        if not target_input and not target_index and not target_fields:
            return self._build_default_conversation(filtered_msgs, verbose)

        subset = []
        if target_input:
            subset = [
                (r, c, n, t, ts) 
                for (r, c, n, t, ts) in filtered_msgs 
                if r == target_input
            ]
        else:
            subset = filtered_msgs[:]

        if not subset:
            return [{"role": "system", "content": self._runtime_system_prompt()}]

        if target_index is not None:
            subset = self._handle_index(target_index, subset)
            if not subset:
                 raise IndexError(f"Requested index/range={target_index} resulted in zero messages from a subset of size {len(filtered_msgs)}.")

        conversation = self._transform_to_conversation(subset, target_fields)
        conversation = [{"role": "system", "content": self._runtime_system_prompt()}] + conversation
        return conversation

    def _build_default_conversation(
        self,
        filtered_msgs: List[tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        conversation = self._transform_to_conversation(filtered_msgs)
        conversation = [{"role": "system", "content": self._runtime_system_prompt()}] + conversation
        return conversation

    def _apply_filters(self, messages: List[tuple]) -> List[tuple]:
        positive_filter = self._runtime_positive_filter()
        negative_filter = self._runtime_negative_filter()
        if not positive_filter and not negative_filter:
            return messages

        def matches_filter(r: str, fltr: str, msg_type) -> bool:
            if fltr == "agent":
                return msg_type == "agent"
            if fltr == "tool":
                return msg_type == "tool"
            if fltr == "process":
                return msg_type == "process"
            return (r == fltr)

        filtered = []
        for (role, content, msg_number, msg_type, timestamp) in messages:
            if positive_filter:
                if not any(matches_filter(role, pf, msg_type) for pf in positive_filter):
                    continue
            if negative_filter:
                if any(matches_filter(role, nf, msg_type) for nf in negative_filter):
                    continue
            filtered.append((role, content, msg_number, msg_type, timestamp))

        return filtered

    def _build_json_schema(self) -> Dict[str, Any]:
        def normalize_prop(spec):
            if not isinstance(spec, dict):
                return {"type": "string", "description": str(spec)}

            t = str(spec.get("type", "") or "").lower()
            if t == "list":  t = "array"
            if t == "dict":  t = "object"
            if not t:
                t = "object" if "properties" in spec else ("array" if "items" in spec else "string")

            out = {k: v for k, v in spec.items() if k not in ("type",)}
            out["type"] = t

            if t == "array" and "items" not in out:
                out["items"] = {}

            return out

        props = {}
        for field_name, desc in Agent._runtime_required_outputs_for_call(self).items():
            props[field_name] = normalize_prop(desc)

        return {
            "name": f"{self.name}_schema",
            "schema": {
                "type": "object",
                "properties": props,
                "additionalProperties": False
            }
        }

    
    def _extract_and_parse_json(self, value):
        JSON_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

        def strip_code_fences(s: str) -> str:
            if not isinstance(s, str):
                return s
            m = JSON_FENCE_RE.match(s)
            return m.group(1).strip() if m else s

        def looks_jsonish(s: str) -> bool:
            if not isinstance(s, str):
                return False
            t = strip_code_fences(s).lstrip()
            return t.startswith("{") or t.startswith("[")

        def json_unwrap_once(s: str):
            if not isinstance(s, str):
                return s
            t = s.strip()
            if not t:
                return s
            if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                t = t[1:-1].strip()
            try:
                return json.loads(t)
            except Exception:
                pass
            t2 = strip_code_fences(t)
            if t2 != t:
                try:
                    return json.loads(t2)
                except Exception:
                    pass
            return s

        def extract_first_json_lenient(s: str):
            if not isinstance(s, str):
                return None
            t = strip_code_fences(s).strip()
            try:
                return json.loads(t)
            except Exception:
                pass
            t2 = re.sub(r'(?<!\\)\n', r'\\n', t)
            try:
                return json.loads(t2)
            except Exception:
                pass
            m = re.match(r'^\s*\{\s*"response"\s*:\s*"(.*)"\s*\}\s*$', t, flags=re.DOTALL)
            if m:
                inner = m.group(1).replace('\\"', '"').replace('\\n', '\n')
                return {"response": inner}
            return None

        def extract_first_json_anywhere(s: str):
            if not isinstance(s, str):
                return None
            t = strip_code_fences(s)
            brace = t.find('{'); bracket = t.find('[')
            if brace == bracket == -1:
                return None
            if brace != -1 and (brace < bracket or bracket == -1):
                opener, closer, start = '{', '}', brace
            else:
                opener, closer, start = '[', ']', bracket
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(t)):
                ch = t[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = t[start:i+1]
                        return extract_first_json_lenient(chunk)
            if depth > 0:
                candidate = t[start:] + closer * depth
                return extract_first_json_lenient(candidate)
            return None

        def flatten_double_response(obj):
            if not isinstance(obj, dict) or "response" not in obj:
                return obj

            inner = obj["response"]

            if isinstance(inner, dict) and "response" in inner:
                return inner

            if isinstance(inner, (dict, list)):
                obj["response"] = inner
                return obj

            if isinstance(inner, str) and looks_jsonish(inner):
                unwrapped = json_unwrap_once(inner)
                if isinstance(unwrapped, dict) and "response" in unwrapped:
                    return unwrapped
                if isinstance(unwrapped, (dict, list)):
                    obj["response"] = unwrapped
                    return obj
                if isinstance(unwrapped, str):
                    parsed = extract_first_json_anywhere(unwrapped)
                    if isinstance(parsed, dict) and "response" in parsed:
                        return parsed
                    if isinstance(parsed, (dict, list)):
                        obj["response"] = parsed
                        return obj

            return obj

        if isinstance(value, (dict, list)):
            return flatten_double_response(value) if isinstance(value, dict) else value

        if not isinstance(value, str):
            return None

        s = value.strip()
        s = re.sub(r"^```json\s*|```$", "", s, flags=re.IGNORECASE).strip()

        unwrapped = json_unwrap_once(s)

        if isinstance(unwrapped, dict):
            return flatten_double_response(unwrapped)
        if isinstance(unwrapped, list):
            return unwrapped

        if isinstance(unwrapped, str):
            extracted = extract_first_json_anywhere(unwrapped)
            if isinstance(extracted, dict):
                return flatten_double_response(extracted)
            if isinstance(extracted, list):
                return extracted
            return unwrapped

        return None

            
    def _flatten_system_content(self, blocks):
        if isinstance(blocks, str):
            return blocks
        if isinstance(blocks, list):
            chunks = []
            for b in blocks:
                if b.get("type") == "text":
                    c = b.get("content")
                    if isinstance(c, (dict, list)):
                        chunks.append(json.dumps(c, ensure_ascii=False))
                    else:
                        chunks.append(str(c))
            return "\n".join(chunks)
        return str(blocks)
    
    def _call_openai_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        schema_for_response = self._build_json_schema()
        
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        
        params = {k: v for k, v in params.items() if v is not None}

        
        new_messages = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                new_messages.append({"role": "developer", "content": content})
            else:
                new_messages.append({"role": role, "content": content})

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_openai_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": new_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema_for_response
            },
            **params
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data, timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "openai",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] OPENAI HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "openai",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "openai",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            return (content, input_tokens, output_tokens)
        else:
            return content

    def _call_openrouter_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        new_messages = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            new_messages.append({"role": role, "content": content})

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_openrouter_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": new_messages,
            **params
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "openrouter",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] OPENROUTER HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "openrouter",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "openrouter",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content


    def _call_lmstudio_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False,
        base_url: Optional[str] = None
    ) -> str:
        api_base = base_url or "http://localhost:1234/v1"
        url = f"{api_base}/chat/completions"
        schema_for_response = self._build_json_schema()
        
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_lmstudio_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": conversation,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema_for_response
            },
            **params
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data, timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "lmstudio",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] LMSTUDIO HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "lmstudio",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "lmstudio",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            return (content, input_tokens, output_tokens)
        else:
            return content
    
    def _call_deepseek_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:

        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        new_messages = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            new_messages.append({"role": role, "content": content})

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_deepseek_api => model={model_name} (params = {params})")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": new_messages,
            **params
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=data, timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "deepseek",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] DEEPSEEK HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "deepseek",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "deepseek",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content


    def _call_google_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:

        system_parts = None
        contents = []
        for msg in conversation:
            role = msg["role"]
            parts = msg.get("parts") or []
            if role == "system":
                system_parts = parts
            else:
                contents.append({"role": "user" if role == "user" else "model", "parts": parts})

        def to_json_schema(spec):
            if not isinstance(spec, dict):
                return {"type": "string", "description": str(spec)}

            t = str(spec.get("type", "") or "").lower()
            if not t:
                t = "object" if "properties" in spec else ("array" if "items" in spec else "string")
            if t == "list": t = "array"
            if t == "dict": t = "object"

            node = {"type": t}
            if "description" in spec: node["description"] = spec["description"]
            if "enum" in spec: node["enum"] = spec["enum"]
            if "format" in spec: node["format"] = spec["format"]

            if t == "array":
                items = spec.get("items")
                node["items"] = to_json_schema(items) if isinstance(items, dict) else {}
                if "minItems" in spec: node["minItems"] = int(spec["minItems"])
                if "maxItems" in spec: node["maxItems"] = int(spec["maxItems"])

            if t == "object":
                props = spec.get("properties", {})
                if isinstance(props, dict) and props:
                    node["properties"] = {k: to_json_schema(v) for k, v in props.items()}
                if "required" in spec and isinstance(spec["required"], list):
                    node["required"] = spec["required"]

            if isinstance(spec.get("type"), list):
                node = {"anyOf": [to_json_schema({**spec, "type": tt}) for tt in spec["type"]]}

            return node

        required_outputs = Agent._runtime_required_outputs_for_call(self)
        prop_names = list(required_outputs.keys())
        json_schema = {
            "type": "object",
            "properties": {k: to_json_schema(required_outputs[k]) for k in prop_names},
            "required": prop_names
        }

        mp = Agent._runtime_model_params_for_call(self)
        gc = {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }
        if mp.get("temperature") is not None:
            gc["temperature"] = float(mp["temperature"])
        if mp.get("max_tokens") is not None:
            gc["maxOutputTokens"] = int(mp["max_tokens"])
        if mp.get("top_p") is not None:
            gc["topP"] = float(mp["top_p"])

        payload = {
            "contents": contents,
            "generationConfig": gc
        }
        if system_parts is not None:
            payload["system_instruction"] = {"parts": system_parts}

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_google_api => model={model_name}")

        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1alpha/models/{model_name}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json=payload, timeout=Agent._runtime_timeout_for_call(self)
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "google",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=resp
                )
            )
            if not resp: raise
            try:
                err = resp.json().get("error", {})
                msg = err.get("message") or resp.text
                det = []
                for d in err.get("details", []) or []:
                    if d.get("@type","").endswith("google.rpc.BadRequest"):
                        for v in d.get("fieldViolations", []) or []:
                            det.append(f"{v.get('field')}: {v.get('description')}")
                rid = resp.headers.get("x-goog-request-id") or resp.headers.get("x-request-id")
                extra = ("\n" + "\n".join(det)) if det else ""
                rid_s = f"\nrequest-id: {rid}" if rid else ""
                raise ValueError(f"Google API {resp.status_code}: {msg}{extra}{rid_s}")
            except ValueError:
                raise ValueError(f"Google API {resp.status_code}: {resp.text}")

        data = r.json()
        usage = data.get("usageMetadata", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "google",
                model_name,
                response=r,
                raw_response=data,
                usage=usage
            )
        )
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "google",
                model_name,
                response=r,
                raw_response=data,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            return (content, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))
        return content


    def _call_anthropic_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens", 4096),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        system_messages = [msg for msg in conversation if msg['role'] == 'system']
        system_prompt = system_messages[0]['content'] if system_messages else None
        messages = [msg for msg in conversation if msg['role'] != 'system']

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_anthropic_api => model={model_name} (params = {params})")
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
            "system": system_prompt,
            **params
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data, timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "anthropic",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] ANTHROPIC HTTP {status} ERROR:\n{body}")
            raise



        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "anthropic",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["content"][0]["text"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "anthropic",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content

    def _call_groq_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_completion_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        new_messages = []
        for msg in conversation:
            role = msg["role"]
            blocks = msg["content"]

            if role == "system":
                sys_text = self._flatten_system_content(blocks)
                new_messages.append({"role": "system", "content": sys_text})
            else:
                new_messages.append({"role": role, "content": blocks})

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_groq_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": new_messages,
            "response_format": {"type": "json_object"},
            **params
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data, timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "groq",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] GROQ HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "groq",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "groq",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content

    def _call_wavespeed_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        schema_for_response = self._build_json_schema()

        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_wavespeed_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": conversation,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema_for_response
            },
            **params
        }

        primary_url = "https://llm.wavespeed.ai/v1/chat/completions"
        fallback_url = "https://tropical-llm.wavespeed.ai/v1/chat/completions"

        def _post(url):
            resp = requests.post(url, headers=headers, json=data, timeout=Agent._runtime_timeout_for_call(self))
            resp.raise_for_status()
            return resp

        try:
            response = _post(primary_url)
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body = getattr(e.response, "text", None)
            if status == 401:
                if verbose:
                    logger.warning(f"[Agent:{self.name}] WAVESPEED 401 on primary host; retrying against tropical-llm fallback.")
                try:
                    response = _post(fallback_url)
                except requests.exceptions.RequestException as e2:
                    status2 = getattr(e2.response, "status_code", None)
                    body2 = getattr(e2.response, "text", None)
                    Agent._remember_provider_response(
                        self,
                        Agent._provider_error_metadata(
                            self,
                            "wavespeed",
                            model_name,
                            error_type=e2.__class__.__name__,
                            message=str(e2),
                            response=getattr(e2, "response", None)
                        )
                    )
                    logger.error(f"[Agent:{self.name}] WAVESPEED HTTP {status2} ERROR (fallback):\n{body2}")
                    raise
            else:
                Agent._remember_provider_response(
                    self,
                    Agent._provider_error_metadata(
                        self,
                        "wavespeed",
                        model_name,
                        error_type=e.__class__.__name__,
                        message=str(e),
                        response=getattr(e, "response", None)
                    )
                )
                logger.error(f"[Agent:{self.name}] WAVESPEED HTTP {status} ERROR:\n{body}")
                raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "wavespeed",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        content = json_response["choices"][0]["message"]["content"]
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "wavespeed",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content

    def _call_nvidia_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False,
        base_url: Optional[str] = None
    ) -> str:
        mp = Agent._runtime_model_params_for_call(self)
        params = {
            "temperature": mp.get("temperature"),
            "max_tokens": mp.get("max_tokens"),
            "top_p": mp.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        api_base = (base_url or mp.get("base_url") or "https://integrate.api.nvidia.com/v1").rstrip("/")
        url = f"{api_base}/chat/completions"

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_nvidia_api => model={model_name} (params = {params})")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model": model_name,
            "messages": conversation,
            "response_format": {"type": "json_object"},
            **params
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=Agent._runtime_timeout_for_call(self)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body = getattr(e.response, "text", None)
            Agent._remember_provider_response(
                self,
                Agent._provider_error_metadata(
                    self,
                    "nvidia",
                    model_name,
                    error_type=e.__class__.__name__,
                    message=str(e),
                    response=getattr(e, "response", None)
                )
            )
            logger.error(f"[Agent:{self.name}] NVIDIA HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        usage = json_response.get("usage", {})
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "nvidia",
                model_name,
                response=response,
                raw_response=json_response,
                usage=usage
            )
        )
        message = json_response["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        Agent._remember_provider_response(
            self,
            Agent._provider_success_metadata(
                self,
                "nvidia",
                model_name,
                response=response,
                raw_response=json_response,
                content=content,
                usage=usage
            )
        )

        if return_token_count:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        return content


class Tool(Component):
    def __init__(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        function: Callable,
        default_output: Optional[Dict[str, Any]] = None,
        description: str = None
    ):

        super().__init__(name)
        self.inputs = copy.deepcopy(inputs) if inputs is not None else {}
        self.outputs = copy.deepcopy(outputs) if outputs is not None else {}
        self.function = function
        self.default_output = copy.deepcopy(default_output) if default_output is not None else {}
        self.description = description if description else "Tool"

        self.sig = inspect.signature(function)
        self.params = list(self.sig.parameters.values())

        self.expects_manager = bool(self.params and self.params[0].name == "manager")

        self.accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self.params)
        
        valid_param_names = {p.name for p in self.params[(1 if self.expects_manager else 0):]}
        if not self.accepts_kwargs:
            unknown = [k for k in self.inputs.keys() if k not in valid_param_names]
            if unknown:
                logger.warning(f"[Tool:{self.name}] inputs absent in function signature: {unknown}")


    def to_string(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}\n"
            f"Default Output: {self.default_output}\n"
            f"Description: {self.description}"
        )

    def _runtime_default_output(self) -> Dict[str, Any]:
        resolver = getattr(self.manager, "resolve_component_runtime_config", None)
        if callable(resolver):
            cfg = resolver(self)
            if isinstance(cfg, dict) and "default_output" in cfg:
                return copy.deepcopy(cfg["default_output"])
        value_resolver = getattr(self.manager, "resolve_runtime_value", None)
        if callable(value_resolver):
            return copy.deepcopy(value_resolver(self.default_output, component_name=self.name))
        return copy.deepcopy(self.default_output)

    def _runtime_outputs(self) -> Dict[str, Any]:
        resolver = getattr(self.manager, "resolve_component_runtime_config", None)
        if callable(resolver):
            cfg = resolver(self)
            if isinstance(cfg, dict) and "outputs" in cfg:
                outputs = cfg["outputs"]
                return copy.deepcopy(outputs) if isinstance(outputs, dict) else {}
        value_resolver = getattr(self.manager, "resolve_runtime_value", None)
        if callable(value_resolver):
            outputs = value_resolver(self.outputs, component_name=self.name)
            return copy.deepcopy(outputs) if isinstance(outputs, dict) else {}
        return copy.deepcopy(self.outputs)

    def run(
            self,
            input_data: Any = None,
            target_input: Optional[str] = None,
            target_index: Optional[int] = None,
            target_fields: Optional[list] = None,
            target_custom: Optional[list] = None,
            verbose: bool = False
        ) -> dict:

            db_conn = self.manager._get_user_db()

            if isinstance(input_data, dict):
                if verbose:
                    logger.debug(f"[Tool:{self.name}] Using provided input_data directly.")
                input_dict = input_data
            else:
                if target_input and any(x in target_input for x in [":", "fn?","fn:", "?"]):
                    parsed = self.manager.parser.parse_input_string(target_input)
                    input_dict = self._gather_data_for_tool_process(parsed, db_conn, verbose)
                else:
                    input_dict = self._resolve_tool_or_process_input(db_conn, target_input, target_index, target_fields, target_custom, verbose)


            sig = getattr(self, "sig", inspect.signature(self.function))
            params = list(getattr(self, "params", list(sig.parameters.values())))

            expects_manager = getattr(self, "expects_manager", bool(params and params[0].name == "manager"))
            start = 1 if expects_manager else 0
            fparams = params[start:]

            POS_ONLY = inspect.Parameter.POSITIONAL_ONLY
            POS_OR_KW = inspect.Parameter.POSITIONAL_OR_KEYWORD
            KW_ONLY  = inspect.Parameter.KEYWORD_ONLY

            pos_only = [p for p in fparams if p.kind == POS_ONLY]
            pos_or_kw = [p for p in fparams if p.kind == POS_OR_KW]
            kw_only   = [p for p in fparams if p.kind == KW_ONLY]

            def _coerce(v):
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in ("null", "none", ""):
                        return None
                    if s == "true":
                        return True
                    if s == "false":
                        return False
                return v

            if isinstance(input_dict, dict):
                input_dict = {k: _coerce(v) for k, v in input_dict.items()}

            missing = []
            missing += [p.name for p in pos_only  if p.default is inspect._empty and p.name not in input_dict]
            missing += [p.name for p in pos_or_kw if p.default is inspect._empty and p.name not in input_dict]
            missing += [p.name for p in kw_only   if p.default is inspect._empty and p.name not in input_dict]
            if missing:
                if verbose:
                    logger.error(f"[Tool:{self.name}] Required arguments are missing: {missing}")
                return self._runtime_default_output()

            pos_args = []
            for p in pos_only:
                if p.name in input_dict:
                    pos_args.append(input_dict[p.name])
                else:
                    break

            kw_names = {p.name for p in pos_or_kw + kw_only}
            func_kwargs = {k: input_dict[k] for k in kw_names if k in input_dict}

            try:
                if expects_manager:
                    result = self.function(self.manager, *pos_args, **func_kwargs)
                else:
                    result = self.function(*pos_args, **func_kwargs)

                if not isinstance(result, dict):
                    raise TypeError(
                        f"[Tool:{self.name}] function must return a dict, "
                        f"got {type(result).__name__}."
                    )

                missing_outputs = [key for key in self._runtime_outputs() if key not in result]
                if missing_outputs:
                    raise ValueError(
                        f"[Tool:{self.name}] function result is missing required "
                        f"output keys: {missing_outputs}"
                    )

                if verbose:
                    logger.debug(f"[Tool:{self.name}] => {result}")

                result.setdefault("metadata", {})
                result["metadata"]["usd_cost"] = self.manager.cost_tool_call(self.name)
                return result

            except Exception as e:
                logger.error(f"[Tool:{self.name}] => error: {e}")
                if verbose:
                    traceback.print_exc()
                return self._runtime_default_output()


    def _resolve_tool_or_process_input(
        self,
        db_conn: sqlite3.Connection,
        target_input: Optional[str],
        target_index: Optional[int],
        target_fields: Optional[list],
        target_custom: Optional[list],
        verbose: bool
    ) -> dict:
        
        if target_custom:
            return self._gather_inputs_from_custom(db_conn, target_custom, verbose)
        else:
            return self._gather_single_input(db_conn, target_input, target_index, target_fields, verbose)


    def _gather_single_input(
        self,
        db_conn: sqlite3.Connection,
        target_input: Optional[str],
        target_index: Optional[int],
        target_fields: Optional[list],
        verbose: bool
    ) -> dict:
        all_messages = self.manager._get_all_messages(db_conn)

        if not target_input:
            subset = all_messages[:]
        else:
            subset = [
                (r, c, n, t, ts) 
                for (r, c, n, t, ts) in all_messages 
                if r == target_input
            ]

        if not subset:
            if verbose:
                logger.debug(f"[Tool:{self.name}] No messages found for target_input={target_input}, returning empty.")
            return {}

        if target_index is None:
            # default to latest for tools
            role, content, msg_number, msg_type, timestamp = subset[-1]
        else:
            if len(subset) < abs(target_index):
                raise IndexError(f"Requested index={target_index} but only {len(subset)} messages found.")
            role, content, msg_number, msg_type, timestamp = subset[target_index]


        blocks = content
        data   = self.manager._blocks_as_tool_input(blocks)

        try:
            data = self.manager._load_files_in_dict(data)

            final_data = {}

            if target_fields:
                for field in target_fields:
                    if isinstance(data, dict) and field in data:
                        final_data[field] = data[field]
            else:
                final_data = data

            return final_data
        except json.JSONDecodeError:
            if verbose:
                logger.error(f"[Tool:{self.name}] Failed to parse content as JSON: {content}")
            return {}

    def _gather_data_for_tool_process(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> dict:
        """
        Tools/Processes must produce a single dictionary of input params from the parse result.
        If multiple sources are specified, we merge them. If there's a single source, we filter fields if needed.
        """

        if verbose:
            logger.debug(f"[Tool:{self.name}] _gather_data_for_tool_process => parsed={parsed}")

        if self._has_timeline_selection(parsed):
            messages = self.manager._get_all_messages(db_conn)
            selected = self._apply_timeline_selection(
                parsed["selection"],
                messages,
                per_source_default="latest"
            )
            return self._messages_to_input_dict(selected)

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp = self.manager._get_component(parsed["component_or_param"])
                if comp:
                    
                    messages = self.manager._get_all_messages(db_conn)
                    subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in messages 
                        if r == parsed['component_or_param']
                    ]


                    if subset:
                        role, content, msg_number, msg_type, timestamp = subset[-1]

                        blocks = content
                        data   = self.manager._blocks_as_tool_input(blocks)

                        data = self.manager._load_files_in_dict(data)
                        if isinstance(data, dict):
                            return data
                        return {}
                    else:
                        return {}
                else:
                    return {}
            else:
                return {}

        if parsed["multiple_sources"]:
            final_input = {}
            for source_item in parsed["multiple_sources"]:
                part = self._collect_one_source_for_tool(source_item, db_conn)
                if isinstance(part, dict):
                    final_input.update(part)
            return final_input
        
        if parsed["single_source"]:
            return self._collect_one_source_for_tool(parsed["single_source"], db_conn)

        return {}


    def _collect_one_source_for_tool(self, source_item: dict, db_conn: sqlite3.Connection) -> dict:
        comp_name = source_item["component"]
        index = source_item["index"]
        fields = source_item["fields"]

        all_messages = self.manager._get_all_messages(db_conn)

        if comp_name:

            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_messages 
                        if r == comp_name
                    ]

        if not subset:
            return {}

        if index is None:
            index = -1
        elif isinstance(index, int):
            index = index
        else:
            raise IndexError(f"Index={index} must be an integer for Tools.")

        if not (-len(subset) <= index < len(subset)):
            return {}

        role, content, msg_number, msg_type, timestamp = subset[index]
        blocks = content
        data   = self.manager._blocks_as_tool_input(blocks)

        data = self.manager._load_files_in_dict(data)

        if not isinstance(data, dict):
            return {}

        if fields:
            filtered = {}
            for f in fields:
                if f in data:
                    filtered[f] = data[f]
            return filtered
        else:
            return data


    def _gather_inputs_from_custom(
        self,
        db_conn: sqlite3.Connection,
        target_custom: list,
        verbose: bool
    ) -> dict:
        final_input = {}
        all_messages = self.manager._get_all_messages(db_conn)

        for item in target_custom:
            comp_name = item.get("component")
            index = item.get("index", None)
            fields = item.get("fields", None)

            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_messages 
                        if r == comp_name
                    ]

            if not subset:
                if verbose:
                    logger.debug(f"[Tool/Process:{self.name}] No messages found for '{comp_name}'. Skipping.")
                continue

            if index is None:
                chosen = [subset[-1]]
            elif isinstance(index, int):
                if not (-len(subset) <= index < len(subset)):
                    raise IndexError(f"Requested index={index} but only {len(subset)} messages found for '{comp_name}'.")
                chosen = [subset[index]]
            else:
                raise IndexError(f"Index={index} must be an integer for Tools.")
                
            for (role, content, msg_number, msg_type, timestamp) in chosen:
                blocks = content
                data   = self.manager._blocks_as_tool_input(blocks)
                    
                data = self.manager._load_files_in_dict(data)
                if not isinstance(data, dict):
                    continue
                if fields:
                    extracted = {}
                    for f in fields:
                        if f in data:
                            extracted[f] = data[f]
                    final_input.update(extracted)
                else:
                    final_input.update(data)

        return final_input


class Process(Component):

    def __init__(self, name: str, function: Callable, description: str = None):
        super().__init__(name)
        self.function = function
        self.description = description if description else "Process"

        sig = inspect.signature(function)
        params = list(sig.parameters.values())
        
        self.expected_params = []
        valid_params = {"manager", "messages"}
        
        for param in params:
            if param.name not in valid_params:
                raise ValueError(
                    f"Process '{name}' has invalid parameter '{param.name}' - "
                    f"only {valid_params} are allowed"
                )
            self.expected_params.append(param.name)

        if len(params) > 2:
            raise ValueError(
                f"Process '{name}' can have maximum 2 parameters, got {len(params)}"
            )

    def to_string(self) -> str:
        return f"Name: {self.name}\nDescription: {self.description}"

    def run(
        self,
        input_data: Any = None,
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False
    ) -> Dict:
        
        db_conn = self.manager._get_user_db()

        if isinstance(input_data, dict) or isinstance(input_data, list):
            if verbose:
                logger.debug(f"[Process:{self.name}] Using user-provided input_data directly.")
            final_input = input_data

        elif target_input and any(x in target_input for x in [":", "fn?", "fn:", "?"]):
            parsed = self.manager.parser.parse_input_string(target_input)
            final_input = self._build_message_list_from_parser_result(parsed, db_conn, verbose)
        elif target_custom:
            final_input = self._build_message_list_from_custom(db_conn, target_custom, verbose)
        else:
            final_input = self._build_message_list_from_fallback(db_conn, target_input, target_index, target_fields, verbose)

        try:
            args = []
            for param_name in self.expected_params:
                if param_name == "manager":
                    args.append(self.manager)
                elif param_name == "messages":
                    args.append(final_input)

            output = self.function(*args)
                
            if not isinstance(output, (dict, list)):
                output = {"result": output}

            if verbose:
                logger.debug(f"[Process:{self.name}] => {output}")

            return output
        except (TypeError, ValueError) as e:
            if verbose:
                logger.exception(f"[Process:{self.name}] => error: {e}")
            return {}


    def _build_message_list_from_parser_result(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> List[dict]:
        if verbose:
            logger.debug(f"[Process:{self.name}] _build_message_list_from_parser_result => parsed={parsed}")

        all_msgs = self.manager._get_all_messages(db_conn)

        if self._has_timeline_selection(parsed):
            selected = self._apply_timeline_selection(
                parsed["selection"],
                all_msgs,
                per_source_default="all"
            )
            return self._transform_to_message_list(selected)

        if not parsed["multiple_sources"] and not parsed["single_source"]:
            if parsed["component_or_param"]:
                comp_name = parsed["component_or_param"]

                subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_msgs 
                        if r == comp_name
                    ]
                
                chosen = subset[-1:] if subset else []
                message_list = self._transform_to_message_list(chosen)
            else:
                chosen = all_msgs[-1:] if all_msgs else []
                message_list = self._transform_to_message_list(chosen)

        elif parsed["multiple_sources"]:
            combined = []
            for source_item in parsed["multiple_sources"]:
                partial = self._collect_msg_snippets(source_item, all_msgs)
                combined.extend(partial)
            combined.sort(key=lambda x: x[2])
            message_list = self._transform_to_message_list(combined)

        else:
            single = self._collect_msg_snippets(parsed["single_source"], all_msgs)
            message_list = self._transform_to_message_list(single)

        return message_list

    def _collect_msg_snippets(self, source_item: dict, all_msgs: List[tuple]) -> List[tuple]:
        comp_name = source_item.get("component")
        index = source_item.get("index")
        fields = source_item.get("fields")

        if comp_name:
            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_msgs 
                        if r == comp_name
                    ]
        else:
            subset = None

        if not subset:
            return []

        chosen = self._handle_index(index, subset)

        final = []
        for (role, content, msg_num, msg_type, timestamp) in chosen:
            data = content
            if fields:
                data = self.manager._filter_blocks_by_fields(
                    self.manager._to_blocks(data), fields
                )
            final.append((role, data, msg_num, msg_type, timestamp))

        return final

    def _handle_index(self, index, subset):
        if not subset:
            return []

        if index is None:
            return subset[:]

        if index == "~":
            # all
            return subset[:]

        if isinstance(index, int):
            if -len(subset) <= index < len(subset):
                return [subset[index]]
            else:
                return []

        if isinstance(index, tuple):
            start, end = index
            if start is None:
                start = 0
            else:
                start = int(start)
            if end is None:
                end = len(subset)
            else:
                end = int(end)
            return subset[start:end]

        return [subset[-1]]

    def _transform_to_message_list(self, msg_tuples: List[tuple]) -> List[dict]:
        sorted_msgs = sorted(msg_tuples, key=lambda x: x[2])

        output = []
        for (role, data, msg_num, msg_type, timestamp) in sorted_msgs:
            data = self.manager._load_files_in_dict(data)

            output.append({
                "source": role,
                "message": data,
                "msg_number": msg_num,
                "type": msg_type,
                "timestamp": timestamp
            })

        output = [{"source": m["source"], "message": m["message"], "type": m["type"], "timestamp": m["timestamp"]} for m in output]

        return output

    def _safe_json_load(self, content: str) -> Any:
        try:
            data = json.loads(content)
            return self.manager._load_files_in_dict(data)
        except json.JSONDecodeError:
            return content

    def _build_message_list_from_custom(
        self,
        db_conn: sqlite3.Connection,
        target_custom: list,
        verbose: bool
    ) -> List[dict]:
        all_msgs = self.manager._get_all_messages(db_conn)
        combined = []

        for item in target_custom:
            comp_name = item.get("component")
            idx = item.get("index", None)
            fields = item.get("fields", None)

            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_msgs 
                        if r == comp_name
                    ]

            if not subset:
                if verbose:
                    logger.debug(f"[Process:{self.name}] No messages found for '{comp_name}'. Skipping.")
                continue

            chosen = self._handle_index(idx, subset)

            final = []
            for (role, content, msg_num, msg_type, timestamp) in chosen:
                data = content
                if fields:
                    data = self.manager._filter_blocks_by_fields(
                        self.manager._to_blocks(data), fields
                    )
                final.append((role, data, msg_num, msg_type, timestamp))

            combined.extend(final)

        combined.sort(key=lambda x: x[2])
        return self._transform_to_message_list(combined)

    def _build_message_list_from_fallback(
        self,
        db_conn: sqlite3.Connection,
        target_input: Optional[str],
        target_index: Optional[int],
        target_fields: Optional[list],
        verbose: bool
    ) -> List[dict]:
        all_msgs = self.manager._get_all_messages(db_conn)

        if not all_msgs:
            return []

        subset = (
            [row for row in all_msgs if row[0] == target_input]
            if target_input else all_msgs
        )
        if not subset:
            return []

        chosen = self._handle_index(target_index, subset)

        if target_fields:
            tmp = []
            for (role, content, num, typ, ts) in chosen:
                blocks = self.manager._filter_blocks_by_fields(content, target_fields)
                tmp.append((role, blocks, num, typ, ts))
            chosen = tmp

        return self._transform_to_message_list(chosen)


class Automation(Component):
    def __init__(self, name: str, sequence: List[Union[str, dict]], description: str = None):
        super().__init__(name)
        self.sequence = sequence
        self.description = description if description else "Automation"

    def to_string(self) -> str:
        return f"Name: {self.name}\nSequence: {self.sequence}\nDescription: {self.description}"

    def run(self, verbose: bool = False,
            on_update: Optional[Callable] = None,
            on_update_params: Optional[Dict] = None,
            return_token_count = False) -> Dict:
        db_conn = self.manager._get_user_db()
        current_output = {}

        for step in self.sequence:
            current_output = self._execute_step(step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)

        if verbose:
            logger.info(f"[Automation:{self.name}] => Execution completed.")
        return current_output

    def _execute_step(self, step, current_output, db_conn, verbose,
                      on_update = None, on_update_params = None,
                      return_token_count = False):
        resolver = getattr(self.manager, "resolve_runtime_value", None)
        if callable(resolver):
            step = resolver(step, component_name=self.name)

        if isinstance(step, str):
            (
                comp_name,
                parsed_target_input,
                parsed_target_index,
                parsed_target_fields,
                parsed_target_custom
            ) = self._parse_automation_step_string(step)

            comp = self.manager._get_component(comp_name)
            if not comp:
                if verbose:
                    logger.warning(f"[Automation:{self.name}] => no such component '{comp_name}'. Skipping.")
                return current_output

            if verbose:
                logger.debug(f"[Automation:{self.name}] => running component '{comp_name}'")

            if isinstance(comp, Automation):
                step_output = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager=None: self.manager._invoke_callback(on_update, messages, manager),
                    return_token_count=return_token_count
                )
            elif isinstance(comp, Agent):
                step_output = comp.run(
                    verbose=verbose,
                    target_input=parsed_target_input,
                    target_index=parsed_target_index,
                    target_fields=parsed_target_fields,
                    target_custom=parsed_target_custom,
                    return_token_count=return_token_count
                )
            else:
                step_output = comp.run(
                    verbose=verbose,
                    target_input=parsed_target_input,
                    target_index=parsed_target_index,
                    target_fields=parsed_target_fields,
                    target_custom=parsed_target_custom
                )
            self.manager._save_component_output(db_conn, comp, step_output, verbose)

            if on_update:
                self.manager._invoke_callback(
                    on_update,
                    self.manager.get_messages(self.manager._active_user_id()),
                    self.manager,
                    on_update_params,
                )
            return step_output

        elif isinstance(step, dict):
            control_flow_type = step.get("control_flow_type")

            if control_flow_type == "branch":
                condition = step.get("condition")
                condition_met = self._evaluate_condition(condition, current_output)

                if verbose:
                    logger.debug(f"[Automation:{self.name}] => branching condition evaluated to {condition_met}.")

                next_steps = step["if_true"] if condition_met else step["if_false"]
                for branch_step in next_steps:
                    current_output = self._execute_step(branch_step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)

            elif control_flow_type == "while":
                run_first_pass = step.get("run_first_pass", True)
                start_condition = step.get("start_condition", step.get("condition", run_first_pass))
                end_condition = step.get("end_condition", step.get("condition"))
                body = step.get("body", [])

                if (isinstance(start_condition, bool) and start_condition) or \
                (isinstance(start_condition, (str, dict)) and self._evaluate_condition(start_condition, current_output)):
                    while True:
                        if verbose:
                            logger.debug(f"[Automation:{self.name}] => executing while loop body.")
                        for nested_step in body:
                            current_output = self._execute_step(nested_step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)

                        if (isinstance(end_condition, bool) and end_condition) or \
                        (isinstance(end_condition, (str, dict)) and self._evaluate_condition(end_condition, current_output)):
                            break
                        if verbose:
                            logger.debug(f"[Automation:{self.name}] => while loop iteration complete.")

            elif control_flow_type == "for":
                items_spec = step.get("items")
                if verbose:
                    logger.debug(f"[Automation:{self.name}] Processing FOR loop with items: {items_spec}")

                items_data = self._resolve_items_spec(items_spec, db_conn, verbose)

                elements = self._generate_elements_from_items(items_data, verbose)
                body = step.get("body", [])

                for idx, element in enumerate(elements):
                    if verbose:
                        logger.debug(f"[Automation:{self.name}] FOR loop iteration {idx+1}/{len(elements)}")
                    
                    iterator_msg = {
                        "item_number": idx,
                        "item": element
                    }
                    self.manager._save_message(db_conn, "iterator", iterator_msg, "iterator")

                    for nested_step in body:
                        current_output = self._execute_step(
                            nested_step, current_output, db_conn, 
                            verbose, on_update, on_update_params,
                            return_token_count
                        )

            elif control_flow_type == "switch":
                switch_value = self._resolve_switch_value(step["value"], db_conn, verbose)
                executed = False
                
                for case in step.get("cases", []):
                    case_value = case.get("case")
                    case_body = case.get("body", [])
                    
                    if case_value == "default":
                        continue
                        
                    if self._case_matches(switch_value, case_value, verbose):
                        if verbose:
                            logger.debug(f"[Automation:{self.name}] Switch matched case: {case_value}")
                        for nested_step in case_body:
                            current_output = self._execute_step(nested_step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)
                        executed = True
                        break
                
                if not executed:
                    for case in step.get("cases", []):
                        if case.get("case") == "default":
                            if verbose:
                                logger.debug(f"[Automation:{self.name}] Executing default case")
                            for nested_step in case.get("body", []):
                                current_output = self._execute_step(nested_step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)
                            break

            else:
                raise ValueError(f"Unsupported control flow type: {control_flow_type}")

        return current_output

    def _resolve_switch_value(self, value_spec, db_conn, verbose):
        """Resolve switch value using parser logic"""
        if isinstance(value_spec, str) and not value_spec.startswith(":"):
            last_blocks = self.manager._get_all_messages(db_conn)[-1][1]
            data_dict   = self.manager._blocks_as_tool_input(last_blocks)
            return data_dict.get(value_spec)
        
        parsed = self.manager.parser.parse_input_string(value_spec)
        resolved = self._gather_data_for_parser_result(parsed, db_conn)
        
        if isinstance(resolved, dict):
            if len(resolved) == 1:
                return next(iter(resolved.values()))
            raise ValueError(f"Switch value resolved to multi-field dict: {resolved.keys()}")
        
        if isinstance(resolved, list):
            if len(resolved) == 1:
                return resolved[0]
            raise ValueError(f"Switch value resolved to multiple messages: {len(resolved)}")
        
        return resolved

    def _case_matches(self, switch_value, case_value, verbose):
        """Compare switch/case values more flexibly."""
        if isinstance(switch_value, list) and len(switch_value) == 1:
            switch_value = switch_value[0]
        if isinstance(case_value, list) and len(case_value) == 1:
            case_value = case_value[0]
        if isinstance(switch_value, str):
            switch_value = switch_value.strip().lower()
        if isinstance(case_value, str):
            case_value = case_value.strip().lower()
        try:
            if isinstance(switch_value, (int, float)) and isinstance(case_value, (int, float)):
                return float(switch_value) == float(case_value)
            return switch_value == case_value
        except (TypeError, ValueError) as e:
            if verbose:
                logger.error(f"[Automation] Case comparison error: {e}")
            return False

    def _resolve_items_spec(self, items_spec, db_conn, verbose):
        if isinstance(items_spec, (int, float)):
            return int(items_spec)
        
        if isinstance(items_spec, list) and all(isinstance(x, (int, float)) for x in items_spec):
            return items_spec
        
        if isinstance(items_spec, str):
            parsed = self.manager.parser.parse_input_string(items_spec)
            resolved_data = self._gather_data_for_parser_result(parsed, db_conn)
            
            if isinstance(resolved_data, list):
                if len(resolved_data) > 1:
                    raise ValueError(f"[Automation] FOR loop items spec '{items_spec}' resolved to multiple messages")
                resolved_data = resolved_data[0] if resolved_data else None
            
            if verbose:
                logger.debug(f"[Automation:{self.name}] Resolved items spec '{items_spec}' to: {resolved_data}")
            return resolved_data
        
        return items_spec

    def _generate_elements_from_items(self, items_data, verbose):
        if items_data is None:
            return []
        
        if isinstance(items_data, (int, float)):
            return list(range(int(items_data)))
        
        if isinstance(items_data, list):
            if len(items_data) in (2, 3) and all(isinstance(x, (int, float)) for x in items_data):
                params = [int(x) for x in items_data]
                return list(range(*params))
            return items_data
        
        if isinstance(items_data, dict):
            if len(items_data) == 1:
                return self._generate_elements_from_items(next(iter(items_data.values())), verbose)
            
            return [{"key": k, "value": v} for k, v in items_data.items()]
        
        return [items_data]

    def _evaluate_condition(self, condition, current_output) -> bool:

        db_conn = self.manager._get_user_db()

        def _truthy(value) -> bool:
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"false", "f", "no", "n", "0", "none", "null", ""}:
                    return False
                if normalized in {"true", "t", "yes", "y", "1"}:
                    return True
            return bool(value)

        if isinstance(condition, str):
            parsed = self.manager.parser.parse_input_string(condition)

            if parsed["is_function"] and parsed["function_name"]:
                input_data = self._gather_data_for_parser_result(parsed, db_conn)
                fn = self.manager._get_function_from_string(parsed["function_name"])
                result = fn(input_data)
                if not isinstance(result, bool):
                    raise ValueError(
                        f"[Automation:{self.name}] Condition function '{parsed['function_name']}' did not return a bool."
                    )
                return result
            else:
                data = self._gather_data_for_parser_result(parsed, db_conn)
                if isinstance(data, dict):
                    return all(_truthy(v) for v in data.values())
                elif isinstance(data, list):
                    return all(_truthy(item) for item in data)
                else:
                    return _truthy(data)

        if isinstance(condition, dict):
            input_spec = condition.get("input")
            target_value = condition.get("value")

            if not input_spec:
                raise ValueError(f"[Automation:{self.name}] Invalid condition: 'input' field is missing in dict condition.")

            parsed = self.manager.parser.parse_input_string(input_spec)

            if parsed["is_function"] and parsed["function_name"]:
                input_data = self._gather_data_for_parser_result(parsed, db_conn)
                fn = self.manager._get_function_from_string(parsed["function_name"])
                result = fn(input_data)
                return (result == target_value)

            data = self._gather_data_for_parser_result(parsed, db_conn)
            return (data == target_value)

        raise ValueError(f"[Automation:{self.name}] Unsupported condition type: {condition}")
    
    def _gather_data_for_parser_result(self, parsed: dict, db_conn: sqlite3.Connection) -> Any:

        if self._has_timeline_selection(parsed):
            messages = self.manager._get_all_messages(db_conn)
            selected = self._apply_timeline_selection(
                parsed["selection"],
                messages,
                per_source_default="latest"
            )
            return self._messages_to_input_dict(selected)

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp = self.manager._get_component(parsed["component_or_param"])
                if comp:
                    messages = self.manager._get_all_messages(db_conn)

                    filtered = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in messages 
                        if r == parsed["component_or_param"]
                    ]

                    if filtered:
                        role, content, msg_number, msg_type, timestamp = filtered[-1]
                        blocks = content
                        data_dict = self.manager._blocks_as_tool_input(blocks)
                        return data_dict
                    else:
                        return None
                else:
                    subset = self.manager._get_all_messages(db_conn)
                    role, content, msg_number, msg_type, timestamp = subset[-1]
                    
                    blocks = content
                    data   = self.manager._blocks_as_tool_input(blocks)

                    return data[parsed["component_or_param"]]
            else:
                return None

        if parsed["multiple_sources"]:
            final_data = {}
            for source_item in parsed["multiple_sources"]:
                partial_data = self._collect_single_source_data(source_item, db_conn)
                if isinstance(partial_data, dict):
                    final_data.update(partial_data)
                else:
                    comp_key = source_item["component"] if source_item["component"] else "unnamed"
                    final_data[comp_key] = partial_data
            return final_data

        if parsed["single_source"]:
            return self._collect_single_source_data(parsed["single_source"], db_conn)
        
        return None


    def _collect_single_source_data(self, source_item: dict, db_conn: sqlite3.Connection) -> Any:
        comp_name = source_item["component"]
        index = source_item["index"]
        fields = source_item["fields"]

        all_messages = self.manager._get_all_messages(db_conn)

        if comp_name:
            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_messages 
                        if r == comp_name
                    ]
        else:
            subset = None

        if not subset:
            return None

        index_to_use = index if index is not None else -1
        if not isinstance(index_to_use, int):
            return None
        if not (-len(subset) <= index_to_use < len(subset)):
            return None

        role, content, msg_number, msg_type, timestamp = subset[index_to_use]

        blocks = content
        base_dict = self.manager._blocks_as_tool_input(blocks)

        if fields:
            return {k: base_dict[k] for k in fields if k in base_dict}
        return base_dict


        
    def _parse_automation_step_string(self, step_str: str) -> tuple:
        component_name = step_str
        target_input = None
        target_index = None
        target_custom = None
        target_fields = None
        

        parsed = self.manager.parser.parse_input_string(step_str)

        if self._has_timeline_selection(parsed):
            component_name = parsed["component_or_param"] if parsed["component_or_param"] else step_str
            if ":" in step_str:
                target_input = ":" + step_str.split(":", 1)[1].strip()
            else:
                target_input = step_str
        elif parsed["multiple_sources"]:
            component_name = parsed["component_or_param"]
            target_custom = parsed["multiple_sources"]
        elif parsed["single_source"]:
             component_name = parsed["component_or_param"] if parsed["component_or_param"] else step_str
             target_input = parsed["single_source"].get("component")
             target_index = parsed["single_source"].get("index")
             target_fields = parsed["single_source"].get("fields")

        elif parsed["component_or_param"]:
             component_name = parsed["component_or_param"]
            
        
        return (component_name, target_input, target_index, target_fields, target_custom)


