# mas.py

import os
import json
import sqlite3
import traceback
import uuid
import threading
from typing import Optional, List, Dict, Callable, Any, Union
import pickle
import importlib.util
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import re
import tempfile
import requests
from dotenv import load_dotenv
import inspect
from datetime import datetime

from importlib import resources
import logging

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Optionally, add a handler just for your logger
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

class Component:
    def __init__(self, name: str):
        self.name = name  # Developer-friendly name (no prefix).
        self.manager = None  # Assigned when the manager creates the component.

    def to_string(self) -> str:
        return f"Component(Name: {self.name})"


    def run(self, input_data: Any = None,  # Not used, but kept for interface consistency
        verbose: bool = False) -> Dict:
        raise NotImplementedError("Subclass must implement .run().")


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
        description: str = None
    ):
        super().__init__(name)
        self.system_prompt = system_prompt
        self.system_prompt_original = system_prompt_original
        if isinstance(required_outputs, str):
            self.required_outputs = {"response": required_outputs}
        else:
            self.required_outputs = required_outputs

        self.models = models
        self.default_output = default_output or {"response": "No valid response."}
        self.positive_filter = positive_filter
        self.negative_filter = negative_filter
        self.general_system_description = general_system_description
        self.model_params = model_params or {}
        self.include_timestamp = include_timestamp
        self.description = description if description else "Agent"

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

    def run(
        self,
        input_data: Any = None,  # Not used, but kept for interface consistency
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> dict:

        if verbose:
            logger.debug(f"[Agent:{self.name}] .run() => reading conversation history from DB")

        db_conn = self.manager._get_user_db()

        # 1) Check for advanced parser usage
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
            logger.debug(f"[Agent:{self.name}] Final conversation for model input:\n\n{json.dumps(conversation, indent=2)}\n")

        # Now we call each model in self.models in order:
        for model_info in self.models:
            provider = model_info["provider"].lower()
            model_name = model_info["model"]
            api_key = self.manager.get_key(provider)

            if not api_key:
                if verbose:
                    logger.warning(f"[Agent:{self.name}] No API key for '{provider}'. Skipping.")
                continue
            
            formatted_conversation = self._provider_format_messages(provider, conversation)

            try:
                response = None
                
                if provider == "openai":
                    response = self._call_openai_api(
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
                else:
                    raise ValueError(f"[Agent:{self.name}] Unknown provider '{provider}'")

                # Now unpack the result if return_token_count is True:
                if return_token_count:
                    response_str, input_tokens, output_tokens = response
                else:
                    response_str = response

                response_dict = self._extract_and_parse_json(response_str)

                if response_dict is None:
                    response_dict = {"response": response_str}

                response_blocks = self.manager._to_blocks(
                    response_dict, self.manager._current_user_id
                )

                self._last_used_model = {"provider": provider, "model": model_name}
                if verbose:
                    logger.debug(f"[Agent:{self.name}] => success from provider={provider}\n{response_dict}")
                
                if return_token_count:
                    # empaquetamos los metadatos en el primer bloque de texto
                    response_blocks[0].setdefault("metadata", {})
                    md = response_blocks[0]["metadata"]
                    md["input_tokens"]  = input_tokens
                    md["output_tokens"] = output_tokens
                    md["usd_cost"]      = self.manager.cost_model_call(
                        provider, model_name, input_tokens, output_tokens
                    )
                return response_blocks
                
            except requests.exceptions.RequestException as req_err:
                if verbose:
                    logger.error(f"[Agent:{self.name}] HTTP error with provider={provider}, model={model_name}: {req_err}")
            except json.JSONDecodeError as json_err:
                if verbose:
                    logger.error(f"[Agent:{self.name}] JSON decoding error with provider={provider}, model={model_name}: {json_err}")
            except ValueError as val_err:
                if verbose:
                    logger.error(f"[Agent:{self.name}] Value error with provider={provider}, model={model_name}: {val_err}")

        if verbose:
            logger.warning(f"[Agent:{self.name}] => All providers failed. Returning default:\n{self.default_output}")
        return self.default_output

    def _as_blocks(self, value):
        """
        Con la normalización global, *todo* lo que viene de la DB
        ya es List[Block]. Este helper sólo garantiza eso para
        casos heredados o tests unitarios.
        """
        if isinstance(value, list) and value and all(
            isinstance(b, dict) and "type" in b and "content" in b for b in value
        ):
            return value

        # Fallback: pedile al Manager que lo normalice
        return self.manager._to_blocks(value)

    def _provider_format_messages(self, provider: str, conversation: list) -> list:
        """
        Convierte cada mensaje (que SIEMPRE es List[Block])
        al formato requerido por cada proveedor.
        """
        import base64, mimetypes, os, requests, json
        provider = provider.lower()
        out = []

        # ---- helpers internos --------------------------------
        def _image_to_datauri(path):
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"

        # ---- conversión --------------------------------------
        for msg in conversation:
            role   = msg["role"]
            blocks = self._as_blocks(msg["content"])

            if provider in {"openai", "groq", "deepseek"}:
                # OpenAI / Groq style: user puede ser multimodal, system/assistant solo texto
                if role == "user":
                    parts = []
                    for blk in blocks:
                        if blk["type"] == "text":
                            parts.append({"type": "text", "text": blk["content"]})
                        elif blk["type"] == "image":
                            src = blk["content"]
                            if src["kind"] == "file":
                                url = _image_to_datauri(src["path"])
                            elif src["kind"] == "url":
                                url = src["url"]
                            else:   # b64 directo
                                url = f"data:image/*;base64,{src['b64']}"
                            parts.append({"type": "image_url",
                                          "image_url": {"url": url,
                                                        "detail": src.get("detail","auto")}})
                    out.append({"role": role, "content": parts})
                else:   # assistant / system
                    text = "\n".join(
                        blk["content"] if blk["type"] == "text"
                        else "[image omitted]"
                        for blk in blocks
                    )
                    out.append({"role": role, "content": text})

            elif provider == "google":        # Gemini
                gem_parts = []
                for blk in blocks:
                    if blk["type"] == "text":
                        gem_parts.append({"text": blk["content"]})
                    elif blk["type"] == "image":
                        src = blk["content"]
                        if src["kind"] == "file":
                            data = open(src["path"],"rb").read()
                        elif src["kind"] == "url":
                            data = requests.get(src["url"]).content
                        else:
                            data = base64.b64decode(src["b64"])
                        gem_parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(data).decode()
                            }
                        })
                if role == "system":
                    out.append({"role": role, "parts": gem_parts})
                else:
                    out.append({"role": "user" if role=="user" else "model",
                                "parts": gem_parts})

            elif provider == "anthropic":     # Claude
                c_blocks = []
                for blk in blocks:
                    if blk["type"] == "text":
                        c_blocks.append({"type":"text", "text": blk["content"]})
                    elif blk["type"] == "image":
                        src = blk["content"]
                        if src["kind"] == "file":
                            data = open(src["path"],"rb").read()
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
                out.append({"role": role, "content": c_blocks})

            else:
                # Por defecto: devolvemos los bloques sin tocar
                out.append({"role": role, "content": blocks})

        return out


    def _build_conversation_from_parser_result(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> List[Dict[str, Any]]:
        if verbose:
            logger.debug(f"[Agent:{self.name}] _build_conversation_from_parser_result => parsed={parsed}")

        all_msgs = self.manager._get_all_messages(db_conn)
        filtered_msgs = self._apply_filters(all_msgs)

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
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
            # Now sort by msg_number ascending
            combined.sort(key=lambda x: x[2])
            conversation = self._transform_to_conversation(combined)
        else:
            partial = self._collect_msg_snippets_for_agent(parsed["single_source"], filtered_msgs)
            conversation = self._transform_to_conversation(partial)

        # Prepend system
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
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
            chosen = subset[:]  # all (default for agents)
        elif isinstance(index, int):
            if len(subset) >= abs(index):
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
            chosen = subset[:] # Fallback to all if index is not recognized

        return chosen


    def _transform_to_conversation(self, msg_tuples: List[tuple], fields: Optional[list] = None, include_message_number: Optional[bool] = False) -> List[Dict[str, str]]:
        conversation = []
        for (role, content, msg_number, msg_type, timestamp) in msg_tuples:
            data = content

            if fields:
                data = self.manager._filter_blocks_by_fields(
                    self._as_blocks(data), fields
                )

            if isinstance(data, list) and all(isinstance(b, dict) and "type" in b for b in data):
                blocks = [b.copy() for b in data]
            else:
                blocks = self._as_blocks(data)

            if role != self.name:
                source = f"{msg_type}: {role}"
                for blk in blocks:
                    if blk["type"] == "text":
                        # envolvemos el texto original dentro del JSON wrapper
                        original = blk["content"]

                        wrapper_obj = {
                            "source": source,
                            "message": original
                        }

                        if self.include_timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp)
                                formatted_ts = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except (ValueError, TypeError):
                                formatted_ts = timestamp

                            wrapper_obj["timestamp"] = formatted_ts

                        # reemplazamos el campo "text" por la serialización
                        blk["content"] = json.dumps(wrapper_obj, ensure_ascii=False)

            conversation.append({
                "role": "assistant" if role == self.name else "user",
                "content": blocks,
                "msg_number": msg_number
            })

        # Sort by msg_number to preserve chronological order
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

                # Filter by component
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
            
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
            return conversation
    
    def _handle_fields(self, msg_tuples, fields):
        conversation = []

        for (role, content, msg_number, msg_type, timestamp) in msg_tuples:
            data = content

            if fields:
                data = self.manager._filter_blocks_by_fields(
                    self._as_blocks(data), fields
                )

            conversation.append((role, data, msg_number, msg_type, timestamp))

        return conversation


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
            return [{"role": "system", "content": self.system_prompt}]

        if target_index is not None:
            if len(subset) < abs(target_index):
                raise IndexError(f"Requested index={target_index} but only {len(subset)} messages available.")
            subset = [subset[target_index]]

        conversation = self._transform_to_conversation(subset, target_fields)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def _build_default_conversation(
        self,
        filtered_msgs: List[tuple],
        verbose: bool
    ) -> List[Dict[str, Any]]:
        conversation = self._transform_to_conversation(filtered_msgs)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        return conversation

    def _apply_filters(self, messages: List[tuple]) -> List[tuple]:
        if not self.positive_filter and not self.negative_filter:
            return messages

        def matches_filter(r: str, fltr: str, msg_type) -> bool:
            if fltr == "agent":
                return msg_type == "agent"
            if fltr == "tool":
                return msg_type == "tool"
            if fltr == "process":
                return msg_type == "process"
            # exact match
            return (r == fltr)

        filtered = []
        for (role, content, msg_number, msg_type, timestamp) in messages:
            if self.positive_filter:
                if not any(matches_filter(role, pf, msg_type) for pf in self.positive_filter):
                    continue
            if self.negative_filter:
                if any(matches_filter(role, nf, msg_type) for nf in self.negative_filter):
                    continue
            filtered.append((role, content, msg_number, msg_type, timestamp))

        return filtered

    def _build_json_schema(self) -> Dict[str, Any]:
        props = {}
        for field_name, desc in self.required_outputs.items():
            if isinstance(desc, dict):
                props[field_name] = {
                    "description": desc.get("description", ""),
                    "type": desc.get("type", "string")
                }
            else:
                props[field_name] = {"description": desc, "type": "string"}

        return {
            "name": f"{self.name}_schema",
            "schema": {
                "type": "object",
                "properties": props,
                "additionalProperties": False
            }
        }
    
    def _extract_and_parse_json(self, text):
        """
        Extracts and parses the first valid JSON object or array from a string.

        :param text: The input text containing JSON and possibly extra markup.
        :return: Parsed JSON as a Python object, or None if no valid JSON is found.
        """
        text = text.strip()  # Remove leading/trailing whitespace
        for start in range(len(text)):
            if text[start] in ('{', '['):  # JSON must start with { or [
                for end in range(len(text), start, -1):
                    try:
                        # Attempt to parse substring as JSON
                        potential_json = text[start:end]
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        continue  # Keep shrinking the window
        return None  # No valid JSON found
    
    def _flatten_system_content(self, blocks):
        # puede venir string o lista de bloques
        if isinstance(blocks, str):
            return blocks
        if isinstance(blocks, list):
            return "\n".join(b["content"] for b in blocks if b.get("type") == "text")
        # fallback
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
        
        params = {
            "temperature": self.model_params.get("temperature"),
            "max_tokens": self.model_params.get("max_tokens"),
            "top_p": self.model_params.get("top_p")
        }
        # Filter out None values
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
                json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] OPENAI HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        content = json_response["choices"][0]["message"]["content"]

        if return_token_count:
            usage = json_response.get("usage", {})
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
        
        params = {
            "temperature": self.model_params.get("temperature"),
            "max_tokens": self.model_params.get("max_tokens"),
            "top_p": self.model_params.get("top_p")
        }
        # Filter out None values
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
                json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] LMSTUDIO HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        content = json_response["choices"][0]["message"]["content"]

        if return_token_count:
            usage = json_response.get("usage", {})
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

        params = {
            "temperature": self.model_params.get("temperature"),
            "max_tokens": self.model_params.get("max_tokens"),
            "top_p": self.model_params.get("top_p")
        }
        # Filter out None values
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
                json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] DEEPSEEK HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        content = json_response["choices"][0]["message"]["content"]

        if return_token_count:
            usage = json_response.get("usage", {})
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
        """
        Calls the Google Gemini API to get a response.
        """
        params = {
            "temperature": self.model_params.get("temperature"),
            "max_output_tokens": self.model_params.get("max_tokens"),
            "top_p": self.model_params.get("top_p")
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        system_parts = None
        contents = []
        for msg in conversation:

            role = msg["role"]
            parts = msg.get("parts")


            if role == "system":
                system_parts = parts or []
            else:
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": parts or []
                })

        if verbose:
            logger.debug(f"[Agent:{self.name}] _call_google_api => model={model_name} (params = {params})")
        
        payload = {
            "contents": contents,
            "generationConfig": params
        }
        if system_parts is not None:
            payload["systemInstruction"] = {"parts": system_parts}

        

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] GOOGLE HTTP {status} ERROR:\n{body}")
            raise




        response_json = response.json()
        
        content = response_json["candidates"][0]["content"]["parts"][0]["text"]

        if return_token_count:
            # The Gemini API response includes a "usageMetadata" field
            usage = response_json.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            return (content, input_tokens, output_tokens)
        else:
            return content

    def _call_anthropic_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool,
        return_token_count: bool = False
    ) -> str:
        
        params = {
            "temperature": self.model_params.get("temperature"),
            "max_tokens": self.model_params.get("max_tokens", 4096),
            "top_p": self.model_params.get("top_p")
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        # Extract system messages (anywhere in conversation)
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
                json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] ANTHROPIC HTTP {status} ERROR:\n{body}")
            raise



        json_response = response.json()
    
        content = json_response["content"][0]["text"]

        if return_token_count:
            usage = json_response.get("usage", {})
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
        
        params = {
            "temperature": self.model_params.get("temperature"),
            "max_completion_tokens": self.model_params.get("max_tokens"),
            "top_p": self.model_params.get("top_p")
        }
        params = {k: v for k, v in params.items() if v is not None}

        new_messages = []
        for msg in conversation:
            role = msg["role"]
            blocks = msg["content"]       # lista de bloques

            if role == "system":
                # Groq (y la 1.0 de OpenAI) exigen string
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
                json=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # imprime status + body
            status = getattr(e.response, "status_code", None)
            body   = getattr(e.response, "text", None)
            logger.error(f"[Agent:{self.name}] GROQ HTTP {status} ERROR:\n{body}")
            raise

        json_response = response.json()
        content = json_response["choices"][0]["message"]["content"]

        if return_token_count:
            usage = json_response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return (content, input_tokens, output_tokens)
        else:
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
        self.inputs = inputs
        self.outputs = outputs
        self.function = function
        self.default_output = default_output or {}
        self.description = description if description else "Tool"

        # Check if function expects manager
        sig = inspect.signature(function)
        params = list(sig.parameters.values())
        
        # Look for "manager" parameter in first position
        self.expects_manager = False
        if len(params) > 0:
            self.expects_manager = params[0].name == "manager"

        # Validate parameter count (either inputs or inputs+1 if has manager)
        expected_params = len(inputs) + (1 if self.expects_manager else 0)
        if len(params) != expected_params:
            raise ValueError(
                f"Tool '{name}' function requires {expected_params} parameters "
                f"(manager? + inputs), but has {len(params)}"
            )

    def to_string(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}\n"
            f"Default Output: {self.default_output}\n"
            f"Description: {self.description}"
        )

    def run(
        self,
        input_data: Any = None,
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False
    ) -> dict:

        if verbose:
            logger.debug(
                f"[Tool:{self.name}] .run() => target_input={target_input}, "
                f"target_index={target_index}, target_fields={target_fields}, target_custom={target_custom}, input_data={input_data}"
            )

        db_conn = self.manager._get_user_db()

        if isinstance(input_data, dict):
            if verbose:
                logger.debug(f"[Tool:{self.name}] Using provided input_data directly.")
            input_dict = input_data
        else:
            # If we have a `target_input` string that might hold advanced syntax => parse
            if target_input and any(x in target_input for x in [":", "fn?","fn:", "?"]):
                parsed = self.manager.parser.parse_input_string(target_input)
                # Now collect data from DB
                input_dict = self._gather_data_for_tool_process(parsed, db_conn, verbose)
            else:
                input_dict = self._resolve_tool_or_process_input(db_conn, target_input, target_index, target_fields, target_custom, verbose)

        func_args = []
        if self.expects_manager:
            func_args.append(self.manager)
            
        for arg_name in self.inputs:
            if arg_name not in input_dict:
                raise ValueError(f"[Tool:{self.name}] Missing required input: {arg_name}")
            func_args.append(input_dict[arg_name])

        # Call the tool function
        try:
            result = self.function(*func_args)

            if len(result) != len(self.outputs):
                raise ValueError(
                    f"[Tool:{self.name}] function returned {len(result)} items, "
                    f"but {len(self.outputs)} expected."
                )

            if verbose:
                logger.debug(f"[Tool:{self.name}] => {result}")

            result.setdefault("metadata", {})
            result["metadata"]["usd_cost"] = self.manager.cost_tool_call(self.name)
            return result

        except (TypeError, ValueError) as e:
            if verbose:
                logger.error(f"[Tool:{self.name}] => error: {e}")
                traceback.print_exc()
            return self.default_output

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
            # Combine multiple partial inputs
            return self._gather_inputs_from_custom(db_conn, target_custom, verbose)
        else:
            # Single input source
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


        blocks = content                           # ya es List[Block]
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

                        blocks = content                           # ya es List[Block]
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

        # If multiple sources => gather them all and merge
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

        if len(subset) < abs(index):
            return {}

        role, content, msg_number, msg_type, timestamp = subset[index]
        blocks = content                           # ya es List[Block]
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
        """
        target_custom: a list of dicts like:
        {
            "component": "myagent" or "user",
            "index": -2 (optional),
            "fields": ["foo", "bar"] (optional)
        }
        We gather each piece, put them in a single dict (merging).
        If there's a field collision, the last one overwrites.
        """
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
                if len(subset) < abs(index):
                    raise IndexError(f"Requested index={index} but only {len(subset)} messages found for '{comp_name}'.")
                chosen = [subset[index]]
            else:
                raise IndexError(f"Index={index} must be an integer for Tools.")
                
            for (role, content, msg_number, msg_type, timestamp) in chosen:
                blocks = content                           # ya es List[Block]
                data   = self.manager._blocks_as_tool_input(blocks)
                    
                data = self.manager._load_files_in_dict(data)
                if not isinstance(data, dict):
                    continue
                if fields:
                    # subset the fields
                    extracted = {}
                    for f in fields:
                        if f in data:
                            extracted[f] = data[f]
                    # merge
                    final_input.update(extracted)
                else:
                    # merge entire data
                    final_input.update(data)

        return final_input


class Process(Component):

    def __init__(self, name: str, function: Callable, description: str = None):
        super().__init__(name)
        self.function = function
        self.description = description if description else "Process"

        # Check if function expects manager
        sig = inspect.signature(function)
        params = list(sig.parameters.values())
        
        # Track expected parameters
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
        
        if verbose:
            logger.debug(f"[Process:{self.name}] .run() => target_input={target_input}, "
                  f"target_index={target_index}, target_fields={target_fields}, target_custom={target_custom}, input_data={input_data}")

        db_conn = self.manager._get_user_db()

        # 1) If user explicitly provided a dict (or list) in 'input_data', pass that directly
        if isinstance(input_data, dict) or isinstance(input_data, list):
            if verbose:
                logger.debug(f"[Process:{self.name}] Using user-provided input_data directly.")
            final_input = input_data

        # 2) If advanced parser syntax is in target_input, build the message list from that
        elif target_input and any(x in target_input for x in [":", "fn?", "fn:", "?"]):
            parsed = self.manager.parser.parse_input_string(target_input)
            if verbose:
                logger.debug(f"[Process:{self.name}] Detected advanced parser usage => parsed={parsed}")
            final_input = self._build_message_list_from_parser_result(parsed, db_conn, verbose)
        elif target_custom:
            if verbose:
                logger.debug(f"[Process:{self.name}] Using target_custom => building message list from multiple items.")
            final_input = self._build_message_list_from_custom(db_conn, target_custom, verbose)
        else:
            if verbose:
                logger.debug(f"[Process:{self.name}] Using fallback => single target_input + target_index.")
            final_input = self._build_message_list_from_fallback(db_conn, target_input, target_index, target_fields, verbose)

        # -- At this point, 'final_input' should be a list of message dicts

        # 5) Call the function
        try:
            args = []
            for param_name in self.expected_params:
                if param_name == "manager":
                    args.append(self.manager)
                elif param_name == "messages":
                    args.append(final_input)

            output = self.function(*args)
                
            if not isinstance(output, dict):
                raise ValueError("[Process] function must return a dict.")
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

        if not parsed["multiple_sources"] and not parsed["single_source"]:
            if parsed["component_or_param"]:
                comp_name = parsed["component_or_param"]

                subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_msgs 
                        if r == comp_name
                    ]
                
                chosen = subset[-1:] if subset else [] # last message is default for processes
                message_list = self._transform_to_message_list(chosen)
            else:
                chosen = all_msgs[-1:] if all_msgs else []
                message_list = self._transform_to_message_list(chosen)

        # 3) If multiple_sources => collect each, combine as a message list, sort by msg_number
        elif parsed["multiple_sources"]:
            combined = []
            for source_item in parsed["multiple_sources"]:
                partial = self._collect_msg_snippets(source_item, all_msgs)
                combined.extend(partial)
            # sort by msg_number
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
        """
        index can be None, "~", int, or (start, end)
        """
        if not subset:
            return []

        if index is None:
            return [subset[-1]]

        if index == "~":
            # all
            return subset[:]

        if isinstance(index, int):
            if len(subset) >= abs(index):
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

        # If we don't recognize index => fallback to last
        return [subset[-1]]

    def _transform_to_message_list(self, msg_tuples: List[tuple]) -> List[dict]:
        # Sort by msg_number ascending
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

        # If you do NOT want "msg_number" in the final structure, remove it:
        output = [{"source": m["source"], "message": m["message"], "type": m["type"]} for m in output]

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

        # sort by msg_number
        combined.sort(key=lambda x: x[2])
        return self._transform_to_message_list(combined)

    def _build_message_list_from_fallback(
        self,
        db_conn: sqlite3.Connection,
        target_input: Optional[str],
        target_index: Optional[int], #should this not handle ranges and stuff now?? what else is wrong and needs changing in this codebase after changing the indexing schema?
        target_fields: Optional[list],
        verbose: bool
    ) -> List[dict]:
        all_msgs = self.manager._get_all_messages(db_conn)

        if not all_msgs:
            return []

        if not target_input:
            # no component specified => entire DB => last message
            chosen = [all_msgs[-1]]
        else:
            subset = [
                        (r, c, n, t, ts) 
                        for (r, c, n, t, ts) in all_msgs 
                        if r == target_input
                    ]

            if not subset:
                if verbose:
                    logger.debug(f"[Process:{self.name}] No messages found for target_input={target_input}, returning empty.")
                return []

            if target_index is None:
                chosen = subset[-1]
            else:
                if len(subset) < abs(target_index):
                    if verbose:
                        logger.warning(f"[Process:{self.name}] index={target_index} but only {len(subset)} messages.")
                    return []
                chosen = subset[target_index] # this might be wrong, if processes can take message lists, indices should be allowed to be all, or ranges, just like agents

        if not isinstance(chosen, list):
            chosen = [chosen]

        if target_fields:
            final_chosen = []

            for ch in chosen:
                new_ch = {}
                for field in target_fields:
                    new_ch[field] = ch[field]
                final_chosen.append(new_ch)
        else:
            final_chosen = chosen

        return self._transform_to_message_list(final_chosen)

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
        if verbose:
            logger.debug(f"[Automation:{self.name}] .run() => executing sequence: {self.sequence}")

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
        """
        Execute a single step, which can be a component name (string) or a control flow dictionary.
        """

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
                logger.debug(f"[Automation:{self.name}] => running component '{comp_name}' with "
                    f"target_input={parsed_target_input}, target_index={parsed_target_index}, "
                    f"target_custom={parsed_target_custom}")

            # Run the component

            # Check if the component is an Automation and pass `on_update`
            if isinstance(comp, Automation):
                step_output = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager=None: on_update(messages, manager) if on_update else None,
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
                if on_update_params:
                    on_update(self.manager.get_messages(self.manager._current_user_id), self.manager, on_update_params)
                else:
                    on_update(self.manager.get_messages(self.manager._current_user_id), self.manager)
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

                # Evaluate start_condition; if it's a bool, use that directly; if it's a string/dict, parse it
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

                # Resolve items specification
                items_data = self._resolve_items_spec(items_spec, db_conn, verbose)

                elements = self._generate_elements_from_items(items_data, verbose)
                body = step.get("body", [])

                for idx, element in enumerate(elements):
                    if verbose:
                        logger.debug(f"[Automation:{self.name}] FOR loop iteration {idx+1}/{len(elements)}")
                    
                    # Create iterator message
                    iterator_msg = {
                        "item_number": idx,
                        "item": element
                    }
                    self.manager._save_message(db_conn, "iterator", iterator_msg, "iterator")

                    # Execute loop body
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
                        continue  # Process default last
                        
                    if self._case_matches(switch_value, case_value, verbose):
                        if verbose:
                            logger.debug(f"[Automation:{self.name}] Switch matched case: {case_value}")
                        for nested_step in case_body:
                            current_output = self._execute_step(nested_step, current_output, db_conn, verbose, on_update, on_update_params, return_token_count)
                        executed = True
                        break
                
                # Process default case if no matches
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
            # Direct field reference from last message
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
        # ── NEW: unwrap single-element lists ─────────────────────────────
        if isinstance(switch_value, list) and len(switch_value) == 1:
            switch_value = switch_value[0]
        if isinstance(case_value, list) and len(case_value) == 1:
            case_value = case_value[0]
        # ── NEW: normalise strings (trim + case-fold) ───────────────────
        if isinstance(switch_value, str):
            switch_value = switch_value.strip().lower()
        if isinstance(case_value, str):
            case_value = case_value.strip().lower()
        # ── ORIGINAL numeric check + strict equality ────────────────────
        try:
            if isinstance(switch_value, (int, float)) and isinstance(case_value, (int, float)):
                return float(switch_value) == float(case_value)
            return switch_value == case_value
        except (TypeError, ValueError) as e:
            if verbose:
                logger.error(f"[Automation] Case comparison error: {e}")
            return False

    def _resolve_items_spec(self, items_spec, db_conn, verbose):
        """Resolve items specification to concrete data with parser logic"""
        # Handle numeric cases first
        if isinstance(items_spec, (int, float)):
            return int(items_spec)
        
        if isinstance(items_spec, list) and all(isinstance(x, (int, float)) for x in items_spec):
            return items_spec
        
        # Handle string specifications with parser
        if isinstance(items_spec, str):
            parsed = self.manager.parser.parse_input_string(items_spec)
            resolved_data = self._gather_data_for_parser_result(parsed, db_conn)
            
            # Enforce single message selection
            if isinstance(resolved_data, list):
                if len(resolved_data) > 1:
                    raise ValueError(f"[Automation] FOR loop items spec '{items_spec}' resolved to multiple messages")
                resolved_data = resolved_data[0] if resolved_data else None
            
            if verbose:
                logger.debug(f"[Automation:{self.name}] Resolved items spec '{items_spec}' to: {resolved_data}")
            return resolved_data
        
        return items_spec

    def _generate_elements_from_items(self, items_data, verbose):
        """Convert resolved items data into iterable elements with type-specific handling"""
        if items_data is None:
            return []
        
        # Handle numeric ranges
        if isinstance(items_data, (int, float)):
            return list(range(int(items_data)))
        
        # Handle list-based ranges or generic lists
        if isinstance(items_data, list):
            if len(items_data) in (2, 3) and all(isinstance(x, (int, float)) for x in items_data):
                # Numeric range case
                params = [int(x) for x in items_data]
                return list(range(*params))
            # Generic list case
            return items_data
        
        # Handle dictionary cases
        if isinstance(items_data, dict):
            # If single field, use its value
            if len(items_data) == 1:
                return self._generate_elements_from_items(next(iter(items_data.values())), verbose)
            
            # Convert dict to list of key-value pairs
            return [{"key": k, "value": v} for k, v in items_data.items()]
        
        # Wrap single items in list
        return [items_data]

    def _evaluate_condition(self, condition, current_output) -> bool:

        db_conn = self.manager._get_user_db()

        if isinstance(condition, str):
            parsed = self.manager.parser.parse_input_string(condition)

            if parsed["is_function"] and parsed["function_name"]:
                input_data = self._gather_data_for_parser_result(parsed, db_conn)
                fn = self.manager._get_function_from_string(parsed["function_name"])
                result = fn(input_data)  # must return a boolean
                if not isinstance(result, bool):
                    raise ValueError(
                        f"[Automation:{self.name}] Condition function '{parsed['function_name']}' did not return a bool."
                    )
                return result
            else:
                # Not a function => interpret as a single param or multiple params => must all be True
                data = self._gather_data_for_parser_result(parsed, db_conn)
                if isinstance(data, dict):
                    return all(bool(v) for v in data.values())
                elif isinstance(data, list):
                    return all(bool(item) for item in data)
                else:
                    return bool(data)

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

            # Otherwise => retrieve data, compare to 'value'
            data = self._gather_data_for_parser_result(parsed, db_conn)

            # If data is a single value or list or dict => do direct equality
            return (data == target_value)

        raise ValueError(f"[Automation:{self.name}] Unsupported condition type: {condition}")
    
    def _gather_data_for_parser_result(self, parsed: dict, db_conn: sqlite3.Connection) -> Any:

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp = self.manager._get_component(parsed["component_or_param"])
                if comp:
                    # It's a component => let's get the latest message from that component
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
                    # It's just a param name or string => interpret it as a param from the last message
                    subset = self.manager._get_all_messages(db_conn)
                    role, content, msg_number, msg_type, timestamp = subset[-1]
                    
                    blocks = content                           # ya es List[Block]
                    data   = self.manager._blocks_as_tool_input(blocks)

                    return data[parsed["component_or_param"]]
            else:
                # Nothing
                return None

        # 2) If multiple_sources is not None, we unify them into a single dict
        if parsed["multiple_sources"]:
            final_data = {}
            for source_item in parsed["multiple_sources"]:
                partial_data = self._collect_single_source_data(source_item, db_conn)
                if isinstance(partial_data, dict):
                    final_data.update(partial_data)
                else:
                    # If partial_data isn't a dict, we put it under a key with the component name or "unnamed"
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

        index_to_use = index if index is not None else -1  # default to latest
        if len(subset) < abs(index_to_use):
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

        if parsed["multiple_sources"]:
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


class AgentSystemManager:

    def __init__(
        self,
        config_json: str = None,
        imports: List[str] = None,
        base_directory: str = os.getcwd(),
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
        log_level=None
    ):
        
        if log_level is not None:
            logger.setLevel(log_level)

        self._current_user_id: Optional[str] = None
        self._current_db_conn: Optional[sqlite3.Connection] = None

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
        
        if config_json and config_json.endswith(".json"):
            try:
                self.build_from_json(config_json)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.exception(f"System build failed: {e}")
                raise
        else:
            self.base_directory = base_directory
            self.general_system_description = general_system_description
            self.on_update = self._resolve_callable(on_update)
            self.on_complete = self._resolve_callable(on_complete)
            self._resolve_api_keys_path(api_keys_path)

        try:
            self.timezone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as e:
            logger.warning(f"Invalid timezone {self.timezone}, defaulting to UTC. Error: {e}")
            self.timezone = ZoneInfo("UTC")

        self.api_keys: Dict[str, str] = {}
        self._load_api_keys()
        self._load_costs()

        self._process_imports()

        if hasattr(self, 'pending_links'):
            for agent_name, tool_name in self.pending_links:
                try:
                    self.link(agent_name, tool_name)
                except ValueError as e:
                    logger.error(f"Warning: Could not link agent '{agent_name}' to tool '{tool_name}'. Error: {e}")

        self._last_known_update = None

        # Setup history folder:
        if history_folder is not None:
            if os.path.isabs(history_folder):
                self.history_folder = history_folder
            else:
                self.history_folder = os.path.join(self.base_directory, history_folder)
        else:
            if getattr(self, "history_folder", None) is None:
                self.history_folder = os.path.join(self.base_directory, "history")
        os.makedirs(self.history_folder, exist_ok=True)

        # Setup files folder:
        if files_folder is not None:
            if os.path.isabs(files_folder):
                self.files_folder = files_folder
            else:
                self.files_folder = os.path.join(self.base_directory, files_folder)
        else:
            if getattr(self, "files_folder", None) is None:
                self.files_folder = os.path.join(self.base_directory, "files")
        os.makedirs(self.files_folder, exist_ok=True)

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
            # Expect something like "[comp1, comp2, ...]"
            if comps_part.startswith("[") and comps_part.endswith("]"):
                inside = comps_part[1:-1].strip()  # remove brackets
                if inside:
                    # Split by comma to get component names
                    components = [c.strip() for c in inside.split(",") if c.strip()]
                else:
                    # "?[ ]" was empty: no specific components => None => import all
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

        temp_manager = AgentSystemManager(config_json=resolved_json_path)
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

        # 1) Agents
        for name, agent_obj in other_mgr.agents.items():
            if only_names and name not in only_names:
                continue
            if name not in self.agents:
                agent_obj.manager = self
                self.agents[name] = agent_obj
                self._component_order.append(name)
            # else: skip if it already exists

        # 2) Tools
        for name, tool_obj in other_mgr.tools.items():
            if only_names and name not in only_names:
                continue
            if name not in self.tools:
                tool_obj.manager = self
                self.tools[name] = tool_obj
                self._component_order.append(name)
            # else: skip if it already exists

        # 3) Processes
        for name, proc_obj in other_mgr.processes.items():
            if only_names and name not in only_names:
                continue
            if name not in self.processes:
                proc_obj.manager = self
                self.processes[name] = proc_obj
                self._component_order.append(name)
            # else: skip if it already exists

        # 4) Automations
        for name, auto_obj in other_mgr.automations.items():
            if only_names and name not in only_names:
                continue
            if name not in self.automations:
                auto_obj.manager = self
                self.automations[name] = auto_obj
                self._component_order.append(name)
            # else: skip if it already exists

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
        
    def clear_file_cache(self):
        """
        Clears the in-memory cache of file-based objects.
        Does NOT delete the actual files on disk.
        """
        self._file_cache = {}
        
    def to_string(self) -> str:
        components_details = []

        # Iterate over each type of component and call its `to_string`
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
            f"  Current User ID: {self._current_user_id}\n"
            f"  General System Description: {self.general_system_description}\n\n"
            f"Components:\n"
            f"{components_summary}\n"
            f")"
        )

    def _load_api_keys(self):
        # Initialize api_keys as an empty dictionary. It will only be populated by the JSON file.
        self.api_keys = {}

        if self.api_keys_path is None or not os.path.exists(self.api_keys_path):
            # No file path provided, so we'll rely solely on environment variables later.
            return
        
        if self.api_keys_path.endswith('.json'):
            try:
                with open(self.api_keys_path, "r", encoding="utf-8") as f:
                    # JSON file has the highest priority, load its keys directly.
                    self.api_keys = json.load(f)
            except OSError as e:
                logger.error(f"Error opening API keys file: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {self.api_keys_path}: {e}")

        elif self.api_keys_path.endswith('.env'):
            # .env file loads variables into the environment. 'override=True' gives it
            # priority over system-level environment variables.
            load_dotenv(self.api_keys_path, override=True)
        else:
            raise ValueError(f"Unsupported API keys format: {self.api_keys_path}")
        
    def _load_costs(self):
        """
        Populate self.costs from a JSON file. Silent-fail → empty dict.
        """
        if self.costs_path and os.path.exists(self.costs_path):
            try:
                with open(self.costs_path, "r", encoding="utf-8") as f:
                    self.costs = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read costs file: {e}")
                self.costs = {}
        else:
            # Try <base_dir>/costs.json automatically
            auto = os.path.join(self.base_directory, "costs.json")
            if os.path.exists(auto):
                self.costs_path = auto
                self._load_costs()
            else:
                self.costs = {}

    # ---------------- price lookup helpers --------------
    def _price_for_model(self, provider: str, model: str):
        prov = self.costs.get("models", {}).get(provider.lower(), {})
        mdl  = prov.get(model, {})
        in_p  = mdl.get("input_per_1m", 0.0)
        out_p = mdl.get("output_per_1m", 0.0)
        return in_p / 1_000_000.0, out_p / 1_000_000.0

    def _price_for_tool(self, tool_name: str):
        return self.costs.get("tools", {}).get(tool_name, {}).get("per_call", 0.0)

    def cost_model_call(self, provider, model, in_tokens, out_tokens):
        in_pp, out_pp = self._price_for_model(provider, model)
        return round(in_tokens * in_pp + out_tokens * out_pp, 10)

    def cost_tool_call(self, tool_name):
        return self._price_for_tool(tool_name)

    def _get_db_path_for_user(self, user_id: str) -> str:
        return os.path.join(self.history_folder, f"{user_id}.sqlite")

    def _ensure_user_db(self, user_id: str) -> sqlite3.Connection:
        db_path = self._get_db_path_for_user(user_id)
        needs_init = not os.path.exists(db_path)
        conn = sqlite3.connect(db_path,  check_same_thread=False)
        if needs_init:
            self._create_table(conn)
        return conn
    
    def export_history(self, user_id: str) -> bytes:
        """
        Exports the full sqlite database for the given user_id.
        Returns the file's bytes.
        """
        db_path = self._get_db_path_for_user(user_id)
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                return f.read()
        else:
            # If no DB exists, return empty bytes (or you could raise an error)
            return b""
        
    def import_history(self, user_id: str, sqlite_bytes: bytes):
        """
        Imports a sqlite database for the given user_id.
        It saves the sqlite file into the history folder (using the user_id as filename).
        If a history file already exists, it overwrites it (logging a warning).
        """
        db_path = self._get_db_path_for_user(user_id)
        if os.path.exists(db_path):
            # Log a warning that an existing DB is being overwritten
            logger.warning(f"Overwriting existing history for user {user_id}")
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
        self._current_user_id = user_id
        self._current_db_conn = self._ensure_user_db(user_id)

    def ensure_user(self, user_id: Optional[str] = None) -> None:
        if user_id:
            self.set_current_user(user_id)
        elif not self._current_user_id:
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)

    def _get_user_db(self) -> sqlite3.Connection:
        self.ensure_user()
        return self._current_db_conn

    def _get_next_msg_number(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(msg_number), 0) FROM message_history")
        max_num = cur.fetchone()[0]
        return max_num + 1

    def _save_message(self, conn: sqlite3.Connection, role: str, content: Union[str, dict, list], type = "user", model: Optional[str] = None):
        timestamp = datetime.now(self.timezone).isoformat()

        if isinstance(content, (dict, list)):
            content_str = json.dumps(
                self._persist_non_json_values(content, self._current_user_id),
                indent=2
            )
        else:
            content_str = str(content)

        msg_number = self._get_next_msg_number(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO message_history (msg_number, role, content, type, model, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_number, role, content_str, type, model, timestamp)
        )
        conn.commit()

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
        """
        Recibe la salida cruda del componente, la convierte en bloques y la guarda.
        """
        # Automatizaciones no se persisten (mantengo comportamiento original)
        if isinstance(component, Automation):
            return

        # ---- info de metadatos ────────────────────────────────
        model_info = getattr(component, "_last_used_model", None)
        if model_info:
            del component._last_used_model
            model_str = f"{model_info['provider']}:{model_info['model']}"
        else:
            model_str = None

        if output_value is None:
            return  # nada que guardar

        # ---- normalización a bloques ──────────────────────────
        blocks = self._to_blocks(output_value, self._current_user_id)

        save_role      = component.name
        component_type = (
            "agent"      if isinstance(component, Agent)      else
            "tool"       if isinstance(component, Tool)       else
            "process"    if isinstance(component, Process)    else
            "automation"
        )

        # ---- persistencia final ───────────────────────────────
        self._save_message(
            db_conn,
            save_role,
            blocks,
            component_type,
            model_str
        )

        
    def _dict_to_json_with_file_persistence(self, data: dict, user_id: str) -> str:
        """
        Recursively persist non-JSON-friendly values into files, replacing them with "file:/path/to.pkl".
        Then return the JSON representation of the modified dictionary.
        """
        data_copy = self._persist_non_json_values(data, user_id)
        return json.dumps(data_copy, indent=2)

    def _persist_non_json_values(self, value: Any, user_id: str) -> Any:
        if isinstance(value, dict):
            return {k: self._persist_non_json_values(v, user_id) for k, v in value.items()}

        if isinstance(value, list):
            return [self._persist_non_json_values(item, user_id) for item in value]

        # ⬇️ NUEVO: bytes / bytearray se guardan igual que otros objetos
        if isinstance(value, (bytes, bytearray)):
            return self._store_file(value, user_id)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return self._store_file(value, user_id)

    def _store_file(self, obj: Any, user_id: str) -> str:
        files_dir = os.path.join(self.files_folder, user_id)
        os.makedirs(files_dir, exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(files_dir, f"{file_id}.pkl")

        try:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        except OSError as e:
            logger.error(f"Failed to store file at {file_path}: {e}")
            raise

        self._file_cache[file_path] = obj

        return f"file:{file_path}"

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

        # ⬇️ file:/  -> cargar del disco / caché
        if isinstance(value, str) and value.startswith("file:"):
            return self._load_file(value[5:])

        return value
       
    def _load_file(self, file_path: str) -> Any:
        if file_path in self._file_cache:
            return self._file_cache[file_path]
        
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
        except OSError as e:
            logger.error(f"Failed to load file at {file_path}: {e}")
            raise

        self._file_cache[file_path] = obj
        return obj


    def add_blocks(
        self,
        content,
        *,
        role: str = "user",
        msg_type: str = "user",
        user_id: str | None = None,
        detail: str = "auto",
        verbose: bool = False,
    ) -> int:
        """
        Normaliza *content* a List[Block] con _to_blocks() y lo guarda.
        Devuelve el msg_number asignado.
        """
        blocks = self._to_blocks(content, user_id=user_id, detail=detail)

        if verbose:
            logger.debug(f"[MAS.add_blocks] role={role} type={msg_type} blocks={blocks}")

        self.ensure_user(user_id)
        conn = self._get_user_db()
        msg_number = self._get_next_msg_number(conn)
        self._save_message(conn, role, blocks, msg_type)
        return msg_number
    
    def _filter_blocks_by_fields(self, blocks, fields):
        """
        • Si no hay bloque text ⇒ devuelve blocks sin tocar.
        • Si hay → decodifica JSON, filtra campos, vuelve a serializar.
        """
        import json, copy
        if not fields:
            return blocks

        new_blocks = copy.deepcopy(blocks)
        for b in new_blocks:
            if b.get("type") == "text":
                raw = b["content"]
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    break                                   # no es JSON
                if isinstance(obj, dict):
                    sub = {k: obj[k] for k in fields if k in obj}
                    b["content"] = json.dumps(sub, ensure_ascii=False)
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
        logger.info(f"=== Message History for user [{self._current_user_id}] ===")
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
    
    def get_usage_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a summary of token- and cost-usage for the given user (defaults to current).
        {
        "models": {
            "<provider:model>": {
            "input_tokens": int,
            "output_tokens": int,
            "usd_cost": float
            },
            ...,
            "overall": { same fields summed }
        },
        "tools": {
            "<tool_name>": {
            "calls": int,
            "usd_cost": float
            },
            ...,
            "overall": { calls, usd_cost }
        },
        "overall": { "usd_cost": float }
        }
        """
        messages = self.get_messages(user_id)
        models = {}
        tools  = {}

        for msg in messages:
            mtype = msg["type"]
            src   = msg["source"]
            content = msg["message"] if isinstance(msg["message"], dict) else {}
            metadata = content.get("metadata", {})
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

        # now build totals
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

        # insert the overall buckets
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
        """
        Combine the general_desc + specific_desc, then build a JSON-like structure
        from required_outputs, showing (type) and description.
        """

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
                # If the user provided a dict with "type" and "description"
                field_type = info.get("type", "string")
                field_desc = info.get("description", str(info))
                line = f'  "{field_name}" ({field_type}): "{field_desc}",'
            else:
                # Just a string
                field_desc = info
                line = f'  "{field_name}": "{field_desc}",'
            lines.append(line)
    
        if len(lines) > 3:  # means we have at least 1 field
            last_line = lines[-1]
            if last_line.endswith(","):
                lines[-1] = last_line[:-1]
    
        lines.append("}")
    
        prompt += "\n".join(lines)

        prompt += (
            "In this system, messages are exchanged between different components (users, agents, tools, etc.).\n"
            "To help you understand the origin of each message, they are wrapped in a JSON structure with two fields:\n"
            "- 'source': This indicates the originator of the message (e.g., 'user', 'my_agent', 'my_tool').\n"
            "- 'message': This contains the actual content of the message.\n"
            "When you receive a message, remember that the 'source' tells you who sent it.\n"
            "You do not need to include the 'source' and 'message' wrapper in your output, since they will be added by the system. Focus solely on producing the JSON structure with the required fields defined above.\n\n"
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
        description: str = None
    ):
        # Automatically assign name if not provided
        if name is None:
            name = self._generate_agent_name()
    
        if name in self.agents:
            raise ValueError(f"[Manager] Agent '{name}' already exists.")

        if models is None:
            models = self.default_models.copy()
    
        # Convert string required_outputs to {response: <string>}
        if isinstance(required_outputs, str):
            required_outputs = {"response": required_outputs}
    
        final_prompt = self._build_agent_prompt(
            self.general_system_description,
            name,
            system,
            required_outputs
        )
    
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
            description=description
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
            # Rebuild the entire system prompt using the same builder
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
        """
        Link a tool and an agent based on their relationship:
        - If the first argument is a tool, link the tool as output to the agent.
        - If the first argument is an agent, link the agent as output to the tool.
        """
        is_tool_1 = name1 in self.tools
        is_tool_2 = name2 in self.tools
        is_agent_1 = name1 in self.agents
        is_agent_2 = name2 in self.agents

        if is_tool_1 and is_agent_2:
            # Tool -> Agent
            self.link_tool_to_agent_as_input(name1, name2)
        elif is_agent_1 and is_tool_2:
            # Agent -> Tool
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
        elif not self._current_user_id:
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)

        db_conn = self._get_user_db()

        # Store user-provided input (if any) in the DB
        if input is not None:
            # por defecto consideramos que el emisor es el propio usuario
            store_role = role or "user"

            if store_role == "user":
                self.add_blocks(input, role="user", msg_type="user")
                if verbose:
                    logger.debug("[Manager] Saved user input")
            else:
                # resto de los casos (system, internal, etc.) se guardan igual
                self._save_message(db_conn, store_role, input, store_role)
                logger.debug("[Manager] Saved developer input")

        if component_name is None:
            # Use default or latest automation logic
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
                    on_update=lambda messages, manager, on_update_params: on_update(messages, manager, on_update_params) if on_update else None,
                    return_token_count=return_token_count
                )
            else:
                output_dict = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager: on_update(messages, manager) if on_update else None,
                    return_token_count=return_token_count
                )
        else:
            # Agents, Tools, and Processes accept input targeting arguments
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
                if on_update_params:
                    on_update(self.get_messages(user_id), self, on_update_params)
                else:
                    on_update(self.get_messages(user_id), self)

        self._save_component_output(db_conn, comp, output_dict, verbose=verbose)
        return output_dict


    def run(
        self,
        component_name: Optional[str] = None,
        input: Optional[Any] = None,
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
        """
        - If user_id is given, we switch to that DB. If none, we use/create the current user.
        - The output is also saved to DB as a JSON (with file references if needed).
        """

        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        if return_token_count and blocking:
            _uid = user_id or self._current_user_id
            prev_usage = self.get_usage_stats(_uid)
        else:
            prev_usage = None

        def task():
            self._run_internal(
                component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params, return_token_count
            )
            if on_complete:
                if on_complete_params:
                    on_complete(self.get_messages(user_id), self, on_complete_params)
                else:
                    on_complete(self.get_messages(user_id), self)

        if blocking:
            result = self._run_internal(
                component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params, return_token_count
            )

            if return_token_count and prev_usage is not None:
                _uid = user_id or self._current_user_id
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

                # Log simple
                if verbose:
                    logger.info(
                        f"[Usage] +{delta_in} in / +{delta_out} out tokens  "
                        f"→ ${usage_summary['total_cost']:.6f} "
                        f"(models: ${usage_summary['models_cost']:.6f}, "
                        f"tools: ${usage_summary['tools_cost']:.6f})"
                    )

                # Y lo devolvemos embebido en el dict resultado
                if isinstance(result, dict):
                    result.setdefault("metadata", {})["usage_summary"] = usage_summary

            if on_complete:
                if on_complete_params:
                    on_complete(self.get_messages(user_id), self, on_complete_params)
                else:
                    on_complete(self.get_messages(user_id), self)
            return result
        else:
            thread = threading.Thread(target=task)
            thread.daemon = True  # Ensures thread exits when the main program exits
            thread.start()
            return None

    def save_file(self, obj: Any, user_id = None) -> str:
        # Asegura que haya un usuario activo
        self.ensure_user(user_id)
        # Ahora lo guarda con el mismo user_id
        return self._store_file(obj, self._current_user_id)


    def build_from_json(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                system_definition = json.load(f)
        except OSError as e:
            logger.error(f"Error opening JSON file at {json_path}: {e}")
            system_definition = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file at {json_path}: {e}")
            system_definition = {}

        general_params = system_definition.get("general_parameters", {})

        self.base_directory = general_params.get("base_directory", os.getcwd())
        self.general_system_description = general_params.get("general_system_description", "This is a multi-agent system.")
        self.functions = general_params.get("functions", "fns.py")
        self.default_models = general_params.get("default_models", self.default_models)
        self.on_update = self._resolve_callable(general_params.get("on_update"))
        self.on_complete = self._resolve_callable(general_params.get("on_complete"))
        self.include_timestamp = general_params.get("include_timestamp", False)
        self.timezone = general_params.get("timezone", 'UTC')

        # Resolve history folder from JSON:
        history_folder = general_params.get("history_folder")
        if history_folder:
            if os.path.isabs(history_folder):
                self.history_folder = history_folder
            else:
                self.history_folder = os.path.join(self.base_directory, history_folder)
        else:
            self.history_folder = os.path.join(self.base_directory, "history")

        # Resolve files folder from JSON:
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

        if isinstance(self.functions, str):
            self.functions = [self.functions]
        elif isinstance(self.functions, list):
            self.functions = self.functions
        else:
            self.functions = []
        
        self._resolve_api_keys_path(general_params.get("api_keys_path"))
        self._load_api_keys()

        self.costs_path = general_params.get("costs_path")
        self._load_costs()

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
                description=description
            )

            # Store the link information to be processed later
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

        

        # If there's a colon that is not just "fn:", it's presumably 'file.py:func'
        if ":" in ref and not ref.startswith("fn:"):
            file_part, func_name = ref.rsplit(":", 1)
            file_path = file_part.strip()
            func_name = func_name.strip()
            return self._load_function_from_file(file_path, func_name)

        # Otherwise, handle the "fn:funcname" style
        if ref.startswith("fn:"):
            func_name = ref[3:].strip()
            # Try each file in self.functions in order
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
        """
        Resolve a sequence in an automation, including control flow objects.
        """
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

                # 4) Now we actually process the single import
                self._process_single_import(import_ref)

                # 5) The final resolved step
                resolved_step = f"{component_name}:{target_params}" if target_params else component_name
                resolved_sequence.append(resolved_step)

            elif isinstance(step, str):
                resolved_sequence.append(step)
            elif isinstance(step, dict):
                control_flow_type = step.get("control_flow_type")
                if control_flow_type == "branch":
                    # Validate mandatory fields for branch
                    if "condition" not in step or "if_true" not in step or "if_false" not in step:
                        raise ValueError("Branch must have 'condition', 'if_true', and 'if_false'.")
                    step["condition"] = self._resolve_condition(step["condition"])
                    step["if_true"] = self._resolve_automation_sequence(step.get("if_true", []))
                    step["if_false"] = self._resolve_automation_sequence(step.get("if_false", []))
                elif control_flow_type == "while":
                    # Validate mandatory fields for while
                    if "body" not in step:
                        raise ValueError("While must have 'body'.")

                    if "condition" not in step and "end_condition" not in step:
                        raise ValueError("While must have either 'condition' or 'end_condition'.")
                    step["condition"] = self._resolve_condition(step.get("condition"))
                    step["run_first_pass"] = step.get("run_first_pass", True)
                    step["start_condition"] = self._resolve_condition(step.get("start_condition", step.get("condition", step["run_first_pass"])))
                    step["end_condition"] = self._resolve_condition(step.get("end_condition", step.get("condition")))
                    step["body"] = self._resolve_automation_sequence(step.get("body", []))
                    
                elif control_flow_type == "for":
                    if "items" not in step or "body" not in step:
                        raise ValueError("For must have 'items' and 'body'.")
                    
                    # Resolve items specification using existing parser logic
                    step["items"] = step["items"]
                    step["body"] = self._resolve_automation_sequence(step.get("body", []))

                elif control_flow_type == "switch":
                    # Validate required fields
                    if "value" not in step or "cases" not in step:
                        raise ValueError("Switch must have 'value' and 'cases'")
                    
                    # Validate cases structure
                    for case in step["cases"]:
                        if "case" not in case or "body" not in case:
                            raise ValueError("Each switch case must have 'case' and 'body'")
                    
                    # Resolve remains as-is since processing happens at runtime
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
        """
        Resolve a condition which can be a string, dict, or callable.
        """
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
        """
        Clears the message history for the specified user.
        If no user_id is provided, it clears the history for the current user.
        If no user_id is set or passed, it does nothing.
        """
        if user_id is not None:
            self.set_current_user(user_id)

        if not self._current_user_id:
            logger.warning("No user ID provided or set. Message history not cleared.")
            return

        conn = self._get_user_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM message_history")
        conn.commit()
        logger.info(f"Message history cleared for user ID: {self._current_user_id}")

    def escape_markdown(self, text: str) -> str:
        """
        Escapes special characters in Markdown v1.
        """
        pattern = r'([_*\[\]\(\)~`>#\+\-=|{}\.!\\])'
        return re.sub(pattern, r'\\\1', text)
    
    def _is_block(self, obj):
        """True si obj es un bloque {'type','content'}."""
        return isinstance(obj, dict) and "type" in obj and "content" in obj
    
    def _is_image_bytes(self, val):
        return isinstance(val, (bytes, bytearray))

    def _is_valid_image_path(self, p):
        if not isinstance(p, str):
            return False
        import mimetypes, os
        mime, _ = mimetypes.guess_type(p)
        return mime and mime.startswith("image/") and os.path.isfile(p)

    def _to_blocks(self, value, user_id=None, detail="auto"):
        """
        Convierte *cualquier* valor Python a List[Block] con schema:
            { "type": "text"|"image", "content": … }

        Reglas:
          • Si ya es lista de bloques válidos → la devuelve intacta.
          • Si ya es bloque único  → lo envuelve en lista.
          • bytes / imagen → bloque image.
          • dict / list / str / cualquier otro → bloque text (json-dump si ≠str).
        """
        import json

        # Aseguramos user_id para guardar archivos cuando haga falta
        self.ensure_user(user_id)
        user_id = user_id or self._current_user_id

        # 1) lista de bloques válida
        if isinstance(value, list) and value and all(self._is_block(b) for b in value):
            return value

        # 2) bloque suelto
        if self._is_block(value):
            return [value]

        # 3) bytes  /  ruta de imagen
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

        # 4) dict / list → serializamos si no son ya bloques
        if isinstance(value, dict) or isinstance(value, list):
            try:
                dumped = json.dumps(value, ensure_ascii=False)
            except TypeError:
                dumped = str(value)
            return [{
                "type": "text",
                "content": dumped
            }]

        # 5) todo lo demás (str, int, float, …)
        return [{
            "type": "text",
            "content": str(value)
        }]
    
    def _first_text_block(self, blocks):
        """Devuelve el primer bloque 'text' o None."""
        for b in blocks:
            if b.get("type") == "text":
                return b["content"]
        return None

    def _blocks_as_tool_input(self, blocks):
        """
        • Obtiene el primer bloque text, lo intenta json.loads().
        • Si no hay bloque text → lanza ValueError (Tools no soportan imagen directa).
        """
        import json
        raw = self._first_text_block(blocks)
        if raw is None:
            raise ValueError("Tool input must contain at least one text block")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw
    
    def get_key(self, name: str) -> Optional[str]:
        name = name.lower()

        variations = [
            name,  # Try exact match first
            f"{name}_api_key",
            f"{name}-api-key",
            f"{name}_key",
            f"{name}-key",
            f"{name}key",
        ]
        
        # Priority 1: Check the api_keys.json file content (stored in self.api_keys)
        for var in variations:
            var_lower = var.lower()
            for key_in_file in self.api_keys:
                if key_in_file.lower() == var_lower:
                    return self.api_keys[key_in_file]

        # Priority 2: Check environment variables (which includes .env file and system vars)
        for var in variations:
            value = os.environ.get(var)

            if value:
                return value
            
            # Environment variables are typically uppercase with underscores
            env_var_name = var.upper().replace('-', '_')
            value = os.environ.get(env_var_name)
            if value:
                return value
        
        # If not found in any source
        return None
    
    def start_telegram_bot(self, telegram_token=None, component_name = None, verbose = False,
              on_complete = None, on_update = None, speech_to_text=None,
              whisper_provider=None, whisper_model=None,
              on_start_msg = "Hey! Talk to me or type '/clear' to erase your message history.",
              on_clear_msg = "Message history deleted.",
              return_token_count = False):

        # Get Telegram token from API keys if not provided
        if telegram_token is None:
            telegram_token = self.get_key("telegram_token")
        
        if not telegram_token:
            raise ValueError("Telegram token required. Provide as argument or in API keys as 'TELEGRAM_TOKEN'")

        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        async def process_audio_message(update, context, file_id):
            file = await context.bot.get_file(file_id)
            file_url = file.file_path
            
            # Download the audio file
            response = requests.get(file_url)
            if response.status_code != 200:
                return None, None

            # Store in manager's file system
            file_data = response.content
            file_uuid = str(uuid.uuid4())
            file_name = f"{file_uuid}.ogg"  # Telegram uses OGG for voice messages
            file_path = os.path.join(self.base_directory, "files", str(update.message.chat.id), file_name)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            try:
                with open(file_path, "wb") as f:
                    f.write(file_data)
            except OSError as e:
                logger.error(f"Error writing audio to file {file_path}: {e}")

            # Store in manager's file cache
            file_ref = self._store_file(file_path, str(update.message.chat.id))

            # Transcribe using appropriate method
            transcript = None
            try:
                if callable(speech_to_text):
                    transcript = speech_to_text({"manager": self, "file_path": file_path, "update": update, "context": context})
                else:
                    # Try automatic transcription
                    transcript = self.stt(file_path, whisper_provider, whisper_model)

                    if verbose:
                        logger.debug(f"Audio transcription succeeded: {transcript}")
            except requests.exceptions.RequestException as e:
                if verbose:
                    logger.error(f"HTTP error during transcription: {e}")
                transcript = None
            except ValueError as e:
                if verbose:
                    logger.error(f"Transcription error: {e}")
                transcript = None

            return transcript, file_ref
        
        async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
            audio = update.message.audio or update.message.voice
            transcript, file_ref = await process_audio_message(update, context, audio.file_id)

            # Armamos la lista de bloques que se grabará en el historial
            blocks = []
            if transcript:                                   # bloque de texto si hubo STT
                blocks.extend(self._to_blocks(transcript, user_id=str(update.message.chat.id)))

            blocks.append({                                  # bloque de audio (siempre)
                "type": "audio",
                "source": {"kind": "file", "path": file_ref, "detail": "auto"}
            })

            await handle_system_text(update, blocks, file_ref)

        async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
            # telegram manda varias resoluciones -> agarramos la mayor
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            img_bytes = await file.download_as_bytearray()
            # guardamos en MAS
            user_id = str(update.message.chat.id)
            img_ref = self._store_file(bytes(img_bytes), user_id)

            # Recogemos el texto que el usuario escribió como caption (si lo hubo)
            caption = update.message.caption or ""
    
            # Armamos una lista de bloques: primero el texto (si existe), luego la imagen
            blocks: List[dict] = []
            if caption:
                blocks.extend(self._to_blocks(caption, user_id=str(update.message.chat.id)))

            blocks.append({
                "type": "image",
                "source": {"kind": "file", "path": img_ref, "detail": "auto"}
            })

            await handle_system_text(update, blocks)

        async def send_telegram_response(update, blocks):
            """
            blocks: List[Block] normalizados.
            • text ⇒ envía el campo 'response' si existe, si no el content tal cual.
            • image ⇒ envía la foto referenciada.
            Se envía cada bloque en un mensaje separado.
            """
            import json

            if not blocks:
                return

            for blk in blocks:
                if blk["type"] == "text":
                    raw = blk["content"]
                    try:
                        # intentamos interpretar como JSON por si es objeto
                        parsed = json.loads(raw)
                        # contrato MAS: por defecto los agentes/tools guardan 'response'
                        txt = parsed.get("response", raw) if isinstance(parsed, dict) else raw
                    except json.JSONDecodeError:
                        txt = raw
                    await update.message.reply_text(txt)
                    #value = self.escape_markdown(value)
                    #await update.message.reply_text(value, parse_mode="MarkdownV2")

                elif blk["type"] == "image":
                    await update.message.reply_photo(blk["content"]["path"])

                elif blk["type"] == "audio":
                    await update.message.reply_voice(blk["content"]["path"])

                


        def on_complete_fn(messages, manager, params):

            blocks = None
            if on_complete is not None:
                response = on_complete(messages, manager, params)
                blocks = self._to_blocks(response, user_id=str(params["update"].message.chat.id))
            
            if blocks is None:
                blocks = messages[-1]["message"] if messages else []

            if blocks is None:
                return

            update = params["update"]
            loop = params["event_loop"]

            def callback():
                asyncio.create_task(send_telegram_response(update, blocks))

            loop.call_soon_threadsafe(callback)

        def on_update_fn(messages, manager, params):
            
            blocks = None
            if on_update is not None:
                response = on_update(messages, manager, params)
                blocks = self._to_blocks(response, user_id=str(params["update"].message.chat.id))
            
            if blocks is None:
                return

            update = params["update"]
            loop = params["event_loop"]

            def callback():
                asyncio.create_task(send_telegram_response(update, blocks))

            loop.call_soon_threadsafe(callback)

        # Define async handlers
        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            # Use await for async operations like reply_text
            await update.message.reply_text(on_start_msg)

        async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            self.clear_message_history(update.message.chat.id)
            await update.message.reply_text(on_clear_msg)

        async def handle_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handle direct text input from user"""
            text = update.message.text
            await handle_system_text(update, text)

        async def handle_system_text(update: Update, input, file_ref=None):
            params = {
                "update": update,
                "event_loop": asyncio.get_running_loop(),
                "file_ref": file_ref
            }

            chat_id = update.message.chat.id

            if verbose:
                logger.debug(f"[Manager] Processing input for user {chat_id}: {input}")

            def run_manager_thread():
                self.run(
                    component_name = component_name,
                    input=input, 
                    user_id=chat_id,
                    role= "user",
                    verbose=verbose,
                    blocking=True,
                    on_update=on_update_fn,
                    on_update_params=params,
                    on_complete = on_complete_fn,
                    on_complete_params=params,
                    return_token_count=return_token_count
                )

            thread = threading.Thread(target=run_manager_thread, daemon=True)
            thread.start()
        
        application = Application.builder().token(telegram_token).build()

        # Add command and message handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("clear", clear))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_text))
        application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
        application.add_handler(MessageHandler(filters.VOICE, handle_audio))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

        # Run the bot
        if verbose:
            logger.info("[Manager] Bot running...")

        application.run_polling()

    def stt(self, file_path, provider=None, model=None):
        # Determine which provider to use
        if provider:
            provider = provider.lower()
        else:
            # Auto-detect provider based on available keys
            if self.get_key("groq"):
                provider = "groq"
            elif self.get_key("openai"):
                provider = "openai"
            else:
                raise ValueError("No supported speech-to-text API keys found")

        # Set default model if not specified
        if not model:
            model = {
                "groq": "whisper-large-v3-turbo",
                "openai": "whisper-1"
            }.get(provider)

        # Read audio file
        try:
            with open(file_path, "rb") as audio_file:
                if provider == "groq":
                    return self._transcribe_groq(audio_file, model)
                elif provider == "openai":
                    return self._transcribe_openai(audio_file, model)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
        except OSError as e:
            logger.error(f"Error reading audio file at {file_path}: {e}")

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
            files=files
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API Error: {response.text}")  # Add debug output
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
            files=files
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
        """
        Convert text to speech using the specified provider.
        
        Returns:
            The file path to the saved MP3 audio file.
        """

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
        # Set default model if not provided
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
        # Only include instructions if provided
        if instruction:
            data["instructions"] = instruction

        response = requests.post(url, headers=headers, json=data)
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
        # Set default model if not provided
        if not model:
            model = "eleven_multilingual_v2"
        
        # Retrieve voices to map the voice name to a voice_id
        voices_url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": api_key}
        voices_response = requests.get(voices_url, headers=headers)
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
        response = requests.post(tts_url, headers=headers, json=payload)
        response.raise_for_status()
        audio_bytes = response.content
        return self._save_tts_file(audio_bytes)

    def _save_tts_file(self, audio_bytes: bytes) -> str:
        """
        Helper function to save the audio file (mp3) in the files/tts folder.
        The file is named as <user_id>_<random_hash>.mp3.
        """
        tts_folder = os.path.join(self.files_folder, "tts")
        os.makedirs(tts_folder, exist_ok=True)
        user_id = self._current_user_id if self._current_user_id else "unknown"
        filename = f"{user_id}_{uuid.uuid4().hex}.mp3"
        file_path = os.path.join(tts_folder, filename)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        return file_path

class Parser:
    """
    The result is a dictionary describing:
      {
        "is_function": bool,
        "function_name": Optional[str],
        "component_or_param": Optional[str],
        "multiple_sources": Optional[List[dict]],
          # if multiple sources, each dict describes: {"component":"...", "index": int, "fields": [..]}
        "single_source": Optional[dict],
          # describes one source with {component, index, fields} if relevant
      }
    """

    def parse_input_string(self, spec: str) -> dict:
        spec = spec.strip()
        if not spec:
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": None,
                "single_source": None,
            }

        if ":" in spec and (spec.startswith("fn:") or ".py:" in spec):
            return self._parse_as_function_reference(spec)

        return self._parse_as_non_function(spec)


    def _parse_as_function_reference(self, full_ref: str) -> dict:
        return {
            "is_function": True,
            "function_name": full_ref.strip(),
            "component_or_param": None,
            "multiple_sources": None,
            "single_source": None,
        }


    def _parse_as_non_function(self, spec: str) -> dict:
        if spec.startswith(":"):
            remainder = spec[1:].strip()
            return self._parse_input_sources(remainder)

        if ":" in spec:
            # e.g. "myComponent:someInputSpec"
            parts = spec.split(":", 1)
            component_name = parts[0].strip()
            remainder = parts[1].strip()

            # If remainder starts with "(" => multiple sources
            if remainder.startswith("(") and remainder.endswith(")"):
                multiple_sources = self._parse_multiple_sources(remainder[1:-1].strip())
                return {
                    "is_function": False,
                    "function_name": None,
                    "component_or_param": component_name,
                    "multiple_sources": multiple_sources,
                    "single_source": None,
                }
            else:
                single_source = self._parse_one_custom_item(remainder)
                return {
                    "is_function": False,
                    "function_name": None,
                    "component_or_param": component_name,
                    "multiple_sources": None,
                    "single_source": single_source,
                }

        # If no colon at all => either a single param or single component
        return {
            "is_function": False,
            "function_name": None,
            "component_or_param": spec,
            "multiple_sources": None,
            "single_source": None,
        }


    def _parse_input_sources(self, remainder: str) -> dict:
        remainder = remainder.strip()
        if not remainder:
            # means just ":" was provided
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": None,
                "single_source": None,
            }

        # If it starts with "(" => multiple sources
        if remainder.startswith("(") and remainder.endswith(")"):
            multiple_sources = self._parse_multiple_sources(remainder[1:-1].strip())
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": multiple_sources,
                "single_source": None,
            }
        else:
            single_source = self._parse_one_custom_item(remainder)
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": None,
                "single_source": single_source,
            }


    def _parse_multiple_sources(self, content: str) -> list:
        segments = [seg.strip() for seg in self._split_by_top_level_comma(content)]
        results = []
        for seg in segments:
            results.append(self._parse_one_custom_item(seg))
        return results


    def _split_by_top_level_comma(self, content: str) -> list:
        parts = []
        current = []
        nesting = 0
        for ch in content:
            if ch in ['(', '[']:
                nesting += 1
                current.append(ch)
            elif ch in [')', ']']:
                nesting -= 1
                current.append(ch)
            elif ch == ',' and nesting == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts


    def _parse_one_custom_item(self, item_str: str) -> dict:
        """
        Always produce: {component, index, fields}
        If component is not found, we default to "user"
        """
        item_str = item_str.strip()
        result = {
            "component": None,
            "index": None,
            "fields": None,
        }

        # 1) Does it contain "?[" ? Extract fields first
        fields_part = None
        bracket_start = item_str.find("[")
        bracket_end = item_str.find("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            fields_part = item_str[bracket_start + 1 : bracket_end].strip()
            item_str = item_str[:bracket_start].strip() + item_str[bracket_end+1:].strip()

            if fields_part:
                field_list = [f.strip() for f in fields_part.split(",") if f.strip()]
                if field_list:
                    result["fields"] = field_list

        # Now handle the index part if '?' is present
        if "?" in item_str:
            parts = item_str.split("?")
            maybe_comp = parts[0].strip()
            if maybe_comp:
                result["component"] = maybe_comp
            index_part = parts[1].strip()

            if index_part == "~":
                result["index"] = "~"  # All messages
            elif "~" in index_part:
                range_parts = index_part.split("~")
                if len(range_parts) == 2:
                    start_str = range_parts[0].strip()
                    end_str = range_parts[1].strip()
                    start = int(start_str) if start_str else None
                    end = int(end_str) if end_str else None
                    result["index"] = (start, end)
                elif len(range_parts) == 1:
                    range_str = range_parts[0].strip()
                    try:
                        index_val = int(range_str)
                        # Determine if it's a start or end based on the original string
                        if index_part.startswith("~"):
                            result["index"] = (None, index_val)
                        elif index_part.endswith("~"):
                            result["index"] = (index_val, None)
                    except ValueError:
                        pass
            elif index_part:
                # Single index
                try:
                    idx_val = int(index_part)
                    result["index"] = idx_val
                except ValueError:
                    pass
        elif item_str:
            # If there's no '?', the entire string is the component name
            result["component"] = item_str

        return result
