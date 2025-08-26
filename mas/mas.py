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
from datetime import datetime, timezone, timedelta
import imghdr, mimetypes, shutil
from importlib import resources
import logging
import base64
import collections
from flask import Flask, request, jsonify

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)


MESSAGE_MAX_AGE_SEC = 120 

def _is_too_old(msg_dt, *, max_age=MESSAGE_MAX_AGE_SEC) -> bool:
    """
    Return True iff *msg_dt* (a datetime) is older than *max_age* seconds.
    Works with naïve or TZ-aware UTC datetimes.
    """
    if not isinstance(msg_dt, datetime):
        return False
    if msg_dt.tzinfo is None:
        msg_dt = msg_dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - msg_dt) > timedelta(seconds=max_age)


def get_readme(owner: str, repo: str, branch: str = None,
               token: str = None) -> str:
    """
    Download the README of a GitHub repo and return it as UTF-8 text.
    Uses the GitHub REST API v3 so it always picks the canonical README,
    regardless of file name or branch.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    if branch:
        url += f"?ref={branch}"

    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["content"]).decode("utf-8")

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
            logger.debug(f"[Agent:{self.name}] Final conversation for model input:\n\n{json.dumps(conversation, indent=2, ensure_ascii=False)}\n")

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
                    response_dict, self.manager._active_user_id()
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
        def _clean_path(path_str):
            # QUITA el prefijo "file:" si existe
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
            # dict/list → json, str → tal cual
            if isinstance(content, (dict, list)):
                return json.dumps(content, ensure_ascii=False)
            return str(content)

        # ---- conversión --------------------------------------
        for msg in conversation:
            role   = msg["role"]
            blocks = self._as_blocks(msg["content"])

            if provider in {"openai", "groq", "deepseek", "lmstudio"}:
                # OpenAI / Groq style: user puede ser multimodal, system/assistant solo texto
                if role == "user":
                    parts = []
                    for blk in blocks:
                        if blk["type"] == "text":
                            parts.append({"type": "text", "text": _text_as_string(blk["content"])})
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
                        _text_as_string(blk["content"]) if blk["type"] == "text"
                        else "[image omitted]"
                        for blk in blocks
                    )
                    out.append({"role": role, "content": text})

            elif provider == "google":        # Gemini
                gem_parts = []
                for blk in blocks:
                    if blk["type"] == "text":
                        gem_parts.append({"text": _text_as_string(blk["content"])})
                    elif blk["type"] == "image":
                        src = blk["content"]
                        if src["kind"] == "file":
                            data = open(_clean_path(src["path"]),"rb").read()
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
                        c_blocks.append({"type":"text", "text": _text_as_string(blk["content"])})
                    elif blk["type"] == "image":
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
    
    def _strip_metadata(self, blocks: list) -> list:
        cleaned = []
        for blk in blocks:
            if isinstance(blk, dict):
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

                if self.include_timestamp:
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
        processed_tuples = []
        for (role, content, msg_number, msg_type, timestamp) in msg_tuples:
            # First, parse the JSON string from the DB into a list of blocks.
            try:
                blocks = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, treat it as a single text block
                blocks = [{"type": "text", "content": str(content)}]

            # Now, if fields are specified, filter the parsed blocks.
            if fields:
                blocks = self.manager._filter_blocks_by_fields(blocks, fields)
            
            # Append the processed tuple. The `blocks` variable is now a Python object.
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
            return [{"role": "system", "content": self.system_prompt}]

        if target_index is not None:
            subset = self._handle_index(target_index, subset)
            if not subset: # _handle_index puede devolver una lista vacía si el índice está fuera de rango
                 raise IndexError(f"Requested index/range={target_index} resulted in zero messages from a subset of size {len(filtered_msgs)}.")

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
    
    def _extract_and_parse_json(self, value):
        """
        Parser robusto para contenidos de agentes.
        - Acepta dict/list/str.
        - Quita code fences (``` y ```json).
        - Puede extraer el PRIMER bloque JSON embebido en un string más largo.
        - Desanida el caso 'doble response': {"response": {"response": ...}} o
        {"response": "```json {\"response\": ... } ```"}.
        - Nunca devuelve la cadena 'None' por errores de parseo.

        Retorna:
        * dict/list si se pudo parsear JSON
        * str original (posiblemente sin fences) si NO es JSON
        * None si el input no es str/dict/list
        """
        import json, re

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
            """Intenta una sola capa de json.loads sobre s o s sin fences."""
            if not isinstance(s, str):
                return s
            t = s.strip()
            if not t:
                return s
            # quita comillas completas si viniera entrecomillado
            if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
                t = t[1:-1].strip()
            # intento directo
            try:
                return json.loads(t)
            except Exception:
                pass
            # intento sin fences
            t2 = strip_code_fences(t)
            if t2 != t:
                try:
                    return json.loads(t2)
                except Exception:
                    pass
            return s  # no parsea, devolver string original

        def extract_first_json_lenient(s: str):
            """Intenta json.loads tolerante sobre un fragmento que ya parece JSON."""
            if not isinstance(s, str):
                return None
            t = strip_code_fences(s).strip()
            # intento directo
            try:
                return json.loads(t)
            except Exception:
                pass
            # tolerancia: escapar saltos de línea no escapados
            t2 = re.sub(r'(?<!\\)\n', r'\\n', t)
            try:
                return json.loads(t2)
            except Exception:
                pass
            # tolerancia especial para {"response":"(multilínea)"}
            m = re.match(r'^\s*\{\s*"response"\s*:\s*"(.*)"\s*\}\s*$', t, flags=re.DOTALL)
            if m:
                inner = m.group(1).replace('\\"', '"').replace('\\n', '\n')
                return {"response": inner}
            return None

        def extract_first_json_anywhere(s: str):
            """Busca el primer bloque {...} o [...] balanceado dentro de s y lo parsea."""
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
            # si quedó abierto, intentamos cerrar de forma conservadora
            if depth > 0:
                candidate = t[start:] + closer * depth
                return extract_first_json_lenient(candidate)
            return None

        def flatten_double_response(obj):
            """
            Si obj = {"response": {"response": ...}} => devuelve el interno.
            Si obj = {"response": <string-jsonish>} => parsea y, si es dict con 'response', devuelve ese interno.
            De lo contrario, preserva lo que haya en 'response'.
            """
            if not isinstance(obj, dict) or "response" not in obj:
                return obj

            inner = obj["response"]

            # Caso 1: inner dict con 'response' -> quedarnos con el interno
            if isinstance(inner, dict) and "response" in inner:
                return inner

            # Caso 2: inner dict/list (sin 'response'): mantener tal cual
            if isinstance(inner, (dict, list)):
                obj["response"] = inner
                return obj

            # Caso 3: inner string que parece JSON -> intentar parsear
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

        # ---- lógica principal ----

        # 1) ya es objeto
        if isinstance(value, (dict, list)):
            return flatten_double_response(value) if isinstance(value, dict) else value

        # 2) si no es str/dict/list
        if not isinstance(value, str):
            return None

        s = value.strip()
        # quita fences sueltos de apertura/cierre si vinieran
        s = re.sub(r"^```json\s*|```$", "", s, flags=re.IGNORECASE).strip()

        # 3) intentar desempaquetar una capa (json.loads o sin fences)
        unwrapped = json_unwrap_once(s)

        # a) dict/list -> posible aplanado
        if isinstance(unwrapped, dict):
            return flatten_double_response(unwrapped)
        if isinstance(unwrapped, list):
            return unwrapped

        # b) sigue siendo string: intentar extraer primer JSON embebido
        if isinstance(unwrapped, str):
            extracted = extract_first_json_anywhere(unwrapped)
            if isinstance(extracted, dict):
                return flatten_double_response(extracted)
            if isinstance(extracted, list):
                return extracted
            # No se pudo parsear a objeto: devolver string (¡nunca "None"!)
            return unwrapped

        # c) cualquier otro caso raro
        return None

            
    def _flatten_system_content(self, blocks):
        # puede venir string o lista de bloques
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
            return subset[:]

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
        target_index: Optional[int],
        target_fields: Optional[list],
        verbose: bool
    ) -> List[dict]:
        all_msgs = self.manager._get_all_messages(db_conn)

        if not all_msgs:
            return []

        # 1) filtrado por componente
        subset = (
            [row for row in all_msgs if row[0] == target_input]
            if target_input else all_msgs
        )
        if not subset:
            return []

        # 2) manejamos el índice / rango
        chosen = self._handle_index(target_index, subset)

        # 3) filtramos fields
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
                    on_update(self.manager.get_messages(self.manager._active_user_id()), self.manager, on_update_params)
                else:
                    on_update(self.manager.get_messages(self.manager._active_user_id()), self.manager)
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
        usage_logging: bool = False
    ):

        
        self.base_directory = os.path.abspath(base_directory)
        self._usage_logging_enabled = bool(usage_logging)
        
        description_mode = (
            isinstance(config, str)
            and not config.lower().endswith(".json")
            and not os.path.exists(config)
        )

        if description_mode:
            config = self._bootstrap_from_description(
                description    = config,
                base_directory = base_directory,
                default_models = bootstrap_models,
                verbose = verbose
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
        else:
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

        if self._usage_logging_enabled:
            self.logs_folder = os.path.join(self.base_directory, "logs")
            os.makedirs(self.logs_folder, exist_ok=True)
            self._usage_log_path   = os.path.join(self.logs_folder, "usage.log")
            self._summary_log_path = os.path.join(self.logs_folder, "summary.log")


    def _bootstrap_from_description(self, *, description: str,
                                    base_directory: str,
                                    default_models: List[Dict[str, str]],
                                    verbose: bool = False) -> str:
        import textwrap

        # ---------- 1. README + combined input ------------------------
        readme_text = get_readme("mneuronico", "multi-agent-system-library")

        combined_input = textwrap.dedent(f"""\
            ## MAS Library README
            {readme_text}

            ## USER REQUIREMENT
            {description}

            """)
        
        if default_models is None:
            default_models = [
                {"provider": "openai",     "model": "o3"},
                {"provider": "google",     "model": "gemini-2.5-pro-preview-06-05"},
                {"provider": "deepseek",   "model": "deepseek-reasoner"},
                {"provider": "anthropic",  "model": "claude-sonnet-4-20250514"},
                {"provider": "groq",       "model": "meta-llama/llama-4-maverick-17b-128e-instruct"}
            ]
        elif isinstance(default_models, dict):
            default_models = [default_models]

        bootstrap_cfg = {
            "general_parameters": {
                "base_directory": base_directory,
                "default_models": default_models
            },
            "components": [
                {
                    "type": "agent",
                    "name": "system_writer",
                    "system": (
                        "You are `system_writer`, an expert MAS engineer.\n"
                        "Your job is to read the single user message you will receive "
                        "(it contains the MAS README followed by the user's requirement) "
                        "and OUTPUT **ONE** JSON object with exactly these keys:\n"
                        "  general_parameters → dict that will go verbatim into config.json\n"
                        "  components         → list of MAS components\n"
                        "  fns                → list[str]  each string is full Python "
                        "source code for a helper function (include imports INSIDE the "
                        "function!).\n"
                        "Return NOTHING ELSE, no markdown, no explanations."
                    ),
                    "required_outputs": {
                        "general_parameters": "Block for config.json",
                        "components": "Component list",
                        "fns": "Python functions"
                    }
                }
            ]
        }
        bootstrap_cfg_path = os.path.join(base_directory,
                                          "_bootstrap_config.json")
        with open(bootstrap_cfg_path, "w", encoding="utf-8") as fp:
            json.dump(bootstrap_cfg, fp, indent=2)

        # ---------- 3. run bootstrap ----------------------------------
        sub_mgr = AgentSystemManager(config=bootstrap_cfg_path,
                                     base_directory=base_directory)

        blocks = sub_mgr.run(
            input=combined_input,
            component_name="system_writer",
            verbose = verbose
        )

        # ----------- extract the JSON object from the first text block ---
        agent_payload = None
        for blk in blocks:
            if blk.get("type") == "text":
                raw = blk.get("content")
                if isinstance(raw, dict):
                    agent_payload = raw
                else:
                    try:
                        agent_payload = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        agent_payload = None
                break

        if not isinstance(agent_payload, dict):
            raise RuntimeError("system_writer did not return the expected JSON object")
        
        gp       = agent_payload["general_parameters"]
        comps    = agent_payload["components"]
        fns_list = agent_payload["fns"]

        # ---------- 4. persist final system ---------------------------
        final_cfg_path = os.path.join(base_directory, "config.json")
        with open(final_cfg_path, "w", encoding="utf-8") as fp:
            json.dump({"general_parameters": gp, "components": comps}, fp, indent=2)

        final_fns_path = os.path.join(base_directory, "fns.py")
        with open(final_fns_path, "w", encoding="utf-8") as fp:
            fp.write("\n\n".join(fns_list))

        # ---------- 4-bis. clean bootstrap artefacts ------------------
        # Close the sub-manager’s DB so Windows lets us delete the folder
        try:
            sub_mgr._close_all_db_conns()
        except Exception:
            pass

        # Remove history/ and files/ produced by the bootstrap run
        for folder in ("history", "files"):
            path = os.path.join(base_directory, folder)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)        

        try:
            os.remove(bootstrap_cfg_path)
        except FileNotFoundError:
            pass

        return final_cfg_path

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
        """User-id vigente para *este* hilo (o None)."""
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
            f"  Current User ID: {self._uid()}\n"
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

    def _price_for_stt(self, provider: str, model: str):
        return (
            self.costs.get("stt", {})           # top-level bucket
                    .get(provider.lower(), {})
                    .get(model, {})
                    .get("per_minute", 0.0)    # $ per audio minute
        )
    
    def _log_stt_call(self,
                      provider: str,
                      model: str,
                      seconds: float,
                      transcription: str):
        usd = round(self._price_for_stt(provider, model) *
                    (seconds / 60.0), 10)      # pro-rate by duration

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
        """Append one JSON-line to usage.log and regenerate the summary."""
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
        """Return True if timestamp is inside the given window span."""
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
        """Recompute aggregates and overwrite summary.log."""
        spans = ["minute", "hour", "day", "week", "month", "year", "lifetime"]
        agg   = {s: collections.defaultdict(lambda: {
                    "calls":0, "input_tokens":0, "output_tokens":0,
                    "seconds":0, "usd_cost":0.0}) for s in spans}

        for row in self._iter_usage_rows():
            for span in spans:
                if span == "lifetime" or self._bucket_for(row["ts"], span):
                    key = row["id"]        # provider:model  OR  tool_name  OR  stt_model
                    bucket = agg[span][key]
                    bucket["calls"]         += 1
                    bucket["usd_cost"]      += row.get("cost", 0.0)
                    bucket["input_tokens"]  += row.get("in_toks", 0)
                    bucket["output_tokens"] += row.get("out_toks", 0)
                    bucket["seconds"]       += row.get("seconds", 0)

        # totals & per-user averages
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
        """
        Return an in-memory dict with the same structure that lives in summary.log.
        Optional filters:
            span            one of minute/hour/day/week/month/year/lifetime
            user            restrict to a single user_id
            model_or_tool   restrict to a single id (provider:model, toolname, stt-model)
        """
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
        return os.path.join(self.history_folder, f"{user_id}.sqlite")

    def _ensure_user_db(self, user_id: str) -> sqlite3.Connection:
        if not hasattr(self, "_db_pool"):
            self._db_pool = {}
        user_id = str(user_id)
        if user_id in self._db_pool:
            return self._db_pool[user_id]
        db_path = self._get_db_path_for_user(user_id)
        needs_init = not os.path.exists(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Mejor concurrencia en SQLite
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        if needs_init:
            self._create_table(conn)
        self._db_pool[user_id] = conn
        return conn
    
    def _active_user_id(self) -> Optional[str]:
        if hasattr(self, "_tls") and getattr(self._tls, "current_user_id", None):
            return self._tls.current_user_id
        return None

    
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
        user_id = str(user_id)

        if not hasattr(self, "_tls"):
            self._tls = threading.local()

        # Si el hilo ya estaba usando ese mismo usuario, nada que hacer
        if getattr(self._tls, "current_user_id", None) == user_id:
            return

        # Conexión sqlite específica de ese usuario
        self._tls.current_user_id = user_id
        self._tls.db_conn         = self._ensure_user_db(user_id)


    def ensure_user(self, user_id: Optional[str] = None) -> None:
        if user_id:
            self.set_current_user(user_id)
            return
        # Si este hilo no tiene usuario aún, crea uno
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

        cur = conn.cursor()
        # Bloquea para evitar carreras con MAX(msg_number)
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
            output_value,
            component_type,
            model_str
        )

        meta = self._extract_metadata_from_blocks(output_value)
        if not meta:
            return                                      # nada que loggear

        entry = {
            "user":   self._active_user_id(),
            "type":   component_type,                   # "agent" | "tool" | …
            "id":     model_str or save_role,           # p.e.  "openai:gpt-4o"
            "in_toks":  meta.get("input_tokens", 0),
            "out_toks": meta.get("output_tokens", 0),
            "seconds": meta.get("seconds", 0),          # STT deja esto en 0
            "cost":    meta.get("usd_cost", 0.0)
        }
        self._log_if_enabled(entry)
        
    def _dict_to_json_with_file_persistence(self, data: dict, user_id: str) -> str:
        """
        Recursively persist non-JSON-friendly values into files, replacing them with "file:/path/to.pkl".
        Then return the JSON representation of the modified dictionary.
        """
        data_copy = self._persist_non_json_values(data, user_id)
        return json.dumps(data_copy, indent=2, ensure_ascii=False)

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
    
    

    def _detect_extension(self, data: bytes, fallback=".bin") -> str:
        """Devuelve '.jpg' | '.png' | '.gif'… si data es imagen, 
        '.ogg' | '.mp3' si es audio (cabecera “ID3”…), etc.
        Si no reconoce, usa fallback."""
        
        ext = imghdr.what(None, h=data)
        if ext:
            return f".{ext}"
        # audio muy simple → ID3, OggS…
        if data[:3] == b"ID3":
            return ".mp3"
        if data[:4] == b"OggS":
            return ".ogg"
        return fallback

    def _store_file(self, obj: Any, user_id: str) -> str:
        user_id = str(user_id)
        files_dir = os.path.join(self.files_folder, user_id)
        os.makedirs(files_dir, exist_ok=True)

        # ── 1. si es ruta a un archivo existente … ──────────────────────
        if isinstance(obj, str) and os.path.isfile(obj):
            ext = os.path.splitext(obj)[1] or ".bin"
            new_path = os.path.join(files_dir, f"{uuid.uuid4()}{ext}")
            shutil.copy2(obj, new_path)
            self._file_cache[new_path] = obj          # opcional: apuntar a path original
            return f"file:{new_path}"

        # ── 2. si es bytes / bytearray … ────────────────────────────────
        if isinstance(obj, (bytes, bytearray)):
            ext = self._detect_extension(obj)
            new_path = os.path.join(files_dir, f"{uuid.uuid4()}{ext}")
            with open(new_path, "wb") as f:
                f.write(obj)
            self._file_cache[new_path] = bytes(obj)
            return f"file:{new_path}"

        # ── 3. cualquier otro objeto → pickle (comportamiento previo) ───
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

        # ⬇️ file:/  -> cargar del disco / caché
        if isinstance(value, str) and value.startswith("file:"):
            return self._load_file(value[5:])

        return value
       
    def _load_file(self, file_path: str) -> Any:
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pkl":                       #  ← solo los .pkl se des-pickle-an
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
        else:                                   #  imágenes / audio → bytes
            with open(file_path, "rb") as f:
                obj = f.read()

        self._file_cache[file_path] = obj
        return obj


    def add_blocks(
        self,
        content,
        *,
        role: str = "user",
        msg_type: str = "user",
        user_id: Union[str, None] = None
    ) -> int:
        """
        Normaliza *content* a List[Block] con _to_blocks() y lo guarda.
        Devuelve el msg_number asignado (atómico).
        """
        self.ensure_user(user_id)
        conn = self._get_user_db()
        msg_number = self._save_message(conn, role, content, msg_type)
        return msg_number

    
    def _filter_blocks_by_fields(self, blocks, fields):
        import copy
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
    
    # ---------- helper: saca metadata de una lista de bloques --------------
    def _extract_metadata_from_blocks(self, content):
        """
        Devuelve el dict metadata que guarda MAS dentro del primer bloque con ella.
        Si no hay, devuelve {}.
        """

        if isinstance(content, dict):
            return content.get("metadata", {})

        if isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                # 1) bloque con metadata explícita
                if "metadata" in blk:
                    return blk["metadata"]

                # 2) bloque type=text con JSON serializado
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
        elif not self._uid():
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
        """
        - If user_id is given, we switch to that DB. If none, we use/create the current user.
        - The output is also saved to DB as a JSON (with file references if needed).
        """

        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        run_user_id = user_id or self._uid()
        if not run_user_id:
            # Si todavía no hay usuario, créalo y establécelo ahora.
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
                if on_complete_params:
                    on_complete(self.get_messages(user_id), self, on_complete_params)
                else:
                    on_complete(self.get_messages(user_id), self)

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

                # Log simple
                if verbose:
                    logger.info(
                        f"[Usage] +{delta_in} in / +{delta_out} out tokens  "
                        f"→ ${usage_summary['total_cost']:.6f} "
                        f"(models: ${usage_summary['models_cost']:.6f}, "
                        f"tools: ${usage_summary['tools_cost']:.6f})"
                    )

                # Y lo devolvemos embebido en el dict resultado
                if final_blocks and isinstance(final_blocks, list):
                    final_blocks[0].setdefault("metadata", {})
                    final_blocks[0]["metadata"]["usage_summary"] = usage_summary

            if on_complete:
                if on_complete_params:
                    on_complete(self.get_messages(user_id), self, on_complete_params)
                else:
                    on_complete(self.get_messages(user_id), self)
            return final_blocks
        else:
            thread = threading.Thread(target=task)
            thread.daemon = True  # Ensures thread exits when the main program exits
            thread.start()
            return None

    def save_file(self, obj: Any, user_id = None) -> str:
        # Usa el user_id explícito o el del hilo actual
        uid = str(user_id) if user_id is not None else str(self._active_user_id() or "unknown")
        self.ensure_user(uid)
        return self._store_file(obj, uid)


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

        if not self._uid():
            logger.warning("No user ID provided or set. Message history not cleared.")
            return

        conn = self._get_user_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM message_history")
        conn.commit()
        logger.info(f"Message history cleared for user ID: {self._uid()}")

    def clear_global_history(self) -> int:
        """
        Borra el historial de TODOS los usuarios (deja los archivos .sqlite y sus tablas vacías).
        No hace chequeos de admin. Pensado para ser llamado por el developer.
        Retorna la cantidad de bases tocadas.
        """
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
                # Garantiza que la tabla exista
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
                # Limpia
                cur.execute("DELETE FROM message_history")
                conn.commit()
                conn.close()
                cleared += 1
            except Exception:
                logger.exception(f"Error clearing history for DB: {db_path}")
        logger.info(f"[Manager] Cleared history in {cleared} sqlite DB(s).")
        return cleared

    def reset_system(self) -> None:
        """
        Resetea el sistema:
        - Cierra conexiones a DBs
        - Elimina TODAS las bases sqlite (history/) y TODOS los archivos en files/
        - Recrea carpetas vacías y limpia caches internas
        NO hace chequeos de admin. Pensado para el developer.
        """
        self._close_all_db_conns()
        # Borra history
        try:
            shutil.rmtree(self.history_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove history folder.")
        # Borra files
        try:
            shutil.rmtree(self.files_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove files folder.")
        # Borra logs (si existen)
        try:
            if hasattr(self, "logs_folder") and self.logs_folder:
                shutil.rmtree(self.logs_folder, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove logs folder.")

        # Recrea carpetas
        os.makedirs(self.history_folder, exist_ok=True)
        os.makedirs(self.files_folder, exist_ok=True)
        if hasattr(self, "logs_folder") and self.logs_folder:
            os.makedirs(self.logs_folder, exist_ok=True)
            # reestablece rutas (por si cambió logs_folder)
            self._usage_log_path   = os.path.join(self.logs_folder, "usage.log")
            self._summary_log_path = os.path.join(self.logs_folder, "summary.log")
            # limpia contador interno
            if hasattr(self, "_lines_since_refresh"):
                self._lines_since_refresh = 0
        # Limpia caches internas
        if hasattr(self, "_db_pool"):
            self._db_pool.clear()
        self._file_cache = {}
        logger.info("[Manager] System reset completed (history/, files/, logs/ wiped).")


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
        # Aseguramos user_id para guardar archivos cuando haga falta
        self.ensure_user(user_id)
        user_id = user_id or self._uid()

        # 1) lista de bloques válida
        if isinstance(value, list) and value and all(self._is_block(b) for b in value):
            return value
        
        # 1-bis) ----- lista *mixta* → procesar elemento por elemento ----
        if isinstance(value, list):
            merged = []
            for item in value:
                merged.extend(self._to_blocks(item, user_id=user_id, detail=detail))
            return merged

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

        # 4) dict → bloque text con el objeto en content
        if isinstance(value, dict):
            data_copy = self._persist_non_json_values(value, user_id)
            return [{
                "type": "text",
                "content": data_copy
            }]
            # try:
            #     dumped = json.dumps(value, ensure_ascii=False)
            # except TypeError:
            #     # Fallback: persist non-serializable values into files
            #     if isinstance(value, dict):
            #         dumped = self._dict_to_json_with_file_persistence(value, user_id)
            #     else:
            #         data_copy = self._persist_non_json_values(value, user_id)
            #         dumped = json.dumps(data_copy, ensure_ascii=False)
            # return [{
            #     "type": "text",
            #     "content": dumped
            # }]

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

        if isinstance(blocks, str):
            try:
                maybe = json.loads(blocks)
                # sólo aceptamos si parece lista de bloques
                if isinstance(maybe, list) and all(isinstance(b, dict) for b in maybe):
                    blocks = maybe
                else:
                    # string plano → lo tratamos como un solo bloque de texto
                    blocks = [{"type": "text", "content": blocks}]
            except json.JSONDecodeError:
                # string plano → idem
                blocks = [{"type": "text", "content": blocks}]

        # ── 2)  Validación mínima ──────────────────────────────────────────
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
            # texto plano → envolvemos
            return {"text_content": raw}


        # cualquier otro tipo lo envolvemos igual
        return {"text_content": str(raw)}
    
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
    
    def _json_unwrap(self, value, max_depth: int = 3):
        """
        Intenta aplicar json.loads repetidamente para convertir:
        '"Hola\\nMundo"'         -> 'Hola\nMundo'
        '"{\\"response\\":...}"' -> '{"response":...}' -> {...}
        Si deja de ser str o falla, devuelve el valor actual.
        """
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
        """
        Convierte un bloque a texto para transporte:
        - Prioriza 'response' si el content es dict y existe.
        - Dict/List → JSON pretty.
        - Str → tal cual (desempaquetando niveles de JSON si hiciera falta).
        """
        import json
        if blk.get("type") != "text":
            return ""
        raw = blk.get("content")

        # Si ya es dict/list
        if isinstance(raw, dict):
            if "response" in raw and isinstance(raw["response"], str):
                return raw["response"]
            return json.dumps(raw, ensure_ascii=False, indent=2)
        if isinstance(raw, list):
            return json.dumps(raw, ensure_ascii=False, indent=2)

        # String → intenta "desempacar" niveles JSON
        val = self._json_unwrap(raw)
        if isinstance(val, dict):
            if "response" in val and isinstance(val["response"], str):
                return val["response"]
            return json.dumps(val, ensure_ascii=False, indent=2)
        if isinstance(val, list):
            return json.dumps(val, ensure_ascii=False, indent=2)
        return str(val)
    
    def start_telegram_bot(self, telegram_token=None, component_name = None, verbose = False,
              on_complete = None, on_update = None, speech_to_text=None,
              whisper_provider=None, whisper_model=None,
              on_start_msg = "Hey! Talk to me or type '/clear' to erase your message history.",
              on_clear_msg = "Message history deleted.",
              return_token_count = False, start_polling = True):
        
        logger = logging.getLogger(__name__)

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

            blocks = []
            if transcript:
                blocks.extend(self._to_blocks(transcript, user_id=str(update.message.chat.id)))

            blocks.append({
                "type": "audio",
                "content": {"kind": "file", "path": file_ref, "detail": "auto"}
            })

            await handle_system_text(update, blocks, file_ref)

        async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            img_bytes = await file.download_as_bytearray()
            user_id = str(update.message.chat.id)
            img_ref = self._store_file(bytes(img_bytes), user_id)
            caption = update.message.caption or ""
    
            blocks: List[dict] = []
            if caption:
                blocks.extend(self._to_blocks(caption, user_id=str(update.message.chat.id)))

            blocks.append({
                "type": "image",
                "content": {"kind": "file", "path": img_ref, "detail": "auto"}
            })

            await handle_system_text(update, blocks)

        async def send_telegram_response(update, blocks):
            if not blocks:
                return

            for blk in blocks:
                if blk["type"] == "text":
                    txt = self._block_to_plain_text(blk)
                    if txt:
                        await update.message.reply_text(txt)
                elif blk["type"] == "image":
                    path = blk["content"]["path"]
                    if path.startswith("file:"):
                        local_path = path[5:]
                        with open(local_path, "rb") as fp:
                            await update.message.reply_photo(fp)
                    else:
                        await update.message.reply_photo(path)
                elif blk["type"] == "audio":
                    path = blk["content"]["path"]
                    if path.startswith("file:"):
                        local_path = path[5:]
                        with open(local_path, "rb") as fp:
                            await update.message.reply_voice(fp)
                    else:
                        await update.message.reply_voice(path)

        def on_complete_fn(messages, manager, params):

            blocks = None
            if on_complete:
                response = on_complete(messages, manager, params)
                if response is not None:
                    blocks = self._to_blocks(response, user_id=str(params["update"].message.chat.id))
            else:
                blocks = messages[-1]["message"] if messages else None

            if blocks is None:
                return

            update = params["update"]
            loop = params["event_loop"]

            def callback(_upd=update, _blks=blocks):
                asyncio.create_task(send_telegram_response(_upd, _blks))
            loop.call_soon_threadsafe(callback)

        def on_update_fn(messages, manager, params):
            blocks = None
            if on_update:
                response = on_update(messages, manager, params)
                if response is not None:
                    blocks = self._to_blocks(response, user_id=str(params["update"].message.chat.id))

            if not blocks:
                return

            update = params["update"]
            loop = params["event_loop"]

            def callback(_upd=update, _blks=blocks):
                asyncio.create_task(send_telegram_response(_upd, _blks))

            loop.call_soon_threadsafe(callback)

        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(on_start_msg)

        async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            self.clear_message_history(update.message.chat.id)
            await update.message.reply_text(on_clear_msg)

        async def clear_all_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not self.admin_user_id:
                await update.message.reply_text("Comando deshabilitado: no hay admin configurado.")
                return
            if str(update.message.chat.id) != str(self.admin_user_id):
                await update.message.reply_text("No autorizado.")
                return
            count = self.clear_global_history()
            await update.message.reply_text(f"Historiales de todos los usuarios borrados ({count} DBs).")

        async def reset_system_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not self.admin_user_id:
                await update.message.reply_text("Comando deshabilitado: no hay admin configurado.")
                return
            if str(update.message.chat.id) != str(self.admin_user_id):
                await update.message.reply_text("No autorizado.")
                return
            try:
                self.reset_system()
                await update.message.reply_text("Sistema reseteado: se eliminaron todos los historiales y archivos, y se recrearon las carpetas vacías.")
            except Exception as e:
                logger.exception("Error en /reset_system:", exc_info=e)
                await update.message.reply_text("Ocurrió un error al resetear el sistema.")

        async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            # Solo admin
            if not self.admin_user_id:
                await update.message.reply_text("Comando deshabilitado: no hay admin configurado.")
                return
            if str(update.message.chat.id) != str(self.admin_user_id):
                print(update.message.chat.id, self.admin_user_id)
                await update.message.reply_text("No autorizado.")
                return

            logs_dir = getattr(self, "logs_folder", None)
            if not logs_dir:
                await update.message.reply_text("usage_logging está deshabilitado.")
                return

            usage_p = os.path.join(logs_dir, "usage.log")
            summary_p = os.path.join(logs_dir, "summary.log")

            # Refresca summary por las dudas
            if getattr(self, "_usage_logging_enabled", False):
                try:
                    self._refresh_cost_summary()
                except Exception:
                    pass

            sent_any = False
            if os.path.isfile(usage_p):
                with open(usage_p, "rb") as fp:
                    await update.message.reply_document(fp, filename="usage.log")
                sent_any = True
            if os.path.isfile(summary_p):
                with open(summary_p, "rb") as fp:
                    # lo mandamos como summary.json para que el cliente lo trate como JSON
                    await update.message.reply_document(fp, filename="summary.json")
                sent_any = True

            if not sent_any:
                await update.message.reply_text("No hay logs para enviar.")



        async def handle_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
            text = update.message.text
            await handle_system_text(update, text)

        async def handle_system_text(update: Update, input, file_ref=None):
            if _is_too_old(update.message.date):
                logging.getLogger(__name__).debug("[Whatsapp Bot] Message ignored because it is too old.")
                return
            
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
                    input=input,
                    component_name = component_name,
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
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("clear", clear))
        application.add_handler(CommandHandler("clear_all_users", clear_all_users))
        application.add_handler(CommandHandler("reset_system", reset_system_cmd))
        application.add_handler(CommandHandler("logs", logs_cmd))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_text))
        application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
        application.add_handler(MessageHandler(filters.VOICE, handle_audio))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

        if verbose:
            logger.info("[Manager] Telegram application ready.")

        if start_polling:
            if verbose:
                logger.info("[Manager] Bot running...")
            application.run_polling()
        
        return application

    # ──────────────────────────────────────────────────────────────────────
    #   WhatsApp Cloud-API integration  (drop into AgentSystemManager)
    # ──────────────────────────────────────────────────────────────────────
    def start_whatsapp_bot(
        self,
        *,
        # WhatsApp Cloud credentials  ─────────────────────────────────────
        whatsapp_token: str = None,
        phone_number_id: str = None,
        webhook_verify_token: str = None,
        # MAS run-options  ────────────────────────────────────────────────
        component_name: str = None,
        verbose: bool = False,
        on_update: Callable = None,
        on_complete: Callable = None,
        return_token_count: bool = False,
        # multimodal helpers
        speech_to_text: Callable = None,
        whisper_provider: str = None,
        whisper_model: str = None,
        # command replies (mirror Telegram defaults)  ─────────────────────
        on_start_msg: str = "Hey! Talk to me or type '/clear' to erase your message history.",
        on_clear_msg: str = "Message history deleted.",
        # server opts
        host: str = "0.0.0.0",
        port: int = 5000,
        base_path: str = "/webhook",
    ):
        """
        Launch a tiny Flask server that bridges WhatsApp Cloud API  ⇄  MAS.

        * multimodal: images, audio (with optional STT), plain text
        * same callback semantics & defaults as start_telegram_bot()
        * supports '/start' and '/clear' commands
        """

        # ──────────────────── credentials / helper vars ──────────────────
        whatsapp_token       = whatsapp_token       or self.get_key("whatsapp_token")
        phone_number_id      = phone_number_id      or self.get_key("whatsapp_phone_number_id")
        webhook_verify_token = webhook_verify_token or self.get_key("webhook_verify_token")

        if not (whatsapp_token and phone_number_id and webhook_verify_token):
            raise ValueError(
                "Missing WhatsApp credentials. Provide WHATSAPP_ACCESS_TOKEN, "
                "WHATSAPP_PHONE_NUMBER_ID and WEBHOOK_VERIFY_TOKEN."
            )
        
        
        log = logging.getLogger(__name__)
        log.debug("[Whatsapp Bot] WA-bot starting – phone_id=%s  base_path=%s", phone_number_id, base_path)

        GRAPH_MSG_URL = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
        HDR_JSON  = {"Authorization": f"Bearer {whatsapp_token}",
                    "Content-Type":  "application/json"}
        HDR_AUTH  = {"Authorization": f"Bearer {whatsapp_token}"}

        # choose callbacks from system defaults if not overridden
        on_update   = on_update   or self.on_update
        on_complete = on_complete or self.on_complete

        

        # ─────────────────── outbound helpers (send to WA) ───────────────
        def _send_text(to: str, txt: str):
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": txt}
            }
            r = requests.post(GRAPH_MSG_URL, headers=HDR_JSON, json=payload, timeout=60)
            log.debug("[Whatsapp Bot] → send %s status=%s to=%s", "text", r.status_code, to)

        def _upload_media(path: str, mime: str) -> str:
            url = f"https://graph.facebook.com/v18.0/{phone_number_id}/media"
            with open(path, "rb") as fh:
                files = {"file": (os.path.basename(path), fh, mime)}
                r = requests.post(url, headers=HDR_AUTH,
                                data={"messaging_product": "whatsapp"},
                                files=files, timeout=60)
            if r.ok:
                return r.json().get("id")
            log.error("Media upload failed: %s", r.text)
            return None

        def _send_image(to: str, path: str, caption: str = None):
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            mid  = _upload_media(path, mime)
            if not mid:
                return
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "image",
                "image": {"id": mid, **({"caption": caption} if caption else {})}
            }
            r = requests.post(GRAPH_MSG_URL, headers=HDR_JSON, json=payload, timeout=60)
            log.debug("[Whatsapp Bot] → send %s status=%s to=%s", "image", r.status_code, to)

        def _send_audio(to: str, path: str):
            mime = "audio/mpeg"
            mid  = _upload_media(path, mime)
            if not mid:
                return
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "audio",
                "audio": {"id": mid}
            }
            r = requests.post(GRAPH_MSG_URL, headers=HDR_JSON, json=payload, timeout=60)
            log.debug("[Whatsapp Bot] → send %s status=%s to=%s", "audio", r.status_code, to)

        def _send_document(to: str, path: str, filename: Optional[str] = None, caption: Optional[str] = None):
            import os, mimetypes
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if not os.path.isfile(path):
                log.error("[Whatsapp Bot] document not found: %s", path)
                return
            mid  = _upload_media(path, mime)
            if not mid:
                log.error("[Whatsapp Bot] media upload failed for: %s", path)
                return
            doc = {"id": mid}
            if filename:
                doc["filename"] = filename
            if caption:
                doc["caption"] = caption
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "document",
                "document": doc
            }
            r = requests.post(GRAPH_MSG_URL, headers=HDR_JSON, json=payload, timeout=60)
            log.debug("[Whatsapp Bot] → send %s status=%s to=%s", "document", r.status_code, to)


        # ───────────── MAS blocks → WhatsApp messages ────────────────────
        def blocks_to_whatsapp(to: str, blocks: list[dict]):
            for blk in blocks:
                if blk["type"] == "text":
                    txt = self._block_to_plain_text(blk)
                    if txt:
                        _send_text(to, txt)

                    # raw = blk["content"]
                    # try:
                    #     obj = json.loads(raw) if isinstance(raw, str) else raw
                    #     txt = obj.get("response", raw) if isinstance(obj, dict) else raw
                    # except json.JSONDecodeError:
                    #     txt = raw
                    # _send_text(to, txt)

                elif blk["type"] == "image":
                    src = blk["content"]
                    if src.get("kind") == "file":
                        local = src["path"][5:] if src["path"].startswith("file:") else src["path"]
                        _send_image(to, local)
                    elif src.get("kind") == "url":
                        _send_text(to, src["url"])          # fallback: just send url
                elif blk["type"] == "audio":
                    src = blk["content"]
                    if src.get("kind") == "file":
                        local = src["path"][5:] if src["path"].startswith("file:") else src["path"]
                        _send_audio(to, local)

        # ───────────────── Flask webhook server ──────────────────────────
        app = Flask(__name__)

        @app.route(base_path, methods=["GET"])
        def verify():
            if request.args.get("hub.verify_token") == webhook_verify_token:
                return request.args.get("hub.challenge", ""), 200
            return "Forbidden", 403

        @app.route(base_path, methods=["POST"])
        def inbox():
            data = request.get_json(force=True, silent=True) or {}
            #log.debug("[Whatsapp Bot] WEBHOOK RAW ⇢ %s ...", json.dumps(data)[:800])
            try:
                for entry in data.get("entry", []):
                    for change in entry.get("changes", []):
                        for msg in change.get("value", {}).get("messages", []):
                            threading.Thread(target=_handle_msg, args=(msg,), daemon=True).start()
            except Exception:
                log.exception("Error processing webhook payload")
            return "OK", 200

        # ───────────────── message dispatcher ────────────────────────────
        def _handle_msg(msg: dict):

            wa_id  = msg.get("from")
            mtype  = msg.get("type")
            blocks = []

            log.debug("[Whatsapp Bot] HANDLE %s id=%s from=%s", mtype, msg.get('id'), wa_id)

            # ⏰ timeout check
            try:
                msg_epoch = int(msg.get("timestamp", "0"))
            except ValueError:
                msg_epoch = 0
            msg_dt = datetime.fromtimestamp(msg_epoch, timezone.utc)
            if _is_too_old(msg_dt):
                log.debug("[Whatsapp Bot] Message ignored because it is too old.")
                return
            

            # ── text / commands ───────────────────────────────────────────
            if mtype == "text":
                body = msg["text"]["body"].strip()
                if body.lower() == "/start":
                    _send_text(wa_id, on_start_msg)
                    return
                if body.lower() == "/clear":
                    self.clear_message_history(wa_id)
                    _send_text(wa_id, on_clear_msg)
                    return
                if body.lower() == "/clear_all_users":
                    if not self.admin_user_id:
                        _send_text(wa_id, "Comando deshabilitado: no hay admin configurado.")
                        return
                    if str(wa_id) != str(self.admin_user_id):
                        _send_text(wa_id, "No autorizado.")
                        return
                    count = self.clear_global_history()
                    _send_text(wa_id, f"Historiales de todos los usuarios borrados ({count} DBs).")
                    return
                if body.lower() == "/reset_system":
                    if not self.admin_user_id:
                        _send_text(wa_id, "Comando deshabilitado: no hay admin configurado.")
                        return
                    if str(wa_id) != str(self.admin_user_id):
                        _send_text(wa_id, "No autorizado.")
                        return
                    try:
                        self.reset_system()
                        _send_text(wa_id, "Sistema reseteado: se eliminaron todos los historiales y archivos, y se recrearon las carpetas vacías.")
                    except Exception as e:
                        logger.exception("Error en /reset_system:", exc_info=e)
                        _send_text(wa_id, "Ocurrió un error al resetear el sistema.")
                    return
                if body.lower() == "/logs":
                    if not self.admin_user_id:
                        _send_text(wa_id, "Comando deshabilitado: no hay admin configurado.")
                        return
                    if str(wa_id) != str(self.admin_user_id):
                        _send_text(wa_id, "No autorizado.")
                        return

                    logs_dir = getattr(self, "logs_folder", None)
                    if not logs_dir:
                        _send_text(wa_id, "usage_logging está deshabilitado.")
                        return

                    usage_p = os.path.join(logs_dir, "usage.log")
                    summary_p = os.path.join(logs_dir, "summary.log")

                    # Refresca summary si corresponde
                    if getattr(self, "_usage_logging_enabled", False):
                        try:
                            self._refresh_cost_summary()
                        except Exception:
                            pass

                    sent = 0
                    if os.path.isfile(usage_p):
                        _send_document(wa_id, usage_p, filename="usage.log")
                        sent += 1
                    if os.path.isfile(summary_p):
                        _send_document(wa_id, summary_p, filename="summary.json")
                        sent += 1

                    if sent == 0:
                        _send_text(wa_id, "No hay logs para enviar.")
                    else:
                        _send_text(wa_id, f"Enviados {sent} archivo(s) de logs.")
                    return


                blocks = self._to_blocks(body, user_id=wa_id)

            # ── images ───────────────────────────────────────────────────
            elif mtype == "image":
                media_id = msg["image"]["id"]
                media_url = requests.get(
                    f"https://graph.facebook.com/v18.0/{media_id}",
                    headers=HDR_AUTH, timeout=60).json().get("url")
                img_bytes = requests.get(media_url, headers=HDR_AUTH, timeout=60).content
                img_ref   = self.save_file(img_bytes, wa_id)
                caption   = msg["image"].get("caption", "")
                if caption:
                    blocks.extend(self._to_blocks(caption, user_id=wa_id))
                blocks.append({"type": "image",
                            "content": {"kind": "file", "path": img_ref, "detail": "auto"}})

            # ── audio / voice ─────────────────────────────────────────────
            elif mtype in ("audio", "voice"):
                media_id = msg[mtype]["id"]
                media_url = requests.get(
                    f"https://graph.facebook.com/v18.0/{media_id}",
                    headers=HDR_AUTH, timeout=60).json().get("url")
                audio_bytes = requests.get(media_url, headers=HDR_AUTH, timeout=60).content
                audio_ref   = self.save_file(audio_bytes, wa_id)

                transcript = None
                if callable(speech_to_text):
                    transcript = speech_to_text({"manager": self, "file_path": audio_ref[5:]})
                else:
                    try:
                        transcript = self.stt(audio_ref, whisper_provider, whisper_model)
                    except Exception as e:
                        log.error("STT failed: %s", e)

                if transcript:
                    blocks.extend(self._to_blocks(transcript, user_id=wa_id))
                blocks.append({"type": "audio",
                            "content": {"kind": "file", "path": audio_ref, "detail": "auto"}})

            else:
                _send_text(wa_id, "Unsupported message type.")
                return

            # ── runner callbacks ─────────────────────────────────────────

            def _cb_update(messages, manager, params=None, _wa_id=wa_id):
                if on_update:
                    custom = on_update(messages, manager, params)
                    if custom is not None:
                        blocks_to_whatsapp(_wa_id, manager._to_blocks(custom, user_id=_wa_id))

            def _cb_complete(messages, manager, params=None, _wa_id=wa_id):
                if on_complete:
                    custom = on_complete(messages, manager, params)
                    if custom is not None:
                        blocks_to_whatsapp(_wa_id, manager._to_blocks(custom, user_id=_wa_id))
                        return
                elif messages:
                    blocks_to_whatsapp(_wa_id, messages[-1]["message"])

            # ── run MAS (thread keeps webhook snappy) ─────────────────────
            def _run():
                log.debug("[Whatsapp Bot] MAS RUN user=%s  blocks=%s", wa_id, blocks[:1])
                self.run(
                    input=blocks,
                    component_name=component_name,
                    user_id=wa_id,
                    role="user",
                    verbose=verbose,
                    blocking=True,
                    on_update=_cb_update,
                    on_complete=_cb_complete,
                    return_token_count=return_token_count
                )
            threading.Thread(target=_run, daemon=True).start()

        # ────────────────────────── launch ───────────────────────────────
        if verbose:
            log.info("[Whatsapp Bot] WhatsApp bot up at http://%s:%d%s", host, port, base_path)
        app.run(host=host, port=port, threaded=True)


    def stt(self, file_path, provider=None, model=None):
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
            """
            Return length in seconds for *any* common audio file.
            Strategy:
            1) .wav/.wave → use Python's wave module (no deps)
            2) If FFmpeg/ffprobe present → use it (handles mp3, ogg, m4a…)
            3) If mutagen installed → use it
            4) Give up → None  (caller decides fallback price)

            Works with absolute paths or our internal 'file:...' refs.
            """
            import os, subprocess, json

            if path.startswith("file:"):
                path = path[5:]

            if not os.path.isfile(path):
                return None

            # --- 1 · plain WAV via stdlib ------------------------------------
            ext = os.path.splitext(path)[1].lower()
            if ext in (".wav", ".wave"):
                import wave
                with wave.open(path, "rb") as w:
                    return w.getnframes() / w.getframerate()

            # --- 2 · ffprobe if available -----------------------------------
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

            # --- 3 · mutagen fallback (pure-python) -------------------------
            try:
                from mutagen import File as mutagen_file
                m = mutagen_file(path)
                if m is not None and m.info.length:
                    return float(m.info.length)
            except Exception:
                pass

            # couldn't figure it out
            return None


        seconds = _audio_duration_sec(file_path) or 60.0   # flat 1-min if unknown

        # 3) call the real HTTP endpoint
        with open(file_path, "rb") as audio_file:
            if provider == "groq":
                transcript = self._transcribe_groq(audio_file, model)
            elif provider == "openai":
                transcript = self._transcribe_openai(audio_file, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        # 4) log the cost so get_usage_stats() can see it
        self._log_stt_call(provider, model, seconds, transcript)

        rate_pm = self._price_for_stt(provider, model)
        cost    = round(rate_pm * (seconds / 60.0), 6)

        self._log_if_enabled({
            "user":  self._active_user_id(),
            "type":  "stt",
            "id":    f"{provider}:{model}",
            "seconds": seconds,
            "in_toks": 0,
            "out_toks": 0,
            "cost":  cost
        })

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
        Helper to save MP3 in files/tts as <user_id>_<random>.mp3, thread-safe.
        """
        tts_folder = os.path.join(self.files_folder, "tts")
        os.makedirs(tts_folder, exist_ok=True)
        uid = str(self._active_user_id() or self._uid() or "unknown")
        filename = f"{uid}_{uuid.uuid4().hex}.mp3"
        file_path = os.path.join(tts_folder, filename)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        return file_path
    
    def _close_all_db_conns(self):
        """Cierra todas las conexiones SQLite abiertas en el pool interno."""
        if hasattr(self, "_db_pool"):
            for _uid, conn in list(self._db_pool.items()):
                try:
                    conn.close()
                except Exception:
                    pass
            self._db_pool.clear()


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
