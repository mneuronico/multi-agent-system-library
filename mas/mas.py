import os
import json
import sqlite3
import traceback
import uuid
import threading
from typing import Optional, List, Dict, Callable, Any, Union
from openai import OpenAI
from groq import Groq
from pydantic import BaseModel
import pickle
import importlib.util
import google.generativeai as genai
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import re

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
        api_keys: Optional[Dict[str, str]] = None,
        general_system_description: str = ""
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
        self.api_keys = api_keys or {}
        self.general_system_description = general_system_description

    def to_string(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"System Prompt: {self.system_prompt}\n"
            f"Models: {self.models}\n"
            f"Default Output: {self.default_output}\n"
            f"Positive Filter: {self.positive_filter}\n"
            f"Negative Filter: {self.negative_filter}\n"
        )

    def run(
        self,
        input_data: Any = None,  # Not used, but kept for interface consistency
        target_input: Optional[str] = None,
        target_index: Optional[int] = None,
        target_fields: Optional[list] = None,
        target_custom: Optional[list] = None,
        verbose: bool = False
    ) -> dict:

        if verbose:
            print(f"[Agent:{self.name}] .run() => reading conversation history from DB")

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
            print(f"[Agent:{self.name}] Final conversation for model input:\n\n{json.dumps(conversation, indent=2)}\n")

        # Now we call each model in self.models in order:
        for model_info in self.models:
            provider = model_info["provider"].lower()
            model_name = model_info["model"]
            api_key = self.api_keys.get(provider)

            if not api_key:
                if verbose:
                    print(f"[Agent:{self.name}] No API key for '{provider}'. Skipping.")
                continue

            try:
                if provider == "openai":
                    response_str = self._call_openai_api(
                        model_name=model_name,
                        conversation=conversation,
                        api_key=api_key,
                        verbose=verbose
                    )
                elif provider == "deepseek":
                    response_str = self._call_deepseek_api(
                        model_name=model_name,
                        conversation=conversation,
                        api_key=api_key,
                        verbose=verbose
                    )
                elif provider == "google":
                    response_str = self._call_google_api(
                        model_name=model_name,
                        conversation=conversation,
                        api_key=api_key,
                        verbose=verbose
                    )
                elif provider == "groq":
                    response_str = self._call_groq_api(
                        model_name=model_name,
                        conversation=conversation,
                        api_key=api_key,
                        verbose=verbose
                    )
                else:
                    raise ValueError(f"[Agent:{self.name}] Unknown provider '{provider}'")

                response_dict = self._extract_and_parse_json(response_str)
                self._last_used_model = {"provider": provider, "model": model_name}
                if verbose:
                    print(f"[Agent:{self.name}] => success from provider={provider}\n{response_dict}")
                return response_dict

            except Exception as e:
                if verbose:
                    print(f"[Agent:{self.name}] => failed with provider={provider}, model={model_name}, error={e}")

        if verbose:
            print(f"[Agent:{self.name}] => All providers failed. Returning default:\n{self.default_output}")
        return self.default_output


    def _build_conversation_from_parser_result(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> List[Dict[str, Any]]:
        if verbose:
            print(f"[Agent:{self.name}] _build_conversation_from_parser_result => parsed={parsed}")

        all_msgs = self.manager._get_all_messages(db_conn)
        filtered_msgs = self._apply_filters(all_msgs)

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp_name = parsed["component_or_param"]
                chosen = [
                    (r, c, n, t) 
                    for (r, c, n, t) in filtered_msgs 
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
        """
        Return a list of (role, content, msg_number, msg_type) from the relevant subset, honoring index if specified.
        """
        comp_name = source_item["component"]
        index = source_item["index"]

        if comp_name:

            subset = [
                (r, c, n, t) 
                for (r, c, n, t) in filtered_msgs 
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
        """
        Convert a list of (role, content, msg_number, msg_type) into conversation format.
        Wrap messages from other roles with {"source": <role>, "message": <content>} except for the agent's own messages.
        """
        conversation = []
        for (role, content, msg_number, msg_type) in msg_tuples:
            try:
                data = json.loads(content)
            except:
                data = content

            if isinstance(data, dict) and fields:
                data = self._extract_fields_from_data(data, fields)

            if isinstance(data, dict):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            
            if role == self.name:
                conversation.append({"role": "assistant", "content": data_str, "msg_number": msg_number})
            else:
                conversation.append({"role": "user", "content": f'{{"source": "{msg_type}: {role}", "message": "{data_str}"}}', "msg_number": msg_number})

        # Sort by msg_number to preserve chronological order
        conversation.sort(key=lambda e: e["msg_number"])

        if not include_message_number:
            conversation = [{"role": e["role"], "content": e["content"]} for e in conversation]
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
                    (r, c, n, t) 
                    for (r, c, n, t) in filtered_msgs 
                    if r == comp_name
                ]

                if not subset:
                    if verbose:
                        print(f"[Agent:{self.name}] No messages found for component/user '{comp_name}'. Skipping.")
                    continue

                subset = self._handle_index(index, subset)
                subset = self._handle_fields(subset, fields)
                
                conversation.extend(subset)

            conversation = self._transform_to_conversation(conversation, fields)
            
            conversation = [{"role": "system", "content": self.system_prompt}] + conversation
            return conversation
    
    def _handle_fields(self, msg_tuples, fields):
        conversation = []

        for (role, content, msg_number, msg_type) in msg_tuples:
            try:
                data = json.loads(content)
            except:
                data = content

            if isinstance(data, dict) and fields:
                data = self._extract_fields_from_data(data, fields)

            conversation.append((role, data, msg_number, msg_type))

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
                (r, c, n, t) 
                for (r, c, n, t) in filtered_msgs 
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


    def _extract_fields_from_data(self, data: dict, fields: list) -> dict:
        """
        Attempt to extract only the specified fields from data.
        Returns a new dict containing only those fields.
        """
        relevant = {}
        for f in fields:
            if f in data:
                relevant[f] = data[f]
        return relevant


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
        for (role, content, msg_number, msg_type) in messages:
            if self.positive_filter:
                if not any(matches_filter(role, pf, msg_type) for pf in self.positive_filter):
                    continue
            if self.negative_filter:
                if any(matches_filter(role, nf, msg_type) for nf in self.negative_filter):
                    continue
            filtered.append((role, content, msg_number, msg_type))

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

    def _call_openai_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool
    ) -> str:
        schema_for_response = self._build_json_schema()

        # Transform system => developer (required for openAI)
        new_messages = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                new_messages.append({"role": "developer", "content": content})
            else:
                new_messages.append({"role": role, "content": content})

        if verbose:
            print(f"[Agent:{self.name}] _call_openai_api => model={model_name}")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=new_messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema_for_response
            }
        )
        return response.choices[0].message.content
    
    def _call_deepseek_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool
    ) -> str:
        """
        For DeepSeek (which works with OpenAI API as well)
        """

        base_url = "https://api.deepseek.com"

        # Transform system => developer (required for openAI)
        new_messages = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                new_messages.append({"role": "system", "content": content})
            else:
                new_messages.append({"role": role, "content": content})

        if verbose:
            print(f"[Agent:{self.name}] _call_deepseek_api => model={model_name}")

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=new_messages#,
            #response_format={'type': 'json_object'} # this sometimes fails, according to deepseek docs
        )
        return response.choices[0].message.content
    
    def _call_google_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool
    ) -> str:
        """
        Calls the Google Gemini API to get a response.
        """
        if verbose:
            print(f"[Agent:{self.name}] _call_google_api => model={model_name}")
        
        genai.configure(api_key=api_key)

        # Separate system instructions and build history (excluding last user message)
        system_instruction = None
        history = []
        last_message_content = None

        for i, msg in enumerate(conversation):
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": msg["content"]})
            elif msg["role"] == "user":
                if i == len(conversation) - 1:
                    last_message_content = msg["content"]
                else:
                    history.append({"role": "user", "parts": msg["content"]})

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None
        )

        chat = model.start_chat(history=history)
        
        # Check if a last user message exists for sending
        if last_message_content:
            response = chat.send_message(last_message_content)
        else:
            if verbose:
                print(f"[Agent:{self.name}] No last user message for Google Gemini API.")
            return json.dumps(self.default_output)

        if verbose:
            print(f"[Agent:{self.name}] _call_google_api response => {response.text}")

        # Google returns text directly, we need to wrap in a json
        return response.text

    def _call_groq_api(
        self,
        model_name: str,
        conversation: List[Dict[str, Any]],
        api_key: str,
        verbose: bool
    ) -> str:
        if verbose:
            print(f"[Agent:{self.name}] _call_groq_api => model={model_name}")

        groq_client = Groq(api_key=api_key)

        chat_completion = groq_client.chat.completions.create(
            messages=conversation,
            model=model_name,
            stream=False,
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content


class Tool(Component):

    def __init__(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        function: Callable,
        default_output: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name)
        self.inputs = inputs
        self.outputs = outputs
        self.function = function
        self.default_output = default_output or {}

    def to_string(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}\n"
            f"Default Output: {self.default_output}\n"
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
            print(
                f"[Tool:{self.name}] .run() => target_input={target_input}, "
                f"target_index={target_index}, target_fields={target_fields}, target_custom={target_custom}, input_data={input_data}"
            )

        db_conn = self.manager._get_user_db()

        if isinstance(input_data, dict):
            if verbose:
                print(f"[Tool:{self.name}] Using provided input_data directly.")
            input_dict = input_data
        else:
            # If we have a `target_input` string that might hold advanced syntax => parse
            if target_input and any(x in target_input for x in [":", "fn?","fn:", "?"]):
                parsed = self.manager.parser.parse_input_string(target_input)
                # Now collect data from DB
                input_dict = self._gather_data_for_tool_process(parsed, db_conn, verbose)
            else:
                input_dict = self._resolve_tool_or_process_input(db_conn, target_input, target_index, target_fields, target_custom, verbose)

        # Now validate that input_dict has the keys for self.inputs
        func_args = []
        for arg_name in self.inputs:
            if arg_name not in input_dict:
                raise ValueError(f"[Tool:{self.name}] Missing required input: {arg_name}")
            func_args.append(input_dict[arg_name])

        # Call the tool function
        try:
            result = self.function(*func_args)
            if not isinstance(result, (list, tuple)):
                result = (result,)

            if len(result) != len(self.outputs):
                raise ValueError(
                    f"[Tool:{self.name}] function returned {len(result)} items, "
                    f"but {len(self.outputs)} expected."
                )

            result_dict = {}
            for (out_name, val) in zip(self.outputs, result):
                result_dict[out_name] = val

            if verbose:
                print(f"[Tool:{self.name}] => {result_dict}")
            return result_dict

        except Exception as e:
            if verbose:
                print(f"[Tool:{self.name}] => error {e}")
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
                (r, c, n, t) 
                for (r, c, n, t) in all_messages 
                if r == target_input
            ]

        if not subset:
            if verbose:
                print(f"[Tool:{self.name}] No messages found for target_input={target_input}, returning empty.")
            return {}

        if target_index is None:
            # default to latest for tools
            role, content, msg_number, msg_type = subset[-1]
        else:
            if len(subset) < abs(target_index):
                raise IndexError(f"Requested index={target_index} but only {len(subset)} messages found.")
            role, content, msg_number, msg_type = subset[target_index]

        try:
            data = json.loads(content)
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
                print(f"[Tool:{self.name}] Failed to parse content as JSON: {content}")
            return {}

    def _gather_data_for_tool_process(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> dict:
        """
        Tools/Processes must produce a single dictionary of input params from the parse result.
        If multiple sources are specified, we merge them. If there's a single source, we filter fields if needed.
        """

        if verbose:
            print(f"[Tool:{self.name}] _gather_data_for_tool_process => parsed={parsed}")

        if parsed["multiple_sources"] is None and parsed["single_source"] is None:
            if parsed["component_or_param"]:
                comp = self.manager._get_component(parsed["component_or_param"])
                if comp:
                    
                    messages = self.manager._get_all_messages(db_conn)
                    subset = [
                        (r, c, n, t) 
                        for (r, c, n, t) in messages 
                        if r == parsed['component_or_param']
                    ]


                    if subset:
                        role, content, msg_number, msg_type = subset[-1]
                        try:
                            data = json.loads(content)
                            data = self.manager._load_files_in_dict(data)
                            if isinstance(data, dict):
                                return data
                            return {}
                        except:
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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_messages 
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

        role, content, msg_number, msg_type = subset[index]
        try:
            data = json.loads(content)
            data = self.manager._load_files_in_dict(data)
        except:
            data = {}

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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_messages 
                        if r == comp_name
                    ]

            if not subset:
                if verbose:
                    print(f"[Tool/Process:{self.name}] No messages found for '{comp_name}'. Skipping.")
                continue

            if index is None:
                chosen = [subset[-1]]
            elif isinstance(index, int):
                if len(subset) < abs(index):
                    raise IndexError(f"Requested index={index} but only {len(subset)} messages found for '{comp_name}'.")
                chosen = [subset[index]]
            else:
                raise IndexError(f"Index={index} must be an integer for Tools.")
                
            for (role, content, msg_number, msg_type) in chosen:
                try:
                    data = json.loads(content)
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
                except:
                    if verbose:
                        print(f"[Tool/Process:{self.name}] Error parsing message for '{comp_name}'.")
                    continue

        return final_input


class Process(Component):

    def __init__(self, name: str, function: Callable):
        super().__init__(name)
        self.function = function

    def to_string(self) -> str:
        return f"Name: {self.name}"

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
            print(f"[Process:{self.name}] .run() => target_input={target_input}, "
                  f"target_index={target_index}, target_fields={target_fields}, target_custom={target_custom}, input_data={input_data}")

        db_conn = self.manager._get_user_db()

        # 1) If user explicitly provided a dict (or list) in 'input_data', pass that directly
        if isinstance(input_data, dict) or isinstance(input_data, list):
            if verbose:
                print(f"[Process:{self.name}] Using user-provided input_data directly.")
            final_input = input_data

        # 2) If advanced parser syntax is in target_input, build the message list from that
        elif target_input and any(x in target_input for x in [":", "fn?", "fn:", "?"]):
            parsed = self.manager.parser.parse_input_string(target_input)
            if verbose:
                print(f"[Process:{self.name}] Detected advanced parser usage => parsed={parsed}")
            final_input = self._build_message_list_from_parser_result(parsed, db_conn, verbose)
        elif target_custom:
            if verbose:
                print(f"[Process:{self.name}] Using target_custom => building message list from multiple items.")
            final_input = self._build_message_list_from_custom(db_conn, target_custom, verbose)
        else:
            if verbose:
                print(f"[Process:{self.name}] Using fallback => single target_input + target_index.")
            final_input = self._build_message_list_from_fallback(db_conn, target_input, target_index, target_fields, verbose)

        # -- At this point, 'final_input' should be a list of message dicts

        # 5) Call the function
        try:
            output = self.function(final_input)
            if not isinstance(output, dict):
                raise ValueError("[Process] function must return a dict.")
            if verbose:
                print(f"[Process:{self.name}] => {output}")
            return output
        except Exception as e:
            if verbose:
                print(f"[Process:{self.name}] => error {e}")
                traceback.print_exc()
            return {}


    def _build_message_list_from_parser_result(self, parsed: dict, db_conn: sqlite3.Connection, verbose: bool) -> List[dict]:
        if verbose:
            print(f"[Process:{self.name}] _build_message_list_from_parser_result => parsed={parsed}")

        all_msgs = self.manager._get_all_messages(db_conn)

        if not parsed["multiple_sources"] and not parsed["single_source"]:
            if parsed["component_or_param"]:
                comp_name = parsed["component_or_param"]

                subset = [
                        (r, c, n, t) 
                        for (r, c, n, t) in all_msgs 
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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_msgs 
                        if r == comp_name
                    ]
        else:
            subset = None

        if not subset:
            return []

        chosen = self._handle_index(index, subset)

        final = []
        for (role, content, msg_num, msg_type) in chosen:
            data = self._safe_json_load(content)
            if isinstance(data, dict) and fields:
                data = {f: data[f] for f in fields if f in data}
            final.append((role, data, msg_num, msg_type))

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
        for (role, data, msg_num, msg_type) in sorted_msgs:
            data = self.manager._load_files_in_dict(data)

            output.append({
                "source": role,
                "message": data,
                "msg_number": msg_num,
                "type": msg_type
            })

        # If you do NOT want "msg_number" in the final structure, remove it:
        output = [{"source": m["source"], "message": m["message"], "type": m["type"]} for m in output]

        return output

    def _safe_json_load(self, content: str) -> Any:
        try:
            data = json.loads(content)
            return self.manager._load_files_in_dict(data)
        except:
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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_msgs 
                        if r == comp_name
                    ]

            if not subset:
                if verbose:
                    print(f"[Process:{self.name}] No messages found for '{comp_name}'. Skipping.")
                continue

            chosen = self._handle_index(idx, subset)

            final = []
            for (role, content, msg_num, msg_type) in chosen:
                data = self._safe_json_load(content)
                if isinstance(data, dict) and fields:
                    data = {f: data[f] for f in fields if f in data}
                final.append((role, data, msg_num, msg_type))

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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_msgs 
                        if r == target_input
                    ]

            if not subset:
                if verbose:
                    print(f"[Process:{self.name}] No messages found for target_input={target_input}, returning empty.")
                return []

            if target_index is None:
                chosen = subset[-1]
            else:
                if len(subset) < abs(target_index):
                    if verbose:
                        print(f"[Process:{self.name}] index={target_index} but only {len(subset)} messages.")
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
    def __init__(self, name: str, sequence: List[Union[str, dict]]):
        super().__init__(name)
        self.sequence = sequence

    def to_string(self) -> str:
        return f"Name: {self.name}\nSequence: {self.sequence}\n"

    def run(self, verbose: bool = False, on_update: Optional[Callable] = None, on_update_params: Optional[Dict] = None) -> Dict:
        if verbose:
            print(f"[Automation:{self.name}] .run() => executing sequence: {self.sequence}")

        db_conn = self.manager._get_user_db()
        current_output = {}

        for step in self.sequence:
            current_output = self._execute_step(step, current_output, db_conn, verbose, on_update, on_update_params)

        if verbose:
            print(f"[Automation:{self.name}] => Execution completed.")
        return current_output

    def _execute_step(self, step, current_output, db_conn, verbose, on_update = None, on_update_params = None):
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
                    print(f"[Automation:{self.name}] => no such component '{comp_name}'. Skipping.")
                return current_output

            if verbose:
                print(f"[Automation:{self.name}] => running component '{comp_name}' with "
                    f"target_input={parsed_target_input}, target_index={parsed_target_index}, "
                    f"target_custom={parsed_target_custom}")

            # Run the component

            # Check if the component is an Automation and pass `on_update`
            if isinstance(comp, Automation):
                step_output = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager=None: on_update(messages, manager) if on_update else None  # Pass on_update callback to nested automation
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
                    print(f"[Automation:{self.name}] => branching condition evaluated to {condition_met}.")

                next_steps = step["if_true"] if condition_met else step["if_false"]
                for branch_step in next_steps:
                    current_output = self._execute_step(branch_step, current_output, db_conn, verbose, on_update, on_update_params)

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
                            print(f"[Automation:{self.name}] => executing while loop body.")
                        for nested_step in body:
                            current_output = self._execute_step(nested_step, current_output, db_conn, verbose, on_update, on_update_params)

                        if (isinstance(end_condition, bool) and end_condition) or \
                        (isinstance(end_condition, (str, dict)) and self._evaluate_condition(end_condition, current_output)):
                            break
                        if verbose:
                            print(f"[Automation:{self.name}] => while loop iteration complete.")

            elif control_flow_type == "for":
                items_spec = step.get("items")
                if verbose:
                    print(f"[Automation:{self.name}] Processing FOR loop with items: {items_spec}")

                # Resolve items specification
                items_data = self._resolve_items_spec(items_spec, db_conn, verbose)

                elements = self._generate_elements_from_items(items_data, verbose)
                body = step.get("body", [])

                for idx, element in enumerate(elements):
                    if verbose:
                        print(f"[Automation:{self.name}] FOR loop iteration {idx+1}/{len(elements)}")
                    
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
                            verbose, on_update, on_update_params
                        )


            else:
                raise ValueError(f"Unsupported control flow type: {control_flow_type}")

        return current_output

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
                print(f"[Automation:{self.name}] Resolved items spec '{items_spec}' to: {resolved_data}")
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
        """
        Evaluate a condition, which can be:
        - a string ("fn:function_name:..."), or just "someParam", or a colon-based syntax for input sources
        - a dict {"input": "...", "value": <literal or list>}
            If "input" starts with "fn:", then we call that function (must return bool).
            Otherwise, we retrieve that input from the DB and compare to 'value'.
        """

        db_conn = self.manager._get_user_db()

        if isinstance(condition, str):
            parsed = self.manager.parser.parse_input_string(condition)

            if parsed["is_function"] and parsed["function_name"]:
                input_data = self._gather_data_for_parser_result(parsed, db_conn)
                fn = self._get_function(parsed["function_name"])
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
                fn = self._get_function(parsed["function_name"])
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
                        (r, c, n, t) 
                        for (r, c, n, t) in messages 
                        if r == parsed["component_or_param"]
                    ]

                    if filtered:
                        role, content, msg_number, msg_type = filtered[-1]
                        try:
                            data = json.loads(content)
                            data = self.manager._load_files_in_dict(data)
                            return data
                        except:
                            return content
                    else:
                        return None
                else:
                    # It's just a param name or string => interpret it as a param from the last message
                    subset = self.manager._get_all_messages(db_conn)
                    role, content, msg_number, msg_type = subset[-1]
                    data = json.loads(content)

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
                        (r, c, n, t) 
                        for (r, c, n, t) in all_messages 
                        if r == comp_name
                    ]
        else:
            subset = None

        if not subset:
            return None

        index_to_use = index if index is not None else -1  # default to latest
        if len(subset) < abs(index_to_use):
            return None

        role, content, msg_number, msg_type = subset[index_to_use]

        try:
            data = json.loads(content)
            data = self.manager._load_files_in_dict(data)
        except:
            data = content

        if not isinstance(data, dict):
            return data

        if fields:
            filtered_data = {}
            for f in fields:
                if f in data:
                    filtered_data[f] = data[f]
            return filtered_data
        else:
            # No field filter => return entire dict
            return data


    def _get_function(self, function_name: str) -> Callable:
        """
        Retrieve a function by name from the manager's loaded fns module or other registry.
        """
        fns_path = os.path.join(self.manager.base_directory, self.manager.functions_file)
        if not os.path.exists(fns_path):
            raise FileNotFoundError(f"File '{self.manager.functions_file}' not found in base directory: {self.manager.base_directory}")

        spec = importlib.util.spec_from_file_location("fns", fns_path)
        fns_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fns_module)

        if not hasattr(fns_module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in 'fns.py'.")
        return getattr(fns_module, function_name)


        
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
        base_directory: str = os.getcwd(),
        api_keys_path: str = os.getcwd() + "/api_keys.json",
        general_system_description: str = "This is a multi agent system.",
        functions_file: str = "fns.py",
        on_update: Optional[Callable] = None,
        on_complete: Optional[Callable] = None
    ):

        self._current_user_id: Optional[str] = None
        self._current_db_conn: Optional[sqlite3.Connection] = None

        self._file_cache = {}
        self._component_order: List[str] = []

        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        self.processes: Dict[str, Process] = {}
        self.automations: Dict[str, Automation] = {}

        self.parser = Parser()
        
        if config_json and config_json.endswith(".json"):
            try:
                self.build_from_json(config_json)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"System build failed: {e}")
        else:
            self.base_directory = base_directory
            self.api_keys_path = api_keys_path
            self.general_system_description = general_system_description
            self.functions_file = functions_file
            self.on_update = self._resolve_callable(on_update)
            self.on_complete = self._resolve_callable(on_complete)

        self.api_keys: Dict[str, str] = {}
        self._load_api_keys()

        self._last_known_update = None

        self.history_folder = os.path.join(self.base_directory, "history")
        os.makedirs(self.history_folder, exist_ok=True)

    def _resolve_callable(self, func):
        if isinstance(func, str) and func.startswith("fn:"):
            return self._get_function_from_reference(func)
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
        if not os.path.exists(self.api_keys_path):
            self.api_keys = {}
            return
        with open(self.api_keys_path, "r", encoding="utf-8") as f:
            self.api_keys = json.load(f)

    def _get_db_path_for_user(self, user_id: str) -> str:
        return os.path.join(self.history_folder, f"{user_id}.sqlite")

    def _ensure_user_db(self, user_id: str) -> sqlite3.Connection:
        db_path = self._get_db_path_for_user(user_id)
        needs_init = not os.path.exists(db_path)
        conn = sqlite3.connect(db_path,  check_same_thread=False)
        if needs_init:
            self._create_table(conn)
        return conn


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

    def _get_user_db(self) -> sqlite3.Connection:
        if not self._current_user_id:
            # create a new user
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)
        return self._current_db_conn

    def _get_next_msg_number(self, conn: sqlite3.Connection) -> int:
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(msg_number), 0) FROM message_history")
        max_num = cur.fetchone()[0]
        return max_num + 1

    def _save_message(self, conn: sqlite3.Connection, role: str, content: Union[str, dict], type = "user", model: Optional[str] = None):
        if isinstance(content, dict):
            # Convert dict -> JSON, persisting non-JSON objects to files
            content_str = self._dict_to_json_with_file_persistence(content, self._current_user_id)
        else:
            content_str = content

        msg_number = self._get_next_msg_number(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO message_history (msg_number, role, content, type, model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (msg_number, role, content_str, type, model)
        )
        conn.commit()

    def _save_component_output(
        self, db_conn, component, output_dict, verbose=False
    ):
        
        if not hasattr(component, '_last_used_model'):
            model_info = None
        else:
            model_info = component._last_used_model
            del component._last_used_model

        if not output_dict:
            return

        save_role = component.name
        component_type = None
        model_str = None

        if model_info:
            model_str = f"{model_info['provider']}:{model_info['model']}"

        if isinstance(component, Agent):
            component_type = "agent"
        elif isinstance(component, Tool):
           component_type = "tool"
        elif isinstance(component, Process):
            component_type = "process"
        elif isinstance(component, Automation):
           component_type = "automation"

        if isinstance(component, Automation):
            return

        self._save_message(db_conn, save_role, output_dict, component_type, model_str)

        
    def _dict_to_json_with_file_persistence(self, data: dict, user_id: str) -> str:
        """
        Recursively persist non-JSON-friendly values into files, replacing them with "file:/path/to.pkl".
        Then return the JSON representation of the modified dictionary.
        """
        data_copy = self._persist_non_json_values(data, user_id)
        return json.dumps(data_copy, indent=2)

    def _persist_non_json_values(self, value: Any, user_id: str) -> Any:
        if isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                new_dict[k] = self._persist_non_json_values(v, user_id)
            return new_dict

        if isinstance(value, list):
            return [self._persist_non_json_values(item, user_id) for item in value]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return self._store_file(value, user_id)

    def _store_file(self, obj: Any, user_id: str) -> str:
        files_dir = os.path.join(self.base_directory, "files", user_id)
        os.makedirs(files_dir, exist_ok=True)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(files_dir, f"{file_id}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

        self._file_cache[file_path] = obj

        return f"file:{file_path}"

    def _load_files_in_dict(self, value: Any) -> Any:
        """
        Recursively walk a dict/list, detect "file:/path" placeholders, load them,
        and replace with the actual object. Return the fully loaded structure.
        """

        try:
            value = json.loads(value)
        except:
            pass

        if isinstance(value, dict):
            for k, v in value.items():
                value[k] = self._load_files_in_dict(v)
            return value
        elif isinstance(value, list):
            return [self._load_files_in_dict(item) for item in value]
        elif isinstance(value, str) and value.startswith("file:"):
            file_path = value[5:]  # remove "file:"
            return self._load_file(file_path)
        elif isinstance(value, str):
            return value
        else:
            return value
       
    def _load_file(self, file_path: str) -> Any:
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        self._file_cache[file_path] = obj
        return obj
    
    def _get_all_messages(self, conn: sqlite3.Connection, include_model = False) -> List[tuple]:
        cur = conn.cursor()

        if include_model:
            cur.execute("SELECT role, content, msg_number, type, model FROM message_history ORDER BY msg_number ASC")
        else:
            cur.execute("SELECT role, content, msg_number, type FROM message_history ORDER BY msg_number ASC")
        return cur.fetchall()

    def show_history(self, user_id: Optional[str] = None, message_char_limit: int = 2000):
        if user_id is not None:
            self.set_current_user(user_id)

        conn = self._get_user_db()
        rows = self._get_all_messages(conn, include_model = True)
        print(f"=== Message History for user [{self._current_user_id}] ===")
        for i, (role, content, msg_number, msg_type, model) in enumerate(rows, start=1):
            model_str = f" ({model})" if model else ""
            role_str = f"{msg_number}. {msg_type} - {role}{model_str}" if role != "user" else f"{msg_number}. {role}{model_str}"
            if len(content) > message_char_limit:
                content = content[:message_char_limit] + "...\nMessage too long to show in history."
            print(f"{role_str}: {content}")
        print("============================================\n")

    def get_messages(self, user_id: Optional[str] = None) -> List[Dict[str, str]]:
        if user_id is not None:
            self.set_current_user(user_id)

        conn = self._get_user_db()
        rows = self._get_all_messages(conn, include_model=True)

        messages = []
        for role, content, msg_number, msg_type, model in rows:
            try:
                content_data = json.loads(content)
            except json.JSONDecodeError:
                content_data = content

            messages.append({
                "source": role,
                "message": content_data,
                "msg_number": msg_number,
                "type": msg_type,
                "model": model
            })

        return messages


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
        lines.append(f"You task has been defined as follows:{specific_desc.strip()}\n")
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
        models: List[Dict[str, str]] = [{"provider": "groq", "model": "llama-3.1-8b-instant"}],
        default_output: Optional[Dict[str, Any]] = {"response": "No valid response."},
        positive_filter: Optional[List[str]] = None,
        negative_filter: Optional[List[str]] = None
    ):
        # Automatically assign name if not provided
        if name is None:
            name = self._generate_agent_name()
    
        if name in self.agents:
            raise ValueError(f"[Manager] Agent '{name}' already exists.")
    
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
            api_keys=self.api_keys,
            general_system_description=self.general_system_description
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
        default_output: Optional[Dict[str, Any]] = None
    ):
        if name in self.tools:
            raise ValueError(f"[Manager] Tool '{name}' already exists.")
        t = Tool(name, inputs, outputs, function, default_output)
        t.manager = self
        self.tools[name] = t
        self._component_order.append(name)

        return name

    def create_process(self, name: str, function: Callable):
        if name in self.processes:
            raise ValueError(f"[Manager] Process '{name}' already exists.")
        p = Process(name, function)
        p.manager = self
        self.processes[name] = p
        self._component_order.append(name)
        return name

    def create_automation(self, name: Optional[str] = None, sequence: List[Union[str, dict]] = None):
        if name is None:
            name = self._generate_automation_name()

        if name in self.automations:
            raise ValueError(f"Automation '{name}' already exists.")

        automation = Automation(name=name, sequence=sequence)
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
            on_update_params: Optional[Dict] = None
        ) -> Dict:

        db_conn = self._get_user_db()
        if user_id:
            self.set_current_user(user_id)
        elif not self._current_user_id:
            new_user = str(uuid.uuid4())
            self.set_current_user(new_user)

        db_conn = self._get_user_db()

        # Store user-provided input (if any) in the DB
        if input is not None:
            if isinstance(input, dict):
                store_role = role if role else "internal"
                self._save_message(db_conn, store_role, input, store_role)
                if verbose:
                    print(f"[Manager] Saved dict input under role='{store_role}'. => {input}")
            else:
                # assume string
                store_role = role if role else "user"
                self._save_message(db_conn, store_role, input, store_role)
                if verbose:
                    print(f"[Manager] Saved string input under role='{store_role}'. => {input}")

        if component_name is None:
            # Use default or latest automation logic
            if not self.automations or len(self.automations) < 1:
                if verbose:
                    print("[Manager] Using default automation.")
                default_automation_name = "default_automation"
                default_automation_sequence = list(self._component_order)
                if default_automation_name not in self.automations:
                    self.create_automation(name=default_automation_name, sequence=default_automation_sequence)
                comp = self._get_component(default_automation_name)
            elif len(self.automations) == 1:
                automation_name = list(self.automations.keys())[0]
                if verbose:
                    print(f"[Manager] Using single existing automation: {automation_name}")
                comp = self._get_component(automation_name)
            else:
                latest_automation_name = list(self.automations.keys())[-1]
                if verbose:
                    print(f"[Manager] Using the latest automation: {latest_automation_name}")
                comp = self._get_component(latest_automation_name)
        else:
            comp = self._get_component(component_name)
            if not comp:
                raise ValueError(f"[Manager] Component '{component_name}' not found.")

        if isinstance(comp, Automation):
            if verbose:
                print(f"[Manager] Running Automation: {component_name or comp.name}")
            
            if on_update_params:
                output_dict = comp.run(
                    verbose=verbose,
                    on_update_params = on_update_params,
                    on_update=lambda messages, manager, on_update_params: on_update(messages, manager, on_update_params) if on_update else None
                )
            else:
                output_dict = comp.run(
                    verbose=verbose,
                    on_update=lambda messages, manager: on_update(messages, manager) if on_update else None
                )
        else:
            # Agents, Tools, and Processes accept input targeting arguments
            if verbose:
                print(f"[Manager] Running {component_name or comp.name} with target_input={target_input}, "
                    f"target_index={target_index}, target_custom={target_custom}")
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
        on_complete_params: Optional[Dict] = None
    ) -> Dict:
        """
        - If user_id is given, we switch to that DB. If none, we use/create the current user.
        - The output is also saved to DB as a JSON (with file references if needed).
        """

        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        def task():
            try:
                self._run_internal(
                    component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params
                )
                if on_complete:
                    if on_complete_params:
                        on_complete(self.get_messages(user_id), self, on_complete_params)
                    else:
                        on_complete(self.get_messages(user_id), self)
            except Exception as e:
                if verbose:
                    print(f"[Manager] Non-blocking execution error: {e}")

        if blocking:
            result = self._run_internal(
                component_name, input, user_id, role, verbose, target_input, target_index, target_custom, on_update, on_update_params
            )
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


    def build_from_json(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            system_definition = json.load(f)

        general_params = system_definition.get("general_parameters", {})
        self.base_directory = general_params.get("base_directory", os.getcwd())
        self.api_keys_path = general_params.get("api_keys_path", os.path.join(self.base_directory, "api_keys.json"))
        self.general_system_description = general_params.get("general_system_description", "This is a multi-agent system.")
        self.functions_file = general_params.get("functions_file", "fns.py")

        self.on_update = self._resolve_callable(general_params.get("on_update"))
        self.on_complete = self._resolve_callable(general_params.get("on_complete"))

        self._load_api_keys()

        components = system_definition.get("components", [])
        for component in components:
            self._create_component_from_json(component)

        links = system_definition.get("links", {})
        for input_component, output_component in links.items():
            try:
                self.link(input_component, output_component)
            except ValueError as e:
                print(f"Warning: Could not create link from '{input_component}' to '{output_component}'. Error: {e}")

         # Process the stored links after components are created
        if hasattr(self, 'pending_links'):
            for agent_name, tool_name in self.pending_links:
                try:
                    self.link(agent_name, tool_name)
                except ValueError as e:
                    print(f"Warning: Could not link agent '{agent_name}' to tool '{tool_name}'. Error: {e}")

    def _create_component_from_json(self, component: dict):
        component_type = component.get("type")
        name = component.get("name")

        if not component_type or not name:
            raise ValueError("Component must have a 'type' and a 'name'.")

        if component_type == "agent":
            agent_name = self.create_agent(
                name=name,
                system=component.get("system", "You are a helpful assistant."),
                required_outputs=component.get("required_outputs", {"response": "Text to send to user."}),
                models=component.get("models", [{"provider": "groq", "model": "llama-3.1-8b-instant"}]),
                default_output=component.get("default_output", {"response": "No valid response."}),
                positive_filter=component.get("positive_filter", None),
                negative_filter=component.get("negative_filter", None)
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
                function=self._get_function_from_reference(fn),
                default_output=component.get("default_output", None)
            )

        elif component_type == "process":
            fn = component.get("function")
            if not fn:
                raise ValueError("Process must have a 'function'.")
            self.create_process(
                name=name,
                function=self._get_function_from_reference(fn)
            )

        elif component_type == "automation":
            self.create_automation(
                name=name,
                sequence=self._resolve_automation_sequence(component.get("sequence", []))
            )

        else:
            raise ValueError(f"Unsupported component type: {component_type}")

    def _load_fns_module(self):
        if not hasattr(self, '_fns_module'):
            fns_path = os.path.join(self.base_directory, self.functions_file)
            
            if not os.path.exists(fns_path):
                raise FileNotFoundError(f"File '{self.functions_file}' not found in base directory: {self.base_directory}")

            spec = importlib.util.spec_from_file_location("fns", fns_path)
            self._fns_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._fns_module)

        return self._fns_module

    def _get_function_from_reference(self, function_ref: str):
        """
        Load a function from fns.py using a reference in the format "fn:<function_name>".
        """
        if not function_ref or not function_ref.startswith("fn:"):
            raise ValueError(f"Invalid function reference: {function_ref}")

        fns_module = self._load_fns_module()

        function_name = function_ref[3:]

        if not hasattr(fns_module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in 'fns.py'.")

        return getattr(fns_module, function_name)

    def _resolve_automation_sequence(self, sequence):
        """
        Resolve a sequence in an automation, including control flow objects.
        """
        resolved_sequence = []

        for step in sequence:
            if isinstance(step, str):
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
            if condition.startswith("fn:"):
                return self._get_function_from_reference(condition)
            return condition
        elif isinstance(condition, dict):
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
            #print("No user ID provided or set. Message history not cleared.")
            return

        conn = self._get_user_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM message_history")
        conn.commit()
        #print(f"Message history cleared for user ID: {self._current_user_id}")

    def escape_markdown(text: str) -> str:
        """
        Escapes special characters in Markdown v1.
        """
        pattern = r'([_*\[\]\(\)~`>#\+\-=|{}\.!\\])'
        return re.sub(pattern, r'\\\1', text)
    
    def start_telegram_bot(self, telegram_token, component_name = None, verbose = False,
              on_complete = None, on_update = None,
              on_start_msg = "Hey! Talk to me or type '/clear' to erase your message history.",
              on_clear_msg = "Message history deleted."):

        on_update = on_update or self.on_update
        on_complete = on_complete or self.on_complete

        async def send_telegram_response(update, response):
            if isinstance(response, str):
                await update.message.reply_text(response)
            elif isinstance(response, dict):
                for key, value in response.items():
                    if key == "text":
                        await update.message.reply_text(value)
                    elif key == "markdown":
                        value = self.escape_markdown(value)
                        await update.message.reply_text(value, parse_mode="MarkdownV2")
                    elif key == "image":
                        await update.message.reply_photo(value)
                    elif key == "audio":
                        await update.message.reply_audio(value)
                    elif key == "voice_note":
                        await update.message.reply_voice(value)
                    elif key == "document":
                        await update.message.reply_document(value)


        def on_complete_fn(messages, manager, on_complete_params):

            if on_complete is not None:
                response = on_complete(messages, manager, on_complete_params)
            else:
                last_message = messages[-1]
                response = last_message.get("message", {}).get("response")

            if response is None:
                return

            update = on_complete_params["update"]
            loop = on_complete_params["event_loop"]

            def callback():
                asyncio.create_task(send_telegram_response(update, response))

            loop.call_soon_threadsafe(callback)

        def on_update_fn(messages, manager, on_update_params):

            if on_update is not None:
                response = on_update(messages, manager, on_update_params)
            else:
                response = None
            
            if response is None:
                return
            
            update = on_update_params["update"]
            loop = on_update_params["event_loop"]

            def callback():
                asyncio.create_task(send_telegram_response(update, response))

            loop.call_soon_threadsafe(callback)

        # Define async handlers
        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            # Use await for async operations like reply_text
            await update.message.reply_text(on_start_msg)

        async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            self.clear_message_history(update.message.chat.id)
            await update.message.reply_text(on_clear_msg)

        async def get_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
            params = {}
            params["update"] = update
            params["event_loop"] = asyncio.get_running_loop()

            chat_id = update.message.chat.id

            if verbose:
                print(f"[Manager] Received message from user {chat_id}. Executing...")

            def run_manager_thread():
                self.run(
                    component_name = component_name,
                    input=update.message.text, 
                    user_id=chat_id,
                    verbose=verbose,
                    blocking=True,
                    on_update=on_update_fn,
                    on_update_params=params,
                    on_complete = on_complete_fn,
                    on_complete_params=params
                )

            thread = threading.Thread(target=run_manager_thread, daemon=True)
            thread.start()
        
        application = Application.builder().token(telegram_token).build()

        # Add command and message handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("clear", clear))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, get_response))

        # Run the bot
        if verbose:
            print("[Manager] Bot running...")

        application.run_polling()


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
        """
        Main entry point: parse 'spec' and return a dictionary describing what to do.
        """
        spec = spec.strip()
        if not spec:
            return {
                "is_function": False,
                "function_name": None,
                "component_or_param": None,
                "multiple_sources": None,
                "single_source": None,
            }

        if spec.startswith("fn:"):
            remainder = spec[3:].strip()
            return self._parse_as_function_reference(remainder)

        return self._parse_as_non_function(spec)


    def _parse_as_function_reference(self, remainder: str) -> dict:

        if ":" in remainder:
            parts = remainder.split(":", 1)
            function_name = parts[0].strip()
            input_part = parts[1].strip()
            parse_result = self._parse_as_non_function(input_part)
            return {
                "is_function": True,
                "function_name": function_name,
                "component_or_param": parse_result["component_or_param"],
                "multiple_sources": parse_result["multiple_sources"],
                "single_source": parse_result["single_source"],
            }
        else:
            function_name = remainder.strip()
            return {
                "is_function": True,
                "function_name": function_name,
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
