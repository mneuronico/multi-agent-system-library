[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# The Multi-Agent System (MAS) Library

The `mas` library is a powerful and flexible framework designed for creating and managing multi-agent systems. Whether you need a simple chatbot, a sophisticated automation workflow, or a fully integrated AI-driven ecosystem, the `mas` library provides the tools you need to build, scale, and maintain your system with ease.

### What Can You Build with the `mas` Library?

- **Intelligent Chatbots**: Create conversational agents with predefined roles and tasks.
- **API Orchestrators**: Seamlessly integrate multiple APIs through tools managed by intelligent agents.
- **Data Processing Pipelines**: Build processes to transform, analyze, and deliver data efficiently.
- **Decision-Making Systems**: Implement branching logic and dynamic workflows to enable adaptive decision-making.
- **Zero-Config Bootstrap**: Give `mas` nothing but a plain-English description of what you want and it will auto-generate a system for you.

### Why Choose the `mas` Library?

- **Minimal Setup**: Define agents, tools, and workflows using JSON configurations or Python scripts. Start with simple setups and expand as needed.
- **Seamless LLM Integration**: Manage interactions with multiple LLM providers (e.g., OpenAI, OpenRouter, Google, Groq, Anthropic, DeepSeek, Wavespeed, NVIDIA NIM, and local models via LM Studio) without adapting your code or message history for each one.
- **Automated System Prompts**: Automatically generate and manage system prompts to reduce the need for complex prompt engineering, ensuring consistent and reliable agent behavior.
- **Error-Tolerant JSON Parsing**: Includes a robust JSON parser that can recognize and correct malformed JSON structures, even when LLMs produce imperfect outputs.
- **Scalability**: Add or modify components like agents, tools, and processes, expanding the capabilities of a system with minimal refactoring.
- **Integration Ready**: Effortlessly connect with external APIs, databases, and custom Python functions for real-world applications.
- **Dynamic Workflows**: Support for branching, looping, and conditional execution in complex automation sequences.
- **Context-Aware Agents**: Use centralized message histories to enable agents to make informed decisions and maintain continuity across interactions.
- **Multi-User Support**: Manage isolated histories for multiple users, making it ideal for systems that handle multiple independent conversations or workflows.
- **Focus on Logic**: Offload low-level details of message management, task sequencing, and system orchestration to the library so you can concentrate on your application's goals.

The `mas` library empowers developers to create robust, flexible, and intelligent systems while minimizing the complexity of setup and orchestration. Whether you are a beginner experimenting with multi-agent architectures or an expert building large-scale AI-driven workflows, the `mas` library adapts to your needs.

### License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

### Currently in alpha
This library was created and is regularly updated by Nicolas Martorell, Neuroscientist and AI researcher from Argentina. It has not yet been extensively tested and is currently under development. Please **do not use it in production yet**, but feel free to test it and post your issues here, or email me at mneuronico@gmail.com to collaborate or know more about this project.

---


# Multi-Agent System (mas) Library Quick Start Guide

Welcome to the `mas` library! This guide will help you get started with setting up and running your first multi-agent system using a simple configuration.

## Installation

Installing `mas` from GitHub is easy and straightforward.

### Prerequisites

-   Python 3.9+
-   [Git](https://git-scm.com/downloads)
-   API keys for services you plan to use (OpenAI, Groq, etc)

### Install from GitHub

```bash
pip install --upgrade git+https://github.com/mneuronico/multi-agent-system-library
```

## Step 1: Create a Minimal JSON Configuration

Let’s create a minimal configuration to get started with a single agent.

### Example `config.json`

```json
{
    "general_parameters": {
        "api_keys_path": ".env"
    },
    "components": [
        {
            "type": "agent",
            "name": "hello_agent",
            "system": "You are an AI that responds with a friendly greeting."
        }
    ]
}
```

### Reference `config.json` Pattern

For most production-shaped MAS projects, a good starting structure is:

1. Put the full system capability map in `general_system_description`, so every agent knows what the system can do.
2. Define narrow components: processes for deterministic preprocessing, agents for judgment or language/vision tasks, tools for validated function calls, and one automation that orchestrates the flow.
3. Preprocess the latest user message first, route into a list of action strings, run one action path at a time with a `for` + `switch`, and finish with one responder that sees what happened.
4. Use MAS input syntax to keep every component's context explicit.

The following example is intentionally complete enough to copy as a reference. It uses one process to add today's weather context, one agent to turn possible images into text, a router agent that emits a list of actions, a `for` loop over those actions, a `switch` for the action-specific paths, and a final responder.

```json
{
  "general_parameters": {
    "api_keys_path": ".env",
    "functions": "fns.py",
    "history_folder": "history",
    "general_system_description": "This MAS can answer normal user questions, inspect user-provided images, fetch deterministic weather context through local functions, save notes through local functions, and combine all intermediate results into one final user-facing response. Components should rely on previous component outputs instead of redoing work.",
    "default_models": [
      {
        "provider": "google",
        "model": "gemini-2.5-flash"
      }
    ],
    "variables": [
      {
        "key": "tone",
        "type": ["concise", "friendly", "technical"],
        "default": "friendly"
      },
      {
        "key": "final_responder:temperature",
        "type": "number",
        "default": 0.4
      }
    ]
  },
  "components": [
    {
      "type": "process",
      "name": "weather_context",
      "description": "Adds deterministic weather context for the latest user request.",
      "function": "fn:get_today_weather_context"
    },
    {
      "type": "agent",
      "name": "image_describer",
      "description": "Describes any image in the latest user message.",
      "system": "If the latest user message contains images, describe them with useful visual details. If there is no image, return image_description as an empty string.",
      "required_outputs": {
        "image_description": "Detailed image description, or an empty string when no image is present."
      },
      "default_output": {
        "image_description": ""
      }
    },
    {
      "type": "agent",
      "name": "router",
      "description": "Chooses one or more action paths.",
      "system": "Read the latest user request and preprocessing outputs. Return every useful action as a list of strings. Allowed actions: answer_weather, describe_image, save_note, small_talk. Use small_talk when no specialized action is needed.",
      "required_outputs": {
        "actions": "List of strings. Each item must be one of: answer_weather, describe_image, save_note, small_talk."
      },
      "default_output": {
        "actions": ["small_talk"]
      }
    },
    {
      "type": "tool",
      "name": "weather_answer",
      "description": "Turns weather context into a user-facing weather result.",
      "function": "fn:format_weather_answer",
      "inputs": {
        "weather_summary": "Weather summary produced by weather_context.",
        "response": "Latest user text."
      },
      "outputs": {
        "weather_answer": "Prepared weather answer."
      },
      "default_output": {
        "weather_answer": ""
      }
    },
    {
      "type": "agent",
      "name": "visual_answer",
      "description": "Uses the image description to answer visual questions.",
      "system": "Use the image_description and the user request to produce a useful visual answer.",
      "required_outputs": {
        "visual_answer": "Answer based on image_description."
      },
      "default_output": {
        "visual_answer": ""
      }
    },
    {
      "type": "process",
      "name": "note_saver",
      "description": "Saves note-like user requests.",
      "function": "fn:save_note"
    },
    {
      "type": "agent",
      "name": "small_talk_answer",
      "description": "Handles general conversation and unknown actions.",
      "system": "Answer conversationally using the available context. Do not claim that tools were used if no tool output exists.",
      "required_outputs": {
        "small_talk_answer": "General conversational answer."
      },
      "default_output": {
        "small_talk_answer": ""
      }
    },
    {
      "type": "agent",
      "name": "final_responder",
      "description": "Combines all action results into one final reply.",
      "system": "Write one coherent final response in a $tone$ tone. Use outputs from the router and action components. Avoid exposing implementation details unless the user asks.",
      "required_outputs": {
        "response": "Final response to send to the user."
      },
      "model_params": {
        "temperature": "$final_responder:temperature$"
      },
      "default_output": {
        "response": "I could not prepare a reliable response."
      }
    },
    {
      "type": "automation",
      "name": "main_flow",
      "description": "Preprocesses the user message, routes actions, runs action paths, and responds.",
      "sequence": [
        "weather_context:user?-1",
        "image_describer:user?-1",
        "router:*?-8~",
        {
          "control_flow_type": "for",
          "items": ":router?-1[actions]",
          "body": [
            {
              "control_flow_type": "switch",
              "value": ":iterator?-1[item]",
              "cases": [
                {
                  "case": "answer_weather",
                  "body": [
                    "weather_answer:(weather_context?-1[weather_summary], user?-1[response])"
                  ]
                },
                {
                  "case": "describe_image",
                  "body": [
                    "visual_answer:(image_describer?-1[image_description], user?-1[response])"
                  ]
                },
                {
                  "case": "save_note",
                  "body": [
                    "note_saver:(user?-1[response], image_describer?-1[image_description])"
                  ]
                },
                {
                  "case": "default",
                  "body": [
                    "small_talk_answer:*?router?-1~"
                  ]
                }
              ]
            }
          ]
        },
        "final_responder:*?router?-1~"
      ]
    }
  ]
}
```

The companion `fns.py` might look like this:

```python
def _latest_user_text(messages):
    for message in reversed(messages or []):
        if message.get("source") == "user":
            for block in message.get("message", []):
                content = block.get("content") if isinstance(block, dict) else block
                if isinstance(content, dict) and "response" in content:
                    return content["response"]
                if isinstance(content, str):
                    return content
    return ""


def get_today_weather_context(messages=None, manager=None):
    text = _latest_user_text(messages)
    return {
        "weather_summary": "Today is mild and partly cloudy. Replace this with a real weather API call.",
        "weather_query": text
    }


def format_weather_answer(manager, weather_summary, response=None):
    target = response or "your request"
    return {"weather_answer": f"For {target}: {weather_summary}"}


def save_note(messages=None, manager=None):
    note_text = _latest_user_text(messages)
    return {
        "note_saved": bool(note_text),
        "note_text": note_text
    }
```

The automation uses several input syntax patterns:

- `weather_context:user?-1` gives the process only the latest user message.
- `router:*?-8~` gives the router the last eight global messages, regardless of source.
- `weather_answer:(weather_context?-1[weather_summary], user?-1[response])` gives the tool only the latest weather summary and latest user text.
- `small_talk_answer:*?router?-1~` gives the agent all messages from the latest router output onward.
- `final_responder:*?router?-1~` gives the final responder the router decision plus everything produced inside the action loop.

### Preparing API Keys

The `mas` library supports a flexible, three-tiered approach to managing API keys, allowing you to use the best method for your environment (e.g., local development vs. production). When you request a key, the manager searches for it in the following order:

1.  **`api_keys.json` file (Highest Priority)**: If you specify an `api_keys_path` pointing to a JSON file, its contents are loaded directly. Keys found here are always checked first. This is useful for project-specific keys that you want to keep separate.

2.  **`.env` file (Second Priority)**: If `api_keys_path` points to a `.env` file, its variables are loaded into the environment. These variables take precedence over any pre-existing system environment variables with the same name.

3.  **System Environment Variables (Fallback)**: If a key is not found in the JSON or `.env` file, the manager checks the system's environment variables. This is the standard method for production environments, CI/CD pipelines, and containerized deployments.

When you initialize the manager, you can point to a specific file with the `api_keys_path` parameter. If no path is provided, the manager will first look for a `.env` file and then an `api_keys.json` file in your `base_directory`.

This flexible system allows you to keep development keys in a local `.env` file while using secure environment variables in production, without changing your code.

To define a `.env` file, you can do something like:

```dotenv
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
LMSTUDIO_API_KEY=your_mock_lmstudio_key
NVIDIA_API_KEY=your_nvidia_key
```

Provider names can be defined as `<PROVIDER>-API-KEY`, `<PROVIDER>`, `<PROVIDER>-KEY`, `<PROVIDER>_KEY`, and other similar variations (the names are handled as case-insensitive).

To use a `json` file to define API keys, you can do something like:

```json
{
    "openai": "your-openai-key",
    "openrouter": "your-openrouter-key",
    "groq": "your-groq-key",
    "google": "your-google-key",
    "anthropic": "your-anthropic-key",
    "deepseek": "your-deepseek-key",
    "lmstudio": "your-mock-lmstudio-key",
    "wavespeed": "your-wavespeed-key",
    "nvidia": "your-nvidia-key"
}
```

Using a `.env` file is recommended. The `api_keys_path` parameter can refer to a file or directory inside the `base_directory`, or to an absolute path.

You must include an API key for any provider you want to use, even for providers where API keys are not needed, such as LM Studio. In that case, you can place a mock key in your API key file, so that the manager will allow the usage of that provider. In particular, LM Studio requires that the specific software is already installed in your computer, and that the requested model is downloaded and currently loaded to memory.

---

## Step 2: Load and Run the System

Use the JSON configuration to initialize and run the system:

### `main.py`

```python
from mas import AgentSystemManager

# Load the system from the JSON configuration
manager = AgentSystemManager(config="config.json")

# Run the system with user input
output = manager.run(input="Hello world!")
print(output)
```

---

## Step 3: View Message History

You can check the interaction history for the current user:

```python
manager.show_history()
```
---

## Step 4: Open the Visual Dashboard

From any MAS project directory, you can open a local dashboard:

```bash
mas dashboard
```

You can also point it at a project explicitly:

```bash
mas dashboard --directory ./my_mas_project
mas dashboard -d ./my_mas_project --port 8777 --auto-port
mas dashboard -d ./my_mas_project --no-browser
```

The dashboard looks for `config.json` in the selected directory. If it finds a MAS project, it loads the configuration through `AgentSystemManager`, reads the configured history folder, and serves a local browser UI. It does not call any model provider or run your automation.

Available options:

- `--directory`, `-d`: Project directory. Defaults to the current directory.
- `--host`: Local server host. Defaults to `127.0.0.1`.
- `--port`: Local server port. Defaults to `8765`.
- `--auto-port`: Use the requested port if it is free, otherwise choose a free port.
- `--no-browser`: Start the server without opening a browser automatically.
- `--history-limit`: Maximum number of messages to load per user history. Defaults to `200`.

The dashboard has four views:

- **Config**: Shows global parameters and each component as an interpreted component card, including component type, description, models, inputs, outputs, and automation step counts.
- **Automation**: Shows each automation as a visual flow. Component calls, `for` loops, `switch` blocks, `while` loops, and branches are represented as separate flow nodes. MAS input syntax is parsed into readable chips so selectors such as `*?-5~`, `*!(debug)`, and `(research?-1[answer], critic?-1[verdict])?planner?-2~` can be inspected without reading raw JSON.
- **History**: Shows one user history at a time. Messages are grouped by user id, then rendered as message cards with role, type, timestamp, model metadata when present, block types, block content, variable blocks, and block metadata.
- **Raw**: Provides the full dashboard state for debugging. This is a fallback view; the main views are intended to be read as MAS objects rather than JSON.

If no `config.json` is found, the dashboard still opens and reports that the selected directory is not a MAS project. This makes it safe to run `mas dashboard` while moving around directories.

---

## Fast Track: Build a System from Plain Text

If you just have an idea and don’t feel like writing JSON yet, pass the idea straight into the `config` parameter.  

The manager will:

1.  Read this very README to learn about the MAS framework.  
2.  Spin up an *internal* agent called **`system_writer`**. By default this
    bootstrap agent tries OpenAI `gpt-5`, Google `gemini-2.5-pro`, and
    Anthropic `claude-sonnet-4` in order; you can override that with the
    `bootstrap_models` constructor argument.
3.  Ask that agent to turn your description into:
    * `general_parameters`
    * a full **components** list
    * any helper python functions (`fns.py`)
4.  Persist everything under your `base_directory`.
5.  Delete the temporary bootstrap file and continue as if you had written the
    JSON yourself.

```python
from mas import AgentSystemManager

description = (
    "I want a system that finds a YouTube video by keyword, "
    "downloads its transcript, then summarizes it for the user."
)

mgr = AgentSystemManager(
    config=description,            # <--     just the text!
)

# now run it exactly as if you had hand-crafted config.json
result = mgr.run(input="Latest Veritasium video?")
print(result)
```

Or in the most succint way possible:

```python
from mas import AgentSystemManager as ASM

ASM("I want a system that gets a youtube video from a link sent by user, gets the transcript from that video, and summarizes it. Use google's gemini-2.0-flash model.").run("www.youtube.com/watch?v=some-video")
```


After the first run you’ll find:

```
yt_summarizer/
├── config.json   # generated
├── fns.py        # generated helper functions
├── history/…     # per-user DB appears automatically
└── files/…
```

You can open & tweak those files at will – and then you can create the manager from `config.json` like any other explicit configuration.


# Multi-Agent System (mas) Library Documentation

## Introduction

The `mas` library provides a robust framework for building complex multi-agent systems. It allows you to create and manage various types of components: `Agents`, `Tools`, `Processes`, and `Automations`. These components interact via a centralized message history mechanism. The library supports both programmatic construction and system definition using JSON files.

## Core Concepts

### Components

Components are the fundamental building blocks of your multi-agent system:

-   **`Agent`**: Agents utilize Large Language Models (LLMs) to generate responses. They are configured with system prompts, required output structures, and filters that specify which messages to use as context. Agents can use tools to accomplish a task. They receive a list of messages (full message history as default) as input and their output is always a dictionary with required fields. The library manages system prompts automatically so that the JSON responses from LLMs always conform with the required outputs.
-   **`Tool`**: Tools perform specific actions (like API calls, database queries, etc.) and are typically used by agents. Tools receive predetermined input fields as a dictionary (typically from a specific agent that is using the tool) and produce an output dictionary, which can then be used by other agents, or it can be processed in some way.
-   **`Process`**: Processes perform data transformations, data loading, or any other kind of data manipulation. They can be used to insert any necessary code or processing that cannot be natively managed by the library. They receive a list of messages and return an output dictionary which can then be used by other agents or tools.
-   **`Automation`**: Automations are workflows that orchestrate the execution of other components (agents, tools and processes). They manage the sequence of steps and support branching and looping structures. Their input is directly passed to the first component in their structure and they return a dictionary with the output of the latest executed component.

### Message History and the Block System

The heart of the `mas` library is its message history and block-based data system. This architecture serves as the central database for all components, ensuring that data, including complex multimodal content, is exchanged in a standardized, persistent, and universally understood format.

#### The Message History Database

All interactions between components are recorded in SQLite history databases. By default, MAS uses one SQLite database per `user_id`, preserving an isolated chronological record for each conversation or workflow. Each message in the history table includes:

-   A unique identifier (`id`).
-   A sequential message number (`msg_number`), which ensures chronological order.
-   The `type` of the component that was run (e.g., `agent`, `tool`, `process`, `user`, `iterator`).
-   The `model` that was called if `type` is `agent` (in the format `'provider:model_name'`).
-   A `role` that indicates the name of the component that produced the message (e.g., `hello_agent`, `my_tool`).
-   The `content` of the message, which is stored as a JSON string representing a **list of blocks**. This is the core of the data exchange system.
-   The `timestamp` indicating when the message was added, in the specified timezone (defaults to `UTC`).

Each `user_id` has its own isolated message history by default, allowing the manager to handle multiple concurrent and independent sessions. For deployments that prefer fewer history files, set `history_mode="shared"` to store all users in shared history databases. In shared mode every row includes a `user_id` column, reads are filtered back to the requested user, and files rotate by `history_max_messages` (default `1000`) or, when selected, by `history_rotation="time_period"` with `history_period` defaulting to one week.

#### The Block System: A Universal Data Format

To handle everything from simple text to images and complex data structures, `mas` normalizes all message content into a **list of blocks**. You rarely need to construct these blocks manually, as the library handles the conversion automatically. Understanding their structure is key to building advanced systems.

##### Standard Block Types

Here are the primary block types the system uses:

-   **Text Block**: For plain text or JSON-serialized data.
    ```json
    {
      "type": "text",
      "content": "This can be a simple string or a JSON-encoded object."
    }
    ```

-   **Image Block**: For visual content. The `content` is a dictionary specifying the image source.
    ```json
    {
      "type": "image",
      "content": {
        "kind": "file" | "url" | "b64",

        // Used when kind is "file". The path points to a file managed by the MAS library.
        "path": "file:/path/to/mas/files/user_id/image_id.jpg",
        
        // Used when kind is "url".
        "url": "https://example.com/some_image.png",

        // Used when kind is "b64".
        "b64": "iVBORw0KGgoAAAANSUhEUg...",

        // (Optional) Image detail level for vision models, e.g., "low", "high", "auto".
        "detail": "auto" 
      }
    }
    ```

-   **Audio Block**: For audio content, typically from speech-to-text or user uploads.
    ```json
    {
      "type": "audio",
      "content": {
        "kind": "file",
        "path": "file:/path/to/mas/files/user_id/audio_id.ogg"
      }
    }
    ```

-   **Variable Block**: For changing runtime variables and component parameters from history.
    ```json
    {
      "type": "variable",
      "key": "tone",
      "value": "casual"
    }
    ```
    Variables are defined globally in `general_parameters.variables`:
    ```json
    {
      "general_parameters": {
        "variables": [
          {"key": "tone", "type": ["formal", "casual"], "default": "formal"},
          {"key": "temperature", "type": "number", "default": 0.2}
        ]
      }
    }
    ```
    Any string in component config can reference a variable with `$key$`. A full-string placeholder keeps the original value type, so `"temperature": "$temperature$"` resolves to a number. The current value is per user: MAS uses the latest `variable` block for that `user_id`, or the configured default when no block exists.

    Component parameters can be overridden with variable keys shaped like `component_name:parameter_name`. For example, `{"type":"variable","key":"assistant:temperature","value":0}` updates the agent's model params for subsequent runs, and `assistant:provider` / `assistant:model` can redirect the model used by that component. Python code can read or write these values with `manager.get_variable(key, user_id=None)` and `manager.set_variable(key, value, user_id=None)`.

#### Automatic Conversion to Blocks (`_to_blocks`)

You don't need to worry about creating these structures yourself. When you call `manager.run(input=...)` or `manager.add_blocks(content=...)`, the library automatically converts your data into the appropriate block format based on these rules:

1.  **If you provide a valid list of blocks**: It's used as-is.
2.  **If you provide a mixed list** (e.g., `['Analyze this:', './img.png']`): Each element is processed individually and the resulting blocks are combined into a single list.
3.  **A `str`**:
    - If the string is a path to a valid local image file, it's read, saved internally, and converted into an `image` block.
    - Otherwise, it's treated as plain text and becomes a `text` block.
4.  **Image `bytes` or `bytearray`**: If the byte header is recognized as an image, the bytes are saved to a file in the `files/` directory and wrapped in an `image` block. Unknown byte payloads are not guessed as media; put them inside a dictionary or an explicit block if you want MAS to persist them as file references.
5.  **A `dict`**: The dictionary is placed inside a `text` block as structured content. If it contains non-JSON values such as bytes or custom class instances, those values are saved under `files/` and replaced with `file:/...` references.

#### Consuming Data from Blocks

Different components access data from the block system in different ways, tailored to their function:

-   **Agents**: Agents receive the raw `List[Block]` for all relevant messages in their context. The library then automatically formats this list into the specific multimodal format required by the target LLM provider (e.g., OpenAI, Google Gemini, Anthropic Claude). This allows you to write provider-agnostic agent logic.

-   **Tools**: Tools are designed to be simple Python functions that expect a dictionary of keyword arguments. Therefore, the manager extracts this dictionary for them by:
    1.  Finding the **first `text` block** in the target message's content.
    2.  Attempting to parse that block's content as a JSON dictionary.
    3.  If successful, that dictionary is passed to the tool.
    4.  If it's not a valid JSON dictionary, the raw string is passed in a `{"text_content": "..."}` wrapper.
    *This means tools cannot directly process images or audio; they must be triggered by an agent that has analyzed the media and produced a structured `text` block as output.*

-   **Processes**: Processes receive the most comprehensive input: a list of full message objects (`{"content": ..., "message": ..., ...}`). The `message` key within each object contains the complete `List[Block]` for that historical entry, giving the process full access to all past data in its original, structured form.

### Roles

Each component is assigned a unique `role` when its output is stored in the database. Roles can simply be the name of the component, or they can also be `user` (messages directly from a user, when calling `run` with `input` string), or any other custom role defined by the developer when calling `manager.run()`.


### Default File Structure

To create a system, it is recommended to build a file structure in your `base_directory` like the one shown below:

```text
base_directory/
├── main.py
├── .env
├── fns.py
└── config.json
```

Key files:

-  `main.py`: Python file where you will import the library and use it. This file is never recognized by the library, it is used to create and manage the system.
-  `.env`: Default location for API keys. `api_keys.json` will also be recognized by the system by default.
-  `fns.py`: Default location for function definitions (tools/processes/others).
-  `config.json`: JSON file where the `manager` and all `components` are specified. This is optional, as the library can be handled programmatically instead. However, JSON configuration is strongly recommended for most use cases.

You can override these defaults by specifying absolute or relative paths in the configuration.

On top of these initial files, the library creates subdirectories inside the `base_directory` when the `manager` is initialized:

-  `history/`: Automatically created if missing. Contains SQLite databases (`.sqlite` files) for each user's conversation history
-  `files/`: Automatically created if missing. Stores serialized objects from components that return non-JSON-serializable data

## The Agent System Manager

### Initialization

```python
from mas import AgentSystemManager
import logging # only necessary if it is required to set a specific log level

manager = AgentSystemManager(
    config="config.json",
    base_directory="path/to/your/base_dir",
    history_folder="path/to/your/history_folder",
    files_folder="path/to/your/files_folder",
    api_keys_path="path/to/your/api_keys.env",
    costs_path="path/to/your/costs.json",
    general_system_description="A description for the overall system.",
    functions="my_fns_file.py",
    default_models=[{"provider": "groq", "model": "llama-3.1-8b-instant"}],
    imports=["common_tools.json"],
    on_update=on_update,
    on_complete=on_complete,
    include_timestamp=False,
    timezone="UTC",
    log_level=logging.INFO,
    admin_user_id="your_telegram_or_whatsapp_id",
    usage_logging=False,
    model_failure_policy={"enabled": True}
)
```

The `AgentSystemManager` manages your system’s components, user histories, and general settings.

-   **`config`**: Path to a JSON file or a plain-English description (triggers automatic bootstrap of system).
-   **`base_directory`**: Specifies the directory where user history databases (`history` subdirectory) and pickled object files (`files` subdirectory) are stored. Also the location of `fns.py`.
-   **`history_folder`**: Path for storing per-user SQLite databases. Defaults to `<base_directory>/history`.
-   **`history_mode`**: History storage mode. Defaults to `per_user`, which creates one SQLite database per `user_id`. Set to `shared` (aliases: `global`, `all_users`) to store all users together in `shared_history_000001.sqlite`, `shared_history_000002.sqlite`, etc., with a `user_id` column on each message row.
-   **`history_rotation`**: Shared-history rotation strategy. Defaults to `message_count`; use `time_period` to rotate by elapsed time, or `both` to rotate on either threshold.
-   **`history_max_messages`**: Maximum messages per shared history file when count rotation is active. Defaults to `1000`.
-   **`history_period`**: Period for time-based shared history rotation. Accepts values like `1w`, `7 days`, or seconds. Defaults to one week.
-   **`files_folder`**: Path for storing serialized object files. Defaults to `<base_directory>/files`.
-   **`api_keys_path`**: Path to a `.env` or `json` file containing API keys. Keys from this file are given priority over system environment variables, making it easy to manage keys for different environments.
-   **`costs_path`**: Path to a `.json` file containing model and tool costs. If provided, the system will calculate and store the USD cost for agent and tool executions when `return_token_count` is set to `True`.
-   **`general_system_description`**: A description appended to the system prompt of each agent.
-   **`functions`**: The name of a Python file, or list of Python files, where function definitions must be located. Files must either exist in the base directory or be referenced as an absolute path. If not defined, this defaults to `fns.py` inside the `base_directory`.
-   **`default_models`**: A list of models to use when executing agents, for agents that don't define specific models. Each element of the list should be a dictionary with two fields, `provider` (like 'groq' or 'openai') and `model` (the specific model name). These models will be tried in order, and failure of a model will trigger a call to the next one in the chain.
-   **`imports`**: List of component import specifications. Each entry can be either `"<your_json>.json"` to import all components from that file, or `"<your_json>.json?[comp1, comp2]"` to import specific components from that file.
-   **`on_update`**: Function to be executed each time an individual component is finished running. It can accept `messages`, `manager`, and optionally a third `params` argument when the caller provides callback parameters. Useful for updating an external database or sending incremental bot messages during an automation.
-   **`on_complete`**: Function to be executed when `manager.run()` reaches completion. This is equivalent to `on_update` when calling `manager.run()` on an individual component (if both are defined, both can run), but it is different for automations because it runs at the end of the automation. It follows the same callback signature flexibility as `on_update`.
-   **`include_timestamp`**: Whether the agents receive the `timestamp` for each message in the conversation history. False by default. This is overriden by the `include_timestamp` parameter associated with each agent, if specified. 
-   **`timezone`**: String defining what timezone should the `timestamp` information be saved in. Defaults to `UTC`.
-   **`log_level`**: Logging level for the library. Defaults to `logging.DEBUG`.
-   **`admin_user_id`**: A specific user ID (e.g., your Telegram chat ID) granted access to administrative commands.
-   **`usage_logging`**: If `True`, enables the persistent usage and cost logging system. Defaults to `False`.
-   **`model_failure_policy`**: Optional manager-wide adaptive fallback policy for model failures. Enabled by default. When a model fails, MAS records that provider/model failure in memory and temporarily prefers the highest-priority model that has not failed recently. The original model order is restored after cooldown or after a later success. Set this to `False` to always probe models strictly in the configured order. Useful tuning fields are `base_cooldown_seconds`, `min_cooldown_seconds`, `max_cooldown_seconds`, `failure_half_life_seconds`, `history_retention_seconds`, and `failure_weights`.

`on_update` and `on_complete` can be defined as callables directly, or they can be strings referring to the name of the function to use, located in one of the `functions` files. To accomplish this, _function syntax_ must be used.

#### Function Syntax

You can refer to functions with strings anywhere you can use functions throughout the system (e.g. tools, processes, `on_update`, `on_complete`, control flow statements in automations, etc). To do so, the string must follow one of the following formats:

- `"fn:<your_function_name>"`: The system will look for the function in one of the files included in `functions`. If `functions` is a list, the files are checked in order, so function name redundancy will be resolved by getting the function included in the file that appears first in this list.
- `"<your_file.py>:<your_function_name>"`: The system will look for the specified file inside the `base_directory`, and then look for the specified function inside that file.
- `"<path/to/your_file.py>:<your_function_name>"`: The system will look for the specified file in the absolute path provided, and then look for the specified function inside that file.

In all of these cases, the `.py` extension is optional, the system will look for a `.py` file with the specified name regardless of whether the extension was included in the string.

You can accomplish the same thing when defining the system from a JSON file:

```json
{
  "general_parameters": {
    "base_directory": "/path/to/your/base_dir",
    "api_keys_path": "/path/to/your/api_keys.env",
    "costs_path": "costs.json",
    "general_system_description": "This is a description for the overall system.",
    "functions": "my_fns_file.py",
    "default_models": [
            {"provider": "deepseek", "model": "deepseek-chat"},
            {"provider": "groq", "model": "llama-3.3-70b-versatile"}
        ],
    "imports": [
        "common_tools.json"
    ],
    "on_update": "fn:on_update_function", 
    "on_complete": "fn:on_complete_function",
    "include_timestamp": true,
    "timezone": "America/Argentina/Buenos_Aires",
    "model_failure_policy": {
        "enabled": true,
        "base_cooldown_seconds": 60,
        "max_cooldown_seconds": 1800
    }
  },
  "components": []
}
```

### Setting the Current User

The `mas` library is designed for multi-user and concurrent applications. To manage this safely, the concept of the "current user" is **thread-specific**. This means each execution thread maintains its own user context, preventing data from different users from getting mixed up in environments like web servers or bots.

#### Explicitly Setting the User

To specify which user's history you want to work with in the current thread, you must call `set_current_user()`. This is the recommended approach, especially in web applications where each request is handled by a different thread.

```python
# In a web request handler or a new thread, always set the user first.
manager.set_current_user("user_id_from_request")
```

Calling this method associates the current thread with the specified user's database. All subsequent calls to methods like `run()`, `get_messages()`, or `show_history()` from the same thread will automatically use that user's history.

#### Automatic User Management

If you call a method that requires a user context (like `run()`) without first setting a user for the current thread, the manager will automatically create a new user with a unique UUID and set it for that thread.

Because the user context is thread-local, you **cannot** set the user once and expect it to apply to all future operations across your application. You **must** call `set_current_user(user_id)` at the beginning of every task that needs to be associated with a specific user (e.g., at the start of each web request handler or when processing a new bot message).

Failure to do so will result in a new, anonymous user (with an empty history) being created for each new thread, which is likely not the desired behavior.


### Creating Components

#### Agents

```python
agent_name = manager.create_agent(
    name="myagent",  # Optional, defaults to agent-<n>
    system="You are an AI assistant that uses tools.",  # Default: "You are a helpful assistant."
    required_outputs={   # Default: {"response": "Text to send to user."}
        "query": "The query to search for",
         "items": {"description": "Items to show", "type": "string"}
    },
    models=[  # Defaults to system's default_models, which, if not defined, defaults to: [{"provider": "groq", "model": "llama-3.1-8b-instant"}]
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "groq", "model": "llama-3.3-70b-versatile"}
    ],
    default_output={"query": "default query", "items": "default items."}, # Default: {"response": "No valid response."}
    positive_filter=["user", "tool-mytool"], # Default: None
    negative_filter=["bot-otheragent"],  # Default: None
    include_timestamp=False, # Default: False
    model_params={"temperature": 1.0, "max_tokens": 4096},
    description="Uses tools to do X." # Default: None
)
```

-   **`name`**:  The unique name of the agent. If not provided, it will automatically be assigned as `agent-<n>`, where `n` is a sequential number.
-   **`system`**:  The system prompt for the agent, defining its role and instructions. The manager will combine this with the general system prompt for the system and with other necessary specifications to ensure the output JSON matches the expected format.
-   **`required_outputs`**: Specifies the structure of the JSON output that the agent should produce. It can be either a simple string (where the type defaults to `string` and the output field is named `response`) or a dictionary providing descriptions and types for each output field:
    ```
        {
            "field_name1": "description of the field",
            "field_name2": { "description": "description of field2", "type": "string"}
        }
    ```
    The manager will automatically add these required outputs to the agent's system prompt so that the model knows to produce a JSON with the required fields.
-   **`models`**:  A list of LLM models to try, in order, to fulfill the request:
    ```
        [
            {"provider": "openai", "model": "gpt-4-turbo-preview"},
            {"provider": "groq", "model": "llama-3.1-8b-instant"}
        ]
    ```
    Supported providers so far are: `"openai"`, `"openrouter"`, `"google"`, `"groq"`, `"anthropic"`, `"deepseek"`, `"wavespeed"`, `"nvidia"`, and `"lmstudio"`. Ensure the corresponding `api_key` is available in your API key file. OpenRouter models should use OpenRouter slugs such as `"openai/gpt-5"` or `"anthropic/claude-sonnet-4"`. LM Studio models can optionally include a `base_url` in the model dictionary when connecting to a non-default server. Wavespeed models use vendor-prefixed slugs such as `"moonshotai/kimi-k2.5"` or `"anthropic/claude-sonnet-4.6"`; the provider calls the Wavespeed LLM gateway (`https://llm.wavespeed.ai/v1`) with an automatic 401-fallback to `https://tropical-llm.wavespeed.ai/v1`. NVIDIA models use NIM/API Catalog slugs such as `"nvidia/llama-3.1-nemotron-nano-8b-v1"` and call the OpenAI-compatible NVIDIA endpoint (`https://integrate.api.nvidia.com/v1`); a model dictionary can also include `base_url` for a self-hosted NIM endpoint.
    Agent responses also keep provider diagnostics in the first output block's `metadata`. Use `provider_response` for the successful raw provider response, `provider_attempts` for every attempted model/provider, and `provider_errors` for normalized errors. If a model is temporarily suppressed after recent failures, `provider_attempts` includes a synthetic skipped attempt with `skipped=True`, `skip_reason`, and `model_health`. These fields are persisted in history and can be read with `manager.get_messages(user_id)` or `manager.read(get_full_dict=True)`.
    User messages produced by bot integrations keep incoming transport metadata in the first input block's `metadata.user_message`, including channel, user id, message type, timestamp, media info, and normalized original payload when available.
    Model availability state is manager-wide and in memory. Use `manager.get_model_health()` to inspect recent failures, cooldowns, and the current status for registered models. Use `manager.clear_model_health()` to reset that operational state during tests or manual intervention.
    ```python
    last = manager.read(get_full_dict=True)
    metadata = last["message"][0].get("metadata", {})
    raw_response = metadata.get("provider_response", {}).get("raw_response")
    attempts = metadata.get("provider_attempts", [])
    errors = metadata.get("provider_errors", [])
    ```
-   **`default_output`**: The output to use when all the models fail, should match the `required_outputs`.
-   **`positive_filter`**: A list of `roles` to be included in the context of the agent (all other roles will be ignored if this is defined).
-   **`negative_filter`**:  A list of `roles` to be excluded from the context.
    You can filter using these values:
    -   `user`: selects messages from the user.
    -   `agent`: selects messages from all roles from the agent type.
    -   `tool`: selects messages from all roles from the tool type.
    -   `process`: selects messages from all roles from the process type.
    -   Exact role names (e.g., `myagent`)
-   **`include_timestamp`**:  Whether this agent should receive `timestamp` information for each message. Defaults to whatever was defined for the manager if not specified, which itself defaults to False.
-   **`model_params`**: Dictionary including params for advanced LLM configuration. Supported params right now are `temperature`, `max_tokens` and `top_p`. Not defining these will use default configuration for each provider.
-   **`description`**: Optional description of the component, solely to be read by the developer.


You can create agents when defining the system from a JSON file by including them in the component list:

```json
{
  "components": [
    {
      "type": "agent",
      "name": "myagent", 
      "system": "You are an AI assistant that uses tools.",
      "required_outputs": {
        "query": "The query to search for",
        "items": {
          "description": "Items to show",
          "type": "string"
        }
      },
      "models": [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "groq",  "model": "llama-3.3-70b-versatile"}
      ],
      "default_output": {
        "query": "default query",
        "items": "default items."
      },
      "positive_filter": ["user", "mytool"],
      "negative_filter": ["otheragent"],
      "include_timestamp": true,
      "model_params": {
        "temperature": 1.0,
        "max_tokens": 4096
      },
      "description": "Uses tools to do X."
    }
  ]
}
```

#### Tools

```python
def my_tool_function(query): # it can also be my_tool_function(manager, query) if the manager object is needed inside the function
    # Make a call to an API based on the query
    return {"items": ["item1", "item2"]}

tool_name = manager.create_tool(
    name="mytool",
    inputs={"query": "query from the agent to call the tool"},
    outputs={"data_field_1": "some data returned by the api",
            "data_field_2": "more data returned by the api"},
    function=my_tool_function,
    default_output={"items": "Default items"},  # Default: {}
    description="Executes API calls for data retrieval"
)
```

-   **`name`**:  The name of the tool.
-   **`inputs`**:  A dictionary describing the input parameters for the `function` using descriptions and names.
-   **`outputs`**:  A dictionary describing the output parameters of the `function` using descriptions and names.
-   **`function`**: A callable (function) that performs the task of the tool. This function receives as many arguments as needed, which must be defined in the same order as the dictionary that will be used as input for this tool (the dictionary from the latest message is used by default, but more complex inputs can be defined as explained below). This function can also optionally receive the `manager` as its first argument, in which case `"manager"` should be the first parameter in the function definition, followed by all parameters in the input dictionary.
-  **`default_output`**: Output to use if there's an error during the function call, or an exception has been raised by the function.
-   **`description`**: Optional description of the component, solely to be read by the developer.

Tools can be included in the component list of the config JSON file just like agents:

```json
{
  "components": [
    {
      "type": "tool",
      "name": "mytool",
      "description": "Executes API calls for data retrieval.",
      "inputs": {
        "query": "query from the agent to call the tool"
      },
      "outputs": {
        "data_field_1": "some data returned by the api",
        "data_field_2": "more data returned by the api"
      },
      "function": "fn:my_tool_function",
      "default_output": {
        "items": "Default items"
      }
    }
  ]
}

```

#### Processes

```python
def my_process_function(manager, messages): # in this case, both manager and messages are optional
    # do anything and return any values that need saving in a dict
    return {"content": "some content from local file"}

process_name = manager.create_process(
    name="myprocess",
    function=my_process_function,
    description="This is what the function does."
)
```

-   **`name`**: The name of the process.
-   **`function`**: A callable (function) that performs data transformations, data loading, etc. This function can receive as argument the `manager` object itself (the parameter must be called `"manager"`) and/or a list of messages (must be named `"messages"`). Each message is a dictionary with two fields ("source", with the source role, and "message" with the actual content, which itself contains `"source"`, `"message"`, `"msg_number"`, `"type"` and `"timestamp"`). The function must return a dictionary, which will be saved in the message history. Both input parameters are optional: you can define process functions which only receive either the `manager` object, or only the `messages` list, or neither. The function call is invariant to the order of the parameters.
-   **`description`**: Optional description of the component, solely to be read by the developer.

You can also define processes in the config JSON file:

```json
{
  "components": [
    {
      "type": "process",
      "name": "myprocess",
      "function": "fn:my_process_function",
      "description": "This is what the function does."
    }
  ]
}
```

#### Automations

Automations allow you to orchestrate multiple components (agents, tools, processes, and even other automations) in a structured workflow. An automation is defined by a name and a sequence of steps. Each step can either be a simple component name or a control flow dictionary (more on that below).

```python
automation_name = manager.create_automation(
    name="myautomation",   # Optional, defaults to automation-<n>
    description="Orchestrates a series of agents and tools for end-to-end processing.",
    sequence=[
        "first_agent",
        "first_tool",
        "decision_agent",
        {
           "control_flow_type": "branch",
           "condition": "is_tool_needed",
            "if_true": [
                "another_tool",
                "a_process"
            ],
            "if_false": [
                "another_agent"
            ]
         },
         {
            "control_flow_type": "while",
            "start_condition": True, # runs the first pass
            "end_condition": "fn:check_if_query_is_valid:second_loop_agent?[query]",
            "body": [
                "first_loop_agent",
                "second_loop_agent"
            ]
          }

    ]
)
```

-   **`name`**: The name of the automation.  If not specified, defaults to `automation-<n>`.
-   **`description`**: Optional description of the component, solely to be read by the developer.
-   **`sequence`**: An ordered list of steps to execute. Steps can be:
    -   A string representing a component name, with an optional input specification (more on **`mas` input syntax** below).
    -   A control flow dictionary (`"branch"`, `"while"`, `"for"`, or `"switch"`) - for more details, please refer to the section below.

Defining an automation in the config JSON file is as simple as including it in the list of components, just like any other component:

```json
{
  "components": [
    {
      "type": "automation",
      "name": "myautomation",
      "description": "Orchestrates a series of agents and tools for end-to-end processing.",
      "sequence": [
        "first_agent",
        "first_tool",
        "decision_agent",
        {
          "control_flow_type": "branch",
          "condition": "is_tool_needed",
          "if_true": [
            "another_tool",
            "a_process"
          ],
          "if_false": [
            "another_agent"
          ]
        },
        {
          "control_flow_type": "while",
          "start_condition": true,
          "end_condition": "fn:check_if_query_is_valid:second_loop_agent?[query]",
          "body": [
            "first_loop_agent",
            "second_loop_agent"
          ]
        }
      ]
    }
  ]
}
```

Note that these examples use the `mas input syntax`, which will be explained below.

### Control Flow Statements

Control flow statements give you the ability to introduce decision points, loops, and iterations into your automation workflows. They come in several types: `branch`, `while`, `for`, and `switch`. Each type supports specific keys and behaviors to control the execution of the automation sequence.

#### Branch Statements

A branch statement allows you to choose between two different sets of steps based on a condition.

**Structure:**

```json
{
  "control_flow_type": "branch",
  "condition": "my_condition",
  "if_true": ["comp1", "comp2"],
  "if_false": ["comp3", "comp4"]
}
```

**Parameters:**

-  **`condition`**: This can be a literal boolean (true or false), a string that will be evaluated as a boolean (for example, by refering to a field produced by a previous component that is itself a boolean) or a dictionary for advanced conditional evaluations. Details on conditionals will be provided in the `MAS input syntax` section.
-  **`if_true`**: A list of steps (components or nested control flow statements) to execute if the condition evaluates to true.
-  **`if_false`**: A list of steps to execute if the condition evaluates to false.

**Example:**

```json
"sequence": [
  "decider",
  {
    "control_flow_type": "branch",
    "condition": ":decider?is_tool_needed",
    "if_true": ["a_tool", "a_process"],
    "if_false": ["an_agent"]
  }
]
```

#### While Loops

A while loop enables repeated execution of a set of steps until a specified condition is met.

**Structure:**

```json
{
  "control_flow_type": "while",
  "start_condition": "condition1",
  "end_condition": "condition2",
  "body": ["comp1", "comp2"]
}
```

**Parameters:**

- **`start_condition`**: A conditional (boolean, string, or dict) that determines whether to run the first iteration. If omitted, the first iteration will run by default.
- **`end_condition`**: A conditional (boolean, string, or dict) that is evaluated before each iteration. When it evaluates to `true`, the loop stops.
- **`body`**: A list of steps to execute on each iteration of the loop.

**Example:**

```json
"sequence": [
  {
    "control_flow_type": "while",
    "start_condition": true,
    "end_condition": ":loop_decider?[is_process_complete]",
    "body": [
      "some_agent",
      "some_tool",
      "loop_decider"
    ]
  }
]
```

#### For Loops

A for loop iterates over a collection of items, executing a body of steps for each item.

**Structure:**

```json
{
  "control_flow_type": "for",
  "items": ":my_input_items",
  "body": ["comp1", "comp2"]
}
```

**Parameters:**

- **`items`**: Defines what to iterate over. This can be:
  - **A numeric range**:
    - A single integer `n` is interpreted as the range `[0, n)`.
    - An array `[a, b]` defines the range from `a` to `b` (including a, excluding b).
    - An array `[a, b, c]` defines the range from `a` to `b` in steps of `c`.
  - **A component reference (via MAS input syntax)**:
    - Single message, list, or dictionary fields can be used for iteration.
- **`body`**:  
  A list of steps to execute on each iteration.

Each time a cycle of the for loop starts, the current item will be added to the message history, under the role of `"iterator"`, wrapped in a dictionary with two fields (`"item_number"` as the cycle number and `"item"` as the current item). These messages behave just like any other message in the history, and allow components inside the loop to easily access the current item being processed, by referring to the latest `iterator` message using `MAS input syntax`.

When using a component reference in the `items` field, these are the possible references you can make and the expected behavior for each:

- **`single message`**: If the input string used in `items` refers to a single message, but this message contains more than one field, the iterator will execute one cycle per field, wrapping each item in a dictionary with fields `"key"` and `"value"`, with the key and value of the current dictionary item. If the message contains only one field, the iterator will assume the content of this field is the object which must be iterated over, and will try to resolve that field as one of the following types.
- **`single number or list of two or three numbers`**: If the field referred to is a number or a list of at most 3 numbers, it will be interpreted as a numeric range, as described above. This is useful when loops must be ran a certain number of times, but than number is not known in advanced but defined by a component.
- **`list of arbitrary length and types`**: If the field referred to is a list, the iterator will execute one cycle per element.
- **`dictionary`**: If the field referred to is itself a dictionary, the iterator treats it the the same way as a single message. If it has more than one field, it iterates through those fields. Else, checks the inner field recursively.

A component reference inside the `items` field can never refer to more than one message, as this is ill-defined for item iteration and will result in an exception.


**Example:**

```json
"sequence": [
  "question_generator",
  {
    "control_flow_type": "for",
    "items": ":question_generator?-1?[questions]",
    "body": [
      "question_responder:(iterator?-1)"
    ]
  }
]
```

In this example, we assume that `question_generator` is an agent which produces a list of questions. Then, we use a `for` loop to iterate through the list named `questions` inside the latest `question_generator` message. For each question, a `question_responder` agent takes as input the latest `iterator` message (which corresponds to the current item being processed, in this case, the current question) and produces a response for that question. The specifics of the `MAS input syntax` used in this example will be explained below.


#### 4. Switch Statements

A switch statement selects one out of several possible branches based on the value of an expression. This is useful, for example, when trying to decide on a set of predefined actions.

**Structure:**

```json
{
  "control_flow_type": "switch",
  "value": "reference_to_switched_value",
  "cases": [
    {
      "case": "case1",
      "body": [ "comp1", "comp2"]
    },
    {
      "case": "case2",
      "body": [ "comp3", "comp4"]
    },
    {
      "case": "default",
      "body": ["comp5", "comp6"]
    }
  ]
}
```

**Parameters:**

- **`value`**: The source value for comparison. It can be a literal (string, number, boolean) or a MAS input syntax reference (e.g., `":my_agent?[field]"`) that must resolve to a single value.
- **`cases`**:  
  An ordered list of case objects, where each case contains:
  - **`case`**: The value to compare against.
  - **`body`**: A list of steps to execute if the value matches the case.
  
The special `case` value of `"default"` will catch any unmatched values.

**Example:**

```json
"sequence": [
  "selector",
  {
    "control_flow_type": "switch",
    "value": ":selector?[action]",
    "cases": [
      {
        "case": "weather",
        "body": ["weather_tool", "weather_agent"]
      },
      {
        "case": "news",
        "body": ["news_tool", "news_agent"]
      },
      {
        "case": "default",
        "body": ["fallback_agent"]
      }
    ]
  }
]
```

Note that for all conditionals and values in control flow statements, starting the string with a colon signals an input from a certain component, while not using a colon is assumed to refer to a specific key inside the first text block of the latest message. For example, in the case of the switch statement in this section's example system, `"value": ":selector?[action]"` is parsed correctly because the string starts with a colon and then refers to a component. The same effect could have been achieved by using the string `"action"`, since `selector` was the last component to be ran before the switch statement.


### Component Imports

The system supports importing components from external JSON files to enable modular architecture and component reuse. This works for both system-wide components and automation-specific references. It also works both when defining the system programmatically and from a JSON file. These JSON files must contain a `"components"` field, which must be a list of components.

#### General Import Syntax

Specify imports using these formats in the `imports` parameter:

```python
imports=[
    # Import all components from a file in the base directory
    "common_tools.json",
    
    # Import specific components from a file
    "external_agents.json?[research_agent, analysis_tool]",

    # Import components from a file outside the base directory
    "path/to/your/file.json"
]
```

#### Automation-Specific Imports

In automation sequences, directly reference external components using inline syntax:

```python
{
    "type": "automation",
    "name": "complex_flow",
    "sequence": [
        "external.json->research_agent",
        "local_component",
        "path/to/your/other_file.json->some_tool"
    ]
}
```

When importing a component directly in an automation step, the string must be resolved to a single component.

Component names must be unique across all imports and local components. Duplicate names will throw an error.

In all of these cases, the `.json` file extension is optional. The system will try to find a `.json` file with the specified name regardless of whether the extension was included in the string.

### The `mas` Standard Library

The `mas` package includes a Standard Library of tools, processes and functions that you can use out-of-the-box without having to write them yourself. To use them, you just need to import the standard library's components like so:

```json
{
    "general_parameters": {
        "imports": "std"
    }
}
```

You may also import it as `std.py`. In general, any `.json` file you try to import, or any `.py` file you try to use, if not an absolute path and not found in your `base_directory`, will be looked for in the `lib` subfolder. This is where you'll be able to find all standard functions and components. If you don't want to import all components from `std`, you may choose specific ones, just like you do with any other import:

```json
{
    "general_parameters": {
        "imports": "std?[comp1, comp2]"
    }
}
```

### Linking Components

```python
manager.link_tool_to_agent_as_output("mytool", "myagent")  # myagent receives mytool inputs
manager.link_tool_to_agent_as_input("mytool", "myagent")  # myagent's context contains mytool's outputs
manager.link("myagent", "mytool") # automatic link
```

-   `link_tool_to_agent_as_output(tool_name, agent_name)`:  This is the function to call when a tool will be used by an agent. In this case, the agent must be ran before the tool. The mangaer updates the agent's `required_outputs` automatically to include the tool's `inputs`. This configures the Agent to produce the required input for the Tool to execute.
-   `link_tool_to_agent_as_input(tool_name, agent_name)`: This is the function to call when a tool will serve as input to an agent. It ensures the agent's message context includes the tool's output, and ensures the agent's filters don't exclude this tool output, as well as updating the system message of the agent so that it will pay special attention to the tool's output.
-   `link(comp1, comp2)`: If `comp1` is a tool and `comp2` an agent, the tool is linked as input to the agent. If `comp1` is an agent and `comp2` a tool, the agent is linked to provide the tool's input.

The simplest way to link an agent and a tool via JSON is to use the "links" section at the top-level of your JSON. For example:

```json
{
  "components": [
    {
      "type": "agent",
      "name": "myagent",
      "system": "You are an AI assistant."
    },
    {
      "type": "tool",
      "name": "mytool",
      "function": "fn:my_tool_function"
    }
  ],
  "links": {
    "my_tool_using_agent": "my_tool",
    "my_tool_input_for_agent": "my_agent"
  }
}
```

Additionally, if you want an agent to declare which tool it uses directly inside its definition, you can add a `"uses_tool"` key:

```json
{
  "components": [
    {
      "type": "agent",
      "name": "myagent",
      "system": "You are an AI assistant.",
      "uses_tool": "mytool"
    },
    {
      "type": "tool",
      "name": "mytool",
      "function": "fn:my_tool_function"
    }
  ]
}
```

Note that these examples are incomplete. You would need to define the inputs and outputs of the tool, as well as `my_tool_function` in your function file.

### Listing Components

The manager provides several methods to list registered components, which is useful for system introspection or building dynamic user interfaces.

#### Type-Specific Methods

You can list components of a specific type with these straightforward methods:

```python
# List all registered agents
agent_names = manager.list_agents()  # Returns ['hello_agent', 'myagent']

# List all tools
tool_names = manager.list_tools() # Returns ['mytool']
```
*   `list_agents() -> List[str]`
*   `list_tools() -> List[str]`
*   `list_processes() -> List[str]`
*   `list_automations() -> List[str]`

#### Advanced Filtering with `list_components`

For more advanced queries, use the `list_components` method, which allows filtering by type, name substring, or regular expression.

```python
manager.list_components(
    types: Optional[List[str]] = None,
    name_contains: Optional[str] = None,
    regex: Optional[str] = None
) -> List[str]
```
**Examples:**
```python
# List only agents and tools
components = manager.list_components(types=['agent', 'tool'])

# List all components whose name contains 'responder'
responders = manager.list_components(name_contains='responder')

# List tools that follow a versioning pattern (e.g., 'yt_tool_v1', 'yt_tool_v2')
yt_tools = manager.list_components(types=['tool'], regex=r'yt_tool_v\d+')
```

### Running Components

```python
output = manager.run(
    input="Some user input",    # Optional input to add to the message history
    component_name="myagent",  # Optional - component to run
    user_id="user123",        # Optional - which database to use
    role="my_custom_role",      # Optional - role to save the input, overrides defaults
    verbose=True,              # Optional: show debug info
    target_input="myagent",   # Optional - retrieve from a single component/user -> this can contain mas input syntax
    target_index=-1,        # Optional - pick a specific message or range of messages
    target_custom=[          # Optional - pick specific fields from multiple components (overrides target_input and target_index)
        {
           "component": "myotheragent",
            "index": -2,
            "fields": ["summary", "keywords"]
        },
          {
            "component": "user",
            "index": -1
        }
   ],
   blocking=True, # Optional -> defaults to True
   on_complete=None, # Optional -> defaults to defined in system build
   on_update=None, # Optional -> defaults to defined in system build
   on_update_params = {"to_use_in_update": value},
   on_complete_params = {"to_use_in_complete": value},
   return_token_count=False
)
print(output)
```
-   **`input`**: Optional value to store before running the component. By default it is saved as a user message using MAS blocks. Strings are wrapped as `{"response": input}` so agents and bot replies can use the same shape; dictionaries are stored as structured text-block content. Pass `role="internal"` or another custom role if you want to save developer-provided context instead of user input.
-   **`component_name`**: The name of the component to run. If not specified it uses the latest created automation, or creates a linear automation if one does not exist using all components available in their order of creation.
-   **`user_id`**: The ID of the user whose database should be used. If not specified, the current user is used, or created if not set.
-   **`role`**: Role to use when saving the input. Defaults to `"user"` for all input types and can be overridden by the developer, for example with `"internal"` for developer-provided context.
-   **`verbose`**: Boolean to enable verbose mode.
-   **`target_input`**: The name of a component or user to use as input. This parameter can contain `mas input syntax` which will be parsed by the parser, equivalent to just calling `target_custom` (more details below).
-   **`target_index`**: An integer or range to select a message by index in the selected `target_input`.  Negative indices count from the end (e.g., `-1` is the last message). If no index is passed, the component will use a default logic to select message(s). Ranges must be passed as tuples, and the string "~" stands for "all messages".
-   **`target_custom`**: Used to select messages from multiple components, each `dict` in the `list` can specify the following keys:
    -   `component`: The name of the component or `user` to select messages from.
    -   `index`: An integer or range to select a message or messages in the history.
    -   `fields`: An optional list of strings to extract fields from messages.
-   **`blocking`**: Boolean that specifies whether manager.run() should block execution thread until completed or run on an independent thread and return immediately.
-   **`on_complete`**: Callable executed when manager.run() completes, overrides function set when defining the system. Useful for using as callback when blocking=False. 
-   **`on_update`**: Similar to on_complete but runs everytime a component is finished running, useful for automations.
-   **`on_complete_params`**: Dictionary containing values that can be accessed inside on_complete.
-   **`on_update_params`**: Dictionary containing values that can be accessed inside on_update.
-   **`return_token_count`**: Boolean that determines whether the output of components should include usage metadata. When `True`, the `metadata` dictionary in an agent's output will contain `input_tokens`, `output_tokens`, and `usd_cost`. For tools, it will add `usd_cost`. This requires a valid `costs.json` file for cost calculation; otherwise, cost will be `0.0`.

### Running the System in Non-Blocking Mode

The `run` method in the `AgentSystemManager` supports both blocking and non-blocking execution modes, making it flexible for various use cases.

#### **Blocking Mode**

In blocking mode, the `run` method waits for the execution of the specified component to complete before returning the output. This is the default behavior.

```python
result = manager.run(component_name="chat_agent", input="Hello", blocking=True)
print(result)  # Processed output after the component finishes execution
```

#### **Non-Blocking Mode**

In non-blocking mode, the `run` method starts the execution in a separate thread and returns immediately. You can use the `on_update` and `on_complete` callbacks to handle progress and final results.

```python
def on_update(messages, manager, on_update_params):
    if messages:
        latest_message = messages[-1]
        if latest_message.get("source") == "specific_role":
            print("[Update] Latest Message from specific_role:", latest_message["message"])

def on_complete(messages, manager, on_complete_params):
    print("[Complete] Final message:", messages[-1] if messages else "No messages.")

manager.run(
    component_name="chat_agent",
    input="Hello",
    blocking=False,
    on_update=on_update,
    on_complete=on_complete
)
```

Note how you can filter messages by role in the callback methods, so you can perform different actions depending on which component has just been executed.

Both methods must include in their definition the `messages` and `manager` arguments in that order, as they will always be provided when these methods are called by the manager. They can both optionally include the `on_update_params` and `on_complete_params` respectively, although they must be provided as arguments to the `manager.run()` function if they are defined in the callback methods.

#### Best Practices

1. **Use Blocking for Simplicity**: Use blocking mode for simple workflows where immediate results are required.
2. **Enable Real-Time Feedback with `on_update`**: Use `on_update` for tasks that require progress tracking or real-time updates.
3. **Handle Completion with `on_complete`**: Always implement an `on_complete` callback for non-blocking tasks to ensure final results are processed.

### Showing History

```python
manager.show_history(user_id="user123") # Show history of "user123"
manager.show_history()  # Show history of current user
```

-   **`user_id`**: Displays the entire message history for the specified user. If not specified, uses the current user.

### Retrieving Messages: `get_messages`

The `get_messages` method retrieves a structured history of all messages for a specific user, useful for analyzing interactions or displaying chat history.

#### Method Signature

```python
manager.get_messages(user_id: Optional[str] = None) -> List[Dict[str, Any]]
```

#### Parameters

- **`user_id`** *(Optional)*: The unique ID of the user whose history you want. Defaults to the current user if set, or creates a new user UUID otherwise.

#### Return Value

A **list** of dictionaries, each representing a message with the following keys:

- **`source`**: Role that sent the message (e.g., `"user"`, `"agent-name"`, etc.).
- **`message`**: Content of the message (JSON strings are parsed into Python dictionaries).
- **`msg_number`**: Sequential message number in the conversation.
- **`type`**: Component type (`"agent"`, `"tool"`, `"process"`, `"user"`, etc.).
- **`model`**: Model used, in the format `"<provider>:<model-name>"`.
- **`timestamp`**: Timestamp indicating when the message was saved to history, in the specified timezone.
- **`user_id`**: Included when `history_mode="shared"` so consumers can inspect the stored owner of each returned message.

### Reading and Querying Messages: `read()`

While `get_messages` returns the entire history, the `read()` method provides a powerful and flexible interface for querying and extracting specific information from messages.

#### Method Signature
```python
manager.read(
    messages: Optional[Union[Dict, List[Dict]]] = None,
    user_id: Optional[str] = None,
    *,
    source: Optional[Union[str, List[str]]] = None,
    index: Optional[Union[int, tuple]] = -1,
    get_full_dict: bool = False,
    block_type: Optional[str] = None,
    block_index: Optional[Union[int, None]] = None
) -> Any
```

#### Parameters
*   **`messages`**: An optional list of message dictionaries to operate on. If `None`, it uses the history of the specified (or current) `user_id`.
*   **`source`**: Filters messages by one or more `source` roles (e.g., `'user'`, `['agent1', 'tool2']`).
*   **`index`**: Selects messages by index: `-1` for the last message (default), `None` for all messages, or a tuple `(start, end)` for a slice.
*   **`get_full_dict`**: If `True`, returns the complete message dictionary (or list of dictionaries if the `index` returns a slice).
*   **`block_type`**: Filters blocks within the selected messages by their type (`'text'`, `'image'`, `'audio'`, `'video'`, `'document'`, or `'sticker'`).
*   **`block_index`**: Selects a specific block by its index after filtering by `block_type`.

#### Usage Examples

```python
# Get the content of the first block from the last message
# Returns a tuple: (source, block_type, content)
source, b_type, content = manager.read()
print(f"Source: {source}, Content: {content}")

# Get the file path of the last image sent by the user
_, _, img_content = manager.read(source='user', block_type='image', block_index=0)
if img_content:
    print(f"Image path: {img_content.get('path')}")

# Get the last 5 full messages from the 'summarizer' agent
latest_messages = manager.read(source='summarizer', index=(-5, None), get_full_dict=True)
print(f"Found {len(latest_messages)} messages.")

# Get all audio blocks from the entire conversation
# Returns a tuple: (source, list_of_blocks)
source, audio_blocks = manager.read(index=None, block_type='audio')
print(f"Audio blocks: {audio_blocks}")
```

The `read()` method dramatically simplifies accessing specific data from the history, removing the need to manually iterate and filter the output of `get_messages`.

### Deleting User History `clear_message_history`

If you need to clear the message history for a user, the `manager` offers a simple method to delete that user's history. In `per_user` mode this clears the user's database; in `shared` mode it deletes only rows for that `user_id` and leaves other users untouched.

```python
manager.clear_message_history(user_id) # if not provided, it will use the current user_id
```

### Adding Messages: `add_message`

```python
manager.add_message(role: str, content: Union[str, dict], msg_type: str = "developer") -> None
```

The add_message method allows the developer to directly insert a message into the conversation history without triggering any component execution. This is useful for manual updates, debugging, or simulating interactions.

#### Usage Example

```python
manager.add_message(
    role="developer",
    content={"info": "Manual update from developer"}
)
```

### Adding Multimodal Messages: `add_blocks`

The `add_blocks` method is a more powerful version of `add_message` designed to intuitively handle multimodal content. It allows you to insert complex messages composed of text, images, and structured data without needing to manually construct the block-based format.

```python
manager.add_blocks(
    content: Union[str, bytes, Dict[str, Any], List[Any]],
    *,
    role: str = "user",
    msg_type: str = "user",
    user_id: Optional[str] = None
) -> int:
```

This method intelligently processes the `content` parameter:

-   **If `content` is a `str`**:
    -   If the string is a path to a valid image file (e.g., `"./images/photo.jpg"`), an `image` block is created.
    -   Otherwise, a `text` block is created.
-   **If `content` is image `bytes`**: An `image` block is created and the system saves the bytes to a file within the `files/` directory. Unknown byte payloads should be wrapped in a dictionary or explicit block if you want them saved as file references.
-   **If `content` is a `dict`**: The dictionary is stored as structured `text` block content. Non-JSON values are persisted under `files/` and replaced with `file:/...` references.
-   **If `content` is a `list`**: Each element in the list is processed using the same logic described above. This allows you to create multi-part messages (e.g., text followed by an image) in a single call.

The method returns the `msg_number` of the newly created message in the database.

#### Usage Example

```python
# Add a message containing text and an image from a file path
manager.add_blocks(
    content=["Please analyze this image:", "path/to/diagram.png"],
    role="user"
)

# Add an image directly from bytes
with open("photo.png", "rb") as f:
    image_bytes = f.read()

manager.add_blocks(
    content=image_bytes,
    role="user"
)
```

This method is the recommended way to programmatically add multimodal content to a user's history.

### Exporting History `export_history`

```python
sqlite_bytes = manager.export_history(user_id)
```

Exports the SQLite history for the specified user and returns its raw bytes. In
`per_user` mode this is the user's database; in `shared` mode the export contains
only that user's rows. This is useful for creating backups or sending the history
to another storage system.

### Importing History `import_history`

```python
manager.import_history(user_id, sqlite_bytes)
```

Loads a SQLite database from bytes for the given user, overwriting any existing
history for that user. In `shared` mode only that user's rows are replaced.

`manager.has_new_updates()` can then be used as a boolean check to detect if new
messages were added to the current user's history since the last check.

### Retrieving API keys with `get_key`

By default, the API keys file is used to store LLM provider keys, as well as the Bot Token for Telegram integration. However, you can also add any other key or sensitive string that you want to that file, and the manager will save it internally under the name you provide for it. Then, if you need to access it (for example, inside a `Tool` or `Process` function, to access an external API, database or anything else), you can use:

```python
manager.get_key("<your-key-name>")
```

When you call this method, the system searches for the key in the following order: 1) The loaded api_keys.json file, 2) variables from the .env file, and 3) System-wide environment variables.


### Clearing Cache

The `mas` library detects when a tool or a process returns a dictionary with values that are not compatible with JSON serializing. In those cases, those values are saved as pickle objects and then loaded when needed for tools or processes (not for agents, as they can only process text input). Files stay loaded in memory for faster retrieval unless explicitely cleared:

```python
manager.clear_file_cache() # Clears the cache that stores pickle objects.
```

### Getting a system description with `to_string()`

Each component and the manager provide a `to_string()` method that outputs a human‐readable summary of its configuration and current state.

#### Manager `to_string()`

The `AgentSystemManager.to_string()` method returns a formatted string that includes:

- The base directory
- The current user ID
- The general system description
- A summary of all components (agents, tools, processes, automations), including:
  - Their names
  - Descriptions (if provided)
  - Types and key configuration parameters

```python
print(manager.to_string())
```

### Loading from JSON and running the system

Below is a minimal example which runs an automation from a multi agent system defined by a JSON file present in the base directory:

```python
from mas import AgentSystemManager
manager = AgentSystemManager(config="<config_file_name>.json")
output = manager.run(input="Hey, how are you today?")
manager.show_history()
```

For maximum brevity, the whole system can be ran in only one line:

```python
AgentSystemManager(config="<config_file_name>.json").run("Hey, how are you today?")
```

This builds the system from a JSON configuration file specified using the `config` parameter, creates all components if the configuration is valid and runs an automation (either a specified one or a default linear automation) with the provided user input (or starting with no input if none is provided).

### Defining Functions
Functions for `Tool` and `Process` components must be defined in any of the Python files included in the `functions` parameter (or the default `fns.py`). They can also be referenced directly from any other `.py` file as explained earlier. When using the library only programatically and not using function syntax (i.e. `"fn:"`) it is possible to define functions elsewhere and use them as callables directly.

### Running a Chat Loop

The `run` method can be used in a loop to implement a simple interactive chat system. This is ideal for continuously processing user inputs and generating responses from the system.

```python
# Initialize the manager
manager = AgentSystemManager(config="config.json") 

# Chat Loop
while True:
    user_input = input("User: ")
    
    # Break the loop if instructed to by user
    if user_input.lower() == "break":
        break
    
    # Get the output from the manager
    manager.run(input=user_input, verbose=False)
```

To run the chat loop, a valid configuration JSON file is required, such as:

```json
{
  "general_parameters": {
    "base_directory": "/path/to/your/base_dir",
    "general_system_description": "This is a description for the overall system.",
    "functions": "fns.py",
    "on_complete": "fn:on_complete_function"
  },
  "components": [
    {
      "type": "agent",
      "name": "chat_agent",
      "system": "You are an AI assistant.",
      "required_outputs": {
        "response": "Text to send to the user."
      },
      "models": [
        {"provider": "openai", "model": "gpt-4o"}
      ]
    }
  ]
}
```

Note that in this case it is important to define an `on_complete` function, since the main chat loop does not include a specific instruction to handle the system's output once an iteration is completed. A minimal `fns.py` file could look like this:

```python
# fns.py

def on_complete_function(messages, manager):
    # Print the content of the latest message
    if messages:
        latest_message = messages[-1]

        # this prints the full list of blocks returned by the last component
        print("Assistant:", latest_message.get("message", "No message content."))
```

In this function, you could possibly send this message to your own database, to a messaging app or to a custom user interface.

### Speech-to-Text (STT) Functionality

The MAS library includes built-in speech-to-text support via the manager’s `stt` method. This function converts an audio file into text that can be used as input for agents, tools, or processes.

- **Method Signature:**  
  `stt(file_path, provider=None, model=None)`  
  - **file_path:** The local path to the audio file to be transcribed.
  - **provider:** Optionally specify the transcription provider, either `"groq"` or `"openai"`. If not provided, the method automatically selects a provider based on available API keys.
  - **model:** Optionally specify the transcription model. Defaults are `"whisper-large-v3-turbo"` for Groq and `"whisper-1"` for OpenAI.

- **Output:**  
  The method returns the transcribed text as a string.


### Text-to-Speech (TTS) Functionality

The MAS library includes built-in text-to-speech support via the manager’s `tts` method. This function converts input text into a speech audio file (MP3), which can be used to generate audio responses in your multi-agent workflows.

- **Method Signature:**  
  `tts(text, voice, provider=None, model=None, instruction=None)`  
  - **text:** The text to be converted into speech.
  - **voice:** The desired voice name. For OpenAI, this value is used directly. For ElevenLabs, the voice name is used to look up the corresponding voice ID.
  - **provider:** Optionally specify the provider, either `"openai"` or `"elevenlabs"`. If not provided, OpenAI is used by default.
  - **model:** Optionally specify the model. Defaults are `"gpt-4o-mini-tts"` for OpenAI and `"eleven_multilingual_v2"` for ElevenLabs.
  - **instruction:** An optional parameter for OpenAI to control the tone or style of the generated speech.

- **Output:**  
  The generated audio (in MP3 format) is saved in the `files` directory, in a special subfolder named `tts`, with a filename that combines the current user ID and a random hash. The method returns the full path to the saved audio file.

### Tracking Costs with `costs.json`

To enable cost tracking, you can provide a `costs.json` file. When `costs_path` is set in the manager and `return_token_count=True` is used in a `run` call, the system will automatically calculate and attach the USD cost to the metadata of agent and tool outputs.

If the file or a specific price is not found, the cost is assumed to be zero, and the system continues to function without interruption.

Create a `costs.json` file in your `base_directory` with the following structure:

```json
{
  "models": {
    "openai": {
      "gpt-4o-mini":  { "input_per_1m": 0.15, "output_per_1m": 0.60 },
      "gpt-4.1-nano": { "input_per_1m": 0.10, "output_per_1m": 0.40 }
    },
    "groq": {
      "llama-3.1-8b-instant": { "input_per_1m": 0.05, "output_per_1m": 0.05 }
    }
  },
  "tools": {
    "vector_store": { "per_call": 0.0002 },
    "s3_uploader":  { "per_call": 0.0001 }
  }
}
```

*   **`models`**: Prices are nested by `provider` and then `model` name. Prices should be specified per million tokens (`input_per_1m` and `output_per_1m`).
*   **`tools`**: Prices are specified per execution (`per_call`).

### Persistent Usage Logging and Reporting

For production environments or detailed analysis, the MAS library provides a persistent usage logging system that tracks every billable API call.

**Enabling Logging:**
To enable this feature, set `usage_logging=True` when initializing the `AgentSystemManager`. This will create a `logs/` directory in your `base_directory` containing two files:

*   **`usage.log`**: A raw, line-by-line JSON log of every model call, tool call, and STT transcription, including tokens, costs (only for runs where `return_token_count=True`), and timestamps.
*   **`summary.log`**: An aggregated JSON summary of costs and usage across different time spans (hour, day, week, etc.). This file is updated periodically.

**Reporting with `get_cost_report`:**
You can programmatically access the aggregated data using the `get_cost_report` method.

#### Method Signature
```python
manager.get_cost_report(span="lifetime", user=None, model_or_tool=None) -> dict
```
#### Parameters
*   **`span`** (Optional): The time window for the report. Can be `"minute"`, `"hour"`, `"day"`, `"week"`, `"month"`, `"year"`, or `"lifetime"` (default).
*   **`user`** (Optional): Filter the report for a specific user ID.
*   **`model_or_tool`** (Optional): Filter for a specific model (`"openai:gpt-4o"`) or tool (`"my_tool"`).

This method returns a dictionary with the same structure as `summary.log`, allowing for detailed, real-time cost analysis.


### Retrieving Usage Statistics: `get_usage_stats`

The `get_usage_stats` method aggregates and returns a detailed summary of token and cost usage for a given user by reading their entire message history.

#### Method Signature

```python
manager.get_usage_stats(user_id: Optional[str] = None) -> Dict[str, Any]
```

#### Parameters

* **`user_id`** (Optional): The unique ID of the user whose history you want to analyze. Defaults to the current user.

#### Return Value

A dictionary containing a detailed breakdown of costs and token counts, structured as follows:

```json
{
  "models": {
    "<provider>:<model-name>": {
      "input_tokens": 124,
      "output_tokens": 210,
      "usd_cost": 0.0132
    },
    "overall": {
      "input_tokens": 124,
      "output_tokens": 210,
      "usd_cost": 0.0132
    }
  },
  "tools": {
    "<tool_name>": {
      "calls": 5,
      "usd_cost": 0.001
    },
    "overall": {
      "calls": 5,
      "usd_cost": 0.001
    }
  },
  "overall": {
    "usd_cost": 0.0142
  }
}
```


### Bot Integrations (Telegram & WhatsApp)

MAS includes bot adapters for Telegram and WhatsApp Cloud API. Both adapters share the same processing pipeline:

1. A platform update arrives.
2. MAS parses it into normalized message blocks.
3. The blocks are saved as the current user's input message.
4. `manager.run(...)` executes the configured component or automation.
5. The last result, or your callback result, is converted back into platform messages.

Install the optional dependencies for the channel you use:

```bash
pip install "mas[telegram]"
pip install "mas[whatsapp]"
```

#### Telegram Startup

```python
from mas import AgentSystemManager

manager = AgentSystemManager(config="config.json")
manager.start_telegram_bot(
    telegram_token=None,
    component_name=None,
    verbose=False,
    on_update=None,
    on_complete=None,
    speech_to_text=None,
    whisper_provider=None,
    whisper_model=None,
    on_start_msg="Hey! Talk to me or type '/clear' to erase your message history.",
    on_clear_msg="Message history deleted.",
    on_help_msg="Here are the available commands:",
    unknown_command_msg="I don't recognize that command. Type /help to see what I can do.",
    custom_commands=None,
    return_token_count=False,
    ensure_delivery=False,
    delivery_timeout=5.0,
    max_allowed_message_delay=120.0,
    start_polling=True,
)
```

`telegram_token` can be passed directly or stored in the manager API keys as `telegram_token`. If `start_polling=True`, polling starts immediately and blocks the current thread. If `start_polling=False`, the method returns a `TelegramBot` object so you can call direct actions or wire your own webhook server.

```python
manager = AgentSystemManager(config="config.json")
bot = manager.start_telegram_bot(start_polling=False)
bot.start_polling()
```

For Telegram webhooks, keep the bot object and feed raw Telegram update dictionaries into `process_webhook_update`:

```python
bot = manager.start_telegram_bot(start_polling=False)
await bot.initialize()
await bot.process_webhook_update(update_json)
```

#### WhatsApp Startup

```python
from mas import AgentSystemManager

manager = AgentSystemManager(config="config.json")
manager.start_whatsapp_bot(
    whatsapp_token=None,
    phone_number_id=None,
    webhook_verify_token=None,
    component_name=None,
    verbose=False,
    on_update=None,
    on_complete=None,
    speech_to_text=None,
    whisper_provider=None,
    whisper_model=None,
    custom_commands=None,
    return_token_count=False,
    ensure_delivery=False,
    delivery_timeout=5.0,
    max_allowed_message_delay=120.0,
    host="0.0.0.0",
    port=5000,
    base_path="/webhook",
    run_server=True,
)
```

WhatsApp credentials can be passed directly or stored in the manager API keys as `whatsapp_token`, `whatsapp_phone_number_id`, and either `webhook_verify_token` or `whatsapp_verify_token`.

If `run_server=True`, MAS starts a Flask server. Meta webhook verification is handled on `GET base_path`, and message webhooks are handled on `POST base_path`.

If `run_server=False`, the method returns a `WhatsappBot` object:

```python
bot = manager.start_whatsapp_bot(run_server=False)

challenge, status = bot.handle_webhook_verification(request.args)
await bot.process_webhook_update(request_json)
```

`process_webhook_update` expects the normal WhatsApp webhook body containing `entry -> changes -> value -> messages`.

#### Shared Bot Parameters

- `component_name`: component or automation to run for each incoming message. If omitted, MAS uses the default `manager.run()` component resolution.
- `on_update`: optional callback called when the run emits an update.
- `on_complete`: optional callback called when the run finishes. If omitted, MAS sends the last generated message to the user.
- `speech_to_text`: optional callable used for audio transcription. If omitted, MAS calls `manager.stt(...)`.
- `whisper_provider` and `whisper_model`: provider/model used by automatic STT. If no provider is set, `manager.stt()` chooses `groq` when available, otherwise `openai`.
- `custom_commands`: a command definition or list of definitions.
- `return_token_count`: passes `return_token_count=True` into the run.
- `ensure_delivery`: when a callback returns a response, wait for the async send operation to finish before continuing.
- `delivery_timeout`: max seconds to wait when `ensure_delivery=True`.
- `max_allowed_message_delay`: incoming messages older than this many seconds are ignored. Set a larger value if your deployment queues webhooks.
- `verbose`: enables bot logging.

#### Incoming Message Conversion

Text messages become one text block:

```json
[
  {"type": "text", "content": {"response": "hello"}}
]
```

Images, videos, documents, and stickers are downloaded and saved through `manager.save_file(...)`. Captions become a preceding text block.

```json
[
  {"type": "text", "content": {"response": "caption text"}},
  {
    "type": "document",
    "content": {
      "kind": "file",
      "path": "file:/path/to/saved/file.pdf",
      "filename": "file.pdf",
      "mime_type": "application/pdf",
      "detail": "auto"
    }
  }
]
```

Audio and voice messages are saved as `audio` blocks. MAS also tries to add a transcription text block first:

```json
[
  {"type": "text", "content": {"response": "transcribed text"}},
  {
    "type": "audio",
    "content": {
      "kind": "file",
      "path": "file:/path/to/saved/audio.ogg",
      "detail": "auto",
      "is_voice_note": true
    }
  }
]
```

WhatsApp reaction messages are parsed as a text block with the reaction metadata:

```json
[
  {
    "type": "text",
    "content": {
      "response": "\uD83D\uDC4D",
      "message_id": "wamid...",
      "emoji": "\uD83D\uDC4D"
    }
  }
]
```

Telegram reaction updates are not parsed as incoming user messages by the current Telegram adapter. Telegram reactions are supported as outgoing bot actions.

#### User Message Metadata

MAS attaches transport metadata to the first generated block for every incoming user message:

```json
{
  "type": "text",
  "content": {"response": "hello"},
  "metadata": {
    "user_message": {
      "channel": "TelegramBot",
      "user_id": "123456",
      "message_type": "text",
      "timestamp": "2026-05-05T12:00:00+00:00",
      "is_voice_note": false,
      "media_info": {},
      "original_payload": {}
    }
  }
}
```

This metadata is persisted in message history. Use it when later components need to inspect where the message came from. In callbacks, MAS also passes the raw platform payload separately in `params["original_payload"]`.

#### Bot Callbacks

Bot callbacks use the same callback mechanism as `manager.run`, with extra bot parameters:

```python
def on_complete(messages, manager, params):
    user_id = params["user_id"]
    original_payload = params["original_payload"]
    event_loop = params["event_loop"]
    return "Done"
```

`params` contains:

- `user_id`: Telegram chat id or WhatsApp sender id as a string.
- `original_payload`: raw Telegram `Update` object for Telegram, or the raw WhatsApp message dictionary for WhatsApp.
- `event_loop`: the bot event loop. Use it to schedule async direct bot actions from a normal synchronous callback.

Callback return values control what is sent:

- `None`: send nothing.
- `str`: send that text.
- `list` of blocks: send each block as a platform message.
- any other value: convert it with `manager._to_blocks(...)` and send it.

If no `on_complete` callback is provided, MAS sends the latest generated message. It prefers a text block with a `"response"` field. If no `"response"` field exists, it sends the first text block converted to plain text or JSON.

For Telegram, callback output is sent as a reply to the incoming message when using the built-in `_send_blocks` path. For WhatsApp, callback output is sent as a normal outbound message unless you use direct actions with `reply_to_message_id`.

For callback-returned media blocks, `video`, `document`, and `sticker` blocks may use `path`, `url`, `link`, `media_id`, `id`, or `file_id`. Telegram `image` and `audio` blocks are sent from saved `file:`/local paths in the built-in block sender; use direct `send_image(...)` or `send_audio(...)` if you need to send a URL or platform id for those media types.

#### Direct Bot Actions

If you keep the returned bot instance, both bot classes expose the same async direct action interface:

```python
await bot.send_text(user_id, "hello")
await bot.send_image(user_id, "file:/path/to/photo.jpg", caption="Photo")
await bot.send_audio(user_id, "file:/path/to/audio.ogg")
await bot.send_video(user_id, "file:/path/to/video.mp4", caption="Video")
await bot.send_document(user_id, "file:/path/to/report.pdf", filename="report.pdf")
await bot.send_sticker(user_id, "file:/path/to/sticker.webp")
await bot.react_to_message(user_id, message_id, "\U0001F44D")
await bot.remove_reaction(user_id, message_id)
```

All direct send methods accept `reply_to_message_id=None` and arbitrary `**kwargs`. For Telegram, extra kwargs are forwarded to the underlying Telegram bot method. For WhatsApp, extra kwargs are merged into the Cloud API JSON payload.

Media arguments can be:

- a local filesystem path such as `"/tmp/photo.jpg"`;
- a MAS file reference such as `"file:/tmp/photo.jpg"`;
- a public `http://` or `https://` URL;
- a platform media/file id;
- a dictionary containing one of `path`, `url`, `link`, `media_id`, `id`, or `file_id`.

Local WhatsApp media is uploaded first, then sent by uploaded media id. WhatsApp URLs are sent as `link`. Any other non-path value is treated as a WhatsApp media `id`.

#### Sending Stickers

Direct sticker send:

```python
await bot.send_sticker(user_id, "file:/path/to/sticker.webp")
await bot.send_sticker(user_id, "https://example.com/sticker.webp")
await bot.send_sticker(user_id, "platform-media-or-file-id")
```

Sticker block response:

```python
def on_complete(messages, manager, params):
    return [
        {
            "type": "sticker",
            "content": {
                "kind": "file",
                "path": "file:/path/to/sticker.webp"
            }
        }
    ]
```

You can also use `url`, `link`, `id`, `media_id`, or `file_id` in `content`. Stickers do not use captions. MAS forwards the media to the platform, but the platform still decides whether the sticker format is valid.

#### Reacting To Messages

Telegram outgoing reaction:

```python
import asyncio

bot_holder = {}

def on_complete(messages, manager, params):
    bot = bot_holder["bot"]
    update = params["original_payload"]
    chat_id = params["user_id"]
    message_id = update.message.message_id
    loop = params["event_loop"]

    asyncio.run_coroutine_threadsafe(
        bot.react_to_message(chat_id, message_id, "\U0001F44D"),
        loop,
    )
    return "Reacted"

manager = AgentSystemManager(config="config.json")
bot = manager.start_telegram_bot(start_polling=False, on_complete=on_complete)
bot_holder["bot"] = bot
bot.start_polling()
```

WhatsApp outgoing reaction:

```python
import asyncio

bot_holder = {}

def on_complete(messages, manager, params):
    bot = bot_holder["bot"]
    inbound = params["original_payload"]
    user_id = params["user_id"]
    message_id = inbound["id"]
    loop = params["event_loop"]

    asyncio.run_coroutine_threadsafe(
        bot.react_to_message(user_id, message_id, "\U0001F44D"),
        loop,
    )
    return None

manager = AgentSystemManager(config="config.json")
bot = manager.start_whatsapp_bot(run_server=False, on_complete=on_complete)
bot_holder["bot"] = bot
bot.run_server()
```

Remove a reaction by sending an empty reaction:

```python
await bot.remove_reaction(user_id, message_id)
```

For WhatsApp incoming reaction messages, the message being reacted to is available in the user's message content as `message_id`, and the emoji is available as `emoji`.

#### Replying To A Specific Message

Direct actions support replies:

```python
await bot.send_text(user_id, "reply text", reply_to_message_id=message_id)
await bot.send_document(user_id, "file:/tmp/report.pdf", reply_to_message_id=message_id)
await bot.send_sticker(user_id, "file:/tmp/sticker.webp", reply_to_message_id=message_id)
```

For Telegram, this sends `reply_to_message_id` to the Telegram API. For WhatsApp, this adds a Cloud API `context` object with that `message_id`.

#### Commands

Built-in commands:

- `/start`: sends `on_start_msg`.
- `/clear`: clears the current user's message history.
- `/help`: lists commands.

Admin-only commands are registered when `manager.admin_user_id` is set:

- `/clear_all_users`: clears every user's history.
- `/reset_system`: deletes histories, saved files, and logs.
- `/logs`: sends `usage.log` and `summary.log` when usage logging is enabled.

Custom commands:

```python
def check_status(user_id, original_message, manager=None):
    return "System is operational."

custom_commands = [
    {
        "command": "/status",
        "description": "Checks the system status.",
        "function": check_status,
        "message": "System is operational.",
        "admin_only": False,
    }
]

manager.start_telegram_bot(custom_commands=custom_commands)
```

`function` can be a callable or a function reference string such as `"fn:check_status"`. Command functions are called with `user_id` and `original_message`; if the function signature includes `manager`, MAS also passes the manager. If the function returns a string, MAS sends it. If it returns `None`, MAS sends the command's `"message"` fallback if one is defined.

#### Operational Behavior

MAS processes one message per user at a time. If another message from the same user arrives while a run is still active, the new message is ignored. This avoids overlapping writes and mixed histories.

Messages older than `max_allowed_message_delay` seconds are ignored. This protects polling and webhook deployments from replaying old updates.

The bot classes are async internally. Direct actions such as `send_sticker` and `react_to_message` must be awaited from async code, or scheduled with `asyncio.run_coroutine_threadsafe(..., params["event_loop"])` inside synchronous callbacks.


## Multimodal Message Support

The library natively supports **multimodal** messages, allowing agents, tools, and processes to exchange structured “blocks” of text, images, and other data types without additional boilerplate. This section explains how to work with multimodal content.

### Block-based Message Format

The `mas` library normalizes all message content into a list of "blocks". Each block is a dictionary with a `type` and a `content` field.

- **Text block**:  
  ```json
  { "type": "text", "content": "Hello, world!" }
  ```
- **Image block**:  
  ```json
  {
    "type": "image",
    "content": {
      "kind": "file" | "url" | "b64",
      "path": "file:/path/to/files/user_id/1234.jpg",  // for `kind: "file"`
      "url": "https://example.com/pic.jpg",             // for `kind: "url"`
      "b64": "iVBORw0KGgoAAAANS...",                     // for `kind: "b64"`
      "detail": "auto"
    }
  }
  ```
- **Audio block**:  
  ```json
  {
    "type": "audio",
    "content": {
      "kind": "file",
      "path": "file:/path/to/files/user_id/5678.ogg"
    }
  }
  ```
- **Video, document, and sticker blocks** use the same file/url/id shape:
  ```json
  {
    "type": "document",
    "content": {
      "kind": "file",
      "path": "file:/path/to/files/user_id/report.pdf",
      "caption": "Quarterly report",
      "filename": "report.pdf"
    }
  }
  ```

When you pass content to `manager.run()` or `manager.add_blocks()`, the library automatically converts it into this block format. Agents then receive this list of blocks as their input context.

### A Simpler Way to Add Multimodal Content: `add_blocks`

While you can manually build the list of blocks, the library includes the `add_blocks` method, which greatly simplifies this process. This method allows you to add text, images (from file paths or bytes), and dictionaries to the user's history, and it handles the conversion to the correct block format automatically.

It is the recommended way to programmatically insert content into the history.

## Input String Parsing

The input string parser controls which messages from the history are passed into a component. It is most often used in automation sequence steps, but the same input spec can also be used in `target_input`, branch conditions, loop conditions, switch values, and `for` loop item specs.

An input string has one of these shapes:

```json
"executed_component"
"executed_component:input_spec"
":input_spec"
```

Use `"executed_component:input_spec"` inside automation steps, because the string must name the component to run. Use `":input_spec"` when the component is already known, such as in `target_input` or in a condition dictionary.

### Defaults

If no input spec is provided:

- Agents receive the full conversation, after excluding `"internal"` messages and after applying the agent's positive and negative filters.
- Tools receive the latest message as a dictionary. That dictionary is unpacked into the tool function arguments.
- Processes receive the latest message as a one-element message list.
- Conditions like `"field_name"` read `field_name` from the latest message.

### Source Items

A source item selects messages from one component's own message history:

```text
component
component?index
component?start~end
component?[field1, field2]
component?index[field1, field2]
component?start~end[field1, field2]
```

Indexes are zero-based. Negative indexes count from the end. Ranges use Python-style slicing: the start is included and the end is excluded.

```json
"agent:research?-1"
"agent:research?-5~"
"agent:research?~-1"
"agent:research?~"
"agent:research?-3~[answer, confidence]"
```

These mean:

- `research?-1`: the latest `research` message.
- `research?-5~`: the last five `research` messages.
- `research?~-1`: all `research` messages except the latest one.
- `research?~`: all `research` messages.
- `research?-3~[answer, confidence]`: the last three `research` messages, keeping only `answer` and `confidence`.

When the input spec is a single bare source item, its range is local to that component. This is important:

```json
"agent:research?-5~"
```

This gives `agent` the last five messages produced by `research`, regardless of how many messages other components produced between them.

### Multiple Positive Sources

Wrap source items in parentheses to select several sources:

```json
"agent:(research, critic)"
"agent:(research?-1, critic?[summary])"
"agent:(research?-3~[answer], critic?-1[verdict])"
```

The result is merged back into chronological order.

For agents and processes, a source item without an explicit index means all messages from that source. For tools and conditions, a source item without an explicit index means the latest message from that source, because their result is a single dictionary by default.

Tools and conditions merge selected messages into one dictionary in chronological order. If two selected messages contain the same field, the later message overwrites the earlier one.

### Timeline Selectors

Sometimes you want to select by the whole conversation timeline first, instead of by each component separately. Timeline selectors do that:

```json
"agent:*"
"agent:*?-5~"
"agent:*!(debug_logger, internal_tool)?-20~"
"agent:(research, critic)?planner?-2~"
"agent:(research?-1[answer], critic?[summary])?planner?-2~"
```

The selector part can be:

- `*`: all messages in the current timeline.
- `*!(component_a, component_b)`: all messages except messages from those components.
- `(source_item, source_item, ...)`: only the listed source items.

The optional range after the selector is global. It is applied to the full chronological conversation before the selector's source-specific filters are applied.

This rule is the main distinction between bare and parenthesized single-component specs:

```json
"agent:research?-5~"
"agent:(research)?-5~"
```

The first line means: take the last five `research` messages.

The second line means: take the last five messages in the whole conversation, then keep only the messages from `research` among those five.

### Global Ranges

A global range follows a timeline selector and starts with `?`:

```json
"agent:*?-5~"
"agent:*?~-1"
"agent:*?-1"
"agent:(research, critic)?planner?-2~"
"agent:(research, critic)?planner~"
```

Global numeric endpoints use the whole conversation:

- `*?-5~`: last five messages in the whole conversation.
- `*?~-1`: all messages except the latest global message.
- `*?-1`: only the latest global message.

Global endpoints can also be anchors to a component message:

- `planner?-2~`: start at the penultimate `planner` message and continue to the end.
- `planner~`: start at the latest `planner` message and continue to the end.
- `planner?-3~planner?-1`: start at the third-to-last `planner` message and stop before the latest `planner` message.

Anchor endpoints identify one message in the global timeline. They can use a component name plus a single integer index. They cannot include fields or ranges. If an anchor does not match any message, the selected global window is empty.

### Evaluation Order

MAS evaluates an input spec in this order:

1. Start from chronological history. For agents, apply the agent's positive and negative filters first.
2. If the input spec has a global range, cut the global timeline to that window.
3. Apply the selector: all messages with `*`, all except named components with `*!(...)`, or only the parenthesized source items with `(...)`.
4. For parenthesized source items, apply each source item's local index, local range, and field filter inside the already-cut global window.
5. Return the selected messages in chronological order. Tools and conditions merge selected dictionaries chronologically, so later fields win on key conflicts.

For example:

```json
"agent:(research?-1[answer], critic?[summary])?planner?-2~"
```

This means:

1. Find the penultimate `planner` message in the whole conversation.
2. Keep only the conversation from that message onward.
3. Inside that window, select the latest `research` message and keep only `answer`.
4. Inside that same window, select `critic` messages and keep only `summary`.
5. Pass the surviving messages to `agent` in chronological order.

### Component Behavior

Agents receive a conversation. Selected messages keep their chronological order, and messages from other components are represented as user-side messages with a source block so the agent can tell where each message came from.

Processes receive a list of dictionaries:

```json
[
  {
    "source": "research",
    "message": [{"type": "text", "content": {"answer": "..."}}],
    "type": "component",
    "timestamp": "..."
  }
]
```

Tools receive one dictionary. The selected messages are converted to dictionaries and merged in chronological order:

```json
"tool:(retriever?[items], ranker?[ranking])"
```

This passes a single dictionary containing fields from the latest `retriever` message and the latest `ranker` message, filtered to `items` and `ranking`.

Conditions use the same input syntax when the condition string starts with `:`:

```json
":state?[ok]"
":*!(debug_logger)?-3~"
":(planner?[should_continue], validator?[is_valid])?planner?-2~"
```

A string condition evaluates the truthiness of the selected value or values. A dictionary condition compares the selected value with `"value"`:

```json
{
  "input": ":*!(debug_logger)?-2~",
  "value": {"ok": true}
}
```

### Examples

```json
"summarizer:*?-5~"
```

Run `summarizer` with the last five messages in the whole conversation.

```json
"writer:(research, critic)?planner?-2~"
```

Run `writer` with `research` and `critic` messages from the penultimate `planner` message onward.

```json
"writer:(research?-1[answer], critic?-1[verdict])?planner?-2~"
```

Run `writer` with the latest `research.answer` and latest `critic.verdict` inside the global window that starts at the penultimate `planner` message.

```json
"cleanup_tool:*!(debug_logger, trace_collector)?-20~"
```

Run `cleanup_tool` with a dictionary merged from the last twenty global messages, excluding messages produced by `debug_logger` and `trace_collector`.

```json
"audit_process:(retriever?~[items], ranker?-1[ranking])?planner~"
```

Run `audit_process` with all `retriever.items` and the latest `ranker.ranking` from the latest `planner` message onward.

```json
"agent:research?-5~"
```

Run `agent` with the last five `research` messages.

```json
"agent:(research)?-5~"
```

Run `agent` with `research` messages that appear inside the last five messages of the whole conversation.

The compact forms keep their usual meaning, and timeline selectors let you express conversation-wide windows, component anchors, and negative component filters without changing the rest of the automation format.

## Error Handling

The library is designed for robust operation, handling various errors gracefully. If an agent fails to produce a response, it will return a `default_output`. If the tool or process fails, the system will return the `default_output`, or an empty dict if no `default_output` was specified. Errors in configuration files or function references are logged for debugging.

## Design Patterns and Recommendations

The `mas` library is designed to be flexible and versatile, and you can use it to fit your own specific needs and requirements. However, certain patterns are commonly repeated and are worth mentioning here for beginners who are learning to build multi agent systems effectively.

### Model Selection Notes

Model availability, pricing, and free tiers change frequently, so treat the examples below as starting points and verify current provider dashboards before production use.

-  **Fast development/testing**: Small Groq, Google Gemini Flash, NVIDIA NIM, or OpenRouter free-tier models are useful for quick iteration when their rate limits fit your workflow.
-  **Low-cost production**: DeepSeek, Gemini Flash, and low-cost OpenAI/OpenRouter models are often good candidates for routine structured-output agents.
-  **Reasoning-heavy workflows**: Use stronger reasoning models only on the steps that need them. MAS makes this practical because each agent can define its own model list.
-  **Local development**: LM Studio can be used through the `lmstudio` provider with a local OpenAI-compatible server and a mock key in your API key file.
-  **Provider fallback**: Put your best-case model order first. MAS preserves that order when providers are healthy, and the manager-wide model failure policy temporarily avoids models with recent failures.

### Design Patterns

Multi-agent systems can usually be thought of as workflows where you can combine decision-making nodes which allow for branching and looping, tool-use nodes which allow for interaction with internal or external information, data-processing nodes which load, transform or export information to usable formats, and user-interaction nodes which allow the system to send outputs to the user.

#### Decision-making Agents

Very commonly, you'll want to decide whether to follow different workflows depending on what the user is asking for. Decision-making agents are perfect for this use case.

```json
{
  "components": [
    {
      "type": "agent",
      "name": "decider",
      "system": "Determine if user request requires an API call or not.",
      "required_outputs": {
        "is_api_needed": "Boolean, determines whether an API call is needed or not."
      },
      "default_output": {
        "is_api_needed": false
      }
    },
    {
      "type": "automation",
      "sequence": [
        "decider",
        {
          "control_flow_type": "branch",
          "condition": "is_api_needed",
          "if_true": ["api_using_agent", "api_tool"],
          "if_false": []
        }
      ]
    }
  ]
}
```

#### Action Switching

Sometimes, you want a single decision-making agent to determine one of several actions to take later. Here's where switching on an action can be useful.

```json
{
  "components": [
    {
      "type": "agent",
      "name": "selector",
      "system": "Determine if the user is asking for weather information, for news information, or just making conversation.",
      "required_outputs": {
        "action": "Action to take given user message. Either 'weather', 'news', or 'other'."
        },
      "default_output": {
        "action": "other"
      }
    },
    {
      "type": "automation",
      "sequence": [
        "selector",
        {
          "control_flow_type": "switch",
          "value": "action",
          "cases": [
            {"case": "weather", "body": ["tool_using_agent", "weather_tool"]},
            {"case": "news", "body": ["other_tool_using_agent", "news_tool"]},
            {"case": "other", "body": []}
          ]
        },
        "conversation_agent"
      ]
    }
  ]
}
```

#### Loops and Verifiers

When you're implementing a complex workflow, it's usually good practice to check whether the result accomplishes the task, or if more work is required. This can sometimes be done programmatically by using processes (e.g. did the API call work or did it return an error?) but other times an intelligent node is needed to analyze the result and decide whether the workflow can end or if we need to try again. This can be done by combining a loop and a verifier agent.

```json
{
  "components": [
    {
      "type": "agent",
      "name": "verifier",
      "system": "Determine if the result from the system is sufficient to fulfill the user's request. If you decide it is not, more work will be done.",
      "required_outputs": {
        "is_result_sufficient": "Boolean, system responds to user if true, system keeps working if false."
        },
      "default_output": {
        "is_result_sufficient": true
      }
    },
    {
      "type": "automation",
      "sequence": [
        {
          "control_flow_type": "while",
          "end_condition": ":verifier?-1?[is_result_sufficient]",
          "body": [
            "some_workflow_steps",
            "verifier"
          ]
        }
      ]
    }
  ]
}
```

#### First Responders and Final Responders

Long workflows can take time, especially when they involve calls to reasoning agents or interactions with external APIs. In these cases, it's good practice to inform the user a process will be taking place by using a first responder agent. Even when this is not the case, almost all chat-based workflows will include a final responder, or conversation agent, which will talk to the user based on available information. Both of these responder-type agents can be handled in the `on_update` function.

```json
{
  "components": [
    {
      "type": "agent",
      "name": "first_responder",
      "system": "Inform the user you will do a search for information and will have results shortly.",
    },
    {
      "type": "agent",
      "name": "final_responder",
      "system": "Respond to the user message given available information.",
    },
    {
      "type": "automation",
      "sequence": [
        "first_responder",
        "some_process_steps",
        "final_responder"
      ]
    }
  ]
}
```

```python
  def on_update_function(messages, manager):
      last_message = messages[-1]
      if last_message["source"] == "first_responder" or last_message["source"] == "final_responder":
          response = last_message["message"]["response"]

          # do something with the response, or return it if you're working with telegram integration
          # for this example, we'll assume we're running a terminal-based chat app and print it

          print(response)
```

#### Message History Filtering

Agents get the full conversation history by default. This is usually unnecessary, especially when using nodes such as decision-making agents or verifiers, which produce outputs that are required for the workflow but often don't need to be seen by other agents. It's important to manage input filtering carefully to save on input token usage when calling LLM providers. This has a profound impact on the multi-agent system cost, as well as its overall effectiveness.

```json
{
  "components": [
    {
      "type": "automation",
      "sequence": [
        "decider:user?-1",
        "some_steps_in_between",
        "final_responder:(user, final_responder)"
      ]
    }
  ]
}
```

Notice how the decider agent may only need, in this example, the last user message, while the final responder may only need user and responder messages, and can ignore decider and intermediate steps. This will usually be more complex and will depend on your specific use case.

Similarly, it's usually good practice to add range filtering to take only the last n messages from the conversation history. This is crucial when building chat-based systems where users can interact with them indefinitely, increasing input token usage without limit if not handled carefully by the developer.

```json
{
  "components": [
    {
      "type": "automation",
      "sequence": [
        "some_steps",
      ]
        "final_responder:(user?-20~, final_responder?-20~)"
    }
  ]
}
```

In this case, the final responder is looking at the last 20 messages from the user and from itself, thereby restricting maximum conversation length. More complex workflows would involve a memory bot which is triggered when the conversation exceeds a certain length, summarizes part of it or all of it, and allows for the `final_responder` agent to always know what happened previously even if it does not have direct access to all messsages.


## Currently Under Development

This is an alpha version of the `mas` library. It has not yet been tested extensively and likely contains many bugs and undesired behavior. Its use on production is **NOT RECOMMENDED**.


# MAWS: 1-Command AWS Deploys for Telegram & WhatsApp Bots (Beta)

`maws` is a tiny companion to the `mas` library that bootstraps a **serverless bot backend** on AWS (API Gateway + Lambda + S3, with optional DynamoDB locks, SQS FIFO processing, S3 file persistence, and SSM Parameter Store). It ships with:

* A **runtime** (`MawsRuntime`) that turns your MAS config into a webhook-driven bot (Telegram or WhatsApp).
* A **CLI** (`maws`) that scaffolds a project, wires AWS resources, and deploys with AWS SAM.
* Python helpers mirroring the CLI, if you prefer scripting over shell.

> Works best on Linux/macOS or Windows Subsystem for Linux (WSL). Native Windows is supported with a guard/override (see flags below).

---

### Fastest Possible Start (3 steps)

```bash
# 0) Install MAS (includes MAWS)
pip install --upgrade git+https://github.com/mneuronico/multi-agent-system-library

# 1) Scaffold a bot project (interactive; chooses provider/model & bot type)
maws start

# 2) Deploy to AWS (builds & uploads a complete stack)
maws update
```

After `maws update` finishes, if you chose to create a WhatsApp bot, in your Meta App, set the webhook URL to <ApiUrl> and the verify token to WHATSAPP_VERIFY_TOKEN (follow instructions in console).

That’s it — send a message to your bot. Conversation history is automatically persisted to S3.

---

### What the scaffold creates

Inside your chosen folder:

```
params.json     # AWS deploy settings (project, region, stack & buckets, etc.)
config.json     # Minimal MAS config with a simple agent (edit as you like)
fns.py          # Where your tools/processes live
.env.prod       # Provider keys & bot tokens (stored in AWS SSM at deploy time)
.samignore      # SAM build ignores
```

On AWS, the stack includes:

* **API Gateway** (public HTTPS endpoint) → **Lambda** (your bot runtime)
* **S3** buckets: one for **history** (per-user SQLite DB) and one for **deployment artifacts**
* **SSM Parameter Store (SecureString)** to hold your `.env` securely
* **(Optional)** DynamoDB table for **user locks** (prevents concurrent processing by multiple invocations)
* **(Optional)** SQS FIFO queue and DLQ for durable, ordered bot work

---

## How MAWS works at runtime

* Loads secrets from **SSM** (`ENV_PARAMETER_NAME`) into `os.environ` on cold start.
* Optionally **syncs “token files”** (API tokens, etc.) from **S3** into `/tmp` and exposes their paths via env vars.
* Optionally verifies Telegram POSTs with `X-Telegram-Bot-Api-Secret-Token` and WhatsApp POSTs with `X-Hub-Signature-256`.
* Optionally syncs user media/files to S3 when `persist_files_s3=true`.
* On each webhook:

  1. Normalizes provider payloads into one job per processable message.
  2. Either self-invokes immediately (`busy_policy="drop"`, default) or enqueues to SQS FIFO (`busy_policy="fifo"`).
  3. **(Optional)** acquires a **DynamoDB distributed lock** for that user, or one global lock in shared-history mode.
  4. Downloads history from S3 and **imports** it into the MAS manager.
  5. Feeds the update to **TelegramBot**/**WhatsappBot** (webhook mode).
  6. **Exports** the updated history and optional files back to S3 and releases the lock.

You don’t have to wire any of this — the CLI’s deploy scripts generate all infra and environment linkage.

---

## Prerequisites

* AWS account with credentials (maws start will install aws, sam and run aws config if necessary)
* For Telegram: `TELEGRAM_TOKEN`
  For WhatsApp Cloud API: `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`

---

## CLI Reference

The CLI ships as `maws`. All commands can also be called from Python (see next section).

### 1) `maws start`

Scaffolds a new project (interactive by default).

```bash
maws start \
  [--project <name>] \
  [--region <aws-region>] \
  [--bot telegram|whatsapp] \
  [--dir <path>] \
  [--overwrite] \
  [--install-deps|--no-install-deps] \
  [--run-config|--no-run-config] \
  [--allow-windows]
```

* Creates `params.json`, `config.json`, `fns.py`, `.env.prod`, `.samignore`.
* Asks for default **LLM provider/model** and **bot type**.
* Optionally installs system deps (AWS CLI, SAM) and runs `aws configure`.

**Tip (non-interactive):**

```bash
maws start --project my-bot --region us-east-1 --bot telegram --no-install-deps --no-run-config
```

---

### 2) `maws update`

Builds & deploys the serverless stack via AWS SAM.

```bash
maws update \
  [-c|--config params.json] \
  [--dir <path>] \
  [--force-copy-script] \
  [--quiet] \
  [--allow-windows] \
  [--project <name>] [--region <region>] [--bot telegram|whatsapp] \
  [--set key=value ...]        # override values in params.json on the fly
```

* Uses `params.json` (and overrides) to deploy.
* Persists `.env.prod` to **SSM** as a secure parameter automatically.
* Prints the **ApiUrl** output on success (use it to set your WhatsApp webhook if necessary).

---

### 3) `maws pull-history`

Downloads your per-user SQLite histories from the **history bucket** into `./history`. Without filters it syncs every history file. Use `--user-id` repeatedly or `--user-ids` with a comma-separated list to download only specific users.

```bash
maws pull-history [-c|--config params.json] [--dir <path>] [--force-copy-script] [--quiet] [--user-id <id>] [--user-ids <id,id>]
```

---

### 4) `maws setup`

Interactive helper to **create or refine** `params.json` safely.

```bash
maws setup [-c|--config params.json] [--dir <path>]
```

Prompts for:

* `project`, `region`, `bot`
* `stack_name`, `history_bucket`, `deployment_bucket`
* `env_param_name` (SSM), `api_path` (API Gateway route), verbosity & token sync prefs

---

### 5) `maws describe`

Shows local files, prints `params.json`, and (if permitted) fetches **CloudFormation outputs** (e.g., **ApiUrl**), bucket existence, and stack status.

```bash
maws describe [-c|--config params.json] [--dir <path>] [--region <region>] [--no-aws]
```

---

### 6) `maws list`

Lists **MAWS stacks** in a region (quick inventory with ApiUrls).

```bash
maws list [--region <region>]
```

---

### 7) `maws remove`

Tears down a project’s AWS resources.

```bash
maws remove \
  [--project <name>] \
  [--region <region>] \
  [-y|--yes] \
  [--keep-deploy-bucket] \
  [--wait] \
  [-c|--config params.json] \
  [--dir <path>]
```

* Empties the **history bucket** and deletes the **CloudFormation stack**.
* Optionally deletes the **deployment bucket** (`--keep-deploy-bucket` to preserve artifacts).
* `--wait` blocks until the stack is fully deleted.

---

## Python API (equivalent helpers)

Everything the CLI can do is also available as Python functions:

```python
from maws import start, update, pull_history, setup, describe, list_projects, remove_project

# Scaffold
start(project="my-bot", region="us-east-1", bot="telegram", install_deps=False, run_config=False)

# Deploy (uses params.json in cwd)
update()

# Pull conversation DBs from S3
pull_history()

# Pull only selected users
pull_history(user_ids=["telegram-chat-id", "whatsapp-user-id"])

# Interactive param refinement (or use programmatic edits to params.json yourself)
setup()

# Print local + AWS state (0 = OK, nonzero = issues)
describe()

# Discover MAWS stacks in a region
list_projects(region="us-east-1")

# Tear down (DANGER)
remove_project(project="my-bot", region="us-east-1", yes=True, wait=True)
```

Return codes mirror the CLI behavior (`0` on success, non-zero on error where applicable).

---

## Configuration & Environment

### `params.json` keys (managed by `maws setup/start`)

* `project`, `region`, `bot` (`telegram`|`whatsapp`)
* `stack_name` – CloudFormation stack name
* `history_bucket` – stores per-user SQLite histories
* `deployment_bucket` – SAM artifacts
* `env_param_name` – SSM parameter path for **.env.prod** (stored encrypted)
* `api_path` – API Gateway route path (default `/webhook`)
* `sync_tokens_s3` (bool) – sync “token files” from S3 into `/tmp`
* `tokens_s3_prefix` – S3 prefix for token files (default `secrets`)
* `verbose` (bool) – extra logging in Lambda

### Production-oriented MAWS params

Optional production params include `busy_policy` (`drop` by default, `fifo` for SQS FIFO), `persist_files_s3`, `files_s3_prefix`, `history_mode`, `history_rotation`, `history_max_messages`, `history_period`, `webhook_security`, `runtime.manager_kwargs`, `runtime.bot_kwargs`, `failure_handling`, `infra`, `requirements_source`, and `requirements_ref`.

Security uses only provider-supported mechanisms: Telegram webhook secret tokens and WhatsApp `X-Hub-Signature-256` verification. These are opt-in so existing projects do not break, but production bots should enable the relevant provider mechanism.

### Lambda environment variables (read by runtime)

* `ENV_PARAMETER_NAME` – SSM parameter name that contains your `.env` lines
* `BUCKET_NAME` – history bucket name (set by the deploy)
* `BOT_TYPE` – `telegram` or `whatsapp` (deploy sets this; defaults to `whatsapp`)
* `SYNC_TOKENS_S3` – `"1"/"0"` (default on)
* `TOKENS_S3_PREFIX` – S3 prefix for token files (default `secrets`)
* `LOCKS_TABLE_NAME` – DynamoDB table name (optional; enables user locks)
* `LOCK_TTL_SECONDS` – seconds for the lock TTL (default `180`)
* `SPECIAL_TOKEN_FILES_JSON` – JSON list of filenames to fetch to `/tmp` (optional)
* `TOKEN_ENV_MAP_JSON` – JSON map `{ "file.ext": "ENV_VAR_NAME" }`; MAWS sets `ENV_VAR_NAME=/tmp/file.ext`
* `VERBOSE` – extra logs (`true/false`)

### Provider & bot secrets (in `.env.prod`, then moved to SSM)

Common entries:

```
# LLMs (put at least one)
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
LMSTUDIO_API_KEY=
NVIDIA_API_KEY=

# Telegram (if using Telegram)
TELEGRAM_TOKEN=
TELEGRAM_WEBHOOK_SECRET_TOKEN=

# WhatsApp Cloud API (if using WhatsApp)
WHATSAPP_TOKEN=
WHATSAPP_PHONE_NUMBER_ID=
WHATSAPP_VERIFY_TOKEN=
WHATSAPP_APP_SECRET=
```

> MAWS loads this env securely from **SSM** at runtime (`ENV_PARAMETER_NAME`).
> To rotate secrets, update `.env.prod` locally and redeploy (`maws update`), or write the SSM parameter directly.

---

## Troubleshooting

* **Windows**: Prefer **WSL**. If you must run natively, add `--allow-windows`, but this won't necessarily play nice with the Lambda environment.
* **Missing AWS/SAM/jq**: Re-run `maws start` on Linux/WSL, or install manually.
* **Telegram webhook not set**: Use the `setWebhook` call shown above with your **ApiUrl** (although it should be set automatically).
* **WhatsApp verification fails**: Ensure your **webhook URL** is exactly the printed **ApiUrl** and **WHATSAPP\_VERIFY\_TOKEN** matches the Meta dashboard value.
* **403 / 401 from LLM provider**: Double-check your provider API key is present (SSM) and the model name in `config.json` is valid.
* **No messages saved**: Verify the **history bucket** exists (`maws describe`) and Lambda has write permissions (the provided stack template handles this automatically).

---

**Status:** MAWS is in **beta**. It’s designed to be safe and ergonomic, but please report issues and ideas — and don’t hesitate to peek into `config.json`/`fns.py` to evolve your bot beyond the minimal scaffold.
