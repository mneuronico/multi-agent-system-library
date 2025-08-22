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
- **Seamless LLM Integration**: Manage interactions with multiple LLM providers (e.g., OpenAI, Google, Groq, Anthropic, DeepSeek, and local models via LM Studio) without adapting your code or message history for each one.
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

-   Python 3.7+
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
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
LMSTUDIO_API_KEY=your_mock_lmstudio_key
```

Provider names can be defined as `<PROVIDER>-API-KEY`, `<PROVIDER>`, `<PROVIDER>-KEY`, `<PROVIDER>_KEY`, and other similar variations (the names are handled as case-insensitive).

To use a `json` file to define API keys, you can do something like:

```json
{
    "openai": "your-openai-key",
    "groq": "your-groq-key",
    "google": "your-google-key",
    "anthropic": "your-anthropic-key",
    "deepseek": "your-deepseek-key",
    "lmstudio": "your-mock-lmstudio-key"
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

## Fast Track: Build a System from Plain Text

If you just have an idea and don’t feel like writing JSON yet, pass the idea straight into the `config` parameter.  

The manager will:

1.  Read this very README to learn about the MAS framework.  
2.  Spin up an *internal* agent called **`system_writer`** (running on the best
    available model chain – `o3`, `gemini-2.5-pro`, `claude-4`, `deepseek-r1`,
    `llama-4@groq`).
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

All interactions between components are recorded in a per-user SQLite database. This provides a complete, chronological record of the conversation or workflow. Each message in the history table includes:

-   A unique identifier (`id`).
-   A sequential message number (`msg_number`), which ensures chronological order.
-   The `type` of the component that was run (e.g., `agent`, `tool`, `process`, `user`, `iterator`).
-   The `model` that was called if `type` is `agent` (in the format `'provider:model_name'`).
-   A `role` that indicates the name of the component that produced the message (e.g., `hello_agent`, `my_tool`).
-   The `content` of the message, which is stored as a JSON string representing a **list of blocks**. This is the core of the data exchange system.
-   The `timestamp` indicating when the message was added, in the specified timezone (defaults to `UTC`).

Each `user_id` has its own isolated message history, allowing the manager to handle multiple concurrent and independent sessions.

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

#### Automatic Conversion to Blocks (`_to_blocks`)

You don't need to worry about creating these structures yourself. When you call `manager.run(input=...)` or `manager.add_blocks(content=...)`, the library automatically converts your data into the appropriate block format based on these rules:

1.  **If you provide a valid list of blocks**: It's used as-is.
2.  **If you provide a mixed list** (e.g., `['Analyze this:', './img.png']`): Each element is processed individually and the resulting blocks are combined into a single list.
3.  **A `str`**:
    - If the string is a path to a valid local image file, it's read, saved internally, and converted into an `image` block.
    - Otherwise, it's treated as plain text and becomes a `text` block.
4.  **`bytes` or `bytearray`**: Assumed to be image or audio data. The bytes are saved to a file in the `files/` directory and wrapped in an `image` or `audio` block.
5.  **A `dict`**: The object is serialized to a JSON string and placed inside a `text` block. If the object contains non-serializable data (like custom class instances), each such value is saved as a `.pkl` file and replaced with a `file:/...` reference.

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
    usage_logging=False
)
```

The `AgentSystemManager` manages your system’s components, user histories, and general settings.

-   **`config`**: Path to a JSON file or a plain-English description (triggers automatic bootstrap of system).
-   **`base_directory`**: Specifies the directory where user history databases (`history` subdirectory) and pickled object files (`files` subdirectory) are stored. Also the location of `fns.py`.
-   **`history_folder`**: Path for storing per-user SQLite databases. Defaults to `<base_directory>/history`.
-   **`files_folder`**: Path for storing serialized object files. Defaults to `<base_directory>/files`.
-   **`api_keys_path`**: Path to a `.env` or `json` file containing API keys. Keys from this file are given priority over system environment variables, making it easy to manage keys for different environments.
-   **`costs_path`**: Path to a `.json` file containing model and tool costs. If provided, the system will calculate and store the USD cost for agent and tool executions when `return_token_count` is set to `True`.
-   **`general_system_description`**: A description appended to the system prompt of each agent.
-   **`functions`**: The name of a Python file, or list of Python files, where function definitions must be located. Files must either exist in the base directory or be referenced as an absolute path. If not defined, this defaults to `fns.py` inside the `base_directory`.
-   **`default_models`**: A list of models to use when executing agents, for agents that don't define specific models. Each element of the list should be a dictionary with two fields, `provider` (like 'groq' or 'openai') and `model` (the specific model name). These models will be tried in order, and failure of a model will trigger a call to the next one in the chain.
-   **`imports`**: List of component import specifications. Each entry can be either `"<your_json>.json"` to import all components from that file, or `"<your_json>.json?[comp1, comp2]"` to import specific components from that file.
-   **`on_update`**: Function to be executed each time an individual component is finished running. The function must receive a list of messages and the manager as the only two arguments. Useful for doing things like updating an independent database or sending messages to user during an automation.
-   **`on_complete`**: Function to be executed when `manager.run`() reaches completion. This is equivalent to `on_update` when calling `manager.run()` on an individual component (if both are defined, both will be executed), but it's different for automations, since it will only be ran at the end of the automation. The function must receive a list of messages and the manager as the only two arguments. Useful for doing things like sending the last message to the user after a complex automation workflow.
-   **`include_timestamp`**: Whether the agents receive the `timestamp` for each message in the conversation history. False by default. This is overriden by the `include_timestamp` parameter associated with each agent, if specified. 
-   **`timezone`**: String defining what timezone should the `timestamp` information be saved in. Defaults to `UTC`.
-   **`log_level`**: Logging level for the library. Defaults to `logging.DEBUG`.
-   **`admin_user_id`**: A specific user ID (e.g., your Telegram chat ID) granted access to administrative commands.
-   **`usage_logging`**: If `True`, enables the persistent usage and cost logging system. Defaults to `False`.

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
        "common_tools.json",
        "external_agents.json->research_agent+analysis_tool"
    ],
    "on_update": "fn:on_update_function", 
    "on_complete": "fn:on_complete_function",
    "include_timestamp": true,
    "timezone": "America/Argentina/Buenos_Aires"
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
    Supported providers so far are: `"openai"`, `"google"`, `"groq"`, `"anthropic"`, `"deepseek"`, and `"lmstudio"`. Ensure the corresponding `api_key` is available in your API key file. LM Studio models can optionally include a `base_url` in the model dictionary when connecting to a non-default server.
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
-   **`input`**: Optional string or dict to store in the message history. If it's a string, it will be stored with the role `"user"`. If it's a dictionary, it will be stored with the role `"internal"`, assumed to be information added by the developer.
-   **`component_name`**: The name of the component to run. If not specified it uses the latest created automation, or creates a linear automation if one does not exist using all components available in their order of creation.
-   **`user_id`**: The ID of the user whose database should be used. If not specified, the current user is used, or created if not set.
-   **`role`**: Role to use when saving the input, defaults to `"user"` if the input is a string, or `"internal"` if input is a dictionary, but can be overriden by developer.
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
manager.get_messages(user_id: Optional[str] = None) -> List[Dict[str, str]]
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

### Deleting User History `clear_message_history`

If you need to clear the message history for a user, the `manager` offers a simple method to delete the database associated to a specific user.

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
    user_id: Optional[str] = None,
    detail: str = "auto",
    verbose: bool = False
) -> int:
```

This method intelligently processes the `content` parameter:

-   **If `content` is a `str`**:
    -   If the string is a path to a valid image file (e.g., `"./images/photo.jpg"`), an `image` block is created.
    -   Otherwise, a `text` block is created.
-   **If `content` is `bytes`**: It is assumed to be image data, and an `image` block is created. The system will save the bytes to a file within the `files/` directory.
-   **If `content` is a `dict`**: The dictionary is serialized to a JSON string and stored as a `text` block. If the dictionary contains non-serializable values, they are persisted as `.pkl` files and referenced internally.
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

Exports the SQLite database for the specified user and returns its raw bytes. This
is useful for creating backups or sending the history to another storage system.

### Importing History `import_history`

```python
manager.import_history(user_id, sqlite_bytes)
```

Loads a SQLite database from bytes for the given user, overwriting any existing
history file.

`manager.has_new_updates()` can then be used as a boolean check to detect if new
messages were added to the current user's history since the last check.

### Retrieving API keys with `get_keys`

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


### Telegram Integration

The `mas` library allows the developer to integrate any system with Telegram seamlessly to allow users to interact with the system through the messaging app without requiring the developer to define custom async logic and event loops. This is possible through the `start_telegram_bot` method:

```python
manager.start_telegram_bot(
  telegram_token = None, # if not provided, the manager looks for it in its API keys
  component_name = "my_automation", # optional, defaults to default or latest automation
  verbose = False, # defaults to False
  on_complete = None, # defaults to sending latest message to user
  on_update = None, # defaults to no operation
  whisper_provider=None, # 'groq' and 'openai' are supported
  whisper_model=None, # defaults to v3-turbo in groq and v2 in openai
  speech_to_text=None, # optional callable if you need to process your audios and voice notes in a custom way
  on_start_msg = "Hey! Talk to me or type '/clear' to erase your message history.", # defaults to this message
  on_clear_msg = "Message history deleted." # defaults to this message,
  return_token_count = False
)
```

-   **`telegram_token`**: The token from Telegram's `BotFather`. If this is not provided, the manager will search for `TELEGRAM_TOKEN` using its standard key retrieval mechanism (`api_keys.json`, `.env`, or environment variables). If the key is not found, an error will be thrown.
-   **`component_name`**: Optional string defining which component should be executed when receiving a user message. If not set, this defaults to the latest or default automation defined, just like `manager.run()`.
-   **`verbose`**: Optional boolean, defines whether the system will run in verbose mode or not (defaults to False).
-   **`on_complete`**: Optional callable, function that will be called when completing execution after a specific user message.
-   **`on_update`**: Optional callable, function that will be called every time a component finishes execution.
-   **`whisper_provider`**: Optional string, provider used to transform voice notes and audio files to text in order for them to be processed by the system. If not set, it looks for a `groq` API key first, and for an `openai` API key later, and uses the `whisper` implementation of the first available provider.
-   **`whisper_model`**: Optional string, `whisper` model used for speech-to-text transformation. Defaults to `whisper-large-v3-turbo` for `groq` and to `whisper-1` for `openai`.
-   **`speech_to_text`**: Optional callable, custom function to be called instead of the default providers.
-   **`on_start_msg`**: Optional string defining what the bot will send to the user when receiving '/start' commnad.
-   **`on_clear_msg`**: Optional string defining what the bot will send to the user when receiving '/clear' command.
-   **`return_token_count`**: Boolean that determines whether component outputs should include usage metadata (`input_tokens`, `output_tokens`, `usd_cost`). This allows for tracking costs and token counts for every user interaction.

After defining the system through JSON and writing the necessary functions, it's possible to run a full Telegram bot with just one line of code:

```python
AgentSystemManager(config="config.json").start_telegram_bot()
```

Defining `on_complete` and `on_update` is optional. If not defined, the system will automatically send the latest message's `"response"` field after execution is finished. If this is not desired behavior, the developer should define `on_complete` to return a string (the response to be sent to user), or `None` if no message should be sent to the user in that step, always taking `messages`, `manager` and `on_complete_params` as arguments. The same applies to `on_update`. In both cases, the developer **does not need to handle Telegram integration**. When using them in conjunction with the `start_telegram_bot` method, they can return a string (which will be sent to the correct user by the system), `None` to send nothing, or a dict for more advanced response patterns, as described below.

Telegram integration also supports image processing out of the box, with no extra effort from the developer.

#### Automatic Speech To Text in Telegram

The `mas` Telegram integration functionality handles speech-to-text transcription for audios and voice notes automatically. You can specify a provider (either `groq` or `openai`) as described above, or they will be used automatically if the API key is available (`groq` is tried first). You may also define your own `speech_to_text` function if you need to. This function must receive a single argument, a dictionary with keys for `manager`, the audio's `file_path`, and Telegram's `update` and `context`. The function must return the text as string.

#### Handling Responses in Telegram

When using `start_telegram_bot`, the `on_complete` and `on_update` callbacks handle how your system responds to the user.

- **Default Behavior**: If you do not provide an `on_complete` function, the system will automatically find the last message generated, look for a `"response"` field inside its content, and send that text to the user.

- **Custom Behavior**: You can define your own `on_complete` or `on_update` functions for full control. These functions should be defined to accept `(messages, manager, params)`. The function's return value determines what is sent to the user:
    - **Return `None`**: Nothing is sent to the user.
    - **Return a `str`**: The string is sent as a plain text message.
    - **Return a `list` of blocks**: The system will iterate through the list and send each block as a separate message.
        - A `text` block will be sent as a text message.
        - An `image` block will be sent as a photo.
        - An `audio` block will be sent as a voice message.
    - **Return any other type**: The value will be converted to blocks using `manager._to_blocks()` and sent accordingly.

This allows for simple text replies or complex, multi-part responses with images and audio.

**Example `on_complete` function:**

```python
# fns.py
def my_on_complete(messages, manager, on_complete_params):
    # El wrapper devuelto por manager.get_messages() → último mensaje real
    blocks = messages[-1]["message"]      # List[Block]

    summary_text = None
    image_path   = None

    # 1. Buscar el primer bloque de texto y parsear su JSON
    for block in blocks:
        if block["type"] == "text":
            try:
                payload = json.loads(block["content"])
            except (json.JSONDecodeError, TypeError):
                payload = {}
            summary_text = payload.get("summary")
            image_path   = payload.get("image_path")
            break        # el dict siempre está en el primer text-block

    # 2. Construir la respuesta en formato bloque
    response_blocks = []
    if summary_text:
        response_blocks.append({
            "type": "text",
            "content": summary_text
        })

    if image_path:
        response_blocks.append({
            "type": "image",
            "content": {"kind": "file", "path": image_path}
        })

    return response_blocks
```

#### Administrative Commands

If you initialize the `AgentSystemManager` with an `admin_user_id` (e.g., your personal Telegram or WhatsApp chat ID), that user gains access to powerful administrative commands in bot integrations:

*   **/clear\_all\_users**: Clears the message history for **every** user of the bot.
*   **/reset\_system**: A destructive command that **deletes all data**: all history databases (`history/`), all saved files (`files/`), and all usage logs (`logs/`). Use with caution.
*   **/logs**: Sends the `usage.log` and `summary.log` files directly to the admin user for inspection.

These commands are protected and can only be executed by the configured admin user.


### WhatsApp Integration

The library offers seamless integration with the WhatsApp Cloud API, allowing you to deploy your multi-agent system as a WhatsApp bot. This is handled by the `start_whatsapp_bot` method, which launches a Flask-based web server to handle WhatsApp webhooks.

```python
manager.start_whatsapp_bot(
  whatsapp_token=None,
  phone_number_id=None,
  webhook_verify_token=None,
  component_name="my_automation",
  verbose=False,
  on_complete=None,
  on_update=None,
  speech_to_text=None,
  host="0.0.0.0",
  port=5000,
  base_path="/webhook"
)
```

**Parameters:**

*   **`whatsapp_token`**: Your WhatsApp Cloud API access token.
*   **`phone_number_id`**: The Phone Number ID from your WhatsApp App settings.
*   **`webhook_verify_token`**: The custom verify token you set up for your webhook.
    *(These three credentials can be provided directly or sourced from your API keys file/environment variables).*
*   **`component_name`, `verbose`, `on_complete`, `on_update`, `speech_to_text`**: These parameters function identically to their counterparts in `start_telegram_bot`.
*   **`host`, `port`, `base_path`**: Standard Flask server configuration for hosting the webhook endpoint.

The WhatsApp bot supports text, images, and audio (voice notes), with automatic STT transcription. It also supports the same administrative commands as the Telegram bot (`/clear`, `/start`, `/clear_all_users`, `/reset_system`, `/logs`), provided an `admin_user_id` is configured in the manager.


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

When you pass content to `manager.run()` or `manager.add_blocks()`, the library automatically converts it into this block format. Agents then receive this list of blocks as their input context.

### A Simpler Way to Add Multimodal Content: `add_blocks`

While you can manually build the list of blocks, the library includes the `add_blocks` method, which greatly simplifies this process. This method allows you to add text, images (from file paths or bytes), and dictionaries to the user's history, and it handles the conversion to the correct block format automatically.

It is the recommended way to programmatically insert content into the history.

## Input String Parsing

The input string parser allows for advanced message selection and transformation based on specifications in a string format. This syntax is usually used when listing components in automations, and allows to specify which messages will each component receive from the history. The syntax can also be used when defining looping and branching conditions within automations.

#### Defining Inputs Using MAS Input Syntax

In an automation, when defining a component to run in a step, you specify the component name in a string:

```json
"executed_component"
```

**Default Behavior**:

- For agents: The component receives the complete conversation (excluding messages labeled with the `"internal"` role, and respecting positive/negative filters defined during creation).
- For tools: The component receives the latest message directly as a dictionary, because its values will be passed as arguments to the tool function, in order.
- For processes: The component receives the latest message as a single-element list containing the content dictionary.

You can override these defaults with more complex input specifications. This is where input string syntax comes into play, starting with a colon which denotes the start of a string that will encode input information.

#### Specifying Input Component

- **Input Component:**
  ```json
  "executed_component:input_component"
  ```
  Specifies which input component’s messages are used as input by the executed component. For agents, this includes the full conversation history filtered to only messages from the specified `input_component` (as well as applying positive and negative filters defined for that agent). For tools and processes, this fetches the latest message from `input_component`.

#### Specifying Input Index

- **Indexing Messages:**
  ```json
  "executed_component:input_component?index"
  ```
  Specifies the index of `input_component`'s message history to fetch.
  - Negative integers indicate positions from the end (e.g., `?-1` is the latest message, which is the default for tools and processes).
  - A range can be specified (e.g., `?-5~-2` fetches all messages from the 5th-to-last to the 2nd-to-last from the specified `input_component`).
  - Open intervals are allowed:
    - `?-4` grabs all messages up to the 4th-to-last.
    - `-4?` grabs the latest 4 messages.
    - `?~` grabs all messages (this is the default for agents).

Ranges are well-defined for agents and processes, which receive lists of messages, but not for tools, as they work with single dictionaries. To retrieve fields from multiple messages for tools, specialized syntax is required (this will be covered below).

#### Specifying Input Fields

- **Selecting Fields:**
  ```json
  "executed_component:input_component?[field1, field2]"
  ```
  Filters the content dictionary of the messages to include only specified fields. For agents, this applies to all fetched messages. For processes, this applies to all fetched messages too, although by default it grabs only the latest message (like in this example, where no index is specified). For tools, it applies to the single fetched message.

#### Combining Index and Field Specification

- **Combining Index and Fields:**
  ```json
  "executed_component:input_component?-3~?[desired_field]"
  ```
  Executes the component with input as the last three messages from `input_component`, filtering each to only include `desired_field` and ignoring other fields that could be part of those messages. This would be valid for agents and processes, but not for tools, as the index specifies a range.

#### Concatenating Multiple Inputs

- **Concatenating Inputs:**
  ```json
  "executed_component:(input_comp_1, input_comp_2)"
  ```
  Specifies multiple input components.
  - For agents: Receives message histories from both components, defaulting to full message histories from both in this case, combining all into one history which preserves conversation order.
  - For processes: Builds a list of messages from each component just like agents, defaulting to full message histories, just like agents. Conversation history is preserved, and each message is wrapped into a dictionary which specifies "source" (the role) and "message" (the actual content) just like with agents.
  - For tools: Merges fetched messages from all components into a single dictionary. Developers must be careful not to merge input messages with fields named the same way, as they can overwrite each other. This is solved by carefully naming fields that will be used by the same tool at the same time.


Input concatenation allows for all the usual input parameter specification for each particular input. For example:
  ```json
  "executed_component:(input_comp_1?-1, input_comp_2?[field1, field2])"
  ```
  - For agents: Combines the latest message from `input_comp_1` with all messages from `input_comp_2` (filtered to `field1` and `field2`), preserving conversation order.
  - For processes: Grabs the latest message from both components, filtering the second component to keep only desired fields, and combines into a single list preserving conversation order.
  - For tools: Combines all fields from the latest message from `input_comp_1` with the specified fields from the latest message of `input_comp_2` into a single dictionary.

#### Specifying Conditionals

**Conditionals in Automations**
Conditionals are used for branching and loops in automations. By default, they evaluate a boolean field from the latest message:

  ```json
  "field1"
  ```

  Fetches `field1` from the latest message, which must be a boolean, and evaluates it. If true, the conditional is true as well.

**Complex Conditionals:**

If the conditional string starts with a colon, the input syntax can be used to specify conditional inputs in the same way as it is used for specifying inputs for components. Conditionals have defaults similar to tools, fetching the latest message by default and merging fields from different messages into a single dict for evaluation. For example:

  ```json
  ":input_comp?[field_1]"
  ```

  Fetches the latest message from `input_comp`, retrieves `field_1`, and evaluates it as a boolean.

  ```json
  ":input_comp?-3?[field1, field2]"
  ```

  Fetches the -3rd message from `input_comp`, retrieves `field1` and `field2`, and evaluates both fields (AND logic is applied by default, both need to be true for the conditional to be true).

  ```json
  ":(input_comp_1?[field1], input_comp_2?-4)"
  ```

  Combines the specified field from `input_comp_1`'s latest message and all fields from `input_comp_2`'s -4th message. All must evaluate to true for the conditional to be true. Just like with tools, the developer must be careful when naming output fields for components that will be evaluated inside the same conditional so as to not have overlap and overwriting.

Just like with tools, ranges are not allowed in conditionals, as they are ill-defined.

**Custom Functions:**
  ```json
  "fn:function_name"
  ```
  Evaluates a custom function, which must return a boolean value to be evaluated by the conditional. By default, this function takes as input a dictionary containing all fields from the latest message. This function must be present in the .py file associated with function calling.
  ```json
  "fn:function_name:input_comp"
  ```
  Takes all fields from the latest message of `input_comp` as input to the function.

  ```json
  "fn:function_name:input_comp?[field1, field2]"
  ```
  Fetches specified fields from the latest message of `input_comp` and passes them to the function.


#### Advanced Conditional Evaluation

Conditionals can also be defined as dictionaries with `"input"` and `"value"` keys. This supports evaluating non-boolean values. The `"input"` field is a string which behaves in the exact same way as the conditional string described above.

- **Basic Equality Check:**

  ```json
  {"input": "field1", "value": 5}
  ```

  Tests if `field1` from the latest message equals `5`.

- **Multiple Values:**

  ```json
  {"input": ":(input_comp_1?-2?[field1], input_comp_2?[field1, field2])", "value": [false, "yes", 0]}
  ```

  Evaluates fields in order of reference inside the input string: `field1` from `input_comp_1`'s second to last message must be `false`, `field1` from `input_comp_2`'s latest message must be the string `"yes"`, and `field2` from that same message must be `0` for the condition to be true.

The input syntax allows the developer to have a remarkable amount of control and flexibility over what is received as input by each component or conditional inside an automation, while keeping the JSON notation concise when working with simple workflows that can rely on reasonable default behaviors.

## Error Handling

The library is designed for robust operation, handling various errors gracefully. If an agent fails to produce a response, it will return a `default_output`. If the tool or process fails, the system will return the `default_output`, or an empty dict if no `default_output` was specified. Errors in configuration files or function references are logged for debugging.

## Design Patterns and Recommendations

The `mas` library is designed to be flexible and versatile, and you can use it to fit your own specific needs and requirements. However, certain patterns are commonly repeated and are worth mentioning here for beginners who are learning to build multi agent systems effectively.

### Model Recommendations

-  **`groq: llama-3.1-8b-instant`**: Free to use and extremely fast when ran in groq's hardware. It works relatively well for very simple tasks, usually good for testing during development. Not good for production as it makes too many mistakes and rate limits are not permissive enough.
-  **`groq: llama-3.3-70b-versatile`**: Free to use and quite fast, as well as intelligent enough for relatively complex tasks. Good for testing complex workflows during development. Not good for production as rate limits are too low.
-  **`deepseek: deepseek-chat`**: Very cheap, fast and intelligent enough for most tasks. Good for production, especially for agents that don't require complex reasoning.
-  **`deepseek: deepseek-reasoner`**: State-of-the-art reasoning, cheaper than commonly used models, such as gpt-4o. Thinking time can take seconds or even minutes, but results are usually robust for high-complexity tasks.
-  **`google: gemini-2.0-flash-exp`**: Free to use as of January 2025, good for its 1M token context window and its speed, quite good at reasoning as well.
-  **`openai: gpt-4o`**: Usually taken as the default model for its reasonably good speed, intelligence and robustness. However, it's quite expensive compared to models such as deepseek's `deepseek-chat` or `deepseek-reasoner`.
-  **`openai: o3`**: State-of-the-art reasoning, but the most expensive model by far in this list, and not necessarily better than `deepseek-reasoner` in most cases.

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
