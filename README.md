# The Multi-Agent System (MAS) Library

The `mas` library is a powerful and flexible framework designed for creating and managing multi-agent systems. Whether you need a simple chatbot, a sophisticated automation workflow, or a fully integrated AI-driven ecosystem, the `mas` library provides the tools you need to build, scale, and maintain your system with ease.

### What Can You Build with the `mas` Library?

- **Intelligent Chatbots**: Create conversational agents with predefined roles and tasks.
- **API Orchestrators**: Seamlessly integrate multiple APIs through tools managed by intelligent agents.
- **Data Processing Pipelines**: Build processes to transform, analyze, and deliver data efficiently.
- **Decision-Making Systems**: Implement branching logic and dynamic workflows to enable adaptive decision-making.

### Why Choose the `mas` Library?

- **Minimal Setup**: Define agents, tools, and workflows using JSON configurations or Python scripts. Start with simple setups and expand as needed.
- **Seamless LLM Integration**: Manage interactions with multiple LLM providers (e.g., OpenAI, Google, Groq) without adapting your code or message history for each one.
- **Automated System Prompts**: Automatically generate and manage system prompts to reduce the need for complex prompt engineering, ensuring consistent and reliable agent behavior.
- **Error-Tolerant JSON Parsing**: Includes a robust JSON parser that can recognize and correct malformed JSON structures, even when LLMs produce imperfect outputs.
- **Scalability**: Add or modify components like agents, tools, and processes, expanding the capabilities of a system with minimal refactoring.
- **Integration Ready**: Effortlessly connect with external APIs, databases, and custom Python functions for real-world applications.
- **Dynamic Workflows**: Support for branching, looping, and conditional execution in complex automation sequences.
- **Context-Aware Agents**: Use centralized message histories to enable agents to make informed decisions and maintain continuity across interactions.
- **Multi-User Support**: Manage isolated histories for multiple users, making it ideal for systems that handle multiple independent conversations or workflows.
- **Focus on Logic**: Offload low-level details of message management, task sequencing, and system orchestration to the library so you can concentrate on your application's goals.

The `mas` library empowers developers to create robust, flexible, and intelligent systems while minimizing the complexity of setup and orchestration. Whether you are a beginner experimenting with multi-agent architectures or an expert building large-scale AI-driven workflows, the `mas` library adapts to your needs.

### Currently in alpha
This library has not yet been extensively tested and is currently under development. Please **do not use it in production yet**, but feel free to test it and post your issues here, or email me at mneuronico@gmail.com to collaborate or know more about this project.

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
pip install git+https://github.com/mneuronico/multi-agent-system-library
```

## Step 1: Create a Minimal JSON Configuration

Let’s create a minimal configuration to get started with a single agent.

### Example `config.json`

```json
{
    "general_parameters": {
        "api_keys_path": "api_keys.json"
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

Ensure you have an `api_keys.json` file in the base directory. Currently, the supported providers are "openai", "groq", "google", "anthropic" and "deepseek".

```json
{
    "openai": "sk-your-openai-api-key",
    "groq": "your-groq-key",
    "google": "your-google-key",
    "anthropic": "your-anthropic-key",
    "deepseek": "your-deepseek-key"
}
```

---

## Step 2: Load and Run the System

Use the JSON configuration to initialize and run the system:

### `main.py`

```python
from mas import AgentSystemManager

# Load the system from the JSON configuration
manager = AgentSystemManager(config_json="config.json")

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

# Multi-Agent System (mas) Library Documentation

## Introduction

The `mas` library provides a robust framework for building complex multi-agent systems. It allows you to create and manage various types of components: `Agents`, `Tools`, `Processes`, and `Automations`. These components interact via a centralized message history mechanism. The library supports both programmatic construction and system definition using JSON files.

## Core Concepts

### Components

Components are the fundamental building blocks of your multi-agent system:

-   **`Agent`**: Agents utilize Large Language Models (LLMs) to generate responses. They are configured with system prompts, required output structures, and filters that specify which messages to use as context. Agents can use tools to accomplish a task. They receive a list of messages (full message history as default) as input and their output is always a dictionary with required fields. The library manages system prompts automatically so that the JSON responses from LLMs always conform with the required outputs.
-   **`Tool`**: Tools perform specific actions (like API calls, database queries, etc.) and are typically used by agents. Tools receive predetermined input fields as a dictionary (typically from a specific agent that is using the tool) and produce an output dictionary, which can then be used by other agents, or it can be processed in some way.
-   **`Process`**: Processes perform data transformations, data loading, or any other kind of data manipulation. They can be used to insert any necessary code or processing that cannot be natively managed by the library. They receive a list of messages (although only the latest message is passed to them by default) and return an output dictionary which can then be used by other agents or tools.
-   **`Automation`**: Automations are workflows that orchestrate the execution of other components (agents, tools and processes). They manage the sequence of steps and support branching and looping structures. Their input is directly passed to the first component in their structure and they return a dictionary with the output of the latest executed component.

### Message History

All interactions between components are recorded in a database, providing context and a message history for agents, tools and processes. Each message includes:

-   A unique identifier (ID).
-   A sequential message number (`msg_number`), stored chronologically and incremented with each new message.
-   The `type` of the component that was ran (agent, tool or process).
-   The `model` that was called, if `type` is 'agent' (in the format 'provider:model_alias').
-   A `role` that indicates the component that produced the message.
-   The `content` of the message, which is typically a dictionary as given by the corresponding component.

Each `user_id` has its own isolated message history.

### Roles

Each component is assigned a unique `role` when its output is stored in the database. Roles can simply be the name of the component, but there are some special terms:
-   `user`: Messages directly from a user (when calling `run` with `input` string).
-   `internal`: Messages not readable by agents but still present in the database (commonly when calling `run` with an `input` dict).


## The Agent System Manager

### Initialization

```python
from mas import AgentSystemManager

manager = AgentSystemManager(
    base_directory="path/to/your/base_dir",  # Default is the current working directory.
    api_keys_path="api_keys.json",  # Default is api_keys.json
    general_system_description="This is a description for the overall system.", # Default: "This is a multi agent system."
    functions_file="my_fns_file.py", # Default: "fns.py"
    default_models=[{"provider": "groq", "model": "llama-3.1-8b-instant"}],
    imports=[
        "common_tools.json",  # Import all components from file
        "external_agents.json->research_agent+analysis_tool"  # Import specific components
    ],
    on_update=on_update, # Default: none
    on_complete=on_complete # Default: none
)
```

The `AgentSystemManager` manages your system’s components, user histories, and general settings.

-   **`base_directory`**: Specifies the directory where user history databases (`history` subdirectory) and pickled object files (`files` subdirectory) are stored. Also the location of `fns.py`.
-   **`api_keys_path`**: The name of a JSON file containing API keys for various LLM providers, which must be in the `base_directory`. Example:

    ```json
    {
        "openai": "sk-...",
        "groq": "groq-..."
    }
    ```
-   **`general_system_description`**: A description appended to the system prompt of each agent.
-   **`functions_file`**: The name of a Python file where function definitions must be located. This file must exist in the base directory.
-   **`default_models`**: A list of models to use when executing agents, for agents that don't define specific models. Each element of the list should be a dictionary with two fields, `provider` (like 'groq' or 'openai') and `model` (the specific model name). These models will be tried in order, and failure of a model will trigger a call to the next one in the chain.
-   **`imports`**: List of component import specifications. Each entry can be either `"<your_json>.json"` to import all components from that file, or `"<your_json>.json"->component1+component2` to import specific components from that file.
-   **`on_update`**: Function to be executed each time an individual component is finished running. The function must receive a list of messages and the manager as the only two arguments. Useful for doing things like updating an independent database or sending messages to user during an automation.
-   **`on_complete`**: Function to be executed when `manager.run`() reaches completion. This is equivalent to `on_update` when calling `manager.run()` on an individual component (if both are defined, both will be executed), but it's different for automations, since it will only be ran at the end of the automation. The function must receive a list of messages and the manager as the only two arguments. Useful for doing things like sending the last message to the user after a complex automation workflow.

`on_update` and `on_complete` can be defined as callables directly, or they can be strings referring to the name of the function to used, located in the `functions_file`. To accomplish this, _function syntax_ must be used, by starting the string with _fn:_, for example:

_"fn:<your_function_name>"_ will attempt to retrieve a function with the specified name from the `functions_file`.

You can accomplish the same thing when defining the system from a JSON file:

```json
{
  "general_parameters": {
    "base_directory": "/path/to/your/base_dir",
    "api_keys_path": "api_keys.json",
    "general_system_description": "This is a description for the overall system.",
    "functions_file": "my_fns_file.py",
    "default_models": [
            {"provider": "deepseek", "model": "deepseek-chat"},
            {"provider": "groq", "model": "llama-3.3-70b-versatile"}
        ],
    "imports": [
        "common_tools.json",
        "external_agents.json->research_agent+analysis_tool"
    ],
    "on_update": "fn:on_update_function", 
    "on_complete": "fn:on_complete_function"
  },
  "components": [...]
}
```

### Setting the Current User

Each user has its own message history, saved as an isolated SQLite database file (.sqlite). This is important because the exact same system manager, with identical structure, can handle many independent conversation histories seamlessly. To specify the user whose history you want to use, call:

```python
manager.set_current_user("user123")  # Creates a new DB for "user123" if it does not exist.
```

If no user is set, a new UUID will be automatically created and the current user will be set to that UUID, which will the be subsequently used until explicitely changed by the developer.

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
    model_params={"temperature": 1.0, "max_tokens": 4096}
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
    Supported providers so far are: `"openai"`, `"google"`, `"groq"`, `"anthropic"`, and `"deepseek"`. Ensure the corresponding `api_key` is available in `api_keys.json`.
-   **`default_output`**: The output to use when all the models fail, should match the `required_outputs`.
-   **`positive_filter`**: A list of `roles` to be included in the context of the agent (all other roles will be ignored if this is defined).
-   **`negative_filter`**:  A list of `roles` to be excluded from the context.
    You can filter using these values:
    -   `user`: selects messages from the user.
    -   `agent`: selects messages from all roles from the agent type.
    -   `tool`: selects messages from all roles from the tool type.
    -   `process`: selects messages from all roles from the process type.
    -   Exact role names (e.g., `myagent`)
-   **`model_params`**: Dictionary including params for advanced LLM configuration. Supported params right now are `temperature`, `max_tokens` and `top_p`. Not defining these will use default configuration for each provider.


You can create agents when defining the system from a JSON file by including them in the component list:

```json
{
  "general_parameters": {
    // ...
  },
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
      "model_params": {
        "temperature": 1.0,
        "max_tokens": 4096
      }
    }
  ]
}
```

#### Tools

```python
def my_tool_function(query):
    # Make a call to an API based on the query
    return {"items": ["item1", "item2"]}

tool_name = manager.create_tool(
    name="mytool",
    inputs={"query": "query from the agent to call the tool"},
    outputs={"data_field_1": "some data returned by the api",
            "data_field_2": "more data returned by the api"},
    function=my_tool_function,
    default_output={"items": "Default items"}  # Default: {}
)
```

-   **`name`**:  The name of the tool.
-   **`inputs`**:  A dictionary describing the input parameters for the `function` using descriptions and names.
-   **`outputs`**:  A dictionary describing the output parameters of the `function` using descriptions and names.
-   **`function`**: A callable (function) that performs the task of the tool. This function receives as many arguments as needed, which must be defined in the same order as the dictionary that will be used as input for this tool (the dictionary from the latest message is used by default, but more complex inputs can be defined as explained below).
-  **`default_output`**: Output to use if there's an error during the function call, or an exception has been raised by the function.

Tools can be included in the component list of the config JSON file just like agents:

```json
{
  "components": [
    {
      "type": "tool",
      "name": "mytool",
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
def my_process_function(message_list):
    # do anything and return any values that need saving in a dict
    return {"content": "some content from local file"}

process_name = manager.create_process(
    name="myprocess",
    function=my_process_function
)
```

-   **`name`**: The name of the process.
-   **`function`**: A callable (function) that performs data transformations, data loading, etc. This function receives as argument a list of messages, each of which is a dictionary with two fields ("source", with the source role, and "message" with the actual content) and it should return a dictionary.

You can also define processes in the config JSON file:

```json
{
  "components": [
    {
      "type": "process",
      "name": "myprocess",
      "function": "fn:my_process_function"
    }
  ]
}
```

#### Automations

```python
automation_name = manager.create_automation(
    name="myautomation",   # Optional, defaults to automation-<n>
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
-   **`sequence`**: An ordered list of steps to execute. Steps can be:
    -   A string representing a component name, with an optional input specification (more on **`mas` input syntax** below).
    -   A control flow dictionary with the following structure:
        -   `control_flow_type`:  `"branch"`, `"while"`, or `"for"`.
        -   For `"branch"`:
            -   `condition`: A boolean, or a string to evaluate as boolean, or a dict. The syntax to specify conditional statements will be covered below.
            -   `if_true`: A list of steps to execute if `condition` is `True` (steps can be themselves control flow types).
            -   `if_false`: A list of steps to execute if `condition` is `False`.
        -   For `"while"`:
            -   `start_condition`: A boolean or a string or a dict.  If `True`, the first iteration is executed. if omitted, the first iteration is executed.
            -   `end_condition`: A boolean or a string to evaluate as boolean, or a dict.
            -   `body`:  A list of steps to execute in each iteration.
        -   For `"for"`:
            -   `items`: Required. Defines what to iterate over. Each time a loop starts, the current item will be added to the message history, under the role of "iterator", wrapped in a dictionary with two fields ("item_number" as the cycle number and "item" as the current item). This field can be:
                - `numeric range`: A single integer n is taken to be the range [0, n). An array of two numeric elements [a, b] is taken to be the range [a, b). An array of three numeric elements [a, b, c] is taken to be a range from a to b in steps of c.
                - `component reference`: Using `mas input syntax` (see below) you can iterate through data produced by a previous component. The input defined here can be of the following types:
                    - `single message`: If the message contains more than one field, the iterator will execute one step per field, wrapping it in a dict with fields "key" and "value", with the key and value of the current dictionary item. If the message contains only one field, the iterator will try to resolve that field as one of the other types.
                    - `single number or list of two or three numbers`: This type of field is resolved to a numeric range.
                    - `list of arbitrary length and types`: The iterator executes one step per element.
                    - `dictionary`: Treats it the same way as a single message, which is itself a dictionary.
                A component reference can never refer to more than one message, as this is ill-defined for item iteration and will result in an exception.
            -   `body`: A list of steps (components or control-flow elements) to execute in each iteration.
        -   For `"switch"`:
            -   `value`: The comparison value source, can be:
                - Literal value (string/number/boolean)
                - MAS input syntax reference (e.g. `":my_agent?[field]"`), which must resolve to a single value to use in the switch statement.
            -   `cases`: Ordered list of potential matches, each containing:
                - `case`: Comparison value (use "default" for catch-all)
                - `body`: Steps to execute on match

Defining an automation in the config JSON file is as simple as including it in the list of components, just like any other component:

```json
{
  "components": [
    {
      "type": "automation",
      "name": "myautomation",
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

### Component Imports

The system supports importing components from external JSON files to enable modular architecture and component reuse. This works for both system-wide components and automation-specific references. It also works both when defining the system programmatically and from a JSON file. These JSON files must contain a `"components"` field, which must be a list of components.

#### General Import Syntax

Specify imports using these formats in the `imports` parameter:

```python
imports=[
    # Import all components from a file in the base directory
    "common_tools.json",
    
    # Import specific components from a file
    "external_agents.json->research_agent+analysis_tool",

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


### Linking Components

```python
manager.link_tool_to_agent_as_output("mytool", "myagent")  # myagent receives mytool inputs
manager.link_tool_to_agent_as_input("mytool", "myagent")  # myagent's context contains mytool's outputs
manager.link("myagent", "mytool") # automatic link
```

-   `link_tool_to_agent_as_output(tool_name, agent_name)`:  This is the function to call when a tool will be used by an agent. In this case, the agent must be ran before the tool. The mangaer updates the agent's `required_outputs` automatically to include the tool's `inputs`. This configures the Agent to produce the required input for the Tool to execute.
-   `link_tool_to_agent_as_input(tool_name, agent_name)`: This is the function to call when a tool will serve as input to an agent. It ensures the agent's message context includes the tool's output, and ensures the agent's filters don't exclude this tool output, as well as updating the system message of the agent so that it will pay special attention to the tool's output.
-   `link(comp1, comp2)`: Automatically links a tool and an agent based on their order: `tool->agent` or `agent->tool`

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
    "myagent": "mytool" // or you could do "mytool: myagent" to link the tool as input to the agent
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
    component_name="myagent",  # Optional - component to run
    input="Some user input",    # Optional input to add to the message history
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
   on_complete_params = {"to_use_in_complete": value}
)
print(output)
```

-   **`component_name`**: The name of the component to run. If not specified it uses the latest created automation, or creates a linear automation if one does not exist using all components available in their order of creation.
-   **`input`**: Optional string or dict to store in the message history. If it's a string, it will be stored with the role `"user"`. If it's a dictionary, it will be stored with the role `"internal"`, assumed to be information added by the developer.
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

### Deleting User History `clear_message_history`

If you need to clear the message history for a user, the `manager` offers a simple method to delete the database associated to a specific user.

```python
manager.clear_message_history(user_id) # if not provided, it will use the current user_id
```

### Clearing Cache

The `mas` library detects when a tool or a process returns a dictionary with values that are not compatible with JSON serializing. In those cases, those values are saved as pickle objects and then loaded when needed for tools or processes (not for agents, as they can only process text input). Files stay loaded in memory for faster retrieval unless explicitely cleared:

```python
manager.clear_file_cache() # Clears the cache that stores pickle objects.
```

### Loading from JSON and running the system

Below is a minimal example which runs an automation from a multi agent system defined by a JSON file present in the base directory:

```python
from mas import AgentSystemManager
manager = AgentSystemManager(config_json="<config_file_name>.json")
output = manager.run(input="Hey, how are you today?")
manager.show_history()
```

For maximum brevity, the whole system can be ran in only one line:

```python
manager = AgentSystemManager(config_json="<config_file_name>.json").run("Hey, how are you today?")
```

This builds the system from a JSON configuration file specified using the `config_json` parameter, creates all components if the configuration is valid and runs an automation (either a specified one or a default linear automation) with the provided user input (or starting with no input if none is provided).

### Defining Functions
Functions for `Tool` and `Process` components must be defined in the Python file specified using the `functions_file` option (or the default `fns.py`). All tools and processes must have their custom function defined in this file. Conditional functions, on_update and on_complete must be defined in this file as well. When using the library only programatically and not using function syntax (i.e. `"fn:"`) it is possible to define functions elsewhere and use them as callables directly, but it is better practice to still define them in the `functions_file`, import it and use it.

### Running a Chat Loop

The `run` method can be used in a loop to implement a simple interactive chat system. This is ideal for continuously processing user inputs and generating responses from the system.

```python
# Initialize the manager
manager = AgentSystemManager(config_json="config.json") 

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
    "api_keys_path": "api_keys.json",
    "general_system_description": "This is a description for the overall system.",
    "functions_file": "fns.py",
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
        {"provider": "openai", "model": "gpt-4-turbo"}
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
        print("Assistant:", latest_message.get("message", "No message content."))
```

In this function, you could possibly send this message to your own database, to a messaging app or to a custom user interface.

### Telegram Integration

The `mas` library allows the developer to integrate any system with Telegram seamlessly to allow users to interact with the system through the messaging app without requiring the developer to define custom async logic and event loops. This is possible through the `start_telegram_bot` method:

```python
manager.start_telegram_bot(telegram_token, # token must be provided
component_name = "my_automation", # optional, defaults to default or latest automation
verbose = False, # defaults to False
on_complete = None, # defaults to sending latest message to user
on_update = None, # defaults to no operation
on_start_msg = "Hey! Talk to me or type '/clear' to erase your message history.", # defaults to this message
on_clear_msg = "Message history deleted." # defaults to this message
)
```

-   **`telegram_token`**: The token given by Telegram's `BotFather` after successful bot creation through the Telegram platform. This lets the library connect with a specific bot to send and receive messages.
-   **`component_name`**: Optional string defining which component should be executed when receiving a user message. If not set, this defaults to the latest or default automation defined, just like `manager.run()`.
-   **`verbose`**: Optional boolean, defines whether the system will run in verbose mode or not (defaults to False).
-   **`on_complete`**: Optional callable, function that will be called when completing execution after a specific user message.
-   **`on_update`**: Optional callable, function that will be called every time a component finishes execution.
-   **`on_start_msg`**: Optional string defining what the bot will send to the user when receiving '/start' commnad.
-   **`on_clear_msg`**: Optional string defining what the bot will send to the user when receiving '/clear' command.

After defining the system through JSON and writing the necessary functions in the `functions_file`, it's possible to run a full Telegram bot with this minimal code example:

```python
manager = AgentSystemManager(config_json="config.json") 
token = "<YOUR_TELEGRAM_TOKEN>"
manager.start_telegram_bot(token)
```

Defining `on_complete` and `on_update` is optional. If not defined, the system will automatically send the latest message's `"response"` field after execution is finished. If this is not desired behavior, the developer should define `on_complete` to return a string (the response to be sent to user), or `None` if no message should be sent to the user in that step, always taking `messages`, `manager` and `on_complete_params` as arguments. The same applies to `on_update`. In both cases, the developer **does not need to handle Telegram integration**. When using them in conjunction with the `start_telegram_bot` method, they can return a string (which will be sent to the correct user by the system), `None` to send nothing, or a dict for more advanced response patterns, as described below.

#### Advanced Response Patterns

For more advanced use cases, `on_complete` and `on_update` can also return a dictionary to define various types of responses to be sent to the user. The system supports the following keys:

-   **`text`**: A plain text message to be sent to the user.
-   **`markdown`**: A MarkdownV2-formatted message.
-   **`image`**: A URL or file representing an image to be sent.
-   **`audio`**: A URL or file representing an audio file to be sent.
-   **`voice_note`**: A URL or file representing a voice note to be sent.
-   **`document`**: A URL or file representing a document to be sent.

The keys are processed in the order they appear in the dictionary, allowing the developer to define the sequence in which data is sent. For example:

```python
def my_on_complete(messages, manager, on_complete_params):
    return {
        "text": "Here's your data:",
        "image": "https://example.com/image.jpg",
        "document": "https://example.com/document.pdf"
    }
```

This will first send the text message, followed by the image, and then the document. This feature ensures flexibility while maintaining simplicity for the developer.


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
  - For processes: Builds a list of messages from each component just like agents, defaulting in this case to the latest message from each (unlike agents). Conversation history is preserved, and each message is wrapped into a dictionary which specifies "source" (the role) and "message" (the actual content) just like with agents.
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


## Currently Under Development

This is an alpha version of the `mas` library. It has not yet been tested extensively and likely contains many bugs and undesired behavior. Its use on production is **NOT RECOMMENDED**.
