# mas/cli/cli.py
import argparse
import sys
import json
from pathlib import Path
import os
import shutil
from mas import AgentSystemManager

def _is_tty():
    """Check whether the session is interactive."""
    return hasattr(sys, "stdin") and sys.stdin.isatty()

def _ask(prompt: str, default: str = None) -> str:
    """Prompt the user for a value interactively."""
    if not _is_tty():
        return default or ""
    suffix = f" [{default}]" if default else ""
    response = input(f"{prompt}{suffix}: ").strip()
    return response or default or ""

def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt the user to choose an option from a list."""
    choices_str = "/".join(choices)
    response = _ask(f"{prompt} ({choices_str})", default).lower()
    if response not in choices:
        print(f"Invalid option '{response}'. Using the default value: '{default}'.")
        return default
    return response

def start(project_dir_str: str = None, config_str: str = None, verbose: bool = False):
    project_dir = Path(project_dir_str or Path.cwd())
    project_dir.mkdir(parents=True, exist_ok=True)

    if config_str:
        # --config mode: from JSON file or description
        print(f"Automatic configuration mode enabled in: {project_dir.resolve()}")

        if config_str.endswith(".json") and os.path.exists(config_str):
            # Case 1: existing JSON file
            target_json_path = project_dir / "config.json"
            shutil.copy(config_str, target_json_path)
            print(f"\nMAS project created successfully from '{config_str}'!")
            print(f"- Configuration copied to: {target_json_path}")
            print("- Remember to create your .env file with the required API keys.")

        else:
            # Case 2: natural language description (bootstrap)
            print("\nWelcome to the MAS system creation wizard.")
            print("An LLM will be used to generate the project configuration.")

            # 1. Ask for the LLM used during bootstrap
            print("\n--- Bootstrap LLM configuration ---")
            bootstrap_providers = {"google": "gemini-2.5-pro", "openai": "gpt-5", "anthropic": "claude-sonnet-4"}
            bootstrap_provider = _ask_choice("Choose a provider to generate the system", list(bootstrap_providers.keys()), "google")
            bootstrap_model = _ask("Enter the bootstrap model name", bootstrap_providers[bootstrap_provider])

            # 2. Ask for the bootstrap API key (required)
            bootstrap_key_name = f"{bootstrap_provider.upper()}_API_KEY"
            bootstrap_api_key = ""
            while not bootstrap_api_key:
                bootstrap_api_key = _ask(f"Enter your {bootstrap_key_name} (required for this process)")
                if not bootstrap_api_key:
                    print("An API key is required so the LLM can generate the system.")

            os.environ[bootstrap_key_name] = bootstrap_api_key

            # 3. Ask for default models for the final system
            print("\n--- Generated system configuration ---")
            target_providers = {"google": "gemini-2.5-flash", "openai": "gpt-5-nano", "anthropic": "claude-sonnet-4"}
            target_provider = _ask_choice("Choose a default provider for your final system", list(target_providers.keys()), "google")
            target_model = _ask("Enter the default model for your final system", target_providers[target_provider])

            # 4. Enrich the prompt with the collected information
            enhanced_description = (
                f"{config_str}\n\n"
                f"Constraint: The user wants the final system to use the provider '{target_provider}' "
                f"with the model '{target_model}' as the default in the 'default_models' section of 'config.json'."
            )
            
            print("\nGenerating system... This may take a moment.")
            try:
                # 5. Run the bootstrap
                AgentSystemManager(
                    config=enhanced_description,
                    bootstrap_models={"provider": bootstrap_provider, "model": bootstrap_model},
                    base_directory=str(project_dir),
                    verbose=verbose
                )

                # 6. Create the .env file for the final system
                target_api_key_name = f"{target_provider.upper()}_API_KEY"
                target_api_key_value = ""

                if target_provider == bootstrap_provider:
                    target_api_key_value = bootstrap_api_key
                else:
                    target_api_key_value = _ask(f"Enter your {target_api_key_name} for the final system (optional)")

                env_content = f"{target_api_key_name}={target_api_key_value}\n" if target_api_key_value else f"# {target_api_key_name}=\n"
                (project_dir / ".env").write_text(env_content)

                print("\nMAS project generated successfully!")
                print("Generated files:")
                print(f"- {project_dir / 'config.json'}")
                print(f"- {project_dir / 'fns.py'}")
                print(f"- {project_dir / '.env'}")
                print("\nNext steps:")
                if not target_api_key_value:
                    print(f"1. Edit the `.env` file and add your {target_api_key_name}.")
                print("2. Review `config.json` and `fns.py` to inspect the generated system.")
                print("3. Run `mas start` again to create a `main.py` for testing, or craft your own!")

            except Exception as e:
                print(f"\nAn error occurred while generating the system: {e}")
                print("Please make sure your bootstrap API key is correct and that the model is accessible.")

    else:
        print("Welcome to the MAS configuration assistant.")
        print(f"The project files will be created at: {project_dir.resolve()}")

        default_models = {
            "google": "gemini-2.5-flash", "openai": "gpt-5-nano", "groq": "openai/gpt-oss-120b",
        }
        provider = _ask_choice("Choose an LLM provider", list(default_models.keys()), "google")
        model = _ask("Enter the model name", default_models[provider])

        # 2. Ask for the API key (optional)
        api_key_name = f"{provider.upper()}_API_KEY"
        api_key_value = _ask(f"Enter your {api_key_name} (optional, you may leave it blank)")

        # 3. Create the files
        config_content = {
            "general_parameters": {
                "api_keys_path": ".env",
                "default_models": [
                    {"provider": provider, "model": model}
                ]
            },
            "components": [
                {
                    "type": "agent",
                    "name": "simple_agent",
                    "system": "You are a basic assistant that answers questions.",
                    "required_outputs": {
                        "response": "A text response to be sent to the user."
                    }
                }
            ]
        }
        (project_dir / "config.json").write_text(json.dumps(config_content, indent=4))

        # .env
        env_content = f"{api_key_name}={api_key_value}\n" if api_key_value else f"# {api_key_name}=\n"
        (project_dir / ".env").write_text(env_content)

        # fns.py
        (project_dir / "fns.py").write_text("# Define the functions for your Tools and Processes here.\n")

        # main.py
        main_py_content = (
            'from mas import AgentSystemManager\n\n'
            '# Initialize the manager from the configuration file\n'
            'try:\n'
            '    manager = AgentSystemManager(config="config.json")\n'
            '    # Run the system with a sample input\n'
            '    output = manager.run(input="Hello World!", verbose=True)\n\n'
            '    print("\\n--- Agent Output ---")\n'
            '    print(output)\n'
            '    print("------------------------\\n")\n'
            '    # Optional: display the full message history\n'
            '    # manager.show_history()\n'
            'except Exception as e:\n'
            '    print(f"An error occurred: {e}")\n'
            '    print("Please make sure the API key in your .env file is correct.")\n'
        )
        (project_dir / "main.py").write_text(main_py_content)

        # 4. Confirmation message
        print("\nMAS project created successfully!")
        print("Generated files:")
        print(f"- {project_dir / 'config.json'}")
        print(f"- {project_dir / '.env'}")
        print(f"- {project_dir / 'fns.py'}")
        print(f"- {project_dir / 'main.py'}")
        print("\nNext steps:")
        if not api_key_value:
            print(f"1. Edit the `.env` file and add your {api_key_name}.")
        print("2. Run `python main.py` to test your agent.")

# >> ADD THIS FUNCTION BEFORE THE `main` FUNCTION <<

def run_system(project_dir_str: str, input_str: str = None, component_name: str = None, verbose: bool = False):
    """
    Load and execute a MAS system from a project directory.
    """
    project_dir = Path(project_dir_str)
    config_path = project_dir / "config.json"

    # 1. Ensure config.json exists
    if not config_path.is_file():
        print(f"Error: 'config.json' was not found in '{project_dir.resolve()}'.")
        print("Make sure you are in the correct directory or create a system with 'mas start'.")
        return 1

    print(f"Loading system from: {config_path.resolve()}")
    
    try:
        manager = AgentSystemManager(
            config=str(config_path),
            base_directory=str(project_dir),
            verbose=verbose
        )

        print("Running the system...")
        output = manager.run(
            input=input_str,
            component_name=component_name,
            verbose=verbose
        )

        print("\n--- Final system output ---")
        if isinstance(output, (dict, list)):
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(output)
        print("--------------------------------\n")
        
        return 0

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Check your 'config.json' file and ensure the API keys in '.env' are correct.")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def main(argv=None):
    """Entry point for the CLI."""
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mas", description="CLI for the Multi-Agent System (MAS) library.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'start' command
    start_parser = subparsers.add_parser("start", help="Create the structure of a MAS project.")
    start_parser.add_argument("--directory", "-d", default=".", help="Directory where the project will be created (default: current).")
    start_parser.add_argument("--config", "-c", help="Natural-language description or path to a config.json to create the system.")
    start_parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed logging.")

    # --- 'run' command ---
    run_parser = subparsers.add_parser("run", help="Run a MAS system from a config.json.")
    run_parser.add_argument("--directory", "-d", default=".", help="MAS project directory (default: current).")
    run_parser.add_argument("--input", "-i", help="Text input to provide to the system.")
    run_parser.add_argument("--component", "-c", help="Name of the specific component to execute.")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed logging.")

    args = parser.parse_args(argv)

    if args.command == "start":
        return start(project_dir_str=args.directory, config_str=args.config, verbose=args.verbose)
    elif args.command == "run":
        return run_system(
            project_dir_str=args.directory,
            input_str=args.input,
            component_name=args.component,
            verbose=args.verbose
        )

    return 1

if __name__ == "__main__":
    sys.exit(main())
