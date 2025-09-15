# mas/cli/cli.py
import argparse
import sys
import json
from pathlib import Path

def _is_tty():
    """Verifica si la sesión es interactiva."""
    return hasattr(sys, "stdin") and sys.stdin.isatty()

def _ask(prompt: str, default: str = None) -> str:
    """Solicita un valor al usuario de forma interactiva."""
    if not _is_tty():
        return default or ""
    suffix = f" [{default}]" if default else ""
    response = input(f"{prompt}{suffix}: ").strip()
    return response or default or ""

def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Solicita al usuario que elija una opción de una lista."""
    choices_str = "/".join(choices)
    response = _ask(f"{prompt} ({choices_str})", default).lower()
    if response not in choices:
        print(f"Opción inválida '{response}'. Usando el valor por defecto: '{default}'.")
        return default
    return response

def start(project_dir_str: str = None):
    """
    Función que ejecuta el scaffolding del proyecto MAS.
    """
    project_dir = Path(project_dir_str or Path.cwd())
    project_dir.mkdir(parents=True, exist_ok=True)

    print("Bienvenido al asistente de configuración de MAS.")
    print(f"Se crearán los archivos del proyecto en: {project_dir.resolve()}")

    # 1. Preguntar por el proveedor y el modelo
    default_models = {
        "google": "gemini-1.5-flash",
        "openai": "gpt-4o-mini",
        "groq": "llama3-8b-8192",
    }
    provider = _ask_choice("Elige un proveedor de LLM", list(default_models.keys()), "google")
    model = _ask("Introduce el nombre del modelo", default_models[provider])

    # 2. Pedir la API Key (opcional)
    api_key_name = f"{provider.upper()}_API_KEY"
    api_key_value = _ask(f"Introduce tu {api_key_name} (opcional, puedes dejarlo en blanco)")

    # 3. Crear los archivos
    # config.json
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
    (project_dir / "fns.py").write_text("# Aquí puedes definir las funciones para tus Tools y Processes.\n")

    # main.py
    main_py_content = (
        'from mas import AgentSystemManager\n\n'
        '# Inicializa el manager desde el archivo de configuración\n'
        'try:\n'
        '    manager = AgentSystemManager(config="config.json")\n'
        '    # Ejecuta el sistema con una entrada de ejemplo\n'
        '    output = manager.run(input="Hello World!", verbose=True)\n\n'
        '    print("\\n--- Salida del Agente ---")\n'
        '    print(output)\n'
        '    print("------------------------\\n")\n'
        '    # Opcional: muestra el historial completo de mensajes\n'
        '    # manager.show_history()\n'
        'except Exception as e:\n'
        '    print(f"Ocurrió un error: {e}")\n'
        '    print("Por favor, asegúrate de que tu API key en el archivo .env es correcta.")\n'
    )
    (project_dir / "main.py").write_text(main_py_content)

    # 4. Mensaje de confirmación
    print("\n¡Proyecto MAS creado con éxito!")
    print("Archivos generados:")
    print(f"- {project_dir / 'config.json'}")
    print(f"- {project_dir / '.env'}")
    print(f"- {project_dir / 'fns.py'}")
    print(f"- {project_dir / 'main.py'}")
    print("\nPróximos pasos:")
    if not api_key_value:
        print(f"1. Edita el archivo `.env` y añade tu {api_key_name}.")
    print("2. Ejecuta `python main.py` para probar tu agente.")

def main(argv=None):
    """Función principal de la CLI."""
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mas", description="CLI para la biblioteca Multi-Agent System (MAS).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Comando 'start'
    start_parser = subparsers.add_parser("start", help="Crea la estructura básica de un proyecto MAS en el directorio actual.")
    start_parser.add_argument("directory", nargs="?", default=".", help="Directorio donde se creará el proyecto (opcional, por defecto es el actual).")

    args = parser.parse_args(argv)

    if args.command == "start":
        start(project_dir_str=args.directory)
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())