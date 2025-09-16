# mas/cli/cli.py
import argparse
import sys
import json
from pathlib import Path
import os
import shutil
from mas import AgentSystemManager

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

def start(project_dir_str: str = None, config_str: str = None, verbose: bool = False):
    project_dir = Path(project_dir_str or Path.cwd())
    project_dir.mkdir(parents=True, exist_ok=True)

    if config_str:
        # Modo --config: Desde archivo JSON o descripción
        print(f"Modo de configuración automática activado en: {project_dir.resolve()}")

        if config_str.endswith(".json") and os.path.exists(config_str):
            # Caso 1: Es un archivo JSON existente
            target_json_path = project_dir / "config.json"
            shutil.copy(config_str, target_json_path)
            print(f"\n¡Proyecto MAS creado con éxito desde '{config_str}'!")
            print(f"- Se ha copiado la configuración a: {target_json_path}")
            print("- Recuerda crear tu archivo .env con las API keys necesarias.")

        else:
            # Caso 2: Es una descripción en lenguaje natural (bootstrap)
            print("\nBienvenido al asistente de creación de sistemas MAS.")
            print("Se usará un LLM para generar la configuración del proyecto.")

            # 1. Preguntar por el LLM para el proceso de bootstrap
            print("\n--- Configuración del LLM de Bootstrap ---")
            bootstrap_providers = {"google": "gemini-2.5-pro", "openai": "gpt-5", "anthropic": "claude-sonnet-4"}
            bootstrap_provider = _ask_choice("Elige un proveedor para generar el sistema", list(bootstrap_providers.keys()), "google")
            bootstrap_model = _ask("Introduce el nombre del modelo de bootstrap", bootstrap_providers[bootstrap_provider])
            
            # 2. Pedir la API Key para el bootstrap (obligatoria)
            bootstrap_key_name = f"{bootstrap_provider.upper()}_API_KEY"
            bootstrap_api_key = ""
            while not bootstrap_api_key:
                bootstrap_api_key = _ask(f"Introduce tu {bootstrap_key_name} (obligatorio para este proceso)")
                if not bootstrap_api_key:
                    print("La API key es necesaria para que el LLM pueda generar el sistema.")
            
            os.environ[bootstrap_key_name] = bootstrap_api_key
            
            # 3. Preguntar por los modelos por defecto para el *sistema final*
            print("\n--- Configuración del Sistema Generado ---")
            target_providers = {"google": "gemini-2.5-flash", "openai": "gpt-5-nano", "anthropic": "claude-sonnet-4"}
            target_provider = _ask_choice("Elige un proveedor por defecto para tu sistema final", list(target_providers.keys()), "google")
            target_model = _ask("Introduce el modelo por defecto para tu sistema final", target_providers[target_provider])

            # 4. Mejorar el prompt con la información recolectada
            enhanced_description = (
                f"{config_str}\n\n"
                f"Constraint: The user wants the final system to use the provider '{target_provider}' "
                f"with the model '{target_model}' as the default in the 'default_models' section of 'config.json'."
            )
            
            print("\nGenerando sistema... Esto puede tardar un momento.")
            try:
                # 5. Ejecutar el bootstrap
                AgentSystemManager(
                    config=enhanced_description,
                    bootstrap_models={"provider": bootstrap_provider, "model": bootstrap_model},
                    base_directory=str(project_dir),
                    verbose=verbose
                )

                # 6. Crear el archivo .env para el sistema final
                target_api_key_name = f"{target_provider.upper()}_API_KEY"
                target_api_key_value = ""

                if target_provider == bootstrap_provider:
                    target_api_key_value = bootstrap_api_key
                else:
                    target_api_key_value = _ask(f"Introduce tu {target_api_key_name} para el sistema final (opcional)")
                
                env_content = f"{target_api_key_name}={target_api_key_value}\n" if target_api_key_value else f"# {target_api_key_name}=\n"
                (project_dir / ".env").write_text(env_content)
                
                print("\n¡Proyecto MAS generado con éxito!")
                print("Archivos generados:")
                print(f"- {project_dir / 'config.json'}")
                print(f"- {project_dir / 'fns.py'}")
                print(f"- {project_dir / '.env'}")
                print("\nPróximos pasos:")
                if not target_api_key_value:
                    print(f"1. Edita el archivo `.env` y añade tu {target_api_key_name}.")
                print("2. Revisa `config.json` y `fns.py` para ver el sistema generado.")
                print("3. Ejecuta `mas start` de nuevo y crea un `main.py` para probarlo, ¡o crea el tuyo propio!")

            except Exception as e:
                print(f"\nOcurrió un error durante la generación del sistema: {e}")
                print("Por favor, verifica que tu API key de bootstrap sea correcta y que el modelo sea accesible.")

    else:
        print("Bienvenido al asistente de configuración de MAS.")
        print(f"Se crearán los archivos del proyecto en: {project_dir.resolve()}")

        default_models = {
            "google": "gemini-2.5-flash", "openai": "gpt-5-nano", "groq": "openai/gpt-oss-120b",
        }
        provider = _ask_choice("Elige un proveedor de LLM", list(default_models.keys()), "google")
        model = _ask("Introduce el nombre del modelo", default_models[provider])

        # 2. Pedir la API Key (opcional)
        api_key_name = f"{provider.upper()}_API_KEY"
        api_key_value = _ask(f"Introduce tu {api_key_name} (opcional, puedes dejarlo en blanco)")

        # 3. Crear los archivos
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

# >> AÑADE ESTA FUNCIÓN COMPLETA ANTES DE LA FUNCIÓN `main` <<

def run_system(project_dir_str: str, input_str: str = None, component_name: str = None, verbose: bool = False):
    """
    Carga y ejecuta un sistema MAS desde un directorio de proyecto.
    """
    project_dir = Path(project_dir_str)
    config_path = project_dir / "config.json"

    # 1. Verificar que el config.json exista
    if not config_path.is_file():
        print(f"Error: No se encontró 'config.json' en el directorio '{project_dir.resolve()}'.")
        print("Asegúrate de estar en el directorio correcto o crea un sistema con 'mas start'.")
        return 1

    print(f"Cargando sistema desde: {config_path.resolve()}")
    
    try:
        manager = AgentSystemManager(
            config=str(config_path),
            base_directory=str(project_dir),
            verbose=verbose
        )

        print("Ejecutando el sistema...")
        output = manager.run(
            input=input_str,
            component_name=component_name,
            verbose=verbose
        )

        print("\n--- Salida Final del Sistema ---")
        if isinstance(output, (dict, list)):
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(output)
        print("--------------------------------\n")
        
        return 0

    except Exception as e:
        print(f"\nOcurrió un error durante la ejecución: {e}")
        print("Verifica tu archivo 'config.json' y asegúrate de que tus API keys en '.env' son correctas.")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def main(argv=None):
    """Función principal de la CLI."""
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mas", description="CLI para la biblioteca Multi-Agent System (MAS).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Comando 'start'
    start_parser = subparsers.add_parser("start", help="Crea la estructura de un proyecto MAS.")
    start_parser.add_argument("--directory", "-d", default=".", help="Directorio donde crear el proyecto (defecto: actual).")
    start_parser.add_argument("--config", "-c", help="Descripción en lenguaje natural o ruta a un config.json para crear el sistema.")
    start_parser.add_argument("--verbose", "-v", action="store_true", help="Activa el logging detallado.")

    # --- Comando 'run' ---
    run_parser = subparsers.add_parser("run", help="Ejecuta un sistema MAS desde un config.json.")
    run_parser.add_argument("--directory", "-d", default=".", help="Directorio del proyecto MAS (defecto: actual).")
    run_parser.add_argument("--input", "-i", help="Input de texto para pasar al sistema.")
    run_parser.add_argument("--component", "-c", help="Nombre del componente específico a ejecutar.")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Activa el logging detallado.")

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