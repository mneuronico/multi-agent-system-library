import argparse
import sys
from .maws import (
    update as _update,
    start as _start,
    pull_history as _pull,
    setup as _setup,           # <- NUEVO
    describe as _describe,     # <- NUEVO
    list_projects as _list,    # <- NUEVO
    remove_project as _remove  # <- NUEVO
)

def _add_common_update_args(p: argparse.ArgumentParser):
    p.add_argument("-c", "--config", dest="config_path", default=None, help="Ruta a params.json (default: params.json)")
    p.add_argument("--dir", dest="project_dir", default=None, help="Directorio del proyecto (default: cwd)")
    p.add_argument("--force-copy-script", action="store_true", help="Copia bootstrap.sh/pull_history.sh al proyecto antes de ejecutar (por defecto usa el script embebido)")
    p.add_argument("--quiet", action="store_true", help="No streamea logs (muestra últimas líneas si falla)")
    p.add_argument("--allow-windows", action="store_true",
                   help="Permite ejecutar en Windows (no recomendado).")
    # overrides
    p.add_argument("--project", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--bot", choices=["whatsapp","telegram"], default=None)
    p.add_argument("--set", dest="set_kv", action="append", default=[],
                   help="Override top-level (clave=valor). Repetible. Ej: --set verbose=true --set tokens_s3_prefix=secrets")


def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser(prog="maws", description="Helpers AWS para MAS bots")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_start = sp.add_parser("start", help="Crea estructura mínima de proyecto")
    sp_start.add_argument("--project", default=None)
    sp_start.add_argument("--region", default=None)
    sp_start.add_argument("--bot", choices=["whatsapp", "telegram"], default=None)
    sp_start.add_argument("--dir", dest="project_dir", default=None)
    
    sp_start.add_argument("--overwrite", action="store_true")


    # install-deps: ON por defecto, con forma de desactivarlo
    sp_start.add_argument("--install-deps", dest="install_deps", action="store_true", default=True,
                        help="Intentar instalar awscli/sam/jq automáticamente (default: ON)")
    sp_start.add_argument("--no-install-deps", dest="install_deps", action="store_false",
                        help="No instalar dependencias del sistema")

    # run-config: ON por defecto, con forma de desactivarlo
    sp_start.add_argument("--run-config", dest="run_config", action="store_true", default=True,
                        help="Ejecutar 'aws configure' al final (default: ON)")
    sp_start.add_argument("--no-run-config", dest="run_config", action="store_false",
                        help="No ejecutar 'aws configure' automáticamente")
    
    
    sp_start.add_argument("--allow-windows", action="store_true",
                      help="Permite ejecutar en Windows (no recomendado).")

    sp_update = sp.add_parser("update", help="Ejecución de bootstrap (deploy)")
    _add_common_update_args(sp_update)

    sp_pull = sp.add_parser("pull-history", help="Descargar ./history desde S3 usando params.json")
    sp_pull.add_argument("-c", "--config", dest="config_path", default=None, help="Ruta a params.json (default: params.json)")
    sp_pull.add_argument("--dir", dest="project_dir", default=None)
    sp_pull.add_argument("--force-copy-script", action="store_true")
    sp_pull.add_argument("--quiet", action="store_true")

    sp_setup = sp.add_parser("setup", help="Guía interactiva para completar params.json")
    sp_setup.add_argument("-c", "--config", dest="config_path", default=None)
    sp_setup.add_argument("--dir", dest="project_dir", default=None)

    sp_desc = sp.add_parser("describe", help="Describe el proyecto actual y su despliegue")
    sp_desc.add_argument("-c", "--config", dest="config_path", default=None)
    sp_desc.add_argument("--dir", dest="project_dir", default=None)
    sp_desc.add_argument("--region", default=None)
    sp_desc.add_argument("--no-aws", action="store_true", help="No consultar AWS")

    sp_list = sp.add_parser("list", help="Lista stacks MAWS de la región actual")
    sp_list.add_argument("--region", default=None)

    sp_rm = sp.add_parser("remove", help="Elimina recursos AWS de un proyecto")
    sp_rm.add_argument("--project", default=None, help="Si no se pasa, lo toma de params.json")
    sp_rm.add_argument("--region", default=None)
    sp_rm.add_argument("-y", "--yes", action="store_true", help="No pedir confirmación")
    sp_rm.add_argument("--keep-deploy-bucket", action="store_true", help="Conservar deployment bucket")
    sp_rm.add_argument("--wait", action="store_true", help="Esperar a que termine el delete-stack")
    sp_rm.add_argument("-c", "--config", dest="config_path", default=None)
    sp_rm.add_argument("--dir", dest="project_dir", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "start":
        _start(project=args.project, region=args.region, bot=args.bot,
               project_dir=args.project_dir, overwrite=args.overwrite,
               install_deps=args.install_deps, run_config=args.run_config, allow_windows=args.allow_windows)
        return 0

    if args.cmd == "update":
        code = _update(
            params=None,
            config_path=args.config_path,
            project_dir=args.project_dir,
            project=args.project,
            region=args.region,
            bot=args.bot,
            set_kv=args.set_kv,
            force_copy_script=args.force_copy_script,
            quiet=args.quiet,
            allow_windows=args.allow_windows
        )
        return code

    if args.cmd == "pull-history":
        code = _pull(config_path=args.config_path, project_dir=args.project_dir,
                     force_copy_script=args.force_copy_script, quiet=args.quiet)
        return code
    
    if args.cmd == "setup":
        _setup(config_path=args.config_path, project_dir=args.project_dir)
        return 0

    if args.cmd == "describe":
        return _describe(config_path=args.config_path, project_dir=args.project_dir,
                         region=args.region, no_aws=args.no_aws)

    if args.cmd == "list":
        _list(region=args.region)
        return 0

    if args.cmd == "remove":
        return _remove(project=args.project, region=args.region, yes=args.yes,
                       keep_deploy_bucket=args.keep_deploy_bucket, config_path=args.config_path,
                       project_dir=args.project_dir, wait=args.wait)

    ap.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
