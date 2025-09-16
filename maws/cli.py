import argparse
import sys
from .maws import (
    update as _update,
    start as _start,
    pull_history as _pull,
    setup as _setup,           # <- NEW
    describe as _describe,     # <- NEW
    list_projects as _list,    # <- NEW
    remove_project as _remove  # <- NEW
)

def _add_common_update_args(p: argparse.ArgumentParser):
    p.add_argument("-c", "--config", dest="config_path", default=None, help="Path to params.json (default: params.json)")
    p.add_argument("--dir", dest="project_dir", default=None, help="Project directory (default: cwd)")
    p.add_argument("--force-copy-script", action="store_true", help="Copy bootstrap.sh/pull_history.sh into the project before running (embedded script is used by default)")
    p.add_argument("--quiet", action="store_true", help="Do not stream logs (print only the last lines on failure)")
    p.add_argument("--allow-windows", action="store_true",
                   help="Allow running on Windows (not recommended).")
    # overrides
    p.add_argument("--project", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--bot", choices=["whatsapp","telegram"], default=None)
    p.add_argument("--set", dest="set_kv", action="append", default=[],
                   help="Override top-level keys (key=value). Repeatable, e.g. --set verbose=true --set tokens_s3_prefix=secrets")


def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser(prog="maws", description="AWS helpers for MAS bots")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_start = sp.add_parser("start", help="Create a minimal project structure")
    sp_start.add_argument("--project", default=None)
    sp_start.add_argument("--region", default=None)
    sp_start.add_argument("--bot", choices=["whatsapp", "telegram"], default=None)
    sp_start.add_argument("--dir", dest="project_dir", default=None)
    
    sp_start.add_argument("--overwrite", action="store_true")


    # install-deps: ON by default, with an option to disable
    sp_start.add_argument("--install-deps", dest="install_deps", action="store_true", default=True,
                        help="Attempt to install awscli/sam/jq automatically (default: ON)")
    sp_start.add_argument("--no-install-deps", dest="install_deps", action="store_false",
                        help="Do not install system dependencies")

    # run-config: ON by default, with an option to disable
    sp_start.add_argument("--run-config", dest="run_config", action="store_true", default=True,
                        help="Run 'aws configure' at the end (default: ON)")
    sp_start.add_argument("--no-run-config", dest="run_config", action="store_false",
                        help="Do not run 'aws configure' automatically")


    sp_start.add_argument("--allow-windows", action="store_true",
                      help="Allow running on Windows (not recommended).")

    sp_update = sp.add_parser("update", help="Run the bootstrap (deploy)")
    _add_common_update_args(sp_update)

    sp_pull = sp.add_parser("pull-history", help="Download ./history from S3 using params.json")
    sp_pull.add_argument("-c", "--config", dest="config_path", default=None, help="Path to params.json (default: params.json)")
    sp_pull.add_argument("--dir", dest="project_dir", default=None)
    sp_pull.add_argument("--force-copy-script", action="store_true")
    sp_pull.add_argument("--quiet", action="store_true")

    sp_setup = sp.add_parser("setup", help="Interactive guide to complete params.json")
    sp_setup.add_argument("-c", "--config", dest="config_path", default=None)
    sp_setup.add_argument("--dir", dest="project_dir", default=None)

    sp_desc = sp.add_parser("describe", help="Describe the current project and deployment")
    sp_desc.add_argument("-c", "--config", dest="config_path", default=None)
    sp_desc.add_argument("--dir", dest="project_dir", default=None)
    sp_desc.add_argument("--region", default=None)
    sp_desc.add_argument("--no-aws", action="store_true", help="Do not query AWS")

    sp_list = sp.add_parser("list", help="List MAWS stacks in the current region")
    sp_list.add_argument("--region", default=None)

    sp_rm = sp.add_parser("remove", help="Remove AWS resources for a project")
    sp_rm.add_argument("--project", default=None, help="If omitted, it is read from params.json")
    sp_rm.add_argument("--region", default=None)
    sp_rm.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    sp_rm.add_argument("--keep-deploy-bucket", action="store_true", help="Keep the deployment bucket")
    sp_rm.add_argument("--wait", action="store_true", help="Wait until delete-stack completes")
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
