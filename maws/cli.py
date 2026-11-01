import argparse
import sys
from .maws import (
    update as _update,
    start as _start,
    pull_history as _pull,
    setup as _setup,
    describe as _describe,
    list_projects as _list,
    remove_project as _remove,
)
from .operations import MawsConfigError

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
                   help="Override params (key=value; dotted keys for nested params). Repeatable, e.g. "
                        "--set verbose=true --set infra.tracing=false --set requirements_ref=v0.3.4. "
                        "Values stay text unless they are exact JSON literals (true, 120, null, {...}).")


def main(argv=None):
    try:
        return _main(argv)
    except MawsConfigError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2


def _main(argv=None):
    argv = argv or sys.argv[1:]
    ap = argparse.ArgumentParser(prog="maws", description="AWS helpers for MAS bots")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_start = sp.add_parser("start", help="Create a minimal project structure")
    sp_start.add_argument("--project", default=None)
    sp_start.add_argument("--region", default=None)
    sp_start.add_argument("--bot", choices=["whatsapp", "telegram"], default=None)
    sp_start.add_argument("--dir", dest="project_dir", default=None)
    
    sp_start.add_argument("--overwrite", action="store_true",
                          help="Rewrite config.json, fns.py, .env.prod and .gitignore, and let --project/--region/--bot "
                               "replace the values in an existing params.json (other params.json keys are kept)")


    # install-deps: opt-in (uses sudo and unverified downloads), like the Python API
    sp_start.add_argument("--install-deps", dest="install_deps", action="store_true", default=False,
                        help="Install curl/unzip/jq (apt-get) and the latest AWS CLI / SAM CLI with sudo "
                             "(downloads are not checksum-verified; default: off)")
    sp_start.add_argument("--no-install-deps", dest="install_deps", action="store_false",
                        help="Do not install system dependencies (default)")

    # run-config: opt-in, and skipped when AWS credentials already work
    sp_start.add_argument("--run-config", dest="run_config", action="store_true", default=False,
                        help="Run 'aws configure' at the end if AWS credentials do not work yet (default: off)")
    sp_start.add_argument("--no-run-config", dest="run_config", action="store_false",
                        help="Do not run 'aws configure' (default)")


    sp_start.add_argument("--allow-windows", action="store_true",
                      help="Allow running on Windows (not recommended).")

    sp_update = sp.add_parser("update", help="Run the bootstrap (deploy)")
    _add_common_update_args(sp_update)

    sp_pull = sp.add_parser("pull-history", help="Download ./history from S3 using params.json")
    sp_pull.add_argument("-c", "--config", dest="config_path", default=None, help="Path to params.json (default: params.json)")
    sp_pull.add_argument("--dir", dest="project_dir", default=None)
    sp_pull.add_argument("--force-copy-script", action="store_true")
    sp_pull.add_argument("--quiet", action="store_true")
    sp_pull.add_argument("--user-id", dest="user_ids", action="append", default=None,
                         help="Download only one user's history. Repeat for multiple users.")
    sp_pull.add_argument("--user-ids", dest="user_ids_csv", default=None,
                         help="Comma-separated list of user IDs to download.")

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
    sp_rm.add_argument("--keep-deploy-bucket", action="store_true",
                       help="Keep the deployment artifacts (by default s3://<deployment_bucket>/<stack>/ is deleted, "
                            "and the bucket too if nothing else is left in it)")
    sp_rm.add_argument("--wait", dest="wait", action="store_true", default=True,
                       help="Wait until the stack is deleted, then delete its secrets and artifacts (default)")
    sp_rm.add_argument("--no-wait", dest="wait", action="store_false",
                       help="Only start the stack deletion; keeps the SSM parameter and deployment files "
                            "(prints the commands to delete them once the stack is gone)")
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
        user_ids = []
        if args.user_ids:
            user_ids.extend(args.user_ids)
        if args.user_ids_csv:
            user_ids.append(args.user_ids_csv)
        code = _pull(config_path=args.config_path, project_dir=args.project_dir,
                     force_copy_script=args.force_copy_script, quiet=args.quiet,
                     user_ids=user_ids or None)
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
