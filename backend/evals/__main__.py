"""Command line entry point: run from backend/ as `python -m evals <command>`."""
import argparse
import sys

from evals.harness import load_tasks, validate_task


def cmd_validate(args) -> int:
    failed = 0
    for task in load_tasks():
        problems = validate_task(task)
        print(f"{'OK  ' if not problems else 'FAIL'} {task.instance_id}")
        for p in problems:
            print(f"     - {p}")
        failed += bool(problems)
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check every local task: base fails, gold patch resolves")
    args = parser.parse_args(argv)
    return {"validate": cmd_validate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
