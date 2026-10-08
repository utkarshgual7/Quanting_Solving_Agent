"""Command line entry point: run from backend/ as `python -m evals <command>`."""
import argparse
import json
import sys

from pathlib import Path

from evals.harness import load_tasks, validate_task
from evals.models import get_model
from evals.runner import run_eval, to_markdown, write_report
from evals.swebench_adapter import fetch_instances, summarize, write_predictions


def cmd_validate(args) -> int:
    failed = 0
    for task in load_tasks():
        problems = validate_task(task)
        print(f"{'OK  ' if not problems else 'FAIL'} {task.instance_id}")
        for p in problems:
            print(f"     - {p}")
        failed += bool(problems)
    return 1 if failed else 0


def cmd_run(args) -> int:
    report = run_eval(get_model(args.model), load_tasks(), n=args.n, ks=args.k)
    print(to_markdown(report))
    for path in write_report(report, Path(args.out)):
        print(f"wrote {path}")
    return 0


def cmd_swebench(args) -> int:
    tasks = fetch_instances(args.dataset, limit=args.limit)
    for task in tasks:
        print(json.dumps(summarize(task)))
    if args.write_gold_preds:
        write_predictions({t.instance_id: t.patch for t in tasks}, "gold", Path(args.write_gold_preds))
        print(f"wrote gold predictions for {len(tasks)} instances to {args.write_gold_preds}")
    print("dry run only: score these with the official harness (see evals/README.md)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check every local task: base fails, gold patch resolves")
    run = sub.add_parser("run", help="sample n patches per task from a model and report pass@k")
    run.add_argument("--model", default="mock", help="mock (offline) or anthropic (needs ANTHROPIC_API_KEY)")
    run.add_argument("-n", type=int, default=10, help="samples per task")
    run.add_argument("-k", type=int, nargs="+", default=[1, 5], help="k values for pass@k")
    run.add_argument("--out", default="eval-results", help="directory for the JSON and Markdown report")
    swe = sub.add_parser("swebench", help="dry run: load SWE-bench Lite/Verified instances from Hugging Face")
    swe.add_argument("--dataset", default="lite", help="lite, verified, or a full HF dataset name")
    swe.add_argument("--limit", type=int, default=5)
    swe.add_argument("--write-gold-preds", metavar="PATH", help="write a predictions JSONL of the gold patches")
    args = parser.parse_args(argv)
    return {"validate": cmd_validate, "run": cmd_run, "swebench": cmd_swebench}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
