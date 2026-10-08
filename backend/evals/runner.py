"""Generate n patches per task, grade each with the harness, report pass@k."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evals.harness import Task, evaluate_patch
from evals.metrics import mean_pass_at_k, pass_at_k


def run_eval(model, tasks: list[Task], n: int, ks: list[int]) -> dict:
    ks = [k for k in ks if k <= n]
    rows = []
    for task in tasks:
        graded: dict[str, dict] = {}  # grading is deterministic, so identical patches are run once
        statuses = []
        for i in range(n):
            patch = model.generate(task, i)
            if patch not in graded:
                graded[patch] = evaluate_patch(task, patch)
            statuses.append(graded[patch]["status"])
        c = statuses.count("resolved")
        rows.append({"instance_id": task.instance_id, "n": n, "c": c,
                     **{f"pass@{k}": pass_at_k(n, c, k) for k in ks}, "samples": statuses})
    return {
        "model": model.name,
        "mock": model.name == "mock",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": n,
        "summary": {f"pass@{k}": mean_pass_at_k([(r["n"], r["c"]) for r in rows], k) for k in ks},
        "tasks": rows,
    }


def to_markdown(report: dict) -> str:
    ks = list(report["summary"])
    lines = [f"# Patch eval report: {report['model']}", ""]
    if report["mock"]:
        lines += ["> **Mock model run.** The mock returns the gold patch at a fixed rate; "
                  "these numbers test the pipeline and are not a score for any real model.", ""]
    lines += [f"- run at: {report['timestamp']}", f"- samples per task (n): {report['n']}",
              f"- tasks: {len(report['tasks'])}", "",
              "| task | correct / n | " + " | ".join(ks) + " |",
              "|---|---|" + "---|" * len(ks)]
    for r in report["tasks"]:
        lines.append(f"| {r['instance_id']} | {r['c']} / {r['n']} | " + " | ".join(f"{r[k]:.3f}" for k in ks) + " |")
    lines.append("| **mean** | | " + " | ".join(f"**{v:.3f}**" for v in report["summary"].values()) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"report_{report['model'].replace('/', '_')}"
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(to_markdown(report))
    return json_path, md_path
