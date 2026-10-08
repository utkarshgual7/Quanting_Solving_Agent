"""Load real SWE-bench Lite / Verified instances and export predictions.

This is metadata only. The official evaluation builds a Docker image per
instance (the real repo at base_commit with its pinned environment) and is run
with the `swebench` package; we do not re-implement or fake it. What this module
does:

- fetch instances from the Hugging Face datasets-server REST API (stdlib only,
  no `datasets` dependency) and map them onto the same Task schema the local
  harness uses;
- print a dry-run summary (repo, base commit, files the gold patch touches,
  number of FAIL_TO_PASS / PASS_TO_PASS tests);
- write a predictions JSONL in the format the official harness reads:
  {"instance_id", "model_name_or_path", "model_patch"}.

Then, on an x86_64 machine with Docker:

    pip install swebench
    python -m swebench.harness.run_evaluation \\
        --dataset_name princeton-nlp/SWE-bench_Lite \\
        --predictions_path preds.jsonl --max_workers 4 --run_id my-run
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from evals.harness import Task, patched_files

DATASETS = {"lite": "princeton-nlp/SWE-bench_Lite", "verified": "princeton-nlp/SWE-bench_Verified"}
ROWS_API = "https://datasets-server.huggingface.co/rows"


def fetch_instances(dataset: str, limit: int = 5, offset: int = 0) -> list[Task]:
    query = urllib.parse.urlencode({"dataset": DATASETS.get(dataset, dataset), "config": "default",
                                    "split": "test", "offset": offset, "length": min(limit, 100)})
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=30) as resp:
        rows = json.load(resp)["rows"]
    return [Task.from_dict(r["row"]) for r in rows]


def summarize(task: Task) -> dict:
    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "base_commit": task.base_commit[:10],
        "gold_patch_files": patched_files(task.patch),
        "FAIL_TO_PASS": len(task.FAIL_TO_PASS),
        "PASS_TO_PASS": len(task.PASS_TO_PASS),
    }


def write_predictions(predictions: dict[str, str], model_name: str, path: Path) -> None:
    """predictions maps instance_id -> model_patch (unified diff)."""
    with open(path, "w") as f:
        for instance_id, patch in predictions.items():
            f.write(json.dumps({"instance_id": instance_id, "model_name_or_path": model_name,
                                "model_patch": patch}) + "\n")
