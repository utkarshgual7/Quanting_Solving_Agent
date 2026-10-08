"""SWE-bench-style task format and patch evaluation harness.

A task mirrors the fields of a SWE-bench instance:

    instance_id, repo, base_commit, problem_statement,
    patch        - the gold (reference) fix, a unified diff
    test_patch   - diff that adds the tests checking the fix (hidden from the model)
    FAIL_TO_PASS - tests that fail before the fix and must pass after it
    PASS_TO_PASS - tests that already pass and must keep passing (no regressions)

Evaluating a candidate patch works like the official harness, minus Docker:
copy the repo snapshot to a temp dir, apply the candidate patch, reset any test
files to their base version and apply test_patch (so a model cannot "fix" the
issue by editing the tests), run the listed tests with pytest, and parse the
per-test status from the log. The task is RESOLVED only if every FAIL_TO_PASS
and every PASS_TO_PASS test passes.

Local tasks keep their repo snapshot in tasks/<instance_id>/repo/, which plays
the role of `repo` checked out at `base_commit`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    repo_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, d: dict, repo_dir: Path | None = None) -> "Task":
        # The Hugging Face SWE-bench datasets store the test lists as JSON strings.
        def tests(v):
            return json.loads(v) if isinstance(v, str) else list(v)

        return cls(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d.get("base_commit", ""),
            problem_statement=d["problem_statement"],
            patch=d.get("patch", ""),
            test_patch=d.get("test_patch", ""),
            FAIL_TO_PASS=tests(d["FAIL_TO_PASS"]),
            PASS_TO_PASS=tests(d["PASS_TO_PASS"]),
            repo_dir=repo_dir,
        )


def load_tasks(root: Path = TASKS_DIR) -> list[Task]:
    return [
        Task.from_dict(json.loads((p / "task.json").read_text()), repo_dir=p / "repo")
        for p in sorted(root.iterdir())
        if (p / "task.json").exists()
    ]


def patched_files(diff: str) -> list[str]:
    """Paths touched by a unified diff (the b/ side of each `diff --git` header)."""
    return re.findall(r"^diff --git a/\S+ b/(\S+)$", diff, flags=re.M)


def parse_pytest_log(log: str) -> dict[str, str]:
    """Map test id -> PASSED/FAILED/ERROR/... from `pytest -rA` summary lines.

    Same approach as SWE-bench's pytest log parser: the short test summary
    prints one `STATUS test_id [- message]` line per test.
    """
    status = {}
    for line in log.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"}:
            status[parts[1]] = parts[0]
    return status


def _git_apply(cwd: Path, diff: str) -> bool:
    proc = subprocess.run(["git", "apply", "-"], cwd=cwd, input=diff, text=True, capture_output=True)
    return proc.returncode == 0


def evaluate_patch(task: Task, model_patch: str, timeout: int = 120) -> dict:
    """Apply `model_patch` to the task repo, run its tests, return a report dict."""
    report = {"instance_id": task.instance_id, "resolved": False, "status": "unresolved", "tests": {}}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(task.repo_dir, work)
        # Own git repo, so `git apply` never resolves paths against an enclosing repo.
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)

        if model_patch.strip() and not _git_apply(work, model_patch):
            report["status"] = "patch_failed"
            return report

        for rel in patched_files(task.test_patch):
            base, dst = task.repo_dir / rel, work / rel
            if base.exists():
                shutil.copy(base, dst)
            else:
                dst.unlink(missing_ok=True)
        if not _git_apply(work, task.test_patch):
            raise RuntimeError(f"{task.instance_id}: test_patch does not apply to the base repo")

        cmd = [sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider",
               *task.FAIL_TO_PASS, *task.PASS_TO_PASS]
        try:
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            report["status"] = "timeout"
            return report

    status = parse_pytest_log(proc.stdout)
    for group in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        tests = getattr(task, group)
        report["tests"][group] = {
            "success": [t for t in tests if status.get(t) == "PASSED"],
            "failure": [t for t in tests if status.get(t) != "PASSED"],
        }
    report["resolved"] = all(not g["failure"] for g in report["tests"].values())
    report["status"] = "resolved" if report["resolved"] else "unresolved"
    return report


def validate_task(task: Task) -> list[str]:
    """Sanity-check a task the way SWE-bench builds its splits.

    With no patch, FAIL_TO_PASS must fail and PASS_TO_PASS must pass; with the
    gold patch the task must be resolved. Returns a list of problems (empty = OK).
    """
    problems = []
    before = evaluate_patch(task, "")
    if before["tests"]["FAIL_TO_PASS"]["success"]:
        problems.append(f"FAIL_TO_PASS already pass without a fix: {before['tests']['FAIL_TO_PASS']['success']}")
    if before["tests"]["PASS_TO_PASS"]["failure"]:
        problems.append(f"PASS_TO_PASS fail on the base repo: {before['tests']['PASS_TO_PASS']['failure']}")
    gold = evaluate_patch(task, task.patch)
    if not gold["resolved"]:
        problems.append(f"gold patch does not resolve the task: {gold['tests']}")
    return problems
