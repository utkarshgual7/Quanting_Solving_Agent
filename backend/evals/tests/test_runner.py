import json

import pytest

from evals.harness import load_tasks
from evals.models import MockModel, extract_diff
from evals.runner import run_eval, write_report

TASKS = load_tasks()


def test_perfect_and_hopeless_mock():
    ids = [t.instance_id for t in TASKS]
    model = MockModel(skill={ids[0]: 1.0, ids[1]: 0.0, ids[2]: 1.0})
    report = run_eval(model, TASKS, n=4, ks=[1, 2])
    by_id = {r["instance_id"]: r for r in report["tasks"]}
    assert by_id[ids[0]]["c"] == 4 and by_id[ids[1]]["c"] == 0
    assert report["summary"]["pass@1"] == pytest.approx(2 / 3)
    assert report["mock"] is True


def test_mock_is_deterministic():
    assert [MockModel().generate(TASKS[0], i) for i in range(10)] == \
           [MockModel().generate(TASKS[0], i) for i in range(10)]


def test_ks_larger_than_n_are_dropped():
    report = run_eval(MockModel(default_skill=1.0), TASKS[:1], n=2, ks=[1, 5])
    assert list(report["summary"]) == ["pass@1"]


def test_write_report(tmp_path):
    report = run_eval(MockModel(default_skill=1.0), TASKS[:1], n=1, ks=[1])
    json_path, md_path = write_report(report, tmp_path)
    assert json.loads(json_path.read_text())["summary"]["pass@1"] == 1.0
    assert "Mock model run" in md_path.read_text()


def test_extract_diff():
    reply = "Here is the fix:\n```diff\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```\nDone."
    assert extract_diff(reply).startswith("diff --git a/x.py")
    assert extract_diff(reply).endswith("+b\n")
    assert extract_diff("no patch here") == ""
