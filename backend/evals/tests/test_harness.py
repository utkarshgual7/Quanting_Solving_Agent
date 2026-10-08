import json

from evals.harness import Task, evaluate_patch, load_tasks, parse_pytest_log, patched_files, validate_task

TASK = next(t for t in load_tasks() if t.instance_id == "quanting__answer-check-1")


def test_gold_patch_resolves():
    report = evaluate_patch(TASK, TASK.patch)
    assert report["resolved"]
    assert report["tests"]["FAIL_TO_PASS"]["failure"] == []


def test_empty_patch_not_resolved():
    report = evaluate_patch(TASK, "")
    assert not report["resolved"]
    # the bug is still there, but nothing regressed
    assert set(report["tests"]["FAIL_TO_PASS"]["failure"]) == set(TASK.FAIL_TO_PASS)
    assert report["tests"]["PASS_TO_PASS"]["failure"] == []


def test_malformed_patch_reported():
    assert evaluate_patch(TASK, "this is not a diff")["status"] == "patch_failed"


def test_fix_that_breaks_existing_behaviour_not_resolved():
    # Makes FAIL_TO_PASS pass by requiring an exact match, which breaks PASS_TO_PASS.
    bad = TASK.patch.replace("    return correct in agent\n", "    return correct == agent\n")
    assert bad != TASK.patch
    report = evaluate_patch(TASK, bad)
    assert report["tests"]["FAIL_TO_PASS"]["failure"] == []
    assert report["tests"]["PASS_TO_PASS"]["failure"]
    assert not report["resolved"]


def test_editing_tests_does_not_help():
    # A "fix" that adds the hidden tests itself, with the assertions gutted. The
    # harness resets test files to base and re-applies the real test_patch.
    cheat = TASK.test_patch.replace("assert not is_correct", "assert True or is_correct")
    assert cheat != TASK.test_patch
    report = evaluate_patch(TASK, cheat)
    assert not report["resolved"]


def test_validate_task():
    assert validate_task(TASK) == []


def test_parse_pytest_log():
    log = "PASSED tests/t.py::a\nFAILED tests/t.py::b - AssertionError\nnoise line\n"
    assert parse_pytest_log(log) == {"tests/t.py::a": "PASSED", "tests/t.py::b": "FAILED"}


def test_patched_files():
    assert patched_files(TASK.patch) == ["answer_check.py"]


def test_from_dict_accepts_hf_json_strings():
    row = {"instance_id": "x__y-1", "repo": "x/y", "base_commit": "abc", "problem_statement": "p",
           "patch": "", "test_patch": "", "FAIL_TO_PASS": json.dumps(["t::a"]), "PASS_TO_PASS": "[]"}
    task = Task.from_dict(row)
    assert task.FAIL_TO_PASS == ["t::a"] and task.PASS_TO_PASS == []
