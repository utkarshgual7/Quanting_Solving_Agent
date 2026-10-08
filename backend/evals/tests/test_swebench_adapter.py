import json

from evals.harness import Task
from evals.swebench_adapter import summarize, write_predictions

# Shape of a row from princeton-nlp/SWE-bench_Lite (test lists are JSON strings).
ROW = {
    "repo": "astropy/astropy", "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607", "problem_statement": "...",
    "patch": "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n",
    "test_patch": "", "FAIL_TO_PASS": '["t::a", "t::b"]', "PASS_TO_PASS": '["t::c"]',
}


def test_summarize_hf_row():
    s = summarize(Task.from_dict(ROW))
    assert s == {"instance_id": "astropy__astropy-12907", "repo": "astropy/astropy",
                 "base_commit": "d16bfe05a7", "gold_patch_files": ["astropy/modeling/separable.py"],
                 "FAIL_TO_PASS": 2, "PASS_TO_PASS": 1}


def test_predictions_format(tmp_path):
    path = tmp_path / "preds.jsonl"
    write_predictions({"a__b-1": "diff"}, "gold", path)
    assert json.loads(path.read_text()) == {"instance_id": "a__b-1", "model_name_or_path": "gold",
                                            "model_patch": "diff"}
