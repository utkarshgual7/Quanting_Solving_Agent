import csv
import json
from pathlib import Path

import pytest

from evals.review import (CRITERIA, Review, agreement, export_for_human_review, import_human_reviews, judge_prompt,
                          llm_judge, load_items, load_reviews, mock_judge, save_reviews)

SAMPLE = Path(__file__).parents[1] / "data" / "review_items.jsonl"
ITEM = {"id": "r1", "prompt": "Solve x + 1 = 3", "reference": "x = 2", "output": "1. Subtract 1.\nx = 2"}


def reply(scores):
    return "Sure:\n```json\n" + json.dumps({c: {"score": s, "rationale": f"because {c}"}
                                            for c, s in zip(CRITERIA, scores)}) + "\n```"


def test_llm_judge_parses_json_reply():
    seen = []
    fake = lambda prompt: seen.append(prompt) or reply([5, 4, 5, 3, 5])
    review = llm_judge(ITEM, fake, "fake")
    assert review.reviewer == "judge:fake"
    assert review.scores["formatting"] == 3
    assert review.overall() == pytest.approx(4.4)
    assert "Solve x + 1 = 3" in seen[0] and "grounding" in seen[0]


@pytest.mark.parametrize("bad", [[6, 4, 5, 3, 5], [0, 4, 5, 3, 5], ["5", 4, 5, 3, 5]])
def test_out_of_range_or_non_int_scores_rejected(bad):
    with pytest.raises(ValueError):
        llm_judge(ITEM, lambda p: reply(bad), "fake")


def test_missing_criterion_or_rationale_rejected():
    with pytest.raises(ValueError):
        llm_judge(ITEM, lambda p: '{"correctness": {"score": 5, "rationale": "ok"}}', "fake")
    with pytest.raises(ValueError):
        Review("x", "h", {c: 3 for c in CRITERIA}, {c: "" for c in CRITERIA}).validate()


def test_mock_judge_on_sample_items():
    items = {i["id"]: i for i in load_items(SAMPLE)}
    assert mock_judge(items["r1"]).scores["correctness"] == 5
    assert mock_judge(items["r3"]).scores["correctness"] == 2
    assert mock_judge(items["r1"]).validate()


def test_human_csv_round_trip(tmp_path):
    path = tmp_path / "review.csv"
    export_for_human_review([ITEM, {**ITEM, "id": "r2"}], path)
    rows = path.read_text().splitlines()
    assert rows[0].startswith("item_id,prompt,output,reference,correctness_score")
    # a reviewer fills in r1 and leaves r2 blank
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    for c in CRITERIA:
        data[0][f"{c}_score"], data[0][f"{c}_rationale"] = "4", f"{c} looks fine"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    reviews = import_human_reviews(path, reviewer="alice")
    assert [r.item_id for r in reviews] == ["r1"]
    assert reviews[0].reviewer == "human:alice" and reviews[0].overall() == 4


def test_jsonl_round_trip(tmp_path):
    r = mock_judge(ITEM)
    save_reviews([r], tmp_path / "r.jsonl")
    assert load_reviews(tmp_path / "r.jsonl") == [r]


def test_prompt_lists_every_criterion():
    assert all(c in judge_prompt(ITEM) for c in CRITERIA)


def test_agreement_between_judge_and_human():
    def rev(item_id, who, s):
        return Review(item_id, who, {c: s for c in CRITERIA}, {c: "r" for c in CRITERIA})

    judge = [rev("a", "judge:x", 5), rev("b", "judge:x", 1), rev("c", "judge:x", 3), rev("only-judge", "judge:x", 2)]
    human = [rev("a", "human:h", 5), rev("b", "human:h", 1), rev("c", "human:h", 4)]
    report = agreement(judge, human)
    corr = report["correctness"]
    assert corr["n"] == 3
    assert corr["exact_agreement"] == pytest.approx(2 / 3)
    assert corr["quadratic_kappa"] > corr["kappa"]  # the one miss is only 1 point off
