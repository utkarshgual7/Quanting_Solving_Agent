# Evaluation tooling

The `evals` package has the evaluation tooling for the maths agent. It needs only the Python standard library and `pytest`, so it runs without the rest of the backend's dependencies (agno, LanceDB, Gemini and so on).

It has three parts:

1. **pass@k metrics** (`metrics.py`): the unbiased estimator from the Codex paper.
2. **A SWE-bench-style patch harness** (`harness.py`, `models.py`, `runner.py`, `tasks/`). It has three local tasks built from real bugs in this repo, a pluggable model interface, and pass@k reports. There is also an adapter (`swebench_adapter.py`) that loads real SWE-bench Lite and Verified instances and exports predictions for the official harness.
3. **Rubric review of free-text outputs** (`review.py`). It scores outputs on five criteria (1-5 each, with rationales), can use an LLM as the judge, round-trips human review through CSV, and measures judge-vs-human agreement with Cohen's kappa.

Run everything from `backend/`:

```bash
pip install pytest
python -m pytest evals -q                         # unit tests
python -m evals validate                          # check every task: base fails, gold patch resolves
python -m evals run --model mock -n 10 -k 1 5     # mock-model eval -> eval-results/report_mock.{json,md}
python -m evals swebench --dataset lite --limit 5 # dry run on real SWE-bench Lite metadata
python -m evals review judge                      # rubric-review the sample outputs (mock judge)
python -m evals review export --out review.csv    # CSV for human reviewers
python -m evals review agree --judged eval-results/reviews.jsonl --human review.csv
```

For a real model, run `pip install anthropic`, set `ANTHROPIC_API_KEY`, then pass `--model anthropic` (or `--judge anthropic` for reviews). `EVAL_MODEL` overrides the default model id (`claude-opus-5-5`).

---

## 1. pass@k

pass@k is the probability that **at least one of k samples** from the model solves a task. To measure it you could draw k samples once, but that estimate is noisy. Chen et al. 2021 (*Evaluating Large Language Models Trained on Code*) instead draw n >= k samples, count the c correct ones, and use the unbiased estimator

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

`C(n - c, k) / C(n, k)` is the chance that a random size-k subset of the n samples contains only failures, so one minus that is the chance it contains at least one success. The binomials overflow for large n, so `pass_at_k` uses the paper's equivalent product form, `1 - prod_{i=n-c+1..n} (1 - k/i)`. The benchmark score is the mean over tasks.

- **pass@1** reduces to c/n, the expected success rate of a single attempt. SWE-bench leaderboards report this, as "% resolved" with one attempt per issue.
- **pass@k** for k > 1 measures whether the model *can* solve the task given several tries, for example with a verifier or tests to pick the right attempt.

A worked example from the tests: with n = 10 and c = 3, pass@1 = 0.3 and pass@5 = 1 - C(7,5)/C(10,5) = 1 - 21/252 = 0.917.

## 2. SWE-bench-style patch harness

### How it maps to SWE-bench

A SWE-bench instance is a real GitHub issue together with the repository at the commit before it was fixed. The model sees the issue text and the code and must produce a patch. Each local task in `tasks/<instance_id>/task.json` uses the same fields:

| field | meaning |
|---|---|
| `instance_id`, `repo`, `base_commit` | which repo, and at which commit. Local tasks use a snapshot in `tasks/<id>/repo/` instead of a real commit (`base_commit: "local-fixture"`). |
| `problem_statement` | the issue text given to the model |
| `patch` | the **gold** fix (unified diff), used to verify the task itself |
| `test_patch` | diff that adds the tests checking the fix; the model never sees it |
| `FAIL_TO_PASS` | tests that fail before the fix and must pass after it |
| `PASS_TO_PASS` | tests that pass before and must still pass (no regressions) |

`evaluate_patch(task, candidate_patch)` follows the official evaluation flow, without Docker:

1. Copy the repo snapshot to a temp dir and apply the candidate patch with `git apply`. If it does not apply, the status is `patch_failed`.
2. Reset every file touched by `test_patch` to its base version, then apply `test_patch`. This means a model cannot pass by editing the tests.
3. Run the FAIL_TO_PASS and PASS_TO_PASS tests with `pytest -rA`, then parse the per-test `PASSED`/`FAILED` lines from the log. SWE-bench's pytest log parser works the same way.
4. The task is **resolved only if every FAIL_TO_PASS and every PASS_TO_PASS test passes**. This is the same definition SWE-bench uses. Fixing the bug while breaking something else does not count, and neither does leaving the bug in place.

`python -m evals validate` checks every task the way SWE-bench builds its splits. With no patch, FAIL_TO_PASS must fail and PASS_TO_PASS must pass. With the gold patch, the task must be resolved. The tests also check four bad patches: an empty patch, a malformed diff, a fix that breaks a PASS_TO_PASS test, and a patch that tries to plant gutted copies of the hidden tests. All four must come back not resolved.

### The local tasks (real bugs from this repo)

Each task is a small, self-contained fixture adapted from code in `backend/app`. The bug is the real one. The fixture strips out framework imports (FastAPI, agno) so the tests run anywhere.

| task | source | bug |
|---|---|---|
| `quanting__answer-check-1` | `app/utils/evaluation.py`, `_evaluate_answer` | the benchmark checker uses substring matching, so expected answer `5` matches `x = 15`, and `2` matches `x = 2.5` |
| `quanting__guardrails-2` | `app/core/guardrails.py`, `_is_educational_content` | `"What is 2 + 2?"` is rejected as not maths, because the arithmetic regex allows no spaces |
| `quanting__rate-limit-3` | `app/core/guardrails.py`, `_check_rate_limit` | "max 20 requests per minute" lets 21 through (`>` instead of `>=`) |

These bugs are still present in the app code. The tasks only use them as eval material.

### Models and the runner

A model is anything with `name` and `generate(task, sample_idx) -> diff`. The runner samples **n** patches per task, grades each one with the harness, and computes per-task pass@k and the mean. It then writes `report_<model>.json` and `report_<model>.md`.

- `MockModel` is deterministic and offline, and is used by the tests and CI. It returns the gold patch at a fixed rate (50% by default). Its other samples are either an empty patch or a malformed one. **Mock reports test the pipeline. They are not a score for any model.** `results/report_mock.md` is one such run.
- `AnthropicModel` sends the issue and the repo files to Claude, asks for a unified diff in a fenced block, and extracts the diff. This provider has not been run in this repo yet, so there are no real-model numbers here.

Security note: grading runs the patched code and its tests on your machine. The official harness uses Docker containers for isolation. Run untrusted model output inside a container or VM.

### Real SWE-bench Lite / Verified

`swebench_adapter.py` fetches instances from the Hugging Face datasets-server API. It uses only the stdlib, with no `datasets` dependency. The instances are `princeton-nlp/SWE-bench_Lite` (300 test instances) and `princeton-nlp/SWE-bench_Verified` (500 instances that human annotators confirmed are well-specified and fairly tested). The adapter maps them onto the same `Task` class, prints a dry-run summary, and can write a predictions file in the official format:

```bash
python -m evals swebench --dataset lite --limit 3 --write-gold-preds preds.jsonl
# {"instance_id": "astropy__astropy-12907", "repo": "astropy/astropy", "base_commit": "d16bfe05a7",
#  "gold_patch_files": ["astropy/modeling/separable.py"], "FAIL_TO_PASS": 2, "PASS_TO_PASS": 13}
```

This repo does **not** score real SWE-bench instances. Scoring needs each instance's repo and environment, which the official harness builds as Docker images. With Docker on an x86_64 machine:

```bash
pip install swebench
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path preds.jsonl --max_workers 4 --run_id my-run
```

Each line of `preds.jsonl` is `{"instance_id", "model_name_or_path", "model_patch"}`. Running the gold predictions first is a sanity check that the setup works.

## 3. Rubric review of free-text outputs

Not every output can be graded by tests. Worked solutions, explanations and chat answers need rubric grading. `review.py` scores each output on five criteria, each from 1 to 5 with a one-sentence rationale:

| criterion | what it checks |
|---|---|
| correctness | final answer and reasoning are mathematically right |
| instruction_following | answers what was asked, in the requested format |
| grounding | no hallucinated theorems, citations or numbers |
| formatting | clear steps, explicit final answer |
| safety | appropriate for a student |

- **LLM-as-judge.** `llm_judge(item, complete, name)` builds a prompt containing the rubric, question, reference and response, and asks for JSON. It then validates the reply: every criterion must be present, scores must be integers from 1 to 5, and every rationale must be non-empty. Invalid replies raise an error instead of being scored silently. `complete` is any `str -> str` function, so a fake can stand in during tests.
- **Mock judge.** It uses crude string checks and exists only for offline demos. It is also a good example of why judges need checking. On sample `r4`, it gives full correctness to "9.0 approximately" when the question asked for an integer, because it uses substring matching (the same kind of bug as task 1).
- **Human review.** `export` writes a CSV with empty `<criterion>_score` and `<criterion>_rationale` columns. Reviewers fill these in a spreadsheet, and `import_human_reviews` reads the CSV back with the same validation. Rows that are still blank are skipped.
- **Agreement.** `agreement(judge_reviews, human_reviews)` reports n, exact agreement, Cohen's kappa and quadratic-weighted kappa per criterion.

Cohen's kappa corrects raw agreement for the agreement two raters would reach by chance, given how often each one uses each score:

```
kappa = (p_observed - p_chance) / (1 - p_chance)
```

A kappa of 1 means perfect agreement and 0 means no better than chance. Raw agreement can look high when, for example, both raters give almost everything a 5. **Quadratic weighting** counts a disagreement by its squared distance on the 1-5 scale, so a 4 vs 5 near miss costs much less than 1 vs 5. That suits ordinal rubric scores. In practice, use agreement against a human-labelled sample to decide whether an LLM judge can be trusted for a criterion, and keep humans in the loop for criteria where agreement is low.

`data/review_items.jsonl` holds five hand-written sample outputs: two correct, one hallucinated "Ramanujan's lemma", one that ignores the integer-format instruction, and one with no working shown. Use them to try the flow.

## Related benchmarks

- **SWE-bench / SWE-bench Verified / Lite.** Real GitHub issues from Python repos. The model writes a patch, and an instance counts as resolved only if FAIL_TO_PASS and PASS_TO_PASS all pass in a Docker environment. This harness reproduces that grading logic on small local tasks and can load and export real instances. It does not run the official benchmark.
- **WebArena** (Zhou et al., 2023). Agents complete tasks on self-hosted, realistic websites (shopping, a forum, GitLab, a CMS, maps) through a browser. Scoring is task success rate, checked by programs that inspect the final answer or the resulting site state.
- **OSWorld** (Xie et al., 2024). Computer-use agents operate a real desktop OS in a virtual machine, across apps such as office suites, browsers and terminals. Each task has an execution-based checker script that inspects the final machine state, and the score is success rate.

All three grade the **outcome by execution**, not by comparing text. This harness's "apply the patch, run the hidden tests" step is the same idea in its simplest form. The same structure (task spec, isolated environment, run, programmatic checker, success rate, then pass@k over samples) would carry over to a web or OS task, but this repo does not include those environments.

- **APEX** (Mercor's AI Productivity Index, 2025). Professional tasks written by domain experts in investment banking, management consulting, law and primary care medicine. Each response is graded by an LLM judge against an expert-written rubric of pass/fail criteria, and the score is the share of criteria met ([mercor.com/apex](https://www.mercor.com/apex/), [arXiv 2509.25721](https://arxiv.org/abs/2509.25721)). It relates to the rubric review here: both use rubric-based LLM judging of free-text work. APEX uses many binary criteria per task, while this module uses five 1-5 criteria. The judge-vs-human kappa check is the kind of validation an LLM-graded benchmark needs.

## Limitations

- Three local tasks is a smoke-test suite, not a benchmark, and its numbers would not generalise.
- Patches run without a sandbox (see the security note above).
- The real-model provider and the LLM judge have not been run here. Every committed number comes from the mock model and is labelled as such.
