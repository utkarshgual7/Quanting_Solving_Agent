# Patch eval report: mock

> **Mock model run.** The mock returns the gold patch at a fixed rate; these numbers test the pipeline and are not a score for any real model.

- run at: 2026-10-08T19:45:31+00:00
- samples per task (n): 10
- tasks: 3

| task | correct / n | pass@1 | pass@5 |
|---|---|---|---|
| quanting__answer-check-1 | 6 / 10 | 0.600 | 1.000 |
| quanting__guardrails-2 | 4 / 10 | 0.400 | 0.976 |
| quanting__rate-limit-3 | 5 / 10 | 0.500 | 0.996 |
| **mean** | | **0.500** | **0.991** |
