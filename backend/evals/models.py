"""Pluggable "model" interface for the patch eval.

A model is anything with a `name` and `generate(task, sample_idx) -> str` that
returns a unified diff. Two implementations:

- MockModel: deterministic and offline, for tests and CI. It does not solve
  anything; it returns the gold patch with a configured probability and a
  failed attempt otherwise. Its scores exercise the pipeline and say nothing
  about any real model.
- AnthropicModel: a real LLM via the Anthropic API. Optional: needs
  `pip install anthropic` and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
import random
import re

from evals.harness import Task


class MockModel:
    def __init__(self, skill: dict[str, float] | None = None, default_skill: float = 0.5, seed: int = 0):
        self.name = "mock"
        self.skill = skill or {}
        self.default_skill = default_skill
        self.seed = seed

    def generate(self, task: Task, sample_idx: int) -> str:
        rng = random.Random(f"{self.seed}:{task.instance_id}:{sample_idx}")
        if rng.random() < self.skill.get(task.instance_id, self.default_skill):
            return task.patch
        # Two failure modes a real model shows: no usable patch, or a malformed one.
        return "" if rng.random() < 0.5 else "--- not a valid diff\n"


def build_prompt(task: Task) -> str:
    files = sorted(p for p in task.repo_dir.rglob("*") if p.is_file())
    listing = "\n\n".join(f"### {p.relative_to(task.repo_dir)}\n```python\n{p.read_text()}```" for p in files)
    return (
        "You are fixing an issue in a small Python repository.\n\n"
        f"## Issue\n{task.problem_statement}\n\n## Repository\n{listing}\n\n"
        "Reply with a single unified diff (git format, paths prefixed a/ and b/) "
        "that fixes the issue, inside one ```diff code block. Do not modify tests."
    )


def extract_diff(text: str) -> str:
    """Pull the patch out of a model reply (```diff block, else from `diff --git`)."""
    fenced = re.search(r"```(?:diff|patch)?\n(.*?)```", text, flags=re.S)
    if fenced and "+++" in fenced.group(1):
        diff = fenced.group(1)
    elif "diff --git" in text:
        diff = text[text.index("diff --git"):]
    else:
        return ""
    return diff if diff.endswith("\n") else diff + "\n"


class AnthropicModel:
    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency, only needed for real runs

        self.client = anthropic.Anthropic()
        self.name = model or os.environ.get("EVAL_MODEL", "claude-opus-5-5")

    def complete(self, prompt: str) -> str:
        response = self.client.beta.messages.create(
            model=self.name,
            max_tokens=16000,
            # On a safety decline the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")

    def generate(self, task: Task, sample_idx: int) -> str:
        return extract_diff(self.complete(build_prompt(task)))


def get_model(name: str):
    if name == "mock":
        return MockModel()
    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set; use --model mock or export a key")
        return AnthropicModel()
    raise SystemExit(f"unknown model {name!r} (choose mock or anthropic)")
