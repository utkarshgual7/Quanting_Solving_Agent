"""Maths-only input check (adapted from app/core/guardrails.py)."""
import re

EDUCATIONAL_KEYWORDS = [
    "math", "mathematics", "equation", "solve", "calculate",
    "algebra", "geometry", "calculus", "statistics", "probability",
]

MATH_PATTERNS = [
    r"\d+[\+\-\*/]\d+",  # basic arithmetic
    r"[a-z]\s*=\s*\d+",  # variable assignments
    r"\\[a-zA-Z]+\{.*?\}",  # LaTeX commands
    r"\b(solve|find|calculate|compute|determine)\b",  # action words
]


def is_math_query(prompt: str) -> bool:
    text = prompt.lower()
    if any(keyword in text for keyword in EDUCATIONAL_KEYWORDS):
        return True
    return any(re.search(pattern, text) for pattern in MATH_PATTERNS)
