"""Token-count proxy, same convention as this portfolio's other repos
(`agent-task-environment-state-benchmark`'s `telemetry.estimate_tokens`,
`per-user-memory-service`'s `src/tokens.py`): word-like substrings in a
transcript string, not a real tokenizer. Used identically for the
skill-library condition and the no-skill baseline, so the percentage
reduction is apples-to-apples even though neither absolute number is a real
model's token count.
"""

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))
