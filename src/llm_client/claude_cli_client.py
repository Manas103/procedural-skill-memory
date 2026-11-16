"""A real `claude -p` attempt generator, implemented for a live sample but
not run this session (the same disclosed convention the sibling
`agent-task-environment-state-benchmark` repo uses for its
`judge/claude_cli_judge.py`: designed and reviewable, but every number in
this repository's README comes from `DeterministicAttemptGenerator` instead).
Running this against all 6 families at the scale `measure.py` uses would
cost one `claude -p` call per instance; that was out of scope for this
session's budget.
"""

from __future__ import annotations

import random
import re
import subprocess

from src.llm_client.base import Attempt, AttemptGenerator
from src.tasks.families import Family

PROMPT_TEMPLATE = """Write a single Python expression, no explanation, that computes the
answer to this task for the given input. Task family: {family}. Bug to avoid if you know it: {bug_hint}.
Input (a Python literal, already bound to the name `inst`): {instance!r}
Reply with only the Python expression, nothing else."""


class ClaudeCliAttemptGenerator(AttemptGenerator):
    """Shells out to `claude -p` for a real completion. Not invoked by
    measure.py or any test in this repository this session; kept here to
    show the pluggable interface accepts a real model, not only the
    deterministic stand-in."""

    def __init__(self, model: str | None = None, timeout_s: int = 60):
        self.model = model
        self.timeout_s = timeout_s

    def attempt(self, family: Family, instance: object, rng: random.Random) -> Attempt:
        prompt = PROMPT_TEMPLATE.format(family=family.name, bug_hint="unknown", instance=instance)
        cmd = ["claude", "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, check=True)
        expression = result.stdout.strip()
        expression = re.sub(r"^```(?:python)?|```$", "", expression).strip()
        output = eval(expression, {"inst": instance})  # noqa: S307 - sandboxed by caller, not run this session
        return Attempt(impl_name="claude_cli", output=output, transcript=prompt + "\n" + expression)
