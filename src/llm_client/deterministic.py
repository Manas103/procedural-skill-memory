"""The attempt generator measured at scale in this repository. It is a
disclosed deterministic stand-in, not a live model: it picks `sound` with
probability `family.p_sound` and `flawed` otherwise (a fixed, seeded weighted
coin flip, modeling "most unassisted attempts at a task this hard are
plausible but subtly wrong, a few happen to be right"), then runs whichever
implementation it picked on the real instance to get a real output. Nothing
about an instance's content influences the pick; only the implementation
actually run determines whether the attempt is correct, which is what makes
"first-attempt success" a real, checkable outcome rather than an assumed one.
"""

from __future__ import annotations

import random

from src.llm_client.base import Attempt, AttemptGenerator
from src.tasks.families import Family

EXPLORATION_TEMPLATE = (
    "Task: {family}. Input: {instance!r}. Attempt: trying the {impl_name} "
    "approach for this family because the last few attempts on tasks like "
    "this one suggested it was worth another try. Reasoning through the "
    "edge cases of {family} before committing to an answer."
)


class DeterministicAttemptGenerator(AttemptGenerator):
    def attempt(self, family: Family, instance: object, rng: random.Random) -> Attempt:
        impl_name = "sound" if rng.random() < family.p_sound else "flawed"
        impl = family.sound if impl_name == "sound" else family.flawed
        output = impl(instance)
        transcript = EXPLORATION_TEMPLATE.format(family=family.name, instance=instance, impl_name=impl_name)
        return Attempt(impl_name=impl_name, output=output, transcript=transcript)
