"""The promotion gate: a candidate skill is admitted only after it passes a
deterministic checker on 3 independently sampled replay instances of the
same family, not the one instance where it first happened to work. This is
the whole point of gating: a flawed implementation can succeed on one
instance by luck (whenever the sampled instance misses its bug's trigger
condition); replaying it on 3 fresh, independently drawn instances makes that
luck much less likely to repeat three times in a row.
"""

from __future__ import annotations

import random

from src.tasks.families import Family

N_REPLAYS = 3


def passes_replay_gate(family: Family, impl, rng: random.Random, n: int = N_REPLAYS) -> bool:
    for _ in range(n):
        replay_instance = family.sample_instance(rng)
        if impl(replay_instance) != family.oracle(replay_instance):
            return False
    return True
