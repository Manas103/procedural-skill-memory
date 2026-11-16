"""The pluggable interface every attempt generator implements. An
`AttemptGenerator` is asked for one attempt at one task instance and returns
the implementation it reached for, that implementation's output on the
instance, and a transcript string whose word count stands in for the tokens
that attempt would have cost (see src/tokens.py).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.tasks.families import Family


@dataclass
class Attempt:
    impl_name: str  # "sound" or "flawed"
    output: object
    transcript: str


class AttemptGenerator(ABC):
    @abstractmethod
    def attempt(self, family: Family, instance: object, rng: random.Random) -> Attempt:
        raise NotImplementedError
