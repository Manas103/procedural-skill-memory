"""Six task families. Each has a ground-truth oracle, a sound implementation
(identical to the oracle by construction, so it can never fail), and exactly
one flawed implementation with a real, input-dependent bug, not a coin flip:
the flawed implementation is ordinary (wrong) Python that fails whenever the
sampled instance happens to hit its edge case, and the instance sampler's own
parameters, not a fudge factor, control how often that edge case is hit.

A Family bundles: sample_instance(rng), oracle(instance), sound(instance),
flawed(instance), and p_sound, the probability an unskilled attempt reaches
for the sound implementation rather than the flawed one (modeling plausible-
but-sometimes-wrong generation, see src/llm_client/README note below).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Family:
    name: str
    sample_instance: Callable[[random.Random], object]
    oracle: Callable[[object], object]
    sound: Callable[[object], object]
    flawed: Callable[[object], object]
    p_sound: float
    bug_description: str


# --- A. clamp_sum --------------------------------------------------------

def _clamp_sum_sample(rng: random.Random):
    return {"values": [rng.randint(0, 12) for _ in range(5)], "cap": 8}


def _clamp_sum_oracle(inst):
    return sum(min(x, inst["cap"]) for x in inst["values"])


def _clamp_sum_flawed(inst):
    return sum(inst["values"])  # forgets the cap entirely


clamp_sum = Family(
    "clamp_sum",
    _clamp_sum_sample,
    _clamp_sum_oracle,
    _clamp_sum_oracle,
    _clamp_sum_flawed,
    p_sound=0.35,
    bug_description="forgets to clamp each element to the cap before summing; "
    "only visible when at least one sampled value exceeds the cap",
)


# --- B. second_largest_unique ---------------------------------------------

def _second_largest_sample(rng: random.Random):
    values = [rng.randint(1, 4) for _ in range(6)]
    if len(set(values)) < 2:
        values[0], values[1] = 1, 2
    return {"values": values}


def _second_largest_oracle(inst):
    uniq = sorted(set(inst["values"]), reverse=True)
    return uniq[1]


def _second_largest_flawed(inst):
    return sorted(inst["values"], reverse=True)[1]  # not dedup'd


second_largest_unique = Family(
    "second_largest_unique",
    _second_largest_sample,
    _second_largest_oracle,
    _second_largest_oracle,
    _second_largest_flawed,
    p_sound=0.35,
    bug_description="picks the second element by position, not the second "
    "distinct value; wrong whenever the maximum value repeats",
)


# --- C. balanced_parens -----------------------------------------------------

def _balanced_parens_sample(rng: random.Random):
    n_pairs = rng.randint(2, 5)
    if rng.random() < 0.15:
        # a genuinely balanced sequence, built by a correct-by-construction
        # random walk that never lets the running count go negative
        chars, depth = [], 0
        remaining_open, remaining_close = n_pairs, n_pairs
        while remaining_open or remaining_close:
            can_open = remaining_open > 0
            can_close = remaining_close > 0 and depth > 0
            if can_open and (not can_close or rng.random() < 0.5):
                chars.append("(")
                remaining_open -= 1
                depth += 1
            else:
                chars.append(")")
                remaining_close -= 1
                depth -= 1
        return {"text": "".join(chars)}
    # equal counts of each bracket, but in uniformly random order, which is
    # usually not validly nested
    chars = ["("] * n_pairs + [")"] * n_pairs
    rng.shuffle(chars)
    return {"text": "".join(chars)}


def _balanced_parens_oracle(inst):
    depth = 0
    for ch in inst["text"]:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def _balanced_parens_flawed(inst):
    return inst["text"].count("(") == inst["text"].count(")")


balanced_parens = Family(
    "balanced_parens",
    _balanced_parens_sample,
    _balanced_parens_oracle,
    _balanced_parens_oracle,
    _balanced_parens_flawed,
    p_sound=0.35,
    bug_description="checks that the two bracket counts match, not that they "
    "nest in valid order; wrong on any equal-count-but-misordered sequence",
)


# --- D. palindrome_ignore_case_and_space ------------------------------------

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _palindrome_sample(rng: random.Random):
    half_len = rng.randint(2, 4)
    half = "".join(rng.choice(_ALPHABET) for _ in range(half_len))
    middle = rng.choice(_ALPHABET) if rng.random() < 0.5 else ""
    core = half + middle + half[::-1]  # a true palindrome, ignoring case
    core = "".join(c.upper() if rng.random() < 0.5 else c for c in core)
    if rng.random() < 0.9:
        # insert a space at an asymmetric position so the raw string itself
        # is not a mirror palindrome, even though it is one once spaces and
        # case are removed
        pos = 1
        text = core[:pos] + " " + core[pos:]
    else:
        text = core
    return {"text": text}


def _palindrome_oracle(inst):
    normalized = inst["text"].lower().replace(" ", "")
    return normalized == normalized[::-1]


def _palindrome_flawed(inst):
    lowered = inst["text"].lower()
    return lowered == lowered[::-1]  # never strips spaces


palindrome_ignore_case_and_space = Family(
    "palindrome_ignore_case_and_space",
    _palindrome_sample,
    _palindrome_oracle,
    _palindrome_oracle,
    _palindrome_flawed,
    p_sound=0.35,
    bug_description="lowercases but never removes spaces, so a phrase that "
    "is a palindrome once spaces are ignored can still fail the raw check",
)


# --- E. most_frequent_tie_smallest ------------------------------------------

def _most_frequent_sample(rng: random.Random):
    return {"values": [rng.randint(1, 7) for _ in range(7)]}


def _mode_counts(values):
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return [v for v, c in counts.items() if c == best]


def _most_frequent_oracle(inst):
    return min(_mode_counts(inst["values"]))


def _most_frequent_flawed(inst):
    return max(_mode_counts(inst["values"]))  # breaks the tie toward the largest, not the smallest


most_frequent_tie_smallest = Family(
    "most_frequent_tie_smallest",
    _most_frequent_sample,
    _most_frequent_oracle,
    _most_frequent_oracle,
    _most_frequent_flawed,
    p_sound=0.35,
    bug_description="breaks a frequency tie toward the largest tied value "
    "instead of the smallest; wrong whenever a genuine tie exists",
)


# --- F. rotate_left_by_k -----------------------------------------------------

def _rotate_sample(rng: random.Random):
    return {"values": [rng.randint(1, 9) for _ in range(6)], "k": rng.randint(1, 5)}


def _rotate_oracle(inst):
    k = inst["k"]
    return inst["values"][k:] + inst["values"][:k]


def _rotate_flawed(inst):
    k = inst["k"]
    return inst["values"][-k:] + inst["values"][:-k]  # rotates right, not left


rotate_left_by_k = Family(
    "rotate_left_by_k",
    _rotate_sample,
    _rotate_oracle,
    _rotate_oracle,
    _rotate_flawed,
    p_sound=0.35,
    bug_description="rotates right by k instead of left by k; on a 6-element "
    "list the two coincide only at k=3, so k in {1,2,4,5} exposes the bug",
)


ALL_FAMILIES = [
    clamp_sum,
    second_largest_unique,
    balanced_parens,
    palindrome_ignore_case_and_space,
    most_frequent_tie_smallest,
    rotate_left_by_k,
]
