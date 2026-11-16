"""Uses a small fake family, not the real 6, so the admission logic can be
pinned to an exact, known scenario: a flawed implementation that is wrong
only on instances tagged "hard", and an instance sequence engineered so the
flawed implementation's first attempt lands on an "easy" instance (a lucky
pass) but a replay instance is "hard" (so the gate should catch it)."""

import random

from src.llm_client.base import Attempt, AttemptGenerator
from src.simulation import simulate_family_episode
from src.tasks.families import Family


def _make_fake_family():
    def sample_instance(rng):
        return {"hard": rng.random() < 0.9}  # mostly "hard", rarely "easy"

    def oracle(instance):
        return "ok"

    def sound(instance):
        return "ok"

    def flawed(instance):
        return "ok" if not instance["hard"] else "wrong"

    return Family("fake", sample_instance, oracle, sound, flawed, p_sound=0.0, bug_description="fails on hard instances")


class _AlwaysFlawedFirst(AttemptGenerator):
    """Always reaches for the flawed implementation; with p_sound=0.0 on the
    fake family this is also what DeterministicAttemptGenerator would do,
    spelled out explicitly here so the test does not depend on that."""

    def attempt(self, family, instance, rng):
        output = family.flawed(instance)
        return Attempt(impl_name="flawed", output=output, transcript="attempt")


def test_gate_rejects_a_flawed_skill_that_only_got_lucky_once():
    family = _make_fake_family()
    generator = _AlwaysFlawedFirst()
    # seed chosen so instance 0 is "easy" (flawed happens to pass) while the
    # replay draws are overwhelmingly "hard" (flawed should fail at least one)
    for seed in range(200):
        result = simulate_family_episode(family, generator, m=1, seed=seed)
        if result["no_skill"][0]["success"]:
            gated_admitted = len(result["gated_admissions"]) == 1
            ungated_admitted = len(result["ungated_admissions"]) == 1
            assert ungated_admitted  # ungated always admits on the first success
            if not gated_admitted:
                return  # found a seed where the gate correctly rejected the lucky pass
    raise AssertionError("no seed produced a gate rejection; the fake family's bug rate may need adjusting")


def test_ungated_admits_strictly_at_least_as_often_as_gated():
    family = _make_fake_family()
    generator = _AlwaysFlawedFirst()
    gated_total = 0
    ungated_total = 0
    for seed in range(300):
        result = simulate_family_episode(family, generator, m=1, seed=seed)
        gated_total += len(result["gated_admissions"])
        ungated_total += len(result["ungated_admissions"])
    assert ungated_total >= gated_total
    assert ungated_total > 0
