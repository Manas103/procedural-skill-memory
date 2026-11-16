import random

from src.tasks.families import ALL_FAMILIES


def test_sound_implementation_always_matches_the_oracle():
    rng = random.Random(1)
    for family in ALL_FAMILIES:
        for _ in range(200):
            instance = family.sample_instance(rng)
            assert family.sound(instance) == family.oracle(instance), family.name


def test_flawed_implementation_has_a_real_input_dependent_bug():
    """The flawed implementation must sometimes match the oracle (it is not
    universally wrong, which would make it an unrealistic strawman) and
    sometimes not (it must have a real bug, not be a second correct
    implementation in disguise)."""
    rng = random.Random(2)
    for family in ALL_FAMILIES:
        matches = 0
        mismatches = 0
        for _ in range(500):
            instance = family.sample_instance(rng)
            truth = family.oracle(instance)
            if family.flawed(instance) == truth:
                matches += 1
            else:
                mismatches += 1
        assert matches > 0, f"{family.name}: flawed impl never matches the oracle"
        assert mismatches > 0, f"{family.name}: flawed impl never diverges from the oracle"
