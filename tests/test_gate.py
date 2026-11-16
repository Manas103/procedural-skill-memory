import random

from src.skills.gate import N_REPLAYS, passes_replay_gate
from src.tasks.families import clamp_sum


def test_sound_implementation_always_passes_the_gate():
    rng = random.Random(3)
    for _ in range(20):
        assert passes_replay_gate(clamp_sum, clamp_sum.sound, rng) is True


def test_an_always_wrong_implementation_never_passes_the_gate():
    rng = random.Random(4)

    def always_wrong(instance):
        return object()

    assert passes_replay_gate(clamp_sum, always_wrong, rng) is False


def test_gate_draws_exactly_n_replay_instances_on_success(monkeypatch):
    rng = random.Random(5)
    calls = {"n": 0}
    real_sample = clamp_sum.sample_instance

    def counting_sample(r):
        calls["n"] += 1
        return real_sample(r)

    monkeypatch.setattr(clamp_sum, "sample_instance", counting_sample)
    passes_replay_gate(clamp_sum, clamp_sum.sound, rng)
    assert calls["n"] == N_REPLAYS
