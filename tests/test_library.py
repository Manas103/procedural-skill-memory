from src.skills.library import SkillLibrary


def test_admit_records_event_and_makes_skill_available():
    lib = SkillLibrary()
    assert lib.has("clamp_sum") is False
    assert lib.get("clamp_sum") is None

    sentinel = object()
    lib.admit("clamp_sum", "sound", sentinel, gated=True)

    assert lib.has("clamp_sum") is True
    assert lib.get("clamp_sum") is sentinel
    assert lib.admission_events == [{"family": "clamp_sum", "impl_name": "sound", "gated": True}]


def test_admitting_a_second_family_does_not_disturb_the_first():
    lib = SkillLibrary()
    lib.admit("clamp_sum", "sound", lambda x: x, gated=True)
    lib.admit("rotate_left_by_k", "flawed", lambda x: x, gated=False)
    assert lib.has("clamp_sum")
    assert lib.has("rotate_left_by_k")
    assert len(lib.admission_events) == 2
