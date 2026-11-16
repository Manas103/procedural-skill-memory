"""Runs one family through three parallel conditions over the identical
sequence of task instances: no skill library at all, a gated library (3
independent replays required before admission), and an ungated library
(admitted on the first success, no replay). All three see the same
instances and the same unskilled-attempt choices until a condition actually
has a skill to fall back on, which is what makes a head-to-head comparison
of their first-attempt success and token cost meaningful.
"""

from __future__ import annotations

import random

from src.llm_client.base import AttemptGenerator
from src.skills.gate import passes_replay_gate
from src.skills.library import SkillLibrary
from src.tasks.families import Family
from src.tokens import estimate_tokens

SKILL_APPLICATION_TEMPLATE = "Applying the {family} skill directly to input {instance!r}."


def simulate_family_episode(family: Family, generator: AttemptGenerator, m: int, seed: int) -> dict:
    rng_instances = random.Random(seed)
    rng_impl = random.Random(seed + 1)
    rng_replay = random.Random(seed + 2)

    instances = [family.sample_instance(rng_instances) for _ in range(m)]
    attempts = [generator.attempt(family, inst, rng_impl) for inst in instances]
    truths = [family.oracle(inst) for inst in instances]

    def explore_record(i: int) -> dict:
        a = attempts[i]
        return {"success": a.output == truths[i], "tokens": estimate_tokens(a.transcript), "impl_name": a.impl_name}

    def skill_record(impl_fn, inst) -> dict:
        output = impl_fn(inst)
        transcript = SKILL_APPLICATION_TEMPLATE.format(family=family.name, instance=inst)
        return {"success": output == family.oracle(inst), "tokens": estimate_tokens(transcript)}

    no_skill_records = [explore_record(i) for i in range(m)]

    def run_gated_condition(gated: bool) -> tuple[list[dict], SkillLibrary]:
        lib = SkillLibrary()
        records = []
        for i in range(m):
            if lib.has(family.name):
                records.append(skill_record(lib.get(family.name), instances[i]))
                continue
            rec = explore_record(i)
            records.append(rec)
            if rec["success"]:
                impl_name = attempts[i].impl_name
                impl_fn = family.sound if impl_name == "sound" else family.flawed
                if gated:
                    if passes_replay_gate(family, impl_fn, rng_replay):
                        lib.admit(family.name, impl_name, impl_fn, gated=True)
                else:
                    lib.admit(family.name, impl_name, impl_fn, gated=False)
        return records, lib

    gated_records, gated_lib = run_gated_condition(gated=True)
    ungated_records, ungated_lib = run_gated_condition(gated=False)

    return {
        "no_skill": no_skill_records,
        "gated": gated_records,
        "gated_admissions": gated_lib.admission_events,
        "ungated": ungated_records,
        "ungated_admissions": ungated_lib.admission_events,
    }
