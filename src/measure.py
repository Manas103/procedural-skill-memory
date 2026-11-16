"""Orchestrates two experiments sharing the same 6 task families:

1. First-attempt success and token cost, over 240 repeated instances (6
   families x 40 instances each), comparing no skill library against a
   gated one.
2. Admission soundness: many independent, short episodes per family, purely
   to get a large enough sample of "a skill got admitted" events to measure
   what fraction of ungated admissions (and, for comparison, gated ones)
   turned out to be the flawed implementation rather than the sound one.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm_client.deterministic import DeterministicAttemptGenerator
from src.simulation import simulate_family_episode
from src.tasks.families import ALL_FAMILIES

EPISODES_PER_FAMILY = 22
M_PER_EPISODE = 2  # 6 families x 22 episodes x 2 instances = 264
EXPERIMENT_1_SEED_BASE = 10_000

ADMISSION_EPISODES_PER_FAMILY = 60
ADMISSION_M_PER_EPISODE = 20
EXPERIMENT_2_SEED_BASE = 90_000


def run_experiment_1(generator) -> dict:
    """Each family is run through several independent, short episodes (a
    fresh skill library each time, modeling separate repeated-task sessions)
    rather than one long episode, so a family that happens to build a skill
    on its very first instance does not single-handedly dominate the 240-
    instance average with one long skill-assisted tail."""
    no_skill_successes = []
    gated_successes = []
    no_skill_tokens = []
    gated_tokens = []
    per_family = {}

    for idx, family in enumerate(ALL_FAMILIES):
        fam_ns, fam_g = [], []
        admitted_at = []
        for episode in range(EPISODES_PER_FAMILY):
            seed = EXPERIMENT_1_SEED_BASE + idx * 1009 + episode * 7
            result = simulate_family_episode(family, generator, M_PER_EPISODE, seed)
            fam_ns += result["no_skill"]
            fam_g += result["gated"]
            first_skill_use = next((i for i, r in enumerate(result["gated"]) if "impl_name" not in r), None)
            admitted_at.append(first_skill_use)
        no_skill_successes += [r["success"] for r in fam_ns]
        gated_successes += [r["success"] for r in fam_g]
        no_skill_tokens += [r["tokens"] for r in fam_ns]
        gated_tokens += [r["tokens"] for r in fam_g]
        per_family[family.name] = {
            "no_skill_success_pct": round(100 * sum(r["success"] for r in fam_ns) / len(fam_ns), 2),
            "gated_success_pct": round(100 * sum(r["success"] for r in fam_g) / len(fam_g), 2),
            "episodes": EPISODES_PER_FAMILY,
            "instances_per_episode": M_PER_EPISODE,
            "skill_first_used_at_instance_by_episode": admitted_at,
        }

    n = len(no_skill_successes)
    no_skill_rate = sum(no_skill_successes) / n
    gated_rate = sum(gated_successes) / n
    avg_no_skill_tokens = sum(no_skill_tokens) / n
    avg_gated_tokens = sum(gated_tokens) / n

    return {
        "total_instances": n,
        "no_skill_first_attempt_success_pct": round(100 * no_skill_rate, 2),
        "gated_first_attempt_success_pct": round(100 * gated_rate, 2),
        "avg_tokens_no_skill": round(avg_no_skill_tokens, 2),
        "avg_tokens_gated": round(avg_gated_tokens, 2),
        "token_reduction_pct": round(100 * (1 - avg_gated_tokens / avg_no_skill_tokens), 2),
        "per_family": per_family,
    }


def run_experiment_2(generator) -> dict:
    gated_events = []
    ungated_events = []
    for idx, family in enumerate(ALL_FAMILIES):
        for episode in range(ADMISSION_EPISODES_PER_FAMILY):
            seed = EXPERIMENT_2_SEED_BASE + idx * 10_007 + episode * 13
            result = simulate_family_episode(family, generator, ADMISSION_M_PER_EPISODE, seed)
            gated_events += result["gated_admissions"]
            ungated_events += result["ungated_admissions"]

    def unsound_pct(events):
        if not events:
            return None
        unsound = sum(1 for e in events if e["impl_name"] == "flawed")
        return {"admissions": len(events), "unsound": unsound, "unsound_pct": round(100 * unsound / len(events), 2)}

    return {"gated": unsound_pct(gated_events), "ungated": unsound_pct(ungated_events)}


def main(docs_dir: Path) -> dict:
    docs_dir.mkdir(parents=True, exist_ok=True)
    generator = DeterministicAttemptGenerator()

    exp1 = run_experiment_1(generator)
    exp2 = run_experiment_2(generator)

    report = {"experiment_1_first_attempt_and_tokens": exp1, "experiment_2_admission_soundness": exp2}
    (docs_dir / "measure_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"repeated instances: {exp1['total_instances']} (6 families x {EPISODES_PER_FAMILY} episodes x "
        f"{M_PER_EPISODE} instances)",
        f"no-skill first-attempt success: {exp1['no_skill_first_attempt_success_pct']}%",
        f"gated-skill first-attempt success: {exp1['gated_first_attempt_success_pct']}%",
        f"avg tokens/instance, no skill: {exp1['avg_tokens_no_skill']}",
        f"avg tokens/instance, gated skill: {exp1['avg_tokens_gated']}",
        f"token reduction: {exp1['token_reduction_pct']}%",
        "",
        f"admission soundness ({ADMISSION_EPISODES_PER_FAMILY} episodes/family x 6 families, "
        f"{ADMISSION_M_PER_EPISODE} instances/episode):",
        f"  gated admissions:   {exp2['gated']}",
        f"  ungated admissions: {exp2['ungated']}",
    ]
    (docs_dir / "benchmark_output.txt").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    import sys

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    result = main(out_dir)
    print(json.dumps(result, indent=2))
