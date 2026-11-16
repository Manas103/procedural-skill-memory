# Procedural Skill Memory with a Replay-Verified Promotion Gate

A Python harness that gives a repeated-task agent a skill library instead of a transcript: an attempt that succeeds once is only promoted into reusable memory after it passes a deterministic checker on 3 independently sampled replay instances, not on the strength of the one instance where it first worked. A second, ungated path admits on the first success alone, to measure how often that shortcut lets an unsound skill into the library. Every number below was measured on this machine this session, not targeted; three attempts at the corpus's episode length landed on the configuration kept, and none of them reproduced the resume's exact figures, which is said plainly below rather than smoothed over.

## Why this exists

This is a student-scale version of a skill-promotion gate: an agent that repeats a family of tasks should get faster and cheaper over time by reusing a verified procedure instead of re-deriving one from scratch every time, but only if "verified" means something more than "worked once." The ungated condition exists to make that distinction checkable rather than asserted.

## Honest framing

- **There is no live LLM agent here (disclosed).** `src/llm_client/deterministic.py`'s `DeterministicAttemptGenerator` is a seeded weighted coin flip between one genuinely correct ("sound") implementation and one genuinely buggy ("flawed") one per task family, run for real against the actual sampled instance to get a real, checkable output. It is not a second correct implementation with a label; `tests/test_families.py` asserts each flawed implementation both passes and fails across many instances, i.e. it has a real, input-dependent bug, not a universal one.
- **The pluggable LLM client interface has a real implementation that was not run.** `src/llm_client/claude_cli_client.py`'s `ClaudeCliAttemptGenerator` shells out to `claude -p` for a real completion; it exists to show the interface (`src/llm_client/base.py`) accepts a real model, not only the deterministic stand-in, but every number in this README comes from the deterministic generator, the same disclosed-stand-in convention the sibling `agent-task-environment-state-benchmark` repo uses for its `claude_cli_judge.py`.
- **The 6 task families are synthetic and hand-designed, not sourced from a real agent benchmark**: `clamp_sum`, `second_largest_unique`, `balanced_parens`, `palindrome_ignore_case_and_space`, `most_frequent_tie_smallest`, `rotate_left_by_k`. Each has one correct implementation and one implementation with a specific, real bug (e.g. `rotate_left_by_k`'s flawed version rotates right instead of left); `src/tasks/families.py` documents each bug in a `bug_description` field.
- **"Tokens" are a disclosed word-count proxy** (`src/tokens.py`), the same `[A-Za-z0-9_]+` convention used in this portfolio's other repos, applied to a constructed transcript string, not a real model's usage.
- **The reported numbers depend on episode length, and three different lengths were tried and are all disclosed** (see Findings), not just the one kept.

## Architecture

```
src/
  tasks/families.py    6 task families: sample_instance, oracle, sound, flawed, p_sound
  llm_client/
    base.py              the pluggable AttemptGenerator interface (Attempt: impl_name, output, transcript)
    deterministic.py     the stand-in measured at scale: seeded weighted sound/flawed pick
    claude_cli_client.py a real claude -p client, implemented, not invoked this session
  skills/
    gate.py               passes_replay_gate: 3 independent oracle-checked replays required
    library.py             SkillLibrary: admitted skill per family + an admission-event log
  simulation.py          one family, one episode, three parallel conditions (no skill,
                          gated, ungated) over the identical instance sequence
  tokens.py               disclosed word-count token proxy
  measure.py              orchestrates both experiments, writes docs/
tests/
  test_families.py, test_gate.py, test_library.py, test_simulation.py
```

**Why the three conditions share one instance sequence and one set of unskilled attempts.** `simulation.simulate_family_episode` draws the task instances once and the unskilled-attempt choices (`sound` or `flawed`) once per instance index, then replays those same choices for the no-skill, gated, and ungated conditions until a condition actually has a skill to fall back on. This is what makes the three conditions' success rates and token costs comparable at all: any difference after that point comes only from whether a skill is available, not from a second independent roll of the dice.

**Why the gate's replay instances are drawn from a separate random stream.** `passes_replay_gate` samples 3 fresh instances via the family's own `sample_instance`, using a dedicated `rng_replay` stream never used for the main instance sequence (`simulation.py`), so a replay check can never accidentally reuse (and trivially re-pass) the exact instance that produced the lucky first success.

**Why admission is logged, not just the final library state.** `SkillLibrary.admission_events` records every admission with which implementation got in and whether it went through the gate, which is the only way `measure.py`'s Experiment 2 can compute "what fraction of admitted skills were actually unsound" after the fact, across hundreds of independent episodes.

## Validation

**The sound implementation can never fail** (`test_families.py::test_sound_implementation_always_matches_the_oracle`): 200 random instances per family, every one matches the oracle exactly, by construction (sound is a literal alias for oracle in 5 of 6 families and an equivalent implementation in the sixth).

**The flawed implementation has a real bug, not a fake one**: the same test file's `test_flawed_implementation_has_a_real_input_dependent_bug` asserts, over 500 instances per family, that the flawed implementation both matches and diverges from the oracle at least once; a flawed implementation that always failed would not model a plausible wrong-but-sometimes-lucky attempt, and one that never failed would not be a bug.

**The gate actually filters on real replay failures** (`test_gate.py`): a sound implementation always passes; a deliberately wrong stand-in never does; a dedicated test confirms the gate draws exactly `N_REPLAYS` (3) fresh instances per check, not fewer.

**A lucky flawed pass gets caught by the gate in a hand-built scenario** (`test_simulation.py::test_gate_rejects_a_flawed_skill_that_only_got_lucky_once`): a fake family's flawed implementation fails on "hard" instances (90% of the distribution); the test searches seeds for one where the flawed implementation's first attempt landed on the rare "easy" instance (a lucky pass) and confirms the ungated condition admits it while the gated condition's replay draws (mostly "hard") reject it.

**Ungated never admits less often than gated, across 300 seeds of the same fake family** (`test_simulation.py::test_ungated_admits_strictly_at_least_as_often_as_gated`), which is the structural property the whole promotion-gate design rests on: skipping the replay check can only let more skills in, never fewer.

Test output (`docs/test_output.txt`): 9 passed.

## Findings

**The gated success rate and the token reduction both depend heavily on how long an episode runs before the skill library resets, and three different episode lengths were tried, in order.** `measure.py`'s `EPISODES_PER_FAMILY` / `M_PER_EPISODE`: 1 episode of 40 instances gave a 98.33% gated success rate and a 67.88% token reduction (the skill, once admitted at instance 1-4 of 40, dominates the remaining ~36-39 instances); 7 episodes of 6 gave 82.54% / 44.87%; 22 episodes of 2, the configuration kept, gave 64.39% / 15.63%. None of the three hit the resume's claimed 71% / 38% exactly; the honest lesson is that "first-attempt success with a skill library" is not one fixed number, it is a function of how much of the measured window happens before versus after the skill is admitted, exactly the kind of parameter a real deployment's episode length would also control.

**The replay gate is not perfect, and the measured gap between gated and ungated unsoundness is the actual point.** Across 360 admission events per condition (60 short episodes x 6 families), the ungated path admitted an unsound (flawed) skill 32.5% of the time; the gated path admitted one 2.5% of the time. The 2.5% is not a bug: a flawed implementation whose bug only triggers on a minority of instances (e.g. `most_frequent_tie_smallest`'s flawed implementation still passes about 54% of instances, see `src/tasks/families.py`) has roughly a 1-in-6 chance of passing all 3 independent replay instances by chance too (0.54^3 ≈ 0.157 for that family alone), so 3 replays shrinks but does not eliminate the risk. This is disclosed as a measured, expected property of a 3-replay gate against a weak-enough bug, not evidence of an implementation error.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, pure standard library (no external services; `requirements.txt` is `pytest` only). Configuration kept: 6 families x 22 episodes x 2 instances = 264 repeated instances (`docs/measure_report.json`, `docs/benchmark_output.txt`).

| claim | measured | met? |
|---|---|---|
| 3 independent replays pass a deterministic checker before admission | `N_REPLAYS = 3` in `src/skills/gate.py`, exercised in `test_gate.py` | met |
| first-attempt success from 54% to 71% | 56.06% (no skill) to 64.39% (gated skill) | partially met: the 54% starting point matched almost exactly; the ending point undershot, see Findings |
| 38% fewer tokens | 15.63% fewer (44.99 to 37.96 avg tokens/instance) | not met at this episode length; 44.87% and 67.88% were measured at the other two lengths tried, see Findings |
| over 240 repeated instances | 264 | met |
| ungated promotion admitted 23% unsound skills | 32.5% (117/360 admissions) | not met exactly; directionally confirmed and contrasted with the gated path's 2.5% (9/360), see Findings |

Per-family detail for all three episode-length attempts, and the full admission-soundness breakdown, are in `docs/measure_report.json`.

## Building and running

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # (.venv/bin/pip on Linux/WSL)

.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m src.measure docs        # writes docs/measure_report.json, benchmark_output.txt
```

No database or external service is required; everything here is pure Python standard library plus pytest.

## Limitations

- No live LLM is called; `DeterministicAttemptGenerator` is a disclosed, seeded stand-in. `ClaudeCliAttemptGenerator` is implemented against the same interface but was not run this session.
- The 6 task families and their single flawed implementation each are hand-designed to have a real, checkable, input-dependent bug; they are not drawn from an existing agent-coding benchmark.
- Token counts are a word-count proxy (`src/tokens.py`), not a real model's tokenizer; both conditions are measured with the same proxy so the percentage comparison is apples-to-apples even though neither absolute number is real.
- The gated-vs-ungated first-attempt-success and token-reduction numbers are sensitive to episode length (`EPISODES_PER_FAMILY`, `M_PER_EPISODE`); three lengths were tried and all three are disclosed in Findings rather than only reporting the one kept.
- The replay gate measurably still admits a small percentage of unsound skills (2.5% in the configuration measured) when a family's bug has a high enough per-instance pass rate; this is an expected property of a fixed 3-replay check, not a defect limited to this implementation.
