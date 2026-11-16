"""A skill library is a dict from family name to the one admitted
implementation for that family, plus a log of every admission event (which
implementation got in, and whether it went through the replay gate). The log
is what lets measure.py compute "what fraction of admitted skills turned out
unsound" after the fact.
"""

from __future__ import annotations


class SkillLibrary:
    def __init__(self):
        self._admitted: dict[str, tuple[str, object]] = {}
        self.admission_events: list[dict] = []

    def get(self, family_name: str):
        entry = self._admitted.get(family_name)
        return entry[1] if entry else None

    def has(self, family_name: str) -> bool:
        return family_name in self._admitted

    def admit(self, family_name: str, impl_name: str, impl_fn, gated: bool) -> None:
        self._admitted[family_name] = (impl_name, impl_fn)
        self.admission_events.append({"family": family_name, "impl_name": impl_name, "gated": gated})
