"""Load and validate config/assumptions.yaml, the single home of every generator number."""
from pathlib import Path

import yaml

from movrec import CONFIG_DIR

DEFAULT_PATH = CONFIG_DIR / "assumptions.yaml"
REQUIRED_FIELDS = ("value", "description", "source", "source_type", "confidence", "used_in")
SOURCE_TYPES = {"company_kb", "public", "public_derived", "generic_reasoning"}
NEEDS_URL = {"public", "public_derived"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}


class AssumptionError(ValueError):
    pass


def find_problems(entries: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems = []
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            problems.append(f"{key}: entry must be a mapping")
            continue
        for field in REQUIRED_FIELDS:
            if field not in entry or entry[field] in (None, ""):
                problems.append(f"{key}: missing '{field}'")
        if entry.get("source_type") not in SOURCE_TYPES:
            problems.append(f"{key}: source_type must be one of {sorted(SOURCE_TYPES)}")
        if entry.get("source_type") in NEEDS_URL and not entry.get("url"):
            problems.append(f"{key}: {entry.get('source_type')} source needs a 'url'")
        if entry.get("confidence") not in CONFIDENCE_LEVELS:
            problems.append(f"{key}: confidence must be one of {sorted(CONFIDENCE_LEVELS)}")
    return problems


class Assumptions:
    """Read-only access to assumption values. Records which keys were read."""

    def __init__(self, entries: dict):
        self._entries = entries
        self.used: set[str] = set()

    def __getitem__(self, key: str):
        if key not in self._entries:
            raise KeyError(f"'{key}' is not defined in assumptions.yaml")
        self.used.add(key)
        return self._entries[key]["value"]

    def entry(self, key: str) -> dict:
        return self._entries[key]

    def keys(self) -> list[str]:
        return list(self._entries)


def load_assumptions(path: Path = DEFAULT_PATH) -> Assumptions:
    with open(path) as f:
        entries = yaml.safe_load(f)
    problems = find_problems(entries)
    if problems:
        raise AssumptionError("Invalid assumptions file:\n  " + "\n  ".join(problems))
    return Assumptions(entries)


def render_markdown(assumptions: Assumptions) -> str:
    """A readable table of every assumption (written to docs/ASSUMPTIONS.md)."""
    lines = [
        "# Assumptions",
        "",
        "Generated from `config/assumptions.yaml` (the source of truth). "
        "All data in this repo is **synthetic**.",
        "",
        "| Key | Description | Source type | Confidence | Source |",
        "|---|---|---|---|---|",
    ]
    for key in assumptions.keys():
        e = assumptions.entry(key)
        source = e["source"] + (f" ({e['url']})" if e.get("url") else "")
        lines.append(
            f"| `{key}` | {e['description']} | {e['source_type']} | {e['confidence']} | "
            f"{source.replace('|', '/')} |"
        )
    counts = {}
    for key in assumptions.keys():
        st = assumptions.entry(key)["source_type"]
        counts[st] = counts.get(st, 0) + 1
    lines += ["", "**Count by source type:** " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))]
    return "\n".join(lines) + "\n"
