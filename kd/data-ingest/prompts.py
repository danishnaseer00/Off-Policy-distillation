"""Teacher prompt sampling: MATH train split, decontaminated against MATH-500.

RECONSTRUCTION NOTE. The original kd/data/prompts.py was swallowed by a
.gitignore rule — a bare `data/` pattern matches kd/data/ at any depth — and was
never committed, so it could not be recovered from git. This is a rewrite, not a
recovery, and it is the reason `modal run distill.py::gen` and `::smoke` died
with ModuleNotFoundError.

The student is scored on MATH-500, so any MATH-500 problem reachable from the
teacher's training split is a leak that inflates the result and makes the whole
comparison meaningless. Contamination is therefore checked on alphanumeric-
normalised text rather than exact match: the same problem is routinely reworded
between splits, and exact matching sails straight past those.
"""

from __future__ import annotations

import random
import re

from ..config import LEVELS, MATH_SOURCE, MATH_SUBJECTS, MATH_TRAIN, SEED

_ALNUM = re.compile(r"[^a-z0-9]+")


def _key(text: str) -> str:
    """Order-insensitive fingerprint: lowercase, strip everything non-alphanumeric."""
    return _ALNUM.sub("", (text or "").lower())


def _math500_keys() -> set[str]:
    from datasets import load_dataset

    rows = load_dataset(MATH_SOURCE, split="test")
    return {_key(r["problem"]) for r in rows}


def _answer_of(row: dict) -> str:
    """MATH rows carry the short answer in `answer`; fall back to the boxed
    value at the end of the worked `solution`."""
    if row.get("answer"):
        return row["answer"]
    from ..eval.extract import extract_boxed

    return extract_boxed(row.get("solution", "")) or ""


def _level_of(row: dict) -> int | None:
    raw = row.get("level", "")
    for ch in str(raw):
        if ch.isdigit():
            return int(ch)
    return None


def sample_prompts(n: int, levels: tuple = LEVELS, seed: int = SEED):
    """Return (rows, dropped_contaminated).

    Sampling is level-balanced so the trace set is not dominated by Level 1,
    which is the majority of MATH train. `rows` entries carry exactly the keys
    teacher_generate writes: problem, answer, level, subject.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    banned = _math500_keys()

    # (subject, level) -> list of candidate problems
    pool: dict[tuple[str, int], list[dict]] = {}
    for subject in MATH_SUBJECTS:
        rows = load_dataset(MATH_TRAIN, subject, split="train")
        for row in rows:
            lvl = _level_of(row)
            if lvl not in levels:
                continue
            if _key(row["problem"]) in banned:
                continue
            ans = _answer_of(row)
            if not ans:
                continue
            pool.setdefault((subject, lvl), []).append(
                {"problem": row["problem"], "answer": ans, "level": lvl, "subject": subject}
            )

    for bucket in pool.values():
        rng.shuffle(bucket)

    per_bucket = -(-n // len(levels))  # ceil, so small n still fills every level
    out: list[dict] = []
    for lvl in levels:
        for subject in MATH_SUBJECTS:
            out.extend(pool.get((subject, lvl), [])[:per_bucket])

    rng.shuffle(out)
    out = out[:n]

    # Report what decontamination actually cost, per level. If this number is
    # not printed the leak check is invisible and therefore untested.
    kept_by_level: dict[int, int] = {}
    for row in out:
        kept_by_level[row["level"]] = kept_by_level.get(row["level"], 0) + 1
    print(f"[prompts] requested {n}, returning {len(out)} after decontamination")
    print(f"[prompts] {len(banned)} MATH-500 keys excluded")
    print(f"[prompts] per level: {dict(sorted(kept_by_level.items()))}")

    return out, len(banned)