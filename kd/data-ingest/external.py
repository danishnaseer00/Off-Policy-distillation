from __future__ import annotations

import json

from ..config import MATH_SOURCE, RASCHKA_SUBSET, RASCHKA_URL  # noqa: F401
from ..eval.extract import last_boxed_or_fallback, match_tier
from .prompts import _key


def _download() -> list[dict]:
    """Fetch the raw partition (126MB).

    Uses huggingface_hub rather than urllib: bare urlopen does not follow the
    CDN redirect the Hub issues for large files, and it fails outright inside
    the container. hf_hub_download also lands in HF_HOME, which is already the
    hf-cache Volume, so the 126MB is paid for once and never again on a retry.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from ..config import RASCHKA_FILE, RASCHKA_REPO

    path = hf_hub_download(
        repo_id=RASCHKA_REPO,
        filename=RASCHKA_FILE,
        repo_type="dataset",
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _benchmark_keys() -> set[str]:
    from datasets import load_dataset

    return {_key(r["problem"]) for r in load_dataset(MATH_SOURCE, split="test")}


def load_external_traces() -> tuple[list[dict], dict]:
    """Return (traces, stats) in the schema build_examples already expects."""
    raw = _download()
    bench = _benchmark_keys()
    seen: set[str] = set()

    traces, dropped_leak, dropped_dup, dropped_unclosed, dropped_wrong = [], 0, 0, 0, 0
    for r in raw:
        k = _key(r["problem"])
        if k in bench:
            dropped_leak += 1
            continue
        if k in seen:
            dropped_dup += 1
            continue
        seen.add(k)
        
        body = f"<think>{r['message_thinking']}</think>\n\n{r['message_content']}"
        pred = last_boxed_or_fallback(body)
        correct = bool(pred) and match_tier(pred, r["gtruth_answer"]) is not None
        closed = "</think>" in body

        if not closed:
            dropped_unclosed += 1
            continue
        if not correct:
            dropped_wrong += 1
            continue

        traces.append(
            {
                "problem": r["problem"],
                "answer": r["gtruth_answer"],
                "generation": body,
                "closed": True,
                "predicted": pred,
                "correct": True,
            }
        )

    stats = {
        "raw": len(raw),
        "dropped_benchmark_leak": dropped_leak,
        "dropped_duplicate": dropped_dup,
        "dropped_unclosed": dropped_unclosed,
        "dropped_teacher_wrong": dropped_wrong,
        "kept": len(traces),
    }
    return traces, stats
