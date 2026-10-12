"""Generation + scoring. No Modal imports, so scoring logic stays portable."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass

from ..config import MATH_SOURCE
from .extract import extract_boxed, last_boxed_or_fallback, match_tier

INSTRUCTION = (
    "Solve the following problem. Reason step by step inside <think> tags, "
    "then give the final answer inside \\boxed{{}}.\n\n{}"
)

# Same instruction with the think-tag clause removed. With enable_thinking=False
# the chat template has ALREADY emitted a closed `<think>\n\n</think>` block,
# so telling the model to "reason step by step inside <think> tags" demands
# something structurally impossible — the block it is told to write into is
# shut. That contradiction, not the model, produced the old 8.6% baseline and
# the "+8.8 improvement" that was built on top of it.
INSTRUCTION_NOTHINK = (
    "Solve the following problem. Reason step by step, "
    "then give the final answer inside \\boxed{{}}.\n\n{}"
)

_MODE_INSTRUCTION = {"chat": INSTRUCTION, "chat_nothink": INSTRUCTION_NOTHINK}

# Bare problem, no instructions. A Base checkpoint has no instruction tuning,
# so the wording of INSTRUCTION may be doing most of the work: the same model
# scored 1.6% here and 41.4% with the full instruction. This variant tests
# whether a baseline is inflated by prompt phrasing rather than by capability.
# The chat template still appends an empty <think></think> block, so "bare"
# describes the user text only, not the whole prompt.
BARE = "{}"


@dataclass
class EvalResult:
    model: str
    benchmark: str
    prompt_mode: str
    n: int
    accuracy: float
    mean_gen_tokens: float
    thinking_rate: float
    mean_think_tokens: float
    empty_think_rate: float
    tokens_per_correct: float
    boxed_found_rate: float
    fallback_rate: float
    tier_counts: dict
    samples: list

    def to_json(self, path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_problems(benchmark: str, n: int) -> list[dict]:
    from datasets import load_dataset

    if benchmark == "math500":
        rows = load_dataset(MATH_SOURCE, split="test")
        return [
            {"problem": r["problem"], "solution": r.get("answer", r["solution"])}
            for r in list(rows)[:n]
        ]
    if benchmark == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        return [
            {"problem": r["question"], "solution": r["answer"].split("####")[-1].strip()}
            for r in list(rows)[:n]
        ]
    raise ValueError(benchmark)


def build_prompt(tokenizer, problem: str, mode: str) -> str:
    """Four modes, and picking the wrong one is the difference between a real
    number and a broken run.

    chat         - Qwen3 template, thinking left open, for the distilled student
    chat_nothink - instruction with no think clause, block pre-closed by the
                   template. The clause is removed because the model cannot
                   write into a block the template already shut.
    chat_bare    - no instruction at all; the template's empty think block is
                   still present. Tests how much of a baseline is prompt phrasing.
    """
    if mode in ("chat", "chat_nothink", "chat_bare"):
        body = problem if mode == "chat_bare" else _MODE_INSTRUCTION[mode].format(problem)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=mode == "chat",
        )
    raise ValueError(mode)


def think_span(generation: str, prompt_opens_think: bool = False) -> str | None:
    """Text the model produced as reasoning, or None if it never started.

    An empty string means the block was opened and closed with nothing inside —
    the failure mode where a model mimics the format without thinking. Must be
    reported separately from thinking_rate, which would score it 100%.

    `prompt_opens_think` must be True whenever the prompt was built with
    enable_thinking=True. The opening tag then lives in the PROMPT, so a
    completion-only search finds no pair of tags and returns None for a model
    that reasoned perfectly. That bug pinned thinking_rate at 0.0 on every run
    ever recorded, which is why a student that had stopped reasoning entirely
    still looked healthy.
    """
    if prompt_opens_think:
        if "<think>" in generation:
            return generation.split("<think>", 1)[1].split("</think>", 1)[0]
        # Opening tag came from the prompt. Everything up to the close is
        # reasoning; if it never closed, the whole generation is an unfinished
        # think block and is reported as such rather than discarded.
        head, sep, _ = generation.partition("</think>")
        return head if sep else generation
    if "<think>" not in generation or "</think>" not in generation:
        return None
    return generation.split("<think>", 1)[1].split("</think>", 1)[0]


def generate(model: str, prompts: list[str], max_new_tokens: int, stop=None,
             gpu_mem: float = 0.92):
    """Greedy generation. `stop` halts on a literal string (unused by the chat
    modes). gpu_mem 0.92 buys KV cache for more concurrent requests.

    The engine is torn down explicitly. Anything that generates more than once
    per container — the prompt sweep, and `compare`, which runs six passes —
    otherwise constructs a second LLM at 0.92 of whatever memory the first
    still holds, and dies on OOM after paying to load the weights twice.
    """
    import gc

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="auto", gpu_memory_utilization=gpu_mem,
              max_model_len=max_new_tokens + 1024)
    try:
        outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                    max_tokens=max_new_tokens, stop=stop))
        return (
            [o.outputs[0].text for o in outs],
            [len(o.outputs[0].token_ids) for o in outs],
        )
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score(model, benchmark, prompt_mode, problems, generations, token_counts, tokenizer) -> EvalResult:
    n = len(problems)
    tiers, samples = {}, []
    correct = boxed = fallback = 0
    think_tokens, tagged, empty_tagged = [], 0, 0
    # Only "chat" sets enable_thinking=True, so only it puts the opening tag in
    # the prompt. Every other mode needs the strict both-tags-in-generation path.
    opens_think = prompt_mode == "chat"

    for prob, gen, ntok in zip(problems, generations, token_counts):
        pred = last_boxed_or_fallback(gen)
        if extract_boxed(gen) is not None:
            boxed += 1
        elif pred is not None:
            fallback += 1

        span = think_span(gen, opens_think)
        if span is not None:
            tagged += 1
            stripped = span.strip()
            if not stripped:
                empty_tagged += 1
            else:
                think_tokens.append(len(tokenizer.encode(stripped, add_special_tokens=False)))

        tier = match_tier(pred, prob["solution"]) if pred is not None else None
        tiers[tier] = tiers.get(tier, 0) + 1
        correct += bool(tier)
        samples.append({
            "problem": prob["problem"][:400], "reference": prob["solution"],
            "predicted": pred, "correct": bool(tier), "match_tier": tier,
            "gen_tokens": ntok,
            "head": gen[:400],
            "think_tokens": len(tokenizer.encode(span, add_special_tokens=False)) if span else 0,
            "completion": gen[-1200:],
        })

    return EvalResult(
        model=model, benchmark=benchmark, prompt_mode=prompt_mode, n=n,
        accuracy=correct / n if n else 0.0,
        mean_gen_tokens=statistics.fmean(token_counts) if token_counts else 0.0,
        thinking_rate=tagged / n if n else 0.0,
        mean_think_tokens=statistics.fmean(think_tokens) if think_tokens else 0.0,
        empty_think_rate=empty_tagged / max(tagged, 1),
        tokens_per_correct=sum(token_counts) / max(correct, 1) if n else 0.0,
        boxed_found_rate=boxed / n if n else 0.0,
        fallback_rate=fallback / n if n else 0.0,
        tier_counts={str(k): v for k, v in tiers.items()},
        samples=samples,
    )


def table(results: list[EvalResult]) -> str:
    cols = ("model", "benchmark", "mode", "n", "acc", "gen_tok", "thinks",
            "think_tok", "empty_think", "tok/corr", "boxed")
    rows = [[
        r.model, r.benchmark, r.prompt_mode, str(r.n),
        f"{r.accuracy:.1%}", f"{r.mean_gen_tokens:.0f}",
        f"{r.thinking_rate:.1%}", f"{r.mean_think_tokens:.0f}",
        f"{r.empty_think_rate:.1%}", f"{r.tokens_per_correct:.0f}",
        f"{r.boxed_found_rate:.1%}",
    ] for r in results]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    out = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    return "\n".join(out)