# Audit trail

Every number in the README, plus what had to be fixed to get there. The README
is deliberately short; this is where the confounds and corrections live.

## Measurements

All at an 8,192-token generation cap, greedy, 500 MATH-500 problems.

| model | mode | n | accuracy | mean gen tok | empty think | boxed | no answer |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | chat | 500 | 41.6% | 1,303 | 0% | 89% | 10% |
| Qwen3-0.6B-Base | chat_nothink | 500 | 1.0% | 7,514 | 0% | 9% | — |
| Qwen3-0.6B-Base | chat_bare | 500 | 1.2% | 7,725 | 0% | 5% | — |
| student06_qwen | chat | 500 | 11.0% | 8,142 | 0% | 11% | — |
| Qwen3-1.7B-Base | chat | 500 | 52.6% | 1,141 | 0% | 90% | 10% |
| Qwen3-1.7B-Base | chat_nothink | 500 | 46.8% | — | 0% | — | — |
| Qwen3-1.7B-Base | chat_bare | 500 | 30.0% | — | 0% | — | — |
| student17 | chat | 500 | 49.2% | 1,239 | 0% | 88% | — |
| student17 | chat_nothink | 500 | 43.8% | 1,802 | 0% | 76% | — |
| student17 | chat_bare | 500 | 26.0% | 2,058 | 0% | 53% | — |
| student17_qwen | chat | 500 | 21.8% | 8,075 | 94% | 20% | 53% |

`chat_nothink` and `chat_bare` force the think block closed. The 0.6B scores
~1% there and has no signal to give; the 1.7B retains 46.8% and 30.0%.

## Paired tests

Exact binomial McNemar on discordant pairs, base against student on identical problems.

| experiment | mode | base | student | Δ | broke | fixed | *p* |
|---|---|---|---|---|---|---|---|
| 1.7B weak traces | chat | 52.6% | 49.2% | −3.4 | 48 | 31 | 0.0712 |
| 1.7B weak traces | chat_nothink | 46.8% | 43.8% | −3.0 | 48 | 33 | 0.119 |
| 1.7B weak traces | chat_bare | 30.0% | 26.0% | −4.0 | 50 | 30 | 0.033 |
| 1.7B strong traces | chat | 52.6% | 21.8% | −30.8 | 180 | 26 | 1.57 × 10⁻²⁹ |
| 0.6B strong traces | chat | 41.6% | 11.0% | −30.6 | 162 | 9 | 1.97 × 10⁻³⁷ |

## Training

| | 0.6B (`student06_qwen`) | 1.7B (`student17_qwen`) |
|---|---|---|
| teacher | Qwen3-235B-A22B | Qwen3-235B-A22B |
| traces | 9,680 | 9,680 |
| epochs / steps | 2 / 606 | 2 / 606 |
| lr | 1e-5 | 1e-5 |
| max len | 8,192 | 8,192 |
| GPU | A100-40GB | A100-80GB |
| loss | 0.78 → 0.63 | 0.78 → 0.63 |

`student17` (the weak-trace 1.7B) used 1,575 examples at lr 2e-6 — a different
configuration, not a like-for-like rerun.

## Data preparation

`rasbt/math_distill` → `qwen3_235b_a22b_math_train`:

```
12,000  raw partition
11,978  −1 benchmark leak, −21 duplicates
11,085  −893 where the teacher answered wrong
 9,680  −1,405 exceeding the 8,192 cap
```

Decontamination and teacher correctness are recomputed in `kd/data/external.py`
at load time. The dataset card claims zero benchmark overlap and is wrong by one
problem; the duplicate check found 21 the card does not mention.

The weak traces in `data/traces.jsonl` are 2,000 self-generated, of which 1,656
are closed and correct, mean 2,743 tokens. 1,575 were trained on.

## Controls

**Hardware.** The `compare` function keeps base and student in one container on
one GPU, because greedy decoding is only reproducible within a fixed kernel and
hardware path. Early runs compared an A10G baseline against an A100 student,
which measures the hardware rather than the weights.

The `chat` baseline was re-measured on an A100-80GB after the eval GPU changed
from L4:

```
L4                 52.6%
A100-80GB          52.6%
per-item agreement 500/500 = 100.0%
```

The original L4 baselines are kept in `qwen3_1.7b/_l4_baseline/`.

**Token cap.** The 0.6B student was measured at both caps:

```
4,096  11.2%
8,192  11.0%
```

Truncation is not the cause of the collapse.

## Bugs found and fixed

Each of these produced a wrong conclusion before it was found.

| bug | effect |
|---|---|
| `think_span` searched the completion for both think tags, but the prompt supplies the opening tag | `thinking_rate` pinned at 0.0% on every run ever recorded |
| `chat_nothink` told the model to use tags the template had already closed | invalidated both sides of that comparison |
| 1.7B baselines were hardcoded strings in a `print` | every delta quoted against them was fiction |
| student `chat` score reported as the `chat_nothink` score | manufactured a −14.4 catastrophe from a −2.6 non-effect |
| train targets used `eos` instead of the chat terminator | taught the wrong turn terminator |
| `TRAIN_MAX_LEN` equal to the teacher generation cap | dropped the longest, hardest traces |
| vLLM engine never released between passes | a second generation in one container OOMs |
| `results = []` shadowed the Modal Volume | crash before commit |
| flat comparison filenames | a later run overwrote an earlier one, pairing one model's base against another's student |
| `.gitignore` `data/` also matched `kd/data/` | swallowed `kd/data/prompts.py` |

## Corrections to earlier writeups

| previously reported | actual |
|---|---|
| 1.7B `chat` 36.6% | 47.4% — 36.2% was `chat_nothink`, mislabelled |
| "−14.4 regression" | −2.6, *p* = 0.177 — nothing |
| 1.7B baselines 51.0 / 35.6 / 30.0 | hardcoded strings, no result file existed |
| "0.6B collapses, 1.7B unaffected" | both regress; the 1.7B on strong traces is −30.8 |
| "±0.6 run-to-run noise" | compared two harness versions, not two runs |

## Not yet resolved

- **The unfilled cell.** No 0.6B was ever distilled from the weak traces. An
  earlier attempt produced degenerate output from a run with the wrong
  terminator and lr 2e-6; it is not a usable data point. This is the one
  experiment that would separate trace quality from trace volume.
- **The 0.6B student has no `chat_nothink` or `chat_bare` pass.** Deliberate.
  Its base scores 1.0% and 1.2% there, so there is nothing to measure.
- **The strong-trace 1.7B was measured on `chat` only.** The other two modes
  would need matching baselines re-run on the same GPU to stay paired.
- **Reasoning is not separated from formatting** for the weak-trace student. 9
  of its 48 broken problems had no extractable answer; the other 39 are genuine
  arithmetic errors. The dominant failure is reasoning, not parsing.