import json

import modal

from kd.config import (BASE_GPU, EVAL_GPU, GEN_CHUNK, MATH_N, MAX_NEW_TOKENS, N_PROMPTS,
                        SMOKE_N, STUDENT_06, STUDENT_17, TEACHER, TEACHER_GPU, TRAIN_EPOCHS,
                        TRAIN_GPU, TRAIN_MAX_LEN, TRAIN_MICRO_BATCH, TRAIN_ACCUM,
                        TRAIN_ALLOC_CONF, TRAIN_SMOKE_STEPS, TRAIN_LR, TRAIN_OUT)

app = modal.App("distillation")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("results", create_if_missing=True)
data = modal.Volume.from_name("kd-data", create_if_missing=True)
artifacts = modal.Volume.from_name("kd-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.11", "torch>=2.9", "transformers>=4.56", "datasets>=5.0", "sympy>=1.13",
        "accelerate>=1.0"
    )
    .env({
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
            "PYTORCH_CUDA_ALLOC_CONF": TRAIN_ALLOC_CONF,
        })
    .add_local_dir("kd", "/root/kd")
)

VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/results": results,
          "/root/data": data, "/root/artifacts": artifacts}

@app.function(image=image, gpu=TEACHER_GPU, volumes=VOLUMES, timeout=60 * 60 * 8)
def teacher_smoke(n: int = SMOKE_N):
    """Run the teacher on a handful of problems so we can SEE a real reasoning
    trace before spending money on thousands.

    Measures what decides whether the full run is viable: does thinking mode
    produce parseable \\boxed{} answers, how long are the traces, is the answer
    actually correct, and what is the real token throughput.
    """
    import time
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import MIN_THROUGHPUT_TPS, TEACHER_MAX_NEW_TOKENS
    from kd.data.prompts import sample_prompts
    from kd.eval.extract import extract_boxed, last_boxed_or_fallback, match_tier, selfcheck
    from kd.eval.harness import INSTRUCTION, generate, think_span

    selfcheck()
    print(f"[smoke] teacher={TEACHER} gpu={TEACHER_GPU} n={n}")

    tok = AutoTokenizer.from_pretrained(TEACHER)
    rows, dropped = sample_prompts(n)
    print(f"[smoke] sampled {len(rows)} prompts, dropped {dropped} contaminated")

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION.format(r["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for r in rows
    ]

    t0 = time.time()
    texts, counts = generate(TEACHER, prompts, TEACHER_MAX_NEW_TOKENS)
    elapsed = time.time() - t0

    total_out = sum(counts)
    tps = total_out / max(elapsed, 1e-9)
    correct = tagged = boxed = truncated = 0
    think_lens = []
    for r, gen, c in zip(rows, texts, counts):
        pred = last_boxed_or_fallback(gen)
        if pred is not None and match_tier(pred, r["answer"]) is not None:
            correct += 1
        if extract_boxed(gen) is not None:
            boxed += 1
        if "</think>" not in gen:
            truncated += 1
        span = think_span(gen)
        if span is not None and span.strip():
            tagged += 1
            think_lens.append(len(tok.encode(span, add_special_tokens=False)))

    n = len(rows)
    print(f"\n[smoke] ---- throughput ----")
    print(f"  wall time        : {elapsed:.0f}s")
    print(f"  output tokens    : {total_out:,}")
    print(f"  throughput       : {tps:.0f} tok/s   (want > {MIN_THROUGHPUT_TPS:.0f})")
    print(f"[smoke] ---- trace quality ----")
    print(f"  correct answers  : {correct}/{n} = {correct / n:.1%}")
    print(f"  has \\boxed{{}}     : {boxed}/{n} = {boxed / n:.1%}")
    print(f"  non-empty think  : {tagged}/{n} = {tagged / n:.1%}")
    print(f"  truncated        : {truncated}/{n} = {truncated / n:.1%}")
    print(f"  mean think tokens: {sum(think_lens) // max(len(think_lens), 1)}")
    print(f"  mean trace tokens: {total_out // n}")

    print(f"\n[smoke] ---- one full trace (so you can see what reasoning looks like) ----")
    print(texts[0][:4000])

    out = Path("/root/results/teacher_smoke.json")
    out.write_text(
        json.dumps(
            {
                "teacher": TEACHER,
                "n": n,
                "contaminated_dropped": dropped,
                "throughput_tps": tps,
                "wall_s": elapsed,
                "accuracy": correct / n,
                "boxed_rate": boxed / n,
                "think_rate": tagged / n,
                "truncated_rate": truncated / n,
                "mean_think_tokens": sum(think_lens) // max(len(think_lens), 1),
                "mean_trace_tokens": total_out // n,
                "traces": [
                    {"problem": r["problem"], "solution": r["answer"], "generation": g}
                    for r, g in zip(rows, texts)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results.commit()

    verdict = "VIABLE" if (tps >= MIN_THROUGHPUT_TPS and correct / n >= 0.4) else "RETHINK"
    print(f"\n[smoke] verdict: {verdict}")
    if verdict == "RETHINK":
        print("  -> if throughput is low, switch to TEACHER_FALLBACK (30B-A3B MoE)")
        print("  -> if accuracy is low, inspect traces before generating 6000")

@app.function(image=image, gpu=TEACHER_GPU, volumes=VOLUMES, timeout=60 * 60 * 12)
def teacher_generate(n: int = N_PROMPTS):
    """The real run. Writes traces to the data volume in chunks and commits
    after each one, so a crash at minute 110 does not lose 110 minutes.

    Resumable: rerun after a failure and it skips prompts already on disk.
    """
    import hashlib
    import time
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import GEN_CHUNK, TEACHER_MAX_NEW_TOKENS
    from kd.data.prompts import sample_prompts
    from kd.eval.extract import last_boxed_or_fallback, match_tier, selfcheck
    from kd.eval.harness import INSTRUCTION, generate, think_span

    selfcheck()
    out_path = Path("/root/data/traces.jsonl")

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["pid"])
        print(f"[gen] resuming: {len(done)} traces already on disk")

    tok = AutoTokenizer.from_pretrained(TEACHER)
    rows, dropped = sample_prompts(n)
    print(f"[gen] {len(rows)} prompts sampled, {dropped} dropped as contaminated")

    def pid(r):
        return hashlib.sha1(r["problem"].encode()).hexdigest()[:16]

    todo = [r for r in rows if pid(r) not in done]
    print(f"[gen] {len(todo)} to generate, cap={TEACHER_MAX_NEW_TOKENS} tokens")

    t0 = time.time()
    written = complete = 0
    for start in range(0, len(todo), GEN_CHUNK):
        chunk = todo[start : start + GEN_CHUNK]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION.format(r["problem"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for r in chunk
        ]
        texts, counts = generate(TEACHER, prompts, TEACHER_MAX_NEW_TOKENS)

        with out_path.open("a", encoding="utf-8") as fh:
            for r, g, c in zip(chunk, texts, counts):
                pred = last_boxed_or_fallback(g)
                correct = bool(pred) and match_tier(pred, r["answer"]) is not None
                span = think_span(g)
                fh.write(
                    json.dumps(
                        {
                            "pid": pid(r),
                            "problem": r["problem"],
                            "answer": r["answer"],
                            "level": r["level"],
                            "subject": r["subject"],
                            "generation": g,
                            "gen_tokens": c,
                            "closed": "</think>" in g,
                            "predicted": pred,
                            "correct": correct,
                            "think_tokens": len(
                                tok.encode(span, add_special_tokens=False)
                            )
                            if span
                            else 0,
                        }
                    )
                    + "\n"
                )
                written += 1
                complete += "</think>" in g
        data.commit()

        el = time.time() - t0
        rate = written / max(el, 1e-9)
        left = (len(todo) - start - len(chunk)) / max(rate, 1e-9)
        print(
            f"[gen] {written}/{len(todo)} traces | closed {complete}/{written} "
            f"({complete / written:.1%}) | {rate:.2f} traces/s | ~{left / 60:.0f} min left"
        )

    print(f"[gen] done -> {out_path}")

@app.function(image=image, gpu=TRAIN_GPU, volumes=VOLUMES, timeout=60 * 60 * 8)
def train_student(smoke_only: bool = False):
    """SFT on the verified teacher traces. Model comes from TRAIN_STUDENT."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import (TRAIN_EPOCHS, TRAIN_LR, TRAIN_MAX_LEN,
                        TRAIN_MICRO_BATCH, TRAIN_OUT, TRAIN_STUDENT)
    from kd.eval.extract import selfcheck
    from kd.eval.harness import INSTRUCTION
    from kd.train.sft import build_examples, check_masking, train

    out_dir = "/root/artifacts/smoke" if smoke_only else TRAIN_OUT
    # Print unconditionally: a smoke run that short-circuits out_dir would
    # never evaluate TRAIN_OUT, and a NameError would only surface on the
    # expensive path. This forces both branches to resolve in 3 steps.
    print(f"[train] model {TRAIN_STUDENT} -> {out_dir}")

    selfcheck()
    from kd.config import LOCAL_TRACES, TRACE_SOURCE

    if TRACE_SOURCE == "external":
        from kd.data.external import load_external_traces

        traces, stats = load_external_traces()
        print(f"[train] source={TRACE_SOURCE}")
        for k, v in stats.items():
            print(f"[train]   {k:<26} {v}")
    else:
        src = Path(LOCAL_TRACES)
        traces = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[train] source={TRACE_SOURCE}  {len(traces)} traces from {src}")

    tok = AutoTokenizer.from_pretrained(TRAIN_STUDENT)
    examples, stats = build_examples(traces, tok, INSTRUCTION, TRAIN_MAX_LEN)
    print(f"[train] kept {len(examples)} | dropped {stats}")
    if not examples:
        raise SystemExit("no usable examples - check the filter stats above")

    check_masking(examples, tok)
    print("[train] prompt masking verified")

    train(
        examples,
        tok,
        out_dir=out_dir,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        micro_batch=TRAIN_MICRO_BATCH,
        accum=TRAIN_ACCUM,
        max_len=TRAIN_MAX_LEN,
        smoke_steps=TRAIN_SMOKE_STEPS if smoke_only else 0,
    )
    artifacts.commit()
    print(f"[train] saved -> {out_dir}")

@app.function(image=image, gpu=BASE_GPU, volumes=VOLUMES, timeout=60 * 60 * 3)
def baseline_17b():
    """Baseline for Qwen3-1.7B-Base, identical harness to the 0.6B numbers.

    from transformers import AutoTokenizer

    from kd.config import MATH_N, MAX_NEW_TOKENS, STUDENT_17
    from kd.eval.extract import selfcheck
    from kd.eval.harness import build_prompt, generate, load_problems, score, table

    selfcheck()
    tok = AutoTokenizer.from_pretrained(STUDENT_17)
    problems = load_problems("math500", MATH_N)

    scored = []
    for mode in ("chat_bare",):
        prompts = [build_prompt(tok, p["problem"], mode) for p in problems]
        print(f"\n[1.7B mode={mode}] prompt tail: {prompts[0][-80:]!r}")
        scored.append(
            score(STUDENT_17, "math500", mode, problems,
                  *generate(STUDENT_17, prompts, MAX_NEW_TOKENS), tokenizer=tok)
        )
        print(table([scored[-1]]))

    print("\n" + "=" * 78)
    print(table(scored))
    print(f"\n[reference] 0.6B base chat = 41.4%  |  Raschka 0.6B = 15.2%")
    print("[note] chat/chat_nothink are NOT run here — use `compare`, which pairs")
    print("       base and student on one GPU so the delta is meaningful.")

@app.local_entrypoint()
def base17():
    from modal import Function

    call = Function.from_name("distillation", "baseline_17b").spawn()
    print(f"[launched] baseline_17b call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")

@app.function(image=image, gpu=EVAL_GPU, volumes=VOLUMES, timeout=60 * 60 * 3)
def baseline_prompt_sweep():
    """Does our 42.6% baseline come from the model, or from our prompt?

    Same 500 problems, same scorer, three prompt wordings. Raschka reports
    15.2% for this checkpoint; we measure 42.6% with a verbose instruction and
    1.6% with empty think tags pre-filled. If the bare prompt lands near 15%,
    our baseline is prompt phrasing, and the honest comparison is against our
    own lowest-instruction variant rather than against his number.
    """
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import MATH_N, MAX_NEW_TOKENS, STUDENT_06
    from kd.eval.extract import selfcheck
    from kd.eval.harness import build_prompt, generate, load_problems, score, table

    selfcheck()
    tok = AutoTokenizer.from_pretrained(STUDENT_06)
    problems = load_problems("math500", MATH_N)

    scored = []
    for mode in ("chat_bare", "chat_nothink", "chat"):
        prompts = [build_prompt(tok, p["problem"], mode) for p in problems]
        print(f"\n[mode={mode}] prompt tail: {prompts[0][-90:]!r}")
        scored.append(
            score(STUDENT_06, "math500", mode, problems,
                  *generate(STUDENT_06, prompts, MAX_NEW_TOKENS), tokenizer=tok)
        )
        print(table([scored[-1]]))

    print("\n" + "=" * 78)
    print(table(scored))
    print("\n[reference] Raschka reports 15.2% for Qwen3-0.6B-Base on MATH-500")

    out = Path("/root/results/baseline_prompt_sweep.json")
    out.write_text(json.dumps([r.__dict__ for r in scored], indent=2, default=str))
    results.commit()

@app.local_entrypoint()
def sweep():
    from modal import Function

    call = Function.from_name("distillation", "baseline_prompt_sweep").spawn()
    print(f"[launched] baseline_prompt_sweep call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")

@app.function(image=image, gpu=TRAIN_GPU, volumes=VOLUMES, timeout=60 * 60 * 6)
def eval_student(modes: tuple = ("chat",), model_dir: str = None):
    """Score the distilled 1.7B student against its own 51.0% baseline.

    thinking_rate says whether it emits the tags; empty_think_rate catches a
    model that copies the format with nothing inside; mean_think_tokens
    measures whether real reasoning came along.
    """
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import MATH_N, MAX_NEW_TOKENS, TRAIN_OUT
    from kd.eval.extract import selfcheck
    from kd.eval.harness import build_prompt, generate, load_problems, score, table

    selfcheck()
    model_dir = model_dir or TRAIN_OUT
    if not Path(model_dir).exists():
        raise SystemExit("no trained student - run train --smoke 0 first")

    tok = AutoTokenizer.from_pretrained(model_dir)
    problems = load_problems("math500", MATH_N)

    # Score at every prompt mode, not just the strongest one. The base model
    # reasons only when prompted to; the distilled student reasons always. A
    # single mode compares those two behaviours and calls the difference a
    # regression, so every row needs its matched baseline.
    scored = []
    for mode in modes:
        prompts = [build_prompt(tok, p["problem"], mode) for p in problems]
        result = score(Path(model_dir).name, "math500", mode, problems,
                       *generate(model_dir, prompts, MAX_NEW_TOKENS), tokenizer=tok)
        scored.append(result)
        result.to_json(Path(f"/root/results/student_{Path(model_dir).name}_{mode}.json"))
        results.commit()
        print(table([result]))

    print("\n" + "=" * 78)
    print(table(scored))
    print("\n[eval] matched baselines for Qwen3-1.7B-Base:")
    print("  chat        51.0%")
    print("  chat_nothink 35.6%")
    print("  chat_bare      ?")
    print("\n[eval] thinking rate / empty-think / mean think tokens per mode:")
    for r in scored:
        print(f"  {r.prompt_mode:<14} {r.thinking_rate:>6.1%} {r.empty_think_rate:>8.1%} "
              f"{r.mean_think_tokens:>8.0f}")


@app.function(image=image, gpu=EVAL_GPU, volumes=VOLUMES, timeout=60 * 60 * 4)
def compare(modes: tuple = ("chat", "chat_nothink", "chat_bare"),
            base: str = STUDENT_17, student_dir: str = None):
    """Base and distilled student on ONE GPU, in ONE container.

    Separate runs are not comparable. `baseline_17b` ran on an A10G while
    `eval_student` ran on an A100; greedy decoding is only reproducible within
    a fixed kernel and hardware path, so a cross-GPU delta carries an error bar
    nobody ever measured. Here both endpoints share the GPU, the vLLM build,
    the problems and the scorer, so the difference is attributable to weights.

    This is also the only path that produces a 1.7B baseline at all. Every
    baseline number previously quoted for this student was a hardcoded string
    in a print statement, with no result file behind it.
    """
    from pathlib import Path

    from transformers import AutoTokenizer

    from kd.config import MATH_N, MAX_NEW_TOKENS, TRAIN_OUT
    from kd.eval.extract import selfcheck
    from kd.eval.harness import EvalResult, build_prompt, generate, load_problems, score, table

    selfcheck()
    student_dir = student_dir or TRAIN_OUT
    if not Path(student_dir).exists():
        raise SystemExit(f"no trained student at {student_dir} - run train --smoke 0 first")

    problems = load_problems("math500", MATH_N)
    print(f"[compare] max_new_tokens={MAX_NEW_TOKENS}  (must exceed trained trace length)")
    by_mode = {}
    for tag, model_dir in (("base", base), ("student", student_dir)):
        tok = AutoTokenizer.from_pretrained(model_dir)
        # Model-scoped filenames. A flat cmp_<tag>_<mode>.json means a second
        # run silently overwrites the first, and reading a half-overwritten set
        # pairs one model's base against another model's student.
        slug = Path(model_dir).name.replace(".", "-").replace("/", "-")
        for mode in modes:
            out = Path(f"/root/results/cmp_{tag}_{slug}_{mode}.json")
            if out.exists():
                # Resume, not redo. A partial run leaves completed passes on
                # disk; re-running blindly pays for them a second time.
                by_mode[(tag, mode)] = EvalResult(**json.loads(out.read_text(encoding="utf-8")))
                print(f"[compare] reusing {out.name} acc={by_mode[(tag, mode)].accuracy:.1%}",
                      flush=True)
                continue
            prompts = [build_prompt(tok, p["problem"], mode) for p in problems]
            print(f"\n[{tag} mode={mode}] prompt tail: {prompts[0][-90:]!r}", flush=True)
            r = score(slug, "math500", mode, problems,
                      *generate(model_dir, prompts, MAX_NEW_TOKENS), tokenizer=tok)
            r.to_json(out)
            by_mode[(tag, mode)] = r
            print(table([r]), flush=True)
            results.commit()

    print("\n" + "=" * 78)
    print("PAIRED  base vs student   (same GPU, same problems, same scorer)")
    print(f"{'mode':<14}{'base':>9}{'student':>10}{'delta':>9}"
          f"{'base_think_tok':>16}{'stu_think_tok':>15}")
    for mode in modes:
        b, s = by_mode[("base", mode)], by_mode[("student", mode)]
        print(f"{mode:<14}{b.accuracy:>8.1%}{s.accuracy:>10.1%}"
              f"{s.accuracy - b.accuracy:>+9.1%}"
              f"{b.mean_think_tokens:>16.0f}{s.mean_think_tokens:>15.0f}")
    print("\nRead the delta against the run-to-run band, never against zero.")


@app.local_entrypoint()
def compare_cli(modes: str = "chat,chat_nothink,chat_bare", base: str = STUDENT_17,
                model: str = ""):
    from modal import Function

    ms = tuple(m.strip() for m in str(modes).split(",") if m.strip())
    call = Function.from_name("distillation", "compare").spawn(ms, base, model or None)
    print(f"[launched] compare call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")


@app.local_entrypoint()
def evaluate(modes: str = "chat,chat_nothink,chat_bare", model: str = ""):
    from modal import Function

    modes = [m.strip() for m in str(modes).split(",") if m.strip()]
    call = Function.from_name("distillation", "eval_student").spawn(tuple(modes), model)
    print(f"[launched] eval_student call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")


@app.local_entrypoint()
def train(smoke: str = "1"):
    """Pass --smoke 0 for the real run. Default is 3 steps: OOM only shows up
    at the first backward pass, so verify before spending an hour."""
    from modal import Function

    smoke_only = str(smoke).lower() not in ("0", "false", "no")
    call = Function.from_name("distillation", "train_student").spawn(smoke_only)
    kind = "3-step memory smoke" if smoke_only else "full 3-epoch training"
    print(f"[launched] {kind}  call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")

@app.local_entrypoint()
def gen(n: int = N_PROMPTS):
    from modal import Function

    call = Function.from_name("distillation", "teacher_generate").spawn(n)
    print(f"[launched] teacher_generate call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")

@app.local_entrypoint()
def smoke(n: int = SMOKE_N):
    from modal import Function

    call = Function.from_name("distillation", "teacher_smoke").spawn(n)
    print(f"[launched] teacher_smoke call_id={call.object_id}")
    print("[monitor] modal app logs distillation --follow")
