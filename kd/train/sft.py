"""SFT on verified teacher traces.

Two things here are load-bearing and easy to get silently wrong:

1. The prompt must be rendered exactly as it is at eval time — same
   instruction, same chat template, enable_thinking=True. Train on one
   format, evaluate on another, and the numbers are noise.
2. Loss is masked to the completion only. Training on the prompt teaches
   the model to generate questions. `check_masking` asserts this rather
   than trusting it.
"""

from __future__ import annotations

IGNORE = -100


def terminator_id(tokenizer) -> int:
    """Token that ends an assistant turn in this model's chat format.

    Qwen3 closes a turn with <|im_end|> (151645), not eos_token
    (<|endoftext|>, 151643). Training on eos teaches the student a stop
    sequence the base model never produced, so at inference the turn boundary
    it actually wants to emit is out of distribution.
    """
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return tokenizer.eos_token_id
    return tid


def build_examples(traces: list[dict], tokenizer, instruction: str, max_len: int):
    """(prompt, completion) pairs with prompt tokens masked out.

    Overlong examples are DROPPED, not truncated: a truncated trace has no
    final answer, and training on it teaches the student to never stop.
    """
    kept, dropped_long, dropped_bad = [], 0, 0
    term = terminator_id(tokenizer)
    for t in traces:
        if not (t.get("closed") and t.get("correct")):
            dropped_bad += 1
            continue

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction.format(t["problem"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        completion = t["generation"].strip()

        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

        if f_ids[: len(p_ids)] != p_ids:
            # Boundary drift would silently cut real supervision.
            raise AssertionError("prompt tokens are not a prefix of the full sequence")

        if f_ids[-1] != term:
            f_ids = f_ids + [term]

        if len(f_ids) > max_len:
            dropped_long += 1
            continue

        labels = [IGNORE] * len(p_ids) + f_ids[len(p_ids) :]
        kept.append({"input_ids": f_ids, "labels": labels})

    return kept, {"dropped_long": dropped_long, "dropped_bad": dropped_bad}


def check_masking(examples: list[dict], tokenizer, n: int = 3) -> None:
    """Decode a sample and assert only the completion is supervised."""
    term = terminator_id(tokenizer)
    for ex in examples[:n]:
        supervised = [i for i, lab in enumerate(ex["labels"]) if lab != IGNORE]
        if not supervised:
            raise AssertionError("no supervised tokens - prompt mask covers everything")
        if supervised[0] == 0:
            raise AssertionError("loss is unmasked from index 0 - prompt is being trained on")
        if supervised[-1] != len(ex["labels"]) - 1:
            raise AssertionError("last token is not supervised - eos/answer not learned")
        first = tokenizer.decode(ex["input_ids"][supervised[0]])
        if not first.strip():
            raise AssertionError(f"supervision starts on whitespace: {first!r}")
        if ex["input_ids"][-1] != term:
            raise AssertionError(
                f"last token is {ex['input_ids'][-1]}, expected turn terminator "
                f"{term}; the model would never learn to stop"
            )


def collate(batch, pad_id: int):
    """Right-pad to the batch width.

    Length-grouped batching is deliberately NOT used: it would pair a 4096-token
    trace with a 300-token one and inflate the padded width that drives the
    logits tensor (151936 vocab x width x 4 bytes in fp32).
    """
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [IGNORE] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def train(
    examples,
    tokenizer,
    out_dir: str,
    lr: float = 2e-6,
    epochs: int = 3,
    micro_batch: int = 2,
    accum: int = 64,
    max_len: int = 4096,
    warmup_ratio: float = 0.03,
    log_every: int = 20,
    smoke_steps: int = 0,
):
    import dataclasses

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    import transformers

    # vLLM resolves its own transformers; the container ships v5, which
    # dropped warmup_ratio in favour of warmup_steps. Compute it explicitly
    # and report anything this version cannot accept, rather than crashing
    # on a renamed argument.
    steps_per_epoch = (len(examples) + micro_batch * accum - 1) // (micro_batch * accum)
    total_steps = max(steps_per_epoch * epochs, 1)

    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(int(round(total_steps * warmup_ratio)), 1),
        weight_decay=0.1,
        adam_beta2=0.95,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=log_every,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # Logits are computed for ALL positions, including the ~160 prompt
        # tokens we mask. At 151936 vocab that is ~1GB per 4096-token row in
        # fp32, wasted on positions with no loss. Slicing before the loss keeps
        # it, without rewriting Trainer internals.
    )
    valid = {f.name for f in dataclasses.fields(TrainingArguments)}
    # max_steps=0 means "zero steps" to transformers, not "unlimited". Only set
    # it for the smoke run, or the real run silently trains for one step.
    if smoke_steps:
        kwargs["max_steps"] = smoke_steps
        kwargs["num_train_epochs"] = 1
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        print(f"[train] transformers {transformers.__version__} ignores: {dropped}")
    args = TrainingArguments(**{k: v for k, v in kwargs.items() if k in valid})
    print(f"[train] {len(examples)} examples | {total_steps} optimiser steps | warmup {kwargs['warmup_steps']}")

    # Load the model ourselves rather than handing Trainer a path string:
    # transformers v5 mishandles the string form inside _move_model_to_device.
    model = AutoModelForCausalLM.from_pretrained(tokenizer.name_or_path, dtype=torch.bfloat16)
    model.config.use_cache = False

    # Without this, gradient checkpointing silently does nothing: the inputs to
    # the first checkpointed block do not require grad, so autograd records no
    # graph and nothing is recomputed. Costs ~4GB here.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=examples,
        data_collator=lambda b: collate(b, pad_id),
    )

    if torch.cuda.is_available():
        print(f"[train] cuda {torch.cuda.get_device_name(0)} "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")

    result = trainer.train()
    done_steps = result.global_step
    expected = smoke_steps or total_steps
    if done_steps < expected:
        raise AssertionError(
            f"trained {done_steps} steps, expected {expected} - checkpoint is not usable"
        )
    print(f"[train] completed {done_steps}/{expected} steps")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    supervised = sum(sum(1 for lab in ex["labels"] if lab != IGNORE) for ex in examples)
    print(f"[train] loss {result.training_loss:.4f}")
    print(f"[train] examples {len(examples)}  supervised tokens {supervised:,}")
    return out_dir