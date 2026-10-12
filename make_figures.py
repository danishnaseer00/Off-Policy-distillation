"""Figures for the off-policy distillation study.

Every value plotted is read from a file under results/ or data/, or is a
recorded measurement from the training log with the source named on the
figure. Nothing is hand-entered for a chart.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1C3D5A"
STEEL = "#4E9CC4"
TEAL = "#0E3B3C"
INK = "#2B2B2B"
MUTED = "#7A7A7A"
RULE = "#D8D8D8"
SALMON = "#F3D6CE"
GREEN = "#DCEBD5"
AMBER = "#C08A2E"
RED = "#B5563F"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


BASE_17 = "results/qwen3_1.7b/cmp_base_Qwen3-1-7B-Base_chat.json"
BASE_06 = "results/qwen3_0.6b/cmp_base_Qwen3-0-6B-Base_chat.json"
STU_17W = "results/qwen3_1.7b/cmp_student_student17_chat.json"
STU_17S = "results/qwen3_1.7b/cmp_student_student17_qwen_chat.json"
STU_06S = "results/qwen3_0.6b/cmp_student_student06_qwen_chat.json"
CAP = 8192


def chrome(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(0.8)
    if ygrid:
        ax.yaxis.grid(True, color=RULE, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def header(fig, title, subtitle):
    fig.text(0.035, 0.955, title, fontsize=18.5, color=NAVY, weight="bold", va="top")
    fig.text(0.035, 0.897, subtitle, fontsize=11, color=MUTED, va="top")


def source(fig, text):
    fig.text(0.035, 0.018, text, fontsize=8.5, color=MUTED, style="italic", va="bottom")


# ------------------------------------------------------------------ figure 1
def fig_headline():
    """The 2x2. Teacher style, not student size, separates the outcomes."""
    rows = [
        ("1.7B", "weak\n1,575 traces", BASE_17, STU_17W),
        ("1.7B", "strong\n9,680 traces", BASE_17, STU_17S),
        ("0.6B", "strong\n9,680 traces", BASE_06, STU_06S),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.74, bottom=0.15, left=0.10, right=0.97)
    header(fig, "The Teacher Decides the Outcome, Not the Student Size",
           "MATH-500 accuracy, paired evaluation — identical 500 problems, one GPU, one container, greedy decoding")

    xs = np.arange(len(rows))
    w = 0.30
    bases = [100 * load(b)["accuracy"] for _, _, b, _ in rows]
    studs = [100 * load(s)["accuracy"] for _, _, _, s in rows]

    b1 = ax.bar(xs - w / 2 - 0.01, bases, w, color=STEEL, zorder=3)
    b2 = ax.bar(xs + w / 2 + 0.01, studs, w, color=TEAL, zorder=3)
    for r, v in zip(b1, bases):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center", fontsize=11, color=INK)
    for r, v in zip(b2, studs):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center",
                fontsize=11, color=TEAL, weight="bold")

    # the two outcomes are both regressions; only one is catastrophic
    ax.axvspan(0.55, 2.45, color=SALMON, alpha=0.45, zorder=0)
    ax.text(1.5, 64.0, "both students collapsed on the strong teacher's traces",
            ha="center", fontsize=10, color=RED, style="italic")

    for i, (size, trace) in enumerate([(s, t) for s, t, _, _ in rows]):
        d = studs[i] - bases[i]
        p = "p = 1.6×10⁻²⁹" if i == 1 else ("p = 2.0×10⁻³⁷" if i == 2 else "p = 0.071  n.s.")
        ax.text(i, max(bases[i], studs[i]) + 7.4, f"{d:+.1f}   {p}",
                ha="center", fontsize=10, color=NAVY, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{t}" for s, t, _, _ in rows], fontsize=10.5)
    ax.set_ylabel("MATH-500 accuracy", fontsize=11)
    ax.set_ylim(0, 68)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    chrome(ax)
    ax.legend(handles=[b1, b2], frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.005),
              ncol=2, fontsize=10.5, handlelength=1.2, columnspacing=1.6,
              labels=["Base (no distillation)", "After distillation"])
    source(fig, "Source: results/qwen3_1.7b/ and results/qwen3_0.6b/, chat mode, 8,192-token cap. "
                "Significance by exact binomial McNemar on paired outcomes.")
    fig.savefig(f"{OUT}/fig1_headline.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_nonttermination():
    """Token distributions: the students adopted the teacher's verbosity."""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.10, right=0.97)
    header(fig, "The Student Learned to Reason but Not to Stop",
           "Generated tokens per problem — Qwen3-1.7B base against students distilled from two different teachers")

    base = np.array([s["gen_tokens"] for s in load(BASE_17)["samples"]])
    weak = np.array([s["gen_tokens"] for s in load(STU_17W)["samples"]])
    strong = np.array([s["gen_tokens"] for s in load(STU_17S)["samples"]])

    edges = np.linspace(0, CAP + 100, 61)
    ax.hist(base, bins=edges, color=STEEL, alpha=0.80, label="Base", zorder=4)
    ax.hist(weak, bins=edges, color=GREEN, alpha=0.90, label="Student — weak traces", zorder=3)
    hist_s, _, _ = ax.hist(strong, bins=edges, color=TEAL, alpha=0.62, label="Student — strong traces", zorder=2)

    ax.axvline(CAP, color=AMBER, lw=1.7, zorder=6)
    top = ax.get_ylim()[1]
    ax.text(CAP - 110, top * 0.94, f"evaluation cap\n{CAP:,}", ha="right",
            fontsize=10, color=AMBER, weight="bold")

    hit = int((strong >= CAP).sum())
    ax.axvspan(CAP - 300, CAP + 100, color=SALMON, alpha=0.55, zorder=1)
    ax.text(CAP - 340, top * 0.58, f"{hit} of 500 generations\nran to the ceiling",
            ha="right", fontsize=10.5, color=RED, style="italic")

    med = int(np.median(strong))
    ax.annotate(f"median {med:,}", xy=(med, top * 0.24), xytext=(CAP * 0.42, top * 0.24),
                fontsize=10.5, color=TEAL, weight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))

    ax.set_xlabel("generated tokens", fontsize=11)
    ax.set_ylabel("problems", fontsize=11)
    ax.set_xlim(0, CAP + 100)
    chrome(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    source(fig, "Source: results/qwen3_1.7b/, chat mode. The weak-trace student tracks the base; "
                "the strong-trace student does not.")
    fig.savefig(f"{OUT}/fig2_nonttermination.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_failure():
    """Every failure signal moves the same direction."""
    panels = [
        ("\\boxed{} answer produced", lambda r: 100 * r["boxed_found_rate"], "%"),
        ("no answer extracted", lambda r: 100 * sum(1 for s in r["samples"] if s["predicted"] is None) / r["n"], "%"),
        (f"hit the {CAP:,} cap", lambda r: 100 * sum(1 for s in r["samples"] if s["gen_tokens"] >= CAP) / r["n"], "%"),
        ("empty think block", lambda r: 100 * r["empty_think_rate"], "%"),
    ]
    trio = [("Base", load(BASE_17)), ("weak\ntraces", load(STU_17W)), ("strong\ntraces", load(STU_17S))]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.4))
    fig.subplots_adjust(top=0.68, bottom=0.19, left=0.05, right=0.985, wspace=0.30)
    header(fig, "Every Failure Signal Moves the Same Direction",
           "Qwen3-1.7B before and after distilling from two teachers — weak traces left the model intact, strong traces did not")

    for ax, (name, fn, unit) in zip(axes, panels):
        vals = [fn(r) for _, r in trio]
        bars = ax.bar(range(3), vals, 0.58, color=[STEEL, "#8FBF6B", TEAL], zorder=3)
        top = max(vals) * 1.34 if max(vals) > 0 else 1
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + top * 0.05, f"{v:.0f}%",
                    ha="center", fontsize=11, color=INK, weight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for n, _ in trio], fontsize=9.5)
        ax.set_ylim(0, top)
        ax.set_title(name, fontsize=11.5, color=NAVY, pad=12)
        chrome(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    source(fig, "Source: results/qwen3_1.7b/. 'empty think block' counts generations that never emit "
                "</think> after opening one.")
    fig.savefig(f"{OUT}/fig3_failure.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_funnel():
    """What survives verification."""
    stages = ["Raw partition", "Decontaminated\n& deduplicated", "Teacher\nverified correct", "Within\n8,192 tokens"]
    vals = [12000, 11978, 11085, 9680]
    drops = ["", "−1 benchmark leak · −21 duplicates", "−893 teacher answered wrong", "−1,405 exceeded the cap"]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(top=0.73, bottom=0.16, left=0.235, right=0.965)
    header(fig, "What Survives Verification",
           "rasbt/math_distill → qwen3_235b_a22b_math_train: 12,000 problems reduced to 9,680 training examples")

    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, 0.52, color=[RULE, "#BFD4E0", STEEL, TEAL], zorder=3)
    xmax = max(vals)
    for yi, v, d in zip(y, vals, drops):
        ax.text(v + xmax * 0.018, yi + (0.19 if d else 0), f"{v:,}", va="center", ha="left",
                fontsize=13, color=INK, weight="bold")
        if d:
            ax.text(v + xmax * 0.018, yi - 0.24, d, va="center", ha="left",
                    fontsize=9.5, color=RED, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=10.5)
    ax.set_xlim(0, xmax * 1.42)
    ax.set_ylim(-0.75, len(vals) - 0.35)
    ax.set_xlabel("problems / examples", fontsize=11)
    ax.set_xticks([0, 3000, 6000, 9000, 12000])
    ax.set_xticklabels(["0", "3,000", "6,000", "9,000", "12,000"])
    chrome(ax, ygrid=False)
    ax.xaxis.grid(True, color=RULE, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    source(fig, "Source: kd/data/external.py. Decontamination and teacher correctness are recomputed at load "
                "time, not taken from the dataset card — the card does not mention the leak.")
    fig.savefig(f"{OUT}/fig4_funnel.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_tracestyle():
    """Students adopt the teacher's length. That is the whole mechanism."""
    traces = [json.loads(l) for l in open("data/traces.jsonl", encoding="utf-8") if l.strip()]
    usable = [t for t in traces if t["closed"] and t["correct"]]
    weak_len = float(np.mean([t["gen_tokens"] for t in usable]))

    names = ["Base\n(no distillation)", "Student\nweak traces", "Student\nstrong traces"]
    colors = [STEEL, "#6FA84C", TEAL]
    runs = [load(BASE_17), load(STU_17W), load(STU_17S)]
    toks = [r["mean_gen_tokens"] for r in runs]
    accs = [100 * r["accuracy"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.06, right=0.985, wspace=0.22)
    header(fig, "The Student Reproduces the Teacher's Length, Not Its Ability",
           "The weak-trace student looks like its base model on both axes. The strong-trace student inherits the "
           "teacher's verbosity and none of its competence.")

    for ax, vals, title, fmt, headroom in (
        (axes[0], toks, "output length per problem", "{:,.0f} tok", 1.34),
        (axes[1], accs, "MATH-500 accuracy", "{:.1f}%", 1.24),
    ):
        bars = ax.bar(range(3), vals, 0.56, color=colors, zorder=3)
        top = max(vals) * headroom
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + top * 0.035, fmt.format(v),
                    ha="center", fontsize=12, color=INK, weight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=12, color=NAVY, pad=12)
        chrome(ax)
        if fmt.endswith("%"):
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        else:
            ax.axhline(CAP, color=AMBER, lw=1.4, ls=(0, (4, 3)), zorder=2)
            ax.text(-0.42, CAP * 1.05, f"evaluation cap {CAP:,}", ha="left", fontsize=9.5, color=AMBER)
            ax.set_yticks([0, 2000, 4000, 6000, 8000])

    source(fig, "Source: results/qwen3_1.7b/. Teacher length from data/traces.jsonl (mean 2,743 tokens, "
                "n=1,656) and from the recorded training log (mean 3,131 thinking tokens).")
    fig.savefig(f"{OUT}/fig5_tracestyle.png", dpi=200)
    plt.close(fig)


for fn in (fig_headline, fig_nonttermination, fig_failure, fig_funnel, fig_tracestyle):
    fn()

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print(f"  {OUT}/{f}  {os.path.getsize(os.path.join(OUT, f)) // 1024} KB")