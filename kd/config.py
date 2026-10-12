"""Every tunable in one place."""

# --- students -------------------------------------------------------------
STUDENT_06 = "Qwen/Qwen3-0.6B-Base"
STUDENT_17 = "Qwen/Qwen3-1.7B-Base"
TRAIN_STUDENT = STUDENT_17
# A fresh directory. Writing into student17/ would overwrite the only complete
# 1.7B result set in the repo (base + student across all three prompt modes).
TRAIN_OUT = "/root/artifacts/student17_qwen"


RASCHKA_REPO = "rasbt/math_distill"
RASCHKA_SUBSET = "qwen3_235b_a22b_math_train"

RASCHKA_FILE = "data/qwen3-235b-a22b-math-train.json"
RASCHKA_URL = f"https://huggingface.co/datasets/{RASCHKA_REPO}/resolve/main/{RASCHKA_FILE}"

TRACE_SOURCE = "external"
LOCAL_TRACES = "/root/data/traces.jsonl"
TRACE_CACHE = "/root/data/qwen3_235b_a22b_math_train.json"


TEACHER = "Qwen/Qwen3-32B-FP8"
TEACHER_FALLBACK = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TEACHER_GPU = "A100-80GB"
TEACHER_MAX_NEW_TOKENS = 6144
N_PROMPTS = 2000
GEN_CHUNK = 250 
SMOKE_N = 50

MIN_THROUGHPUT_TPS = 400

# --- data -----------------------------------------------------------------
MATH_SOURCE = "HuggingFaceH4/MATH-500"
MATH_TRAIN = "EleutherAI/hendrycks_math"
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
LEVELS = (1, 2, 3, 4, 5)

SEED = 0

# --- evaluation -----------------------------------------------------------
MATH_N = 500

MAX_NEW_TOKENS = 8192
# A100-80GB, not L4. Two independent reasons, and the second matters more:
#   bandwidth - L4 is 300 GB/s against the A100's 2,039 GB/s
#   KV cache  - this is the real lever. L4 left 16.17 GiB of KV, which at a
#               9,216-token context allowed just ~16 concurrent requests out
#               of 500. The 80GB card gives ~5x that concurrency.
# Cost: every base pass measured on L4 is now a cross-hardware delta and has
# to be recomputed alongside the student, or the McNemar test is meaningless.
EVAL_GPU = "A100-80GB"
BASE_GPU = "A10G"

# --- training -------------------------------------------------------------
# 1.7B needs the 80GB card. At 8192 wide the logits tensor alone is
# 151936 vocab x 8192 x 4 bytes ~= 5GB, and AdamW adds ~13.6GB of fp32 state
# on top of 3.4GB of bf16 weights. The 40GB card fitted the 0.6B and has ~22GB
# less headroom than this does, so it OOMs on the first backward pass.
TRAIN_GPU = "A100-80GB"

TRAIN_MAX_LEN = 8192
TRAIN_EPOCHS = 2

TRAIN_LR = 1e-5
TRAIN_MICRO_BATCH = 1
TRAIN_ACCUM = 32
TRAIN_ALLOC_CONF = "expandable_segments:True"
TRAIN_SMOKE_STEPS = 3

PUBLISHED_BASELINE = 0.152