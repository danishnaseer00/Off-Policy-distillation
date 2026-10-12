"""Answer extraction + equivalence for MATH-style outputs.

The one thing that must not be wrong here: a scoring bug looks exactly like a
failed experiment. Self-checked by `selfcheck()` before any GPU is billed.
"""

from __future__ import annotations

import re
from functools import lru_cache

_FN = r"sqrt|sin|cos|tan|log|ln|exp|abs|min|max|mod"
_FN_BRACED = r"\\(" + _FN + r")\{([^{}]*)\}"
_FN_BARE = r"\\(" + _FN + r")\s*([a-zA-Z0-9.])"
_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FRAC = re.compile(r"\\[dtc]?frac")

_FALLBACK = (
    re.compile(r"(?:final answer|answer)\s*(?:is|=|:)?\s*([^\n]+)", re.I),
    re.compile(r"####\s*(.+)"),
)


def _group(s: str, i: int) -> tuple[str, int] | None:
    """Brace group whose opening brace is at s[i]. Returns (body, index after)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    return None


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, brace-counted so nesting survives."""
    found = None
    for m in _BOXED.finditer(text or ""):
        g = _group(text, m.end() - 1)
        if g:
            found = g[0]
    return found


def last_boxed_or_fallback(text: str) -> str | None:
    """Boxed first; else a common answer marker. Fallback precision is lower,
    so `harness` reports how often it fires."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    for pattern in _FALLBACK:
        hits = pattern.findall(text or "")
        if hits:
            return hits[-1].strip()
    return None


def _fractions(s: str) -> str:
    r"""\\frac{a}{b} -> (a)/(b), innermost first, so nesting unwinds correctly.

    A regex with [^{}]* cannot do this: \frac{\sqrt{2}}{2} and
    \frac{2^{1/2}}{2} both occur in real MATH answers.
    """
    for _ in range(6):
        hit = None
        for m in _FRAC.finditer(s):
            i = m.end()
            while i < len(s) and s[i] == " ":
                i += 1
            num = _group(s, i) if i < len(s) and s[i] == "{" else None
            if not num:
                continue
            j = num[1]
            while j < len(s) and s[j] == " ":
                j += 1
            den = _group(s, j) if j < len(s) and s[j] == "{" else None
            if den:
                hit = (m.start(), den[1], num[0], den[0])
                break
        if not hit:
            return s
        start, after, num, den = hit
        s = s[:start] + f"({num})/({den})" + s[after:]
    return s


_ALIASES = {
    "\\cdot": "*", "\\times": "*", "\\div": "/",
    "\\pi": "pi", "\\infty": "oo", "\\infty ": "oo",
}


def normalize(raw: str) -> str:
    """Reduce a LaTeX answer to a comparable form. Only removes what is never
    semantically meaningful; never rewrites arithmetic."""
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"\\[dtc]?frac", r"\\frac", s)

    for w in ("text", "mbox", "textbf", "textit", "mathrm", "mathbf", "mathit"):
        s = s.replace(f"\\{w}{{", "{").replace(f"\\{w} (", "(")

    s = re.sub(r"\\(?:left|right|displaystyle|!|,|;|:)", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\text\{\s*\}", "", s)
    s = s.replace("\\%", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+", "", s).rstrip(".")

    # LaTeX function calls -> sympy-parseable calls, braced or bare.
    s = re.sub(_FN_BRACED, r"\1(\2)", s)
    s = re.sub(_FN_BARE, r"\1(\2)", s)
    for cmd, name in _ALIASES.items():
        s = s.replace(cmd, name)

    s = re.sub(r"(?<=[0-9)])(sqrt|sin|cos|tan|log|ln|exp|pi)\b", r"*\1", s)
    s = re.sub(r"(?<=\d)\(", "*(", s)

    s = _fractions(s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)

    s = s.replace("{,}", ",")
    # Thousands separators only. A comma is a separator when it sits between
    # digits and is followed by exactly three digits before a non-digit, which
    # keeps tuples like (3,\frac{\pi}{2}) intact. The previous r"(\d),(\d{3})"
    # consumed just the first group of "1,234,567" and left "1234,567".
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)

    if s.startswith("{") and s.endswith("}") and _group(s, 0):
        s = s[1:-1]
    return s


@lru_cache(maxsize=8192)
def _as_float(s: str) -> float | None:
    """Numeric value of a normalised answer, or None. Covers the large
    majority of MATH answers: ints, decimals, ratios, signed values."""
    if not s or any(c in s for c in "=<>\\^_"):
        return None
    neg = s.count("-") % 2 == 1
    t = s.lstrip("-+").strip()
    try:
        if "/" in t:
            parts = t.split("/")
            if len(parts) != 2:
                return None
            num, den = float(parts[0]), float(parts[1])
            if den == 0:
                return None
            val = num / den
        else:
            val = float(t.replace(",", ""))
    except ValueError:
        return None
    return -val if neg else val


def _as_sympy(s: str):
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if not s or len(s) > 200 or any(c in s for c in "=<>"):
            return None
        return sympy.nsimplify(
            parse_expr(
                s,
                transformations=standard_transformations
                + (convert_xor, implicit_multiplication_application),
            ),
            rational=False,
        )
    except Exception:
        return None


def match_tier(predicted: str, reference: str) -> int | None:
    """1 = string, 2 = numeric, 3 = symbolic, None = no match.

    Tier 3 is best-effort: it returns None on parse failure rather than
    raising, so a symbolic regression shows as a slightly lower score, not a
    crash. Reporting the tier tells you when that fragile path is load-bearing.
    """
    if not predicted or not reference:
        return None
    p, r = normalize(predicted), normalize(reference)
    if not p or not r:
        return None
    if p == r:
        return 1
    pf, rf = _as_float(p), _as_float(r)
    if pf is not None and rf is not None:
        scale = max(1.0, abs(pf), abs(rf))
        return 2 if abs(pf - rf) <= 1e-6 * scale else None
    ps, rs = _as_sympy(p), _as_sympy(r)
    if ps is None or rs is None:
        return None
    try:
        import sympy

        return 3 if bool(sympy.simplify(ps - rs) == 0) else None
    except Exception:
        return None


def answers_match(predicted: str, reference: str) -> bool:
    return match_tier(predicted, reference) is not None


_BOXED_CASES = (
    ("The result is \\boxed{42}.", "42"),
    ("\\boxed{\\boxed{7}}", "7"),
    ("first \\boxed{1} then \\boxed{2}", "2"),
    ("no box here", None),
)

_FALLBACK_CASES = (
    ("#### 17", "17"),
    ("The final answer is 42", "42"),
    ("\\boxed{5} and also final answer is 9", "5"),
)

_MATCH_CASES = (
    ("42", "42", 1),
    ("$42$", "42", 1),
    ("\\dfrac{1}{2}", "\\frac{1}{2}", 1),
    ("1/2", "\\frac{1}{2}", 3),
    ("0.5", "\\frac{1}{2}", 3),
    ("1{,}234", "1234", 1),
    ("50\\%", "50", 1),
    ("2\\sqrt{2}", "2\\sqrt2", 1),
    ("\\frac{\\sqrt{2}}{2}", "\\frac{2^{1/2}}{2}", 3),
    ("x^2+2x+1", "(x+1)^2", 3),
    ("42", "43", None),
    ("x^2", "x^3", None),
    ("", "42", None),
)


def selfcheck() -> None:
    """Cheap parser check. Raises on drift; no framework, no GPU."""
    for raw, want in _BOXED_CASES:
        if (got := extract_boxed(raw)) != want:
            raise AssertionError(f"extract_boxed({raw!r}) -> {got!r}, want {want!r}")
    for raw, want in _FALLBACK_CASES:
        if (got := last_boxed_or_fallback(raw)) != want:
            raise AssertionError(f"answer({raw!r}) -> {got!r}, want {want!r}")
    for pred, ref, want in _MATCH_CASES:
        if (got := match_tier(pred, ref)) != want:
            raise AssertionError(f"match_tier({pred!r}, {ref!r}) -> {got}, want {want}")