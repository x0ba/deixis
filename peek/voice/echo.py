"""Removing Peek's own voice from a transcript.

The mic stays live while Peek talks so the user can interrupt, which means
the speakers leak into the transcript. Peek knows what it said, so runs of
transcript words that line up, in order, with that text are its own echo.
Speech recognition mangles some echoed words, so short unmatched gaps
between echoed runs count as echo too. Everything else is the user.
"""

import re
from difflib import SequenceMatcher

WORD_MATCH = 0.75
MAX_GAP = 2

BACKCHANNEL = {
    "um", "uh", "mm", "hmm", "mhm", "mmhmm", "uhhuh", "ok", "okay", "yeah", "yep", "right",
    "sure", "oh", "ah", "huh", "cool", "nice", "alright",
}  # fmt: skip


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    return min(len(a), len(b)) > 2 and SequenceMatcher(None, a, b).ratio() >= WORD_MATCH


def _alignment(t: list[str], r: list[str]) -> list[tuple[int, int]]:
    """Longest in-order fuzzy match of t against r, preferring the earliest t words."""
    n, m = len(t), len(r)
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = max(lcs[i + 1][j], lcs[i][j + 1])
            if t[i] and _same(t[i], r[j]):
                best = max(best, lcs[i + 1][j + 1] + 1)
            lcs[i][j] = best
    pairs, i, j = [], 0, 0
    while i < n and j < m and lcs[i][j]:
        if t[i] and _same(t[i], r[j]) and lcs[i][j] == lcs[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif lcs[i][j + 1] == lcs[i][j]:
            j += 1
        else:
            i += 1
    return pairs


def _swapped_next_to(pair: tuple[int, int], run: list[tuple[int, int]]) -> bool:
    (t, r), (end_t, end_r), (start_t, start_r) = pair, run[-1], run[0]
    after = 0 <= t - end_t - 1 == r - end_r - 1 <= MAX_GAP
    before = 0 <= start_t - t - 1 == start_r - r - 1 <= MAX_GAP
    return after or before


def strip_echo(transcript: str, spoken: list[str]) -> str:
    """The part of `transcript` that isn't Peek repeating `spoken` back."""
    words = transcript.split()
    reference = [w for w in (_norm(x) for x in " ".join(spoken).split()) if w]
    if not words or not reference:
        return transcript.strip()
    pairs = _alignment([_norm(w) for w in words], reference)

    # Group matches into runs that are consecutive in both texts. A lone
    # matching word is usually the user reusing a common word, unless the
    # reference itself is that short ("Done.").
    runs: list[list[tuple[int, int]]] = []
    for pair in pairs:
        if runs and pair[0] == runs[-1][-1][0] + 1 and pair[1] == runs[-1][-1][1] + 1:
            runs[-1].append(pair)
        else:
            runs.append([pair])
    strong = [run for run in runs if len(run) >= min(2, len(reference))]
    if not strong:
        return transcript.strip()
    # A lone match next to an echoed run, with the same number of words
    # skipped on both sides, is a garbled word swapped in place: still echo.
    pending = [run for run in runs if run not in strong]
    grew = True
    while grew:
        grew = False
        for run in list(pending):
            if any(_swapped_next_to(run[0], other) for other in strong):
                strong.append(run)
                pending.remove(run)
                grew = True
    runs = sorted(strong)

    echo = [False] * len(words)
    for k, run in enumerate(runs):
        for i, _ in run:
            echo[i] = True
        if k + 1 < len(runs):
            gap_start, gap_end = run[-1][0] + 1, runs[k + 1][0][0]
            if gap_end - gap_start <= MAX_GAP:
                for i in range(gap_start, gap_end):
                    echo[i] = True
    kept = " ".join(w for w, e in zip(words, echo, strict=True) if not e)
    return kept.lstrip(" ,.;:-").strip()


def is_backchannel(text: str) -> bool:
    """Only filler like "okay" or "mm-hmm": listening noises, not a request."""
    words = [_norm(w) for w in text.split()]
    words = [w for w in words if w]
    return not words or all(w in BACKCHANNEL for w in words)


def content_words(text: str) -> int:
    return sum(1 for w in (_norm(x) for x in text.split()) if w and w not in BACKCHANNEL)
