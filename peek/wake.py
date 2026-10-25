"""Finding the wake word in a transcript.

Speech-to-text spells the wake word several ways ("Peek's", "Peak"), so
matching is fuzzy. The wake word must start a clause: the transcript itself,
or the text after punctuation. In a noisy room the transcript often starts
with other people's speech ("so loud in here. Peek, ..."), so every clause is
checked and the last match wins.
"""

import re
from difflib import SequenceMatcher

SEARCH_WORDS = 4
THRESHOLD = 0.72
_FILLERS = {"hey", "ok", "okay", "hi", "yo", "um", "uh", "so"}
_CLAUSE_END = re.compile(r"[.?!,;:]$")


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _match_at(
    words: list[str], normalized: list[str], clause: int, wake_words: list[str]
) -> str | None:
    for start in range(clause, min(clause + SEARCH_WORDS, len(words))):
        if any(normalized[i] not in _FILLERS for i in range(clause, start)):
            break
        for wake in wake_words:
            parts = [_norm(p) for p in wake.split()]
            span = normalized[start : start + len(parts)]
            if len(span) != len(parts):
                continue
            score = SequenceMatcher(None, "".join(span), "".join(parts)).ratio()
            if score >= THRESHOLD:
                rest = " ".join(words[start + len(parts) :])
                return rest.lstrip(" ,.!?;:-").strip()
    return None


def split_command(transcript: str, wake_words: list[str]) -> str | None:
    """The command after the wake word, "" if only the wake word was said, else None."""
    words = transcript.split()
    normalized = [_norm(w) for w in words]
    clauses = [0] + [i + 1 for i, w in enumerate(words[:-1]) if _CLAUSE_END.search(w)]
    for clause in reversed(clauses):
        command = _match_at(words, normalized, clause, wake_words)
        if command is not None:
            return command
    return None
