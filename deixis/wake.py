"""Finding the wake word in a transcript.

Speech-to-text spells an invented word like "Deixis" many ways ("Dixie's",
"Daxis", "Deixes"), so matching is fuzzy and only looks near the start.
"""

import re
from difflib import SequenceMatcher

SEARCH_WORDS = 4
THRESHOLD = 0.72
_FILLERS = {"hey", "ok", "okay", "hi", "yo", "um", "uh", "so"}


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def split_command(transcript: str, wake_words: list[str]) -> str | None:
    """The command after the wake word, "" if only the wake word was said, else None."""
    words = transcript.split()
    normalized = [_norm(w) for w in words]
    for start in range(min(SEARCH_WORDS, len(words))):
        if any(normalized[i] not in _FILLERS for i in range(start)):
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
