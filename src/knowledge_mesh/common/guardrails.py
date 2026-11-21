"""Signals for untrusted content that tries to steer the model (prompt-injection heuristics)."""

from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts|messages)",
    r"disregard (all |any )?(previous|prior|above)",
    r"you are now (a|an|the) ",
    r"(reveal|print|show) (your|the) (system|hidden) prompt",
    r"system prompt",
    r"do not (tell|inform) the user",
    r"</?(system|assistant|instructions?)>",
]
_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def suspicious_instruction(text: str) -> bool:
    return bool(text) and bool(_RE.search(text))
