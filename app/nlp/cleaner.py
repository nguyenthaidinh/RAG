"""
Deterministic text cleaner for the NLP pipeline.

Operations (in order):
  1. Decode HTML entities  (``&amp;`` → ``&``)
  2. Strip HTML/XML tags   (``<p>text</p>`` → ``text``)
  3. Remove null / control characters (U+0000–U+001F except \\n \\t)
  4. Normalise whitespace   (collapse runs, strip leading/trailing)

The cleaner is a pure function with zero side effects and
zero external dependencies (stdlib only).
"""
from __future__ import annotations

import html
import re

# Compiled once at module level for performance.
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_WHITESPACE_RUNS = re.compile(r"[^\S\n]+")       # horizontal WS runs
_RE_BLANK_LINES = re.compile(r"\n{3,}")              # 3+ newlines → 2


class TextCleaner:
    """Stateless, deterministic text cleaner."""

    __slots__ = ()

    def clean(self, text: str) -> str:
        """
        Return a cleaned copy of *text*.

        Guarantees:
        - Same input always produces the same output.
        - No HTML tags or entities remain.
        - No null / control characters remain.
        - Whitespace is normalised (no leading/trailing, no excessive blanks).
        """
        if not text:
            return ""

        # 1. Decode HTML entities (may need two passes for double-encoded)
        out = html.unescape(html.unescape(text))

        # 2. Strip HTML / XML tags
        out = _RE_HTML_TAGS.sub(" ", out)

        # 3. Remove control characters (keep \n and \t)
        out = _RE_CONTROL_CHARS.sub("", out)

        # 4. Normalise whitespace
        out = _RE_WHITESPACE_RUNS.sub(" ", out)   # collapse horizontal runs
        out = _RE_BLANK_LINES.sub("\n\n", out)     # cap vertical runs at 2
        out = "\n".join(line.strip() for line in out.split("\n"))  # strip each line

        return out.strip()
