"""
Tokenizer providers for the NLP pipeline.

Three concrete implementations of the ``Tokenizer`` protocol:

* **LocalTokenizer**  – 1 word = 1 token. Zero dependencies, safe fallback.
* **OpenAITokenizer** – Uses ``tiktoken`` when installed for exact BPE counts;
  falls back to a word-ratio approximation otherwise.
* **GeminiTokenizer** – Word-ratio approximation tuned for SentencePiece
  models (~1.3 tokens per English word).

All three honour the contract: for every segment returned by ``split``,
``count(segment) <= max_tokens`` holds.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────

def _split_words_windowed(
    words: Sequence[str],
    max_words: int,
    overlap_words: int,
) -> list[str]:
    """Sliding-window split over a word list."""
    if not words:
        return []
    step = max(1, max_words - overlap_words)
    segments: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        segments.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return segments


# ── LocalTokenizer ────────────────────────────────────────────────────

class LocalTokenizer:
    """
    1 whitespace-delimited word = 1 token.

    No external dependencies.  Suitable as an always-available fallback
    when exact sub-word tokenisation is not required.
    """

    __slots__ = ()

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def split(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        words = text.split()
        return _split_words_windowed(words, max_tokens, overlap_tokens)


# ── OpenAITokenizer ──────────────────────────────────────────────────

class OpenAITokenizer:
    """
    Adapter for OpenAI BPE tokenisation.

    * If ``tiktoken`` is installed → exact token counts via *cl100k_base*
      (GPT-4 / GPT-3.5-turbo default encoding).
    * Otherwise → word-ratio approximation (~1.3 tokens per word).

    No API calls are ever made.
    """

    # Approximation: avg English word ≈ 1.3 BPE tokens (cl100k_base).
    _TOKENS_PER_WORD: float = 1.3

    def __init__(self, model: str = "gpt-4") -> None:
        self._encoding = None
        try:
            import tiktoken  # type: ignore[import-untyped]

            self._encoding = tiktoken.encoding_for_model(model)
            logger.info("nlp.tokenizer provider=openai mode=exact model=%s", model)
        except ImportError:
            logger.info(
                "nlp.tokenizer provider=openai mode=approx "
                "(tiktoken not installed, using word-ratio fallback)"
            )
        except KeyError:
            logger.warning(
                "nlp.tokenizer provider=openai mode=approx "
                "model=%s unknown to tiktoken, using fallback",
                model,
            )

    @property
    def exact(self) -> bool:
        return self._encoding is not None

    # ── count ─────────────────────────────────────────────────────

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        if self._encoding:
            return len(self._encoding.encode(text))
        return max(1, round(len(text.split()) * self._TOKENS_PER_WORD))

    # ── split ─────────────────────────────────────────────────────

    def split(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        if self._encoding:
            return self._split_exact(text, max_tokens, overlap_tokens)
        return self._split_approx(text, max_tokens, overlap_tokens)

    def _split_exact(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        enc = self._encoding
        assert enc is not None
        tokens = enc.encode(text)
        if not tokens:
            return []
        step = max(1, max_tokens - overlap_tokens)
        segments: list[str] = []
        for start in range(0, len(tokens), step):
            end = min(start + max_tokens, len(tokens))
            segments.append(enc.decode(tokens[start:end]))
            if end >= len(tokens):
                break
        return segments

    def _split_approx(
        self, text: str, max_tokens: int, overlap_tokens: int
    ) -> list[str]:
        words = text.split()
        # Floor guarantees count(segment) <= max_tokens for ratio-based counting.
        max_words = max(1, int(max_tokens / self._TOKENS_PER_WORD))
        overlap_words = max(0, int(overlap_tokens / self._TOKENS_PER_WORD))
        return _split_words_windowed(words, max_words, overlap_words)


# ── GeminiTokenizer ──────────────────────────────────────────────────

class GeminiTokenizer:
    """
    Word-ratio approximation for Google Gemini (SentencePiece).

    ~1.3 tokens per English word on average.
    No external dependencies.
    """

    _TOKENS_PER_WORD: float = 1.3

    __slots__ = ()

    def count(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return max(1, round(len(text.split()) * self._TOKENS_PER_WORD))

    def split(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
    ) -> list[str]:
        words = text.split()
        # Floor guarantees count(segment) <= max_tokens for ratio-based counting.
        max_words = max(1, int(max_tokens / self._TOKENS_PER_WORD))
        overlap_words = max(0, int(overlap_tokens / self._TOKENS_PER_WORD))
        return _split_words_windowed(words, max_words, overlap_words)
