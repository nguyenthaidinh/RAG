"""
Factory functions for the NLP pipeline.

Provider selection is driven by ``Settings`` (env-var backed):

    NLP_TOKENIZER_PROVIDER  =  "local" | "openai" | "gemini"
    NLP_CHUNK_MAX_TOKENS    =  512
    NLP_CHUNK_OVERLAP_TOKENS =  50

Usage::

    cleaner   = get_cleaner()
    tokenizer = get_tokenizer()
    chunker   = get_chunker()
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.nlp.chunker import SemanticChunker
from app.nlp.cleaner import TextCleaner
from app.nlp.tokenizer import GeminiTokenizer, LocalTokenizer, OpenAITokenizer
from app.nlp.types import Tokenizer

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, type] = {
    "local": LocalTokenizer,
    "openai": OpenAITokenizer,
    "gemini": GeminiTokenizer,
}


def get_cleaner() -> TextCleaner:
    """Return a ``TextCleaner`` instance (stateless, safe to cache)."""
    return TextCleaner()


def get_tokenizer(provider: str | None = None) -> Tokenizer:
    """
    Return a tokenizer for *provider* (defaults to ``NLP_TOKENIZER_PROVIDER``).

    Falls back to ``LocalTokenizer`` if the requested provider is unknown.
    """
    name = (provider or settings.NLP_TOKENIZER_PROVIDER).lower().strip()
    cls = _PROVIDERS.get(name)
    if cls is None:
        logger.warning(
            "nlp.factory unknown provider=%s, falling back to local", name,
        )
        cls = LocalTokenizer
    return cls()  # type: ignore[call-arg]


def get_chunker(
    tokenizer: Tokenizer | None = None,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> SemanticChunker:
    """
    Return a ``SemanticChunker`` wired to *tokenizer* (defaults to
    the env-configured one).
    """
    tok = tokenizer or get_tokenizer()
    mt = max_tokens if max_tokens is not None else settings.NLP_CHUNK_MAX_TOKENS
    ot = overlap_tokens if overlap_tokens is not None else settings.NLP_CHUNK_OVERLAP_TOKENS
    return SemanticChunker(tok, max_tokens=mt, overlap_tokens=ot)
