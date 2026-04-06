"""
NLP Core Engine — vendor-agnostic text processing pipeline.

Public API::

    from app.nlp import get_cleaner, get_tokenizer, get_chunker, Chunk

    cleaner = get_cleaner()
    chunker = get_chunker()

    cleaned  = cleaner.clean(raw_html)
    chunks   = chunker.chunk(cleaned, tenant_id=tid, document_id=did, version_id=vid)
"""

from app.nlp.factory import get_chunker, get_cleaner, get_tokenizer
from app.nlp.types import Chunk, Tokenizer

__all__ = [
    "Chunk",
    "Tokenizer",
    "get_chunker",
    "get_cleaner",
    "get_tokenizer",
]
