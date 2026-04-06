"""
Vendor-agnostic embedding provider abstraction.

Protocol + implementations:

* **LocalEmbeddingProvider** — deterministic hash-based vectors for dev/test.
* **OpenAIEmbeddingProvider** — real OpenAI API via ``httpx``.
* **HuggingFaceEmbeddingProvider** — stub (TODO Phase 4).

No provider logic is allowed outside these classes.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Vendor-agnostic embedding interface."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced embeddings."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for *texts*.

        Returns a list of float vectors, one per input text.
        The length of each vector equals ``embedding_dim``.
        """
        ...


# ── local (dev / test) ───────────────────────────────────────────────

class LocalEmbeddingProvider:
    """
    Deterministic dummy embeddings for development and testing.

    Vectors are derived from SHA-256 hashes — same input always
    produces the same embedding.  Zero external dependencies.
    """

    __slots__ = ("_dim",)

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec: list[float] = []
            data = digest
            while len(vec) < self._dim:
                if not data:
                    data = hashlib.sha256(digest + len(vec).to_bytes(4, "big")).digest()
                vec.append(data[0] / 255.0)
                data = data[1:]
            result.append(vec[: self._dim])
        return result


# ── openai ────────────────────────────────────────────────────────────

class OpenAIEmbeddingProvider:
    """
    Real embedding provider using the OpenAI Embeddings API via ``httpx``.

    Configuration (from ENV):
        ``OPENAI_API_KEY``, ``EMBEDDING_MODEL``, ``EMBEDDING_BATCH_SIZE``

    Handles batching safely — splits large input lists into batches of
    ``EMBEDDING_BATCH_SIZE`` and concatenates results.
    """

    _MODEL_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    __slots__ = ("_api_key", "_model", "_batch_size", "_dim")

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model = model or settings.EMBEDDING_MODEL
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        if not self._api_key:
            raise ValueError(
                "OpenAIEmbeddingProvider requires OPENAI_API_KEY to be set"
            )
        self._dim = self._MODEL_DIMS.get(self._model, 1536)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start: batch_start + self._batch_size]

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()

            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            batch_embeddings = [item["embedding"] for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

            logger.info(
                "embedding.batch provider=openai model=%s batch_size=%d "
                "batch_start=%d total_tokens=%s",
                self._model,
                len(batch),
                batch_start,
                data.get("usage", {}).get("total_tokens", "n/a"),
            )

        return all_embeddings


# ── huggingface (stub) ────────────────────────────────────────────────

class HuggingFaceEmbeddingProvider:
    """
    HuggingFace-backed embedding provider.

    TODO (Phase 4): Implement using sentence-transformers or the
    HuggingFace Inference API.
    """

    __slots__ = ("_dim",)

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "TODO (Phase 4): implement HuggingFace embedding provider"
        )


# ── factory ───────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type] = {
    "local": LocalEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
    "hf": HuggingFaceEmbeddingProvider,
}


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """
    Return an ``EmbeddingProvider`` based on config or *provider* override.

    Falls back to ``LocalEmbeddingProvider`` for unknown providers.
    """
    name = (provider or settings.EMBEDDING_PROVIDER).lower().strip()
    cls = _PROVIDERS.get(name)
    if cls is None:
        logger.warning(
            "embedding.factory unknown provider=%s, falling back to local",
            name,
        )
        cls = LocalEmbeddingProvider

    if cls is LocalEmbeddingProvider:
        return cls(dim=settings.EMBEDDING_DIM)
    return cls()  # type: ignore[call-arg]
