from __future__ import annotations

import asyncio
import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAIPlannerProvider:
    __slots__ = ("_api_key", "_model")

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._model = model or settings.LLM_QUERY_PLANNER_MODEL

    async def plan(self, query_text: str) -> dict:
        if not self._api_key:
            return {}

        async def _call() -> dict:
            import httpx

            system_prompt = (
                "You are a query-planning component for a search engine. "
                "Output ONLY valid JSON. No prose."
            )
            user_prompt = (
                "query_text:\n"
                f"{query_text}\n\n"
                "Required schema:\n"
                '- normalized_query: string\n'
                '- subqueries: array of strings (max 3)\n'
                "- filters: { doc_ids: array of positive integers } (usually empty)\n"
                '- preferred_mode: "vector" | "bm25" | "hybrid" | null\n'
                "Rules:\n"
                "- do NOT include document content\n"
                "- do NOT include secrets\n"
                "- do NOT output anything except JSON\n"
                "Example JSON:\n"
                '{"normalized_query":"annual budget review","subqueries":["annual budget","budget review"],'
                '"filters":{"doc_ids":[]},"preferred_mode":"hybrid"}'
            )

            payload = {
                "model": self._model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            timeout_s = float(getattr(settings, "LLM_QUERY_PLANNER_TIMEOUT_S", 2.5))
            async with httpx.AsyncClient(timeout=timeout_s + 0.5) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            try:
                content = data["choices"][0]["message"]["content"]
            except Exception:
                return {}

            if not isinstance(content, str):
                return {}

            try:
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        try:
            timeout_s = float(getattr(settings, "LLM_QUERY_PLANNER_TIMEOUT_S", 2.5))
            return await asyncio.wait_for(_call(), timeout=timeout_s)
        except Exception:
            logger.warning("retrieval.planner_provider_failed provider=openai")
            return {}
