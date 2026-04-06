from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileServiceSyncClient:
    async def sync_document_metadata(
        self,
        *,
        file_id: int,
        temporary_url: str,
        metadata_json: dict[str, Any],
        ai_document_id: int | None,
        semantic_status: str = "ready",
    ) -> None:
        """
        Sync semantic JSON metadata back to file-service.

        We derive the callback base URL from the same temporary_url domain:
        Example:
            temporary_url = https://file-service.domain/api/v1/files/12/temporary-download?...
            callback      = https://file-service.domain/api/v1/internal/files/12/document-meta
        """
        if not getattr(settings, "FILE_SERVICE_CALLBACK_ENABLED", False):
            return

        token = getattr(settings, "FILE_SERVICE_INTERNAL_TOKEN", "")
        if not token:
            logger.warning("file_service_sync.skipped missing internal token")
            return

        base_url = self._extract_base_url(temporary_url)
        callback_url = f"{base_url}/api/v1/internal/files/{file_id}/document-meta"

        timeout = float(getattr(settings, "FILE_SERVICE_CALLBACK_TIMEOUT_S", 8.0))

        payload = {
            "metadata_json": metadata_json,
            "ai_document_id": ai_document_id,
            "semantic_status": semantic_status,
        }

        headers = {
            "X-Internal-Token": token,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(callback_url, json=payload, headers=headers)
            resp.raise_for_status()

    @staticmethod
    def _extract_base_url(temporary_url: str) -> str:
        parts = temporary_url.split("/")
        if len(parts) < 3:
            raise ValueError("Invalid temporary_url")
        return f"{parts[0]}//{parts[2]}"