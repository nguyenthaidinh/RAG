"""
Internal API source connector (Phase 2).

Reads knowledge/content items from an internal web/app API and maps
them into ``CanonicalKnowledgeItem`` for ingestion through the
existing document pipeline.

This connector is **content-oriented** — it fetches articles, policies,
FAQs, process docs, etc.  It is NOT the same as
``system_context.CorePlatformConnector`` which reads user/tenant/
permission context.

HTTP patterns are modelled after ``CorePlatformConnector``:
  - httpx with explicit timeouts
  - oversized-response protection
  - response envelope normalisation (``data``, ``items``, ``results``)
  - tenant safety checks
  - structured logging

Endpoint configuration
~~~~~~~~~~~~~~~~~~~~~~

The connector accepts lightweight endpoint hints either at construction
time or at call time via ``params``:

* ``list_path``  — API path for listing items
  (default: ``/api/internal/knowledge/items``)
* ``detail_path_template`` — f-string template for item detail
  (default: ``/api/internal/knowledge/items/{external_id}``)

Field mapping
~~~~~~~~~~~~~

``map_to_canonical_item`` is intentionally lenient.  It probes common
field names for body/content (``body``, ``content``, ``body_text``,
``description``, ``text``) and composes the best available body text.

This allows the same connector to serve multiple knowledge domains
(policy, FAQ, article …) without hard-coding a schema.
"""
from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from typing import Any

import httpx

from app.services.source_platform.canonical_item import CanonicalKnowledgeItem

logger = logging.getLogger(__name__)


# ── Internal exceptions (not exported) ───────────────────────────────


class _NoBaseURLError(RuntimeError):
    """Raised when base_url is not configured."""


class _OversizedResponseError(RuntimeError):
    """Raised when response exceeds max_response_bytes."""


# ── Helpers ──────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html_simple(text: str) -> str:
    """Best-effort lightweight HTML tag removal.

    Decodes entities and collapses whitespace.  Not a full HTML parser
    — good enough for stripping simple inline markup from API payloads.
    """
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v) if not isinstance(v, str) else v


def _safe_datetime(v: Any) -> datetime | None:
    """Try to parse an ISO datetime string.  Returns None on failure."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except (ValueError, TypeError):
        return None


# ── Connector ────────────────────────────────────────────────────────


class InternalApiConnector:
    """
    HTTP connector for reading knowledge/content items from an
    internal web/app API.

    Implements ``BaseSourceConnector`` protocol.

    Parameters
    ----------
    base_url : str | None
        Root URL of the internal API.  Resolved from settings
        (``SOURCE_PLATFORM_API_BASE_URL``) if not given.
    api_key : str | None
        Bearer token / API key.  Resolved from settings
        (``SOURCE_PLATFORM_API_AUTH_TOKEN``) if not given.
    timeout_s : float
        Overall request timeout in seconds.
    connect_timeout_s : float
        TCP connect timeout.
    read_timeout_s : float
        Read timeout.
    max_response_bytes : int
        Safety limit for response body size.
    list_path : str
        Default API path for listing items.
    detail_path_template : str
        Default API path template for item detail.
        Must contain ``{external_id}`` placeholder.
    client : httpx.AsyncClient | None
        Optional pre-built client (useful for testing).
    """

    # ── Default endpoint paths ───────────────────────────────────
    _DEFAULT_LIST_PATH = "/api/internal/knowledge/items"
    _DEFAULT_DETAIL_TEMPLATE = "/api/internal/knowledge/items/{external_id}"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        max_response_bytes: int | None = None,
        list_path: str | None = None,
        detail_path_template: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url_override = base_url
        self._api_key_override = api_key
        self._timeout_s_override = timeout_s
        self._connect_timeout_s_override = connect_timeout_s
        self._read_timeout_s_override = read_timeout_s
        self._max_response_bytes_override = max_response_bytes
        self._list_path = list_path or self._DEFAULT_LIST_PATH
        self._detail_path_template = (
            detail_path_template or self._DEFAULT_DETAIL_TEMPLATE
        )
        self._client_override = client

    # ── Lazy config resolution (from settings if no override) ────

    def _cfg(self, key: str, override: Any, default: Any) -> Any:
        if override is not None:
            return override
        from app.core.config import settings

        return getattr(settings, key, default)

    @property
    def _base_url(self) -> str:
        v = self._cfg(
            "SOURCE_PLATFORM_API_BASE_URL", self._base_url_override, ""
        )
        return v.rstrip("/") if v else ""

    @property
    def _api_key(self) -> str:
        return self._cfg(
            "SOURCE_PLATFORM_API_AUTH_TOKEN", self._api_key_override, ""
        )

    @property
    def _timeout_s(self) -> float:
        return float(
            self._cfg(
                "SOURCE_PLATFORM_API_TIMEOUT_S",
                self._timeout_s_override,
                10.0,
            )
        )

    @property
    def _connect_timeout_s(self) -> float:
        return float(
            self._cfg(
                "SOURCE_PLATFORM_API_CONNECT_TIMEOUT_S",
                self._connect_timeout_s_override,
                3.0,
            )
        )

    @property
    def _read_timeout_s(self) -> float:
        return float(
            self._cfg(
                "SOURCE_PLATFORM_API_READ_TIMEOUT_S",
                self._read_timeout_s_override,
                10.0,
            )
        )

    @property
    def _max_response_bytes(self) -> int:
        return int(
            self._cfg(
                "SOURCE_PLATFORM_API_MAX_RESPONSE_BYTES",
                self._max_response_bytes_override,
                2_097_152,  # 2 MB — content items can be larger than context
            )
        )

    # ── Protocol properties ──────────────────────────────────────

    @property
    def connector_name(self) -> str:
        return "internal-api"

    @property
    def source_type(self) -> str:
        return "internal_api"

    # ── Private HTTP helpers ─────────────────────────────────────

    def _build_url(self, path: str) -> str:
        base = self._base_url
        if not base:
            raise _NoBaseURLError(
                "SOURCE_PLATFORM_API_BASE_URL is empty — "
                "cannot call internal API"
            )
        return f"{base}{path}"

    def _headers(self, *, tenant_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token = self._api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning(
                "source_platform.internal_api no auth_token configured — "
                "requests will be unauthenticated"
            )
        if tenant_id:
            headers["X-Tenant-Id"] = str(tenant_id)
        return headers

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self._timeout_s,
            connect=self._connect_timeout_s,
            read=self._read_timeout_s,
        )

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> Any:
        """Execute GET request and return parsed JSON.

        Handles: timeout, connection errors, HTTP error statuses,
        oversized response protection, safe JSON parsing.
        """
        url = self._build_url(path)
        max_bytes = self._max_response_bytes

        client = self._client_override
        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout())

        try:
            resp = await client.get(
                url,
                headers=self._headers(tenant_id=tenant_id),
                params=params,
            )

            # Early content-length check
            cl_raw = resp.headers.get("content-length")
            if cl_raw:
                try:
                    if int(cl_raw) > max_bytes:
                        raise _OversizedResponseError(
                            f"Response too large: {cl_raw} bytes "
                            f"(max {max_bytes})"
                        )
                except ValueError:
                    pass

            resp.raise_for_status()

            body = resp.content
            if len(body) > max_bytes:
                raise _OversizedResponseError(
                    f"Response body too large: {len(body)} bytes "
                    f"(max {max_bytes})"
                )

            try:
                return resp.json()
            except Exception:
                import json as _json

                return _json.loads(body)

        finally:
            if should_close:
                await client.aclose()

    def _extract_payload(
        self,
        data: Any,
        *,
        expect_list: bool = False,
    ) -> Any:
        """Normalize API response envelope.

        Accepts:
          - raw list / dict directly
          - ``{"data": ...}``
          - ``{"items": [...]}``
          - ``{"results": [...]}``
        """
        if isinstance(data, dict):
            for key in ("data", "items", "results"):
                if key in data:
                    val = data[key]
                    if val is None:
                        return [] if expect_list else None
                    if expect_list:
                        return val if isinstance(val, list) else []
                    return val
            return data
        return data

    def _check_tenant(self, payload: Any, local_tenant_id: str) -> bool:
        """Return True if payload tenant_id matches or is absent."""
        if not isinstance(payload, dict):
            return True
        remote_tid = payload.get("tenant_id")
        if remote_tid is None:
            return True
        if str(remote_tid) != str(local_tenant_id):
            logger.warning(
                "source_platform.internal_api.tenant_mismatch "
                "local=%s remote=%s — discarding item",
                local_tenant_id,
                remote_tid,
            )
            return False
        return True

    # ── BaseSourceConnector protocol methods ─────────────────────

    async def test_connection(self) -> bool:
        """Check reachability of the upstream API.

        Tries a lightweight GET to the list endpoint with ``limit=1``.
        Returns True on any successful HTTP response, False on failure.
        """
        try:
            await self._get_json(
                self._list_path,
                params={"limit": "1"},
            )
            logger.info("source_platform.internal_api.test_connection ok")
            return True
        except _NoBaseURLError:
            logger.warning(
                "source_platform.internal_api.test_connection "
                "no_base_url — not configured"
            )
            return False
        except Exception:
            logger.warning(
                "source_platform.internal_api.test_connection failed",
                exc_info=True,
            )
            return False

    async def fetch_item_refs(
        self,
        *,
        tenant_id: str,
        source_key: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch lightweight item references from the list endpoint.

        Supported ``params`` keys:
          - ``list_path``: override the list endpoint path
          - ``page``, ``page_size`` / ``limit``: pagination hints
          - ``kind`` / ``category`` / ``status``: optional filters
          - any other keys are passed through as query params
        """
        p = dict(params or {})

        # Allow per-call endpoint override
        path = p.pop("list_path", None) or self._list_path

        # Ensure tenant_id is passed to the API
        p.setdefault("tenant_id", tenant_id)

        try:
            raw = await self._get_json(path, params=p, tenant_id=tenant_id)
        except _NoBaseURLError:
            logger.error(
                "source_platform.internal_api.fetch_refs "
                "no_base_url source_key=%s",
                source_key,
            )
            raise
        except _OversizedResponseError:
            logger.error(
                "source_platform.internal_api.fetch_refs "
                "oversized_response source_key=%s",
                source_key,
            )
            raise
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "source_platform.internal_api.fetch_refs "
                "http_error=%d source_key=%s",
                exc.response.status_code,
                source_key,
            )
            raise
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "source_platform.internal_api.fetch_refs "
                "timeout/connect_error source_key=%s",
                source_key,
            )
            raise
        except Exception:
            logger.warning(
                "source_platform.internal_api.fetch_refs "
                "unexpected_error source_key=%s",
                source_key,
                exc_info=True,
            )
            raise

        items = self._extract_payload(raw, expect_list=True)
        if not isinstance(items, list):
            items = []

        refs: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            # Resolve external_id from common key names
            ext_id = (
                item.get("external_id")
                or item.get("id")
                or item.get("item_id")
                or item.get("slug")
            )
            if not ext_id:
                continue

            ref: dict[str, Any] = {"external_id": str(ext_id)}

            # Carry forward useful hints
            if item.get("title"):
                ref["title"] = item["title"]
            if item.get("updated_at"):
                ref["updated_at"] = item["updated_at"]
            if item.get("kind"):
                ref["kind"] = item["kind"]
            if item.get("uri") or item.get("url"):
                ref["uri"] = item.get("uri") or item.get("url")

            refs.append(ref)

        logger.info(
            "source_platform.internal_api.fetch_refs "
            "source_key=%s tenant=%s raw_count=%d ref_count=%d",
            source_key,
            tenant_id,
            len(items),
            len(refs),
        )

        return refs

    async def fetch_item_detail(
        self,
        *,
        tenant_id: str,
        source_key: str,
        external_id: str,
        ref: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch full item payload from the detail endpoint.

        Supports ``detail_path_template`` override via ref metadata or
        constructor config.  Returns None on 404 or any retrieval error.
        """
        r = ref or {}

        # Allow per-call template override
        template = r.get("detail_path_template") or self._detail_path_template

        try:
            path = template.format(external_id=external_id)
        except (KeyError, ValueError):
            logger.warning(
                "source_platform.internal_api.fetch_detail "
                "bad_path_template template=%s external_id=%s",
                template,
                external_id,
            )
            return None

        try:
            raw = await self._get_json(
                path,
                params={"tenant_id": tenant_id},
                tenant_id=tenant_id,
            )
        except _NoBaseURLError:
            logger.error(
                "source_platform.internal_api.fetch_detail "
                "no_base_url external_id=%s",
                external_id,
            )
            return None
        except _OversizedResponseError:
            logger.warning(
                "source_platform.internal_api.fetch_detail "
                "oversized_response external_id=%s",
                external_id,
            )
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug(
                    "source_platform.internal_api.fetch_detail "
                    "not_found external_id=%s",
                    external_id,
                )
            else:
                logger.warning(
                    "source_platform.internal_api.fetch_detail "
                    "http_error=%d external_id=%s",
                    exc.response.status_code,
                    external_id,
                )
            return None
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "source_platform.internal_api.fetch_detail "
                "timeout/connect_error external_id=%s",
                external_id,
            )
            return None
        except Exception:
            logger.warning(
                "source_platform.internal_api.fetch_detail "
                "unexpected_error external_id=%s",
                external_id,
                exc_info=True,
            )
            return None

        payload = self._extract_payload(raw)
        if not isinstance(payload, dict):
            return None

        if not self._check_tenant(payload, tenant_id):
            return None

        return payload

    def map_to_canonical_item(
        self,
        *,
        source_key: str,
        raw_detail: dict[str, Any],
    ) -> CanonicalKnowledgeItem | None:
        """Map a raw API detail payload to a ``CanonicalKnowledgeItem``.

        Field resolution is lenient — probes multiple common key names
        so the same connector works for different knowledge domains.

        Returns ``None`` if the item lacks meaningful textual content.
        """
        # ── Required identity ────────────────────────────────────
        external_id = _safe_str(
            raw_detail.get("external_id")
            or raw_detail.get("id")
            or raw_detail.get("item_id")
            or raw_detail.get("slug")
        )
        if not external_id:
            return None

        # ── Title ────────────────────────────────────────────────
        title = _safe_str(
            raw_detail.get("title")
            or raw_detail.get("name")
            or raw_detail.get("heading")
        )

        # ── Body text (compose from best available fields) ───────
        body_text = self._compose_body_text(raw_detail)
        if not body_text or len(body_text.strip()) < 10:
            # No meaningful content → skip
            return None

        # ── Summary ──────────────────────────────────────────────
        summary = _safe_str(
            raw_detail.get("summary")
            or raw_detail.get("excerpt")
            or raw_detail.get("abstract")
        )

        # ── Source URI ───────────────────────────────────────────
        source_uri = _safe_str(
            raw_detail.get("source_uri")
            or raw_detail.get("uri")
            or raw_detail.get("url")
            or raw_detail.get("link")
        )

        # ── Timestamps ───────────────────────────────────────────
        updated_at = _safe_datetime(
            raw_detail.get("updated_at")
            or raw_detail.get("modified_at")
            or raw_detail.get("last_modified")
        )

        # ── Checksum from source (if provided) ───────────────────
        checksum = _safe_str(
            raw_detail.get("checksum")
            or raw_detail.get("content_hash")
            or raw_detail.get("etag")
        )

        # ── Domain metadata (keep useful fields) ─────────────────
        metadata = self._extract_metadata(raw_detail)

        # ── Access scope ─────────────────────────────────────────
        access_scope = self._extract_access_scope(raw_detail)

        return CanonicalKnowledgeItem(
            external_id=external_id,
            source_key=source_key,
            source_type=self.source_type,
            title=title,
            body_text=body_text,
            summary=summary,
            source_uri=source_uri,
            updated_at=updated_at,
            checksum=checksum,
            metadata=metadata,
            access_scope=access_scope,
        )

    # ── Body text composition ────────────────────────────────────

    @staticmethod
    def _compose_body_text(detail: dict[str, Any]) -> str:
        """Build body text from available content fields.

        Priorities:
          1. Explicit ``body_text`` / ``body`` / ``content``
          2. ``description`` / ``text``
          3. Fallback to ``summary`` + ``title`` combination

        Sections (``sections``, ``parts``) are appended if present.
        Simple HTML tags are stripped.
        """
        # Primary content candidates
        primary = None
        for key in ("body_text", "body", "content", "main_content"):
            val = detail.get(key)
            if val and isinstance(val, str) and val.strip():
                primary = val.strip()
                break

        # Secondary fallback
        if not primary:
            for key in ("description", "text", "full_text"):
                val = detail.get(key)
                if val and isinstance(val, str) and val.strip():
                    primary = val.strip()
                    break

        # Last resort: summary or title
        if not primary:
            summary = detail.get("summary") or detail.get("excerpt") or ""
            title = detail.get("title") or ""
            primary = f"{title}\n\n{summary}".strip() if (summary or title) else ""

        if not primary:
            return ""

        parts: list[str] = [_strip_html_simple(primary)]

        # Append sections if available
        sections = detail.get("sections") or detail.get("parts")
        if isinstance(sections, list):
            for sec in sections[:50]:  # sanity cap
                if isinstance(sec, dict):
                    heading = sec.get("heading") or sec.get("title") or ""
                    body = sec.get("body") or sec.get("content") or sec.get("text") or ""
                    if body:
                        sec_text = f"{heading}\n{body}" if heading else str(body)
                        parts.append(_strip_html_simple(sec_text))
                elif isinstance(sec, str) and sec.strip():
                    parts.append(_strip_html_simple(sec))

        return "\n\n".join(parts)

    # ── Metadata extraction ──────────────────────────────────────

    @staticmethod
    def _extract_metadata(detail: dict[str, Any]) -> dict[str, Any]:
        """Pick domain-useful metadata fields from the raw payload."""
        meta: dict[str, Any] = {}
        for key in (
            "kind",
            "category",
            "tags",
            "author",
            "author_id",
            "published_at",
            "created_at",
            "updated_at",
            "status",
            "version",
            "language",
            "department",
            "priority",
            "source_uri",
        ):
            val = detail.get(key)
            if val is not None:
                meta[key] = val

        # Nested metadata dict from the source (if any)
        raw_meta = detail.get("metadata") or detail.get("meta")
        if isinstance(raw_meta, dict):
            for k, v in raw_meta.items():
                if k not in meta and v is not None:
                    meta[k] = v

        return meta

    # ── Access scope extraction ──────────────────────────────────

    @staticmethod
    def _extract_access_scope(detail: dict[str, Any]) -> dict[str, Any]:
        """Extract access control hints from the raw payload."""
        scope: dict[str, Any] = {}

        # Explicit access_scope / permissions block
        raw_scope = detail.get("access_scope") or detail.get("access")
        if isinstance(raw_scope, dict):
            scope.update(raw_scope)

        # Common individual fields
        for key in ("visibility", "tenant_id", "roles", "permission_keys"):
            val = detail.get(key)
            if val is not None and key not in scope:
                scope[key] = val

        return scope
