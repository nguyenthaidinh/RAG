"""
Core-platform system connector (Phase 2A).

Implements real HTTP calls to the core-platform internal context API.
Replaces the Phase 1.1 placeholder while keeping backward-compatible
construction — ConnectorRegistry calls ``CorePlatformConnector()``
with no arguments, so ``__init__`` must work with zero args.

API contract:
  GET /api/internal/context/user
  GET /api/internal/context/tenant
  GET /api/internal/context/permissions
  GET /api/internal/context/stats
  GET /api/internal/context/records
  GET /api/internal/context/workflows

Fail semantics:
  - user / tenant / stats: fail-open → return None
  - records / workflows: fail-open → return []
  - permissions: FAIL-CLOSED → return empty PermissionSnapshot (deny-all)

Tenant safety:
  - local tenant_id is the source of truth
  - if remote payload tenant_id diverges, discard and log warning
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.schemas.system_context import (
    MetricValue,
    PermissionDecision,
    PermissionSnapshot,
    RecordSummary,
    SystemContextBundle,
    TenantContext,
    TenantStats,
    UserContext,
    WorkflowSummary,
)

logger = logging.getLogger(__name__)


class CorePlatformConnector:
    """
    HTTP connector for core-platform internal context API.

    Constructor accepts optional overrides for testing.  When called
    with no arguments (as ConnectorRegistry does), all values are
    resolved lazily from ``app.core.config.settings``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        auth_token: str | None = None,
        timeout_s: float | None = None,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        max_response_bytes: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url_override = base_url
        self._auth_token_override = auth_token
        self._timeout_s_override = timeout_s
        self._connect_timeout_s_override = connect_timeout_s
        self._read_timeout_s_override = read_timeout_s
        self._max_response_bytes_override = max_response_bytes
        self._client_override = client

    # ── Config resolution (lazy, from settings if no override) ────────

    def _cfg(self, key: str, override: Any, default: Any) -> Any:
        if override is not None:
            return override
        from app.core.config import settings
        return getattr(settings, key, default)

    @property
    def _base_url(self) -> str:
        v = self._cfg("SYSTEM_CONTEXT_CORE_BASE_URL", self._base_url_override, "")
        return v.rstrip("/") if v else ""

    @property
    def _auth_token(self) -> str:
        return self._cfg("SYSTEM_CONTEXT_CORE_AUTH_TOKEN", self._auth_token_override, "")

    @property
    def _timeout_s(self) -> float:
        return float(self._cfg("SYSTEM_CONTEXT_CORE_TIMEOUT_S", self._timeout_s_override, 3.0))

    @property
    def _connect_timeout_s(self) -> float:
        return float(self._cfg("SYSTEM_CONTEXT_CORE_CONNECT_TIMEOUT_S", self._connect_timeout_s_override, 1.0))

    @property
    def _read_timeout_s(self) -> float:
        return float(self._cfg("SYSTEM_CONTEXT_CORE_READ_TIMEOUT_S", self._read_timeout_s_override, 3.0))

    @property
    def _max_response_bytes(self) -> int:
        return int(self._cfg("SYSTEM_CONTEXT_CORE_MAX_RESPONSE_BYTES", self._max_response_bytes_override, 65536))

    # ── Protocol property ─────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "core-platform"

    # ── Private helpers ───────────────────────────────────────────────

    def _build_url(self, path: str) -> str:
        base = self._base_url
        if not base:
            raise _NoBaseURLError("SYSTEM_CONTEXT_CORE_BASE_URL is empty")
        return f"{base}{path}"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        token = self._auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning(
                "system_context.core_platform no auth_token configured — "
                "requests will be unauthenticated"
            )
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
    ) -> Any:
        """Execute GET request and return parsed JSON.

        Handles:
          - timeout / connection errors
          - HTTP error statuses
          - oversized response protection (content-length + body)
          - safe JSON parsing (tolerant of wrong content-type)
          - safe content-length parsing (non-int values)
        """
        url = self._build_url(path)
        max_bytes = self._max_response_bytes

        client = self._client_override
        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout())

        try:
            resp = await client.get(url, headers=self._headers(), params=params)

            # Check content-length header early if available (safe parse)
            content_length_raw = resp.headers.get("content-length")
            if content_length_raw:
                try:
                    cl_int = int(content_length_raw)
                    if cl_int > max_bytes:
                        raise _OversizedResponseError(
                            f"Response too large: {cl_int} bytes "
                            f"(max {max_bytes})"
                        )
                except ValueError:
                    # Non-integer content-length — ignore, check body later
                    pass

            resp.raise_for_status()

            # Read body and enforce size limit
            body = resp.content
            if len(body) > max_bytes:
                raise _OversizedResponseError(
                    f"Response body too large: {len(body)} bytes "
                    f"(max {max_bytes})"
                )

            # Parse JSON — tolerate wrong content-type if body is valid JSON
            try:
                return resp.json()
            except Exception:
                import json
                return json.loads(body)

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
          - raw object/list directly
          - {"data": ...}
          - {"items": [...]}
          - {"results": [...]}

        For list endpoints (expect_list=True), prefers data/items/results
        if present and is a list.

        Phase 2C hardening:
          - data=None on object endpoint → returns None
          - data/items/results not a list on list endpoint → returns []
        """
        if isinstance(data, dict):
            for key in ("data", "items", "results"):
                if key in data:
                    val = data[key]
                    if val is None:
                        # explicit null — no data
                        return [] if expect_list else None
                    if expect_list:
                        if isinstance(val, list):
                            return val
                        # shape mismatch: expected list but got non-list → []
                        return []
                    return val
            # None of the envelope keys found — return dict itself
            return data

        return data

    def _check_tenant(self, payload: Any, local_tenant_id: str) -> bool:
        """Return True if payload tenant_id matches or is absent.

        Log warning and return False if mismatch.
        """
        if not isinstance(payload, dict):
            return True

        remote_tid = payload.get("tenant_id")
        if remote_tid is None:
            return True

        if str(remote_tid) != str(local_tenant_id):
            logger.warning(
                "system_context.core_platform.tenant_mismatch "
                "local=%s remote=%s — discarding payload",
                local_tenant_id, remote_tid,
            )
            return False

        return True

    # ── Public connector methods ──────────────────────────────────────

    async def get_user_context(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
    ) -> UserContext | None:
        """GET /api/internal/context/user — fail-open."""
        try:
            raw = await self._get_json(
                "/api/internal/context/user",
                params={"tenant_id": tenant_id, "user_id": str(actor_user_id)},
            )
            payload = self._extract_payload(raw)
            if not isinstance(payload, dict):
                return None

            if not self._check_tenant(payload, tenant_id):
                return None

            # Phase 2C: safe type coercion for roles/scopes/attributes
            roles_raw = payload.get("roles", [])
            scopes_raw = payload.get("scopes", [])
            attrs_raw = payload.get("attributes", {})

            return UserContext(
                user_id=payload.get("user_id", actor_user_id),
                tenant_id=tenant_id,
                email=payload.get("email"),
                display_name=payload.get("display_name"),
                role=_safe_str(payload.get("role")),
                roles=roles_raw if isinstance(roles_raw, list) else [],
                scopes=scopes_raw if isinstance(scopes_raw, list) else [],
                attributes=attrs_raw if isinstance(attrs_raw, dict) else {},
                is_active=payload.get("is_active", True),
            )

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_user_context failed "
                "tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_user_context "
                    "endpoint_not_found tenant_id=%s — capability not deployed",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_user_context "
                    "http_error=%d tenant_id=%s",
                    exc.response.status_code, tenant_id,
                )
            return None
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_user_context "
                "timeout/connect_error tenant_id=%s",
                tenant_id,
            )
            return None
        except Exception:
            logger.warning(
                "system_context.core_platform.get_user_context "
                "unexpected_error tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None

    async def get_tenant_context(
        self,
        *,
        tenant_id: str,
    ) -> TenantContext | None:
        """GET /api/internal/context/tenant — fail-open."""
        try:
            raw = await self._get_json(
                "/api/internal/context/tenant",
                params={"tenant_id": tenant_id},
            )
            payload = self._extract_payload(raw)
            if not isinstance(payload, dict):
                return None

            if not self._check_tenant(payload, tenant_id):
                return None

            # Phase 2C: safe type coercion for attributes
            attrs_raw = payload.get("attributes", {})

            return TenantContext(
                tenant_id=tenant_id,
                tenant_name=payload.get("tenant_name"),
                tenant_slug=payload.get("tenant_slug"),
                attributes=attrs_raw if isinstance(attrs_raw, dict) else {},
            )

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_tenant_context "
                "failed tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_tenant_context "
                    "endpoint_not_found tenant_id=%s",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_tenant_context "
                    "http_error=%d tenant_id=%s",
                    exc.response.status_code, tenant_id,
                )
            return None
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_tenant_context "
                "timeout/connect_error tenant_id=%s",
                tenant_id,
            )
            return None
        except Exception:
            logger.warning(
                "system_context.core_platform.get_tenant_context "
                "unexpected_error tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None

    async def get_permission_snapshot(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        resource_types: list[str] | None = None,
    ) -> PermissionSnapshot:
        """GET /api/internal/context/permissions — FAIL-CLOSED.

        On any error, returns an empty snapshot (no decisions = deny-all).
        Never returns None for actual errors.
        """
        empty = PermissionSnapshot(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
        )

        try:
            params: dict[str, Any] = {
                "tenant_id": tenant_id,
                "user_id": str(actor_user_id),
            }
            if resource_types:
                params["resource_types"] = ",".join(resource_types)

            raw = await self._get_json(
                "/api/internal/context/permissions",
                params=params,
            )
            payload = self._extract_payload(raw)
            if not isinstance(payload, dict):
                return empty

            if not self._check_tenant(payload, tenant_id):
                return empty

            decisions_raw = payload.get("decisions", [])
            if not isinstance(decisions_raw, list):
                decisions_raw = []

            decisions: list[PermissionDecision] = []
            for d in decisions_raw:
                if not isinstance(d, dict):
                    continue
                try:
                    # Phase 2C: safe type coercion for string fields + field_masking
                    fm_raw = d.get("field_masking", {})
                    decisions.append(PermissionDecision(
                        resource_type=_safe_str(d.get("resource_type", "unknown")),
                        action=_safe_str(d.get("action", "unknown")),
                        allowed=bool(d.get("allowed", False)),
                        scope=_safe_str(d.get("scope")),
                        reason=d.get("reason"),
                        field_masking=fm_raw if isinstance(fm_raw, dict) else {},
                    ))
                except Exception:
                    logger.debug(
                        "system_context.core_platform.permission_decision_skip",
                        exc_info=True,
                    )

            return PermissionSnapshot(
                tenant_id=tenant_id,
                actor_user_id=actor_user_id,
                decisions=decisions,
            )

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_permission_snapshot "
                "failed tenant_id=%s behavior=FAIL_CLOSED",
                tenant_id, exc_info=True,
            )
            return empty
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_permission_snapshot "
                    "endpoint_not_found tenant_id=%s — returning empty snapshot",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_permission_snapshot "
                    "http_error=%d tenant_id=%s behavior=FAIL_CLOSED",
                    exc.response.status_code, tenant_id,
                )
            return empty
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_permission_snapshot "
                "timeout/connect_error tenant_id=%s behavior=FAIL_CLOSED",
                tenant_id,
            )
            return empty
        except Exception:
            logger.warning(
                "system_context.core_platform.get_permission_snapshot "
                "unexpected_error tenant_id=%s behavior=FAIL_CLOSED",
                tenant_id, exc_info=True,
            )
            return empty

    async def get_tenant_stats(
        self,
        *,
        tenant_id: str,
        period: str | None = None,
    ) -> TenantStats | None:
        """GET /api/internal/context/stats — fail-open."""
        try:
            params: dict[str, Any] = {"tenant_id": tenant_id}
            if period:
                params["period"] = period

            raw = await self._get_json(
                "/api/internal/context/stats",
                params=params,
            )
            payload = self._extract_payload(raw)
            if not isinstance(payload, dict):
                return None

            if not self._check_tenant(payload, tenant_id):
                return None

            metrics_raw = payload.get("metrics", [])
            if not isinstance(metrics_raw, list):
                metrics_raw = []

            metrics: list[MetricValue] = []
            for m in metrics_raw:
                if not isinstance(m, dict):
                    continue
                try:
                    metrics.append(MetricValue(
                        key=m.get("key", "unknown"),
                        value=m.get("value", 0),
                        label=m.get("label"),
                        unit=m.get("unit"),
                    ))
                except Exception:
                    logger.debug(
                        "system_context.core_platform.metric_skip",
                        exc_info=True,
                    )

            return TenantStats(
                tenant_id=tenant_id,
                metrics=metrics,
                period=payload.get("period", period),
            )

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_tenant_stats "
                "failed tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_tenant_stats "
                    "endpoint_not_found tenant_id=%s",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_tenant_stats "
                    "http_error=%d tenant_id=%s",
                    exc.response.status_code, tenant_id,
                )
            return None
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_tenant_stats "
                "timeout/connect_error tenant_id=%s",
                tenant_id,
            )
            return None
        except Exception:
            logger.warning(
                "system_context.core_platform.get_tenant_stats "
                "unexpected_error tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return None

    async def get_record_summaries(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int | None = None,
        record_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[RecordSummary]:
        """GET /api/internal/context/records — fail-open."""
        try:
            params: dict[str, Any] = {
                "tenant_id": tenant_id,
                "limit": str(limit),
            }
            if actor_user_id is not None:
                params["user_id"] = str(actor_user_id)
            if record_types:
                params["record_types"] = ",".join(record_types)

            raw = await self._get_json(
                "/api/internal/context/records",
                params=params,
            )
            items = self._extract_payload(raw, expect_list=True)
            if not isinstance(items, list):
                items = []

            records: list[RecordSummary] = []
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Tenant filter: keep only matching or absent tenant_id
                item_tid = item.get("tenant_id")
                if item_tid is not None and str(item_tid) != str(tenant_id):
                    logger.debug(
                        "system_context.core_platform.record_skip_tenant "
                        "local=%s remote=%s",
                        tenant_id, item_tid,
                    )
                    continue

                try:
                    # Phase 2C: safe type coercion
                    meta_raw = item.get("metadata", {})
                    records.append(RecordSummary(
                        record_type=_safe_str(item.get("record_type", "unknown")),
                        record_id=str(item.get("record_id", "")),
                        title=item.get("title"),
                        status=item.get("status"),
                        owner_id=item.get("owner_id"),
                        tenant_id=tenant_id,
                        summary=item.get("summary"),
                        metadata=meta_raw if isinstance(meta_raw, dict) else {},
                    ))
                except Exception:
                    logger.debug(
                        "system_context.core_platform.record_parse_skip",
                        exc_info=True,
                    )

            return records

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_record_summaries "
                "failed tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_record_summaries "
                    "endpoint_not_found tenant_id=%s",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_record_summaries "
                    "http_error=%d tenant_id=%s",
                    exc.response.status_code, tenant_id,
                )
            return []
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_record_summaries "
                "timeout/connect_error tenant_id=%s",
                tenant_id,
            )
            return []
        except Exception:
            logger.warning(
                "system_context.core_platform.get_record_summaries "
                "unexpected_error tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return []

    async def get_workflow_summaries(
        self,
        *,
        tenant_id: str,
        workflow_types: list[str] | None = None,
    ) -> list[WorkflowSummary]:
        """GET /api/internal/context/workflows — fail-open."""
        try:
            params: dict[str, Any] = {"tenant_id": tenant_id}
            if workflow_types:
                params["workflow_types"] = ",".join(workflow_types)

            raw = await self._get_json(
                "/api/internal/context/workflows",
                params=params,
            )
            items = self._extract_payload(raw, expect_list=True)
            if not isinstance(items, list):
                items = []

            workflows: list[WorkflowSummary] = []
            for item in items:
                if not isinstance(item, dict):
                    continue

                item_tid = item.get("tenant_id")
                if item_tid is not None and str(item_tid) != str(tenant_id):
                    logger.debug(
                        "system_context.core_platform.workflow_skip_tenant "
                        "local=%s remote=%s",
                        tenant_id, item_tid,
                    )
                    continue

                try:
                    # Phase 2C: safe type coercion
                    bs_raw = item.get("by_status", {})
                    workflows.append(WorkflowSummary(
                        workflow_type=_safe_str(item.get("workflow_type", "unknown")),
                        tenant_id=tenant_id,
                        total=item.get("total"),
                        by_status=bs_raw if isinstance(bs_raw, dict) else {},
                        pending_count=item.get("pending_count"),
                        completed_count=item.get("completed_count"),
                    ))
                except Exception:
                    logger.debug(
                        "system_context.core_platform.workflow_parse_skip",
                        exc_info=True,
                    )

            return workflows

        except (_NoBaseURLError, _OversizedResponseError):
            logger.warning(
                "system_context.core_platform.get_workflow_summaries "
                "failed tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info(
                    "system_context.core_platform.get_workflow_summaries "
                    "endpoint_not_found tenant_id=%s",
                    tenant_id,
                )
            else:
                logger.warning(
                    "system_context.core_platform.get_workflow_summaries "
                    "http_error=%d tenant_id=%s",
                    exc.response.status_code, tenant_id,
                )
            return []
        except (httpx.TimeoutException, httpx.ConnectError):
            logger.warning(
                "system_context.core_platform.get_workflow_summaries "
                "timeout/connect_error tenant_id=%s",
                tenant_id,
            )
            return []
        except Exception:
            logger.warning(
                "system_context.core_platform.get_workflow_summaries "
                "unexpected_error tenant_id=%s",
                tenant_id, exc_info=True,
            )
            return []

    async def build_context_bundle(
        self,
        *,
        tenant_id: str,
        actor_user_id: str | int,
        include_user: bool = True,
        include_tenant: bool = True,
        include_permissions: bool = False,
        include_stats: bool = False,
        include_records: bool = False,
        include_workflows: bool = False,
    ) -> SystemContextBundle:
        """Assemble a full bundle by calling individual methods.

        Delegates to SystemContextBuilder in production usage,
        but this method provides a direct implementation matching
        the BaseSystemConnector protocol.
        """
        user = None
        tenant = None
        permissions = None
        stats = None
        records: list[RecordSummary] = []
        workflows: list[WorkflowSummary] = []

        if include_user:
            user = await self.get_user_context(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
        if include_tenant:
            tenant = await self.get_tenant_context(tenant_id=tenant_id)
        if include_permissions:
            permissions = await self.get_permission_snapshot(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
        if include_stats:
            stats = await self.get_tenant_stats(tenant_id=tenant_id)
        if include_records:
            records = await self.get_record_summaries(
                tenant_id=tenant_id, actor_user_id=actor_user_id,
            )
        if include_workflows:
            workflows = await self.get_workflow_summaries(tenant_id=tenant_id)

        return SystemContextBundle(
            user=user,
            tenant=tenant,
            permissions=permissions,
            tenant_stats=stats,
            records=records,
            workflows=workflows,
            source="core-platform",
        )


# ── Helpers (not exported) ───────────────────────────────────────────


def _safe_str(v: Any) -> str | None:
    """Safely coerce a value to str or None.

    Handles cases where API returns int/float/bool where string is expected.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return str(v)


# ── Internal exceptions (not exported) ───────────────────────────────


class _NoBaseURLError(RuntimeError):
    """Raised when SYSTEM_CONTEXT_CORE_BASE_URL is not configured."""


class _OversizedResponseError(RuntimeError):
    """Raised when response exceeds max_response_bytes."""
