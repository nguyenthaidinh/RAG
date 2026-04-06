"""
Connector registry (Phase 1.1 — Hardened).

Central registry for system context connectors.  Allows the application
to resolve a connector by provider name at runtime.

Phase 1.1 hardening:
  - Mock connector is only auto-registered when SYSTEM_CONTEXT_ALLOW_MOCK=True
  - get_required(name) raises ConnectorNotFoundError for fast-fail
  - Default resolution respects SYSTEM_CONTEXT_PROVIDER config
  - Production path never silently falls back to mock

Usage::

    from app.services.system_context.connector_registry import get_connector_registry

    registry = get_connector_registry()
    connector = registry.get_required("core-platform")

Thread-safety: the registry is a module-level singleton.
Registration happens at startup; reads are lock-free.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.system_context.base_connector import BaseSystemConnector

logger = logging.getLogger(__name__)


class ConnectorNotFoundError(RuntimeError):
    """Raised when a required connector is not registered."""

    def __init__(self, name: str, available: list[str]):
        self.provider_name = name
        self.available = available
        super().__init__(
            f"System context connector '{name}' not found. "
            f"Available: {available or ['(none)']}"
        )


class ConnectorRegistry:
    """
    Maps provider names → connector instances.

    Methods:
      register(name, connector)  — add/replace a connector
      get(name)          → connector | None
      get_required(name) → connector  (raises ConnectorNotFoundError)
      get_default()      → connector | None
      list_providers()   → list[str]
    """

    __slots__ = ("_connectors", "_default_name")

    def __init__(self) -> None:
        self._connectors: dict[str, BaseSystemConnector] = {}
        self._default_name: str | None = None

    def register(
        self,
        name: str,
        connector: BaseSystemConnector,
        *,
        default: bool = False,
    ) -> None:
        """Register a connector under *name*.  Overwrites silently."""
        self._connectors[name] = connector
        if default or self._default_name is None:
            self._default_name = name
        logger.info(
            "system_context.registry.registered provider=%s default=%s",
            name, default,
        )

    def get(self, name: str) -> BaseSystemConnector | None:
        """Resolve a connector by provider name.  Returns None if absent."""
        return self._connectors.get(name)

    def get_required(self, name: str) -> BaseSystemConnector:
        """
        Resolve a connector by provider name.

        Raises ConnectorNotFoundError if the provider is not registered.
        Use this on production paths to fail fast on misconfiguration.
        """
        connector = self._connectors.get(name)
        if connector is None:
            raise ConnectorNotFoundError(name, self.list_providers())
        return connector

    def get_default(self) -> BaseSystemConnector | None:
        """Return the default connector (first registered or explicit default)."""
        if self._default_name is None:
            return None
        return self._connectors.get(self._default_name)

    @property
    def default_name(self) -> str | None:
        return self._default_name

    def list_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._connectors.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._connectors


# ── Module-level singleton ───────────────────────────────────────────

_registry: ConnectorRegistry | None = None


def get_connector_registry() -> ConnectorRegistry:
    """Return the global ConnectorRegistry, lazily initialized."""
    global _registry

    if _registry is None:
        _registry = ConnectorRegistry()
        _bootstrap_default_connectors(_registry)

    return _registry


def _bootstrap_default_connectors(registry: ConnectorRegistry) -> None:
    """Register built-in connectors respecting config flags.

    Phase 1.1 hardening:
      - Mock is registered ONLY when SYSTEM_CONTEXT_ALLOW_MOCK is True
        (defaults to True for dev safety, set False in production).
      - Default provider is set from SYSTEM_CONTEXT_PROVIDER config.
      - Warns loudly if configured provider couldn't be registered.
    """
    from app.core.config import settings

    allow_mock: bool = getattr(settings, "SYSTEM_CONTEXT_ALLOW_MOCK", True)
    configured_provider: str = getattr(settings, "SYSTEM_CONTEXT_PROVIDER", "mock")

    # ── Register mock (gated) ─────────────────────────────────────
    if allow_mock:
        from app.services.system_context.mock_connector import MockSystemConnector

        is_default = configured_provider == "mock"
        registry.register("mock", MockSystemConnector(), default=is_default)
    else:
        logger.info(
            "system_context.registry mock_connector skipped "
            "SYSTEM_CONTEXT_ALLOW_MOCK=False"
        )

    # ── Register core-platform placeholder ────────────────────────
    try:
        from app.services.system_context.core_platform_connector import (
            CorePlatformConnector,
        )
        is_default = configured_provider == "core-platform"
        registry.register("core-platform", CorePlatformConnector(), default=is_default)
    except Exception:
        logger.debug(
            "system_context.registry core-platform connector not available",
            exc_info=True,
        )

    # ── Validate configured provider was registered ───────────────
    if configured_provider not in registry:
        logger.error(
            "system_context.registry MISCONFIGURATION: "
            "SYSTEM_CONTEXT_PROVIDER='%s' but that provider was not registered. "
            "Available: %s. System context features will not work correctly.",
            configured_provider, registry.list_providers(),
        )
