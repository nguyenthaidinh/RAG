"""
Source connector registry (Phase 1 — Foundation).

Central registry for source-platform connectors.  Mirrors the design
of ``system_context.connector_registry`` for consistency.

Usage::

    from app.services.source_platform.source_registry import get_source_registry

    registry = get_source_registry()
    registry.register("internal-api", my_connector, default=True)
    connector = registry.get_required("internal-api")

Thread-safety: module-level singleton.  Registration at startup;
reads are lock-free.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.source_platform.base_source_connector import (
        BaseSourceConnector,
    )

logger = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────


class ConnectorNotRegisteredError(RuntimeError):
    """Raised when a required source connector is not registered."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.provider_name = name
        self.available = available
        super().__init__(
            f"Source connector '{name}' not registered. "
            f"Available: {available or ['(none)']}"
        )


# ── Registry ─────────────────────────────────────────────────────────


class SourceRegistry:
    """
    Maps provider names → source connector instances.

    Methods
    -------
    register(name, connector, default=False)
        Add or replace a connector.
    get(name)
        Resolve by name; returns ``None`` if absent.
    get_required(name)
        Resolve by name; raises ``ConnectorNotRegisteredError`` if absent.
    list_providers()
        Return all registered provider names.
    """

    __slots__ = ("_connectors", "_default_name")

    def __init__(self) -> None:
        self._connectors: dict[str, BaseSourceConnector] = {}
        self._default_name: str | None = None

    def register(
        self,
        name: str,
        connector: BaseSourceConnector,
        *,
        default: bool = False,
    ) -> None:
        """Register a connector under *name*.

        If a connector with the same name already exists it is replaced
        and a warning is logged.
        """
        if name in self._connectors:
            logger.warning(
                "source_platform.registry.overwrite provider=%s", name
            )
        self._connectors[name] = connector
        if default or self._default_name is None:
            self._default_name = name
        logger.info(
            "source_platform.registry.registered provider=%s default=%s",
            name,
            default,
        )

    def get(self, name: str) -> BaseSourceConnector | None:
        """Resolve a connector by name.  Returns ``None`` if absent."""
        return self._connectors.get(name)

    def get_required(self, name: str) -> BaseSourceConnector:
        """Resolve a connector by name.

        Raises ``ConnectorNotRegisteredError`` if the provider is not
        registered.  Use on production paths for fast-fail.
        """
        connector = self._connectors.get(name)
        if connector is None:
            raise ConnectorNotRegisteredError(name, self.list_providers())
        return connector

    def get_default(self) -> BaseSourceConnector | None:
        """Return the default connector, or ``None``."""
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

_registry: SourceRegistry | None = None


def get_source_registry() -> SourceRegistry:
    """Return the global SourceRegistry, lazily initialised."""
    global _registry

    if _registry is None:
        _registry = SourceRegistry()

    return _registry
