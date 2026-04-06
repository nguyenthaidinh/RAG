"""
First source config examples (Phase 3 — Multi-Source Onboarding).

This file shows how to define ``OnboardedSourceConfig`` instances for
real sources.  Each config describes a single web/API source.

**These are examples, not hard-coded bootstraps.**  In production, configs
may come from env vars, a config file, or eventually a DB table.

To onboard a new web, copy one of these examples, adjust the fields, and
call ``sync_onboarded_source(db, config=my_config, tenant_id="...")``.

Example sync::

    from app.services.source_platform.source_onboarding_service import (
        sync_onboarded_source,
    )
    from app.services.source_platform.first_source_examples import (
        make_first_source_config,
    )

    config = make_first_source_config(
        base_url="https://api.myapp.com",
        auth_token="my-secret-token",
    )
    result = await sync_onboarded_source(db, config=config, tenant_id="t1")
"""
from __future__ import annotations

from app.services.source_platform.onboarded_source_config import (
    OnboardedSourceConfig,
)


def make_first_source_config(
    *,
    base_url: str,
    auth_token: str = "",
    source_key: str = "first-source-policies",
    content_kind: str = "policy",
    list_path: str = "/api/internal/knowledge/items",
    detail_path_template: str = "/api/internal/knowledge/items/{external_id}",
    default_params: dict | None = None,
    enabled: bool = True,
) -> OnboardedSourceConfig:
    """Create an ``OnboardedSourceConfig`` for the first onboarded source.

    This factory makes it easy to spin up the first source with sensible
    defaults while keeping everything explicit and overridable.

    The first source is a **policy / knowledge-article** domain — the
    most natural fit for the Source Connector Platform because it is
    content-oriented (articles, regulations, FAQs, procedures).

    Parameters
    ----------
    base_url : str
        Root URL of the source API.
    auth_token : str
        Bearer token.  Pass empty string for unauthenticated.
    source_key : str
        Logical source identifier.  Default is generic.
    content_kind : str
        Domain hint used as ``kind`` filter on the list endpoint.
    list_path : str
        Override list endpoint path.
    detail_path_template : str
        Override detail endpoint template.
    default_params : dict | None
        Extra default query params for every list request.
    enabled : bool
        Whether this source should be synced.

    Returns
    -------
    OnboardedSourceConfig
    """
    return OnboardedSourceConfig(
        source_key=source_key,
        connector_name="internal-api",
        base_url=base_url,
        auth_token=auth_token,
        list_path=list_path,
        detail_path_template=detail_path_template,
        content_kind=content_kind,
        default_params=default_params or {"status": "published"},
        enabled=enabled,
    )


# ── Quick reference: how would web #2 look? ─────────────────────────
#
# SECOND_SOURCE = OnboardedSourceConfig(
#     source_key="another-app-faqs",
#     connector_name="internal-api",
#     base_url="https://api.another-app.com",
#     auth_token="another-token",
#     list_path="/api/v2/faqs",
#     detail_path_template="/api/v2/faqs/{external_id}",
#     content_kind="faq",
#     default_params={"language": "vi", "status": "active"},
#     enabled=True,
# )
#
# To sync it:
#   result = await sync_onboarded_source(db, config=SECOND_SOURCE, tenant_id="t2")
#
# No new service, no new module — just a new config instance.
