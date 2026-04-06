"""
Alerting & SLO definitions (Phase 7.0).

This module:
  - Defines SLO targets as config-driven thresholds
  - Provides Prometheus alert rules as structured data
  - Application code ONLY exposes metrics — it does NOT hardcode
    alert thresholds.  Alert rules are consumed by Prometheus/Alertmanager.

Alert rules below are expressed as Python dicts that can be serialized
to YAML for a ``prometheus_rules.yml`` file.
"""
from __future__ import annotations

from app.core.config import settings


# ── SLO definitions (read from config) ────────────────────────────────

def get_slo_definitions() -> dict:
    """
    Return current SLO thresholds.

    All values come from config / environment so operators can tune
    without code changes.
    """
    return {
        "query_success_rate": settings.SLO_QUERY_SUCCESS_RATE,
        "query_p95_latency_seconds": settings.SLO_QUERY_P95_LATENCY,
        "error_rate": settings.SLO_ERROR_RATE,
    }


# ── Prometheus alert rules (config-driven) ────────────────────────────
#
# These are meant to be exported as a ``groups:`` YAML block for
# ``prometheus_rules.yml`` or loaded into Alertmanager.
#
# Thresholds reference the SLO_* config keys so operators can change
# them via environment variables without touching code.
#

def get_alert_rules() -> list[dict]:
    """
    Return Prometheus alert rules as structured dicts.

    Each rule references the config-driven SLO thresholds.
    Operators should render these into ``prometheus_rules.yml``.
    """
    if not settings.ALERTING_ENABLED:
        return []

    slo = get_slo_definitions()

    return [
        # 🚨 5xx error spike
        {
            "alert": "HighErrorRate",
            "expr": (
                "sum(rate(http_requests_total{status=~\"5..\"}[5m]))"
                " / "
                "sum(rate(http_requests_total[5m]))"
                f" > {slo['error_rate']}"
            ),
            "for": "2m",
            "labels": {"severity": "critical"},
            "annotations": {
                "summary": "5xx error rate exceeds SLO",
                "description": (
                    f"Error rate is above {slo['error_rate']*100:.1f}% "
                    "for the last 2 minutes."
                ),
            },
        },
        # 🚨 p95 latency above SLO
        {
            "alert": "HighQueryLatency",
            "expr": (
                "histogram_quantile(0.95, "
                "sum(rate(ai_query_latency_seconds_bucket[5m])) by (le)"
                f") > {slo['query_p95_latency_seconds']}"
            ),
            "for": "5m",
            "labels": {"severity": "warning"},
            "annotations": {
                "summary": "p95 query latency exceeds SLO",
                "description": (
                    f"p95 query latency above {slo['query_p95_latency_seconds']}s "
                    "for 5 minutes."
                ),
            },
        },
        # 🚨 rate-limit hits spike
        {
            "alert": "RateLimitSpike",
            "expr": (
                "sum(rate(rate_limit_hits_total[5m])) by (tenant_id) > 10"
            ),
            "for": "2m",
            "labels": {"severity": "warning"},
            "annotations": {
                "summary": "Rate-limit hits spiking for tenant",
                "description": (
                    "Tenant {{ $labels.tenant_id }} is hitting rate limits "
                    "at >10 req/s for the last 2 minutes."
                ),
            },
        },
        # 🚨 token quota exceeded spike
        {
            "alert": "TokenQuotaExceededSpike",
            "expr": (
                "sum(rate(token_quota_exceeded_total[5m])) by (tenant_id) > 5"
            ),
            "for": "2m",
            "labels": {"severity": "warning"},
            "annotations": {
                "summary": "Token quota exceeded events spiking",
                "description": (
                    "Tenant {{ $labels.tenant_id }} is exceeding token quota "
                    "at >5 events/s for 2 minutes."
                ),
            },
        },
        # 🚨 Query success rate below SLO
        {
            "alert": "LowQuerySuccessRate",
            "expr": (
                "sum(rate(ai_queries_total{status=\"success\"}[5m]))"
                " / "
                "sum(rate(ai_queries_total[5m]))"
                f" < {slo['query_success_rate']}"
            ),
            "for": "5m",
            "labels": {"severity": "critical"},
            "annotations": {
                "summary": "Query success rate below SLO",
                "description": (
                    f"Success rate is below {slo['query_success_rate']*100:.1f}% "
                    "for the last 5 minutes."
                ),
            },
        },
    ]


def render_alert_rules_yaml() -> str:
    """
    Render alert rules as a Prometheus-compatible YAML string.

    This is a convenience helper — operators can also use
    ``get_alert_rules()`` directly.
    """
    import yaml  # noqa: delay import — only used by admin tooling

    rules = get_alert_rules()
    if not rules:
        return "# Alerting disabled\n"

    doc = {
        "groups": [
            {
                "name": "ai_server_slo",
                "rules": rules,
            },
        ],
    }
    return yaml.dump(doc, default_flow_style=False)
