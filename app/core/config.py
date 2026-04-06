from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AI Server"
    ENV: str = "prod"

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 3600

    # ── NLP pipeline ──────────────────────────────────────────────
    NLP_TOKENIZER_PROVIDER: str = "local"       # "local" | "openai" | "gemini"
    NLP_CHUNK_MAX_TOKENS: int = 512
    NLP_CHUNK_OVERLAP_TOKENS: int = 50

    # ── Embedding & Vector (Phase 3.2) ────────────────────────────
    EMBEDDING_PROVIDER: str = "local"               # "local" | "openai" | "hf"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100
    EMBEDDING_DIM: int = 128                        # dimension for local provider

    VECTOR_INDEX: str = "null"                      # "null" | "pgvector" | "qdrant" | "faiss"

    OPENAI_API_KEY: str = ""

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    # ── Query / Retrieval (Phase 4.0) ──────────────────────────────
    QUERY_VECTOR_LIMIT: int = 50
    QUERY_BM25_LIMIT: int = 50
    QUERY_FINAL_LIMIT: int = 10
    HYBRID_VECTOR_WEIGHT: float = 0.7
    HYBRID_BM25_WEIGHT: float = 0.3
    HYBRID_THRESHOLD: float = 0.0
    LLM_QUERY_PLANNER_ENABLED: bool = False
    LLM_QUERY_PLANNER_PROVIDER: str = "none"  # "none" | "openai"
    LLM_QUERY_PLANNER_MODEL: str = "gpt-4.1-mini"
    LLM_QUERY_PLANNER_TIMEOUT_S: float = 2.5
    LLM_QUERY_PLANNER_MAX_SUBQUERIES: int = 3
    LLM_QUERY_PLANNER_MAX_QUERY_CHARS: int = 1200
    LLM_QUERY_PLANNER_MAX_TERM_CHARS: int = 120
    LLM_QUERY_PLANNER_CACHE_TTL_S: int = 300

    # ── LLM Answer (RAG synthesis) ────────────────────────────────────
    LLM_ANSWER_ENABLED: bool = False
    LLM_ANSWER_PROVIDER: str = "openai"   # openai | none
    LLM_ANSWER_MODEL: str = "gpt-4o-mini"
    LLM_ANSWER_TIMEOUT_S: float = 12.0
    LLM_ANSWER_MAX_TOKENS: int = 600
    LLM_ANSWER_TEMPERATURE: float = 0.2

    # context guards (avoid huge prompts)
    LLM_ANSWER_MAX_CONTEXT_CHARS: int = 12000
    LLM_ANSWER_MAX_SNIPPET_CHARS: int = 1200
    LLM_ANSWER_MAX_RESULTS: int = 6

    # ── Document Synthesis (Phase 9.0) ────────────────────────────────
    SYNTHESIS_ENABLED: bool = False
    SYNTHESIS_MODEL: str = "gpt-4o-mini"
    SYNTHESIS_TIMEOUT_S: float = 60.0
    SYNTHESIS_MAX_TOKENS: int = 4096
    SYNTHESIS_TEMPERATURE: float = 0.15

    # ── Retrieval Representation (Step 5) ─────────────────────────────
    # "balanced" | "summary_first" | "source_first"
    RETRIEVAL_REPRESENTATION_MODE: str = "balanced"

    # ── Query Rewrite (Phase 3A) ──────────────────────────────────────
    QUERY_REWRITE_ENABLED: bool = False
    QUERY_REWRITE_PROVIDER: str = "openai"  # "openai" | "none"
    QUERY_REWRITE_MODEL: str = "gpt-4o-mini"
    QUERY_REWRITE_TIMEOUT_S: float = 3.0
    QUERY_REWRITE_MAX_TOKENS: int = 300
    QUERY_REWRITE_TEMPERATURE: float = 0.1
    QUERY_REWRITE_MAX_SUBQUERIES: int = 2
    QUERY_REWRITE_MAX_QUERY_CHARS: int = 1200
    QUERY_REWRITE_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Metadata-Aware Retrieval (Phase 3B) ───────────────────────────
    METADATA_RETRIEVAL_ENABLED: bool = False
    METADATA_RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.6
    METADATA_RETRIEVAL_MAX_TITLE_TERMS: int = 3
    METADATA_RETRIEVAL_MAX_TAGS: int = 3
    METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT: float = 0.08
    METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT: float = 0.10
    METADATA_RETRIEVAL_TITLE_BIAS_WEIGHT: float = 0.06
    METADATA_RETRIEVAL_RECENCY_BIAS_WEIGHT: float = 0.05

    # ── Representation Policy (Phase 3D) ──────────────────────────────
    REPRESENTATION_POLICY_ENABLED: bool = False
    REPRESENTATION_POLICY_OVERVIEW_SYNTH_WEIGHT: float = 0.12
    REPRESENTATION_POLICY_EXACT_ORIGINAL_WEIGHT: float = 0.14
    REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT: float = 0.18
    REPRESENTATION_POLICY_MIXED_SYNTH_WEIGHT: float = 0.05
    REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD: float = 0.6

    # ── Query Billing (Phase 4.1) ────────────────────────────────────
    QUERY_BILLING_ENABLED: bool = True
    QUERY_USAGE_TYPE: str = "query"

    # ── Retention (Phase 4.3) ──────────────────────────────────────────
    QUERY_USAGE_RETENTION_DAYS: int = 90

    # ── Quota & Rate Limit (Phase 5.0) ───────────────────────────────
    QUOTA_ENABLED: bool = True
    RATE_LIMIT_ENABLED: bool = True
    PLANS_ENABLED: bool = True
    DEFAULT_PLAN_CODE: str = "free"
    RATE_LIMIT_WINDOW_SEC: int = 60
    RATE_LIMIT_BUCKET_KEY: str = "qpm"
    TOKEN_QUOTA_PRECHECK_ENABLED: bool = False
    TOKEN_QUOTA_CONTEXT_ESTIMATE: int = 300

    # ── Observability & Audit (Phase 6.0) ──────────────────────────────
    AUDIT_ENABLED: bool = True
    AUDIT_DEFAULT_WINDOW_DAYS: int = 7
    AUDIT_MAX_WINDOW_DAYS: int = 90
    AUDIT_PAGE_LIMIT_MAX: int = 200
    AUDIT_PAGE_LIMIT_DEFAULT: int = 50

    # ── Operations & Reliability (Phase 7.0) ─────────────────────────
    METRICS_ENABLED: bool = True
    TRACING_ENABLED: bool = True
    ALERTING_ENABLED: bool = True

    # ── Operational Hardening (Phase 8.0) ──────────────────────────
    HEALTHCHECKS_ENABLED: bool = True
    GRACEFUL_SHUTDOWN_ENABLED: bool = True
    BACKPRESSURE_ENABLED: bool = True
    OPS_DASHBOARD_ENABLED: bool = True

    # Health check timeouts (milliseconds)
    HEALTH_DB_TIMEOUT_MS: int = 80
    HEALTH_VECTOR_TIMEOUT_MS: int = 80

    # Graceful shutdown
    SHUTDOWN_WAIT_SECONDS: int = 30          # max wait for in-flight requests

    # Backpressure / load shedding
    BACKPRESSURE_MAX_CONCURRENT_GLOBAL: int = 100   # max concurrent expensive ops
    BACKPRESSURE_MAX_CONCURRENT_PER_TENANT: int = 10 # per-tenant concurrent limit
    BACKPRESSURE_MAX_QUEUE_SIZE: int = 500           # bounded async queue depth

    # Usage logging dispatcher
    USAGE_LOG_DB_TIMEOUT_SEC: float = 2.0           # max seconds per DB write
    USAGE_LOG_QUEUE_MAXSIZE: int = 2000             # bounded queue depth
    USAGE_LOG_CONCURRENCY: int = 3                  # max concurrent DB writers
    USAGE_LOG_DRAIN_TIMEOUT_SEC: float = 5.0        # shutdown drain budget

    # Tracing
    TRACING_SAMPLE_RATE: float = 0.1          # 10% sampling by default
    TRACING_EXPORTER: str = "none"            # "none" | "otlp" | "console"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""

    # SLO thresholds (Phase 7.2 — consumed by Prometheus alert rules only)
    SLO_QUERY_SUCCESS_RATE: float = 0.99      # 99%
    SLO_QUERY_P95_LATENCY: float = 2.0        # 2 seconds
    SLO_ERROR_RATE: float = 0.01              # 1%

    # ── System Context (Phase 1.1) ────────────────────────────────────
    SYSTEM_CONTEXT_ENABLED: bool = False
    SYSTEM_CONTEXT_PROVIDER: str = "mock"   # "mock" | "core-platform"
    SYSTEM_CONTEXT_DEBUG_ENABLED: bool = False  # debug endpoint
    SYSTEM_CONTEXT_ALLOW_MOCK: bool = True  # set False in production

    # ── System Context: Core-Platform Connector (Phase 2A) ────────────
    SYSTEM_CONTEXT_CORE_BASE_URL: str = ""
    SYSTEM_CONTEXT_CORE_AUTH_TOKEN: str = ""
    SYSTEM_CONTEXT_CORE_TIMEOUT_S: float = 3.0
    SYSTEM_CONTEXT_CORE_CONNECT_TIMEOUT_S: float = 1.0
    SYSTEM_CONTEXT_CORE_READ_TIMEOUT_S: float = 3.0
    SYSTEM_CONTEXT_CORE_MAX_RESPONSE_BYTES: int = 65536

    # ── System Context: Phase 2C Production Readiness ─────────────────
    SYSTEM_CONTEXT_DEBUG_TRACE_ENABLED: bool = False  # log full provenance traces
    SYSTEM_CONTEXT_LOG_ITEM_COUNTS: bool = True       # log connector item counts

    # ── Remote Fetch Hardening (Phase 1.1) ────────────────────────────
    # Comma-separated allowlist of hosts for ingest-reference fetch.
    # Empty string = allow all (legacy behavior, NOT recommended).
    REMOTE_FETCH_ALLOWED_HOSTS: str = ""
    REMOTE_FETCH_ENFORCE_ALLOWLIST: bool = False  # set True in production
    REMOTE_FETCH_MAX_REDIRECTS: int = 3           # max redirect hops

    # ── Document ingest strategy ─────────────────────────────────────
    DOCUMENT_INGEST_MODE: str = "legacy"   # legacy | semantic
    DOCUMENT_INGEST_ALLOW_OVERRIDE: bool = True

    # ── Source Sync Scheduler (Phase 7) ────────────────────────────
    SOURCE_SYNC_SCHEDULER_ENABLED: bool = False
    SOURCE_SYNC_SCHEDULER_TICK_SECONDS: int = 30
    SOURCE_SYNC_MAX_RETRIES: int = 3
    SOURCE_SYNC_RETRY_BASE_SECONDS: int = 30
    SOURCE_SYNC_STALE_RUN_TIMEOUT_MINUTES: int = 60

    # ── File-Service callback sync ─────────────────────────────
    FILE_SERVICE_CALLBACK_ENABLED: bool = True
    FILE_SERVICE_INTERNAL_TOKEN: str = ""
    FILE_SERVICE_CALLBACK_TIMEOUT_S: float = 8.0

settings = Settings()
        