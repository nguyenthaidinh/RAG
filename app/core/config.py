from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "CTDT AI Server"
    ENV: str = "prod"

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 3600

    # ── NLP pipeline ──────────────────────────────────────────────
    NLP_TOKENIZER_PROVIDER: str = "local"       # "local" | "openai" | "gemini"
    NLP_CHUNK_MAX_TOKENS: int = 512
    NLP_CHUNK_OVERLAP_TOKENS: int = 50

    # ── Embedding & Vector ────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "local"               # "local" | "openai" | "hf"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_BATCH_SIZE: int = 100
    EMBEDDING_DIM: int = 128                        # dimension for local provider

    VECTOR_INDEX: str = "null"                      # "null" | "pgvector" | "qdrant" | "faiss"

    OPENAI_API_KEY: str = ""

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"

    # ── Query / Retrieval ──────────────────────────────────────────
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

    # ── LLM Answer (RAG synthesis) ────────────────────────────────
    LLM_ANSWER_ENABLED: bool = False
    LLM_ANSWER_PROVIDER: str = "openai"   # openai | none
    LLM_ANSWER_MODEL: str = "gpt-4o-mini"
    LLM_ANSWER_TIMEOUT_S: float = 12.0
    LLM_ANSWER_MAX_TOKENS: int = 600
    LLM_ANSWER_TEMPERATURE: float = 0.2
    LLM_ANSWER_MAX_CONTEXT_CHARS: int = 12000
    LLM_ANSWER_MAX_SNIPPET_CHARS: int = 1200
    LLM_ANSWER_MAX_RESULTS: int = 6

    # ── Document Synthesis ────────────────────────────────────────
    SYNTHESIS_ENABLED: bool = False
    SYNTHESIS_MODEL: str = "gpt-4o-mini"
    SYNTHESIS_TIMEOUT_S: float = 60.0
    SYNTHESIS_MAX_TOKENS: int = 4096
    SYNTHESIS_TEMPERATURE: float = 0.15

    # ── Retrieval Representation ──────────────────────────────────
    RETRIEVAL_REPRESENTATION_MODE: str = "balanced"

    # ── Query Rewrite ─────────────────────────────────────────────
    QUERY_REWRITE_ENABLED: bool = False
    QUERY_REWRITE_PROVIDER: str = "openai"  # "openai" | "none"
    QUERY_REWRITE_MODEL: str = "gpt-4o-mini"
    QUERY_REWRITE_TIMEOUT_S: float = 3.0
    QUERY_REWRITE_MAX_TOKENS: int = 300
    QUERY_REWRITE_TEMPERATURE: float = 0.1
    QUERY_REWRITE_MAX_SUBQUERIES: int = 2
    QUERY_REWRITE_MAX_QUERY_CHARS: int = 1200
    QUERY_REWRITE_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Metadata-Aware Retrieval ───────────────────────────────────
    METADATA_RETRIEVAL_ENABLED: bool = False
    METADATA_RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.6
    METADATA_RETRIEVAL_MAX_TITLE_TERMS: int = 3
    METADATA_RETRIEVAL_MAX_TAGS: int = 3
    METADATA_RETRIEVAL_SOURCE_BIAS_WEIGHT: float = 0.08
    METADATA_RETRIEVAL_REPRESENTATION_BIAS_WEIGHT: float = 0.10
    METADATA_RETRIEVAL_TITLE_BIAS_WEIGHT: float = 0.06
    METADATA_RETRIEVAL_RECENCY_BIAS_WEIGHT: float = 0.05

    # ── Representation Policy ─────────────────────────────────────
    REPRESENTATION_POLICY_ENABLED: bool = False
    REPRESENTATION_POLICY_OVERVIEW_SYNTH_WEIGHT: float = 0.12
    REPRESENTATION_POLICY_EXACT_ORIGINAL_WEIGHT: float = 0.14
    REPRESENTATION_POLICY_CITATION_ORIGINAL_WEIGHT: float = 0.18
    REPRESENTATION_POLICY_MIXED_SYNTH_WEIGHT: float = 0.05
    REPRESENTATION_POLICY_CONFIDENCE_THRESHOLD: float = 0.6

    # ── Observability & Audit ──────────────────────────────────────
    AUDIT_ENABLED: bool = True
    AUDIT_DEFAULT_WINDOW_DAYS: int = 7
    AUDIT_MAX_WINDOW_DAYS: int = 90
    AUDIT_PAGE_LIMIT_MAX: int = 200
    AUDIT_PAGE_LIMIT_DEFAULT: int = 50

    # ── Metrics ──────────────────────────────────────────────────
    METRICS_ENABLED: bool = True

    # ── Health checks ─────────────────────────────────────────────
    HEALTHCHECKS_ENABLED: bool = True
    HEALTH_DB_TIMEOUT_MS: int = 80
    HEALTH_VECTOR_TIMEOUT_MS: int = 80

    # ── Document ingest strategy ──────────────────────────────────
    DOCUMENT_INGEST_MODE: str = "legacy"   # legacy | semantic
    DOCUMENT_INGEST_ALLOW_OVERRIDE: bool = True

    # ── File-Service callback sync ─────────────────────────────────
    FILE_SERVICE_CALLBACK_ENABLED: bool = True
    FILE_SERVICE_INTERNAL_TOKEN: str = ""
    FILE_SERVICE_CALLBACK_TIMEOUT_S: float = 8.0

    # ── Moodle Integration (CTDT) ─────────────────────────────────
    MOODLE_BASE_URL: str = ""
    MOODLE_WSTOKEN: str = ""
    MOODLE_TIMEOUT_S: float = 10.0


settings = Settings()
