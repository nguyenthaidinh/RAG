"""
Phase 4.0 — Retrieval & Query Engine.

Provides vector similarity search, BM25 lexical search, hybrid fusion,
deterministic re-ranking, and strict tenant-level access control.

Architecture::

    QueryService
     ├── AccessPolicy
     ├── QueryPlanner
     │     ├── BM25Retriever
     │     ├── VectorRetriever
     │     └── HybridStrategy
     ├── ResultMerger
     ├── ReRanker
     └── ResponseBuilder

All components are Protocol-based, dependency-injected, and vendor-agnostic.
"""
# self._metadata_first = MetadataFirstRetrievalService()