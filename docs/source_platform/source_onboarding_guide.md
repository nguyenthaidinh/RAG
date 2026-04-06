# Source Platform — First Source Onboarding Guide

End-to-end guide for onboarding the first knowledge source into AI Server
using the source platform admin API.

## Prerequisites

- AI Server running with DB migrated (`alembic upgrade head`)
- Admin user with JWT or API key
- Upstream source API accessible from AI Server
- Recommended first source: **policy / FAQ / knowledge items** that are tenant-wide readable

## Step 1 — Create the Source

```bash
curl -X POST http://localhost:8000/api/v1/admin/source-platform/sources \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_key": "company-policies",
    "name": "Company Policy Knowledge Base",
    "description": "Internal policies and guidelines",
    "connector_type": "internal-api",
    "base_url": "https://your-api.example.com",
    "auth_type": "bearer",
    "auth_config": {"token": "your-upstream-api-token"},
    "list_path": "/api/internal/knowledge/items",
    "detail_path_template": "/api/internal/knowledge/items/{external_id}",
    "default_metadata": {
      "kind": "policy",
      "status": "published"
    },
    "is_active": true
  }'
```

**Expected**: 201 with source detail (auth_config masked as `****xxxx`).

Save the returned `id` for subsequent steps.

## Step 2 — Verify Source Created

```bash
curl http://localhost:8000/api/v1/admin/source-platform/sources/$SOURCE_ID \
  -H "Authorization: Bearer $TOKEN"
```

**Check**:
- `source_key` matches
- `auth_config.token` shows `****` (masked)
- `is_active` = true

## Step 3 — Trigger Sync

```bash
curl -X POST http://localhost:8000/api/v1/admin/source-platform/sources/$SOURCE_ID/sync \
  -H "Authorization: Bearer $TOKEN"
```

**Expected**: 200 with sync result:
```json
{
  "sync_run_id": 1,
  "source_id": 1,
  "source_key": "company-policies",
  "status": "success",
  "items_fetched": 15,
  "items_upserted": 15,
  "items_failed": 0,
  "message": "Synced 15 items (0 failed)"
}
```

## Step 4 — Check Sync History

```bash
curl http://localhost:8000/api/v1/admin/source-platform/sources/$SOURCE_ID/sync-runs \
  -H "Authorization: Bearer $TOKEN"
```

**Check**:
- `status` = "success"
- `items_fetched` > 0
- `items_failed` = 0
- `finished_at` is set
- `triggered_by` = "api"

## Step 5 — Verify Documents in DB

```bash
# List documents filtered by source
curl "http://localhost:8000/api/v1/admin/documents?source=company-policies" \
  -H "Authorization: Bearer $TOKEN"
```

**Check**:
- Documents exist with `source` = "company-policies"
- Each has `title`, `external_id`, `status` = "ready"
- `metadata` contains `source_platform.source_key`, `source_platform.ingest_mode`

```bash
# View a specific document detail
curl "http://localhost:8000/api/v1/admin/documents/$DOC_ID" \
  -H "Authorization: Bearer $TOKEN"
```

**Check metadata shape**:
```json
{
  "kind": "policy",
  "source_platform": {
    "source_key": "company-policies",
    "source_type": "internal_api",
    "kind": "policy",
    "synced_at": "2026-03-20T01:30:00+00:00",
    "ingest_mode": "source_platform",
    "source_uri": "https://...",
    "access_scope": {},
    "updated_at": "...",
    "checksum": "..."
  }
}
```

## Step 6 — Test Retrieval

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company policy on remote work?",
    "mode": "retrieval"
  }'
```

**Check**:
- Results include snippets from the newly synced source
- `source` field in references shows "company-policies"
- Content is relevant to the query

## Step 7 — Test Answer Quality

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company policy on remote work?"
  }'
```

**Check**:
- Answer references knowledge from the synced source
- Answer is more specific than generic "I don't have information"
- Citations/evidence trace back to source documents

## Step 8 — Re-sync (Idempotency Check)

```bash
# Trigger sync again
curl -X POST http://localhost:8000/api/v1/admin/source-platform/sources/$SOURCE_ID/sync \
  -H "Authorization: Bearer $TOKEN"
```

**Check**:
- If content unchanged: most items should be `skipped` (noop), not re-upserted
- No duplicate documents created
- `last_synced_at` updated on source

## Troubleshooting

### Sync fails with connection error
- Check `base_url` is reachable from AI Server
- Check `auth_config.token` is valid
- Check upstream API returns expected format

### Sync succeeds but 0 items fetched
- Check `list_path` returns data
- Check `default_metadata` filters (e.g. `status=published`) match existing items
- Check upstream API response format: expects `data`, `items`, or `results` envelope

### Documents created but retrieval returns nothing
- Check document `status` = "ready" (not "error" or "pending")
- Check vector index is configured (not NullIndex)
- Check embedding provider is working
- Wait for full pipeline: chunking → embedding → indexing

### Sync succeeds but items_failed > 0
- Check sync run `error_message` for details
- Common: items with empty content are skipped (by design, min 10 chars)
- Common: items without `external_id` / `id` field are skipped
