

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    source VARCHAR(50) NOT NULL,
    external_id VARCHAR(512) NOT NULL,
    title VARCHAR(512),
    content_raw TEXT NOT NULL,
    content_text TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    checksum VARCHAR(64) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Enforce idempotency
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_tenant_source_external
    ON documents(tenant_id, source, external_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_tenant
    ON documents(tenant_id);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_status
    ON documents(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status);

-- Status lifecycle constraint
ALTER TABLE documents
ADD CONSTRAINT IF NOT EXISTS ck_documents_status
CHECK (status IN ('pending', 'processing', 'ready', 'error'));

-- updated_at trigger (idempotent)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;

CREATE TRIGGER update_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
