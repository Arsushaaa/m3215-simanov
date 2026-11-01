-- Observe HTTP attempts without storing request bodies, tokens or URLs.
ALTER TABLE labbi_outbox
    ADD COLUMN telegram_message_id BIGINT CHECK(telegram_message_id>0),
    ADD COLUMN sent_at TIMESTAMPTZ CHECK(sent_at IS NULL OR isfinite(sent_at)),
    ADD COLUMN uncertain BOOLEAN NOT NULL DEFAULT FALSE,
    ADD CONSTRAINT labbi_outbox_receipt_pair CHECK((telegram_message_id IS NULL)=(sent_at IS NULL)),
    ADD CONSTRAINT labbi_outbox_receipt_sent CHECK(sent_at IS NULL OR state='sent');
CREATE TABLE labbi_delivery_attempts (
    event_id UUID NOT NULL REFERENCES labbi_outbox(id) ON DELETE RESTRICT,
    attempt INTEGER NOT NULL CHECK(attempt BETWEEN 1 AND 3),
    lease_version BIGINT NOT NULL CHECK(lease_version>0),
    lease_token UUID NOT NULL,
    trace_id UUID REFERENCES labbi_traces(id) ON DELETE RESTRICT,
    started_at TIMESTAMPTZ NOT NULL CHECK(isfinite(started_at)),
    finished_at TIMESTAMPTZ CHECK(finished_at IS NULL OR (isfinite(finished_at) AND finished_at>=started_at)),
    outcome TEXT NOT NULL CHECK(outcome IN ('running','sent','retry','failed','unknown')),
    error_code TEXT CHECK(error_code IS NULL OR error_code ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    telegram_message_id BIGINT CHECK(telegram_message_id>0),
    PRIMARY KEY(event_id,attempt),
    UNIQUE(event_id,lease_version),
    CHECK((outcome='running')=(finished_at IS NULL)),
    CHECK((outcome='sent')=(telegram_message_id IS NOT NULL))
);
