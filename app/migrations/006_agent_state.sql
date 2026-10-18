-- LR2 storage foundation. IANA names and future/DST validation belong to Python;
-- no time-dependent CHECK can replace validation under the commit transaction.
ALTER TABLE labbi_users
    ADD COLUMN timezone TEXT
        CHECK (timezone IS NULL OR (timezone = btrim(timezone) AND length(timezone) BETWEEN 1 AND 100)),
    ADD COLUMN confirmation_epoch BIGINT NOT NULL DEFAULT 0 CHECK (confirmation_epoch >= 0);

-- This registry is independent of labbi_messages and survives /reset.
CREATE TABLE labbi_agent_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    telegram_message_id BIGINT NOT NULL CHECK (telegram_message_id > 0),
    telegram_chat_id BIGINT NOT NULL CHECK (telegram_chat_id = user_id),
    status TEXT NOT NULL DEFAULT 'processing'
        CHECK (status IN ('processing', 'completed', 'cancelled', 'failed')),
    decision TEXT CHECK (decision IN ('respond', 'clarify', 'call_tool', 'read_resource')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    UNIQUE (user_id, telegram_message_id),
    UNIQUE (id, user_id),
    UNIQUE (id, user_id, telegram_message_id),
    CHECK ((status = 'processing') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR (isfinite(finished_at) AND finished_at >= created_at))
);

CREATE TABLE labbi_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    request_id UUID NOT NULL,
    telegram_message_id BIGINT NOT NULL CHECK (telegram_message_id > 0),
    kind TEXT NOT NULL DEFAULT 'add_reminder' CHECK (kind = 'add_reminder'),
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'approved', 'committed', 'cancelled', 'expired', 'failed')),
    confirmation_epoch BIGINT NOT NULL CHECK (confirmation_epoch >= 0),
    text TEXT NOT NULL CHECK (text = btrim(text) AND length(text) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL CHECK (isfinite(remind_at)),
    timezone TEXT NOT NULL CHECK (timezone = btrim(timezone) AND length(timezone) BETWEEN 1 AND 100),
    prepared_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '300 seconds'),
    approved_at TIMESTAMPTZ,
    capability_hash BYTEA CHECK (octet_length(capability_hash) = 32),
    capability_issued_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    receipt_reminder_id UUID,
    UNIQUE (user_id, telegram_message_id),
    UNIQUE (request_id),
    UNIQUE (id, user_id),
    UNIQUE (id, user_id, state),
    UNIQUE (id, user_id, request_id),
    UNIQUE (id, user_id, text, remind_at, timezone),
    FOREIGN KEY (request_id, user_id, telegram_message_id)
        REFERENCES labbi_agent_requests(id, user_id, telegram_message_id) ON DELETE RESTRICT,
    CHECK (expires_at > prepared_at AND expires_at <= prepared_at + INTERVAL '300 seconds'),
    CHECK (approved_at IS NULL OR (approved_at >= prepared_at AND approved_at < expires_at)),
    CHECK ((capability_hash IS NULL) = (capability_issued_at IS NULL)),
    CHECK (capability_issued_at IS NULL OR
        (approved_at IS NOT NULL AND capability_issued_at >= approved_at AND capability_issued_at < expires_at)),
    CHECK (finished_at IS NULL OR (isfinite(finished_at) AND finished_at >= prepared_at)),
    CHECK (
        (state = 'pending' AND approved_at IS NULL AND capability_hash IS NULL
            AND finished_at IS NULL AND receipt_reminder_id IS NULL)
        OR (state = 'approved' AND approved_at IS NOT NULL AND capability_hash IS NOT NULL
            AND finished_at IS NULL AND receipt_reminder_id IS NULL)
        OR (state = 'committed' AND approved_at IS NOT NULL AND capability_hash IS NOT NULL
            AND finished_at IS NOT NULL AND receipt_reminder_id IS NOT NULL)
        OR (state IN ('cancelled', 'expired', 'failed')
            AND finished_at IS NOT NULL AND receipt_reminder_id IS NULL)
    )
);

-- Elapsed time alone never releases an active slot: expiration is an explicit
-- transition under the same user -> action locking order as confirmation.
CREATE UNIQUE INDEX labbi_actions_one_active_per_user_idx
    ON labbi_actions(user_id) WHERE state IN ('pending', 'approved');

CREATE TABLE labbi_reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    action_id UUID NOT NULL UNIQUE,
    committed_state TEXT NOT NULL DEFAULT 'committed' CHECK (committed_state = 'committed'),
    text TEXT NOT NULL CHECK (text = btrim(text) AND length(text) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL CHECK (isfinite(remind_at)),
    timezone TEXT NOT NULL CHECK (timezone = btrim(timezone) AND length(timezone) BETWEEN 1 AND 100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (id, action_id, user_id),
    -- A reminder cannot silently change canonical parameters or the owner.
    FOREIGN KEY (action_id, user_id, text, remind_at, timezone)
        REFERENCES labbi_actions(id, user_id, text, remind_at, timezone) ON DELETE RESTRICT,
    FOREIGN KEY (action_id, user_id, committed_state)
        REFERENCES labbi_actions(id, user_id, state)
        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
);

-- Deferred so one transaction may insert the reminder and mark its action
-- committed in either order. A receipt always names this action's own reminder.
ALTER TABLE labbi_actions ADD CONSTRAINT labbi_actions_receipt_fkey
    FOREIGN KEY (receipt_reminder_id, id, user_id)
    REFERENCES labbi_reminders(id, action_id, user_id)
    ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX labbi_reminders_owner_time_idx ON labbi_reminders(user_id, remind_at, id);

-- Only bounded, sanitized explanations belong here, never raw prompts, tool
-- payloads, Telegram profiles, credentials or capability values/hashes.
CREATE TABLE labbi_agent_events (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    request_id UUID,
    action_id UUID,
    decision TEXT NOT NULL CHECK (decision IN
        ('respond', 'clarify', 'call_tool', 'read_resource', 'confirm', 'cancel')),
    target TEXT CHECK (target IN
        ('get_weather', 'get_schedule', 'add_reminder', 'schedule://current-week')),
    status TEXT NOT NULL CHECK (status IN
        ('OK', 'CLARIFY', 'REJECTED', 'FAILED', 'CANCELLED', 'OUTCOME_UNKNOWN')),
    argument_summary TEXT NOT NULL DEFAULT '' CHECK (length(argument_summary) <= 500),
    reason TEXT NOT NULL CHECK (length(btrim(reason)) BETWEEN 1 AND 500),
    error_code TEXT CHECK (error_code ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    llm_calls SMALLINT NOT NULL DEFAULT 0 CHECK (llm_calls BETWEEN 0 AND 2),
    mcp_calls SMALLINT NOT NULL DEFAULT 0 CHECK (mcp_calls BETWEEN 0 AND 1),
    duration_ms BIGINT NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (request_id, user_id)
        REFERENCES labbi_agent_requests(id, user_id) ON DELETE RESTRICT,
    FOREIGN KEY (action_id, user_id, request_id)
        REFERENCES labbi_actions(id, user_id, request_id) ON DELETE RESTRICT,
    CHECK (action_id IS NULL OR request_id IS NOT NULL)
);

CREATE INDEX labbi_agent_events_owner_history_idx ON labbi_agent_events(user_id, id DESC);
