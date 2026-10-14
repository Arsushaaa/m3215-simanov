CREATE TABLE labbi_users (
    user_id BIGINT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'quiz')),
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.3
        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0)),
    revision BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE labbi_messages (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES labbi_users(user_id) ON DELETE CASCADE,
    telegram_message_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (length(content) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, telegram_message_id, role)
);

CREATE INDEX labbi_messages_history_idx ON labbi_messages (user_id, id DESC);
