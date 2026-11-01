-- A scheduler cursor is tied to a profile version; it never changes consent.
CREATE TABLE labbi_briefing_cursors (
 user_id BIGINT PRIMARY KEY REFERENCES labbi_notification_profiles(user_id),
 profile_version BIGINT NOT NULL CHECK(profile_version>=0),
 next_at TIMESTAMPTZ NOT NULL CHECK(isfinite(next_at))
);
-- Public course content only. Failed/interrupted generation is not regenerated.
CREATE TABLE labbi_focus_cache (
 catalog_version TEXT NOT NULL CHECK(catalog_version ~ '^[0-9a-f]{64}$'),
 topic TEXT NOT NULL CHECK(topic ~ '^[a-z0-9][a-z0-9_-]{0,63}$'),
 token UUID NOT NULL,
 body TEXT CHECK(body IS NULL OR length(body) BETWEEN 1 AND 1600),
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(catalog_version,topic)
);
