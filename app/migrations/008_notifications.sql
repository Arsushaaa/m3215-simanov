-- Explicit destinations, confirmed places, opt-in and an outbox. No worker/HTTP here.
ALTER TABLE labbi_reminders ADD CONSTRAINT labbi_reminders_id_owner UNIQUE(id,user_id);
CREATE TABLE labbi_notification_profiles (
    user_id BIGINT PRIMARY KEY REFERENCES labbi_users(user_id) ON DELETE RESTRICT,
    destination_chat_id BIGINT CHECK(destination_chat_id > 0),
    city JSONB CHECK(city IS NULL OR jsonb_typeof(city)='object'),
    city_confirmed_at TIMESTAMPTZ CHECK(city_confirmed_at IS NULL OR isfinite(city_confirmed_at)),
    briefing_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    consented_at TIMESTAMPTZ CHECK(consented_at IS NULL OR isfinite(consented_at)),
    version BIGINT NOT NULL DEFAULT 0 CHECK(version>=0),
    CHECK((city IS NULL)=(city_confirmed_at IS NULL)),
    CHECK(NOT briefing_enabled OR (city IS NOT NULL AND destination_chat_id IS NOT NULL AND consented_at IS NOT NULL))
);
CREATE TABLE labbi_city_menus (
    user_id BIGINT PRIMARY KEY REFERENCES labbi_notification_profiles(user_id) ON DELETE CASCADE,
    token_hash BYTEA NOT NULL CHECK(octet_length(token_hash)=32),
    profile_version BIGINT NOT NULL CHECK(profile_version>=0),
    expires_at TIMESTAMPTZ NOT NULL CHECK(isfinite(expires_at)),
    options JSONB NOT NULL DEFAULT '[]' CHECK(jsonb_typeof(options)='array' AND jsonb_array_length(options)<=50),
    selected INTEGER CHECK(selected IS NULL OR selected BETWEEN 0 AND 49)
);
CREATE TABLE labbi_outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES labbi_notification_profiles(user_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK(kind IN ('reminder','briefing')),
    reminder_id UUID UNIQUE,
    local_date DATE,
    destination_chat_id BIGINT NOT NULL CHECK(destination_chat_id>0),
    profile_version BIGINT NOT NULL CHECK(profile_version>=0),
    timezone TEXT NOT NULL CHECK(length(timezone) BETWEEN 1 AND 100),
    due_at TIMESTAMPTZ NOT NULL CHECK(isfinite(due_at)),
    deadline_at TIMESTAMPTZ NOT NULL CHECK(isfinite(deadline_at)),
    body TEXT CHECK(body IS NULL OR length(body) BETWEEN 1 AND 16000),
    state TEXT NOT NULL DEFAULT 'pending'
      CHECK(state IN ('pending','preparing','ready','sending','retry_wait','sent','failed','missed','cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
    next_attempt_at TIMESTAMPTZ NOT NULL,
    lease_until TIMESTAMPTZ,
    lease_token UUID,
    lease_version BIGINT NOT NULL DEFAULT 0 CHECK(lease_version>=0),
    llm_reserved INTEGER NOT NULL DEFAULT 0 CHECK(llm_reserved BETWEEN 0 AND 1),
    mcp_reserved INTEGER NOT NULL DEFAULT 0 CHECK(mcp_reserved BETWEEN 0 AND 3),
    error_code TEXT CHECK(error_code IS NULL OR error_code ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    finished_at TIMESTAMPTZ,
    FOREIGN KEY(reminder_id,user_id) REFERENCES labbi_reminders(id,user_id),
    UNIQUE(user_id,local_date),
    CHECK((kind='reminder' AND reminder_id IS NOT NULL AND local_date IS NULL)
       OR (kind='briefing' AND reminder_id IS NULL AND local_date IS NOT NULL)),
    CHECK(deadline_at=due_at + CASE WHEN kind='reminder' THEN INTERVAL '15 minutes' ELSE INTERVAL '60 minutes' END),
    CHECK((state IN ('preparing','sending'))=(lease_until IS NOT NULL AND lease_token IS NOT NULL)),
    CHECK((lease_until IS NULL)=(lease_token IS NULL)),
    CHECK(lease_until IS NULL OR isfinite(lease_until)),
    CHECK(state NOT IN ('ready','sending','retry_wait','sent') OR body IS NOT NULL),
    CHECK(state NOT IN ('sending','sent') OR attempts>=1),
    CHECK((state IN ('sent','failed','missed','cancelled'))=(finished_at IS NOT NULL)),
    CHECK(finished_at IS NULL OR (isfinite(finished_at) AND finished_at>=created_at)),
    CHECK(isfinite(next_attempt_at) AND next_attempt_at>=due_at)
);
CREATE INDEX labbi_outbox_due_idx ON labbi_outbox(state,next_attempt_at,due_at);

CREATE FUNCTION labbi_notification_profile_before() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.briefing_enabled AND NOT EXISTS(SELECT 1 FROM labbi_users WHERE user_id=NEW.user_id AND timezone IS NOT NULL) THEN
        RAISE EXCEPTION 'Timezone required' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN
        IF ROW(NEW.destination_chat_id,NEW.city,NEW.city_confirmed_at,NEW.briefing_enabled,NEW.consented_at)
            IS DISTINCT FROM ROW(OLD.destination_chat_id,OLD.city,OLD.city_confirmed_at,OLD.briefing_enabled,OLD.consented_at) THEN
            NEW.version := OLD.version+1;
        ELSIF NEW.version NOT IN (OLD.version,OLD.version+1) THEN
            RAISE EXCEPTION 'Invalid profile version' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER labbi_notification_profile_before BEFORE INSERT OR UPDATE ON labbi_notification_profiles
    FOR EACH ROW EXECUTE FUNCTION labbi_notification_profile_before();

CREATE FUNCTION labbi_notification_profile_changed() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.version<>OLD.version THEN
        DELETE FROM labbi_city_menus WHERE user_id=NEW.user_id;
        UPDATE labbi_outbox SET state='cancelled',finished_at=clock_timestamp(),
            lease_until=NULL,lease_token=NULL,error_code='PROFILE_CHANGED'
            WHERE user_id=NEW.user_id AND kind='briefing' AND state IN ('pending','preparing','ready','retry_wait');
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER labbi_notification_profile_changed AFTER UPDATE ON labbi_notification_profiles
    FOR EACH ROW EXECUTE FUNCTION labbi_notification_profile_changed();

CREATE FUNCTION labbi_notification_timezone_changed() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.timezone IS DISTINCT FROM OLD.timezone THEN
        UPDATE labbi_notification_profiles SET version=version+1,
            briefing_enabled=CASE WHEN NEW.timezone IS NULL THEN FALSE ELSE briefing_enabled END
            WHERE user_id=NEW.user_id;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER labbi_notification_timezone_changed AFTER UPDATE OF timezone ON labbi_users
    FOR EACH ROW EXECUTE FUNCTION labbi_notification_timezone_changed();

CREATE FUNCTION labbi_enqueue_saved_reminder() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO labbi_outbox(user_id,kind,reminder_id,destination_chat_id,profile_version,
        timezone,due_at,deadline_at,next_attempt_at)
      SELECT NEW.user_id,'reminder',NEW.id,p.destination_chat_id,p.version,
        NEW.timezone,NEW.remind_at,NEW.remind_at+INTERVAL '15 minutes',NEW.remind_at
      FROM labbi_notification_profiles p WHERE p.user_id=NEW.user_id AND p.destination_chat_id IS NOT NULL
      ON CONFLICT(reminder_id) DO NOTHING;
    RETURN NEW;
END $$;
CREATE TRIGGER labbi_enqueue_saved_reminder AFTER INSERT ON labbi_reminders
    FOR EACH ROW EXECUTE FUNCTION labbi_enqueue_saved_reminder();
