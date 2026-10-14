ALTER TABLE labbi_messages
    ADD COLUMN parts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD CONSTRAINT labbi_messages_parts_array_check CHECK (jsonb_typeof(parts) = 'array');
