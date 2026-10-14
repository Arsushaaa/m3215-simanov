ALTER TABLE labbi_messages
    ADD COLUMN is_complete BOOLEAN NOT NULL DEFAULT true,
    ADD CONSTRAINT labbi_messages_user_complete_check CHECK (role = 'assistant' OR is_complete);
