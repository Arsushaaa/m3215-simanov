ALTER TABLE labbi_users
    ADD COLUMN quiz_topic TEXT,
    ADD COLUMN quiz_question INTEGER NOT NULL DEFAULT 0 CHECK (quiz_question BETWEEN 0 AND 5),
    ADD COLUMN quiz_finished BOOLEAN NOT NULL DEFAULT false;
