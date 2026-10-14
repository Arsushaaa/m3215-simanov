ALTER TABLE labbi_users
    DROP CONSTRAINT labbi_users_temperature_check,
    ADD CONSTRAINT labbi_users_temperature_check CHECK (temperature BETWEEN 0.0 AND 1.0);
