# Проверка и демонстрация ЛР1

Карта требований составлена по разделу ЛР1 в [методичке](labs-guide.md#lab1).
Она связывает требования с кодом и проверками. Итоговый результат запуска тестов,
эксперимент и выводы приведены в отчёте ЛР1; наличие теста само по себе не является
утверждением, что последний запуск прошёл успешно.

## Карта обязательных требований

| Требование | Реализация | Проверки |
|---|---|---|
| Личные текстовые сообщения, `/start`, обработка без echo | Фильтры и порядок обработчиков в `app/handlers/assistant.py`; `typing` обновляется раз в 4 секунды во время генерации | `test_start_and_commands_do_not_call_model`, `test_regular_message_calls_model_and_sends_plain_answer`, `test_groups_and_nontext_are_ignored` |
| OpenAI-совместимый асинхронный LLM API, отдельный клиент | `app/llm.py`: общая `aiohttp.ClientSession`, `/chat/completions`, роли, модель, `temperature`, выходной лимит и запрет платных провайдеров | `test_client_sends_roles_parameters_and_free_price_limit` |
| Начальный `/study`, отдельные `/translate` и собственный `/quiz` | `app/prompts.py`; `/quiz` хранит тему и прогресс пяти вопросов в БД | `test_mode_clears_own_history_preserves_temperature_and_changes_prompt`, `test_quiz_has_five_questions_hint_keeps_progress_and_restart_restores_it` |
| Хотя бы один режим с few-shot, включая пограничный случай | В `/study` показано различие `=` и `==`; в `/translate` есть перевод и уточнение отсутствующего целевого языка | Промпты передаются отдельным сообщением `system`; это проверяет `test_history_roles_order_and_user_isolation` |
| Смена режима хранится в PostgreSQL, очищает только свою историю и не вызывает модель | `DialogService.mode`, `DialogRepository.change_mode` | Тест смены режима выше; `test_start_and_commands_do_not_call_model`; интеграционный тест репозитория |
| Раздельная постоянная история, роли и порядок | `app/repository.py`, миграции `001`–`005`; SQL-параметры `$1` и далее, общий пул | `test_history_roles_order_and_user_isolation`, `test_service_recreation_restores_settings_and_history`, `test_dialog_migrations_isolation_persistence_atomicity_and_revision` |
| Ограничение контекста по числу сообщений и оценочному объёму | `build_context` и `estimate_tokens` в `app/dialog.py`; `HISTORY_MAX_MESSAGES`, `HISTORY_MAX_TOKENS` в конфигурации | `test_context_trims_oldest_complete_pairs_and_preserves_current`, `test_context_obeys_estimated_token_budget`, `test_oversized_current_message_is_rejected_without_removing_system` |
| При сокращении сохраняются системная инструкция, текущий запрос и связные упорядоченные пары | Удаляются самые старые полные пары; архив БД не урезается. В продолжении неполного ответа при необходимости сохраняется его конец | Предыдущие три теста и `test_continuation_preserves_tail_when_partial_answer_exceeds_context` |
| `/settings`: режим, модель, температура; изменения изолированы и применяются к следующему запросу | `DialogService.preferences`, `DialogRepository.set_temperature`; разрешены все конечные числа 0–1, включая обязательные 0.0/0.3/0.7/1.0 | `test_settings_accepts_allowed_values`, `test_settings_rejects_bad_values_without_api_or_state_change`, `test_mode_clears_own_history_preserves_temperature_and_changes_prompt` |
| `/reset` очищает только свою историю, сохраняя режим и температуру | `DialogService.reset`, транзакция `DialogRepository.reset` | `test_reset_clears_only_history`, `test_reset_interrupts_model_immediately_and_preserves_other_users` |
| Ошибки LLM не сохраняют ложный успешный обмен | Ответ проверяется до сохранения; вопрос и ответ записываются одной транзакцией после успешной генерации | `test_model_failure_does_not_store_a_partial_exchange`, интеграционный тест атомарности |
| Таймаут, HTTP-ошибка, недоступность, пустой или некорректный ответ | `ModelError` с безопасными русскими сообщениями в `app/llm.py`; нет скрытого повторного вызова модели | `test_http_errors_are_safe_and_not_retried`, `test_empty_or_invalid_result_is_rejected`, `test_timeout_is_safe`, `test_network_error_is_safe`, `test_safe_error_and_database_error_do_not_expose_details` |
| Длинные ответы делятся, текст не теряется, порядок сохраняется | `app/formatting.py`: Markdown → Telegram entities, части до 4096 единиц UTF-16; `send_part` отправляет последовательно | `test_long_answer_is_sent_in_order_without_text_loss`, `test_long_entity_is_split_without_losing_text_or_unicode`, `test_long_code_keeps_formatting_across_messages` |
| Неполная генерация отличается от пустого/ошибочного ответа | При `finish_reason=length` сохраняются полученный текст и `is_complete=false`; продолжение запускается новым сообщением | `test_truncated_result_preserves_all_received_text_without_retry`, `test_truncated_text_is_saved_and_can_continue_after_service_restart`, `test_truncated_answer_is_delivered_in_parts_instead_of_refusal` |
| Конфигурация и секреты вынесены из кода | `app/config.py`, `.env.example`; настоящий `.env` исключён из Git; обязательные поля проверяются безопасно | `test_llm_key_is_required_for_application_but_not_database_setup`, `test_llm_configuration_errors_do_not_leak_credentials`, `test_deploy_archive_never_contains_env_or_keys` |
| Ответственность разделена | Telegram — `app/handlers/assistant.py`, LLM — `app/llm.py`, промпты — `app/prompts.py`, диалог — `app/dialog.py`, SQL — `app/repository.py`, конфигурация — `app/config.py` | Тесты подменяют LLM и Telegram; интеграционные тесты используют отдельную тестовую БД |
| Безопасное логирование внешнего вызова | `app/llm.py`: начало с UUID и моделью, результат со статусом и длительностью; `app/logging_setup.py`: `SecretFilter`, включая исключения | `test_log_filter_redacts_secrets_and_exception`, `test_logging_masks_openrouter_key` |
| Повторный запуск, миграции, healthcheck и завершение | `app/__main__.py`, `app/migration.py`, `app/health.py`, `app/healthcheck.py`; текущие данные не удаляются при запуске | `test_real_db_password_restart_health_and_persistence`, `test_dialog_migrations_isolation_persistence_atomicity_and_revision`, `tests/test_lifecycle.py` |
| Эксперимент: 12 запусков, четыре температуры, сравнение инструкций | Скрипт и исходные результаты эксперимента прилагаются к отчёту | Проверяется по фактическим ответам внешней модели, а не подменяется автотестами |
| Отчёт, AI-декларация, ссылка на репозиторий, тег `lab1` | Комплект документов и аннотированный Git-тег фиксируются после финальных проверок | Проверить `git show lab1` и чистоту рабочего дерева |

## Воспроизводимая демонстрация

Автотесты не отправляют сообщения настоящему Telegram-боту и не расходуют квоту
OpenRouter. Для демонстрации настоящих режимов понадобятся свой бот, секреты
в локальном `.env` и доступность бесплатного провайдера.

1. Запустить приложение по README. Для настроенного локального rootless Docker:

   ```bash
   source scripts/env.sh
   bash scripts/docker-local.sh start
   bash scripts/start.sh
   ```

   Запускается один экземпляр polling для одного Telegram-токена. В другом
   терминале `source scripts/env.sh` и `python -m app.healthcheck` проверяют БД
   и готовность polling без чтения пользовательской истории.

2. В личном чате A отправить `/start`, `/study`, `/settings 0.3`, затем
   «Объясни разницу между списком и кортежем в Python». Отправить уточнение
   «А какой из них можно изменить?» — модель должна получить предыдущую пару.
3. Из второго Telegram-аккаунта B отправить `/study`, `/settings 0.7` и вопрос
   по другой теме. Показать, что `/settings` в A по-прежнему возвращает 0.3,
   а диалоги независимы. Для повторяемой проверки изоляции без второго аккаунта
   используется интеграционный тест, а не выдуманный снимок Telegram.
4. Остановить только Python-процесс через Ctrl+C, не удаляя БД, и запустить снова.
   Повторить `/settings` и контекстное уточнение в каждом чате. История и настройки
   восстанавливаются из PostgreSQL.
5. В A переключиться на `/translate`, выполнить «На английский: лабораторная
   работа завершена», затем `/quiz` и тему «Циклы Python». Показать один вопрос,
   подсказку, проверку ответа и сохранение прогресса. Смена режима очищает
   прежний контекст A, не затрагивая B.
6. Выполнить `/reset` в A: история очищена, его текущие режим и температура
   сохранены, данные B не меняются. `/settings 2` отклоняется без LLM-вызова.
7. Показать контролируемые сбои внешнего API и длинный ответ автотестами:

   ```bash
   python -m pytest -q tests/test_llm.py tests/test_handlers.py
   ```

   Настоящая сеть не обязана случайно ломаться во время защиты. Тесты задают
   таймауты, ошибки, пустые ответы, ограничение длины и ответы длиннее 4096,
   проверяя конкретные безопасные сообщения, части и порядок отправки.
8. Показать таблицу всех 12 экспериментальных ответов, сравнение исходной
   и улучшенной инструкции, выводы, AI-декларацию и `git show lab1`.

## Известный пограничный случай доставки

Сначала вопрос и полученный ответ фиксируются в PostgreSQL, затем ответ
отправляется в Telegram. Если отправка завершилась сетевой ошибкой, ответ
остаётся в БД, хотя пользователь мог не получить весь текст. Для `429` есть
до двух повторов текущей части, для ошибки entities — отправка той же части
обычным текстом. Постоянной очереди исходящих сообщений и возобновления
недоставленных частей после перезапуска нет. `is_complete` описывает полноту
генерации модели, а не доставку.

Это ограничение явно раскрывается в отчёте; доставка через transactional outbox
не является обязательным требованием ЛР1. Временные ошибки самого LLM не
повторяются автоматически: следующее обращение требует нового сообщения
пользователя, что бережёт бесплатную квоту и исключает прежнюю цепочку генераций.
