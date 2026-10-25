# Проверка и демонстрация ЛР2

Карта составлена по [требованиям ЛР2](labs-guide.md#lab2) и [спецификации](../spec.md).
Она связывает наблюдаемое поведение с реализацией и доказательствами проверки.
Наличие теста означает, что сценарий предусмотрен; результат запуска берётся
из сохранённого протокола. Автотесты используют подменённые Telegram, LLM и
погодный HTTP; PostgreSQL и MCP stdio проверяются настоящими процессами.

## Источники результатов

- [Матрица 28 критериев](evaluation/lab2-stage6-cases.json) и
  [фактический протокол этапа 6](evaluation/lab2-stage6-checks.json): успешная
  приёмка, без упавших и пропущенных проверок. Подробности —
  [воспроизведение и границы проверки](lab2-reliability.md).
- [Финальная приёмка](evaluation/lab2-final-checks.json): 1107 успешных тестов,
  включая 24 PostgreSQL, все 28 критериев, без ошибок и пропусков.
- [Итоговая контейнерная проверка](evaluation/lab2-final-container.json): сборка,
  непривилегированный пользователь, файловая система только для чтения,
  MCP stdio и учебное расписание без сети. Эта проверка не заменяет запуск
  Telegram polling с настоящим токеном.
- [Сверка материалов](evaluation/lab2-artifacts-checks.json): JSON, локальные
  ссылки, исключение локальных данных и сравнение сдаваемых файлов с известными
  секретами длиной от восьми символов без вывода значений.
- [Свежий протокол независимого клиента](evaluation/lab2-stage7-probe.json): discovery,
  чтение дня/недели и отказ `UNAUTHORIZED` для записи без доверенного права.
- [Неизменный набор эксперимента](evaluation/lab2-routing.json): 12 запросов,
  фиксированные часы и ожидаемые действия. Результаты реальной модели сохраняет
  `scripts/lab2_experiment.py` в `docs/evaluation/lab2-results.json`.
  Итог — 12/12 действий и 9/9 аргументов; исходный и промежуточный неуспешные
  прогоны сохранены в [анализе](evaluation/lab2-experiment.md). В каждом прогоне
  11 LLM-запросов и один локальный ответ, суммарно 33 реальных вызова.

## Карта обязательных требований

Текущее дополнение: [исправления и 1312 тестов](live-fixes.md),
[предварительно зарегистрированное сравнение моделей](evaluation/lab2-comparison.md).
dots T=0,3 прошла 10/12 и 8/9 с фактическими 12/12 и 8/9; дополнительные
формулировки и один реальный MCP-read проверены отдельно. Исторические прогоны
с изменением промпта не заменены и не используются как независимая оценка.

Имена в столбце «Проверка» — существующие функции pytest. Полные пути и исходы
критериев этапа 6 указаны в его машинном протоколе.

| Требование методички и спецификации | Реализация | Проверка и доказательство |
|---|---|---|
| Продолжение ЛР1: режимы, история, настройки, разбиение длинного текста | `app/dialog.py`, `app/repository.py`, `app/prompts.py`, `app/formatting.py`, `app/handlers/assistant.py`; неизменный тег `lab1` | `test_other_modes_keep_legacy_model_path`, `test_dialog_persists_only_user_and_final_and_respects_revision`; набор тестов ЛР1 входит в полную приёмку; `source_checks.lab1_commit_unchanged` |
| Собственный MCP-сервер: три инструмента, стандартные входные/выходные схемы и `structuredContent` | `app/mcp/server.py`, `app/mcp/contracts.py`; закреплённый официальный SDK в `requirements.txt` | `test_discovery_exposes_exact_contracts_and_read_only_resource`, `test_weather_result_is_structured_and_matches_discovered_schema` в `tests/test_mcp_server.py`; независимый probe |
| `get_weather`: нормализованный город/страна, температура, условия, ветер с единицей, момент наблюдения и зона | `app/mcp/weather.py`; фиксированные Open-Meteo geocoding/weather endpoints | `test_weather_uses_fixed_endpoints_checked_units_and_utc_to_iana`; критерий `weather_success` |
| Неизвестный или неоднозначный город, HTTP-ошибка, тайм-аут, некорректный ответ | Проверка кандидатов и результата в `app/mcp/weather.py`; без автоматического выбора первого города | `test_missing_city_is_not_found_and_does_not_fetch_weather`, `test_same_named_places_require_country_region_without_population_selection`, `test_both_http_calls_share_one_total_timeout`; критерии `weather_not_found`, `weather_failure` |
| `get_schedule`: дата ISO, поля `start/end/title/location`, пустой день; источник задан конфигурацией | `app/mcp/schedule.py`, `app/mcp/contracts.py`, `app/config.py`, учебный `data/schedule.json`; путь не входит в аргументы модели | `test_schedule_returns_exact_observable_lesson_fields`, `test_empty_day_and_outside_coverage_are_distinct`, `test_schedule_date_is_real_iso_and_in_range`; критерии `schedule_fields_and_empty_day`, `invalid_arguments` |
| Read-only ресурс `schedule://current-week`: понедельник–воскресенье, границы месяца/года и зона | `app/mcp/schedule.py`, `app/mcp/server.py`; внедряемые часы | `test_week_crosses_month_and_year_without_changing_system_clock`, `test_week_uses_configured_zone_before_selecting_calendar_date`; критерий `week_boundary` |
| Discovery клиента без копии схем в Telegram-обработчиках; отсутствие MCP сохраняет обычный диалог | `app/mcp/client.py`, `app/agent.py`; обработчик `/tools` читает обнаруженный каталог | `test_discovery_is_canonical_and_child_environment_excludes_bot_secrets`, `test_failed_real_client_preserves_plain_dialog_without_reconnect`; критерий `mcp_failure_plain_dialog` |
| Выбор ответа, уточнения либо одного инструмента по строгому решению; разрешённый allowlist | `app/llm.py`, `app/agent.py`, `app/mcp/contracts.py`; заранее выбранные `structured`/`native`, без извлечения команд из свободного текста | `test_strict_decisions_reject_free_text_extra_fields_and_untrusted_targets`, `test_canonical_but_undiscovered_tool_is_not_executed`, `test_invalid_calls_never_reach_sdk`; критерий `tool_allowlist` |
| Ограниченный цикл: методичка допускает максимум 2 MCP; проект использует максимум 1 прикладной MCP и 2 LLM на чтение | `app/agent.py`; финальный LLM не может вызвать новый инструмент, discovery не входит в бюджет сообщения | `test_read_uses_two_models_one_mcp_and_final_disables_tools`, `test_multiple_native_calls_are_refused_before_partial_execution`, `test_context_budget_accounts_for_schema_and_tool_result`; критерий `bounded_calls_context` |
| Результат чтения передаётся модели с источником; неверная схема/дата не становится фактом | `app/mcp/client.py`, `app/llm.py`, `app/agent.py`; отдельный временный tool-контекст, в истории только вопрос/итог | `test_well_formed_schedule_for_another_day_is_not_a_fact`, `test_dialog_persists_only_user_and_final_and_respects_revision`; критерии `result_binding`, `why_and_history` |
| `add_reminder`: текст 1–500, будущий момент с ISO-offset, сохранённая IANA-зона, проверка DST | `app/mcp/contracts.py`, `app/reminder_intent.py`, `app/confirmations.py`; время сверяется с пользовательской просьбой | `test_reminder_future_context_is_explicit_and_normalized`, `test_reminder_text_limits`, `test_dst_gap_and_overlap_need_clarification`, `test_confirmation_preparation_checks_saved_zone_db_time_and_dst`; критерии `invalid_arguments`, `timezone_dst_expiry` |
| Точная карточка и явное подтверждение владельца в течение 5 минут; изменение/отмена/истечение отзывают право | `app/confirmation_models.py`, `app/confirmations.py`, `app/agent_repository.py`, `app/handlers/assistant.py`; отдельный `confirmation_epoch` | `test_reminder_dialog_confirmation_real_stdio_postgres_and_no_extra_model_calls`, `test_confirmation_revocation_covers_pending_and_approved_cards_and_expiry`; критерий `confirmation_real_stdio` |
| Модель не задаёт владельца и право записи; независимый MCP-клиент без права получает отказ | Идентичность из Telegram update; capability минимум 256 бит передаётся в доверенном MCP `_meta`, хеш и связанное действие проверяет PostgreSQL | `test_arguments_trim_strings_but_do_not_accept_model_owner`, `test_confirmed_write_sends_capability_only_in_trusted_metadata`, `test_reminder_repository_rejects_unauthorized_stale_or_changed_actions`; критерий `ownership` и независимый probe |
| Повтор update даёт одну подготовку; конкурентные подтверждения — одно напоминание | `app/agent_repository.py`, `app/migrations/006_agent_state.sql`; блокировки и уникальные ограничения PostgreSQL | `test_confirmation_prepare_is_concurrent_atomic_and_separate_from_dialog_revision`, `test_confirmation_services_race_across_pools_commits_once_and_replays_without_mcp`; критерии `update_deduplication`, `concurrent_confirmation`, `idempotent_write` |
| PostgreSQL хранит зоны, состояния, напоминания, квитанции и owner-аудит после перезапуска | `app/agent_repository.py`, миграция `006`; реестр операций отделён от удаляемой истории | `test_confirmation_database_restart_preserves_zone_actions_receipts_and_private_audit`, `test_agent_audit_is_concurrent_idempotent_isolated_and_survives_reset`; критерий `restart_persistence` |
| `/timezone`: просмотр/сохранение IANA и локального времени; зона не угадывается моделью | `app/confirmations.py`, `app/timezone_ui.py`, `app/timezone_lookup.py`, `app/time_display.py`; дополнительно меню поиска стран/городов | `test_timezones_use_explicit_iana_and_database_clock`, `test_timezone_menu_only_applies_after_owner_confirmation_and_persists`, `test_reminder_rejects_missing_or_nonportable_zone`; критерии `timezone_dst_expiry`, `timezone_readonly` |
| `/start`, `/tools`, `/why`; безопасные аргументы/статусы/время и краткая причина, изоляция владельца | `app/handlers/assistant.py`, `app/confirmations.py`, `app/agent_repository.py`; команды без LLM | `test_why_shows_owner_event_safe_arguments_and_fixed_reason`, `test_why_without_decisions_is_explicit`; `tests/test_confirmation_handlers.py`; критерий `why_and_history` |
| Общее представление ошибок; сбои MCP/HTTP/схемы, неизвестный инструмент, лимит, отмена и просрочка не дают ложного успеха | Типизированные ошибки `app/mcp/contracts.py`; ограниченные дедлайны клиента/агента; без скрытых повторов LLM/MCP | `test_tool_http_errors_are_safe_and_never_retried`, `test_routing_modes_are_selected_before_http_not_fallback_after_error`, `test_timeout_with_uncommitted_action_reports_unknown_not_success`; критерии `no_hidden_retries`, `unknown_commit` |
| Ошибка после commit и ошибка Telegram не создают новую операцию; квитанция восстанавливается по прежнему action | `app/confirmations.py`, `app/agent_repository.py`, `app/handlers/assistant.py`; удаление карточки после доставки квитанции | `test_reminder_repository_unknown_commit_ack_recovers_same_action`, `test_receipt_delivery_failure_keeps_card_for_receipt_recovery`; критерии `unknown_commit`, `telegram_delivery` |
| Четыре категории негативных сценариев: неизвестный инструмент/секреты, инъекция данных, плохое время, недоступность сервиса | Строгие контракты, фиксированные endpoints, минимум окружения MCP, отключённые инструменты финала | `test_invalid_calls_never_reach_sdk`, `test_injected_tool_data_cannot_start_another_action_in_read_final`, `test_dst_gap_and_overlap_need_clarification`, `test_failed_real_client_preserves_plain_dialog_without_reconnect`; критерии `tool_allowlist`, `untrusted_result`, `timezone_dst_expiry`, `mcp_failure_plain_dialog` |
| Секреты, Telegram identity, сообщения, системные инструкции и capability не попадают в текстовые логи | `app/logging_setup.py`; безопасный owner-аудит в PostgreSQL; серверное окружение без ключей Telegram/LLM | `test_library_payloads_are_replaced_before_formatting_and_never_logged`, `test_actual_dispatcher_send_failure_does_not_log_update_identity_or_telegram_payload`, `test_mcp_child_environment_is_a_server_only_allowlist`; критерий `privacy_logs` |
| Миграции и конфигурация воспроизводимы; SQL/расписание включены в Docker и деплой; персональных данных в учебном файле нет | `app/migration.py`, `app/config.py`, `.env.example`, Compose, Dockerfile, `scripts/cloud.py`, `.gitignore` | `test_agent_migration_upgrades_lab1_and_preserves_history_and_registry`, `test_repository_fixture_covers_all_fourteen_days_and_has_no_personal_fields`, `tests/test_compose.py`, `tests/test_scripts.py`; `source_checks.migrations_001_to_006_unchanged` и контейнерный протокол |
| `AGENTS.md`, `spec.md`, `plan.md` соответствуют реализации; запуск и независимая демонстрация описаны | Документы в корне, README, документы этапов 2–6, `scripts/mcp_probe.py` | Ручная сверка карты с исходниками; `test_independent_stdio_probe_needs_no_database_or_telegram_keys`, `test_independent_stdio_client_commits_confirmed_reminder_in_real_postgres` |
| Реальный эксперимент: все 12 запросов, действие ≥10/12, аргументы ≥8/9, исходные данные и разбор ошибок | Неизменный `docs/evaluation/lab2-routing.json`; скрипт `scripts/lab2_experiment.py` | Итоговый `docs/evaluation/lab2-results.json`: 12/12 и 9/9. В `lab2-experiment.md` разобраны оба неуспешных прогона и повтор всех кейсов после уточнения одного блока правил |
| Итоговый отчёт, использование AI, ограничения, зафиксированное состояние `lab2` | Финальный отчёт/инструкция защиты и Git-тег после приёмки | `docs/lab2-report.md` и `docs/lab2-defense.md`; после финального коммита проверить `git show lab2` и чистоту рабочего дерева |

## Воспроизводимая демонстрация

1. Подготовить окружение и запустить единственный экземпляр Telegram polling
   по README. Настоящие ключи остаются в локальном `.env`:

   ```bash
   source scripts/env.sh
   bash scripts/docker-local.sh start
   bash scripts/start.sh
   ```

2. В личном чате выбрать `/study`, посмотреть `/tools` и `/timezone`.
   Сохранить `/timezone Europe/Moscow` либо выбрать зону через меню; сверить
   показанное локальное время. Запросить текущую погоду с указанием города
   и страны. `/why` должен показать чтение `get_weather` и безопасные статусы.
   Погодный прогноз на завтра не входит в контракт `get_weather`.
3. Запросить «Покажи расписание на 20 октября 2026 года». Показать источник
   учебного файла, точное время занятий и `/why`. Вопрос о дне вне покрытия
   должен дать отсутствие данных; пустой день внутри покрытия означает,
   что занятий нет. Личного расписания в Git нет.
4. Попросить напомнить о действии в явно указанное будущее время. Сверить
   текст, дату, время и зону карточки. До подтверждения напоминание не создаётся.
   Подтвердить, получить ID и квитанцию; карточка исчезает после её доставки.
   Напоминание только сохраняется: уведомление по времени относится к ЛР3.
5. Отдельным запросом подготовить карточку и отменить её. Двойное подтверждение,
   чужой callback, просрочку, конкурентные процессы и неизвестный исход записи
   показать воспроизводимыми тестами, а не пытаться случайно вызвать сбой в Telegram:

   ```bash
   python -m scripts.lab2_reliability
   ```

6. Остановить Python-процесс и снова запустить его, сохранив PostgreSQL.
   Проверить `/timezone`, `/settings`, `/why` и продолжение обычной истории.
   Настоящий stop/start PostgreSQL и две независимые конкурентные сессии
   также покрыты интеграционными тестами приёмки.
7. Запустить независимый MCP-клиент. Он обнаружит схемы, прочитает учебный день
   и неделю и получит отказ записи без capability. Эта команда не вызывает
   Telegram, OpenRouter и погодный HTTP и не требует их ключей:

   ```bash
   python -m scripts.mcp_probe --date 2026-10-12
   ```

   Разрешённая запись через настоящий stdio проверяется тестом
   `test_independent_stdio_client_commits_confirmed_reminder_in_real_postgres`
   в отдельной временной БД с тестовой идентичностью.
8. Показать все 12 результатов реальной модели, расчёт двух метрик, объяснение
   ошибок и сохранённый повтор после изменения одного элемента, если он потребовался.
   После финальной приёмки показать аннотированный тег `lab2` и сохранённый `lab1`.

## Границы результата

Реальные Telegram и OpenRouter требуют сети и доступного бесплатного провайдера.
Успешный запуск тестов не гарантирует их текущую доступность. Ошибка отправки
Telegram после commit не откатывает напоминание; повторная доставка квитанции
возможна, строгая однократная доставка не обещается.

Учебное расписание покрывает 12–25 октября 2026 года. Погода — текущее наблюдение,
уведомления по времени — будущая ЛР3. Автоматического продолжения длинных ответов
и скрытых повторных генераций нет. Соблюдение этих границ проверяется отдельно
от качества маршрутизации настоящей моделью.
