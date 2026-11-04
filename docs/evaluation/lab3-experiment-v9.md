# Повтор Router v9, 4 ноября 2026

Результат: **13/15**, 15 started / 15 HTTP. Все неуспехи входят в знаменатель; исходные запросы и ожидания неизменны.

[Протокол](lab3-experiment-v9-protocol.json), [результат](lab3-experiment-v9-results.json).

| Случай | Ожидание | Получено |
|---|---|---|
| schedule_1 | schedule | schedule |
| schedule_2 | schedule | schedule |
| schedule_3 | schedule | schedule |
| schedule_4 | schedule | schedule |
| knowledge_1 | knowledge | knowledge |
| knowledge_2 | knowledge | knowledge |
| knowledge_3 | knowledge | knowledge |
| knowledge_4 | knowledge | knowledge |
| lifestyle_1 | lifestyle | lifestyle |
| lifestyle_2 | lifestyle | lifestyle |
| lifestyle_3 | lifestyle | lifestyle |
| lifestyle_4 | lifestyle | ModelError |
| clarify_1 | clarify | clarify |
| clarify_2 | clarify | clarify |
| clarify_3 | clarify | ModelError |

Реальная бесплатная модель, один production Router на запрос, T=0.2, native; специалисты/MCP/Telegram не вызываются. Ожидания в модель не передавались. Started-попытки не повторялись, платных моделей/fallback нет.

v9 → v10: изменён AgentService для отбрасывания вступления при валидном native tool call. Сам Router не менялся; все 15 хешей payload совпадают. Поэтому разница 13/15 → 15/15 не приписывается улучшению семантики Router. Две ошибки v9 возникли до наблюдаемого ответа модели, точная HTTP-причина не записана старым измерителем и не угадывается.

Два конкретных вопроса преподавателя и отдельные контролирующие формулировки проверены в [общем отчёте](teacher-feedback-20261104.md).
