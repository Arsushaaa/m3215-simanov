# Все 15 исходных запросов: реальный эксперимент ЛР3

Текущий результат v4: **14/15**, порог 12/15 пройден. Ни один неуспех не исключён.

| ID | Запрос | Ожидание | Фактический маршрут v4 | Верно |
|---|---|---|---|---|
| schedule_1 | Какие пары у меня завтра? | schedule | schedule | Да |
| schedule_2 | Напомни в пятницу в 17:00 отправить лабораторную. | schedule | schedule | Да |
| schedule_3 | Когда сегодня начинается первая пара? | schedule | schedule | Да |
| schedule_4 | Покажи расписание занятий на эту неделю. | schedule | schedule | Да |
| knowledge_1 | По учебным материалам объясни назначение MCP. | knowledge | knowledge | Да |
| knowledge_2 | Найди в материалах определение контекстного окна. | knowledge | knowledge | Да |
| knowledge_3 | Какие правила в конспекте относятся к структурированному выводу? | knowledge | knowledge | Да |
| knowledge_4 | Составь краткое объяснение идемпотентности по учебным документам. | knowledge | knowledge | Да |
| lifestyle_1 | Какая сейчас погода в моём городе? | lifestyle | lifestyle | Да |
| lifestyle_2 | Стоит ли сегодня брать зонт? | lifestyle | lifestyle | Да |
| lifestyle_3 | Сильный ли сейчас ветер? | lifestyle | lifestyle | Да |
| lifestyle_4 | Как одеться с учётом текущей погоды? | lifestyle | lifestyle | Да |
| clarify_1 | Подготовь меня к завтрашнему дню. | clarify | schedule | Нет |
| clarify_2 | Сделай что-нибудь полезное. | clarify | clarify | Да |
| clarify_3 | Расскажи, что для меня важно сегодня. | clarify | clarify | Да |

Источник — [v4 results](lab3-experiment-v4-results.json), [протокол и анализ](lab3-experiment-v4.md).
Baseline v2 и повтор v3 сохранены отдельно: оба13/15. Все15 payload v4 совпадают с v2; разница не доказывает эффект исправления вывода.
