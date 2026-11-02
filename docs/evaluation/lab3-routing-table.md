# Все 15 исходных запросов: реальный эксперимент ЛР3

Финальный результат v5: **13/15**, порог 12/15 пройден. Все неуспехи сохранены.

| ID | Запрос | Ожидание | Фактический маршрут v5 | Верно |
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
| clarify_1 | Подготовь меня к завтрашнему дню. | clarify | clarify | Да |
| clarify_2 | Сделай что-нибудь полезное. | clarify | legacy_study | Нет |
| clarify_3 | Расскажи, что для меня важно сегодня. | clarify | legacy_study | Нет |

Источник — [v5 results](lab3-experiment-v5-results.json), [протокол и анализ](lab3-experiment-v5.md).
История v2/v3/v4:13/15,13/15,14/15. Все15 входныхpayload одинаковы; различие не доказывает улучшение классификатора.
