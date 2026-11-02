# Все15 исходных запросов: реальный эксперимент ЛР3

Текущая версия lab3-final-v2, v6: **12/15**, порог12/15 пройден. Все3ошибки сохранены.

| ID | Запрос | Ожидание | Факт v6 | Верно |
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
| clarify_2 | Сделай что-нибудь полезное. | clarify | legacy_study | Нет |
| clarify_3 | Расскажи, что для меня важно сегодня. | clarify | legacy_study | Нет |

[Протокол/анализ](lab3-experiment-v6.md),[результаты](lab3-experiment-v6-results.json).
Историяv2/v3/v4/v5:13/15,13/15,14/15,13/15. Всеpayloadv2/v4/v5/v6 одинаковы; вариативность результата не доказывает эффект фикса.
