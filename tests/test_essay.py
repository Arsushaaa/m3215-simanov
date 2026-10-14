import pytest

from app.essay import ESSAY_TOPIC_QUESTION, needs_essay_topic


@pytest.mark.parametrize(
    "text",
    [
        "напиши эссе",
        "напиши эссе на 10000 слов",
        "напиши эссе на10000слов",
        "Пожалуйста, напиши мне эссе на 10 000 слов",
        "Напиши мне огромное эссе, насколько большое, насколько ты можешь",
        "Напиши максимально длинное эссе на русском языке",
        "Напиши эссе в академическом стиле",
        "Напиши первую часть эссе на 500 слов",
        "Write an essay of 10000 words",
    ],
)
def test_bare_essay_needs_topic_even_with_unrelated_history(text):
    history = [
        {"role": "user", "content": "Как устроены циклы Python?"},
        {"role": "assistant", "content": "Цикл повторяет действия."},
    ]
    assert needs_essay_topic(text, [])
    assert needs_essay_topic(text, history)


@pytest.mark.parametrize(
    "text",
    [
        "Напиши эссе о Python на 500 слов",
        "Напиши эссе про роль образования",
        "Напиши эссе на тему искусственного интеллекта",
        "Напиши эссе: будущее ИИ",
        "Напиши эссе «1984»",
        "Напиши эссе о том, как можно считать слова",
        "Write an essay about education",
        "Что такое эссе?",
        "Напиши функцию, которая печатает слово эссе",
        "Переведи: напиши эссе на 10000 слов",
        "Сделай ответ в три раза длиннее",
    ],
)
def test_explicit_topics_and_other_tasks_are_not_blocked(text):
    assert not needs_essay_topic(text, [])


@pytest.mark.parametrize(
    "text",
    [
        "на 10000 слов",
        "10000",
        "на русском языке",
        "огромное",
        "любую тему",
        "в три раза длиннее",
        "в 3 раза длиннее",
    ],
)
def test_pending_clarification_keeps_asking_when_topic_still_absent(text):
    history = [
        {"role": "user", "content": "Напиши эссе"},
        {"role": "assistant", "content": ESSAY_TOPIC_QUESTION},
    ]
    assert needs_essay_topic(text, history)


def test_pending_clarification_accepts_topic_and_establishes_context():
    history = [
        {"role": "user", "content": "Напиши эссе"},
        {"role": "assistant", "content": ESSAY_TOPIC_QUESTION},
    ]
    assert not needs_essay_topic("О роли образования", history)
    history.extend(
        [
            {"role": "user", "content": "О роли образования"},
            {"role": "assistant", "content": "Образование помогает людям развиваться."},
        ]
    )
    assert not needs_essay_topic("Напиши первую часть эссе на 500 слов", history)


def test_only_recent_essay_topic_is_reused():
    history = [
        {"role": "user", "content": "Напиши эссе о Python"},
        {"role": "assistant", "content": "Python — язык программирования."},
    ]
    assert not needs_essay_topic("Напиши эссе на 500 слов", history)
    history.extend(
        [
            {"role": "user", "content": "Сколько будет 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    )
    assert needs_essay_topic("Напиши эссе на 500 слов", history)
