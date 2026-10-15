import pytest

from app.llm import ModelError
from app.output_limits import LARGE_OUTPUT, check_output_size


@pytest.mark.parametrize(
    "text",
    [
        "напиши мне эссе по теме c++ на огромное количество слов",
        "напиши мне огромное эссе",
        "Напиши эссе о C++ на очень большое число слов",
        "Напиши эссе о C++, используй как можно больше слов",
        "Напиши эссе о C++, пусть в нём будет очень много слов",
        "Напиши очень много текста о C++",
        "Напиши огромный текст о C++",
        "Напиши максимально длинный ответ",
        "Напиши самое большое эссе о C++, какое сможешь",
        "Напиши эссе о C++ на 10000 слов",
        "Составь текст объёмом 10к слов",
        "Напиши на 10000 слов",
        "Можешь, пожалуйста, написать эссе о C++ на огромное количество слов?",
        "Write a huge essay about C++",
        "Please write an essay about C++ with an enormous number of words",
        "Can you write an essay about C++ with a huge amount of words?",
        "Generate an essay about C++ with as many words as possible",
        "Write an essay about C++ as long as possible",
        "Write an essay about C++ of 10000 words",
    ],
)
def test_requested_excessive_output_is_rejected(text):
    with pytest.raises(ModelError) as error:
        check_output_size(text, 2048)
    assert str(error.value) == LARGE_OUTPUT


@pytest.mark.parametrize(
    "text",
    [
        "Напиши эссе о C++ на 500 слов",
        "Напиши первую часть эссе о C++ на 500 слов",
        "Write a short essay about C++ of 500 words",
        "Напиши эссе на тему огромного количества слов в словаре",
        "Напиши эссе о гигантских текстах",
        "Напиши текст про огромное количество слов в словаре",
        "Напиши функцию для обработки текста на 10000 слов",
        "Напиши программу, которая выводит огромное эссе из файла",
        "Объясни обработку строки из 10000 символов",
        "Переведи: напиши эссе на 10000 слов",
        'Проанализируй запрос "напиши мне огромное эссе"',
        'Напиши перевод фразы "напиши эссе на 10000 слов"',
        "Напиши ответ на запрос «Напиши эссе на 10000 слов»",
        "Напиши эссе на тему «Огромное количество слов»",
        "Напиши текст по образцу: `напиши эссе на 10000 слов`",
        "Напиши текст по образцу:\n```\nнапиши огромное эссе\n```",
        "Напиши ответ о тексте на 10000 слов",
        "Напиши краткий ответ для входного текста на 10000 слов",
        "Write a response about a huge text of 10000 words",
    ],
)
def test_reasonable_output_input_size_and_quoted_requests_are_allowed(text):
    check_output_size(text, 2048, previous_answer="Длинный предыдущий ответ. " * 500)


@pytest.mark.parametrize(
    "text",
    [
        "О C++ на огромное количество слов",
        "О C++ на 10000 слов",
        "C++, как можно больше слов",
    ],
)
def test_pending_essay_topic_reply_still_checks_requested_size(text):
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        check_output_size(text, 2048, output_followup=True)
    # Отдельная фраза с темой без подтверждённого контекста не считается приказом.
    check_output_size(text, 2048)


def test_pending_essay_topic_reply_allows_bounded_first_part():
    check_output_size("О C++, первую часть на 500 слов", 2048, output_followup=True)


def test_relative_expansion_is_checked_before_model_and_does_not_affect_translation():
    previous = "Длинный ответ. " * 500
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        check_output_size("Сделай в 3 раза длиннее", 2048, previous)
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        check_output_size("В три раза длиннее", 2048, previous)
    check_output_size("Переведи: сделай в 3 раза длиннее", 2048, previous)
    check_output_size('Напиши перевод фразы "Сделай в 3 раза длиннее"', 2048, previous)
    check_output_size("Сделай в 3 раза длиннее", 2048, "Короткий ответ.")
