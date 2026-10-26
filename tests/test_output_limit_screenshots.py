import pytest

from app.dialog import DialogService
from app.essay import ESSAY_TOPIC_QUESTION
from app.llm import ModelError
from app.output_limits import check_output_size

SCREENSHOT_REQUEST = (
    "О C++, на огромное количество слов. Когда достигнешь лимита, "
    "автоматически продолжай новыми сообщениями без моего участия."
)


@pytest.mark.parametrize(
    "text",
    [
        SCREENSHOT_REQUEST,
        "C++, на огромное количество слов. Самостоятельно продолжай после лимита.",
        "О C++, на 10000 слов. Продолжай новыми сообщениями без моего участия.",
        "Про C++, на 10000 символов. Автоматически продолжай после лимита.",
        "About C++, with an enormous number of words. Automatically continue at the limit.",
    ],
)
def test_implicit_topic_with_automatic_continuation_is_bounded(text):
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        check_output_size(text, 2048)


@pytest.mark.parametrize(
    "text",
    [
        f"Переведи на английский: {SCREENSHOT_REQUEST}",
        f"На английский: {SCREENSHOT_REQUEST}",
        f"Проанализируй запрос: {SCREENSHOT_REQUEST}",
        f"Объясни, почему опасен запрос: {SCREENSHOT_REQUEST}",
        f"Напиши перевод фразы «{SCREENSHOT_REQUEST}»",
        f"Проверь этот текст: `{SCREENSHOT_REQUEST}`",
        "Напиши программу, которая выводит огромное эссе из файла. "
        "Если файл длинный, автоматически продолжай обработку следующими частями.",
        "Напиши короткий ответ о тексте на 10000 слов. "
        "Автоматически продолжай, если не поместится.",
        "Как обработать строку из 10000 символов в Python? "
        "Ответь кратко, до 100 слов, с одним примером.",
    ],
)
def test_automatic_continuation_in_input_or_program_is_not_output_size(text):
    check_output_size(text, 2048)


async def test_exact_screenshot_does_not_spend_llm_or_create_followup_context(dialog_setup):
    service, repository, llm = dialog_setup
    await service.respond(1, 1, "Покажи короткий пример кода Python")
    history = await repository.history(1, 20)
    llm.complete.reset_mock()

    with pytest.raises(ModelError, match="не поместится в один ответ"):
        await service.respond(1, 2, SCREENSHOT_REQUEST)

    assert await repository.history(1, 20) == history
    llm.complete.assert_not_awaited()
    llm.decide.assert_not_awaited()


async def test_topic_clarification_with_comma_is_bounded_after_service_restart(dialog_setup):
    service, repository, llm = dialog_setup
    assert await service.respond(1, 1, "Напиши эссе") == ESSAY_TOPIC_QUESTION
    restarted = DialogService(repository, llm, service.settings)

    with pytest.raises(ModelError, match="не поместится в один ответ"):
        await restarted.respond(1, 2, SCREENSHOT_REQUEST)

    llm.complete.assert_not_awaited()
    await restarted.respond(1, 3, "О параллельности C++, первую часть на 300 слов")
    llm.complete.assert_awaited_once()


async def test_translation_mode_keeps_literal_implicit_output_request(dialog_setup):
    service, _, llm = dialog_setup
    await service.mode(1, "translate")
    await service.respond(1, 1, SCREENSHOT_REQUEST)

    llm.complete.assert_awaited_once()
    assert llm.complete.call_args.args[0][-1]["content"] == SCREENSHOT_REQUEST
