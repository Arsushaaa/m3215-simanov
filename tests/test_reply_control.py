import pytest

from app.dialog import DialogService
from app.essay import ESSAY_TOPIC_QUESTION
from app.llm import ModelError, RoutingCompletion


async def test_bare_10000_word_essay_asks_topic_without_calling_model(dialog_setup):
    service, repo, llm = dialog_setup
    assert await service.respond(1, 1, "напиши эссе на 10000 слов") == ESSAY_TOPIC_QUESTION
    llm.complete.assert_not_awaited()
    assert await repo.answer(1, 1) == ESSAY_TOPIC_QUESTION
    restarted = DialogService(repo, llm, service.settings)
    assert await restarted.respond(1, 2, "на русском языке") == ESSAY_TOPIC_QUESTION
    llm.complete.assert_not_awaited()
    await restarted.respond(1, 3, "О роли образования, небольшой текст")
    llm.complete.assert_awaited_once()
    assert llm.complete.call_args.args[0][-1]["content"] == "О роли образования, небольшой текст"


async def test_essay_topic_guard_does_not_interfere_with_translation_or_quiz(dialog_setup):
    service, _, llm = dialog_setup
    await service.mode(1, "translate")
    await service.respond(1, 1, "На английский: напиши эссе")
    await service.mode(1, "quiz")
    llm.decide.return_value = RoutingCompletion(
        '{"kind":"clarify","feedback":"","text":"Назови тему по программированию."}', []
    )
    reply = await service.respond(1, 2, "Тема: как написать эссе")
    assert reply == "Уточнение для тренировки\n\nНазови тему по программированию."
    llm.complete.assert_awaited_once()
    llm.decide.assert_awaited_once()


async def test_huge_volume_after_topic_clarification_is_checked_without_model(dialog_setup):
    service, repo, llm = dialog_setup
    assert await service.respond(1, 1, "Напиши мне огромное эссе") == ESSAY_TOPIC_QUESTION
    before = await repo.history(1, 20)
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        await service.respond(1, 2, "О C++ на огромное количество слов")
    assert await repo.history(1, 20) == before
    llm.complete.assert_not_awaited()
    await service.respond(1, 3, "О C++ на 300 слов")
    llm.complete.assert_awaited_once()


async def test_translation_of_large_output_request_does_not_trigger_volume_guard(dialog_setup):
    service, _, llm = dialog_setup
    await service.mode(1, "translate")
    await service.respond(
        1, 1, "На английский: «напиши мне эссе по теме C++ на огромное количество слов»"
    )
    llm.complete.assert_awaited_once()
