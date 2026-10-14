from app.dialog import DialogService
from app.essay import ESSAY_TOPIC_QUESTION


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
    await service.respond(1, 2, "Тема: как написать эссе")
    assert llm.complete.await_count == 2
