"""Регрессии явных переходов тренировки без дополнительных LLM-запросов."""

import json
from dataclasses import replace

import pytest

from app.dialog import TRUNCATED_NOTICE, DialogService, build_context
from app.llm import ModelError, RoutingCompletion


def quiz_turn(
    kind="question", text="Чем отличаются список и кортеж?", *, feedback="", truncated=False
):
    return RoutingCompletion(
        json.dumps({"kind": kind, "feedback": feedback, "text": text}, ensure_ascii=False),
        [],
        truncated=truncated,
    )


def quiz_state(user):
    return user.quiz_question, user.quiz_topic, user.quiz_finished


async def start_quiz(dialog_setup, question=0, *, user_id=1):
    service, repo, llm = dialog_setup
    await service.mode(user_id, "quiz")
    if question:
        user = await repo.user(user_id)
        repo.users[user_id] = replace(user, quiz_question=question, quiz_topic="Python")
    llm.decide.return_value = quiz_turn()
    return service, repo, llm


async def test_rejected_personal_topic_then_python_starts_question_one(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    llm.decide.return_value = quiz_turn(
        "clarify", "Я не знаю твоего имени. Назови тему по программированию."
    )
    clarification = await service.respond(1, 1, "мое имя")
    assert "Уточнение для тренировки" in clarification
    assert "Вопрос" not in clarification
    assert quiz_state(await repo.user(1)) == (0, None, False)

    llm.decide.return_value = quiz_turn(text="Вопрос 1/5\n\nДля чего нужен список в Python?")
    first = await service.respond(1, 2, "python")
    assert first == "Вопрос 1/5\n\nДля чего нужен список в Python?"
    assert quiz_state(await repo.user(1)) == (1, "python", False)

    llm.decide.return_value = quiz_turn(
        text="Что делает len?", feedback="Верно, список хранит элементы."
    )
    second = await service.respond(1, 3, "Хранит элементы")
    assert second.index("Верно, список хранит элементы.") < second.index("Вопрос 2/5")
    assert second.count("Вопрос") == 1
    assert quiz_state(await repo.user(1)) == (2, "python", False)
    assert llm.decide.await_count == 3
    llm.complete.assert_not_awaited()
    for call in llm.decide.call_args_list:
        assert len(call.args) == 2
        assert call.kwargs["decision_schema"]


@pytest.mark.parametrize("kind", ["clarify", "hint"])
async def test_non_question_turn_does_not_advance_or_replace_topic(dialog_setup, kind):
    service, repo, llm = await start_quiz(dialog_setup, 2)
    llm.decide.return_value = quiz_turn(kind, "Попробуй уточнить свой ответ.")
    await service.respond(1, 20, "А можно подробнее?")
    assert quiz_state(await repo.user(1)) == (2, "Python", False)
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


async def test_fifth_question_allows_hint_and_clarification_before_final_answer(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup, 5)
    llm.decide.return_value = quiz_turn("hint", "Вспомни, изменяем ли этот тип.")
    hint = await service.respond(1, 1, "Дай подсказку")
    assert "Подсказка к вопросу 5/5" in hint
    assert quiz_state(await repo.user(1)) == (5, "Python", False)

    llm.decide.return_value = quiz_turn("clarify", "Что ты имеешь в виду под копией?")
    await service.respond(1, 2, "Наверное, копия?")
    assert quiz_state(await repo.user(1)) == (5, "Python", False)

    llm.decide.return_value = quiz_turn(
        "finished", "Повтори изменяемость типов.", feedback="Верно."
    )
    final = await service.respond(1, 3, "Кортеж неизменяем")
    assert "Тренировка завершена." in final
    assert "Вопрос 6" not in final
    assert quiz_state(await repo.user(1)) == (5, "Python", True)
    before = llm.decide.await_count
    with pytest.raises(ModelError, match="Для новой тренировки"):
        await service.respond(1, 4, "Дальше")
    assert llm.decide.await_count == before
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "question,kind", [(0, "hint"), (0, "finished"), (2, "finished"), (5, "question")]
)
async def test_invalid_state_transition_never_changes_history_or_progress(
    dialog_setup, question, kind
):
    service, repo, llm = await start_quiz(dialog_setup, question)
    before = await repo.user(1)
    llm.decide.return_value = quiz_turn(kind)
    with pytest.raises(ModelError):
        await service.respond(1, 1, "Ответ")
    assert await repo.user(1) == before
    assert await repo.history(1, 20) == []
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "raw,truncated",
    [
        ("Обычный текст вместо протокола.", False),
        ('{"kind":"question","feedback":"","text":"Оборванный', True),
        ('{"kind":"question","feedback":"","text":"Вопрос","counter":5}', False),
        ('{"kind":"question","kind":"clarify","feedback":"","text":"Вопрос"}', False),
        ('{"kind":"unknown","feedback":"","text":"Вопрос"}', False),
        ('{"kind":"question","feedback":"","text":""}', False),
        ('{"kind":"question","text":"Вопрос"}', False),
        ('{"kind":"question","feedback":null,"text":"Вопрос"}', False),
        ('[{"kind":"question","feedback":"","text":"Вопрос"}]', False),
    ],
)
async def test_invalid_quiz_protocol_has_no_prose_fallback_or_retry(dialog_setup, raw, truncated):
    service, repo, llm = await start_quiz(dialog_setup)
    before = await repo.user(1)
    llm.decide.return_value = RoutingCompletion(raw, [], truncated=truncated)
    with pytest.raises(ModelError):
        await service.respond(1, 1, "Python")
    assert await repo.user(1) == before
    assert await repo.history(1, 20) == []
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


async def test_quiz_rejects_unexpected_tool_calls_without_retry(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    valid = quiz_turn()
    llm.decide.return_value = RoutingCompletion(
        valid.text,
        [
            {
                "id": "unexpected",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    )
    with pytest.raises(ModelError):
        await service.respond(1, 1, "Python")
    assert quiz_state(await repo.user(1)) == (0, None, False)
    assert await repo.history(1, 20) == []
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "heading",
    ["Вопрос 1/5", "### Вопрос 1/5", "Вопрос 1 из 5:", "**Вопрос 1/5:**", "__Вопрос 1/5:__"],
)
async def test_generated_leading_question_heading_is_removed(dialog_setup, heading):
    service, _, llm = await start_quiz(dialog_setup)
    llm.decide.return_value = quiz_turn(text=f"{heading}\n\nЧто делает print?")
    answer = await service.respond(1, 1, "Python")
    assert answer == "Вопрос 1/5\n\nЧто делает print?"


async def test_leading_question_label_preserves_bold_question_body(dialog_setup):
    service, _, llm = await start_quiz(dialog_setup)
    llm.decide.return_value = quiz_turn(text="Вопрос 1/5: **Что делает print?**")
    answer = await service.respond(1, 1, "Python")
    assert answer == "Вопрос 1/5\n\n**Что делает print?**"


@pytest.mark.parametrize(
    "kind,question,label,heading",
    [
        ("clarify", 0, "**Уточнение для тренировки:**", "Уточнение для тренировки"),
        ("finished", 5, "**Тренировка завершена.**", "Тренировка завершена."),
    ],
)
async def test_control_label_is_rendered_once_without_broken_bold(
    dialog_setup, kind, question, label, heading
):
    service, _, llm = await start_quiz(dialog_setup, question)
    llm.decide.return_value = quiz_turn(kind, text=f"{label}\n\nТекст ответа.")
    reply = await service.respond(1, 1, "Python")
    assert reply == f"{heading}\n\nТекст ответа."


async def test_second_question_heading_inside_body_is_rejected(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    llm.decide.return_value = quiz_turn(text="Что делает print?\n\nВопрос 2/5\n\nА что делает len?")
    with pytest.raises(ModelError):
        await service.respond(1, 1, "Python")
    assert quiz_state(await repo.user(1)) == (0, None, False)
    assert await repo.history(1, 20) == []


@pytest.mark.parametrize(
    "kind,question,header,final_question,finished",
    [
        ("question", 0, "Вопрос 1/5", 1, False),
        ("question", 2, "Вопрос 3/5", 3, False),
        ("hint", 2, "Подсказка к вопросу 2/5", 2, False),
        ("clarify", 0, "Уточнение для тренировки", 0, False),
        ("clarify", 2, "Уточнение для тренировки", 2, False),
        ("finished", 5, "Тренировка завершена.", 5, True),
    ],
)
async def test_truncated_turn_restores_its_kind_after_service_restart(
    dialog_setup, kind, question, header, final_question, finished
):
    service, repo, llm = await start_quiz(dialog_setup, question)
    llm.decide.return_value = quiz_turn(
        kind, "Незаконченная часть", feedback="Обратная связь.", truncated=True
    )
    partial = await service.respond(1, 1, "Python")
    assert partial.startswith(TRUNCATED_NOTICE)
    assert header in partial
    topic = "Python" if question or kind == "question" else None
    assert quiz_state(await repo.user(1)) == (question, topic, False)
    assert await repo.has_incomplete_answer(1)

    service = DialogService(repo, llm, service.settings)
    llm.decide.return_value = quiz_turn(kind, "Завершённая часть")
    continued = await service.respond(1, 2, "Продолжи")
    assert header in continued
    assert TRUNCATED_NOTICE not in continued
    assert quiz_state(await repo.user(1)) == (final_question, topic, finished)
    assert not await repo.has_incomplete_answer(1)
    assert llm.decide.await_count == 2
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "pending,question,wrong",
    [
        ("question", 0, "clarify"),
        ("hint", 2, "question"),
        ("clarify", 2, "question"),
        ("finished", 5, "hint"),
    ],
)
async def test_continuation_cannot_change_pending_kind(dialog_setup, pending, question, wrong):
    service, repo, llm = await start_quiz(dialog_setup, question)
    llm.decide.return_value = quiz_turn(pending, truncated=True)
    await service.respond(1, 1, "Python")
    before = await repo.user(1)
    history = await repo.history(1, 20)
    service = DialogService(repo, llm, service.settings)
    llm.decide.return_value = quiz_turn(wrong)
    with pytest.raises(ModelError):
        await service.respond(1, 2, "Продолжи")
    assert await repo.user(1) == before
    assert await repo.history(1, 20) == history
    assert await repo.has_incomplete_answer(1)
    assert llm.decide.await_count == 2
    llm.complete.assert_not_awaited()


async def test_duplicate_update_reuses_accepted_quiz_turn_without_new_llm_call(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    first = await service.respond(1, 10, "Python")
    before = await repo.user(1)
    assert await service.respond(1, 10, "Python") == first
    assert await repo.user(1) == before
    assert len(await repo.history(1, 20)) == 2
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


async def test_quiz_reset_during_decision_discards_stale_turn(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)

    async def decide(*args, **kwargs):
        await repo.reset(1)
        return quiz_turn()

    llm.decide.side_effect = decide
    with pytest.raises(ModelError, match="Диалог изменился"):
        await service.respond(1, 1, "Python")
    assert quiz_state(await repo.user(1)) == (0, None, False)
    assert await repo.history(1, 20) == []
    llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


async def test_clarification_and_pending_continuation_are_isolated_by_owner(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    await service.mode(2, "quiz")
    llm.decide.return_value = quiz_turn(truncated=True)
    await service.respond(1, 1, "Python")
    alice = await repo.user(1)
    llm.decide.return_value = quiz_turn("clarify", "Назови тему по программированию.")
    await service.respond(2, 1, "мое имя")
    assert quiz_state(await repo.user(2)) == (0, None, False)
    assert await repo.user(1) == alice
    assert await repo.has_incomplete_answer(1)
    assert not await repo.has_incomplete_answer(2)

    llm.decide.return_value = quiz_turn(text="Что такое SELECT?")
    await service.respond(2, 2, "SQL")
    bob = await repo.user(2)
    assert quiz_state(bob) == (1, "SQL", False)
    service = DialogService(repo, llm, service.settings)
    llm.decide.return_value = quiz_turn(text="Что такое переменная?")
    await service.respond(1, 2, "Продолжи")
    assert quiz_state(await repo.user(1)) == (1, "Python", False)
    assert await repo.user(2) == bob
    assert "SELECT" not in (await repo.history(1, 20))[-1]["content"]
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("truncated", [False, True])
async def test_unclosed_feedback_code_cannot_hide_program_question_heading(
    dialog_setup, fence, truncated
):
    service, repo, llm = await start_quiz(dialog_setup)
    feedback = f"{fence}python\nprint('Вопрос внутри кода')"
    llm.decide.return_value = quiz_turn(
        feedback=feedback, text="Что выведет этот код?", truncated=truncated
    )
    answer = await service.respond(1, 1, "Python")
    assert feedback in answer
    assert answer.count("Вопрос 1/5") == 1
    before_heading = answer.split("Вопрос 1/5", 1)[0]
    assert before_heading.rstrip().endswith(fence)
    assert before_heading.rfind(fence) > before_heading.index("print(")
    assert quiz_state(await repo.user(1)) == (0 if truncated else 1, "Python", False)

    if truncated:
        assert answer.startswith(TRUNCATED_NOTICE)
        assert await repo.has_incomplete_answer(1)
        service = DialogService(repo, llm, service.settings)
        llm.decide.return_value = quiz_turn(text="Какой результат у print?")
        continued = await service.respond(1, 2, "Продолжи")
        assert continued.startswith("Вопрос 1/5\n\n")
        assert quiz_state(await repo.user(1)) == (1, "Python", False)
        assert not await repo.has_incomplete_answer(1)
        assert llm.decide.await_count == 2
    else:
        assert TRUNCATED_NOTICE not in answer
        llm.decide.assert_awaited_once()
    llm.complete.assert_not_awaited()


async def test_numbered_text_inside_code_is_preserved_in_question_and_continuation(dialog_setup):
    service, repo, llm = await start_quiz(dialog_setup)
    body = "Что выведет код?\n\n```text\nВопрос 4/5\n```\n\nОбъясни результат."
    llm.decide.return_value = quiz_turn(text=body, truncated=True)
    answer = await service.respond(1, 1, "Строки Python")
    assert answer.endswith(body)
    assert quiz_state(await repo.user(1)) == (0, "Строки Python", False)
    service = DialogService(repo, llm, service.settings)
    llm.decide.return_value = quiz_turn(text="Продолжение того же вопроса.")
    await service.respond(1, 2, "Продолжи")
    assert quiz_state(await repo.user(1)) == (1, "Строки Python", False)


def test_schema_is_included_in_context_budget(bot_settings):
    settings = replace(bot_settings, history_max_tokens=200)
    assert build_context("Инструкция", [], "Python", settings)
    with pytest.raises(ModelError, match="контекста"):
        build_context("Инструкция", [], "Python", settings, schema={"description": "x" * 1000})
