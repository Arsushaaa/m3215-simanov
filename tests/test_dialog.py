import json
from dataclasses import replace

import pytest

from app.dialog import TRUNCATED_NOTICE, DialogService, build_context, estimate_tokens
from app.llm import Completion, ModelError, RoutingCompletion
from app.prompts import PROMPTS


def quiz_reply(text, *, kind="question", truncated=False):
    return RoutingCompletion(
        json.dumps({"kind": kind, "feedback": "", "text": text}, ensure_ascii=False),
        [],
        truncated=truncated,
    )


async def test_history_roles_order_and_user_isolation(dialog_setup):
    service, repo, llm = dialog_setup
    await service.respond(1, 10, "Первый вопрос")
    await service.respond(2, 10, "Чужой вопрос")
    await service.respond(1, 11, "Уточнение")
    messages, temperature = llm.complete.call_args.args
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(PROMPTS["study"])
    assert "2048 токенов" in messages[0]["content"]
    assert messages[1:] == [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Ответ модели."},
        {"role": "user", "content": "Уточнение"},
    ]
    assert temperature == 0.3
    assert await repo.history(2, 20) == [
        {"role": "user", "content": "Чужой вопрос"},
        {"role": "assistant", "content": "Ответ модели."},
    ]


async def test_duplicate_message_reuses_successful_answer(dialog_setup):
    service, repo, llm = dialog_setup
    first = await service.respond(1, 5, "Привет")
    second = await service.respond(1, 5, "Привет")
    assert first == second == "Ответ модели."
    llm.complete.assert_awaited_once()
    assert len(await repo.history(1, 20)) == 2


async def test_mode_clears_own_history_preserves_temperature_and_changes_prompt(dialog_setup):
    service, repo, llm = dialog_setup
    await service.respond(1, 1, "Старый контекст")
    await service.respond(2, 1, "Чужой контекст")
    await service.preferences(1, "0.7")
    reply = await service.mode(1, "translate")
    assert reply == "Режим: Перевод. История очищена. Укажи язык перевода и отправь текст."
    assert await repo.history(1, 20) == []
    assert len(await repo.history(2, 20)) == 2
    await service.respond(1, 2, "На английский: привет")
    messages, temperature = llm.complete.call_args.args
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(PROMPTS["translate"])
    assert messages[1:] == [{"role": "user", "content": "На английский: привет"}]
    assert temperature == 0.7


async def test_reset_clears_only_history(dialog_setup):
    service, repo, _ = dialog_setup
    await service.mode(1, "quiz")
    await service.preferences(1, "1.0")
    await service.respond(1, 1, "Python")
    await service.respond(2, 1, "SQL")
    assert await service.reset(1) == "История очищена. Режим и температура сохранены."
    user = await repo.user(1)
    assert (user.mode, user.temperature) == ("quiz", 1.0)
    assert await repo.history(1, 20) == []
    assert len(await repo.history(2, 20)) == 2


@pytest.mark.parametrize(
    "value", ["0", "0.0", "0.3", "0.7", "1", "1.0", "0,3", "0.55", "0,125", "0.123456789"]
)
async def test_settings_accepts_allowed_values(dialog_setup, value):
    service, repo, llm = dialog_setup
    reply = await service.preferences(1, value)
    assert (await repo.user(1)).temperature == float(value.replace(",", "."))
    assert "Модель: nvidia/nemotron-3-super-120b-a12b:free" in reply
    assert f"Температура: {float(value.replace(',', '.'))}" in reply
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "value", ["-0.001", "1.001", "-1", "2", "NaN", "sNaN", "Infinity", "0.3 extra"]
)
async def test_settings_rejects_bad_values_without_api_or_state_change(dialog_setup, value):
    service, repo, llm = dialog_setup
    await service.preferences(1, "0.7")
    assert await service.preferences(1, value) == (
        "Укажи число от 0 до 1 включительно, например /settings 0.55."
    )
    assert (await repo.user(1)).temperature == 0.7
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "question",
    [
        "дай мне эссе о Python на 10000 слов",
        "Напиши эссе о Python на 10 000 слов",
        "Составь текст объёмом 10к слов",
        "Напиши рассказ на 10000 символов",
        "Write an essay about Python of 10000 words",
        "напиши мне огромное эссе о Python, насколько большое, насколько ты можешь",
        "Напиши самое большое эссе о Python, какое сможешь",
        "Напиши максимально длинный ответ",
    ],
)
async def test_excessive_output_is_rejected_before_model_call(dialog_setup, question):
    service, repo, llm = dialog_setup
    await service.respond(1, 1, "Первый вопрос")
    before = await repo.history(1, 20)
    llm.complete.reset_mock()
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        await service.respond(1, 2, question)
    llm.complete.assert_not_awaited()
    assert await repo.history(1, 20) == before


@pytest.mark.parametrize(
    "question",
    [
        "Напиши первую часть эссе о Python на 500 слов",
        "Как подсчитать символы в тексте на 10000 слов?",
        "Объясни обработку строки из 10000 символов",
    ],
)
async def test_reasonable_output_and_input_size_discussion_are_allowed(dialog_setup, question):
    service, _, llm = dialog_setup
    await service.respond(1, 1, question)
    llm.complete.assert_awaited_once()


async def test_relative_request_is_estimated_from_last_answer_before_model_call(dialog_setup):
    service, repo, llm = dialog_setup
    llm.complete.return_value = Completion("Пример текста. " * 250)
    await service.respond(1, 1, "Напиши эссе о Python")
    before = await repo.history(1, 20)
    llm.complete.reset_mock()
    with pytest.raises(ModelError, match="не поместится в один ответ"):
        await service.respond(
            1, 2, "напиши мне такой же ответ, только в три раза длиннее и на русском языке"
        )
    llm.complete.assert_not_awaited()
    assert await repo.history(1, 20) == before


async def test_small_relative_request_can_be_generated(dialog_setup):
    service, _, llm = dialog_setup
    await service.respond(1, 1, "Напиши небольшой пример")
    await service.respond(1, 2, "Сделай в 3 раза длиннее")
    assert llm.complete.await_count == 2


async def test_truncated_text_is_saved_and_can_continue_after_service_restart(dialog_setup):
    service, repo, llm = dialog_setup
    partial = "### Эссе\n\nПервая часть и незавершённое предложение"
    llm.complete.return_value = Completion(partial, truncated=True)
    reply = await service.respond(1, 1, "Напиши эссе о Python")
    assert reply == TRUNCATED_NOTICE + "\n\n" + partial
    assert await repo.has_incomplete_answer(1)
    assert await repo.answer(1, 1) == reply
    assert (await repo.history(1, 20))[-1]["content"] == reply
    assert llm.complete.await_count == 1
    assert await service.respond(1, 1, "Напиши эссе о Python") == reply
    assert llm.complete.await_count == 1

    service = DialogService(repo, llm, service.settings)
    llm.complete.return_value = Completion("Продолжение и завершение эссе.")
    assert await service.respond(1, 2, "Продолжи") == "Продолжение и завершение эссе."
    messages = llm.complete.call_args.args[0]
    assert messages[2]["content"] == reply
    assert "Продолжи его с места остановки" in messages[0]["content"]
    assert not await repo.has_incomplete_answer(1)


async def test_continuation_preserves_tail_when_partial_answer_exceeds_context(dialog_setup):
    service, _, llm = dialog_setup
    service.settings = replace(service.settings, history_max_tokens=1600)
    tail = "ПОСЛЕДНЯЯ ФРАЗА"
    llm.complete.return_value = Completion("я" * 15000 + tail, truncated=True)
    await service.respond(1, 1, "Напиши эссе о Python")
    llm.complete.return_value = Completion("Продолжение")
    await service.respond(1, 2, "Продолжи")
    messages = llm.complete.call_args.args[0]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"].endswith(tail)
    assert messages[-3] == {"role": "user", "content": "Напиши эссе о Python"}
    assert estimate_tokens(messages) <= 1600


async def test_incomplete_quiz_question_does_not_advance_and_continuation_keeps_topic(dialog_setup):
    service, repo, llm = dialog_setup
    await service.mode(1, "quiz")
    llm.decide.return_value = quiz_reply("Незаконченный вопрос о циклах", truncated=True)
    reply = await service.respond(1, 1, "Тема: циклы Python")
    assert reply.startswith(TRUNCATED_NOTICE)
    user = await repo.user(1)
    assert user.quiz_question == 0 and user.quiz_topic == "Тема: циклы Python"
    llm.decide.return_value = quiz_reply("Какие значения выдаёт range(3)?")
    await service.respond(1, 2, "Продолжи")
    user = await repo.user(1)
    assert user.quiz_question == 1 and user.quiz_topic == "Тема: циклы Python"
    assert "Не считай слово «Продолжи» ответом" in llm.decide.call_args.args[0][0]["content"]


async def test_model_failure_does_not_store_a_partial_exchange(dialog_setup):
    service, repo, llm = dialog_setup
    await service.respond(1, 1, "Первый вопрос")
    before = await repo.history(1, 20)
    llm.complete.side_effect = ModelError("Модель не успела ответить.")
    with pytest.raises(ModelError, match="не успела"):
        await service.respond(1, 2, "Второй вопрос")
    assert await repo.history(1, 20) == before


async def test_service_recreation_restores_settings_and_history(dialog_setup):
    service, repo, llm = dialog_setup
    await service.mode(1, "quiz")
    await service.preferences(1, "0.0")
    await service.respond(1, 1, "Тема Python")
    recreated = DialogService(repo, llm, service.settings)
    await recreated.respond(1, 2, "Мой ответ")
    messages, temperature = llm.decide.call_args.args
    assert messages[0]["content"].startswith(PROMPTS["quiz"])
    assert "Сейчас пользователь отвечает на вопрос 1 из пяти" in messages[0]["content"]
    assert "вопрос 2 из пяти" in messages[0]["content"]
    assert messages[1]["content"] == "Тема Python"
    assert temperature == 0.0


def test_context_trims_oldest_complete_pairs_and_preserves_current(bot_settings):
    settings = replace(bot_settings, history_max_messages=4)
    history = [
        {"role": "user", "content": "Старый вопрос"},
        {"role": "assistant", "content": "Старый ответ"},
        {"role": "user", "content": "Новый вопрос"},
        {"role": "assistant", "content": "Новый ответ"},
    ]
    result = build_context("Инструкция", history, "Текущий вопрос", settings)
    assert result == [
        {"role": "system", "content": "Инструкция"},
        *history[2:],
        {"role": "user", "content": "Текущий вопрос"},
    ]
    assert len(history) == 4


def test_context_obeys_estimated_token_budget(bot_settings):
    history = [
        {"role": "user", "content": "ж" * 1200},
        {"role": "assistant", "content": "я" * 1200},
        {"role": "user", "content": "Недавний вопрос"},
        {"role": "assistant", "content": "Недавний ответ"},
    ]
    settings = replace(bot_settings, history_max_tokens=100)
    result = build_context("Инструкция", history, "Текущий вопрос", settings)
    assert result[1:-1] == history[2:]
    assert result[-1]["content"] == "Текущий вопрос"
    assert estimate_tokens(result) <= 100


def test_oversized_current_message_is_rejected_without_removing_system(bot_settings):
    settings = replace(bot_settings, history_max_tokens=20)
    with pytest.raises(ModelError, match="слишком длинное"):
        build_context("Инструкция", [], "д" * 200, settings)


async def test_concurrent_reset_from_another_instance_discards_stale_answer(dialog_setup):
    service, repo, llm = dialog_setup

    async def complete(*args):
        await repo.reset(1)
        return Completion("Устаревший ответ")

    llm.complete.side_effect = complete
    with pytest.raises(ModelError, match="Диалог изменился"):
        await service.respond(1, 1, "Вопрос")
    assert await repo.history(1, 20) == []


async def test_quiz_has_five_questions_hint_keeps_progress_and_restart_restores_it(dialog_setup):
    service, repo, llm = dialog_setup
    await service.mode(1, "quiz")
    answer = await service.respond(1, 1, "Тема: Python")
    assert answer.startswith("Вопрос 1/5\n\n")
    llm.decide.return_value = quiz_reply("Вспомни изменяемость типов.", kind="hint")
    hint = await service.respond(1, 2, "Дай подсказку")
    assert hint.startswith("Подсказка к вопросу 1/5\n\n")
    assert (await repo.user(1)).quiz_question == 1
    assert "не задавая новый вопрос" in llm.decide.call_args.args[0][0]["content"]
    service = DialogService(repo, llm, service.settings)
    llm.decide.return_value = quiz_reply("Чем отличаются список и кортеж?")
    for number in range(2, 6):
        answer = await service.respond(1, number + 1, "Ответ на предыдущий вопрос")
        assert answer.startswith(f"Вопрос {number}/5\n\n")
    llm.decide.return_value = quiz_reply("Повтори изменяемость типов.", kind="finished")
    finished = await service.respond(1, 7, "Ответ на пятый вопрос")
    assert finished.startswith("Тренировка завершена.\n\n")
    assert (await repo.user(1)).quiz_finished is True
    calls = llm.decide.await_count
    with pytest.raises(ModelError, match="Для новой тренировки"):
        await service.respond(1, 8, "Дальше")
    assert llm.decide.await_count == calls
    await service.reset(1)
    user = await repo.user(1)
    assert (user.mode, user.quiz_question, user.quiz_topic, user.quiz_finished) == (
        "quiz",
        0,
        None,
        False,
    )


async def test_quiz_topic_remains_in_context_after_history_trimming(dialog_setup):
    service, repo, llm = dialog_setup
    service.settings = replace(service.settings, history_max_messages=4)
    await service.mode(1, "quiz")
    await service.respond(1, 1, "Тема: циклы Python")
    await service.respond(1, 2, "Первый ответ")
    await service.respond(1, 3, "Второй ответ")
    messages = llm.decide.call_args.args[0]
    assert messages[1] == {"role": "user", "content": "Тема: циклы Python"}
    assert messages[2]["content"] == "Первый ответ"
    assert messages[3]["role"] == "assistant"
    assert messages[4]["content"] == "Второй ответ"
    assert len(messages) - 1 == 4
