from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.dialog import estimate_tokens
from app.llm import Completion, ModelError
from app.output_limits import output_target


async def test_truncated_response_continues_automatically_and_saves_one_exchange(dialog_setup):
    service, repo, llm = dialog_setup
    llm.complete.side_effect = [
        Completion("Первая часть", truncated=True),
        Completion("Конец ответа"),
    ]
    sent = AsyncMock()
    answer = await service.respond(1, 1, "Напиши эссе", on_part=sent)
    assert answer == "Первая часть\n\nКонец ответа"
    assert [call.args[0] for call in sent.await_args_list] == ["Первая часть", "Конец ответа"]
    assert llm.complete.await_count == 2
    assert (await repo.reply(1, 1)).is_complete
    assert len(await repo.history(1, 20)) == 2
    assert llm.complete.call_args.args[0][-2]["content"] == "Первая часть"
    sent.reset_mock()
    assert await service.respond(1, 1, "Напиши эссе", on_part=sent) == answer
    assert llm.complete.await_count == 2
    assert [call.args[0] for call in sent.await_args_list] == ["Первая часть", "Конец ответа"]


async def test_explicit_long_target_continues_even_when_each_part_has_stop_reason(dialog_setup):
    service, repo, llm = dialog_setup
    service.settings = replace(
        service.settings, llm_max_output_tokens=128, llm_max_response_parts=4
    )
    texts = [f"Раздел{n} " + "слово " * 34 for n in range(3)]
    llm.complete.side_effect = [Completion(text) for text in texts]
    answer = await service.respond(1, 1, "Напиши эссе на 100 слов")
    assert llm.complete.await_count == 3
    assert answer == "\n\n".join(texts)
    assert (await repo.reply(1, 1)).is_complete


async def test_first_part_is_delivered_before_second_generation(dialog_setup):
    service, _, llm = dialog_setup
    sent = []

    async def deliver(text):
        sent.append(text)

    async def complete(messages, temperature):
        if not sent:
            return Completion("Начало", truncated=True)
        assert sent == ["Начало"]
        return Completion("Конец")

    llm.complete.side_effect = complete
    await service.respond(1, 1, "Эссе", on_part=deliver)
    assert sent == ["Начало", "Конец"]


async def test_continuation_preserves_tail_when_whole_answer_exceeds_context(dialog_setup):
    service, _, llm = dialog_setup
    service.settings = replace(service.settings, history_max_tokens=1600)
    tail = "ПОСЛЕДНЯЯ ФРАЗА"
    llm.complete.side_effect = [Completion("я" * 15000 + tail, truncated=True), Completion("Конец")]
    await service.respond(1, 1, "Напиши эссе")
    messages = llm.complete.call_args.args[0]
    assert messages[-2]["content"].endswith(tail)
    assert messages[-3] == {"role": "user", "content": "Напиши эссе"}
    assert estimate_tokens(messages) <= 1600


async def test_quota_error_preserves_and_delivers_generated_parts(dialog_setup):
    service, repo, llm = dialog_setup
    llm.complete.side_effect = [Completion("Текст", truncated=True), ModelError("Квота исчерпана.")]
    sent = AsyncMock()
    answer = await service.respond(1, 1, "Эссе", on_part=sent)
    assert answer.startswith("Текст\n\nОтвет прерван: Квота исчерпана.")
    assert [call.args[0] for call in sent.await_args_list][0] == "Текст"
    assert len(sent.await_args_list) == 2
    assert not (await repo.reply(1, 1)).is_complete
    assert "напиши «Продолжи»" not in answer
    assert llm.complete.await_count == 2


async def test_generation_cap_is_finite_and_saves_every_part(dialog_setup):
    service, repo, llm = dialog_setup
    service.settings = replace(service.settings, llm_max_response_parts=2)
    llm.complete.side_effect = [
        Completion("Раз", truncated=True),
        Completion("Два", truncated=True),
    ]
    answer = await service.respond(1, 1, "Текст")
    assert answer.startswith("Раз\n\nДва\n\nДостигнут предел")
    assert llm.complete.await_count == 2
    assert len((await repo.reply(1, 1)).parts) == 3


async def test_repeated_part_stops_without_duplicate_delivery(dialog_setup):
    service, _, llm = dialog_setup
    llm.complete.return_value = Completion("Повтор", truncated=True)
    sent = AsyncMock()
    answer = await service.respond(1, 1, "Текст", on_part=sent)
    assert llm.complete.await_count == 2
    assert [call.args[0] for call in sent.await_args_list].count("Повтор") == 1
    assert "Модель повторила" in answer


async def test_quiz_progress_advances_only_when_automatic_chain_finishes(dialog_setup):
    service, repo, llm = dialog_setup
    await service.mode(1, "quiz")
    llm.complete.side_effect = [
        Completion("Начало вопроса", truncated=True),
        Completion("Конец вопроса?"),
    ]
    progress = []

    async def deliver(text):
        progress.append((await repo.user(1)).quiz_question)

    answer = await service.respond(1, 1, "Тема: циклы", on_part=deliver)
    assert answer.startswith("Вопрос 1/5")
    assert progress == [0, 1]
    assert (await repo.user(1)).quiz_topic == "Тема: циклы"


async def test_reset_from_other_process_stops_automatic_chain_without_saving_stale_part(
    dialog_setup,
):
    service, repo, llm = dialog_setup
    sent = AsyncMock()
    calls = 0

    async def complete(messages, temperature):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Completion("Первое", truncated=True)
        await repo.reset(1)
        return Completion("Устаревшее")

    llm.complete.side_effect = complete
    with pytest.raises(ModelError, match="Диалог изменился"):
        await service.respond(1, 1, "Текст", on_part=sent)
    assert [call.args[0] for call in sent.await_args_list] == ["Первое"]
    assert await repo.history(1, 20) == []


def test_long_target_uses_total_budget_instead_of_single_generation_budget():
    assert output_target("Напиши эссе на 10000 слов", 2048 * 24).amount == 10000
    with pytest.raises(ModelError, match="предел одного задания"):
        output_target("Напиши эссе на 1000000 слов", 2048 * 24)
