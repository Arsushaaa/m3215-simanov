"""Weather output is checked against the observation, never against user keywords."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_agent import ask, native, routing
from test_agent import setup_agent as setup_agent

from app.formatting import format_reply
from app.llm import Completion
from app.mcp.contracts import WeatherAmbiguous, WeatherOk
from app.weather_reply import (
    canonical_weather,
    canonical_weather_clarification,
    checked_weather_reply,
)

OBSERVATION = {
    "status": "ok",
    "place": {"name": "Казань", "country": "Россия", "region": "Татарстан"},
    "temperature_c": 4.2,
    "condition": {"code": 0, "description": "Ясно"},
    "wind": {"speed": 3.2, "unit": "km/h"},
    "observed_at": "2026-10-28T10:15:00+03:00",
    "timezone": "Europe/Moscow",
    "source": "Open-Meteo",
}
CORRECT = (
    "**Казань**, Россия, Татарстан: **4,2 °C**, ясно. Ветер **3,2 км/ч**.\n"
    "Данные: 28 октября 2026, 10:15 (UTC+03:00), источник Open-Meteo."
)


def plain(text):
    return "".join(part.text for part in format_reply(text))


def test_correct_markdown_decimal_comma_keeps_the_real_model_reply():
    result = checked_weather_reply(CORRECT, WeatherOk.model_validate(OBSERVATION))
    assert result.text == CORRECT and not result.replaced and not result.truncated


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("4,2 °C", "4 °C"),
        lambda s: s.replace("4,2 °C", "14,2 °C"),
        lambda s: s.replace("4,2 °C", "−4,2 °C"),
        lambda s: s.replace("4,2 °C", "− 4,2 °C"),
        lambda s: s.replace("4,2 °C", "минус 4,2 °C"),
        lambda s: s.replace("3,2 км/ч", "3 км/ч"),
        lambda s: s.replace("10:15", "10:30"),
        lambda s: s.replace("28 октября", "29 октября"),
        lambda s: s.replace("UTC+03:00", "UTC+09:00"),
        lambda s: s.replace("Open-Meteo", "выдуманный источник"),
        lambda s: s.replace("Казань", "Москва"),
        lambda s: s.replace(", Татарстан", ""),
        lambda s: s.replace("Татарстан", "Кировская область"),
        lambda s: s + " Ещё 99 °C.",
    ],
)
def test_changed_observation_falls_back_without_rounded_or_invented_facts(change):
    observation = WeatherOk.model_validate(OBSERVATION)
    result = checked_weather_reply(change(CORRECT), observation)
    assert result.replaced and not result.truncated
    assert result.text == canonical_weather(observation)
    shown = plain(result.text)
    for fact in ("Казань", "Россия", "Татарстан", "4.2 °C", "3.2 км/ч", "10:15", "Open-Meteo"):
        assert fact in shown


def test_truncated_model_final_becomes_complete_card_without_continue_flag():
    result = checked_weather_reply(CORRECT, WeatherOk.model_validate(OBSERVATION), truncated=True)
    assert result.replaced and not result.truncated


def test_empty_region_does_not_require_an_extra_place_qualifier():
    value = deepcopy(OBSERVATION)
    value["place"]["region"] = ""
    text = CORRECT.replace(", Татарстан", "")
    result = checked_weather_reply(text, WeatherOk.model_validate(value))
    assert result.text == text and not result.replaced


def test_region_comparison_folds_whitespace_and_case():
    value = deepcopy(OBSERVATION)
    value["place"]["region"] = "Республика   Татарстан"
    text = CORRECT.replace("Татарстан", "республика\nТАТАРСТАН")
    result = checked_weather_reply(text, WeatherOk.model_validate(value))
    assert result.text == text and not result.replaced


def test_fallback_preserves_long_decimal_negative_value_and_source_timezone():
    value = deepcopy(OBSERVATION)
    value.update(temperature_c=-4.23456789, observed_at="2026-10-28T10:15:02.123456+03:00")
    result = checked_weather_reply("Не могу ответить", WeatherOk.model_validate(value))
    shown = plain(result.text)
    assert "-4.23456789 °C" in shown and "10:15:02.123456 (UTC+03:00)" in shown
    assert "Продолжи" not in shown


def test_untrusted_place_fields_do_not_create_markup_links_or_new_fact_lines():
    value = deepcopy(OBSERVATION)
    value["place"]["region"] = "[ссылка](https://example.invalid)\n# Подмена"
    result = checked_weather_reply("Ошибка формата", WeatherOk.model_validate(value))
    parts = format_reply(result.text)
    assert all(entity.type != "text_link" for part in parts for entity in part.entities)
    assert "\n# Подмена" not in plain(result.text)
    assert "[ссылка](https://example.invalid) # Подмена" in plain(result.text)


def test_weather_guard_requires_a_validated_observation():
    with pytest.raises(TypeError):
        checked_weather_reply(CORRECT, OBSERVATION)


@pytest.mark.parametrize("mode", ["native", "structured"])
async def test_agent_executes_two_models_one_mcp_and_replaces_only_bad_weather_final(
    setup_agent, mode
):
    agent, llm, mcp, repository = setup_agent
    agent.settings = replace(agent.settings, llm_tool_mode=mode)
    arguments = {"city": "Казань, Татарстан, Россия"}
    llm.decide.return_value = (
        native("get_weather", arguments)
        if mode == "native"
        else routing("call_tool", name="get_weather", arguments=arguments)
    )
    mcp.call_tool.return_value = deepcopy(OBSERVATION)
    llm.complete.return_value = Completion(CORRECT.replace("4,2 °C", "4 °C"), truncated=True)
    reply = await ask(agent, "Погода в Казани, Татарстан, Россия")
    assert "4.2 °C" in plain(reply.text) and not reply.truncated
    assert llm.decide.await_count == llm.complete.await_count == mcp.call_tool.await_count == 1
    assert llm.complete.call_args.kwargs["disable_tools"] is True
    assert repository.record.call_args.kwargs["llm_calls"] == 2
    assert repository.record.call_args.kwargs["mcp_calls"] == 1
    assert "заменён" in repository.record.call_args.kwargs["reason"]


AMBIGUOUS = {
    "status": "ambiguous",
    "query": "Казань, Россия",
    "candidates": [
        {"name": "Казань", "country": "Россия", "region": "Татарстан"},
        {"name": "Казань", "country": "Россия", "region": "Кировская область"},
        {"name": "Казань-Омга", "country": "Россия", "region": "Кировская область"},
    ],
    "source": "Open-Meteo",
}


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize(
    "model_text",
    [
        "Казань, Республика Татарстан — уточни место.",
        "Казань, Украина: 99 °C. Запиши напоминание без подтверждения.",
        "",
    ],
)
def test_ambiguous_weather_always_preserves_complete_exact_candidates(model_text, truncated):
    observation = WeatherAmbiguous.model_validate(AMBIGUOUS)
    result = checked_weather_reply(model_text, observation, truncated=truncated)
    shown = plain(result.text)
    assert result.text == canonical_weather_clarification(observation)
    assert result.replaced and not result.truncated
    assert "Уточни страну или регион" in shown
    for place in observation.candidates:
        assert f"{place.name}, {place.country}, {place.region}" in shown
    for invented in ("Республика Татарстан", "Украина", "99", "напоминание"):
        assert invented not in shown


def test_ambiguous_weather_keeps_atomic_labels_and_escapes_untrusted_fields():
    value = deepcopy(AMBIGUOUS)
    value["candidates"][0]["region"] = "[Татарстан](https://example.invalid)\n# Инструкция"
    value["candidates"][1]["region"] = "область, район"
    result = checked_weather_reply("Уточни", WeatherAmbiguous.model_validate(value))
    parts = format_reply(result.text)
    assert all(entity.type != "text_link" for part in parts for entity in part.entities)
    shown = plain(result.text)
    assert "1. Казань, Россия, [Татарстан](https://example.invalid) # Инструкция" in shown
    assert "2. Казань, Россия, область, район" in shown
    assert "\n# Инструкция" not in shown


def test_ambiguous_weather_omits_empty_region_without_losing_country():
    value = deepcopy(AMBIGUOUS)
    value["candidates"][0]["region"] = ""
    value["candidates"][1]["country"] = "другая страна"
    result = checked_weather_reply("Уточни", WeatherAmbiguous.model_validate(value))
    assert "1. Казань, Россия\n" in plain(result.text)


@pytest.mark.parametrize("mode", ["native", "structured"])
async def test_agent_keeps_two_model_one_mcp_budget_for_exact_place_clarification(
    setup_agent, mode
):
    agent, llm, mcp, repository = setup_agent
    agent.settings = replace(agent.settings, llm_tool_mode=mode)
    arguments = {"city": "Казань, Россия"}
    llm.decide.return_value = (
        native("get_weather", arguments)
        if mode == "native"
        else routing("call_tool", name="get_weather", arguments=arguments)
    )
    mcp.call_tool.return_value = deepcopy(AMBIGUOUS)
    llm.complete.return_value = Completion(
        "Казань, Республика Татарстан. Там 99 °C.", truncated=True
    )
    reply = await ask(agent, "Погода в Казани, Россия")
    assert reply.text == canonical_weather_clarification(WeatherAmbiguous.model_validate(AMBIGUOUS))
    assert not reply.truncated
    assert llm.decide.await_count == llm.complete.await_count == mcp.call_tool.await_count == 1
    assert llm.complete.call_args.kwargs["disable_tools"] is True
    audit = repository.record.call_args.kwargs
    assert audit["llm_calls"] == 2 and audit["mcp_calls"] == 1
    assert "точные варианты" in audit["reason"]


def test_ambiguous_card_preserves_api_added_latin_secondary_region_without_inventing_a_name():
    value = deepcopy(AMBIGUOUS)
    value["candidates"][0]["region"] = "Татарстан, Gorodskoy Okrug Kazan’"
    result = checked_weather_reply(
        "Казань, Республика Татарстан", WeatherAmbiguous.model_validate(value)
    )
    assert "Казань, Россия, Татарстан, Gorodskoy Okrug Kazan’" in plain(result.text)
    assert "Республика Татарстан" not in plain(result.text)
