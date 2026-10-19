"""Выбор часового пояса требует явной кнопки владельца и свежего меню."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.timezone_lookup import TimezoneLookupError, TimezoneOption
from app.timezone_ui import TimezoneUI, TimezoneView

NOW = datetime(2026, 10, 19, 6, tzinfo=UTC)
OPTIONS = [
    TimezoneOption("Санкт-Петербург — Россия", "Europe/Moscow"),
    TimezoneOption("Saint Petersburg — США, Florida", "America/New_York"),
]


def buttons(view):
    assert isinstance(view, TimezoneView)
    return [button for row in view.rows for button in row]


def callback(view, prefix):
    matches = [
        button.callback_data for button in buttons(view) if button.callback_data.startswith(prefix)
    ]
    assert matches, f"Не найдена кнопка {prefix}"
    return matches[0]


def saved_calls(confirmations):
    return [call for call in confirmations.timezone.await_args_list if len(call.args) > 1]


@pytest.fixture
def setup_ui():
    clock = [0.0]

    async def timezone(_owner, argument=None):
        return f"Часовой пояс: {argument or 'Europe/Moscow'}."

    confirmations = SimpleNamespace(
        timezone=AsyncMock(side_effect=timezone),
        repository=SimpleNamespace(
            timezone=AsyncMock(return_value="Europe/Moscow"),
            now=AsyncMock(return_value=NOW),
        ),
    )
    lookup = SimpleNamespace(search=AsyncMock(return_value=OPTIONS))
    return TimezoneUI(confirmations, lookup, clock=lambda: clock[0]), confirmations, lookup, clock


async def test_open_only_shows_saved_zone_options_and_search_without_changing_preference(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    view = await ui.open(1)
    assert "Europe/Moscow" in view.text
    callback(view, "tz:pick:")
    callback(view, "tz:search:")
    callback(view, "tz:cancel:")
    assert not saved_calls(confirmations)
    lookup.search.assert_not_awaited()
    assert not ui.accepts_input(1)


async def test_ambiguous_city_lists_both_countries_and_saves_only_explicit_preview_use(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    view = await ui.open(1, "Saint Petersburg")
    lookup.search.assert_awaited_once_with("Saint Petersburg")
    choices = [button for button in buttons(view) if button.callback_data.startswith("tz:pick:")]
    assert len(choices) == 2
    assert "Россия" in choices[0].text
    assert "США" in choices[1].text
    assert not saved_calls(confirmations)

    preview = await ui.handle(1, choices[1].callback_data)
    assert "America/New_York" in preview.text
    assert "19 октября 2026" in preview.text
    assert "02:00" in preview.text and "UTC-04:00" in preview.text
    assert not saved_calls(confirmations)

    await ui.handle(1, callback(preview, "tz:use:"))
    assert saved_calls(confirmations)[0].args == (1, "America/New_York")
    assert len(saved_calls(confirmations)) == 1


async def test_direct_iana_retains_command_compatibility_without_geocoding(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    await ui.open(1, "Europe/Moscow")
    confirmations.timezone.assert_awaited_once_with(1, "Europe/Moscow")
    lookup.search.assert_not_awaited()


async def test_search_button_routes_next_text_to_lookup_then_requires_selection(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    menu = await ui.open(1)
    await ui.handle(1, callback(menu, "tz:search:"))
    assert ui.accepts_input(1)
    assert not ui.accepts_input(2)
    result = await ui.search_input(1, "Санкт-Петербург")
    assert isinstance(result, TimezoneView)
    lookup.search.assert_awaited_once_with("Санкт-Петербург")
    callback(result, "tz:pick:")
    assert not saved_calls(confirmations)
    assert await ui.search_input(2, "Санкт-Петербург") is None


@pytest.mark.parametrize("failure", [None, "unavailable", "invalid_query"])
async def test_unsuccessful_search_explains_how_to_retry_without_setting_a_zone(setup_ui, failure):
    ui, confirmations, lookup, _ = setup_ui
    if failure is None:
        lookup.search.return_value = []
    else:
        lookup.search.side_effect = TimezoneLookupError(failure, "Укажи город или IANA-зону.")
    result = await ui.open(1, "Непонятный город")
    text = result.text if isinstance(result, TimezoneView) else result
    assert text
    assert "IANA" in text or "поиск" in text.lower() or "город" in text.lower()
    assert not saved_calls(confirmations)


async def test_foreign_or_forged_choice_cannot_save_and_owner_can_still_finish(setup_ui):
    ui, confirmations, _, _ = setup_ui
    menu = await ui.open(1, "Saint Petersburg")
    pick = callback(menu, "tz:pick:")
    _, _, token, _ = pick.split(":")
    await ui.handle(2, pick)
    await ui.handle(1, f"tz:use:{token}:0")
    await ui.handle(1, f"tz:pick:{token}:9999")
    assert not saved_calls(confirmations)

    preview = await ui.handle(1, pick)
    use = callback(preview, "tz:use:")
    wrong_index = 1 if use.endswith(":0") else 0
    await ui.handle(1, f"tz:use:{token}:{wrong_index}")
    assert not saved_calls(confirmations)
    await ui.handle(1, use)
    assert len(saved_calls(confirmations)) == 1


async def test_expired_menu_and_replaced_menu_cannot_apply_old_selection(setup_ui):
    ui, confirmations, _, clock = setup_ui
    menu = await ui.open(1, "Saint Petersburg")
    preview = await ui.handle(1, callback(menu, "tz:pick:"))
    use = callback(preview, "tz:use:")
    clock[0] = 601
    await ui.handle(1, use)
    assert not saved_calls(confirmations)

    fresh = await ui.open(1, "Saint Petersburg")
    preview = await ui.handle(1, callback(fresh, "tz:pick:"))
    old_use = callback(preview, "tz:use:")
    await ui.open(1)
    await ui.handle(1, old_use)
    assert not saved_calls(confirmations)


@pytest.mark.parametrize("cancel", [False, True])
async def test_clear_or_cancel_stops_text_search_and_invalidates_buttons(setup_ui, cancel):
    ui, confirmations, _, _ = setup_ui
    menu = await ui.open(1)
    pick = callback(menu, "tz:pick:")
    await ui.handle(1, callback(menu, "tz:search:"))
    assert ui.accepts_input(1)
    if cancel:
        await ui.handle(1, callback(menu, "tz:cancel:"))
    else:
        ui.clear(1)
    assert not ui.accepts_input(1)
    assert await ui.search_input(1, "Казань") is None
    await ui.handle(1, pick)
    assert not saved_calls(confirmations)


async def test_country_pagination_keeps_all_fifty_choices_and_callback_bytes_bounded(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    lookup.search.return_value = [
        TimezoneOption(f"Страна — город {index}", "Europe/Moscow") for index in range(50)
    ]
    view = await ui.open(1, "страна")
    seen = set()
    while True:
        for button in buttons(view):
            assert len(button.callback_data.encode("utf-8")) <= 64
            if button.callback_data.startswith("tz:pick:"):
                seen.add(button.callback_data.rsplit(":", 1)[1])
        unseen = [
            button.callback_data
            for button in buttons(view)
            if button.callback_data.startswith("tz:page:") and "Дальше" in button.text
        ]
        if not unseen:
            break
        view = await ui.handle(1, unseen[0])
    assert seen == {str(index) for index in range(50)}
    lookup.search.assert_awaited_once()
    assert not saved_calls(confirmations)


async def test_back_to_choices_invalidates_previous_apply_button(setup_ui):
    ui, confirmations, _, _ = setup_ui
    choices = await ui.open(1, "Санкт-Петербург")
    preview = await ui.handle(1, callback(choices, "tz:pick:"))
    use = callback(preview, "tz:use:")
    assert ui.is_apply(1, use)
    await ui.handle(1, callback(preview, "tz:page:"))
    assert not ui.is_apply(1, use)
    assert isinstance(await ui.handle(1, use), str)
    assert not saved_calls(confirmations)


async def test_close_button_remains_valid_during_http_search(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    view = await ui.open(1)
    prompt = await ui.handle(1, callback(view, "tz:search:"))
    close = callback(prompt, "tz:cancel:")
    started, release = asyncio.Event(), asyncio.Event()

    async def search(_query):
        started.set()
        await release.wait()
        return OPTIONS

    lookup.search.side_effect = search
    task = asyncio.create_task(ui.search_input(1, "Санкт-Петербург"))
    await asyncio.wait_for(started.wait(), 1)
    closed = await ui.handle(1, close)
    assert closed.dismiss_previous
    release.set()
    assert isinstance(await asyncio.wait_for(task, 1), str)
    assert not ui.accepts_input(1)
    assert not saved_calls(confirmations)


async def test_old_lookup_completion_cannot_replace_newer_menu_or_its_choices(setup_ui):
    ui, confirmations, lookup, _ = setup_ui
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_lookup(_query):
        started.set()
        await release.wait()
        return [TimezoneOption("Старый результат", "America/New_York")]

    lookup.search.side_effect = slow_lookup
    old = asyncio.create_task(ui.open(1, "старый поиск"))
    await started.wait()
    current = await ui.open(1)
    release.set()
    old_result = await old
    if isinstance(old_result, TimezoneView):
        assert all("Старый результат" not in button.text for button in buttons(old_result))
    assert not saved_calls(confirmations)
    preview = await ui.handle(1, callback(current, "tz:pick:"))
    await ui.handle(1, callback(preview, "tz:use:"))
    assert len(saved_calls(confirmations)) == 1


@pytest.mark.parametrize(
    "data",
    ["tz:pick", "tz:use:missing:0", "tz:page:missing:-1", "tz:unknown:value", "tz:pick:x:NaN"],
)
async def test_malformed_callbacks_never_set_timezone(setup_ui, data):
    ui, confirmations, lookup, _ = setup_ui
    await ui.handle(1, data)
    assert not saved_calls(confirmations)
    lookup.search.assert_not_awaited()
