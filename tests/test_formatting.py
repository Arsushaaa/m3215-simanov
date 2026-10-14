import pytest

from app.formatting import FormattedText, format_reply, split_formatted, utf16_length


def entity_text(reply, entity):
    data = reply.text.encode("utf-16-le")
    return data[entity.offset * 2 : (entity.offset + entity.length) * 2].decode("utf-16-le")


def test_headings_lists_code_and_emphasis_from_screenshot():
    parts = format_reply(
        "### Как это работает:\n- Значение: `0.7`\n\n"
        "**Пример** и *ошибка*\n\n```python\nx = 0.7\nprint(x)\n```"
    )
    assert len(parts) == 1
    reply = parts[0]
    assert "###" not in reply.text and "```" not in reply.text
    assert "• Значение: 0.7" in reply.text
    assert [(e.type, entity_text(reply, e)) for e in reply.entities] == [
        ("bold", "Как это работает:"),
        ("code", "0.7"),
        ("bold", "Пример"),
        ("italic", "ошибка"),
        ("pre", "x = 0.7\nprint(x)"),
    ]
    assert reply.entities[-1].language == "python"


def test_links_unicode_special_characters_and_literal_html():
    reply = format_reply("👋 **русский** [сайт](https://example.com/a?q=1&x=2) <b> & `x < y`")[0]
    assert reply.text == "👋 русский сайт <b> & x < y"
    assert [(e.type, entity_text(reply, e)) for e in reply.entities] == [
        ("bold", "русский"),
        ("text_link", "сайт"),
        ("code", "x < y"),
    ]
    assert reply.entities[0].offset == 3
    assert reply.entities[1].url == "https://example.com/a?q=1&x=2"


def test_nested_styles_do_not_overlap_code():
    reply = format_reply("**Жирный `код` и _курсив_**")[0]
    code = next(e for e in reply.entities if e.type == "code")
    for entity in reply.entities:
        if entity != code:
            assert (
                entity.offset + entity.length <= code.offset
                or entity.offset >= code.offset + code.length
            )
    assert {e.type for e in reply.entities} == {"bold", "italic", "code"}


@pytest.mark.parametrize(
    "source",
    ["**незакрытый", "`незакрытый", "```python\nx = '<&>'", "```\n```", "[ссылка](неполная"],
)
def test_incomplete_model_markdown_is_sendable(source):
    parts = format_reply(source)
    assert parts and all(part.text.strip() for part in parts)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("**" + "😀я" * 4000 + "**", "😀я" * 4000),
        ("```python\n" + "x = '😀'\n" * 2000 + "```", ("x = '😀'\n" * 2000).removesuffix("\n")),
    ],
)
def test_long_entity_is_split_without_losing_text_or_unicode(source, expected):
    parts = format_reply(source)
    rendered = "".join(part.text for part in parts)
    assert rendered == expected
    assert len(parts) > 2
    for part in parts:
        assert 0 < utf16_length(part.text) <= 4096
        assert len(part.entities) == 1
        for entity in part.entities:
            assert 0 <= entity.offset < entity.offset + entity.length <= utf16_length(part.text)
            assert entity_text(part, entity) == part.text


def test_split_prefers_word_boundaries_and_preserves_whitespace():
    text = "первая строка\n" + "ещё слово " * 1000
    parts = split_formatted(FormattedText(text))
    assert "".join(part.text for part in parts) == text
    assert all(part.text[-1].isspace() for part in parts[:-1])


def test_nested_lists_quotes_and_tables_are_readable():
    reply = format_reply(
        "1. Первый\n2. Второй\n   - Вложенный\n\n> **Цитата**\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    )[0]
    assert "1. Первый\n2. Второй\n  • Вложенный" in reply.text
    assert "A | B\n1 | 2" in reply.text
    assert "blockquote" in {e.type for e in reply.entities}
