"""HTML-подобные данные из ответа не теряются и не становятся разметкой Telegram."""

import pytest

from app.formatting import format_reply


def entity_text(reply, entity):
    raw = reply.text.encode("utf-16-le")
    return raw[entity.offset * 2 : (entity.offset + entity.length) * 2].decode("utf-16-le")


@pytest.mark.parametrize(
    "source",
    [
        "<b>жирный</b> & <script>alert(1)</script>",
        '<pre>\ns = "</pre><b>не тег</b>"\n</pre>',
        "<script>\nalert(1)\n</script>\n\nСледующий абзац остаётся.",
        '<b title="😀 & данные">буквальные теги</b>',
    ],
)
def test_raw_html_is_literal_text_without_telegram_entities(source):
    parts = format_reply(source)

    assert "".join(part.text for part in parts) == source
    assert all(not part.entities for part in parts)


def test_literal_html_response_retains_following_code_and_styled_summary():
    source = (
        "<b>жирный</b> & <script>alert(1)</script>\n\n"
        '```python\ns = "</pre><b>не тег</b>"\n```\n\n'
        "**Итог:** данные показаны буквально."
    )

    reply = format_reply(source)[0]

    assert reply.text == (
        "<b>жирный</b> & <script>alert(1)</script>\n\n"
        's = "</pre><b>не тег</b>"\n\n'
        "Итог: данные показаны буквально."
    )
    assert [(entity.type, entity_text(reply, entity)) for entity in reply.entities] == [
        ("pre", 's = "</pre><b>не тег</b>"'),
        ("bold", "Итог:"),
    ]
    assert reply.entities[0].language == "python"
