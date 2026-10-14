"""Markdown → текст и Telegram entities; разбиение без повреждения разметки."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from aiogram.types import MessageEntity
from markdown_it import MarkdownIt
from markdown_it.token import Token

MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable(["strikethrough", "table"])


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


@dataclass
class FormattedText:
    text: str
    entities: list[MessageEntity] = field(default_factory=list)


class Renderer:
    def __init__(self):
        self.parts: list[str] = []
        self.entities: list[MessageEntity] = []
        self.offset = 0
        self.tail = ""

    def append(self, text: str) -> None:
        self.parts.append(text)
        self.offset += utf16_length(text)
        self.tail = (self.tail + text)[-2:]

    def newline(self, count: int = 2) -> None:
        if self.offset:
            missing = count - (len(self.tail) - len(self.tail.rstrip("\n")))
            if missing > 0:
                self.append("\n" * missing)

    def entity(self, kind: str, start: int, **attributes) -> None:
        if self.offset > start:
            self.entities.append(
                MessageEntity(type=kind, offset=start, length=self.offset - start, **attributes)
            )

    def inline(self, tokens: list[Token]) -> None:
        stack = []
        styles = {"strong": "bold", "em": "italic", "s": "strikethrough", "link": "text_link"}
        for token in tokens:
            if token.type in {"text", "html_inline"}:
                self.append(token.content)
            elif token.type in {"softbreak", "hardbreak"}:
                self.append("\n")
            elif token.type == "code_inline":
                start = self.offset
                self.append(token.content)
                self.entity("code", start)
            elif token.type == "image":
                # Показываем подпись, не загружаем картинку по адресу из ответа модели.
                self.append(token.content)
            elif token.nesting == 1 and token.type.removesuffix("_open") in styles:
                style = token.type.removesuffix("_open")
                attributes = {}
                if style == "link":
                    url = token.attrGet("href") or ""
                    if urlsplit(url).scheme.lower() in {"https", "http", "tg", "mailto"}:
                        attributes["url"] = url
                stack.append((styles[style], self.offset, attributes))
            elif token.nesting == -1 and stack:
                kind, start, attributes = stack.pop()
                if kind != "text_link" or attributes:
                    self.entity(kind, start, **attributes)

    def render(self, tokens: list[Token]) -> FormattedText:
        lists, headings, quotes = [], [], []
        previous, column = "", 0
        for token in tokens:
            kind = token.type
            if kind == "inline":
                self.inline(token.children or [])
            elif kind == "paragraph_open":
                if previous != "list_item_open":
                    self.newline(1 if lists else 2)
            elif kind == "heading_open":
                self.newline()
                headings.append(self.offset)
            elif kind == "heading_close":
                self.entity("bold", headings.pop())
            elif kind in {"fence", "code_block"}:
                self.newline()
                start = self.offset
                self.append(token.content.removesuffix("\n"))
                language = token.info.strip().split()[0] if token.info.strip() else None
                self.entity("pre", start, language=language)
            elif kind in {"bullet_list_open", "ordered_list_open"}:
                if not lists:
                    self.newline()
                lists.append(int(token.attrGet("start") or 1) if kind.startswith("ordered") else 0)
            elif kind in {"bullet_list_close", "ordered_list_close"}:
                lists.pop()
            elif kind == "list_item_open":
                self.newline(1)
                self.append("  " * (len(lists) - 1))
                self.append(f"{lists[-1]}. " if lists[-1] else "• ")
                if lists[-1]:
                    lists[-1] += 1
            elif kind == "blockquote_open":
                self.newline()
                quotes.append(self.offset)
            elif kind == "blockquote_close":
                start = quotes.pop()
                if not quotes:
                    self.entity("blockquote", start)
            elif kind == "hr":
                self.newline()
                self.append("────────")
            elif kind == "table_open":
                self.newline()
            elif kind == "tr_open":
                self.newline(1)
                column = 0
            elif kind in {"th_open", "td_open"}:
                if column:
                    self.append(" | ")
                column += 1
            previous = kind

        # Telegram запрещает выделения внутри code/pre: сохраняем стиль вокруг кода.
        code = [e for e in self.entities if e.type in {"code", "pre"}]
        entities = []
        for entity in self.entities:
            ranges = [(entity.offset, entity.offset + entity.length)]
            if entity.type not in {"code", "pre"}:
                for literal in code:
                    left, right = literal.offset, literal.offset + literal.length
                    ranges = [
                        piece
                        for start, end in ranges
                        for piece in ((start, min(end, left)), (max(start, right), end))
                        if piece[0] < piece[1]
                    ]
            entities.extend(
                entity.model_copy(update={"offset": start, "length": end - start})
                for start, end in ranges
            )
        return FormattedText(
            "".join(self.parts), sorted(entities, key=lambda e: (e.offset, -e.length))
        )


def split_formatted(reply: FormattedText, limit: int = 4096) -> list[FormattedText]:
    if limit < 2:
        raise ValueError("Лимит должен вмещать хотя бы один символ Unicode.")
    chunks, index, offset = [], 0, 0
    while index < len(reply.text):
        end, units, boundary = index, 0, index
        while end < len(reply.text):
            width = utf16_length(reply.text[end])
            if units + width > limit:
                break
            units += width
            end += 1
            if reply.text[end - 1].isspace():
                boundary = end
        # Предпочитаем границу строки/слова во второй половине части, сохраняя пробелы.
        if end < len(reply.text) and boundary > index + (end - index) // 2:
            end = boundary
        part = reply.text[index:end]
        units = utf16_length(part)
        entities = []
        for entity in reply.entities:
            start = max(offset, entity.offset)
            finish = min(offset + units, entity.offset + entity.length)
            if start < finish:
                entities.append(
                    entity.model_copy(update={"offset": start - offset, "length": finish - start})
                )
        chunks.append(FormattedText(part, entities))
        index, offset = end, offset + units
    return chunks


def format_reply(markdown: str) -> list[FormattedText]:
    reply = Renderer().render(MARKDOWN.parse(markdown))
    # Ответ из одной разметки (например, пустой fence) тоже должен быть отправляемым.
    if not reply.text.strip():
        reply = FormattedText(markdown)
    return split_formatted(reply)
