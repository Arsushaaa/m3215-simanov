"""Small immutable text corpus loaded only from a trusted registry, not user paths."""

import hashlib
import re
from pathlib import Path

from app.materials.contracts import CatalogOk, MaterialMatch, SearchArguments, SearchData

ROOT = Path(__file__).resolve().parents[2] / "data" / "materials"
MAX_DOCUMENT_BYTES = 64 * 1024
STOP = frozenset(
    (
        "по в на из и или что как для это мне найди материалах материалов учебным "
        "объясни определение какие правила относятся краткое объяснение составь"
    ).split()
)


class MaterialSourceError(ValueError):
    pass


def expected_catalog(root=ROOT):
    try:
        path = root / "registry.json"
        if path.is_symlink() or path.stat().st_size > 16 * 1024:
            raise ValueError
        return CatalogOk.model_validate_json(path.read_bytes())
    except (OSError, ValueError, TypeError, RecursionError):
        raise MaterialSourceError("Material registry unavailable") from None


def terms(text):
    # Lexical retrieval only; this is never an intent/action parser.
    words = re.findall(r"[^\W_]+", text.casefold().replace("ё", "е"))
    return {word[:6] if len(word) >= 6 else word for word in words if word not in STOP}


class Corpus:
    def __init__(self, catalog, chunks):
        self._catalog, self._chunks = catalog, chunks

    @classmethod
    def load(cls, root=ROOT):
        catalog = expected_catalog(root)
        chunks = {}
        try:
            for document in catalog.documents:
                path = root / (document.document_id + ".md")
                if path.is_symlink() or path.stat().st_size > MAX_DOCUMENT_BYTES:
                    raise ValueError
                raw = path.read_bytes()
                if (
                    len(raw) > MAX_DOCUMENT_BYTES
                    or hashlib.sha256(raw).hexdigest() != document.version
                ):
                    raise ValueError
                text = raw.decode("utf-8")
                if not text.strip() or "\0" in text:
                    raise ValueError
                # Bounded fragments keep the complete MCP envelope under its transport limit.
                pieces = [
                    part[i : i + 900]
                    for part in text.split("\n\n")
                    if part.strip()
                    and not (part.lstrip().startswith("#") and "\n" not in part.strip())
                    for i in range(0, len(part), 900)
                ]
                chunks[document.document_id] = tuple(pieces)
        except (OSError, ValueError, TypeError, UnicodeError):
            raise MaterialSourceError("Material corpus unavailable") from None
        return cls(catalog, chunks)

    def catalog(self):
        return self._catalog.model_copy(deep=True)

    def search(self, arguments: SearchArguments):
        arguments = SearchArguments.model_validate(arguments)
        if arguments.document_id is not None and arguments.document_id not in self._chunks:
            raise ValueError("Unknown document ID")
        query = terms(arguments.query)
        matches = []
        for document in self._catalog.documents:
            if arguments.document_id is not None and arguments.document_id != document.document_id:
                continue
            candidates = []
            for index, snippet in enumerate(self._chunks[document.document_id]):
                overlap = query & terms(snippet)
                if overlap:
                    candidates.append((len(overlap) / len(query), -index, snippet))
            if candidates:
                score, _, snippet = max(candidates)
                matches.append(
                    MaterialMatch(
                        document_id=document.document_id,
                        title=document.title,
                        version=document.version,
                        snippet=snippet,
                        score=float(score),
                    )
                )
        matches.sort(key=lambda m: (-m.score, m.document_id))
        matches = matches[: arguments.limit]
        return SearchData(
            status="ok" if matches else "not_found",
            query=arguments.query,
            document_id=arguments.document_id,
            limit=arguments.limit,
            catalog_version=self._catalog.catalog_version,
            matches=matches,
        )
