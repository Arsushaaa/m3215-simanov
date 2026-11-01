"""Canonical material MCP schemas. Paths and delivery identities are never model arguments."""

import hashlib
import json
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, RootModel, StringConstraints, model_validator

from app.mcp.contracts import ErrorResult, ShortText, StrictModel

DocumentId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class MaterialModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class MaterialMatch(MaterialModel):
    document_id: DocumentId
    title: ShortText
    version: Digest
    snippet: Annotated[str, StringConstraints(min_length=1, max_length=1200)] = Field(repr=False)
    score: Confidence


class CatalogDocument(MaterialModel):
    document_id: DocumentId
    title: ShortText
    version: Digest
    summary: Annotated[str, StringConstraints(min_length=1, max_length=300)]


def catalog_digest(documents):
    raw = [d.model_dump(mode="json") for d in sorted(documents, key=lambda d: d.document_id)]
    return hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class CatalogOk(MaterialModel):
    status: Literal["ok"] = "ok"
    catalog_version: Digest
    documents: Annotated[list[CatalogDocument], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def consistent(self):
        if len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document ID")
        if self.catalog_version != catalog_digest(self.documents):
            raise ValueError("Invalid catalog digest")
        return self


class CatalogResult(RootModel[Annotated[CatalogOk | ErrorResult, Field(discriminator="status")]]):
    pass


class SearchArguments(MaterialModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    document_id: DocumentId | None = None
    limit: Annotated[int, Field(strict=True, ge=1, le=5)] = 3


class SearchData(MaterialModel):
    status: Literal["ok", "not_found"]
    query: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    document_id: DocumentId | None
    limit: Annotated[int, Field(strict=True, ge=1, le=5)]
    catalog_version: Digest
    matches: Annotated[list[MaterialMatch], Field(max_length=5)]

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "ok") != bool(self.matches) or len(self.matches) > self.limit:
            raise ValueError("Inconsistent search outcome")
        if len({m.document_id for m in self.matches}) != len(self.matches):
            raise ValueError("Duplicate match")
        if self.document_id is not None and any(
            m.document_id != self.document_id for m in self.matches
        ):
            raise ValueError("Wrong requested document")
        return self


class SearchResult(RootModel[Annotated[SearchData | ErrorResult, Field(discriminator="status")]]):
    pass
