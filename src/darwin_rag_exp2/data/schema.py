"""Schema definitions for the read-only notice export."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NoticeRecord(BaseModel):
    """One source notice row from ``scatch_notices.jsonl``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    url: str
    slug: str
    title: str
    date: str
    category: str
    text: str
    text_length: int
    source: str
    collected_at: str
