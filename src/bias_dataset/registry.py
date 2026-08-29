from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from . import LABELS


@dataclass(frozen=True)
class Outlet:
    source_id: str
    source_name: str
    domain: str
    weak_label: str
    rating_scope: str
    rating_url: str
    rating_checked_at: str
    feed_url: str
    feed_kind: str
    enabled: bool
    holdout: bool


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false, got {value!r}")
    return normalized == "true"


def load_registry(path: Path) -> list[Outlet]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    outlets: list[Outlet] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        outlet = Outlet(
            source_id=row["source_id"].strip(),
            source_name=row["source_name"].strip(),
            domain=row["domain"].strip().lower(),
            weak_label=row["weak_label"].strip(),
            rating_scope=row["rating_scope"].strip(),
            rating_url=row["rating_url"].strip(),
            rating_checked_at=row["rating_checked_at"].strip(),
            feed_url=row["feed_url"].strip(),
            feed_kind=row["feed_kind"].strip().lower(),
            enabled=_as_bool(row["enabled"]),
            holdout=_as_bool(row["holdout"]),
        )
        if not outlet.source_id or outlet.source_id in seen_ids:
            raise ValueError(f"Duplicate or empty source_id at line {line_number}")
        if outlet.weak_label not in LABELS:
            raise ValueError(f"Invalid label {outlet.weak_label!r} at line {line_number}")
        if outlet.feed_kind not in {"rss", "atom"}:
            raise ValueError(f"Unsupported feed_kind at line {line_number}")
        if urlparse(outlet.feed_url).scheme not in {"http", "https"}:
            raise ValueError(f"Invalid feed URL at line {line_number}")
        seen_ids.add(outlet.source_id)
        outlets.append(outlet)

    if set(LABELS) != {outlet.weak_label for outlet in outlets if outlet.enabled}:
        raise ValueError("Every bias class must have at least one enabled outlet")
    return outlets

