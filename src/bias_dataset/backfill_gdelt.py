from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .collect import MAX_FEED_BYTES, USER_AGENT, attach_provenance, load_existing_ids, tls_context
from .feeds import canonicalize_url, digest, normalized_headline
from .registry import Outlet, load_registry

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
POLITICAL_QUERY = (
    '(election OR congress OR senate OR "white house" OR president OR government '
    'OR politics OR immigration OR parliament OR minister OR legislation OR tariff '
    'OR sanctions OR military OR court)'
)


def gdelt_timestamp(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return ""
    return parsed.isoformat().replace("+00:00", "Z")


def query_url(outlet: Outlet, start: date, end: date, max_records: int) -> str:
    query = f"{POLITICAL_QUERY} domainis:{outlet.domain} sourcelang:english"
    parameters = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": str(max_records),
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
    }
    return GDELT_ENDPOINT + "?" + urllib.parse.urlencode(parameters)


def fetch_articles(url: str, timeout: float) -> list[dict[str, object]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout, context=tls_context()) as response:
        payload = response.read(MAX_FEED_BYTES + 1)
        if len(payload) > MAX_FEED_BYTES:
            raise ValueError("GDELT response exceeded size limit")
    document = json.loads(payload)
    articles = document.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("GDELT response did not contain an article list")
    return articles


def article_item(article: dict[str, object]) -> dict[str, str] | None:
    headline = str(article.get("title", "")).strip()
    url = canonicalize_url(str(article.get("url", "")))
    if not headline or not url:
        return None
    normalized = normalized_headline(headline)
    return {
        "headline": headline,
        "feed_summary": "",
        "canonical_url": url,
        "published_at": gdelt_timestamp(str(article.get("seendate", ""))),
        "published_raw": str(article.get("seendate", "")),
        "headline_hash": digest(normalized),
        "text_hash": digest(f"{normalized}\n"),
    }


def backfill(
    registry_path: Path,
    output_path: Path,
    start: date,
    end: date,
    timeout: float,
    delay: float,
    max_records: int,
    source_ids: set[str],
) -> dict[str, object]:
    outlets = [outlet for outlet in load_registry(registry_path) if outlet.enabled]
    if source_ids:
        outlets = [outlet for outlet in outlets if outlet.source_id in source_ids]
        missing = source_ids - {outlet.source_id for outlet in outlets}
        if missing:
            raise ValueError(f"Unknown or disabled source IDs: {', '.join(sorted(missing))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(output_path)
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stats: dict[str, object] = {"sources_attempted": len(outlets), "sources": {}, "new_records": 0, "errors": 0}

    with output_path.open("a", encoding="utf-8") as output:
        for index, outlet in enumerate(outlets):
            if index and delay:
                time.sleep(delay)
            url = query_url(outlet, start, end, max_records)
            try:
                articles = fetch_articles(url, timeout)
                added = 0
                for article in articles:
                    item = article_item(article)
                    if item is None:
                        continue
                    record = attach_provenance(item, outlet, observed_at, url)
                    record["feed_url"] = ""
                    record["acquisition_method"] = "archive_backfill"
                    record["discovery_source"] = "GDELT DOC 2.0"
                    if record["record_id"] in existing_ids:
                        continue
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    existing_ids.add(str(record["record_id"]))
                    added += 1
                stats["sources"][outlet.source_id] = {"returned": len(articles), "added": added, "status": "ok"}
                stats["new_records"] += added
            except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
                stats["sources"][outlet.source_id] = {"status": "error", "error": str(exc)}
                stats["errors"] += 1
    return stats


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    today = datetime.now(UTC).date()
    parser = argparse.ArgumentParser(description="Backfill registered publications through GDELT discovery")
    parser.add_argument("--registry", type=Path, default=Path("config/outlets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/feed_items.jsonl"))
    parser.add_argument("--start", type=parse_date, default=today - timedelta(days=89))
    parser.add_argument("--end", type=parse_date, default=today)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--max-records", type=int, choices=range(1, 251), default=250)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not be after --end")
    try:
        stats = backfill(
            args.registry, args.output, args.start, args.end, args.timeout,
            args.delay, args.max_records, set(args.source),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

