from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .feeds import digest, parse_feed
from .registry import Outlet, load_registry

USER_AGENT = "bias-project-research-collector/0.1 (+noncommercial research)"
MAX_FEED_BYTES = 10 * 1024 * 1024
SYSTEM_CA_BUNDLES = (Path("/etc/ssl/cert.pem"), Path("/private/etc/ssl/cert.pem"))


def tls_context() -> ssl.SSLContext:
    """Use platform defaults, with the macOS system bundle as a safe fallback."""
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile and Path(defaults.cafile).is_file():
        return ssl.create_default_context()
    for bundle in SYSTEM_CA_BUNDLES:
        if bundle.is_file():
            return ssl.create_default_context(cafile=str(bundle))
    return ssl.create_default_context()


def paged_feed_url(feed_url: str, page: int) -> str:
    if page <= 1:
        return feed_url
    parsed = urllib.parse.urlparse(feed_url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, value) for key, value in query if key.lower() not in {"page", "paged"}]
    query.append(("paged", str(page)))
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def fetch_feed(outlet: Outlet, timeout: float, page: int = 1) -> tuple[bytes, str]:
    feed_url = paged_feed_url(outlet.feed_url, page)
    request = urllib.request.Request(
        feed_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"},
    )
    with urllib.request.urlopen(request, timeout=timeout, context=tls_context()) as response:
        payload = response.read(MAX_FEED_BYTES + 1)
        if len(payload) > MAX_FEED_BYTES:
            raise ValueError(f"Feed exceeded {MAX_FEED_BYTES} bytes")
        return payload, response.geturl()


def raw_path(root: Path, outlet: Outlet, observed_at: datetime, payload: bytes) -> Path:
    stamp = observed_at.strftime("%Y%m%dT%H%M%SZ")
    short_hash = hashlib.sha256(payload).hexdigest()[:12]
    return root / observed_at.strftime("%Y-%m-%d") / outlet.source_id / f"{stamp}_{short_hash}.xml"


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["record_id"])
    return ids


def attach_provenance(item: dict[str, str], outlet: Outlet, observed_at: str, fetched_url: str) -> dict[str, object]:
    record_id = digest(f"{outlet.source_id}\n{item['canonical_url']}\n{item['text_hash']}")
    return {
        "record_id": record_id,
        "source_id": outlet.source_id,
        "source_name": outlet.source_name,
        "source_domain": outlet.domain,
        "headline": item["headline"],
        "feed_summary": item["feed_summary"],
        "subheadline": "",
        "subheadline_provenance": "not_collected",
        "canonical_url": item["canonical_url"],
        "published_at": item["published_at"],
        "published_raw": item["published_raw"],
        "observed_at": observed_at,
        "headline_hash": item["headline_hash"],
        "text_hash": item["text_hash"],
        "weak_label": outlet.weak_label,
        "label_scope": "outlet",
        "label_provider": "AllSides",
        "rating_scope": outlet.rating_scope,
        "rating_url": outlet.rating_url,
        "rating_checked_at": outlet.rating_checked_at,
        "feed_url": outlet.feed_url,
        "fetched_url": fetched_url,
        "acquisition_method": "publisher_feed",
        "discovery_source": outlet.source_name,
        "holdout_outlet": outlet.holdout,
    }


def collect(
    registry_path: Path,
    output_path: Path,
    raw_root: Path,
    timeout: float,
    source_ids: set[str],
    pages: tuple[int, ...] = (1,),
    override_urls: tuple[str, ...] = (),
) -> dict[str, object]:
    outlets = [outlet for outlet in load_registry(registry_path) if outlet.enabled]
    if source_ids:
        outlets = [outlet for outlet in outlets if outlet.source_id in source_ids]
        missing = source_ids - {outlet.source_id for outlet in outlets}
        if missing:
            raise ValueError(f"Unknown or disabled source IDs: {', '.join(sorted(missing))}")

    if override_urls:
        if len(outlets) != 1:
            raise ValueError("--feed-url requires exactly one --source")
        tasks = [
            (replace(outlets[0], feed_url=url), f":feed_{index}")
            for index, url in enumerate(override_urls, start=1)
        ]
    else:
        tasks = [(outlet, "") for outlet in outlets]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(output_path)
    stats: dict[str, object] = {"sources_attempted": len(tasks) * len(pages), "sources": {}, "new_records": 0, "errors": 0}

    with output_path.open("a", encoding="utf-8") as output:
        for outlet, task_suffix in tasks:
            for page in pages:
                task_id = f"{outlet.source_id}{task_suffix}"
                if page != 1:
                    task_id += f":page_{page}"
                observed_dt = datetime.now(UTC)
                observed_at = observed_dt.isoformat().replace("+00:00", "Z")
                try:
                    payload, fetched_url = fetch_feed(outlet, timeout, page)
                    destination = raw_path(raw_root, outlet, observed_dt, payload)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    items = parse_feed(payload)
                    if not items:
                        raise ValueError("Response contained no RSS/Atom items")
                    added = 0
                    for item in items:
                        record = attach_provenance(item, outlet, observed_at, fetched_url)
                        if record["record_id"] in existing_ids:
                            continue
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        existing_ids.add(str(record["record_id"]))
                        added += 1
                    stats["sources"][task_id] = {"parsed": len(items), "added": added, "status": "ok"}
                    stats["new_records"] += added
                except (OSError, ValueError, urllib.error.URLError, ET.ParseError) as exc:
                    stats["sources"][task_id] = {"status": "error", "error": str(exc)}
                    stats["errors"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect publication-labelled headlines from registered feeds")
    parser.add_argument("--registry", type=Path, default=Path("config/outlets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/feed_items.jsonl"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/feeds"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--source", action="append", default=[], help="Collect only this source_id; repeatable")
    parser.add_argument(
        "--page", action="append", type=int, default=[],
        help="Request this feed page; repeatable. Defaults to page 1.",
    )
    parser.add_argument(
        "--feed-url", action="append", default=[],
        help="Override feed URL for one selected source; repeatable for category feeds.",
    )
    args = parser.parse_args()
    try:
        pages = tuple(dict.fromkeys(args.page or [1]))
        if any(page < 1 for page in pages):
            raise ValueError("Feed pages must be positive integers")
        stats = collect(
            args.registry, args.output, args.raw_root, args.timeout,
            set(args.source), pages, tuple(args.feed_url),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
