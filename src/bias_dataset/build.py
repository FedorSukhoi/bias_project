from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .relevance import assess_political_relevance
from .registry import load_registry

FIELDS = (
    "record_id",
    "source_id",
    "source_name",
    "source_domain",
    "headline",
    "feed_summary",
    "feed_summary_truncated",
    "subheadline",
    "subheadline_provenance",
    "canonical_url",
    "published_at",
    "observed_at",
    "headline_hash",
    "text_hash",
    "weak_label",
    "label_scope",
    "label_provider",
    "rating_scope",
    "rating_url",
    "rating_checked_at",
    "holdout_outlet",
    "acquisition_method",
    "discovery_source",
    "political_relevance_score",
    "political_relevance_reason",
    "cross_source_headline_count",
)


def hostname_matches(hostname: str, expected: str) -> bool:
    return hostname == expected or hostname.endswith("." + expected)


def build(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    registry_path: Path | None = None,
    politics_only: bool = True,
) -> dict[str, object]:
    registry_domains = (
        {outlet.source_id: outlet.domain for outlet in load_registry(registry_path)}
        if registry_path is not None
        else {}
    )
    candidates: list[dict[str, object]] = []
    rejected = Counter()
    seen_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["record_id"] in seen_ids:
                rejected["duplicate_record_id"] += 1
                continue
            seen_ids.add(record["record_id"])
            expected_domain = registry_domains.get(record["source_id"], record["source_domain"])
            hostname = (urlparse(record["canonical_url"]).hostname or "").lower()
            if not hostname_matches(hostname, expected_domain):
                rejected["off_domain_url"] += 1
                continue
            record["source_domain"] = expected_domain
            if len(record["headline"].split()) < 3:
                rejected["headline_too_short"] += 1
                continue
            relevance = assess_political_relevance(
                record["headline"], record["feed_summary"], record["canonical_url"]
            )
            record["political_relevance_score"] = relevance.score
            record["political_relevance_reason"] = relevance.reason
            if politics_only and not relevance.relevant:
                rejected["non_political"] += 1
                continue
            original_summary = record["feed_summary"]
            record["feed_summary_truncated"] = len(original_summary) > 1000
            record["feed_summary"] = original_summary[:1000].strip()
            candidates.append(record)

    # An archive and a publisher feed can discover the same headline/URL with
    # different summaries. Keep one version, preferring first-party feed data
    # and then the richer summary. Different headlines at the same URL remain
    # separate revisions.
    best_by_story: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in candidates:
        key = (str(record["source_id"]), str(record["canonical_url"]), str(record["headline_hash"]))
        previous = best_by_story.get(key)
        if previous is None:
            best_by_story[key] = record
            continue
        rejected["duplicate_story_version"] += 1
        current_rank = (
            record.get("acquisition_method") == "publisher_feed",
            len(str(record.get("feed_summary", ""))),
        )
        previous_rank = (
            previous.get("acquisition_method") == "publisher_feed",
            len(str(previous.get("feed_summary", ""))),
        )
        if current_rank > previous_rank:
            best_by_story[key] = record
    records = list(best_by_story.values())

    sources_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sources_by_hash[record["headline_hash"]].add(record["source_id"])
    for record in records:
        record["cross_source_headline_count"] = len(sources_by_hash[record["headline_hash"]])

    records.sort(key=lambda row: (str(row["published_at"]), str(row["source_id"]), str(row["record_id"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    report = {
        "accepted": len(records),
        "rejected": dict(rejected),
        "by_label": dict(sorted(Counter(str(row["weak_label"]) for row in records).items())),
        "by_source": dict(sorted(Counter(str(row["source_id"]) for row in records).items())),
        "holdout_records": sum(bool(row["holdout_outlet"]) for row in records),
        "missing_published_at": sum(not row["published_at"] for row in records),
        "with_feed_summary": sum(bool(row["feed_summary"]) for row in records),
        "cross_source_exact_headlines": sum(int(row["cross_source_headline_count"]) > 1 for row in records),
        "political_relevance_reasons": dict(
            sorted(Counter(str(row["political_relevance_reason"]) for row in records).items())
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a model-ready publication-proxy CSV")
    parser.add_argument("--input", type=Path, default=Path("data/interim/feed_items.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/publication_proxy_headlines.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/dataset_report.json"))
    parser.add_argument("--registry", type=Path, default=Path("config/outlets.csv"))
    parser.add_argument(
        "--include-all-topics",
        action="store_true",
        help="Keep non-political records in the processed CSV (disabled by default)",
    )
    args = parser.parse_args()
    report = build(
        args.input,
        args.output,
        args.report,
        args.registry,
        politics_only=not args.include_all_topics,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
