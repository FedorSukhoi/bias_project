from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: str | None) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, *names: str) -> str:
    wanted = set(names)
    for child in element:
        if local_name(child.tag) in wanted:
            return "".join(child.itertext()).strip()
    return ""


def _entry_link(element: ET.Element) -> str:
    for child in element:
        if local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href", "").strip()
        rel = child.attrib.get("rel", "alternate")
        if href and rel in {"alternate", ""}:
            return href
        if child.text and child.text.strip():
            return child.text.strip()
    return ""


def canonicalize_url(value: str) -> str:
    parsed = urlparse(value.strip())
    scheme = parsed.scheme.lower() or "https"
    hostname = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    netloc = hostname + port
    query = urlencode(
        [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
         if key.lower() not in TRACKING_PARAMETERS]
    )
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", query, ""))


def parse_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def normalized_headline(value: str) -> str:
    value = plain_text(value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_feed(payload: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(payload)
    entries = [element for element in root.iter() if local_name(element.tag) in {"item", "entry"}]
    records: list[dict[str, str]] = []
    for entry in entries:
        title = plain_text(_child_text(entry, "title"))
        url = canonicalize_url(_entry_link(entry))
        summary = plain_text(_child_text(entry, "description", "summary", "content", "encoded"))
        published_raw = _child_text(entry, "pubdate", "published", "updated", "date")
        if not title or not url:
            continue
        normalized = normalized_headline(title)
        records.append(
            {
                "headline": title,
                "feed_summary": summary,
                "canonical_url": url,
                "published_at": parse_timestamp(published_raw),
                "published_raw": published_raw,
                "headline_hash": digest(normalized),
                "text_hash": digest(f"{normalized}\n{summary.casefold().strip()}"),
            }
        )
    return records

