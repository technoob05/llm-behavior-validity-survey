"""Reproducible existence and metadata checks for every cited bibliography entry.

This script checks local citation integrity, arXiv identifiers and downloaded
source availability, DOI registry metadata, and stable URLs. It does not decide
whether a source supports a manuscript claim; that requires full-text reading.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import difflib
import json
import pathlib
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "latex" / "paper.tex"
BIB_PATH = ROOT / "latex" / "references.bib"
VERIFY_ROOT = ROOT / "verify" / "full"
JSON_OUTPUT = ROOT / "analysis" / "citation_verification.json"
MD_OUTPUT = ROOT / "analysis" / "CITATION_VERIFICATION.md"
USER_AGENT = "llm-behavior-validity-survey/1.0 citation integrity audit"


@dataclasses.dataclass
class Entry:
    entry_type: str
    key: str
    fields: dict[str, str]


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def cited_keys(tex: str) -> list[str]:
    clean = strip_tex_comments(tex)
    result: list[str] = []
    for match in re.finditer(r"\\cite\w*\s*\{([^}]+)\}", clean, re.S):
        result.extend(key.strip() for key in match.group(1).split(",") if key.strip())
    return result


def balanced_value(text: str, start: int) -> tuple[str, int]:
    opener = text[start]
    if opener == "{":
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif char == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    return text[start + 1 : index], index + 1
        raise ValueError("unclosed braced BibTeX value")
    if opener == '"':
        index = start + 1
        while index < len(text):
            if text[index] == '"' and text[index - 1] != "\\":
                return text[start + 1 : index], index + 1
            index += 1
        raise ValueError("unclosed quoted BibTeX value")
    end = text.find(",", start)
    if end < 0:
        end = len(text)
    return text[start:end].strip(), end


def parse_fields(body: str) -> tuple[str, dict[str, str]]:
    comma = body.find(",")
    if comma < 0:
        raise ValueError("BibTeX entry has no key separator")
    key = body[:comma].strip()
    fields: dict[str, str] = {}
    cursor = comma + 1
    while cursor < len(body):
        while cursor < len(body) and (body[cursor].isspace() or body[cursor] == ","):
            cursor += 1
        if cursor >= len(body):
            break
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[cursor:])
        if not match:
            break
        name = match.group(1).lower()
        cursor += match.end()
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        value, cursor = balanced_value(body, cursor)
        fields[name] = value.strip()
    return key, fields


def parse_bibtex(text: str) -> list[Entry]:
    entries: list[Entry] = []
    for match in re.finditer(r"@([A-Za-z]+)\s*\{", text):
        entry_type = match.group(1).lower()
        start = match.end() - 1
        depth = 0
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if char == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif char == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            raise ValueError(f"unclosed BibTeX entry near byte {match.start()}")
        key, fields = parse_fields(text[start + 1 : end])
        entries.append(Entry(entry_type, key, fields))
    return entries


def arxiv_id(entry: Entry) -> str | None:
    haystack = " ".join(entry.fields.get(field, "") for field in ("eprint", "url", "journal", "note"))
    match = re.search(r"(?:arxiv[:./ ]|^)(\d{4}\.\d{4,5})(?:v\d+)?", haystack, re.I)
    return match.group(1) if match else None


def source_status(identifier: str | None) -> str:
    if not identifier:
        return "not-applicable"
    extracted = VERIFY_ROOT / identifier
    if extracted.is_dir() and any(extracted.rglob("*")):
        return "latex-source"
    for suffix in (".source", ".tar", ".tar.gz", ".bin"):
        path = VERIFY_ROOT / f"{identifier}{suffix}"
        if path.is_file() and path.stat().st_size > 100:
            if path.read_bytes()[:4] == b"%PDF":
                return "pdf-only-from-arxiv"
            return "archive-present"
    return "missing"


def plain_text(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", value)
    value = value.replace("{", " ").replace("}", " ")
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return " ".join(value.lower().split())


def title_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, plain_text(left), plain_text(right)).ratio()


def fetch_arxiv_metadata(identifiers: list[str]) -> dict[str, dict[str, str]]:
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    metadata: dict[str, dict[str, str]] = {}
    for offset in range(0, len(identifiers), 25):
        batch = identifiers[offset : offset + 25]
        url = "https://export.arxiv.org/api/query?id_list=" + ",".join(batch)
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("atom:entry", namespace):
            raw_id = item.findtext("atom:id", default="", namespaces=namespace)
            match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?$", raw_id)
            if not match:
                continue
            metadata[match.group(1)] = {
                "title": " ".join(item.findtext("atom:title", default="", namespaces=namespace).split()),
                "published": item.findtext("atom:published", default="", namespaces=namespace),
            }
        time.sleep(1)
    return metadata


def check_url(url: str) -> dict[str, object]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            allow_redirects=True,
            stream=True,
        )
        status = response.status_code
        final_url = response.url
        response.close()
        return {"ok": status < 400, "status": status, "final_url": final_url}
    except requests.RequestException as error:
        return {"ok": False, "error": str(error)}


def crossref_metadata(doi: str) -> dict[str, object]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if response.status_code != 200:
            return {"ok": False, "status": response.status_code}
        message = response.json()["message"]
        titles = message.get("title") or []
        return {
            "ok": True,
            "title": titles[0] if titles else "",
            "published": message.get("published-print") or message.get("published-online") or {},
        }
    except (requests.RequestException, KeyError, ValueError) as error:
        return {"ok": False, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip URL, arXiv API, and Crossref requests")
    args = parser.parse_args()

    tex = TEX_PATH.read_text(encoding="utf-8")
    entries = parse_bibtex(BIB_PATH.read_text(encoding="utf-8"))
    entry_map = {entry.key: entry for entry in entries}
    used = cited_keys(tex)
    unique_used = sorted(set(used))

    records: dict[str, dict[str, object]] = {}
    for key in unique_used:
        entry = entry_map.get(key)
        if entry is None:
            records[key] = {"key": key, "error": "missing-bibliography-entry"}
            continue
        identifier = arxiv_id(entry)
        records[key] = {
            "key": key,
            "entry_type": entry.entry_type,
            "title": entry.fields.get("title", ""),
            "year": entry.fields.get("year", ""),
            "url": entry.fields.get("url", ""),
            "doi": entry.fields.get("doi", ""),
            "arxiv_id": identifier,
            "source_status": source_status(identifier),
        }

    if not args.offline:
        identifiers = sorted({record["arxiv_id"] for record in records.values() if record.get("arxiv_id")})
        arxiv_metadata: dict[str, dict[str, str]] = {}
        arxiv_error = ""
        try:
            arxiv_metadata = fetch_arxiv_metadata(identifiers)
        except (requests.RequestException, ET.ParseError) as error:
            arxiv_error = str(error)

        urls = sorted({str(record["url"]) for record in records.values() if record.get("url")})
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            url_results = dict(zip(urls, executor.map(check_url, urls)))

        dois = sorted({str(record["doi"]) for record in records.values() if record.get("doi")})
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            doi_results = dict(zip(dois, executor.map(crossref_metadata, dois)))

        for record in records.values():
            identifier = record.get("arxiv_id")
            if identifier:
                remote = arxiv_metadata.get(str(identifier))
                record["arxiv_registry_ok"] = remote is not None
                if remote:
                    record["arxiv_title"] = remote["title"]
                    record["arxiv_title_similarity"] = round(
                        title_similarity(str(record["title"]), remote["title"]), 3
                    )
                elif arxiv_error:
                    record["arxiv_registry_error"] = arxiv_error
            url = record.get("url")
            if url:
                record["url_check"] = url_results[str(url)]
            doi = record.get("doi")
            if doi:
                remote_doi = doi_results[str(doi)]
                record["doi_check"] = remote_doi
                if remote_doi.get("title"):
                    record["doi_title_similarity"] = round(
                        title_similarity(str(record["title"]), str(remote_doi["title"])), 3
                    )

    duplicate_keys = sorted(
        key for key in {entry.key for entry in entries} if sum(item.key == key for item in entries) > 1
    )
    duplicate_titles: dict[str, list[str]] = {}
    for entry in entries:
        normalized = plain_text(entry.fields.get("title", ""))
        duplicate_titles.setdefault(normalized, []).append(entry.key)
    duplicate_titles = {
        title: keys for title, keys in duplicate_titles.items() if title and len(keys) > 1
    }

    summary = {
        "network_checks": "skipped" if args.offline else "attempted",
        "citation_key_mentions": len(used),
        "unique_cited_keys": len(unique_used),
        "bibliography_entries": len(entries),
        "missing_keys": sorted(set(unique_used) - set(entry_map)),
        "unused_keys": sorted(set(entry_map) - set(unique_used)),
        "duplicate_keys": duplicate_keys,
        "duplicate_titles": duplicate_titles,
        "arxiv_entries": sum(bool(record.get("arxiv_id")) for record in records.values()),
        "arxiv_source_missing": sorted(
            key for key, record in records.items() if record.get("source_status") == "missing"
        ),
        "arxiv_registry_unverified": sorted(
            key for key, record in records.items() if record.get("arxiv_id") and record.get("arxiv_registry_ok") is False
        ),
        "low_arxiv_title_similarity": sorted(
            key
            for key, record in records.items()
            if isinstance(record.get("arxiv_title_similarity"), float)
            and record["arxiv_title_similarity"] < 0.75
        ),
        "doi_indeterminate": sorted(
            key
            for key, record in records.items()
            if record.get("doi")
            and not record.get("doi_check", {}).get("ok", False)
            and (
                record.get("doi_check", {}).get("status") == 429
                or record.get("doi_check", {}).get("status", 0) >= 500
                or "error" in record.get("doi_check", {})
            )
        ),
        "doi_failures": sorted(
            key
            for key, record in records.items()
            if record.get("doi")
            and not record.get("doi_check", {}).get("ok", False)
            and record.get("doi_check", {}).get("status") not in (429, None)
            and record.get("doi_check", {}).get("status", 0) < 500
        ),
        "low_doi_title_similarity": sorted(
            key
            for key, record in records.items()
            if isinstance(record.get("doi_title_similarity"), float)
            and record["doi_title_similarity"] < 0.75
        ),
        "url_failures": sorted(
            key
            for key, record in records.items()
            if record.get("url")
            and "url_check" in record
            and not record.get("url_check", {}).get("ok", False)
        ),
    }
    payload = {"summary": summary, "records": records}
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Citation verification",
        "",
        "Generated by `python analysis/verify_bibliography.py`.",
        "",
        "This report verifies citation-key integrity, bibliographic identity, arXiv source",
        "availability, DOI registry metadata, and URL reachability. Claim-level support is",
        "reported separately because it requires reading each source in context.",
        "",
        (
            "Network checks were skipped; local integrity and source availability "
            "were still checked."
            if args.offline
            else "`arxiv_registry_unverified` and `doi_indeterminate` record inconclusive"
        ),
        (
            ""
            if args.offline
            else "rate-limited registry requests, not invalid citations. Local arXiv source"
        ),
        "" if args.offline else "availability is checked independently.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, list):
            lines.append(f"- `{key}`: {len(value)}")
        elif isinstance(value, dict):
            lines.append(f"- `{key}`: {len(value)}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Items requiring review", ""])
    for field in (
        "arxiv_source_missing",
        "arxiv_registry_unverified",
        "low_arxiv_title_similarity",
        "doi_indeterminate",
        "doi_failures",
        "low_doi_title_similarity",
        "url_failures",
    ):
        values = summary[field]
        lines.append(f"### {field}")
        lines.append("")
        lines.append(", ".join(f"`{item}`" for item in values) if values else "None.")
        lines.append("")
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
