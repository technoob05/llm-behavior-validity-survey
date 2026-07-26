"""Export every cited work to a reader-friendly Markdown catalogue."""

from __future__ import annotations

import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.verify_bibliography import cited_keys, parse_bibtex


TEX = ROOT / "latex" / "paper.tex"
BIB = ROOT / "latex" / "references.bib"
OUTPUT = ROOT / "docs" / "surveyed-literature.md"


def clean_tex(value: str) -> str:
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\textendash": "-",
        r"\textemdash": "-",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = re.sub(r"\\(?:textit|textbf|emph|url)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\['\"`^~=.uvHtcdbkr]\{?([A-Za-z])\}?", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+\*?", "", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value)
    return html.unescape(value).strip()


def preferred_url(fields: dict[str, str]) -> str:
    if fields.get("url"):
        return fields["url"]
    if fields.get("doi"):
        return "https://doi.org/" + fields["doi"]
    eprint = fields.get("eprint", "")
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", eprint):
        return "https://arxiv.org/abs/" + eprint
    return ""


def short_authors(value: str) -> str:
    authors = []
    for item in value.split(" and "):
        name = clean_tex(item)
        if "," in name:
            family, given = (part.strip() for part in name.split(",", 1))
            name = f"{given} {family}".strip()
        authors.append(name)
    if len(authors) > 6:
        return ", ".join(authors[:5]) + ", et al."
    return ", ".join(authors)


def main() -> None:
    entries = parse_bibtex(BIB.read_text(encoding="utf-8"))
    entry_map = {entry.key: entry for entry in entries}
    used = set(cited_keys(TEX.read_text(encoding="utf-8")))
    cited = [entry_map[key] for key in used]
    cited.sort(
        key=lambda entry: (
            -(int(entry.fields.get("year", "0")) if entry.fields.get("year", "").isdigit() else 0),
            clean_tex(entry.fields.get("author", "")).lower(),
            clean_tex(entry.fields.get("title", "")).lower(),
        )
    )

    lines = [
        "# Surveyed literature",
        "",
        (
            f"This catalogue lists all **{len(cited)} works cited by the paper**. "
            "It is generated from `latex/references.bib`, so the catalogue and "
            "manuscript cannot silently drift apart."
        ),
        "",
        "For claim-level checks, see "
        "[the citation audit](../analysis/CITATION_VERIFICATION.md).",
        "",
        "| Year | Paper | Authors | Citation key |",
        "|---:|---|---|---|",
    ]
    for entry in cited:
        fields = entry.fields
        title = clean_tex(fields.get("title", "Untitled"))
        url = preferred_url(fields)
        paper = f"[{title}]({url})" if url else title
        authors = short_authors(fields.get("author", ""))
        lines.append(
            f"| {fields.get('year', 'n.d.')} | {paper} | {authors} | `{entry.key}` |"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(cited)} cited works to {OUTPUT}")


if __name__ == "__main__":
    main()
