"""Prepare auditable CPU-only inputs for multilingual and multimodal follow-ups.

This script never calls a model and never reports a fragility result.  It checks
that public benchmark files have the expected paired structure, then emits
frozen JSONL prompt pools whose variant fields make later inference auditable.
Run from the repository root after downloading the pinned datasets:

    python experiments/prepare_extension_datasets.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = ROOT / "experiments" / "prepared"

# A deliberately diverse, declared primary language set.  The full downloaded
# Belebele release remains available for sensitivity checks and future variants.
BELEBELE_LANGUAGES = [
    "eng_Latn", "arb_Arab", "ben_Beng", "deu_Latn", "fra_Latn", "hin_Deva",
    "jpn_Jpan", "kor_Hang", "rus_Cyrl", "swh_Latn", "tel_Telu", "zho_Hans",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_mgsm(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path, columns=["question", "answer", "answer_number", "equation_solution"])
    return table.to_pylist()


def option_prompt(question: str, options: list[str], order: str) -> str:
    """Render a label-free option order so labels cannot be mistaken for content."""
    if order == "reverse":
        options = list(reversed(options))
    lines = [question, "", "Choices:"]
    lines.extend(f"{index}. {option}" for index, option in enumerate(options, start=1))
    lines.append("Reply with only the number of the best answer.")
    return "\n".join(lines)


def prepare_belebele(report: dict[str, object]) -> None:
    # The Dataset Server's Parquet conversion is used here because the Hub
    # archive is a convenience mirror and may be incompletely transferred by
    # some local clients.  Each file below is a 900-row test configuration.
    root = DATA / "belebele" / "parquet"
    files = {path.stem: path for path in root.glob("*.parquet")}
    missing = [language for language in BELEBELE_LANGUAGES if language not in files]
    if missing:
        raise FileNotFoundError(f"Missing declared Belebele languages: {missing}")

    pools: list[dict[str, object]] = []
    row_counts: dict[str, int] = {}
    answer_counts: Counter[str] = Counter()
    for language in BELEBELE_LANGUAGES:
        rows = pq.read_table(files[language]).to_pylist()
        row_counts[language] = len(rows)
        for row_index, row in enumerate(rows, start=1):
            options = [str(row[f"mc_answer{i}"]) for i in range(1, 5)]
            correct = int(row["correct_answer_num"])
            answer_counts[str(correct)] += 1
            for order in ("forward", "reverse"):
                expected_position = correct if order == "forward" else 5 - correct
                pools.append({
                    "dataset": "facebook/belebele",
                    "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
                    # Belebele's question_number repeats (it is not a stable
                    # item key), so keep the published, aligned row order.
                    "published_row_index": row_index,
                    "language": language,
                    "probe_axis": "option_order",
                    "variant": order,
                    "correct_position": expected_position,
                    "prompt": option_prompt(str(row["question"]), options, order),
                })
    if set(row_counts.values()) != {900}:
        raise ValueError(f"Belebele selected languages must each have 900 rows: {row_counts}")
    with (OUT / "belebele_language_order_pool.jsonl").open("w", encoding="utf-8") as handle:
        for row in pools:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["belebele"] = {
        "languages": BELEBELE_LANGUAGES,
        "aligned_items": 900,
        "prompt_records": len(pools),
        "original_correct_position_counts": dict(sorted(answer_counts.items())),
        "pairing_assumption": "published row index; question_number and link are not unique IDs",
        "design": "published-row x language x two label-free option orders",
    }


def prepare_mgsm(report: dict[str, object]) -> None:
    root = DATA / "mgsm"
    languages = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path / "test-00000-of-00001.parquet").is_file()
    )
    rows_by_language = {
        language: read_mgsm(root / language / "test-00000-of-00001.parquet")
        for language in languages
    }
    lengths = {language: len(rows) for language, rows in rows_by_language.items()}
    if set(lengths.values()) != {250}:
        raise ValueError(f"MGSM test rows must be 250 per language, found {lengths}")
    # MGSM publishes aligned translations without a stable cross-language item ID.
    # Preserve the published row index explicitly and record that assumption.
    records: list[dict[str, object]] = []
    for language, rows in rows_by_language.items():
        for index, row in enumerate(rows, start=1):
            records.append({
                "dataset": "juletxara/mgsm",
                "revision": "b2f13d426afe3be8d69a7e739b36724db8b66bbc",
                "published_row_index": index,
                "language": language,
                "probe_axis": "language",
                "prompt": str(row["question"]),
                "reference_answer": str(row["answer"]),
                "reference_number": int(row["answer_number"]),
            })
    with (OUT / "mgsm_language_pool.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["mgsm"] = {
        "languages": languages,
        "test_rows_per_language": 250,
        "prompt_records": len(records),
        "pairing_assumption": "published row index; confirm alignment against the upstream release before inference",
        "design": "open-ended item x language pool with frozen numeric reference answers",
    }


def inspect_parquet(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": parquet.metadata.num_rows,
        "columns": [field.name for field in parquet.schema_arrow],
        "bytes": path.stat().st_size,
    }


def inspect_secondary_sets(report: dict[str, object]) -> None:
    multilingual = DATA / "multilingual-benchmark"
    mmmu = DATA / "mmmu-pro"
    report["multilingual_benchmark"] = {
        "revision": "801297609d9203fefeeec8a0ca0601e58e18f549",
        "jsonl_files": sorted(str(path.relative_to(ROOT)).replace("\\", "/") for path in multilingual.rglob("*.jsonl")),
        "warning": "Machine-translated benchmark; use as a secondary robustness set, not interchangeable evidence.",
    }
    mmmu_files = sorted(mmmu.rglob("*.parquet"))
    report["mmmu_pro"] = {
        "revision": "563f3e84bb3b90893083a1f039cfa13077f2302b",
        "shards": [inspect_parquet(path) for path in mmmu_files],
        "warning": "Do not treat image availability or a changed option count as a neutral probe without an item-level construct check.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"generated_by": "experiments/prepare_extension_datasets.py"}
    prepare_belebele(report)
    prepare_mgsm(report)
    inspect_secondary_sets(report)
    write_json(OUT / "extension_data_audit.json", report)
    with (OUT / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "These files are frozen CPU-prepared inputs, not model results.\\n"
            "Run inference with a declared model, decoding configuration, and scoring plan before estimating phi.\\n"
        )
    print(json.dumps({key: report[key] for key in ("belebele", "mgsm")}, indent=2))


if __name__ == "__main__":
    main()
