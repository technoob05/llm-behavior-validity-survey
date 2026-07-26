"""Lightweight no-GPU checks for the public-core artifact."""
from __future__ import annotations
import csv
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def require(path: Path) -> None:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise AssertionError(f"Missing or empty: {path}")

def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(
            f"{label}: expected {expected} +/- {tolerance}, got {actual}"
        )

def main() -> None:
    for directory in ("moralchoice", "prompteval", "mtbench"):
        require(ROOT / "analysis" / directory)
    for script in ("harmonised_crosscorpus.py", "phi_openended.py", "hierarchical_bootstrap.py", "paired_ranking_and_rho.py"):
        require(ROOT / "analysis" / script)
    manifest = ROOT / "analysis" / "CLAIM_MANIFEST.csv"
    require(manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 12 or any(not row["claim_id"] for row in rows):
        raise AssertionError("Claim manifest is incomplete")

    analysis = ROOT / "analysis"
    moral = pd.read_csv(analysis / "phi_moralchoice.csv")
    close(
        moral.loc[moral.ambiguity == "high", "phi_broad"].median(),
        0.382,
        0.002,
        "dilemma broad phi",
    )
    close(
        moral.loc[moral.ambiguity == "low", "phi_broad"].median(),
        0.665,
        0.002,
        "clear-answer broad phi",
    )
    draws = pd.read_csv(analysis / "harmonised_template_draws.csv")
    close(
        draws.dilemma_to_capability.median(),
        20.9,
        0.1,
        "template-draw ratio",
    )
    paired = pd.read_csv(analysis / "paired_ranking_high.csv")
    n_supported = int(
        (
            (paired.model != "openai_gpt-4")
            & (paired.p_more_fragile_broad > 0.975)
        ).sum()
    )
    if n_supported != 5:
        raise AssertionError(f"paired survivor count: expected 5, got {n_supported}")
    print(
        f"PASS: inputs, {len(rows)} claim mappings, and headline numeric "
        "regressions are consistent"
    )

if __name__ == "__main__":
    main()
