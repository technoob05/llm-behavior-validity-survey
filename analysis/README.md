# Reproduction guide

This directory contains the analyses behind the survey's quantitative examples.
It is intentionally organized around claims rather than around exploratory
notebooks: each claim below names the source, command, output, and limitation.

## Environment

Use Python 3.11 or later and install the exact packages in
`requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r analysis\requirements.txt
```

All commands below are run from the repository root. Each script fixes its RNG
seed in source and writes its derived CSV beside the script.

## Fully local, public-source analyses

| Claim | Command | Output | Interpretation boundary |
|---|---|---|---|
| Harmonised strict phi comparison | `python analysis/harmonised_crosscorpus.py` | `harmonised_crosscorpus.csv`, `harmonised_template_draws.csv` | The cross-corpus result is illustrative and claim-specific. Report the 200-draw distribution, not a fixed multiplier. |
| Repeat-aware MoralChoice phi | `python analysis/phi_behavioural.py` | `phi_moralchoice.csv` | Broad phi removes estimated run noise from its numerator but retains run noise in total cell-mean variance. |
| Open-ended order audit | `python analysis/phi_openended.py` | `mtbench_winrates.csv` | Reports strict phi, signed bias, and disagreement. Broad phi is intentionally not estimated because released MT-Bench has one judge outcome per item-by-order cell. |
| Hierarchical uncertainty | `python analysis/hierarchical_bootstrap.py` | `hierarchical_bootstrap.csv` | Resamples items and probe conditions; thin probe pools yield wide intervals. |
| Pairwise ranking and POSIX comparison | `python analysis/paired_ranking_and_rho.py` | `paired_ranking_high.csv`, `paired_ranking_low.csv`, `probe_rho.csv` | Named comparisons require paired resampling. |
| Remedy and interaction analyses | `python analysis/mitigation_efficacy.py`; `python analysis/interaction_changes_verdict.py` | `mitigation_*.csv`, `interaction_verdict_*.csv` | Applies only to the released MoralChoice design. |
| Structural check | `python analysis/structural_replication.py` | `structural_*.csv`, `structural_summary_*.csv` | Rank correlations and the first singular component are descriptive, not a latent-mechanism identification. |

## Inputs and provenance

- `analysis/moralchoice/`: released MoralChoice response matrices.
- `analysis/prompteval/`: released PromptEval template matrices.
- `analysis/mtbench/`: released MT-Bench pairwise judgments.

Before archival submission, record a SHA-256 checksum for every input file and
the exact source-release URL/commit in `CLAIM_MANIFEST.csv`. Do not silently
substitute a newer hosted or benchmark release.

The packaged PromptEval core contains 84 subject-model matrices from 7 models
and 12 subjects. It supports the matched-item analysis. The full-corpus
category-level summary in the manuscript has derived CSVs but not every raw
matrix in this package, and is labelled accordingly.

## Results intentionally excluded from the reproducible core

The manuscript labels two companion-project checks as exploratory. Their code,
derived outputs, and unavailable input matrices are deliberately absent from
this public repository. They must not be used as load-bearing evidence until
their inputs can be released under an appropriate licence or fetched from an
immutable public source.

## Required release checks

1. Run every command in a fresh environment.
2. Compare each CSV against its expected schema and headline values.
3. Record input checksums, source version, model identifier, seed, and command.
4. Ensure no script contains an absolute machine-local input path.
5. Map every manuscript number to an output row in `CLAIM_MANIFEST.csv`.
