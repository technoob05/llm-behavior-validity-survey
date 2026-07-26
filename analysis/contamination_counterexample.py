"""A ceiling-mediated contamination counterexample.

The mechanism. A 0/1 outcome with mean p has variance exactly p(1-p): for binary
data the spread is determined by the level. An item a model has memorised is
answered correctly under every phrasing, so p approaches 1 and its probe variance
approaches 0, not because the probe stopped mattering but because there is no
room left to move. Contamination, which pushes items toward the ceiling, therefore
compresses probe variance as a side effect of raising the level. At the item
level this is an identity, not an empirical finding, so the question worth asking
is what it does to phi, which is defined one level up on the REPORTED score.

The test. PromptEval gives, for each (subject, model), an items x 100-template
matrix of 0/1 correctness on MMLU, among the most exposed benchmarks in
circulation. We compute phi on the template-level accuracies twice: over all
items, and over only the items that are not near a ceiling or floor. If
contamination merely shifted levels, the two would agree. The gap between them is
the size of the bias that near-ceiling items inject into phi.

Note on the decomposition: PromptEval holds one observation per (item, template)
cell, so the item-by-template interaction cannot be separated from the residual.
We therefore report phi_strict, the probe MAIN effect share, which is the
conservative half of the index.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "prompteval"


def phi_strict(A):
    """Crossed item x template decomposition on a 0/1 matrix (items x templates).

    Returns the template (probe) main-effect share of total variance."""
    A = np.asarray(A, dtype=float)
    n_i, n_k = A.shape
    g = A.mean()
    a = A.mean(axis=1) - g                       # item effects
    b = A.mean(axis=0) - g                       # template (probe) effects
    ss_item = n_k * (a ** 2).sum()
    ss_probe = n_i * (b ** 2).sum()
    resid = A - (g + a[:, None] + b[None, :])
    ss_res = float((resid ** 2).sum())
    tot = ss_item + ss_probe + ss_res
    return (ss_probe / tot) if tot > 0 else np.nan


def main():
    rows = []
    for f in sorted(RAW.glob("*.parquet")):
        subject, model = f.stem.split("__", 1)
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        num = df.select_dtypes(include=[np.number])
        # PromptEval stores rows = prompt templates, columns = examples
        # (column names are example_*), so transpose to items x templates.
        A = num.to_numpy(dtype=float).T
        if A.shape[0] < 30 or A.shape[1] < 20:
            continue
        A = A[np.isfinite(A).all(axis=1)]
        if A.shape[0] < 30:
            continue
        p = A.mean(axis=1)
        keep = (p > 0.2) & (p < 0.8)             # drop ceiling and floor items
        if keep.sum() < 20:
            continue
        rows.append(dict(
            subject=subject, model=model,
            n_items=A.shape[0], n_templates=A.shape[1],
            frac_near_ceiling=float((p >= 0.9).mean()),
            frac_near_floor=float((p <= 0.1).mean()),
            phi_all=phi_strict(A),
            phi_mid=phi_strict(A[keep]),
            acc=float(A.mean()),
        ))

    t = pd.DataFrame(rows)
    print("subject-model matrices analysed: %d  (median %d items x %d templates)"
          % (len(t), t.n_items.median(), t.n_templates.median()))
    print("\nshare of items at a ceiling (p>=0.9): median %.1f%%   floor (p<=0.1): %.1f%%"
          % (100 * t.frac_near_ceiling.median(), 100 * t.frac_near_floor.median()))

    t["ratio"] = t.phi_mid / t.phi_all
    print("\n=== phi_strict computed with and without ceiling/floor items ===")
    print("  all items                : median phi = %.4f" % t.phi_all.median())
    print("  mid-difficulty items only: median phi = %.4f" % t.phi_mid.median())
    print("  ratio (mid / all)        : median %.2fx  (IQR %.2f to %.2f)"
          % (t.ratio.median(), t.ratio.quantile(.25), t.ratio.quantile(.75)))
    print("  matrices where dropping ceiling/floor RAISES phi: %d of %d"
          % (int((t.ratio > 1).sum()), len(t)))

    # does the bias track how many extreme items a matrix has?
    r = t[["frac_near_ceiling", "ratio"]].dropna()
    if len(r) > 5:
        rho = r.corr(method="spearman").iloc[0, 1]
        print("\n  Spearman(share of ceiling items, phi inflation when they are dropped)"
              " = %.2f" % rho)

    print("\ninterpretation: contamination raises p, and for binary outcomes a higher p")
    print("mechanically shrinks probe variance, so phi is DEFLATED on contaminated")
    print("items. The bias is conservative (it cannot make a fragile finding look")
    print("robust) but it means phi is not comparable across corpora of very")
    print("different difficulty without saying so.")

    t.to_csv(Path(__file__).resolve().parent / "contamination_phi_bias.csv", index=False)


if __name__ == "__main__":
    main()
