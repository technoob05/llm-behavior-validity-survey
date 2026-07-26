"""
Empirical estimation of the fragility fraction phi from PromptEval MMLU data.

Data: PromptEval/PromptEval_MMLU_correctness. For each (MMLU subject, model)
cell it stores a binary correctness matrix of K prompt templates x n items.

Model: two-way random-effects decomposition without replication
    y_ik = mu + a_i + b_k + eps_ik
    a_i  ~ item (signal) effect        -> sigma2_a
    b_k  ~ prompt template (probe)     -> sigma2_b
    eps  ~ residual                    -> sigma2_e
    phi  = sigma2_b / (sigma2_a + sigma2_b + sigma2_e)

Estimation is the standard ANOVA method-of-moments on the K x n matrix.
Binary outcomes are treated with a linear probability model, which is the
usual approximation for variance partitioning of accuracy; this is stated
as a caveat in the paper.
"""
import json
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0"}
PARQUET_API = ("https://huggingface.co/api/datasets/"
               "PromptEval/PromptEval_MMLU_correctness/parquet")


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def variance_components(Y):
    """Y: K x n matrix (rows = prompt templates, cols = items).
    Returns (sigma2_item, sigma2_template, sigma2_resid) clamped at 0."""
    K, n = Y.shape
    if K < 2 or n < 2:
        return None
    grand = Y.mean()
    row_m = Y.mean(axis=1)          # per template
    col_m = Y.mean(axis=0)          # per item

    ms_b = n * ((row_m - grand) ** 2).sum() / (K - 1)          # templates
    ms_a = K * ((col_m - grand) ** 2).sum() / (n - 1)          # items
    resid = Y - row_m[:, None] - col_m[None, :] + grand
    ms_e = (resid ** 2).sum() / ((K - 1) * (n - 1))

    s2_b = max((ms_b - ms_e) / n, 0.0)
    s2_a = max((ms_a - ms_e) / K, 0.0)
    s2_e = max(ms_e, 0.0)
    return s2_a, s2_b, s2_e


def one_cell(job):
    subject, model, urls = job
    try:
        dfs = [pd.read_parquet(u) for u in urls]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        return ("skip", subject, model, type(e).__name__)
    Y = df.to_numpy(dtype=float)          # K templates x n items
    vc = variance_components(Y)
    if vc is None:
        return None
    s2_a, s2_b, s2_e = vc
    tot = s2_a + s2_b + s2_e
    if tot <= 0:
        return None
    return dict(
        subject=subject, model=model,
        K=Y.shape[0], n=Y.shape[1],
        acc=float(Y.mean()),
        sigma2_item=s2_a, sigma2_template=s2_b, sigma2_resid=s2_e,
        phi=s2_b / tot,
        phi_excl_resid=(s2_b / (s2_a + s2_b)) if (s2_a + s2_b) > 0 else np.nan,
        acc_spread=float(Y.mean(axis=1).max() - Y.mean(axis=1).min()),
    )


def main():
    from concurrent.futures import ThreadPoolExecutor
    print("fetching parquet index ...", flush=True)
    idx = get_json(PARQUET_API)
    jobs = []
    for subject, splits in idx.items():
        for model, files in splits.items():
            urls = [f if isinstance(f, str) else f["url"] for f in files]
            jobs.append((subject, model, urls))
    print(f"{len(jobs)} (subject x model) cells to process", flush=True)

    rows, n_done = [], 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(one_cell, jobs):
            n_done += 1
            if isinstance(res, tuple):
                print("  skip", res[1], res[2], res[3], flush=True)
            elif res:
                rows.append(res)
            if n_done % 150 == 0:
                print(f"  {n_done}/{len(jobs)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phi_promptEval_cells.csv", index=False)
    print(f"\nTOTAL CELLS: {len(out)}  (subjects={out.subject.nunique()}, models={out.model.nunique()})")

    print("\n=== phi over all (subject x model) cells ===")
    print(out.phi.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())
    print(f"\nfraction of cells with phi >= 0.5 : {(out.phi >= .5).mean():.4f}")
    print(f"fraction of cells with phi >= 0.25: {(out.phi >= .25).mean():.4f}")
    print(f"median template-induced accuracy spread (max-min over templates): {out.acc_spread.median():.4f}")
    print(f"max template-induced accuracy spread: {out.acc_spread.max():.4f}")

    per_model = (out.groupby("model")
                    .agg(phi_median=("phi", "median"),
                         phi_mean=("phi", "mean"),
                         spread_median=("acc_spread", "median"),
                         acc=("acc", "mean"),
                         cells=("phi", "size"))
                    .sort_values("phi_median", ascending=False).round(4))
    per_model.to_csv(OUT / "phi_by_model.csv")
    print("\n=== phi by model ===")
    print(per_model.to_string())

    per_subj = (out.groupby("subject")
                   .agg(phi_median=("phi", "median"), cells=("phi", "size"))
                   .sort_values("phi_median", ascending=False).round(4))
    per_subj.to_csv(OUT / "phi_by_subject.csv")
    print("\n=== 10 subjects with highest median phi ===")
    print(per_subj.head(10).to_string())
    print("\n=== 5 subjects with lowest median phi ===")
    print(per_subj.tail(5).to_string())


if __name__ == "__main__":
    main()
