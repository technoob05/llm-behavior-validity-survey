"""
Fragility fraction phi at the level a dispositional claim is actually made:
the REPORTED ACCURACY, not the per-item outcome.

Stage 1: for every (MMLU subject, model) cell of PromptEval_MMLU_correctness,
reduce the K x n binary matrix to K template-level accuracies (row means).
This yields a tensor acc[subject, model, template].

Stage 2: two-way random-effects decompositions of that accuracy.
  (a) phi_subject  (within a model): template variance vs subject variance.
      "Does the prompt move the score more than the topic does?"
  (b) phi_model    (within a subject): template variance vs model variance.
      "Does the prompt move the score more than the model identity does?"
      This is the decisive test of the survey's thesis.

Binary outcomes are aggregated before decomposition, so the linear model is
applied to accuracies (bounded 0-1), the standard practice for variance
partitioning of benchmark scores.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CACHE = OUT / "template_acc.csv"
UA = {"User-Agent": "Mozilla/5.0"}
PARQUET_API = ("https://huggingface.co/api/datasets/"
               "PromptEval/PromptEval_MMLU_correctness/parquet")


def get_json(url, tries=6):
    import time
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(15 * (k + 1))


def fetch_cell(job, tries=5):
    import time
    subject, model, urls = job
    df = None
    for k in range(tries):
        try:
            df = pd.concat([pd.read_parquet(u) for u in urls], ignore_index=True)
            break
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(8 * (k + 1))
    if df is None:
        return None
    Y = df.to_numpy(dtype=float)            # K templates x n items
    acc = Y.mean(axis=1)                    # per-template accuracy
    return pd.DataFrame({"subject": subject, "model": model,
                         "template": np.arange(len(acc)), "acc": acc})


def build_cache():
    idx = get_json(PARQUET_API)
    jobs = [(s, m, [f if isinstance(f, str) else f["url"] for f in files])
            for s, splits in idx.items() for m, files in splits.items()]
    print(f"fetching {len(jobs)} cells ...", flush=True)
    parts = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for i, r in enumerate(ex.map(fetch_cell, jobs), 1):
            if r is not None:
                parts.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    long = pd.concat(parts, ignore_index=True)
    long.to_csv(CACHE, index=False)
    return long


def two_way(M):
    """M: A x B matrix. Returns (sigma2_A, sigma2_B, sigma2_resid)."""
    A, B = M.shape
    if A < 2 or B < 2:
        return None
    g = M.mean()
    ra, cb = M.mean(axis=1), M.mean(axis=0)
    ms_a = B * ((ra - g) ** 2).sum() / (A - 1)
    ms_b = A * ((cb - g) ** 2).sum() / (B - 1)
    res = M - ra[:, None] - cb[None, :] + g
    ms_e = (res ** 2).sum() / ((A - 1) * (B - 1))
    return (max((ms_a - ms_e) / B, 0.0),
            max((ms_b - ms_e) / A, 0.0),
            max(ms_e, 0.0))


def main():
    long = pd.read_csv(CACHE) if CACHE.exists() else build_cache()
    print(f"\nrows={len(long)}  subjects={long.subject.nunique()}  "
          f"models={long.model.nunique()}  templates={long.template.nunique()}")

    # ---------- (a) within model: template vs subject ----------
    rec = []
    for model, g in long.groupby("model"):
        M = g.pivot_table(index="subject", columns="template", values="acc").to_numpy()
        M = M[~np.isnan(M).any(axis=1)]
        vc = two_way(M)
        if not vc:
            continue
        s2_subj, s2_tmpl, s2_e = vc
        tot = s2_subj + s2_tmpl + s2_e
        rec.append(dict(model=model, sigma2_subject=s2_subj,
                        sigma2_template=s2_tmpl, sigma2_resid=s2_e,
                        phi_subject=s2_tmpl / tot,
                        ratio_tmpl_over_subj=s2_tmpl / s2_subj if s2_subj > 0 else np.nan,
                        mean_acc=float(M.mean()),
                        tmpl_spread=float(M.mean(axis=0).max() - M.mean(axis=0).min())))
    a = pd.DataFrame(rec).sort_values("phi_subject", ascending=False)
    a.round(5).to_csv(OUT / "phi_within_model.csv", index=False)
    print("\n=== (a) WITHIN MODEL: template variance vs subject variance ===")
    print(a[["model", "phi_subject", "ratio_tmpl_over_subj", "tmpl_spread", "mean_acc"]]
          .round(4).to_string(index=False))
    print(f"\nmedian phi_subject = {a.phi_subject.median():.4f}")

    # ---------- (b) within subject: template vs MODEL identity ----------
    rec = []
    for subj, g in long.groupby("subject"):
        M = g.pivot_table(index="model", columns="template", values="acc").to_numpy()
        M = M[~np.isnan(M).any(axis=1)]
        vc = two_way(M)
        if not vc:
            continue
        s2_model, s2_tmpl, s2_e = vc
        tot = s2_model + s2_tmpl + s2_e
        rec.append(dict(subject=subj, sigma2_model=s2_model,
                        sigma2_template=s2_tmpl, sigma2_resid=s2_e,
                        phi_model=s2_tmpl / tot,
                        ratio_tmpl_over_model=s2_tmpl / s2_model if s2_model > 0 else np.nan))
    b = pd.DataFrame(rec).sort_values("phi_model", ascending=False)
    b.round(5).to_csv(OUT / "phi_within_subject.csv", index=False)
    print("\n=== (b) WITHIN SUBJECT: template variance vs MODEL-IDENTITY variance ===")
    print(b.head(12)[["subject", "phi_model", "ratio_tmpl_over_model"]].round(4).to_string(index=False))
    print(f"\nmedian phi_model = {b.phi_model.median():.4f}")
    print(f"median (template var / model var) = {b.ratio_tmpl_over_model.median():.4f}")
    print(f"subjects where template var EXCEEDS model var: "
          f"{(b.ratio_tmpl_over_model > 1).sum()} / {len(b)}")
    print(f"subjects with phi_model >= 0.25: {(b.phi_model >= .25).sum()} / {len(b)}")
    print(f"subjects with phi_model >= 0.50: {(b.phi_model >= .50).sum()} / {len(b)}")

    # ---------- (c) ranking instability ----------
    piv = long.pivot_table(index="model", columns=["subject", "template"], values="acc")
    per_tmpl = long.pivot_table(index="model", columns="template", values="acc")
    ranks = per_tmpl.rank(ascending=False, axis=0)
    top1 = per_tmpl.idxmax(axis=0)
    print("\n=== (c) leaderboard instability across templates ===")
    print(f"distinct models taking rank 1 across {per_tmpl.shape[1]} templates: {top1.nunique()}")
    print(top1.value_counts().to_string())
    spread = ranks.max(axis=1) - ranks.min(axis=1)
    print("\nrank range per model across templates:")
    print(spread.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
