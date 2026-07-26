"""Uncertainty from sampling both items and probe conditions.

With K = 6 probe conditions, a scenario-only bootstrap conditions on the
observed probe pool and therefore omits uncertainty from probe selection.

This script recomputes the interval three ways on the same data:

  (a) scenarios only          - what was reported; treats the probe pool as fixed
  (b) probe conditions only   - the neglected axis, on its own
  (c) hierarchical, both      - resample scenarios AND probe conditions

and adds a delete-one jackknife over probe conditions.  The six compound
form-by-order cells are treated as a finite pool.  These resamples are
sensitivity intervals for that pool, not population confidence intervals over
all possible wordings and orders.

It then asks the question that actually matters for the paper's conclusions: does
the ORDERING of models by fragility survive the wider intervals? The claim at
risk is "GPT-4 is the least probe-dependent model".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from fragility import fragility_pair

HERE = Path(__file__).resolve().parent
MC = HERE / "moralchoice"
RNG = np.random.default_rng(11)
B = 2000

MODELS = [
    "openai_gpt-4", "openai_gpt-3.5-turbo", "openai_text-davinci-003",
    "openai_text-davinci-002", "anthropic_claude-v1.3",
    "anthropic_claude-instant-v1.1", "google_text-bison-001",
    "google_flan-t5-xl", "cohere_command-xlarge", "ai21_j2-jumbo-instruct",
    "bigscience_bloomz-7b1", "meta_opt-iml-max-small",
]


def load(amb, model):
    f = MC / f"{amb}__{model}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                 "question_ordering", "eval_sample_nb",
                                 "decision"])
    df = df[df.decision.isin(["action1", "action2"])].copy()
    df["y"] = (df.decision == "action1").astype(float)
    df["cond"] = df.question_type.astype(str) + "|" + df.question_ordering.astype(str)
    piv = df.pivot_table(index="scenario_id", columns="cond", values="y",
                         aggfunc="mean").dropna()
    within = df.pivot_table(index="scenario_id", columns="cond", values="y",
                            aggfunc="var").loc[piv.index, piv.columns]
    return (piv.to_numpy(float), within.to_numpy(float)) if piv.shape[0] >= 30 else None


def boot(M, V, mode, reps=B):
    """mode: 'items' | 'probes' | 'both'."""
    I, K = M.shape
    out = []
    for _ in range(reps):
        ri = RNG.integers(0, I, I) if mode in ("items", "both") else np.arange(I)
        rk = RNG.integers(0, K, K) if mode in ("probes", "both") else np.arange(K)
        s, b = fragility_pair(
            M[np.ix_(ri, rk)], V[np.ix_(ri, rk)], repeats=5
        )
        if np.isfinite(s):
            out.append((s, b))
    if not out:
        return None
    a = np.array(out)
    return a


def ci(a, col):
    lo, hi = np.percentile(a[:, col], [2.5, 97.5])
    return lo, hi, hi - lo


def main():
    recs = []
    for amb in ("high", "low"):
        print(f"\n=== {amb.upper()} AMBIGUITY ===", flush=True)
        print(f"{'model':32s} {'phi_s':>7s} | {'items-only CI':>22s} "
              f"{'probes-only':>22s} {'hierarchical':>22s} {'jack range':>12s}")
        for m in MODELS:
            loaded = load(amb, m)
            if loaded is None:
                continue
            M, V = loaded
            s_hat, b_hat = fragility_pair(M, V, repeats=5)
            res = {}
            for mode in ("items", "probes", "both"):
                a = boot(M, V, mode)
                res[mode] = ci(a, 0) if a is not None else (np.nan,) * 3
                res[mode + "_broad"] = ci(a, 1) if a is not None else (np.nan,) * 3
            # delete-one jackknife over probe conditions
            jk = [
                fragility_pair(
                    np.delete(M, k, axis=1),
                    np.delete(V, k, axis=1),
                    repeats=5,
                )[0]
                for k in range(M.shape[1])
            ]
            jk = [v for v in jk if np.isfinite(v)]
            jrange = (max(jk) - min(jk)) if jk else np.nan

            recs.append(dict(ambiguity=amb, model=m, n_items=M.shape[0],
                             K=M.shape[1], phi_strict=s_hat, phi_broad=b_hat,
                             ci_items_w=res["items"][2],
                             ci_probes_w=res["probes"][2],
                             ci_both_w=res["both"][2],
                             ci_both_lo=res["both"][0], ci_both_hi=res["both"][1],
                             ci_items_lo=res["items"][0], ci_items_hi=res["items"][1],
                             ci_both_broad_w=res["both_broad"][2],
                             ci_items_broad_w=res["items_broad"][2],
                             jack_range=jrange,
                             jack_min=min(jk) if jk else np.nan,
                             jack_max=max(jk) if jk else np.nan))
            print(f"{m:32s} {s_hat:7.4f} | "
                  f"[{res['items'][0]:.3f},{res['items'][1]:.3f}] w={res['items'][2]:.3f} "
                  f"[{res['probes'][0]:.3f},{res['probes'][1]:.3f}] w={res['probes'][2]:.3f} "
                  f"[{res['both'][0]:.3f},{res['both'][1]:.3f}] w={res['both'][2]:.3f} "
                  f"{jrange:12.3f}", flush=True)

    t = pd.DataFrame(recs)
    t.to_csv(HERE / "hierarchical_bootstrap.csv", index=False)

    print("\n" + "=" * 78)
    print("MEDIAN 95% INTERVAL WIDTH FOR phi_strict, BY WHAT IS RESAMPLED")
    print("=" * 78)
    for amb, d in t.groupby("ambiguity"):
        print(f"  {amb:5s}  items only {d.ci_items_w.median():.3f}   "
              f"probes only {d.ci_probes_w.median():.3f}   "
              f"BOTH {d.ci_both_w.median():.3f}   "
              f"inflation {d.ci_both_w.median()/max(d.ci_items_w.median(),1e-9):.1f}x")
    print("\n  phi_broad: items only %.3f -> both %.3f"
          % (t.ci_items_broad_w.median(), t.ci_both_broad_w.median()))
    print("  jackknife-over-probes range of phi_strict: median %.3f, max %.3f"
          % (t.jack_range.median(), t.jack_range.max()))

    # ---- does the fragility ordering survive?
    print("\n" + "=" * 78)
    print("WHICH CONCLUSIONS SURVIVE THE WIDER INTERVAL")
    print("=" * 78)
    for amb, d in t.groupby("ambiguity"):
        d = d.sort_values("phi_strict")
        best = d.iloc[0]
        print(f"\n  [{amb}] least fragile by point estimate: {best.model} "
              f"(phi={best.phi_strict:.4f})")
        # count models whose hierarchical CI excludes the best model's point est.
        sep_h = int((d.ci_both_lo > best.phi_strict).sum())
        sep_i = int((d.ci_items_lo > best.phi_strict).sum())
        print(f"        models separated from it: {sep_i} of {len(d)-1} "
              f"under items-only, {sep_h} of {len(d)-1} under hierarchical")
        # pairwise: how many of the 66 ordered pairs are still distinguishable?
        n = len(d)
        pairs = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = d.iloc[i], d.iloc[j]
                tot += 1
                if a.ci_both_hi < b.ci_both_lo or b.ci_both_hi < a.ci_both_lo:
                    pairs += 1
        print(f"        pairs of models still distinguishable: {pairs}/{tot} "
              f"({100*pairs/max(tot,1):.0f}%) under hierarchical intervals")


if __name__ == "__main__":
    main()
