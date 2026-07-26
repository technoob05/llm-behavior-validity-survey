"""Sensitivity of phi to the declared probe pool.

Phi is defined relative to a probe pool selected by the study, so a narrow or
convenient pool can make the estimate look artificially small.

A warning is also not a measurement. This script measures the exploit, which is
the first step to bounding it:

  (1) ADVERSARIAL POOL SELECTION. For a target pool size K', enumerate (or sample)
      the subsets of available probe variants and report the MINIMUM, median and
      MAXIMUM phi a reporter could obtain. The min-to-max range is the discretion
      an author has. On PromptEval, with 100 templates, the range is what an
      author picking 6 templates could report.

  (2) SATURATION. phi as a function of K', to answer "how many variants before
      phi stops moving". If phi has not saturated by the K' a study used, the
      reported number is a lower bound of unknown tightness.

  (3) A POOL-ADEQUACY DIAGNOSTIC that a reader can check without trusting the
      author: compare phi on a random half of the declared pool against phi on
      the whole pool. A pool that is already saturated gives the same answer on
      half of itself. A pool chosen to look good does not, because the variants
      that carry the fragility are the ones that were left out, and any random
      half of a narrow pool is narrower still. We report the statistic and its
      null distribution under honest (random) pool selection, so the diagnostic
      has a reference point rather than a vibe.

  (4) WHAT A PRE-DECLARED POOL BUYS. The min-max range shrinks as K' grows; we
      report the K' at which the adversarial range falls below the decision
      thresholds the paper publishes, which converts "declare your pool" from
      advice into a number.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MC, PE = HERE / "moralchoice", HERE / "prompteval"
RNG = np.random.default_rng(99)

MODELS = [
    "openai_gpt-4", "openai_gpt-3.5-turbo", "openai_text-davinci-003",
    "openai_text-davinci-002", "anthropic_claude-v1.3",
    "anthropic_claude-instant-v1.1", "google_text-bison-001",
    "google_flan-t5-xl", "cohere_command-xlarge", "ai21_j2-jumbo-instruct",
    "bigscience_bloomz-7b1", "meta_opt-iml-max-small",
]
MAX_SUBSETS = 1500


def phi_strict(M):
    M = np.asarray(M, float)
    I, K = M.shape
    if I < 3 or K < 2:
        return np.nan
    g = M.mean()
    ai, bk = M.mean(axis=1) - g, M.mean(axis=0) - g
    ms_i = K * (ai ** 2).sum() / (I - 1)
    ms_k = I * (bk ** 2).sum() / (K - 1)
    res = M - (g + ai[:, None] + bk[None, :])
    ms_e = (res ** 2).sum() / ((I - 1) * (K - 1))
    s2i, s2k, s2e = (max((ms_i - ms_e) / K, 0.), max((ms_k - ms_e) / I, 0.),
                     max(ms_e, 0.))
    tot = s2i + s2k + s2e
    return (s2k / tot) if tot > 0 else np.nan


def subsets(n, k, cap=MAX_SUBSETS):
    """All k-subsets of range(n) if there are few, else a random sample."""
    from math import comb
    if comb(n, k) <= cap:
        return list(combinations(range(n), k))
    seen = set()
    while len(seen) < cap:
        seen.add(tuple(sorted(RNG.choice(n, k, replace=False))))
    return list(seen)


def greedy(M, k, sense):
    """Greedily build a K'-subset that minimises (sense=-1) or maximises
    (sense=+1) phi. Random subset sampling understates the discretion an author
    has, because a motivated author searches rather than samples. This is the
    threat model: someone who tries."""
    M = np.asarray(M, float)
    K = M.shape[1]
    chosen = []
    remaining = list(range(K))
    # seed with the best pair, since phi needs K>=2
    best = None
    for i in range(K):
        for j in range(i + 1, K):
            v = phi_strict(M[:, [i, j]])
            if not np.isfinite(v):
                continue
            if best is None or sense * v > sense * best[0]:
                best = (v, [i, j])
    if best is None:
        return np.nan, []
    chosen = list(best[1])
    remaining = [c for c in remaining if c not in chosen]
    cur = best[0]
    while len(chosen) < k and remaining:
        cand = None
        for c in remaining:
            v = phi_strict(M[:, chosen + [c]])
            if not np.isfinite(v):
                continue
            if cand is None or sense * v > sense * cand[0]:
                cand = (v, c)
        if cand is None:
            break
        cur = cand[0]
        chosen.append(cand[1])
        remaining.remove(cand[1])
    return cur, chosen


def sweep(M, name, corpus, ks):
    """min / median / max phi over K'-subsets of the columns of M, plus a greedy
    adversarial min and max at each K'."""
    M = np.asarray(M, float)
    K = M.shape[1]
    rows = []
    for k in ks:
        if k > K:
            continue
        vals = [phi_strict(M[:, list(c)]) for c in subsets(K, k)]
        vals = np.array([v for v in vals if np.isfinite(v)])
        if not len(vals):
            continue
        # greedy search only in the range real studies use; at large K' the
        # search is quadratic in K and the exploit is not the interesting case
        if k <= 10:
            g_lo, _ = greedy(M, k, -1)
            g_hi, _ = greedy(M, k, +1)
        else:
            g_lo = g_hi = np.nan
        rows.append(dict(corpus=corpus, cell=name, K_available=K, K_pool=k,
                         n_subsets=len(vals), phi_min=vals.min(),
                         phi_p25=np.percentile(vals, 25),
                         phi_med=np.median(vals),
                         phi_p75=np.percentile(vals, 75),
                         phi_max=vals.max(), phi_full=phi_strict(M),
                         phi_greedy_min=g_lo, phi_greedy_max=g_hi))
    return rows


def half_pool_diagnostic(M, reps=400):
    """phi(random half of the pool) / phi(full pool). Returns median ratio."""
    M = np.asarray(M, float)
    K = M.shape[1]
    full = phi_strict(M)
    if not np.isfinite(full) or full <= 0 or K < 4:
        return np.nan
    h = K // 2
    r = []
    for _ in range(reps):
        cols = RNG.choice(K, h, replace=False)
        v = phi_strict(M[:, cols])
        if np.isfinite(v):
            r.append(v / full)
    return float(np.median(r)) if r else np.nan


def main():
    rows, diag = [], []

    # ------------------------------------------------------- MoralChoice, K=6
    print("loading MoralChoice ...", flush=True)
    for amb in ("high", "low"):
        for m in MODELS:
            f = MC / f"{amb}__{m}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                         "question_ordering", "eval_sample_nb",
                                         "decision"])
            df = df[df.decision.isin(["action1", "action2"])].copy()
            df["y"] = (df.decision == "action1").astype(float)
            df["cond"] = (df.question_type.astype(str) + "|"
                          + df.question_ordering.astype(str))
            df = df[df.eval_sample_nb == df.eval_sample_nb.min()]
            piv = df.pivot_table(index="scenario_id", columns="cond",
                                 values="y", aggfunc="mean").dropna()
            if piv.shape[0] < 30:
                continue
            M = piv.to_numpy(float)
            rows += sweep(M, f"{amb}|{m}", "MoralChoice", (2, 3, 4, 5, 6))
            diag.append(dict(corpus="MoralChoice", cell=f"{amb}|{m}",
                             ratio=half_pool_diagnostic(M)))

    # -------------------------------------------------- PromptEval, K=100
    print("loading PromptEval ...", flush=True)
    for f in sorted(PE.glob("*.parquet")):
        subj, model = f.stem.split("__", 1)
        A = pd.read_parquet(f).select_dtypes(include=[np.number]).to_numpy(float).T
        A = A[np.isfinite(A).all(axis=1)]
        if A.shape[0] < 30 or A.shape[1] < 20:
            continue
        rows += sweep(A, f"{subj}|{model}", "PromptEval",
                      (2, 3, 4, 5, 6, 8, 10, 20, 50, 100))
        diag.append(dict(corpus="PromptEval", cell=f"{subj}|{model}",
                         ratio=half_pool_diagnostic(A)))

    t = pd.DataFrame(rows)
    t.to_csv(HERE / "pool_gaming.csv", index=False)
    d = pd.DataFrame(diag)
    d.to_csv(HERE / "pool_adequacy.csv", index=False)

    # ------------------------------------------------------------ (1) exploit
    print("\n" + "=" * 90)
    print("(1) HOW MUCH DISCRETION DOES THE POOL CHOICE GIVE? "
          "phi_strict over K'-subsets")
    print("=" * 90)
    for corpus, g in t.groupby("corpus"):
        print(f"\n  {corpus}")
        print(f"  {'K_pool':>7s} {'phi min':>9s} {'phi median':>11s} "
              f"{'phi max':>9s} {'max/min':>9s} {'median n subsets':>17s}")
        for k, gg in g.groupby("K_pool"):
            lo, me, hi = gg.phi_min.median(), gg.phi_med.median(), gg.phi_max.median()
            ratio = (hi / lo) if lo > 1e-6 else np.inf
            print(f"  {k:7d} {lo:9.4f} {me:11.4f} {hi:9.4f} "
                  f"{ratio:9.1f} {gg.n_subsets.median():17.0f}")
    print("\n  reading: at the pool size studies actually use, an author choosing")
    print("  which variants to report moves phi between the min and max columns")
    print("  WITHOUT reporting anything false. That range is the exploit.")

    # -------------------------------------------------- against the thresholds
    print("\n" + "=" * 90)
    print("(2) DOES THE DISCRETION CROSS A PUBLISHED DECISION THRESHOLD?")
    print("=" * 90)
    print("  thresholds from the paper: phi* = 0.03 (K=1), 0.17 (K=5), 0.67 (K=20)")
    for corpus, g in t.groupby("corpus"):
        for k in (5, 6):
            gg = g[g.K_pool == k]
            if not len(gg):
                continue
            thr = 0.17
            straddle = ((gg.phi_min < thr) & (gg.phi_max > thr)).mean()
            print(f"  {corpus:12s} K'={k}: {100*straddle:.0f}% of cells have a pool "
                  f"choice that puts phi on EITHER side of phi*={thr}")

    # --------------------------------------------------------- (3) saturation
    print("\n" + "=" * 90)
    print("(3) SATURATION: phi as the pool grows (PromptEval, K up to 100)")
    print("=" * 90)
    pe = t[t.corpus == "PromptEval"]
    if len(pe):
        base = pe[pe.K_pool == 100].set_index("cell").phi_med
        print(f"  {'K_pool':>7s} {'median phi':>11s} {'as share of K=100':>19s} "
              f"{'IQR of that share':>20s}")
        for k, gg in pe.groupby("K_pool"):
            sh = (gg.set_index("cell").phi_med / base).dropna()
            if not len(sh):
                continue
            print(f"  {k:7d} {gg.phi_med.median():11.4f} {sh.median():19.2f} "
                  f"[{sh.quantile(.25):.2f}, {sh.quantile(.75):.2f}]".rjust(21))
        print("\n  reading: if the share column has not reached 1.0 by the K a study")
        print("  used, that study's phi is a lower bound.")

    # ---------------------------------------------------- (4) the diagnostic
    print("\n" + "=" * 90)
    print("(4) POOL-ADEQUACY DIAGNOSTIC: phi(random half of pool) / phi(full pool)")
    print("=" * 90)
    for corpus, g in d.groupby("corpus"):
        v = g.ratio.dropna()
        if not len(v):
            continue
        print(f"  {corpus:12s} median {v.median():.2f}   "
              f"IQR [{v.quantile(.25):.2f}, {v.quantile(.75):.2f}]   n={len(v)}")
    print("\n  A saturated pool returns a ratio near 1. A ratio well below 1 says")
    print("  the pool is still adding fragility as it grows, so the reported phi")
    print("  understates it. This is computable by a READER from released")
    print("  probe-level outputs, which is what makes it a safeguard rather than")
    print("  a request for good behaviour.")


if __name__ == "__main__":
    main()
