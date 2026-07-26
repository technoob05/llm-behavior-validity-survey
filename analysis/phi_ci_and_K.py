"""
(a) Bootstrap confidence intervals for phi, and a non-parametric check on the
    normal approximation used in the inversion-risk derivation.
(b) A practical K-selection table: given an estimated phi and a target
    inversion risk, how many probe variants does a study need?

Both were asked for by all three Round-5 reviews.
"""
from pathlib import Path
from itertools import combinations
from math import erf, sqrt, ceil

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"
rng = np.random.default_rng(0)


def Phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def two_way(M):
    A, B = M.shape
    if A < 2 or B < 2:
        return None
    g = M.mean()
    ra, cb = M.mean(axis=1), M.mean(axis=0)
    ms_a = B * ((ra - g) ** 2).sum() / (A - 1)
    ms_b = A * ((cb - g) ** 2).sum() / (B - 1)
    res = M - ra[:, None] - cb[None, :] + g
    ms_e = (res ** 2).sum() / ((A - 1) * (B - 1))
    return (max((ms_a - ms_e) / B, 0.0), max((ms_b - ms_e) / A, 0.0), max(ms_e, 0.0))


def load(amb):
    out = {}
    for f in sorted(RAW.glob(f"{amb}__*.csv")):
        model = f.name.split("__", 1)[1].replace(".csv", "")
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        if not len(df):
            continue
        df["y"] = (df.decision == "action1").astype(float)
        df["cond"] = df.question_type.astype(str) + "|" + df.question_ordering.astype(str)
        piv = df.pivot_table(index="scenario_id", columns="cond",
                             values="y", aggfunc="mean").dropna()
        if piv.shape[0] > 50:
            out[model] = piv
    return out


def boot_phi(M, B=600):
    """Bootstrap over SCENARIOS (the exchangeable unit)."""
    n = M.shape[0]
    strict, broad = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        vc = two_way(M[idx])
        if not vc:
            continue
        a, b, e = vc
        tot = a + b + e
        if tot <= 0:
            continue
        strict.append(b / tot)
        broad.append((b + e) / tot)
    return np.array(strict), np.array(broad)


def part_a(amb="high"):
    data = load(amb)
    rows = []
    for m, piv in data.items():
        M = piv.to_numpy()
        vc = two_way(M)
        a, b, e = vc
        tot = a + b + e
        s, br = boot_phi(M)
        rows.append(dict(
            model=m,
            phi_strict=b / tot,
            s_lo=np.percentile(s, 2.5), s_hi=np.percentile(s, 97.5),
            phi_broad=(b + e) / tot,
            b_lo=np.percentile(br, 2.5), b_hi=np.percentile(br, 97.5),
        ))
    t = pd.DataFrame(rows).sort_values("phi_broad", ascending=False)
    t.to_csv(OUT / f"phi_ci_{amb}.csv", index=False)
    print(f"=== bootstrap 95% CIs for phi ({amb} ambiguity, resampling scenarios) ===")
    print(t.round(3).to_string(index=False))
    print(f"\n  median CI width, strict: "
          f"{(t.s_hi - t.s_lo).median():.3f}   broad: {(t.b_hi - t.b_lo).median():.3f}")
    return t


def part_b_normality(amb="high"):
    """Non-parametric check: is the normal approximation for inversion risk ok?"""
    data = load(amb)
    models = list(data)
    conds = list(next(iter(data.values())).columns)
    scores = {m: np.array([data[m][c].mean() for c in conds]) for m in models}
    # empirical: probability that two independently drawn conditions reverse a pair
    emp, norm = [], []
    for a, b in combinations(models, 2):
        da = scores[a]
        db = scores[b]
        delta = abs(da.mean() - db.mean())
        diffs = []
        for i in range(len(conds)):
            for j in range(len(conds)):
                diffs.append((da[i] - db[j]))
        diffs = np.array(diffs)
        sgn = np.sign(da.mean() - db.mean())
        emp.append(float((np.sign(diffs) != sgn).mean()))
        s2 = 0.5 * (da.var(ddof=1) + db.var(ddof=1))
        norm.append(Phi(-delta / sqrt(2 * s2)) if s2 > 0 else 0.0)
    emp, norm = np.array(emp), np.array(norm)
    print(f"\n=== normal approximation vs empirical inversion rate ({amb}) ===")
    print(f"  mean empirical {emp.mean():.3f}  vs  mean normal-model {norm.mean():.3f}")
    print(f"  median absolute discrepancy: {np.median(np.abs(emp - norm)):.3f}")
    print(f"  Spearman correlation: "
          f"{pd.Series(emp).corr(pd.Series(norm), method='spearman'):.3f}")


def part_c_table(amb="high"):
    """K needed to hold inversion risk below a target, given phi and delta."""
    data = load(amb)
    conds = list(next(iter(data.values())).columns)
    means = {m: np.mean([data[m][c].mean() for c in conds]) for m in data}
    gaps = [abs(a - b) for a, b in combinations(means.values(), 2)]
    s2_probe_obs = float(np.mean([np.var([data[m][c].mean() for c in conds], ddof=1)
                                  for m in data]))
    s2_total = s2_probe_obs / 0.5     # same anchoring as the threshold analysis
    print(f"\n=== K needed to hold inversion risk below a target ===")
    print(f"  anchored to observed scale: sigma^2_total = {s2_total:.4f}")
    for name, d in (("small gap (25th pct)", float(np.percentile(gaps, 25))),
                    ("typical gap (median)", float(np.median(gaps))),
                    ("large gap (75th pct)", float(np.percentile(gaps, 75)))):
        print(f"\n  {name}: delta = {d:.3f}")
        print(f"    {'phi':>6} | {'K for 10% risk':>15} | {'K for 5% risk':>14}")
        for phi in (0.05, 0.10, 0.25, 0.50, 0.75):
            out = []
            for tol in (0.10, 0.05):
                k = None
                for K in range(1, 2001):
                    if Phi(-d / sqrt(2 * phi * s2_total / K)) <= tol:
                        k = K
                        break
                out.append(str(k) if k else ">2000")
            print(f"    {phi:>6.2f} | {out[0]:>15} | {out[1]:>14}")


if __name__ == "__main__":
    part_a("high")
    part_b_normality("high")
    part_b_normality("low")
    part_c_table("high")
