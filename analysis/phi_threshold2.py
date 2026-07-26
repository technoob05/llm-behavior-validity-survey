"""
Anchoring the phi threshold, corrected.

Why phi is decision-relevant. A study measures a model on N items with S samples
under K probe variants. Seed noise and item noise shrink as 1/(N*S); the probe
effect does NOT shrink with more items or more samples, only with more probes.
So in the regime real studies operate in (large N, S; K = 1) the residual share
of variance washes out and the probe share is exactly what is left. That is the
error that survives, and phi is its share.

For a pair of models truly separated by delta, with per-model probe effects of
variance s2_probe = phi * s2_total averaged over K probes,

    P(inversion) = Phi( -delta / sqrt(2 * phi * s2_total / K) )

We verify the algebra against simulation and then provide a conditional
sensitivity illustration. The illustration uses observed MoralChoice gaps but
sets a reference score scale by convention; it is not a data-derived universal
phi cutoff. For a real ranking claim, use the empirical probe variance and a
paired probe bootstrap instead.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from math import erf, sqrt

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"
rng = np.random.default_rng(0)


def Phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def observed_scale(amb="high"):
    """Total variance of the measured statistic and the observed gap sizes."""
    frames = []
    for f in sorted(RAW.glob(f"{amb}__*.csv")):
        model = f.name.split("__", 1)[1].replace(".csv", "")
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        if not len(df):
            continue
        df["y"] = (df.decision == "action1").astype(float)
        df["cond"] = df.question_type.astype(str) + "|" + df.question_ordering.astype(str)
        g = df.groupby("cond")["y"].mean()
        g.name = model
        frames.append(g)
    M = pd.concat(frames, axis=1).dropna()        # conditions x models
    per_model_mean = M.mean(axis=0)
    s2_probe_obs = float(M.var(axis=0, ddof=1).mean())     # within-model, across probes
    s2_between = float(per_model_mean.var(ddof=1))          # across models
    gaps = [abs(a - b) for a, b in combinations(per_model_mean.tolist(), 2)]
    return dict(s2_probe_obs=s2_probe_obs, s2_between=s2_between,
                gaps=np.array(gaps), models=list(M.columns), M=M)


def p_inversion_analytic(delta, s2_probe, K=1):
    if s2_probe <= 0:
        return 0.0
    return Phi(-delta / sqrt(2.0 * s2_probe / K))


def p_inversion_sim(delta, s2_probe, K=1, trials=200000):
    i1 = rng.normal(0, sqrt(s2_probe), size=(trials, K)).mean(axis=1)
    i2 = rng.normal(0, sqrt(s2_probe), size=(trials, K)).mean(axis=1)
    return float(((delta + i1 - i2) < 0).mean())


def main():
    for amb in ("high", "low"):
        o = observed_scale(amb)
        s2p = o["s2_probe_obs"]
        gaps = o["gaps"]
        print(f"=== {amb.upper()} AMBIGUITY: observed scale ===")
        print(f"  probe variance of a model's measured score (within-model, "
              f"across the 6 probe conditions): {s2p:.5f}  (sd {sqrt(s2p):.3f})")
        print(f"  between-model variance of the score: {o['s2_between']:.5f}")
        print(f"  pairwise |gap| between models: median {np.median(gaps):.3f}, "
              f"25th pct {np.percentile(gaps,25):.3f}")

        # predicted vs empirical flip rate at K=1
        pred = np.mean([p_inversion_analytic(d, s2p, K=1) for d in gaps])
        print(f"  predicted P(inversion) for a single-probe study, averaged over "
              f"observed gaps: {pred:.3f}")
        emp = pd.read_csv(OUT / f"inversions_{amb}.csv") if (OUT / f"inversions_{amb}.csv").exists() else None
        if emp is not None:
            print(f"  empirical flip rate across the 6 probe conditions: "
                  f"{emp.flipped.mean():.3f}")
        print()

    # ---- conditional sensitivity curve ----
    o = observed_scale("high")
    # Convention for plotting phi values on a common score scale. This is an
    # explicit sensitivity anchor, not an estimate of total variance.
    reference_phi = 0.5
    s2_total = o["s2_probe_obs"] / reference_phi
    print("=== P(inversion) vs phi, for a pair at a typical gap ===")
    print(f"  (conditional reference scale: phi={reference_phi:.2f} gives the observed "
          f"probe variance {o['s2_probe_obs']:.5f}; not a data-derived cutoff)")
    med_gap = float(np.median(o["gaps"]))
    small_gap = float(np.percentile(o["gaps"], 25))
    print(f"  typical gap delta = {med_gap:.3f}; small gap = {small_gap:.3f}\n")

    rows = []
    for phi in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        s2p = phi * s2_total
        for name, d in (("median_gap", med_gap), ("small_gap", small_gap)):
            for K in (1, 5, 20):
                pa = p_inversion_analytic(d, s2p, K)
                rows.append(dict(phi=phi, gap=name, delta=d, K=K, p_inv=pa))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "phi_threshold_curve.csv", index=False)
    for name in ("median_gap", "small_gap"):
        print(f"  gap = {name}")
        piv = t[t.gap == name].pivot(index="phi", columns="K", values="p_inv")
        print(piv.round(3).to_string(), "\n")

    print("=== conditional threshold: largest phi keeping P(inversion) <= tolerance ===")
    for tol in (0.05, 0.10):
        for name, d in (("median_gap", med_gap), ("small_gap", small_gap)):
            for K in (1, 5, 20):
                ok = [phi for phi in np.arange(0.01, 0.96, 0.01)
                      if p_inversion_analytic(d, phi * s2_total, K) <= tol]
                v = f"{max(ok):.2f}" if ok else "none"
                print(f"  tol={tol:.2f} gap={name:10s} K={K:2d} -> phi* = {v}")

    # analytic vs simulation check
    print("\n=== analytic vs simulation check ===")
    for phi in (0.1, 0.5, 0.9):
        s2p = phi * s2_total
        a = p_inversion_analytic(med_gap, s2p, 1)
        s = p_inversion_sim(med_gap, s2p, 1)
        print(f"  phi={phi:.1f}  analytic {a:.4f}  simulated {s:.4f}")


if __name__ == "__main__":
    main()
