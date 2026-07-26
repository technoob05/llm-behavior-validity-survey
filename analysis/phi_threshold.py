"""
Anchoring the phi threshold in a decision-theoretic outcome.

An uncalibrated rule such as "phi >= 0.5 is too fragile" is intuitive but ad
hoc. We instead anchor it in the outcome a reader actually cares about: the probability that
a single-probe measurement REVERSES the true ordering of two models (a false
dispositional conclusion of the form "M1 is more X than M2").

Part A (empirical). MoralChoice high-ambiguity: for every pair of models, how
often does the sign of (mean_y[M1] - mean_y[M2]) flip across the six probe
conditions (3 formats x 2 orders)? This is measured, not simulated.

Part B (simulation). A model's measured score under probe k is
    y_mk = mu_m + b_k + e_mk,   b_k ~ N(0, s2_probe),  e_mk ~ N(0, s2_resid)
with signal variance s2_signal = Var(mu). Given phi = s2_probe / total, we
compute P(inversion) for a pair separated by delta, under K probes averaged.
This yields a phi-to-error curve from which a threshold can be read off for a
stated error tolerance, instead of asserting 0.5.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- Part A
def empirical_inversions(amb="high"):
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
    if not frames:
        return None
    M = pd.concat(frames, axis=1).dropna()          # conditions x models
    conds = list(M.index)
    models = list(M.columns)

    flips, pairs = 0, 0
    detail = []
    for a, b in combinations(models, 2):
        d = M[a] - M[b]
        signs = np.sign(d.to_numpy())
        signs = signs[signs != 0]
        if len(signs) < 2:
            continue
        pairs += 1
        flipped = not (np.all(signs > 0) or np.all(signs < 0))
        flips += flipped
        detail.append(dict(m1=a, m2=b, flipped=flipped,
                           gap_mean=float(abs(d.mean())),
                           gap_min=float(abs(d).min()),
                           gap_max=float(abs(d).max())))
    det = pd.DataFrame(detail)
    print(f"=== Part A: empirical ranking inversions ({amb} ambiguity) ===")
    print(f"  {len(models)} models, {len(conds)} probe conditions, {pairs} pairs")
    print(f"  pairs whose ordering FLIPS across probe conditions: "
          f"{flips}/{pairs} = {flips/pairs:.3f}")
    for thr in (0.02, 0.05, 0.10, 0.20):
        sub = det[det.gap_mean < thr]
        if len(sub):
            print(f"  pairs with mean gap < {thr:.2f}: flip rate "
                  f"{sub.flipped.mean():.3f}  (n={len(sub)})")
    # how many distinct models are "the most moral" depending on the probe
    top = M.idxmax(axis=1)
    print(f"  distinct models ranked FIRST depending on probe condition: "
          f"{top.nunique()} of {len(models)}")
    print(f"    {dict(top.value_counts())}")
    det.to_csv(OUT / f"inversions_{amb}.csv", index=False)
    return det


# ---------------------------------------------------------------- Part B
def sim_inversion(phi, delta, K=1, n_models=2, trials=20000, total_var=0.05):
    """P(measured ordering reverses the true ordering) for a pair at gap delta."""
    s2_probe = phi * total_var
    s2_resid = (1 - phi) * total_var
    # shared probe effect b_k hits both models (a common format/order prior),
    # plus independent model-probe interaction
    b = rng.normal(0, np.sqrt(s2_probe), size=(trials, K))
    e1 = rng.normal(0, np.sqrt(s2_resid), size=(trials, K))
    e2 = rng.normal(0, np.sqrt(s2_resid), size=(trials, K))
    # the probe prior shifts the two models differently (interaction term):
    i1 = rng.normal(0, np.sqrt(s2_probe), size=(trials, K))
    i2 = rng.normal(0, np.sqrt(s2_probe), size=(trials, K))
    y1 = (delta + b + i1 + e1).mean(axis=1)
    y2 = (0.0 + b + i2 + e2).mean(axis=1)
    return float((y1 < y2).mean())


def threshold_curve():
    print("\n=== Part B: simulated P(ranking inversion) vs phi ===")
    deltas = [0.02, 0.05, 0.10]
    phis = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    for K in (1, 5, 10):
        for d in deltas:
            for phi in phis:
                rows.append(dict(K=K, delta=d, phi=phi,
                                 p_inv=sim_inversion(phi, d, K=K)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "phi_threshold_curve.csv", index=False)
    for K in (1, 5, 10):
        print(f"\n  K = {K} probe(s)")
        piv = t[t.K == K].pivot(index="phi", columns="delta", values="p_inv")
        print(piv.round(3).to_string())

    print("\n  phi at which P(inversion) first exceeds 0.05, by (K, delta):")
    for K in (1, 5, 10):
        for d in deltas:
            sub = t[(t.K == K) & (t.delta == d)].sort_values("phi")
            hit = sub[sub.p_inv > 0.05]
            v = f"{hit.phi.iloc[0]:.2f}" if len(hit) else ">0.90"
            print(f"    K={K:2d} delta={d:.2f} -> phi* = {v}")
    return t


if __name__ == "__main__":
    empirical_inversions("high")
    print()
    empirical_inversions("low")
    threshold_curve()
