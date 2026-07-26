"""
The practitioner table Round-8 asked for: the largest fragility fraction a study
can tolerate, as a function of how many probe variants it runs (K) and how much
ranking error its reader will accept.

Same anchoring as phi_threshold2.py: the total variance is fixed so that the
observed MoralChoice probe variance corresponds to phi = 0.5, and delta is the
median observed gap between two models. Nothing here is assumed; both scales come
off the data.
"""
from itertools import combinations
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "moralchoice"
rng = np.random.default_rng(0)


def Phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def observed_scale(amb="high"):
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
    M = pd.concat(frames, axis=1).dropna()
    per_model = M.mean(axis=0)
    gaps = np.array([abs(a - b) for a, b in combinations(per_model.tolist(), 2)])
    return float(M.var(axis=0, ddof=1).mean()), gaps


def main():
    s2_probe_obs, gaps = observed_scale("high")
    s2_total = s2_probe_obs / 0.5              # anchor, as in phi_threshold2
    delta = float(np.median(gaps))
    print(f"anchored total variance = {s2_total:.5f}; median gap delta = {delta:.3f}\n")

    Ks = (1, 5, 10, 20)
    tols = (0.01, 0.05, 0.10)
    grid = np.arange(0.005, 1.0005, 0.005)
    tab = {}
    for tol in tols:
        row = {}
        for K in Ks:
            ok = [p for p in grid
                  if Phi(-delta / sqrt(2.0 * (p * s2_total) / K)) <= tol]
            row[K] = max(ok) if ok else float("nan")
        tab[tol] = row

    print("largest tolerable phi, by target inversion risk and probe-pool size K")
    print("risk \\ K   " + "".join("%8d" % K for K in Ks))
    for tol in tols:
        cells = "".join(
            ("     >1.0" if np.isnan(tab[tol][K]) else "%8.2f" % tab[tol][K])
            for K in Ks)
        print("  %3d%%     %s" % (int(tol * 100), cells))

    # sanity: analytic vs simulation at one cell
    def sim(d, s2p, K, trials=200000):
        a = rng.normal(0, sqrt(s2p), size=(trials, K)).mean(axis=1)
        b = rng.normal(0, sqrt(s2p), size=(trials, K)).mean(axis=1)
        return float(((d + a - b) < 0).mean())

    print("\nanalytic vs simulation (median gap, K=1):")
    for phi in (0.03, 0.17, 0.38):
        s2p = phi * s2_total
        print("  phi=%.2f  analytic %.4f  simulated %.4f"
              % (phi, Phi(-delta / sqrt(2.0 * s2p)), sim(delta, s2p, 1)))

    pd.DataFrame(tab).to_csv(Path(__file__).resolve().parent / "phi_practitioner_table.csv")


if __name__ == "__main__":
    main()
