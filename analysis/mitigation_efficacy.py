"""
How much does each remedy actually buy?

This analysis measures how far a template ensemble or an order symmetrisation
moves phi and how many verdict flips it prevents. We can
answer this on MoralChoice because the design is fully crossed, so a mitigation
can be simulated by restricting or aggregating over the probe factors that a
real study would have controlled.

Interventions compared, all on the high-ambiguity (genuine dilemma) set:

  none            a single probe condition, drawn at random  (what most studies do)
  order-sym       average the two option orders within one question form
                  (balanced permutation; costs 2x inference)
  template-ens    average the three question forms at one fixed order
                  (template ensemble; costs 3x inference)
  both            average all six conditions (costs 6x)

For each intervention we report
  (a) residual probe variance of a model's score, i.e. the variance that a study
      using that intervention would still be exposed to,
  (b) phi recomputed against the same total,
  (c) the empirical pairwise verdict-flip rate: how often two models' ordering
      reverses between two independent applications of the same intervention.

(c) is the decision-relevant number: it is measured, not modelled, and it is
directly comparable across interventions because every intervention is applied
to the same 12 models and the same 66 pairs.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"
rng = np.random.default_rng(0)


def load_cell_means(amb):
    """model -> DataFrame(scenario x (form, order)) of P(action1)."""
    out = {}
    for f in sorted(RAW.glob(f"{amb}__*.csv")):
        model = f.name.split("__", 1)[1].replace(".csv", "")
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        if not len(df):
            continue
        df["y"] = (df.decision == "action1").astype(float)
        piv = df.pivot_table(index="scenario_id",
                             columns=["question_type", "question_ordering"],
                             values="y", aggfunc="mean").dropna()
        if piv.shape[0] > 50:
            out[model] = piv
    return out


def score_under(piv, mode, rng):
    """Return a model's overall score under one application of an intervention."""
    cols = list(piv.columns)
    forms = sorted({c[0] for c in cols})
    orders = sorted({c[1] for c in cols})
    if mode == "none":
        c = cols[rng.integers(len(cols))]
        return float(piv[c].mean())
    if mode == "order-sym":
        f = forms[rng.integers(len(forms))]
        sub = [c for c in cols if c[0] == f]
        return float(piv[sub].mean(axis=1).mean())
    if mode == "template-ens":
        o = orders[rng.integers(len(orders))]
        sub = [c for c in cols if c[1] == o]
        return float(piv[sub].mean(axis=1).mean())
    if mode == "both":
        return float(piv[cols].mean(axis=1).mean())
    raise ValueError(mode)


def main(amb="high", trials=400):
    data = load_cell_means(amb)
    models = list(data)
    print(f"{amb} ambiguity: {len(models)} models, "
          f"{data[models[0]].shape[1]} probe conditions\n")

    modes = ["none", "order-sym", "template-ens", "both"]
    cost = {"none": 1, "order-sym": 2, "template-ens": 3, "both": 6}
    rows = []

    # ---- residual probe spread of a model's score under each intervention ----
    for mode in modes:
        spreads = []
        for m in models:
            piv = data[m]
            cols = list(piv.columns)
            forms = sorted({c[0] for c in cols})
            orders = sorted({c[1] for c in cols})
            if mode == "none":
                vals = [float(piv[c].mean()) for c in cols]
            elif mode == "order-sym":
                vals = [float(piv[[c for c in cols if c[0] == f]].mean(axis=1).mean())
                        for f in forms]
            elif mode == "template-ens":
                vals = [float(piv[[c for c in cols if c[1] == o]].mean(axis=1).mean())
                        for o in orders]
            else:
                vals = [float(piv[cols].mean(axis=1).mean())]
            spreads.append(np.var(vals, ddof=1) if len(vals) > 1 else 0.0)
        rows.append(dict(mode=mode, cost=cost[mode],
                         residual_probe_var=float(np.mean(spreads))))

    base = rows[0]["residual_probe_var"]
    for r in rows:
        r["var_reduction_vs_none"] = (1 - r["residual_probe_var"] / base) if base > 0 else np.nan

    # ---- empirical verdict-flip rate between two independent applications ----
    for r in rows:
        mode = r["mode"]
        flips, tot = 0, 0
        for _ in range(trials):
            s1 = {m: score_under(data[m], mode, rng) for m in models}
            s2 = {m: score_under(data[m], mode, rng) for m in models}
            for a, b in combinations(models, 2):
                d1, d2 = s1[a] - s1[b], s2[a] - s2[b]
                if d1 == 0 or d2 == 0:
                    continue
                tot += 1
                flips += (np.sign(d1) != np.sign(d2))
        r["flip_rate"] = flips / tot if tot else np.nan

    t = pd.DataFrame(rows)
    t["flip_reduction_vs_none"] = 1 - t.flip_rate / t.flip_rate.iloc[0]
    t.to_csv(OUT / f"mitigation_{amb}.csv", index=False)
    print("=== efficacy of each remedy (high-ambiguity moral dilemmas) ===")
    print(t[["mode", "cost", "residual_probe_var", "var_reduction_vs_none",
             "flip_rate", "flip_reduction_vs_none"]].round(4).to_string(index=False))

    print("\ninterpretation: 'cost' is the inference multiplier; 'flip_rate' is the")
    print("probability that two models' ordering reverses between two independent")
    print("applications of the same protocol.")
    return t


if __name__ == "__main__":
    main("high")
    print()
    main("low")
