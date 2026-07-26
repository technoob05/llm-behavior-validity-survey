"""When the item-by-probe interaction changes a study conclusion.

Table 1 of
the paper says a study running K probe variants may carry at most a certain phi
if it wants its model ranking to survive. Take a study that runs K=5 probes and
accepts a 5% chance of reporting two models in the wrong order: its budget is
phi <= 0.17.

For each of the 12 models in the crossed MoralChoice design we compute both
readings of the same data:

  phi_strict  the probe MAIN effect only, which is what an additive model reports
  phi_broad   the main effect plus the item-by-probe interaction, Eq. 1

and ask which side of the 0.17 line each falls on. Every model that passes under
phi_strict and fails under phi_broad is a study that would have been published as
sound and is not.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from fragility import fragility_pair

RAW = Path(__file__).resolve().parent / "moralchoice"

# budget from Table 1: K = 5 probes, 5% accepted inversion risk
BUDGET = {1: 0.03, 5: 0.17, 10: 0.34, 20: 0.67}


def load(amb="high"):
    out = {}
    for f in sorted(RAW.glob(f"{amb}__*.csv")):
        model = f.name.split("__", 1)[1].replace(".csv", "")
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "eval_sample_nb",
                                     "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        if not len(df):
            continue
        df["y"] = (df.decision == "action1").astype(float)
        keys = ["scenario_id", "question_type", "question_ordering"]
        piv = df.pivot_table(index="scenario_id", columns=keys[1:],
                             values="y", aggfunc="mean").dropna()
        within = df.pivot_table(index="scenario_id", columns=keys[1:],
                                values="y", aggfunc="var").loc[
                                    piv.index, piv.columns
                                ]
        if piv.shape[0] > 50:
            out[model] = (piv, within)
    return out


def components(piv):
    forms = sorted({c[0] for c in piv.columns})
    orders = sorted({c[1] for c in piv.columns})
    ni, nf, no = piv.shape[0], len(forms), len(orders)
    Y = np.zeros((ni, nf, no))
    for fi, f in enumerate(forms):
        for oi, o in enumerate(orders):
            Y[:, fi, oi] = piv[(f, o)].to_numpy()
    g = Y.mean()
    a = Y.mean(axis=(1, 2)) - g
    b = Y.mean(axis=(0, 2)) - g
    c = Y.mean(axis=(0, 1)) - g
    ab = Y.mean(axis=2) - g - a[:, None] - b[None, :]
    ac = Y.mean(axis=1) - g - a[:, None] - c[None, :]
    bc = Y.mean(axis=0) - g - b[:, None] - c[None, :]
    resid = Y - (g + a[:, None, None] + b[None, :, None] + c[None, None, :]
                 + ab[:, :, None] + ac[:, None, :] + bc[None, :, :])
    ss = dict(item=nf * no * (a ** 2).sum(),
              form=ni * no * (b ** 2).sum(),
              order=ni * nf * (c ** 2).sum(),
              item_form=no * (ab ** 2).sum(),
              item_order=nf * (ac ** 2).sum(),
              form_order=ni * (bc ** 2).sum(),
              resid=float((resid ** 2).sum()))
    tot = sum(ss.values())
    return {k: v / tot for k, v in ss.items()}


def main():
    for amb, label in (("high", "genuine dilemmas"), ("low", "clear-answer items")):
        data = load(amb)
        rows = []
        for m, (piv, within) in data.items():
            strict, broad = fragility_pair(
                piv.to_numpy(float), within.to_numpy(float), repeats=5
            )
            rows.append(dict(model=m, phi_strict=strict, phi_broad=broad))
        t = pd.DataFrame(rows).sort_values("phi_broad")
        if not len(t):
            continue

        print("=" * 74)
        print("%s  (%d models)" % (label.upper(), len(t)))
        print("=" * 74)
        for K in (5, 10):
            b = BUDGET[K]
            t["pass_strict"] = t.phi_strict <= b
            t["pass_broad"] = t.phi_broad <= b
            flip = t[t.pass_strict & ~t.pass_broad]
            print("\n  study design: K=%d probes, 5%% accepted inversion risk "
                  "-> budget phi <= %.2f" % (K, b))
            print("    would pass on the additive (strict) reading : %2d of %d"
                  % (int(t.pass_strict.sum()), len(t)))
            print("    actually passes once the interaction counts : %2d of %d"
                  % (int(t.pass_broad.sum()), len(t)))
            print("    VERDICT REVERSED for                        : %2d models"
                  % len(flip))
            if len(flip):
                for _, r in flip.iterrows():
                    print("        %-34s strict %.3f (pass)  broad %.3f (fail)"
                          % (r.model, r.phi_strict, r.phi_broad))
        print("\n  median phi_strict %.3f   median phi_broad %.3f   (ratio %.1fx)"
              % (t.phi_strict.median(), t.phi_broad.median(),
                 t.phi_broad.median() / max(t.phi_strict.median(), 1e-9)))
        print()
        t.to_csv(Path(__file__).resolve().parent / f"interaction_verdict_{amb}.csv",
                 index=False)


if __name__ == "__main__":
    main()
