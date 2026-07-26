"""
A full variance decomposition of a behavioural measurement.

The two-way estimator used earlier attributes variance to item and to probe.
Three questions are left open by it: how much variance sits in the item-by-probe
INTERACTION, how much is run-to-run SAMPLING noise, and how the probe share
splits between the FORMAT and the ORDER factor. MoralChoice answers all three,
because its design is fully crossed:

    scenario  x  question form (3)  x  option order (2)  x  sample (5)

We fit the crossed random-effects model

    y_{i,f,o,s} = mu + a_i + b_f + c_o + (ab)_{if} + (ac)_{io} + (bc)_{fo} + eps

by ANOVA sums of squares on the balanced cell means. The between-cell components
are reported as shares of between-cell variation. The within-cell sampling
variance is retained separately on its raw response scale and is therefore not
another share in the same partition. This diagnostic refines the joint-probe
decomposition but is not used to recompute the headline fragility fraction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"


def load(amb):
    frames = []
    for f in sorted(RAW.glob(f"{amb}__*.csv")):
        model = f.name.split("__", 1)[1].replace(".csv", "")
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "eval_sample_nb",
                                     "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        if not len(df):
            continue
        df["y"] = (df.decision == "action1").astype(float)
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def components(d):
    """Balanced-cell ANOVA over scenario x form x order, with samples as reps."""
    cell = (d.groupby(["scenario_id", "question_type", "question_ordering"])["y"]
              .agg(["mean", "var", "count"]))
    # within-cell (sampling / seed) variance
    s2_seed = float(cell["var"].dropna().mean())

    m = cell["mean"].reset_index()
    piv = m.pivot_table(index="scenario_id",
                        columns=["question_type", "question_ordering"],
                        values="mean").dropna()
    if piv.shape[0] < 30:
        return None
    forms = sorted({c[0] for c in piv.columns})
    orders = sorted({c[1] for c in piv.columns})
    n_i, n_f, n_o = piv.shape[0], len(forms), len(orders)
    Y = np.zeros((n_i, n_f, n_o))
    for fi, f in enumerate(forms):
        for oi, o in enumerate(orders):
            Y[:, fi, oi] = piv[(f, o)].to_numpy()

    g = Y.mean()
    a = Y.mean(axis=(1, 2)) - g                     # item
    b = Y.mean(axis=(0, 2)) - g                     # form
    c = Y.mean(axis=(0, 1)) - g                     # order
    ab = Y.mean(axis=2) - g - a[:, None] - b[None, :]
    ac = Y.mean(axis=1) - g - a[:, None] - c[None, :]
    bc = Y.mean(axis=0) - g - b[:, None] - c[None, :]

    ss = dict(
        item=n_f * n_o * (a ** 2).sum(),
        form=n_i * n_o * (b ** 2).sum(),
        order=n_i * n_f * (c ** 2).sum(),
        item_x_form=n_o * (ab ** 2).sum(),
        item_x_order=n_f * (ac ** 2).sum(),
        form_x_order=n_i * (bc ** 2).sum(),
    )
    resid = Y - (g + a[:, None, None] + b[None, :, None] + c[None, None, :]
                 + ab[:, :, None] + ac[:, None, :] + bc[None, :, :])
    ss["resid_3way"] = float((resid ** 2).sum())
    tot = sum(ss.values())
    share = {k: v / tot for k, v in ss.items()}
    share["seed_within_cell_var"] = s2_seed
    share["_n_items"] = n_i
    return share


def main():
    for amb in ("high", "low"):
        d = load(amb)
        rows = []
        for model, g in d.groupby("model"):
            r = components(g)
            if r:
                r["model"] = model
                rows.append(r)
        t = pd.DataFrame(rows).set_index("model")
        t.to_csv(OUT / f"variance_components_{amb}.csv")

        print(f"\n{'='*72}\n{amb.upper()} AMBIGUITY  ({len(t)} models, "
              f"{int(t._n_items.median())} scenarios)\n{'='*72}")
        cols = ["item", "form", "order", "item_x_form", "item_x_order",
                "form_x_order", "resid_3way"]
        print("median share of total variance in the scenario x form x order design:")
        for c in cols:
            print(f"   {c:16s} {t[c].median():.4f}")
        # With the joint probe k=(form, order), form-by-order is part of the
        # probe main effect and the three-way term is item-by-joint-probe.
        probe_main = t["form"] + t["order"] + t["form_x_order"]
        probe_inter = t["item_x_form"] + t["item_x_order"] + t["resid_3way"]
        print(f"\n   joint-probe MAIN effect              median {probe_main.median():.4f}")
        print(f"   probe x ITEM interactions            median {probe_inter.median():.4f}")
        print(f"   ratio interaction / main             median "
              f"{(probe_inter/probe_main.replace(0,np.nan)).median():.1f}x")
        print(f"\n   within-cell sampling variance (seed) median "
              f"{t.seed_within_cell_var.median():.4f}")
        print(f"   models where interaction > main      "
              f"{(probe_inter > probe_main).sum()}/{len(t)}")


if __name__ == "__main__":
    main()
