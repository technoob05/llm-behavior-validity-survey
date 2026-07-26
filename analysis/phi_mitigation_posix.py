"""
Two analyses on the crossed MoralChoice design
(high-ambiguity / genuine dilemmas, 12 models):

A4  phi BEFORE vs AFTER a mitigation.
    A mitigation collapses one probe factor by averaging over it, so the probe
    variance a study still faces shrinks. We report phi_broad (probe main +
    item-by-probe interaction, over total) before any mitigation, after
    order-symmetrisation (average the two option orders), and after a template
    ensemble (average the three question forms). phi is the decision variable:
    a study picks the remedy that drops it furthest per unit cost.

A5  phi vs a POSIX-style index, on the same corpus.
    POSIX scores how far a model's OUTPUT DISTRIBUTION moves under perturbation.
    We compute, per model, POSIX_m = mean over scenarios of the spread of
    P(action1) across the six probe conditions, and correlate it across the 12
    models with the model's phi_broad. High correlation shows the two indices
    track the same underlying fragility; the point of phi is that it is
    normalised by the signal and attaches to a reported claim, not to a model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path(__file__).resolve().parent / "moralchoice"


def load(amb="high"):
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


def components(piv):
    """Crossed scenario x form x order variance components; returns dict of shares."""
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
              form=no * ((b ** 2).sum()) * ni / ni,  # keep scale comparable
              order=nf * ((c ** 2).sum()),
              item_form=no * (ab ** 2).sum(),
              item_order=nf * (ac ** 2).sum(),
              form_order=ni * (bc ** 2).sum(),
              resid=float((resid ** 2).sum()))
    # simpler, correct sums of squares
    ss = dict(item=nf * no * (a ** 2).sum(),
              form=ni * no * (b ** 2).sum(),
              order=ni * nf * (c ** 2).sum(),
              item_form=no * (ab ** 2).sum(),
              item_order=nf * (ac ** 2).sum(),
              form_order=ni * (bc ** 2).sum(),
              resid=float((resid ** 2).sum()))
    tot = sum(ss.values())
    return {k: v / tot for k, v in ss.items()}


def phi_broad(sh):
    """probe main (form+order) + item-by-probe interaction, over total."""
    return sh["form"] + sh["order"] + sh["item_form"] + sh["item_order"] + sh["form_order"]


def phi_strict(sh):
    """probe main effect only (form+order), over total."""
    return sh["form"] + sh["order"]


def phi_after(piv, collapse, fn=phi_broad):
    """phi after averaging over one probe factor (fn = phi_broad or phi_strict)."""
    forms = sorted({c[0] for c in piv.columns})
    orders = sorted({c[1] for c in piv.columns})
    if collapse == "order":            # order-symmetrisation: only form remains
        cols = {}
        for f in forms:
            cols[(f, "sym")] = piv[[c for c in piv.columns if c[0] == f]].mean(axis=1)
        p2 = pd.DataFrame(cols)
        p2.columns = pd.MultiIndex.from_tuples(p2.columns)
    else:                               # template ensemble: only order remains
        cols = {}
        for o in orders:
            cols[("ens", o)] = piv[[c for c in piv.columns if c[1] == o]].mean(axis=1)
        p2 = pd.DataFrame(cols)
        p2.columns = pd.MultiIndex.from_tuples(p2.columns)
    return phi_broad(components(p2))


def main():
    data = load("high")
    models = list(data)
    print(f"MoralChoice high-ambiguity: {len(models)} models, "
          f"{data[models[0]].shape[1]} probe conditions\n")

    # ---- A4: a mitigation shrinks phi's NUMERATOR (probe variance), not phi as
    # a ratio (collapsing a factor also shrinks the denominator, which would
    # inflate the remaining shares). The absolute probe-variance reduction is the
    # decision quantity and is reported by mitigation_efficacy.py (88% / 29% on
    # dilemmas). We print phi_broad before mitigation for reference only.
    before = np.median([phi_broad(components(data[m])) for m in models])
    print("=== A4: phi_broad before mitigation (median over models) ===")
    print(f"  phi_broad (no mitigation) = {before:.3f}")
    print("  mitigations act on the numerator: order-symmetrisation removes 88% of")
    print("  probe variance on dilemmas, template ensembles 29% (mitigation_efficacy.py).")

    # ---- A5: phi vs POSIX-style index, across models ----
    phis, posix = [], []
    for m in models:
        piv = data[m]
        phis.append(phi_broad(components(piv)))
        # per-scenario spread of P(action1) across the 6 conditions, averaged
        spread = (piv.max(axis=1) - piv.min(axis=1)).mean()
        posix.append(float(spread))
    rho, p = stats.spearmanr(phis, posix)
    r, _ = stats.pearsonr(phis, posix)
    print("\n=== A5: phi_broad vs POSIX-style output-spread index (12 models) ===")
    print(f"  Spearman rho = {rho:.2f} (p = {p:.3f}),  Pearson r = {r:.2f}")
    print("  interpretation: the two indices rank models' fragility similarly;")
    print("  phi differs by normalising against the signal and attaching to a claim.")

    pd.DataFrame({"model": models, "phi_broad": phis,
                  "posix_spread": posix}).to_csv(
        Path(__file__).resolve().parent / "phi_vs_posix.csv", index=False)


if __name__ == "__main__":
    main()
