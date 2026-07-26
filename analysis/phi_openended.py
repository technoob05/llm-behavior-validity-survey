"""
The fragility fraction on OPEN-ENDED generation.

Our earlier estimates used forced-choice probes. This one does not. MT-Bench
(Zheng et al., 2023) asks models to produce free multi-turn text; a GPT-4 judge
then compares two answers. The released judgments record the SAME comparison run
in BOTH presentation orders (g1 = model_1 first, g2 = model_2 first), so
presentation order is a fully crossed probe factor over an open-ended behaviour.

Item      = (question_id, model_1, model_2, turn)   -> the thing being measured
Probe     = presentation order (2 levels)
Outcome   = does the judge prefer model_1?   y in {0, 0.5, 1}   (0.5 = tie)

We report
  (a) the signed position-bias statistic beta-hat with a bootstrap interval,
  (b) the raw disagreement rate between the two orders, and
  (c) strict phi, the main-effect share assigned to presentation order.

There is one released judge decision per item-by-order cell. Consequently the
item-by-order interaction and judge/sampling noise are confounded. This script
intentionally does not report broad phi: a repeated-judgment design is required
to estimate that quantity.
"""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SRC = OUT / "mtbench" / "gpt4_pair.jsonl"
rng = np.random.default_rng(0)


def score(w):
    """Preference for model_1 as seen in one presentation order."""
    if w == "model_1":
        return 1.0
    if w == "model_2":
        return 0.0
    if w == "tie":
        return 0.5
    return np.nan


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
    return (max((ms_a - ms_e) / B, 0.0),
            max((ms_b - ms_e) / A, 0.0),
            max(ms_e, 0.0))


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf8")]
    df = pd.DataFrame(rows)
    df["y_order1"] = df.g1_winner.map(score)
    df["y_order2"] = df.g2_winner.map(score)
    df = df.dropna(subset=["y_order1", "y_order2"]).copy()
    print(f"open-ended comparisons with both orders: {len(df)}")
    print(f"distinct questions: {df.question_id.nunique()}   "
          f"distinct model pairs: {df.groupby(['model_1','model_2']).ngroups}")

    # ---------- (a) signed position bias ----------
    # In order 1 the judge sees model_1 first; in order 2 it sees model_2 first.
    # A judge with no position prior scores the same either way.
    # Keep the two orders for an item together.  Ties contribute one half.
    item_first_pref = (
        df.y_order1.to_numpy() + 1.0 - df.y_order2.to_numpy()
    ) / 2.0
    beta = item_first_pref.mean() - 0.5
    b = [
        item_first_pref[
            rng.integers(0, len(item_first_pref), len(item_first_pref))
        ].mean() - 0.5
        for _ in range(2000)
    ]
    print(f"\n(a) signed position bias beta-hat = {beta:+.4f}  "
          f"95% CI [{np.percentile(b,2.5):+.4f}, {np.percentile(b,97.5):+.4f}]")
    print(f"    the judge prefers whichever answer is shown FIRST "
          f"{50 + 100*beta:.1f}% of the time")

    # ---------- (b) raw disagreement between orders ----------
    disagree = (df.y_order1 != df.y_order2).mean()
    strict = ((df.y_order1 - df.y_order2).abs() == 1.0).mean()
    print(f"\n(b) the two orders disagree on {100*disagree:.1f}% of comparisons; "
          f"{100*strict:.1f}% are outright reversals (not tie-related)")

    # ---------- (c) phi ----------
    long = pd.concat([
        df.assign(order="first", y=df.y_order1),
        df.assign(order="second", y=df.y_order2),
    ])
    long["item"] = (long.question_id.astype(str) + "|" + long.model_1 + "|"
                    + long.model_2 + "|" + long.turn.astype(str))
    piv = long.pivot_table(index="item", columns="order", values="y").dropna()
    M = piv.to_numpy()
    s2_item, s2_probe, s2_res = two_way(M)
    tot = s2_item + s2_probe + s2_res
    phi_strict = s2_probe / tot
    # The residual contains both item-by-order interaction and judge noise.
    # With one judgment per cell, broad phi is not identifiable.
    print(f"\n(c) variance decomposition over {M.shape[0]} items x 2 orders")
    print(f"    sigma^2_item  = {s2_item:.5f}")
    print(f"    sigma^2_probe = {s2_probe:.5f}   (presentation order)")
    print(f"    sigma^2_resid = {s2_res:.5f}   (interaction + judge noise, confounded)")
    print(f"    phi_strict = {phi_strict:.4f}")
    print("    phi_broad  = not identifiable without repeated judge decisions")

    # bootstrap phi over items
    n = M.shape[0]
    ps = []
    for _ in range(600):
        idx = rng.integers(0, n, n)
        vc = two_way(M[idx])
        if not vc:
            continue
        a, bb, e = vc
        t = a + bb + e
        if t > 0:
            ps.append(bb / t)
    print(f"    phi_strict 95% CI [{np.percentile(ps,2.5):.4f}, {np.percentile(ps,97.5):.4f}]")

    # ---------- per-model win rate stability ----------
    def winrate(col, flip):
        d = df.copy()
        d["w"] = (1.0 - d[col]) if flip else d[col]
        a = d.groupby("model_1").w.mean()
        b = 1.0 - d.groupby("model_2").w.mean()
        return pd.concat([a, b], axis=1).mean(axis=1)
    w1 = winrate("y_order1", False)
    w2 = winrate("y_order2", False)
    common = w1.index.intersection(w2.index)
    w1, w2 = w1[common], w2[common]
    flips = sum(1 for a, b in combinations(common, 2)
                if np.sign(w1[a] - w1[b]) != np.sign(w2[a] - w2[b]))
    tot_pairs = len(list(combinations(common, 2)))
    print(f"\n(d) model win rates recomputed under each order: "
          f"{flips}/{tot_pairs} model pairs "
          f"({100*flips/tot_pairs:.1f}%) reverse their ordering")
    print(f"    max |win-rate shift| between orders: "
          f"{(w1 - w2).abs().max():.3f}")

    pd.DataFrame(dict(order_first=w1, order_second=w2)).to_csv(
        OUT / "mtbench_winrates.csv")


if __name__ == "__main__":
    main()
