"""
Worked structural-replication test (condition (iii) of the model-native
construct criterion, Appendix H), run on MoralChoice.

The question a structural test answers is NOT "does the model score high on X"
but "does the pattern of dependencies among items look the same when the probe
changes". A genuine construct predicts a stable structure; a shortcut or a
presentation prior does not.

Design. For each model we build the scenario x probe-condition matrix of
P(action1). We then ask three things.

  (1) Split-half structural replication ACROSS PROBES.
      Split the six probe conditions into two halves, average within each half
      to get two independent scenario profiles, and correlate them. A construct
      that survives probe variation gives a high correlation; a probe-driven
      measurement gives a low one. Spearman-Brown corrected.

  (2) Structural agreement ACROSS MODELS.
      Correlate scenario profiles between models within a fixed probe
      condition. If models share a moral structure, this is high.

  (3) Which dominates: does changing the PROBE move the scenario profile more
      than changing the MODEL does? This is the structural analogue of phi and
      it is the decisive quantity: if swapping the question form perturbs the
      profile more than swapping to a different model, the "moral profile" is a
      property of the instrument.

  (4) Rank-one descriptive adequacy. Fit a rank-1 approximation to
      the scenario x condition matrix by truncated SVD and report the share of
      variance it explains, per ambiguity level. A unidimensional moral
      disposition predicts a dominant first factor.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"


def load(amb):
    """dict model -> DataFrame(scenario x condition) of P(action1)."""
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
        piv = df.pivot_table(index="scenario_id", columns="cond", values="y",
                             aggfunc="mean").dropna()
        if piv.shape[0] > 50 and piv.shape[1] >= 4:
            out[model] = piv
    return out


def sb(r, n=2):
    """Spearman-Brown correction for a split-half correlation."""
    if r <= -1 or np.isnan(r):
        return np.nan
    return n * r / (1 + (n - 1) * r)


def analyse(amb):
    data = load(amb)
    if not data:
        return None
    conds = sorted(next(iter(data.values())).columns)
    print(f"\n{'='*66}\n{amb.upper()} AMBIGUITY  |  {len(data)} models, "
          f"{len(conds)} probe conditions\n{'='*66}")

    # ---------- (1) split-half across probes ----------
    rows = []
    rng = np.random.default_rng(0)
    for model, piv in data.items():
        rs = []
        for _ in range(50):
            perm = list(rng.permutation(conds))
            h1, h2 = perm[:len(perm)//2], perm[len(perm)//2:]
            a = piv[h1].mean(axis=1).to_numpy()
            b = piv[h2].mean(axis=1).to_numpy()
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            rs.append(spearmanr(a, b).statistic)
        if rs:
            r = float(np.mean(rs))
            rows.append(dict(model=model, split_half_r=r, sb_corrected=sb(r)))
    sh = pd.DataFrame(rows).sort_values("sb_corrected", ascending=False)
    print("\n(1) split-half structural replication ACROSS PROBE CONDITIONS")
    print("    (Spearman rho between two disjoint halves of the probe pool)")
    print(sh.round(3).to_string(index=False))
    print(f"    median Spearman-Brown corrected: {sh.sb_corrected.median():.3f}")

    # ---------- (2) cross-model agreement within one probe ----------
    cross = []
    for c in conds:
        prof = {m: piv[c].to_numpy() for m, piv in data.items()}
        common = None
        for m, piv in data.items():
            idx = set(piv.index)
            common = idx if common is None else (common & idx)
        common = sorted(common)
        prof = {m: data[m].loc[common, c].to_numpy() for m in data}
        for a, b in combinations(prof, 2):
            if prof[a].std() < 1e-9 or prof[b].std() < 1e-9:
                continue
            cross.append(spearmanr(prof[a], prof[b]).statistic)
    print(f"\n(2) cross-MODEL structural agreement within a fixed probe: "
          f"median rho = {np.median(cross):.3f}  (n={len(cross)} pairs)")

    # ---------- (3) probe vs model as sources of structural change ----------
    common = sorted(set.intersection(*[set(p.index) for p in data.values()]))
    within_model_across_probe = []
    for m, piv in data.items():
        sub = piv.loc[common]
        for c1, c2 in combinations(conds, 2):
            if sub[c1].std() < 1e-9 or sub[c2].std() < 1e-9:
                continue
            within_model_across_probe.append(
                spearmanr(sub[c1].to_numpy(), sub[c2].to_numpy()).statistic)
    across_model_same_probe = []
    for c in conds:
        for a, b in combinations(data, 2):
            x = data[a].loc[common, c].to_numpy()
            y = data[b].loc[common, c].to_numpy()
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            across_model_same_probe.append(spearmanr(x, y).statistic)
    wm = float(np.median(within_model_across_probe))
    am = float(np.median(across_model_same_probe))
    print(f"\n(3) same model, different probe : median rho = {wm:.3f}")
    print(f"    different model, same probe  : median rho = {am:.3f}")
    if wm >= am:
        print("    -> the profile is more stable under a probe change than under a "
              "model change on this descriptive rank-correlation comparison")
    else:
        print("    -> the profile is LESS stable under a probe change than under a "
              "model change on this descriptive rank-correlation comparison")

    # ---------- (4) one-factor adequacy ----------
    fac = []
    for m, piv in data.items():
        M = piv.to_numpy()
        M = M - M.mean(axis=0, keepdims=True)
        if not np.isfinite(M).all():
            continue
        s = np.linalg.svd(M, compute_uv=False)
        var = s ** 2
        fac.append(dict(model=m, first_factor_share=float(var[0] / var.sum())))
    ff = pd.DataFrame(fac)
    print(f"\n(4) share of variance on the FIRST singular component "
          f"(scenario x condition, column-centred)")
    print(f"    median {ff.first_factor_share.median():.3f}  "
          f"min {ff.first_factor_share.min():.3f}  "
          f"max {ff.first_factor_share.max():.3f}")

    sh.to_csv(OUT / f"structural_{amb}.csv", index=False)
    summary = pd.DataFrame([dict(
        ambiguity=amb,
        n_models=len(data),
        n_conditions=len(conds),
        split_half_sb_median=float(sh.sb_corrected.median()),
        cross_model_same_probe_rho_median=float(np.median(cross)),
        same_model_different_probe_rho_median=wm,
        different_model_same_probe_rho_median=am,
        first_singular_component_share_median=float(
            ff.first_factor_share.median()
        ),
    )])
    summary.to_csv(OUT / f"structural_summary_{amb}.csv", index=False)
    return dict(amb=amb, sh=sh, cross=np.median(cross), wm=wm, am=am,
                ff=ff.first_factor_share.median())


if __name__ == "__main__":
    res = [analyse(a) for a in ("low", "high")]
    print("\n\n================ SUMMARY ================")
    for r in res:
        if r:
            print(f"{r['amb']:5s} | split-half SB median {r['sh'].sb_corrected.median():.3f} "
                  f"| cross-model rho {r['cross']:.3f} "
                  f"| same-model/diff-probe {r['wm']:.3f} "
                  f"| diff-model/same-probe {r['am']:.3f} "
                  f"| 1st-factor share {r['ff']:.3f}")
