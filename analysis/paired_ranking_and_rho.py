"""Paired ranking uncertainty and cross-model probe correlation.

(1) The simple inversion-risk derivation takes
Var(difference) = 2*phi*sigma2/K, which
    assumes the probe disturbance hitting model A is independent of the one
    hitting model B. The paper itself shows this is false: there IS a probe MAIN
    effect, a format that hurts every model at once. Positive correlation makes
    the closed form CONSERVATIVE, by a factor that depends on rho, which nobody
    has measured. We measure it.

(2) A comparative claim is that GPT-4 is less
    probe-dependent than the other eleven models. The appendix defends it by
    noting that GPT-4's marginal interval does not overlap the median model's.
    Comparing two marginal intervals is the wrong test, and it is wrong in BOTH
    directions: it ignores that the two models were measured under the SAME six
    probe conditions (which makes the paired difference more precise than two
    marginals suggest), while the marginals themselves were computed without
    probe-level resampling at all (which makes them far too narrow).

    The right test is a paired hierarchical bootstrap: resample scenarios and
    probe conditions ONCE, apply the same resample to every model, and look at
    the distribution of the DIFFERENCE in phi. That is what a ranking claim
    means. This script runs it.

Outputs: paired_ranking.csv, probe_rho.csv, and a rho sensitivity table for the
inversion threshold.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from fragility import fragility_pair

HERE = Path(__file__).resolve().parent
MC = HERE / "moralchoice"
RNG = np.random.default_rng(4242)
B = 3000

MODELS = [
    "openai_gpt-4", "openai_gpt-3.5-turbo", "openai_text-davinci-003",
    "openai_text-davinci-002", "anthropic_claude-v1.3",
    "anthropic_claude-instant-v1.1", "google_text-bison-001",
    "google_flan-t5-xl", "cohere_command-xlarge", "ai21_j2-jumbo-instruct",
    "bigscience_bloomz-7b1", "meta_opt-iml-max-small",
]
REF = "openai_gpt-4"


def load_panel(amb):
    """All models on a COMMON scenario set and common conditions.
    Returns (scenarios, conditions, dict model -> matrix)."""
    frames = {}
    within_frames = {}
    for m in MODELS:
        f = MC / f"{amb}__{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                     "question_ordering", "eval_sample_nb",
                                     "decision"])
        df = df[df.decision.isin(["action1", "action2"])].copy()
        df["y"] = (df.decision == "action1").astype(float)
        df["cond"] = (df.question_type.astype(str) + "|"
                      + df.question_ordering.astype(str))
        piv = df.pivot_table(index="scenario_id", columns="cond", values="y",
                             aggfunc="mean").dropna()
        within = df.pivot_table(index="scenario_id", columns="cond", values="y",
                                aggfunc="var").loc[piv.index, piv.columns]
        if piv.shape[0] >= 30:
            frames[m] = piv
            within_frames[m] = within
    if not frames:
        return None
    scen = None
    conds = None
    for piv in frames.values():
        scen = piv.index if scen is None else scen.intersection(piv.index)
        conds = piv.columns if conds is None else conds.intersection(piv.columns)
    mats = {m: piv.loc[scen, conds].to_numpy(float) for m, piv in frames.items()}
    variances = {
        m: within_frames[m].loc[scen, conds].to_numpy(float) for m in frames
    }
    return list(scen), list(conds), mats, variances


def main():
    all_rho = []
    for amb in ("high", "low"):
        got = load_panel(amb)
        if got is None:
            continue
        scen, conds, mats, variances = got
        I, K = len(scen), len(conds)
        models = [m for m in MODELS if m in mats]
        print("=" * 88)
        print(f"{amb.upper()} AMBIGUITY: {len(models)} models on a common panel of "
              f"{I} scenarios x {K} conditions")
        print("=" * 88)

        # -------------------------------------------------- measured rho
        # per-condition probe deviation for each model: the K condition means,
        # centred. rho between two models says how much of the probe
        # disturbance is shared.
        dev = {m: (mats[m].mean(axis=0) - mats[m].mean()) for m in models}
        rhos = []
        for a, b in combinations(models, 2):
            va, vb = dev[a], dev[b]
            if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
                continue
            r = float(np.corrcoef(va, vb)[0, 1])
            rhos.append(r)
            all_rho.append(dict(ambiguity=amb, model_a=a, model_b=b, rho=r))
        rhos = np.array(rhos)
        print(f"\n  correlation of the probe disturbance between two models "
              f"(n={len(rhos)} pairs)")
        print(f"    median rho = {np.median(rhos):+.3f}   "
              f"IQR [{np.percentile(rhos,25):+.3f}, {np.percentile(rhos,75):+.3f}]   "
              f"share positive = {100*(rhos>0).mean():.0f}%")
        print("    (the closed form assumes rho = 0; positive rho makes it "
              "conservative)")

        # -------------------------------------------------- paired bootstrap
        pt = {
            m: fragility_pair(mats[m], variances[m], repeats=5) for m in models
        }
        draws_s = {m: [] for m in models}
        draws_b = {m: [] for m in models}
        ref_min_s = ref_min_b = 0
        n_ok = 0
        for _ in range(B):
            ri = RNG.integers(0, I, I)
            rk = RNG.integers(0, K, K)
            cur_s, cur_b = {}, {}
            ok = True
            for m in models:
                s, b = fragility_pair(
                    mats[m][np.ix_(ri, rk)],
                    variances[m][np.ix_(ri, rk)],
                    repeats=5,
                )
                if not np.isfinite(s):
                    ok = False
                    break
                cur_s[m], cur_b[m] = s, b
            if not ok:
                continue
            n_ok += 1
            for m in models:
                draws_s[m].append(cur_s[m])
                draws_b[m].append(cur_b[m])
            if REF in cur_s and min(cur_s, key=cur_s.get) == REF:
                ref_min_s += 1
            if REF in cur_b and min(cur_b, key=cur_b.get) == REF:
                ref_min_b += 1

        rows = []
        for m in models:
            ds = np.array(draws_s[m])
            db = np.array(draws_b[m])
            d_s = ds - np.array(draws_s[REF])
            d_b = db - np.array(draws_b[REF])
            rows.append(dict(
                ambiguity=amb, model=m, n_scen=I, K=K,
                phi_strict=pt[m][0], phi_broad=pt[m][1],
                ci_s_lo=np.percentile(ds, 2.5), ci_s_hi=np.percentile(ds, 97.5),
                ci_b_lo=np.percentile(db, 2.5), ci_b_hi=np.percentile(db, 97.5),
                d_strict=float(d_s.mean()),
                d_strict_lo=np.percentile(d_s, 2.5),
                d_strict_hi=np.percentile(d_s, 97.5),
                p_more_fragile_strict=float((d_s > 0).mean()),
                d_broad=float(d_b.mean()),
                d_broad_lo=np.percentile(d_b, 2.5),
                d_broad_hi=np.percentile(d_b, 97.5),
                p_more_fragile_broad=float((d_b > 0).mean()),
            ))
        t = pd.DataFrame(rows)
        t.to_csv(HERE / f"paired_ranking_{amb}.csv", index=False)

        print(f"\n  paired hierarchical bootstrap, {n_ok} usable draws")
        print(f"  reference model = {REF}\n")
        print(f"{'model':32s} {'phi_b':>7s} {'d_broad vs ref':>22s} "
              f"{'P(more fragile)':>16s}")
        for _, r in t.sort_values("phi_broad").iterrows():
            star = "  <- ref" if r.model == REF else ""
            print(f"{r.model:32s} {r.phi_broad:7.3f} "
                  f"[{r.d_broad_lo:+.3f},{r.d_broad_hi:+.3f}]".rjust(23)
                  + f" {r.p_more_fragile_broad:16.3f}{star}")

        nb = int((t[t.model != REF].p_more_fragile_broad > 0.975).sum())
        ns = int((t[t.model != REF].p_more_fragile_strict > 0.975).sum())
        print(f"\n  models MORE fragile than {REF} at the 97.5% level: "
              f"{nb} of {len(t)-1} on phi_broad, {ns} of {len(t)-1} on phi_strict")
        print(f"  P({REF} is the least fragile of all {len(models)}) = "
              f"{ref_min_b/max(n_ok,1):.3f} (broad), "
              f"{ref_min_s/max(n_ok,1):.3f} (strict)")

        # marginal-overlap test, for contrast with the paired one
        med_model = t.iloc[(t.phi_broad - t.phi_broad.median()).abs().argsort()].iloc[0]
        ref = t[t.model == REF].iloc[0]
        overlap = not (ref.ci_b_hi < med_model.ci_b_lo or
                       med_model.ci_b_hi < ref.ci_b_lo)
        print(f"\n  marginal intervals, {REF} [{ref.ci_b_lo:.3f},{ref.ci_b_hi:.3f}] "
              f"vs median model {med_model.model} "
              f"[{med_model.ci_b_lo:.3f},{med_model.ci_b_hi:.3f}]: "
              f"{'OVERLAP' if overlap else 'disjoint'}")
        print("  (the paired test above is the correct one; the marginal test is "
              "shown\n   only because the appendix currently relies on it)\n")

    pd.DataFrame(all_rho).to_csv(HERE / "probe_rho.csv", index=False)

    # ------------------------------------------- rho sensitivity of the threshold
    print("=" * 88)
    print("SENSITIVITY OF THE INVERSION THRESHOLD TO rho")
    print("=" * 88)
    r = pd.DataFrame(all_rho)
    print("\n  Var(difference of two model scores) = phi*sigma2_total*(2-2rho)/K,")
    print("  so the largest tolerable phi scales as 1/(2-2rho): rho>0 RELAXES the")
    print("  threshold, rho<0 tightens it. phi*(rho) / phi*(0) = 1/(1-rho).\n")
    print(f"  {'rho':>6s} {'factor on tolerable phi':>26s}")
    for rho in (-0.25, 0.0, 0.25, 0.5, 0.75):
        print(f"  {rho:+6.2f} {1.0/(1.0-rho):26.2f}x")
    if len(r):
        for amb, g in r.groupby("ambiguity"):
            med = g.rho.median()
            print(f"\n  measured on the {amb}-ambiguity panel: median rho = {med:+.3f}"
                  f"  ->  tolerable phi is {1.0/(1.0-med):.2f}x the rho=0 value")
        print("\n  So the published thresholds are conservative where rho>0. The")
        print("  correction is not negligible and it is not uniform across tasks,")
        print("  which is why the table needs the rho column rather than a footnote.")


if __name__ == "__main__":
    main()
