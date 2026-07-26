"""
Fragility fraction phi on a BEHAVIOURAL probe, with a factorial probe design.

Data: MoralChoice (Scherrer et al., NeurIPS 2023), released responses of 28 LLMs
to moral scenarios. The design is fully crossed:

    scenario_id       687 (low) / 680 (high) moral scenarios   <- SIGNAL
    question_type     ab | compare | repeat                    <- probe: FORMAT
    question_ordering 0 | 1                                    <- probe: ORDER
    eval_sample_nb    0..4  (top-p sampling, temperature 1.0)  <- probe: SEED
    decision          action1 | action2 | invalid | refusal

Outcome y = 1[decision == action1]. For the low-ambiguity set action1 is the
commonsense-moral action; for the high-ambiguity set the two actions are a
genuine dilemma, so y is a preference, not a correctness score. Either way a
dispositional claim ("model M prefers/does X") is read off the mean of y.

Decomposition. Aggregate over seeds to get y[scenario, condition] where
condition = format x order (6 cells), then a two-way random-effects partition
    sigma2_scenario  (signal)
    sigma2_condition (probe)
    sigma2_resid     (scenario x condition interaction: the probe changing the
                      answer differently per scenario -- also probe-attributable)
Seed variance is estimated separately from the replicate samples.

    phi_strict = sigma2_condition / (sigma2_scenario + sigma2_condition + sigma2_resid)
    phi_broad  = (sigma2_condition + sigma2_interaction) / total
       (sigma2_interaction subtracts the estimated run-noise variance of the
        five-sample cell mean; run noise remains in the denominator)

Also reported: format-only and order-only variance components, seed variance,
and the refusal/invalid rate (a T3 signal in its own right).
"""
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from fragility import fragility_pair, two_way_components

OUT = Path(__file__).resolve().parent
RAW = OUT / "moralchoice"
RAW.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}
BASE = ("https://raw.githubusercontent.com/ninodimontalcino/moralchoice/"
        "9f1dbced7ecf70e334af9a88c3d93be5af0f37b8/"
        "data/responses/paper/{amb}/results_{model}.csv")

MODELS = [
    "openai_gpt-4", "openai_gpt-3.5-turbo", "openai_text-davinci-003",
    "openai_text-davinci-002", "anthropic_claude-v1.3",
    "anthropic_claude-instant-v1.1", "google_text-bison-001",
    "google_flan-t5-xl", "cohere_command-xlarge", "ai21_j2-jumbo-instruct",
    "bigscience_bloomz-7b1", "meta_opt-iml-max-small",
]


def fetch(args):
    amb, model = args
    dest = RAW / f"{amb}__{model}.csv"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest
    url = BASE.format(amb=amb, model=model)
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                dest.write_bytes(r.read())
            return dest
        except Exception:
            if k == 3:
                print("  FAILED", amb, model, flush=True)
                return None
    return None


def one_way(groups):
    """Variance between group means, bias-corrected by within-group variance."""
    means = np.array([g.mean() for g in groups])
    if len(means) < 2:
        return 0.0
    return max(float(means.var(ddof=1)), 0.0)


def analyse(path, amb, model):
    df = pd.read_csv(path, usecols=["scenario_id", "question_type",
                                    "question_ordering", "eval_sample_nb",
                                    "decision"])
    n_all = len(df)
    bad = df.decision.isin(["invalid", "refusal"]).mean()
    df = df[df.decision.isin(["action1", "action2"])].copy()
    if len(df) < 500:
        return None
    df["y"] = (df.decision == "action1").astype(float)
    df["cond"] = df.question_type.astype(str) + "|" + df.question_ordering.astype(str)

    # scenario x condition, averaged over seeds
    piv = df.pivot_table(index="scenario_id", columns="cond", values="y", aggfunc="mean")
    piv = piv.dropna()
    if piv.shape[0] < 30 or piv.shape[1] < 2:
        return None
    M = piv.to_numpy()
    vc = two_way_components(M)
    if vc is None:
        return None
    s2_scen, s2_cond, s2_res = vc
    tot = s2_scen + s2_cond + s2_res
    if tot <= 0:
        return None

    # format-only and order-only components, from condition means
    cond_mean = piv.mean(axis=0)
    fmt = {}
    ordr = {}
    for c, v in cond_mean.items():
        f, o = c.split("|")
        fmt.setdefault(f, []).append(v)
        ordr.setdefault(o, []).append(v)
    s2_fmt = one_way([np.array(v) for v in fmt.values()])
    s2_ord = one_way([np.array(v) for v in ordr.values()])

    # Run-to-run variance in an individual response.  A cell mean averages five
    # samples, so only seed_var / 5 remains in the reported cell mean.
    cell_seed_var = (df.groupby(["scenario_id", "cond"])["y"]
                       .var(ddof=1).dropna())
    seed_var = float(cell_seed_var.mean())
    phi_strict, phi_broad = fragility_pair(
        M, within_cell_variance=seed_var, repeats=5
    )
    interaction_var = max(s2_res - seed_var / 5.0, 0.0)

    return dict(
        ambiguity=amb, model=model,
        n_rows=n_all, refusal_invalid_rate=bad,
        n_scenarios=piv.shape[0], n_conditions=piv.shape[1],
        mean_y=float(M.mean()),
        sigma2_scenario=s2_scen, sigma2_condition=s2_cond, sigma2_resid=s2_res,
        sigma2_format=s2_fmt, sigma2_order=s2_ord, seed_var=float(seed_var),
        sigma2_interaction=interaction_var,
        sigma2_run_mean=seed_var / 5.0,
        phi_strict=phi_strict,
        phi_broad=phi_broad,
        cond_spread=float(cond_mean.max() - cond_mean.min()),
    )


def main():
    jobs = [(a, m) for a in ("low", "high") for m in MODELS]
    print(f"downloading {len(jobs)} response files ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        paths = list(ex.map(fetch, jobs))

    rows = []
    for (amb, model), path in zip(jobs, paths):
        if path is None:
            continue
        try:
            r = analyse(path, amb, model)
        except Exception as e:
            print("  err", amb, model, type(e).__name__, flush=True)
            continue
        if r:
            rows.append(r)
            print(f"  {amb:4s} {model:32s} phi_strict={r['phi_strict']:.3f} "
                  f"phi_broad={r['phi_broad']:.3f} spread={r['cond_spread']:.3f}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phi_moralchoice.csv", index=False)
    print(f"\nmodels analysed: {out.model.nunique()}  rows: {len(out)}")

    for amb in ("low", "high"):
        d = out[out.ambiguity == amb]
        if not len(d):
            continue
        print(f"\n=== {amb.upper()} AMBIGUITY ===")
        print(f"  phi_strict  median {d.phi_strict.median():.4f}  "
              f"IQR [{d.phi_strict.quantile(.25):.4f}, {d.phi_strict.quantile(.75):.4f}]  "
              f"max {d.phi_strict.max():.4f}")
        print(f"  phi_broad   median {d.phi_broad.median():.4f}  "
              f"max {d.phi_broad.max():.4f}")
        print(f"  cells with phi_broad >= 0.5 : {(d.phi_broad >= .5).sum()}/{len(d)}")
        print(f"  format var median {d.sigma2_format.median():.5f} | "
              f"order var median {d.sigma2_order.median():.5f} | "
              f"seed var median {d.seed_var.median():.5f}")
        print(f"  probe-condition spread in mean y: median "
              f"{d.cond_spread.median():.4f}  max {d.cond_spread.max():.4f}")
        print(f"  refusal/invalid rate median {d.refusal_invalid_rate.median():.4f}")

    print("\n=== per model (high ambiguity = genuine dilemmas) ===")
    h = out[out.ambiguity == "high"].sort_values("phi_broad", ascending=False)
    if len(h):
        print(h[["model", "phi_strict", "phi_broad", "sigma2_format",
                 "sigma2_order", "seed_var", "cond_spread",
                 "refusal_invalid_rate"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
