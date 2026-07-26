"""Like-for-like capability-versus-behavioural comparison.

An unharmonised contrast between PromptEval and MoralChoice differs in several
ways at once:

  (1) aggregation level : PromptEval phi was reported over subject-level
                          accuracies, MoralChoice over scenario-level outcomes;
  (2) estimand          : the capability value is strict phi while the original
                          MoralChoice summary used broad phi;
  (3) estimator         : the PromptEval path used a raw sum-of-squares ratio,
                          the MoralChoice path used method-of-moments variance
                          components (which subtract the residual mean square);
  (4) probe count       : K = 100 templates vs K = 6 conditions;
  and difficulty filtering was applied to one corpus and not the other.

This script fixes all five and reports the contrast that survives. Every corpus
is reduced to the SAME object: an (items x conditions) matrix with ONE
observation per cell, item = the unit a dispositional claim is read off (a
question, a scenario), and phi is computed with ONE estimator.

Comparability of the two estimands, stated plainly:

  phi_strict = sigma2_probe / total                is comparable across corpora.
  phi_broad  = (sigma2_probe + sigma2_resid)/total is NOT, when one corpus has
               replicates and the other does not. With a single sample per cell
               a binary outcome carries Bernoulli noise p(1-p) that cannot be
               separated from the item-by-probe interaction, so phi_broad is
               inflated by noise on PromptEval and only partly on MoralChoice.
               We therefore report phi_strict as the comparable quantity and
               give phi_broad only where the replicates make it identifiable.

Outputs: harmonised_crosscorpus.csv (per corpus-model cell) and a printed table.
The companion harmonised_template_draws.csv records the capability median and
capability-to-dilemma ratio for every template draw. Point estimates must be
read with that distribution, not as a fixed cross-corpus multiplier.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from fragility import fragility_pair

HERE = Path(__file__).resolve().parent
MC = HERE / "moralchoice"
PE = HERE / "prompteval"
RNG = np.random.default_rng(20260725)

MODELS = [
    "openai_gpt-4", "openai_gpt-3.5-turbo", "openai_text-davinci-003",
    "openai_text-davinci-002", "anthropic_claude-v1.3",
    "anthropic_claude-instant-v1.1", "google_text-bison-001",
    "google_flan-t5-xl", "cohere_command-xlarge", "ai21_j2-jumbo-instruct",
    "bigscience_bloomz-7b1", "meta_opt-iml-max-small",
]

# ---------------------------------------------------------------- estimators


def components(M):
    """Method-of-moments variance components for a crossed items x conditions
    matrix with one observation per cell.

    Returns (s2_item, s2_probe, s2_resid, n_clamped). The residual here is the
    item-by-probe interaction confounded with any within-cell noise; with one
    observation per cell the two are not separable, which is why phi_broad is
    reported only where replicates exist.
    """
    M = np.asarray(M, float)
    I, K = M.shape
    g = M.mean()
    ai = M.mean(axis=1) - g
    bk = M.mean(axis=0) - g
    ms_i = K * (ai ** 2).sum() / (I - 1)
    ms_k = I * (bk ** 2).sum() / (K - 1)
    resid = M - (g + ai[:, None] + bk[None, :])
    ms_e = (resid ** 2).sum() / ((I - 1) * (K - 1))
    raw_i, raw_k = (ms_i - ms_e) / K, (ms_k - ms_e) / I
    n_clamped = int(raw_i < 0) + int(raw_k < 0)
    return max(raw_i, 0.0), max(raw_k, 0.0), max(ms_e, 0.0), n_clamped


def phi_mom(M):
    s2i, s2k, s2e, _ = components(M)
    tot = s2i + s2k + s2e
    return (s2k / tot) if tot > 0 else np.nan


def phi_ssratio(M):
    """The uncorrected estimator the PromptEval path used: a raw sum-of-squares
    share, with no residual mean square subtracted. Kept to show that the
    estimator choice alone moves the number."""
    M = np.asarray(M, float)
    I, K = M.shape
    g = M.mean()
    ai = M.mean(axis=1) - g
    bk = M.mean(axis=0) - g
    ss_i = K * (ai ** 2).sum()
    ss_k = I * (bk ** 2).sum()
    resid = M - (g + ai[:, None] + bk[None, :])
    tot = ss_i + ss_k + float((resid ** 2).sum())
    return (ss_k / tot) if tot > 0 else np.nan


def mid_mask(M, lo=0.2, hi=0.8):
    p = np.asarray(M, float).mean(axis=1)
    return (p > lo) & (p < hi)


def subsample_K(M, K, reps, fn):
    """Median of fn over `reps` random draws of K columns."""
    M = np.asarray(M, float)
    if M.shape[1] <= K:
        return fn(M), 0.0
    vals = []
    for _ in range(reps):
        cols = RNG.choice(M.shape[1], size=K, replace=False)
        v = fn(M[:, cols])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.median(vals)), float(np.std(vals))


# ---------------------------------------------------------------- corpora


def load_moralchoice(amb, model):
    """Return (M_single, M_seedavg, seed_var) at SCENARIO level, conditions =
    format x order (K=6). M_single uses one sample per cell to match a corpus
    without replicates; M_seedavg averages the five samples."""
    f = MC / f"{amb}__{model}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, usecols=["scenario_id", "question_type",
                                 "question_ordering", "eval_sample_nb",
                                 "decision"])
    df = df[df.decision.isin(["action1", "action2"])].copy()
    if len(df) < 500:
        return None
    df["y"] = (df.decision == "action1").astype(float)
    df["cond"] = df.question_type.astype(str) + "|" + df.question_ordering.astype(str)

    avg = df.pivot_table(index="scenario_id", columns="cond", values="y",
                         aggfunc="mean").dropna()
    one = (df[df.eval_sample_nb == df.eval_sample_nb.min()]
           .pivot_table(index="scenario_id", columns="cond", values="y",
                        aggfunc="mean").dropna())
    common = avg.index.intersection(one.index)
    avg, one = avg.loc[common], one.loc[common]
    if avg.shape[0] < 30 or avg.shape[1] < 2:
        return None
    seed_var = float(df.groupby(["scenario_id", "cond"])["y"]
                       .var(ddof=1).dropna().mean())
    return one.to_numpy(float), avg.to_numpy(float), seed_var


def load_prompteval(path):
    """Return items x templates 0/1 matrix at QUESTION level (K=100)."""
    df = pd.read_parquet(path)
    A = df.select_dtypes(include=[np.number]).to_numpy(float).T  # items x templates
    A = A[np.isfinite(A).all(axis=1)]
    return A if (A.shape[0] >= 30 and A.shape[1] >= 20) else None


# ---------------------------------------------------------------- run

def row_for(matrix, corpus, claim, model, unit, extra=None):
    """Compute the harmonised battery for one items x conditions matrix."""
    M = np.asarray(matrix, float)
    keep = mid_mask(M)
    out = dict(corpus=corpus, claim=claim, model=model, unit=unit,
               n_items=M.shape[0], K_full=M.shape[1],
               frac_ceiling=float((M.mean(axis=1) >= 0.9).mean()),
               frac_floor=float((M.mean(axis=1) <= 0.1).mean()),
               n_items_mid=int(keep.sum()),
               mean_y=float(M.mean()),
               phi_fullK_mom=phi_mom(M),
               phi_fullK_ssratio=phi_ssratio(M))
    # matched K = 6, with and without the difficulty filter
    out["phi_K6_mom"], out["phi_K6_sd"] = subsample_K(M, 6, 200, phi_mom)
    if keep.sum() >= 20:
        out["phi_K6_mid_mom"], _ = subsample_K(M[keep], 6, 200, phi_mom)
        out["phi_fullK_mid_mom"] = phi_mom(M[keep])
    else:
        out["phi_K6_mid_mom"] = np.nan
        out["phi_fullK_mid_mom"] = np.nan
    s2i, s2k, s2e, ncl = components(M)
    out.update(sigma2_item=s2i, sigma2_probe=s2k, sigma2_resid=s2e,
               n_clamped=ncl)
    if extra:
        out.update(extra)
    return out


def main():
    rows = []
    prompteval_mid = []

    print("=== MoralChoice (behavioural), scenario level, K=6 ===", flush=True)
    for amb, claim in (("high", "behavioural-dilemma"),
                       ("low", "behavioural-clear-answer")):
        for m in MODELS:
            got = load_moralchoice(amb, m)
            if got is None:
                continue
            one, avg, sv = got
            # matched design: one observation per cell
            r = row_for(one, "MoralChoice", claim, m, "scenario",
                        extra=dict(design="single-sample", seed_var=sv))
            rows.append(r)
            # replicate-averaged design: phi_broad becomes identifiable because
            # the seed variance can be netted out of the residual
            s2i, s2k, s2e, ncl = components(avg)
            noise = sv / 5.0                      # variance of a 5-sample mean
            s2_inter = max(s2e - noise, 0.0)
            # Broad phi is on the five-sample cell-mean scale.  Run noise is
            # removed from its numerator but remains in total measured variance.
            tot = s2i + s2k + s2e
            rows.append(dict(
                corpus="MoralChoice", claim=claim, model=m, unit="scenario",
                design="seed-averaged", n_items=avg.shape[0], K_full=avg.shape[1],
                mean_y=float(avg.mean()), seed_var=sv,
                sigma2_item=s2i, sigma2_probe=s2k, sigma2_resid=s2e,
                n_clamped=ncl,
                phi_fullK_mom=phi_mom(avg), phi_fullK_ssratio=phi_ssratio(avg),
                phi_broad_noisecorrected=((s2k + s2_inter) / tot)
                if tot > 0 else np.nan,
                frac_ceiling=float((avg.mean(axis=1) >= 0.9).mean()),
                frac_floor=float((avg.mean(axis=1) <= 0.1).mean()),
                n_items_mid=int(mid_mask(avg).sum()),
                phi_fullK_mid_mom=(phi_mom(avg[mid_mask(avg)])
                                   if mid_mask(avg).sum() >= 20 else np.nan),
            ))
            print(f"  {amb:4s} {m:32s} phi_strict(single,K6)={r['phi_fullK_mom']:.4f} "
                  f"mid={r['phi_fullK_mid_mom'] if np.isfinite(r['phi_fullK_mid_mom']) else float('nan'):.4f} "
                  f"ceil={100*r['frac_ceiling']:.0f}%", flush=True)

    print("\n=== PromptEval / MMLU (capability), question level ===", flush=True)
    for f in sorted(PE.glob("*.parquet")):
        subject, model = f.stem.split("__", 1)
        A = load_prompteval(f)
        if A is None:
            continue
        keep = mid_mask(A)
        if keep.sum() >= 20:
            prompteval_mid.append(dict(subject=subject, model=model, matrix=A[keep]))
        r = row_for(A, "PromptEval", "capability", model, "question",
                    extra=dict(design="single-sample", subject=subject,
                               seed_var=np.nan))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "harmonised_crosscorpus.csv", index=False)
    pe = t[t.corpus == "PromptEval"]
    print(f"  {len(pe)} subject-model matrices, "
          f"median {pe.n_items.median():.0f} questions x {pe.K_full.median():.0f} templates")

    # ------------------------------------------------------------- the table
    def blk(d, col):
        v = d[col].dropna()
        if not len(v):
            return "     n/a"
        return f"{v.median():.4f}"

    print("\n" + "=" * 92)
    print("HARMONISED phi_strict  (probe main-effect share; one obs per cell; MoM estimator)")
    print("=" * 92)
    hdr = (f"{'corpus / claim':32s} {'n':>4s} {'fullK':>8s} {'K=6':>8s} "
           f"{'K=6,mid':>8s} {'fullK,mid':>10s} {'ceil%':>6s}")
    print(hdr)
    print("-" * 92)
    for (corpus, claim), d in t[t.design == "single-sample"].groupby(["corpus", "claim"]):
        print(f"{corpus + ' / ' + claim:32s} {len(d):4d} "
              f"{blk(d,'phi_fullK_mom'):>8s} {blk(d,'phi_K6_mom'):>8s} "
              f"{blk(d,'phi_K6_mid_mom'):>8s} {blk(d,'phi_fullK_mid_mom'):>10s} "
              f"{100*d.frac_ceiling.median():6.0f}")

    print("\n--- the contrast, computed like for like (K=6, mid-difficulty items) ---")
    cap = t[(t.corpus == "PromptEval")].phi_K6_mid_mom.dropna()
    beh = t[(t.claim == "behavioural-dilemma") & (t.design == "single-sample")].phi_K6_mid_mom.dropna()
    bec = t[(t.claim == "behavioural-clear-answer") & (t.design == "single-sample")].phi_K6_mid_mom.dropna()
    for name, v in (("capability (MMLU)", cap), ("behavioural, dilemmas", beh),
                    ("behavioural, clear-answer", bec)):
        if len(v):
            print(f"  {name:28s} median {v.median():.4f}  IQR "
                  f"[{v.quantile(.25):.4f}, {v.quantile(.75):.4f}]  n={len(v)}")
    if len(cap) and len(beh) and cap.median() > 0:
        print(f"\n  ratio dilemmas / capability      = {beh.median()/cap.median():.1f}x")
    if len(cap) and len(bec) and cap.median() > 0:
        print(f"  ratio clear-answer / capability  = {bec.median()/cap.median():.1f}x")

    # The per-cell K=6 median above is useful for a table, but it hides how
    # template subsampling moves the corpus-level comparison. Keep the complete
    # draw distribution as a release artifact. MoralChoice has exactly six
    # conditions, so its mid-difficulty estimate is fixed in this comparison.
    beh_mid = float(beh.median()) if len(beh) else np.nan
    draw_rows = []
    for draw in range(200):
        vals = []
        for record in prompteval_mid:
            M = record["matrix"]
            cols = RNG.choice(M.shape[1], size=6, replace=False)
            v = phi_mom(M[:, cols])
            if np.isfinite(v):
                vals.append(v)
        cap_draw = float(np.median(vals)) if vals else np.nan
        draw_rows.append(dict(draw=draw, capability_phi=cap_draw,
                              dilemma_phi=beh_mid,
                              dilemma_to_capability=(beh_mid / cap_draw)
                              if np.isfinite(cap_draw) and cap_draw > 0 else np.nan,
                              n_capability_cells=len(vals)))
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(HERE / "harmonised_template_draws.csv", index=False)
    if len(draws.dropna(subset=["dilemma_to_capability"])):
        ratios = draws.dilemma_to_capability.dropna()
        print("\n--- template-draw uncertainty for the illustrative contrast ---")
        print(f"  dilemma/capability ratio: median {ratios.median():.1f}x; "
              f"95% central range [{ratios.quantile(.025):.1f}, "
              f"{ratios.quantile(.975):.1f}] across {len(ratios)} draws")

    print("\n--- estimator choice alone (full K, all items) ---")
    for (corpus, claim), d in t[t.design == "single-sample"].groupby(["corpus", "claim"]):
        print(f"  {corpus + ' / ' + claim:32s} MoM {d.phi_fullK_mom.median():.4f}   "
              f"SS-ratio {d.phi_fullK_ssratio.median():.4f}")

    print("\n--- clamping of negative variance components ---")
    tot_cells = len(t)
    print(f"  cells with >=1 component clamped at zero: {int((t.n_clamped>0).sum())}"
          f" of {tot_cells}")

    print("\n--- phi_broad, only where replicates make it identifiable ---")
    sa = t[t.design == "seed-averaged"]
    for claim, d in sa.groupby("claim"):
        print(f"  {claim:28s} phi_broad(noise-corrected) median "
              f"{d.phi_broad_noisecorrected.median():.4f}   "
              f"phi_strict median {d.phi_fullK_mom.median():.4f}   n={len(d)}")
    print("  (PromptEval carries one sample per cell, so its interaction term is")
    print("   confounded with Bernoulli noise and phi_broad is NOT identifiable there.)")


if __name__ == "__main__":
    main()
