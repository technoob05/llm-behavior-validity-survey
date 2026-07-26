"""Optional analysis of probe-variance composition in a crossed-design run.

The input contains one frozen model answering each item under a crossed panel
of persona, paraphrase, and option-order views, with a G-theory variance
decomposition stored per benchmark and model. It covers twelve benchmarks and
five open-weight models. Several benchmarks are cultural or cross-national:
NormAd, GlobalOpinionQA, DICES, and D3CODE.

What this does and does not give us. The run stored the variance components
WITHIN each item, averaged over items, and did not retain the between-item term.
So we can report the composition of phi's NUMERATOR, which is what the probe
contributes and how it splits across the three axes, but not phi itself, which
needs the signal in the denominator. We label it that way rather than quietly
calling it phi.

This optional companion analysis is not load bearing because its source matrix
is not distributed with the public artifact. Set CROSSBENCH_COMPONENTS_JSON to
an authorised local copy before running it.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.environ.get("CROSSBENCH_COMPONENTS_JSON", ""))

CULTURAL = {"normad_none", "normad_country", "normad_value", "normad_cval",
            "normad_rot", "globalopinions", "dices350", "dices990", "d3code"}


def main():
    if not os.environ.get("CROSSBENCH_COMPONENTS_JSON") or not SRC.is_file():
        print("Set CROSSBENCH_COMPONENTS_JSON to an authorised local JSON file.")
        return
    d = json.loads(SRC.read_text(encoding="utf8"))
    rows = []
    for model, payload in d["backbones"].items():
        fv = payload.get("facet_variance", {})
        for bench, v in fv.items():
            tot = v.get("framing_total", 0.0)
            if not tot or tot <= 0:
                continue
            rows.append(dict(
                model=model, bench=bench,
                persona=v["sigma2_persona"], para=v["sigma2_para"],
                order=v["sigma2_order"], residual=v["sigma2_residual"],
                framing_total=tot,
                pct_order=v["pct_order"], pct_persona=v["pct_persona"],
                pct_para=v["pct_para"], n_items=v.get("n_items", np.nan),
                cultural=bench in CULTURAL))
    t = pd.DataFrame(rows)
    print("cells (benchmark x model): %d  |  benchmarks: %d  |  models: %d"
          % (len(t), t.bench.nunique(), t.model.nunique()))
    print("benchmarks:", ", ".join(sorted(t.bench.unique())))
    print("models    :", ", ".join(sorted(t.model.unique())))

    print("\n=== which probe axis carries the framing variance ===")
    print("  median share of the probe (framing) variance, over all %d cells:" % len(t))
    print("     option order      %5.1f%%" % t.pct_order.median())
    print("     persona           %5.1f%%" % t.pct_persona.median())
    print("     paraphrase        %5.1f%%" % t.pct_para.median())
    print("  cells where option order is the largest of the three: %d of %d (%.0f%%)"
          % (int((t.pct_order >= t[["pct_order", "pct_persona", "pct_para"]].max(axis=1)).sum()),
             len(t),
             100 * (t.pct_order >= t[["pct_order", "pct_persona", "pct_para"]].max(axis=1)).mean()))

    print("\n=== cultural / cross-national benchmarks vs the rest ===")
    for flag, lab in ((True, "cultural or cross-national"), (False, "other")):
        s = t[t.cultural == flag]
        if not len(s):
            continue
        print("  %-28s n=%2d   order %5.1f%%   persona %5.1f%%   paraphrase %5.1f%%"
              % (lab, len(s), s.pct_order.median(), s.pct_persona.median(),
                 s.pct_para.median()))

    print("\n=== per benchmark (median over the %d models) ===" % t.model.nunique())
    g = (t.groupby("bench")
           .agg(order=("pct_order", "median"), persona=("pct_persona", "median"),
                para=("pct_para", "median"), probe_var=("framing_total", "median"),
                resid=("residual", "median"), n=("bench", "size"))
           .sort_values("order", ascending=False))
    print(g.round(2).to_string())

    print("\n  probe variance relative to within-item residual, median over cells: %.2f"
          % (t.framing_total / t.residual.replace(0, np.nan)).median())

    t.to_csv(Path(__file__).resolve().parent / "crossbench_probe_composition.csv",
             index=False)


if __name__ == "__main__":
    main()
