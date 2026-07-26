"""Estimate repeat-aware strict and broad probe fractions from new judge repeats.

Input must be produced by repeated_judge_mtbench.py with at least two repeats
per item/order cell. The residual variance is estimated from within-cell repeat
variation, allowing the item-by-order interaction to be separated from judge
sampling noise. This estimator is specific to the newly chosen judge model.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SCORE = {"A": 1.0, "B": 0.0, "TIE": 0.5}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    args = p.parse_args()
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    cells = {}
    totals = {"g1": 0, "g2": 0}
    unparsed = {"g1": 0, "g2": 0}
    for r in rows:
        order = r["order"]
        totals[order] = totals.get(order, 0) + 1
        if r["winner"] not in SCORE:
            unparsed[order] = unparsed.get(order, 0) + 1
            continue
        cells.setdefault((r["item_id"], r["order"]), []).append(SCORE[r["winner"]])
    items = sorted({item for item, _ in cells})
    if not items: raise ValueError("No parseable decisions")
    if any(len(cells.get((item, order), [])) < 2 for item in items for order in ("g1", "g2")):
        raise ValueError("Every item/order cell needs at least two parseable repeats")
    counts = {len(cells[item, order]) for item in items for order in ("g1", "g2")}
    if len(counts) != 1:
        raise ValueError(
            "Balanced ANOVA requires the same number of parseable repeats in "
            "every item/order cell"
        )
    repeat_count = counts.pop()
    values = np.array([
        [cells[item, order] for order in ("g1", "g2")] for item in items
    ], dtype=float)
    n, k, repeat_count = values.shape
    grand = values.mean()
    item_means = values.mean(axis=(1, 2))
    order_means = values.mean(axis=(0, 2))
    cell_means = values.mean(axis=2)
    interaction = (
        cell_means - item_means[:, None] - order_means[None, :] + grand
    )

    ms_item = (
        k * repeat_count * ((item_means - grand) ** 2).sum() / (n - 1)
    )
    ms_probe = (
        n * repeat_count * ((order_means - grand) ** 2).sum() / (k - 1)
    )
    ms_interaction = (
        repeat_count * (interaction ** 2).sum() / ((n - 1) * (k - 1))
    )
    residual = values - cell_means[:, :, None]
    ms_error = (residual ** 2).sum() / (n * k * (repeat_count - 1))

    s2_item = max((ms_item - ms_interaction) / (k * repeat_count), 0.0)
    s2_probe = max((ms_probe - ms_interaction) / (n * repeat_count), 0.0)
    s2_interaction = max((ms_interaction - ms_error) / repeat_count, 0.0)
    s2_error = max(float(ms_error), 0.0)

    total_individual = s2_item + s2_probe + s2_interaction + s2_error
    total_cell_mean = (
        s2_item + s2_probe + s2_interaction + s2_error / repeat_count
    )
    print(f"items={n}; parseable repeats/cell={repeat_count}")
    for order in ("g1", "g2"):
        rate = unparsed.get(order, 0) / max(totals.get(order, 0), 1)
        print(f"unparsed_{order}={unparsed.get(order, 0)}/{totals.get(order, 0)} "
              f"({rate:.2%})")
    print(
        f"sigma2_item={s2_item:.6f}; sigma2_probe={s2_probe:.6f}; "
        f"sigma2_interaction={s2_interaction:.6f}; "
        f"sigma2_within_judge_run={s2_error:.6f}"
    )
    print(
        "individual-judgment scale: "
        f"phi_strict={s2_probe/total_individual:.6f}; "
        f"phi_broad={(s2_probe+s2_interaction)/total_individual:.6f}"
    )
    print(
        "reported cell-mean scale: "
        f"phi_strict={s2_probe/total_cell_mean:.6f}; "
        f"phi_broad={(s2_probe+s2_interaction)/total_cell_mean:.6f}"
    )
    print(
        "Interpretation: repeated stochastic calls separate interaction from "
        "within-judge run noise for this one judge model and prompt. They do "
        "not estimate disagreement across judge models."
    )

if __name__ == "__main__": main()
