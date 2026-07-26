"""Shared estimators for the survey's fragility analyses.

The MoralChoice files contain R repeated binary responses in every
item-by-probe cell.  We analyse the cell mean because that is the scale of the
reported preference rate.  Its two-way residual contains both item-by-probe
interaction and run noise divided by R.  Strict phi uses the probe main effect
over all variance of the reported cell mean.  Broad phi removes the estimated
run-noise contribution from the numerator, but retains it in the denominator.
"""

from __future__ import annotations

import numpy as np


def two_way_components(matrix):
    """Return non-negative item, probe, and residual variance components."""
    matrix = np.asarray(matrix, dtype=float)
    n_item, n_probe = matrix.shape
    if n_item < 2 or n_probe < 2:
        return None
    grand = matrix.mean()
    item_mean = matrix.mean(axis=1)
    probe_mean = matrix.mean(axis=0)
    ms_item = n_probe * ((item_mean - grand) ** 2).sum() / (n_item - 1)
    ms_probe = n_item * ((probe_mean - grand) ** 2).sum() / (n_probe - 1)
    residual = matrix - item_mean[:, None] - probe_mean[None, :] + grand
    ms_residual = (residual ** 2).sum() / ((n_item - 1) * (n_probe - 1))
    return (
        max(float((ms_item - ms_residual) / n_probe), 0.0),
        max(float((ms_probe - ms_residual) / n_item), 0.0),
        max(float(ms_residual), 0.0),
    )


def fragility_pair(matrix, within_cell_variance=None, repeats=1):
    """Return strict and broad phi on the cell-mean scale.

    ``within_cell_variance`` may be a scalar or an item-by-probe matrix.  Broad
    phi is unidentified when it is omitted.  The denominator always includes
    the run noise remaining in a cell mean.
    """
    components = two_way_components(matrix)
    if components is None:
        return np.nan, np.nan
    item_var, probe_var, residual_var = components
    total = item_var + probe_var + residual_var
    if total <= 0:
        return np.nan, np.nan
    strict = probe_var / total
    if within_cell_variance is None:
        return strict, np.nan
    noise_of_mean = float(np.nanmean(within_cell_variance)) / float(repeats)
    interaction_var = max(residual_var - noise_of_mean, 0.0)
    broad = (probe_var + interaction_var) / total
    return strict, broad
