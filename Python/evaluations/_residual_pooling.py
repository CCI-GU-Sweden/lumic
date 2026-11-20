"""Experimental, observed-data-only residual selection and spatial diagnostics.

These helpers support a bounded evaluation, not a public LUMIC selection API.
Neither morphology nor spatial correlation establishes correct factor recovery.
"""

import numpy as np
from lumic.evaluation import residual_neighbor_correlation
from lumic.evaluation.sampling import _candidate_masks


def candidate_masks(scores, eligible, *, quantile=.8, opening_radius=1, informative=True):
    """Evaluation adapter to the shared threshold/opening primitive.

    Full-mask opening precedes eligibility intersection so hull exclusions
    do not create artificial support holes. See select_residual_pixels for
    public sampling semantics and limits; this adapter retains pilot labels.
    """
    return _candidate_masks(scores, eligible, quantile=quantile,
                            opening_radius=opening_radius, informative=informative)


def addition_pools(hull, eligible, masks, requested_budget, seed):
    """Keep the hull and sample equal-budget additions without ground truth.

    Empty shortlists are unavailable, not silently replaced by another strategy.
    Among available strategies, reduce the common budget to the smallest
    candidate set so every compared addition has the same size. Uniform pixel
    sampling within spatial regions weights larger regions more heavily; it
    does not guarantee spectral or region balance. Exact indices are reproducible.
    """
    eligible = np.asarray(eligible, dtype=bool).copy()
    hull = np.asarray(hull)
    if (eligible.ndim != 2 or hull.ndim != 1 or not hull.size or hull.dtype.kind not in "iu"
            or np.any(hull < 0) or np.any(hull >= eligible.size)):
        raise ValueError("hull must contain valid integer indices for a 2D eligibility map")
    if isinstance(requested_budget, (bool, np.bool_)) or not isinstance(requested_budget, (int, np.integer)) or requested_budget < 0:
        raise ValueError("requested_budget must be a nonnegative integer")
    hull = np.unique(hull)
    eligible.flat[hull] = False
    choices = {"uniform": np.flatnonzero(eligible)}
    for name, key in (("high_residual", "high"), ("spatial_residual", "spatial")):
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape != eligible.shape:
            raise ValueError("candidate masks must match eligibility")
        choices[name] = np.flatnonzero(mask & eligible)
    available = {name: indices for name, indices in choices.items() if indices.size}
    budget = min([requested_budget] + [len(indices) for indices in available.values()]) if available else 0
    pools = {}
    for name, candidates in available.items():
        added = np.random.default_rng(seed).permutation(candidates)[:budget]
        pools[name] = np.union1d(hull, added)
    return {"pools": pools, "actual_budget": int(budget),
            "candidate_counts": {name: len(indices) for name, indices in choices.items()},
            "unavailable": [name for name in choices if name not in available]}


def spatial_diagnostics(observed, residuals, *, relative_floor=1e-10):
    """Describe residual size and spatial support; no ground truth is needed.

    Channel-last 2D images have shape (y, x, channels). Neighbor correlation is
    the normalized cross-product of horizontal/vertical residual pairs after
    centering each channel on each side. Channels with more residual variance
    contribute more. RMS-versus-signal correlation measures brightness
    association, not shot-noise significance. Both are descriptive, not tests
    for randomness. A flatter map is not evidence of correct spectra/counts.

    When the global relative residual is below the explicit practical floor,
    spatial correlations are NaN and residual-guided selection should be
    skipped. This prevents ranking near-zero arithmetic differences. The floor
    is a diagnostic convention, not a noise calibration or universal precision
    bound. Mean/max RMS remain available in input intensity units.
    """
    observed, residuals = np.asarray(observed, dtype=float), np.asarray(residuals, dtype=float)
    if (observed.ndim != 3 or observed.shape != residuals.shape or not observed.size
            or not np.isfinite(observed).all() or not np.isfinite(residuals).all()):
        raise ValueError("observed and residuals must be finite matching channel-last 2D images")
    if not np.isscalar(relative_floor) or not np.isfinite(relative_floor) or relative_floor < 0:
        raise ValueError("relative_floor must be finite and nonnegative")
    signal_norm = np.linalg.norm(observed)
    relative = float(np.linalg.norm(residuals)/signal_norm) if signal_norm else np.nan
    rms = np.sqrt(np.mean(residuals**2, axis=-1))
    result = {"relative_l2": relative, "mean_rms": float(rms.mean()), "max_rms": float(rms.max()),
              "residual_above_floor": bool(np.isfinite(relative) and relative > relative_floor),
              "neighbor_correlation": np.nan, "rms_signal_correlation": np.nan}
    if not result["residual_above_floor"]:
        return result
    result["neighbor_correlation"] = residual_neighbor_correlation(residuals)
    a, b = rms.ravel()-rms.mean(), observed.sum(axis=-1).ravel()-observed.sum(axis=-1).mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if denominator:
        result["rms_signal_correlation"] = float(np.dot(a,b)/denominator)
    return result
