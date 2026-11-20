"""Traceable residual-guided pixel selection, without pool fitting or file I/O."""

from dataclasses import dataclass

import numpy as np
from skimage import measure, morphology


@dataclass(frozen=True)
class ResidualPixelSelection:
    """Added original-image indices and the masks/settings that selected them.

    ``indices`` contains sorted distinct ADDED indices only, in C-order spatial
    flattening. ``existing_indices`` contains the canonical excluded pool.
    ``eligible_mask`` is input eligibility after excluding existing pixels.
    Dictionary keys are pc1_positive, pc1_negative, etc. ``threshold_masks``
    and ``opened_masks`` retain full spatial support, including excluded pixels;
    ``candidate_masks`` intersects opened masks with eligible_mask. Candidate
    masks can overlap; groups records overlap exclusions and exact sampled rows.
    ``settings`` contains JSON-friendly parameters and actual seed entropy.
    Frozen fields do not make arrays/dictionaries immutable.
    """

    indices: np.ndarray
    existing_indices: np.ndarray
    eligible_mask: np.ndarray
    threshold_masks: dict
    opened_masks: dict
    candidate_masks: dict
    groups: dict
    settings: dict


def _candidate_masks(scores, eligible, *, quantile=.8, opening_radius=1, informative=True):
    """Shared nonnegative threshold/opening primitive used by evaluation pilots."""
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.ndim != 2 or scores.shape != eligible.shape or not scores.size:
        raise ValueError("scores and eligibility must have the same nonempty 2D shape")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be finite and nonnegative")
    try:
        valid_quantile = (not isinstance(quantile, (bool, np.bool_)) and np.isscalar(quantile)
                          and np.isfinite(quantile) and 0 <= quantile < 1)
    except TypeError:
        valid_quantile = False
    if not valid_quantile:
        raise ValueError("quantile must lie in [0, 1)")
    if isinstance(opening_radius, (bool, np.bool_)) or not isinstance(opening_radius, (int, np.integer)) or opening_radius < 0:
        raise ValueError("opening_radius must be a nonnegative integer")
    empty = np.zeros(scores.shape, dtype=bool)
    if not informative or not eligible.any():
        return {"high": empty.copy(), "spatial": empty.copy(),
                "labels": np.zeros(scores.shape, dtype=int), "threshold": None,
                "reason": "residual_below_floor" if not informative else "no_eligible_pixels"}
    threshold = float(np.quantile(scores[eligible], quantile))
    high = (scores >= threshold) & (scores > 0)
    if opening_radius:
        padded = np.pad(high, opening_radius, constant_values=False)
        opened = morphology.opening(padded, footprint=morphology.disk(opening_radius))
        spatial = opened[opening_radius:-opening_radius, opening_radius:-opening_radius].astype(bool)
    else:
        spatial = high.copy()
    return {"high": high, "spatial": spatial, "labels": measure.label(spatial, connectivity=1),
            "threshold": threshold, "reason": "available"}


def select_residual_pixels(scores, *, pcs, eligible=None, existing_indices=None,
                           per_sign_budget=512, score_quantile=.90,
                           opening_radius=1, seed=None):
    """Sample original image pixels from user-chosen signed residual-PC regions.

    Purpose
    -------
    Use after inspecting residual PCA maps when coherent spatial patterns merit
    adding examples to a fitting pool. The function selects pixels; it does not
    increase the component count, fit spectra, write files or establish that a
    missing fluorophore exists. Feed selected ORIGINAL measured spectra to a
    solver, not the signed residual scores themselves.

    Parameters
    ----------
    scores : real finite array, shape (y, x, PCs)
        Signed spatial scores, for example ResidualPCAResult.scores. Currently
        supports two spatial dimensions. Scores are raw pattern amplitudes,
        not calibrated noise-significance or purity values.
    pcs : sequence of distinct positive integers
        Human-facing one-based PC numbers, e.g. [1, 2] or [1]. Selection uses
        both signs of each PC. PC count and requested species count are separate.
    eligible : boolean array, shape (y, x), optional
        Pixels permitted for addition; defaults to all pixels. Use the mask
        from select_analysis_pixels to preserve intensity/validity exclusions.
    existing_indices : integer sequence, optional
        Existing fitting-pool indices in C-order spatial flattening. These are
        excluded from sampling and percentile calculation, never returned as
        additions. Repeated input indices are canonicalized.
    per_sign_budget : positive int, default 512
        Maximum distinct samples per selected PC/sign. Two PCs can add up to
        2048 pixels. Short groups supply fewer; no unrelated replacement pixels
        are used. Earlier groups exclude selected pixels from later groups.
    score_quantile : float in [0, 1), default .90
        Quantile over eligible non-pool pixels of max(sign*score, 0). Opposite
        signs contribute zeros to the percentile population; this is NOT a
        percentile computed only among that sign's nonzero pixels. Retain ties
        and strictly positive values. No exact candidate fraction is guaranteed.
    opening_radius : nonnegative int, default 1
        Opening with a scikit-image disk: radius one is a 3x3 cross. Zero disables
        opening. False padding treats outside-field pixels as absent support.
        Apply opening to the FULL threshold mask before intersecting eligibility
        so excluded/hull pixels do not create artificial holes in support.
    seed : nonnegative integer or None, default None
        Fixed seed for reproducible sampling; None chooses fresh entropy. The
        actual entropy is recorded and can be reused as a seed. Each PC/sign
        has a deterministic random stream; chosen PC order still affects overlap.

    Returns
    -------
    ResidualPixelSelection
        Added indices, full threshold/opened masks, eligible candidate masks,
        per-group thresholds, counts, reasons and sampled indices, and settings.
        Existing pool membership is not changed. Combine explicitly with
        np.union1d(existing_indices, result.indices) in the workflow layer.

    Limits
    ------
    Uniform sampling weights larger regions more heavily; no spectral/region
    balancing is imposed. Opening can erase real thin/small features and retain
    chance noise clusters. Raw scores can reflect brightness-dependent noise,
    detector effects or inaccurate existing spectra. More pool pixels or flatter
    same-image residuals do not prove recovery or significance. Fresh PCA can
    rotate/reorder directions: choose PCs from the current diagnostic result.
    Applying this to pixels used to fit the model is exploratory, not held-out
    validation. Callers manage stale diagnostics, pool history and rank decisions.

    Examples
    --------
    >>> scores = np.ones((2, 3, 1))
    >>> selected = select_residual_pixels(scores, pcs=[1], existing_indices=[0],
    ...                                  per_sign_budget=2, opening_radius=0, seed=7)
    >>> len(selected.indices)
    2
    >>> 0 in selected.indices
    False

    Typical workflow: selection = select_residual_pixels(pca.scores, pcs=[1, 2],
    eligible=retained.mask, existing_indices=pool_indices, seed=2026); then
    expanded_indices = np.union1d(pool_indices, selection.indices).
    """
    scores = np.asarray(scores)
    if (scores.ndim != 3 or not scores.size or np.iscomplexobj(scores)
            or not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all()):
        raise ValueError("scores must be a finite real nonempty (y, x, PC) array")
    if eligible is None:
        eligible = np.ones(scores.shape[:2], dtype=bool)
    else:
        eligible = np.asarray(eligible)
        if eligible.shape != scores.shape[:2] or eligible.dtype.kind != "b":
            raise ValueError("eligible must be a boolean mask matching the score spatial shape")
        eligible = eligible.copy()
    try:
        pcs = list(pcs)
    except TypeError:
        raise ValueError("pcs must contain distinct valid one-based PC numbers") from None
    if (not pcs or any(isinstance(pc, (bool, np.bool_)) or not isinstance(pc, (int, np.integer))
            or not 1 <= pc <= scores.shape[-1] for pc in pcs) or len(set(pcs)) != len(pcs)):
        raise ValueError("pcs must contain distinct valid one-based PC numbers")
    pcs = [int(pc) for pc in pcs]
    if isinstance(per_sign_budget, (bool, np.bool_)) or not isinstance(per_sign_budget, (int, np.integer)) or per_sign_budget < 1:
        raise ValueError("per_sign_budget must be a positive integer")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    existing = np.asarray([] if existing_indices is None else existing_indices)
    if (existing.ndim != 1 or (existing.size and existing.dtype.kind not in "iu")
            or np.any(existing < 0) or np.any(existing >= eligible.size)):
        raise ValueError("existing_indices must contain valid flattened integer pixel indices")
    existing = np.unique(existing.astype(int))
    eligible.flat[existing] = False
    entropy = int(np.random.SeedSequence(None if seed is None else int(seed)).entropy)
    added = np.array([], dtype=int)
    thresholds, opened, candidates_by_group, groups = {}, {}, {}, {}
    for pc in pcs:
        k = pc-1
        for sign in (1, -1):
            name = f"pc{pc}_{'positive' if sign == 1 else 'negative'}"
            masks = _candidate_masks(np.maximum(sign*scores[..., k].astype(float), 0), eligible,
                quantile=score_quantile, opening_radius=opening_radius)
            candidate_mask = masks["spatial"] & eligible
            all_candidates = np.flatnonzero(candidate_mask)
            available = np.setdiff1d(all_candidates, added)
            chosen = np.random.default_rng(np.random.SeedSequence([entropy, k, sign+1])).choice(
                available, min(per_sign_budget, len(available)), replace=False)
            added = np.union1d(added, chosen)
            thresholds[name], opened[name] = masks["high"], masks["spatial"]
            candidates_by_group[name] = candidate_mask
            groups[name] = dict(threshold=masks["threshold"], candidate_count=len(all_candidates),
                available_after_exclusion=len(available), selected=len(chosen), indices=chosen.tolist(),
                reason="selected" if chosen.size else ("overlap_exhausted" if all_candidates.size else
                    (masks["reason"] if masks["reason"] != "available" else "no_candidates")))
    settings = dict(pcs=pcs, per_sign_budget=int(per_sign_budget), score_quantile=float(score_quantile),
                    opening_radius=int(opening_radius), seed=None if seed is None else int(seed), seed_entropy=entropy)
    return ResidualPixelSelection(added, existing, eligible, thresholds, opened,
                                  candidates_by_group, groups, settings)
