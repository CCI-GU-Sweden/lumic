"""Batch nonnegative least squares for a small number of components."""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np


MAX_BATCH_COMPONENTS = 6
# Retained subset matrices must stay modest even for unusual tall bases.
MAX_PREPARATION_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class PreparedBatchSolver:
    """Private, call-local subset decompositions for fixed components."""

    component_scales: np.ndarray
    active: np.ndarray
    basis: np.ndarray
    subsets: Optional[tuple]


def _subset_decompositions(basis):
    """Yield one subset at a time, retaining the original enumeration order."""
    for size in range(1, basis.shape[1] + 1):
        for subset in combinations(range(basis.shape[1]), size):
            indices = np.asarray(subset)
            selected_basis = basis[:, indices]
            # SVD avoids the squared condition number of normal equations.
            yield indices, selected_basis, np.linalg.pinv(selected_basis)


def prepare_small_batch(components, *, cache_subsets=True):
    """Prepare fixed-component matrices once, or return None for legacy NNLS.

    With four components there are only 16 possible sets of nonzero
    coefficients. We can solve each set for all observations together, reject
    negative coefficients, and choose the smallest residual for each row.
    This minimizes the same objective as NNLS without thousands of separate
    solver calls. Enumeration grows exponentially, so use it only for at most
    six components. Rank-deficient or ill-conditioned systems use the existing
    NNLS/BVLS path instead. Preparation does not depend on pixel observations,
    so every batch in a fitting call can reuse it. ALS must prepare again when
    it changes the components. Nothing is cached globally or between calls.
    For a single batch, leave subsets lazy: the spectral ALS step has only
    channel-many observations but may have millions of basis rows (pixels).
    Holding every subset there would increase memory without any reuse.
    Even with multiple batches, stream if the estimated retained matrices
    would exceed 8 MiB; large allocations must not be a cost of this fast path.
    """
    if (components.shape[0] > MAX_BATCH_COMPONENTS or not components.size
            or np.iscomplexobj(components)):
        return None
    # Match the original solver's double precision even for float32 images.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            components = np.asarray(components, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(components).all():
        return None

    # Scaling each basis vector improves conditioning without changing the
    # possible reconstructions. A vanished ALS component stays at zero.
    component_scales = np.max(np.abs(components), axis=1)
    active = np.flatnonzero(component_scales > 0)
    if not active.size:
        return PreparedBatchSolver(component_scales, active,
                                   np.empty((components.shape[1], 0)), ())
    basis = (components[active] / component_scales[active, None]).T
    singular_values = np.linalg.svd(basis, compute_uv=False)
    if (len(singular_values) < len(active)
            or singular_values[-1] <= singular_values[0] * 1e-8):
        return None

    # Each component occurs in half the nonempty subsets. Both the selected
    # basis and its inverse contain that many floats, with swapped dimensions.
    retained_bytes = 2 * basis.shape[0] * len(active) * 2**(len(active)-1) * basis.itemsize
    cache_subsets = cache_subsets and retained_bytes <= MAX_PREPARATION_BYTES
    subsets = tuple(_subset_decompositions(basis)) if cache_subsets else None
    return PreparedBatchSolver(component_scales, active, basis, subsets)


def solve_prepared_batch(mixed_data, prepared):
    """Solve observations using prepared matrices; None requests legacy NNLS.

    Observation scaling, feasibility checks, subset order, residual comparison
    and the preference for simpler exact reconstructions match the original
    batch solver. Only component-dependent decompositions are reused. Bad
    observations or nonfinite results still delegate to the existing fallback.
    """
    if not mixed_data.size or np.iscomplexobj(mixed_data):
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            mixed_data = np.asarray(mixed_data, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mixed_data).all():
        return None
    active, component_scales = prepared.active, prepared.component_scales
    answer = np.zeros((mixed_data.shape[0], len(component_scales)))
    if not active.size:
        return answer

    # Scale observations too, so squared residuals remain representable for
    # bright and dim inputs. Each observation has its own positive scale.
    observation_scales = np.max(np.abs(mixed_data), axis=1)
    observation_scales[observation_scales == 0] = 1
    targets = (mixed_data / observation_scales[:, None]).T
    best_coefficients = np.zeros((len(active), mixed_data.shape[0]))
    best_error = np.sum(targets**2, axis=0)  # The all-zero solution is allowed.
    residual_floor = (100 * np.finfo(float).eps)**2 * best_error

    subsets = (prepared.subsets if prepared.subsets is not None
               else _subset_decompositions(prepared.basis))
    for indices, selected_basis, inverse in subsets:
        coefficients = inverse @ targets
        feasible = (coefficients >= 0).all(axis=0)
        residual = targets - selected_basis @ coefficients
        error = np.sum(residual**2, axis=0)
        # Keep simpler exact reconstructions, including vanished components.
        improved = feasible & (error < best_error) & (best_error > residual_floor)
        best_coefficients[:, improved] = 0
        best_coefficients[np.ix_(indices, np.flatnonzero(improved))] = (
            coefficients[:, improved]
        )
        best_error[improved] = error[improved]

    answer[:, active] = (
        best_coefficients.T * observation_scales[:, None]
        / component_scales[active]
    )
    return answer if np.isfinite(answer).all() else None


def solve_small_batch(mixed_data, components):
    """Convenience entry point for a single batch, retaining the private API."""
    prepared = prepare_small_batch(components, cache_subsets=False)
    return None if prepared is None else solve_prepared_batch(mixed_data, prepared)
