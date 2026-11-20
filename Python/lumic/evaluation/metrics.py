"""Compare spectral factors with reference estimates or simulation truth.

Spectra are put on a unit-sum scale and their paired abundances are rescaled
before comparison. This preserves the reconstructed image while making each
abundance an expected count contribution across the detector channels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class FactorEvaluation:
    """Matched factors and errors for one fitted spectral image."""

    component_order: np.ndarray
    matching_costs: np.ndarray
    estimated_abundances: np.ndarray
    estimated_spectra: np.ndarray
    reconstruction_relative_l2: float
    spectral_relative_l2_per_component: np.ndarray
    spectral_relative_l2_global: float
    max_channel_absolute: float
    max_error_component_index: int
    max_error_channel_index: int
    abundance_rmse_per_component: np.ndarray
    abundance_rmse_global: float
    abundance_relative_l2_per_component: np.ndarray
    abundance_relative_l2_global: float


@dataclass(frozen=True)
class AlignedFactors:
    """Unit-sum factors in reference order, with original estimated row indices."""

    abundances: np.ndarray
    spectra: np.ndarray
    component_order: np.ndarray
    matching_costs: np.ndarray


@dataclass(frozen=True)
class SpectralComparison:
    """Per-component differences after optional spectral row matching.

    ``shape_distance`` uses unit-L2 spectra; ``max_channel_change`` and
    ``max_channel_index`` use unit-sum spectra. ``component_order`` indexes
    the original estimated rows. ``matching_costs`` are unit-L2 distances
    for the chosen pairing, even when row order is supplied directly.
    """

    component_order: np.ndarray
    matching_costs: np.ndarray
    shape_distance: np.ndarray
    max_channel_change: np.ndarray
    max_channel_index: np.ndarray


def _matrix(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2D array")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _unit_rows(spectra):
    lengths = np.linalg.norm(spectra, axis=1, keepdims=True)
    return np.divide(
        spectra, lengths, out=np.zeros_like(spectra), where=lengths != 0
    )


def _canonical_factors(abundances, spectra):
    """Give spectra unit sum and transfer each removed scale to abundances."""
    abundances = _matrix(abundances, "abundances")
    spectra = _matrix(spectra, "spectra")
    if abundances.shape[1] != spectra.shape[0]:
        raise ValueError("abundances and spectra must have matching components")
    scale = spectra.sum(axis=1)
    normalized = np.divide(
        spectra, scale[:, None], out=np.zeros_like(spectra),
        where=scale[:, None] != 0,
    )
    # A zero spectrum contributes nothing to the image, irrespective of its
    # abundance coefficient. Its canonical abundance is therefore zero.
    return abundances * scale, normalized


def match_components(true_spectra, estimated_spectra):
    """Return estimated row indices in true-component order and their costs.

    Matching minimizes summed squared Euclidean distance between unit-L2
    spectral shapes. The first argument can also be a previous fit or blind
    starting spectra; its historical name does not require ground truth.
    Matching gives a correspondence, not proof of biological identity. A zero
    estimated spectrum is allowed and has unit distance from every nonzero
    true spectrum; a zero true spectrum cannot define a spectral identity.
    """
    true_spectra = _matrix(true_spectra, "true_spectra")
    estimated_spectra = _matrix(estimated_spectra, "estimated_spectra")
    if true_spectra.shape != estimated_spectra.shape or not true_spectra.size:
        raise ValueError("true and estimated spectra must have the same nonempty shape")
    if np.any(np.linalg.norm(true_spectra, axis=1) == 0):
        raise ValueError("each true spectrum must be nonzero")

    true_unit = _unit_rows(true_spectra)
    estimated_unit = _unit_rows(estimated_spectra)
    costs = np.linalg.norm(
        true_unit[:, None, :] - estimated_unit[None, :, :], axis=2
    )
    true_indices, estimated_indices = linear_sum_assignment(costs ** 2)
    order = np.empty(len(true_indices), dtype=int)
    order[true_indices] = estimated_indices
    return order, costs[np.arange(len(order)), order]


def align_factors(reference_spectra, abundances, spectra):
    """Match fitted factors to reference spectra and give spectra unit sum.

    Purpose
    -------
    ALS can return components in a different order or with different scales.
    Use this before comparing repeated fits, plotting concentration maps, or
    collecting sampling/initialization results. Reference spectra may be known
    signatures, blind starts, or a previous fit: ground truth is not required.

    Parameters and returns
    ----------------------
    ``reference_spectra`` and ``spectra`` have the same nonempty shape
    (components, channels). Reference rows must be nonzero. ``abundances``
    is (pixels, components). All inputs must be finite and nonnegative.
    Returns an AlignedFactors object. Spectral matching minimizes total squared
    unit-L2 distance. Each fitted spectrum is divided by its sum and its paired
    abundance multiplied by that sum, then both are reordered together.
    The product C @ S is preserved. Inputs are not modified. A zero fitted
    spectrum gets zero canonical abundance because it contributes no signal.

    Interpretation and limits
    -------------------------
    Canonical abundance measures each component's contribution summed across
    the supplied channels, in the data's intensity units (counts only when
    data are counts). It is not automatically a molecular concentration.
    Matching assumes equal component counts; it does not find missing species
    or establish identity. Similar signatures can make assignments ambiguous.
    Alignment alone provides neither accuracy nor uncertainty estimates.

    Examples
    --------
    A swapped and rescaled fit is made comparable to a previous fit:

    >>> aligned = align_factors([[1, 0], [0, 1]], [[3, 4]],
    ...                         [[0, 2], [5, 0]])
    >>> aligned.component_order.tolist()
    [1, 0]
    >>> aligned.abundances.tolist()
    [[20.0, 6.0]]
    >>> aligned.spectra.tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """
    abundances, spectra = _canonical_factors(abundances, spectra)
    order, costs = match_components(reference_spectra, spectra)
    return AlignedFactors(abundances[:, order], spectra[order], order, costs)


def compare_spectra(reference_spectra, estimated_spectra, *, match=True):
    """Measure spectral shape and channel changes relative to a reference.

    Purpose
    -------
    Use across fitting pools, sampling seeds, initializations or solver versions
    to identify which recovered signatures change most. With known true spectra
    these are recovery errors; with another fit or blind starts they measure
    change only. This function works without known abundances or ground truth.

    Parameters and returns
    ----------------------
    Both arrays must be finite, nonnegative, and have the same nonempty shape
    (components, channels). With ``match=True``, reference rows must be nonzero
    and estimated rows are matched by total squared unit-L2 distance.
    With ``match=False``, rows are compared in their supplied order, allowing
    zero references; use this after alignment to a common reference so that
    trajectories keep the same component IDs.

    Returns SpectralComparison. ``shape_distance`` is the Euclidean distance
    between unit-L2 rows (0 for identical shapes; sqrt(2) for orthogonal nonzero
    nonnegative shapes). ``max_channel_change`` is the largest absolute change
    in each unit-sum row, a fraction of total channel signal; multiplying by
    100 gives percentage points. ``max_channel_index`` identifies that channel
    (first on a tie). Pure scaling causes no change in either metric. A zero
    row remains zero under normalization; its shape distance from a nonzero
    row is 1, a convention for absent signal rather than a defined shape.

    Interpretation and limits
    -------------------------
    A small change establishes stability relative to this reference, not
    correctness. No confidence interval, noise significance, identity, or
    convergence guarantee is inferred. Scale/intensity differences are removed
    deliberately; inspect abundances separately. Similar components may have
    ambiguous assignments, and missing/extra components are not supported.

    Examples
    --------
    Reordering and scaling alone do not constitute spectral change:

    >>> change = compare_spectra([[1, 0], [0, 1]], [[0, 2], [3, 0]])
    >>> change.shape_distance.tolist()
    [0.0, 0.0]
    >>> change.max_channel_change.tolist()
    [0.0, 0.0]

    For an already aligned trajectory, preserve its component IDs:

    >>> change = compare_spectra([[1, 0]], [[1, 1]], match=False)
    >>> change.max_channel_change.tolist()
    [0.5]
    """
    reference = _matrix(reference_spectra, "reference_spectra")
    estimate = _matrix(estimated_spectra, "estimated_spectra")
    if reference.shape != estimate.shape or not reference.size:
        raise ValueError("reference and estimated spectra must have the same nonempty shape")
    if not isinstance(match, (bool, np.bool_)):
        raise ValueError("match must be a boolean")
    if match:
        order, costs = match_components(reference, estimate)
    else:
        order = np.arange(len(reference))
        costs = np.linalg.norm(_unit_rows(reference) - _unit_rows(estimate), axis=1)
    estimate = estimate[order]
    # Unit-sum differences answer a different question from unit-L2 shape
    # differences: how much of the total spectrum shifted in one channel?
    def unit_sum(rows):
        totals = rows.sum(axis=1, keepdims=True)
        return np.divide(rows, totals, out=np.zeros_like(rows), where=totals != 0)
    difference = np.abs(unit_sum(estimate) - unit_sum(reference))
    channels = np.argmax(difference, axis=1)
    return SpectralComparison(order, costs,
        np.linalg.norm(_unit_rows(estimate) - _unit_rows(reference), axis=1),
        difference[np.arange(len(reference)), channels], channels)


def evaluate_factors(observed, true_abundances, true_spectra,
                     estimated_abundances, estimated_spectra):
    """Measure image fit and factor recovery after scale and order alignment.

    Inputs have shapes ``(pixels, channels)``, ``(pixels, components)``, and
    ``(components, channels)``. Relative abundance error is NaN for a true
    component with zero abundance; its count RMSE remains defined.
    """
    observed = _matrix(observed, "observed")
    if not observed.size:
        raise ValueError("observed must contain pixels and channels")
    true_abundances, true_spectra = _canonical_factors(
        true_abundances, true_spectra
    )
    estimated_abundances = _matrix(estimated_abundances, "estimated_abundances")
    estimated_spectra = _matrix(estimated_spectra, "estimated_spectra")
    if (true_spectra.shape != estimated_spectra.shape
            or observed.shape != (true_abundances.shape[0], true_spectra.shape[1])
            or estimated_abundances.shape != true_abundances.shape):
        raise ValueError("observed image and factor shapes must agree")

    aligned = align_factors(true_spectra, estimated_abundances, estimated_spectra)
    order, matching_costs = aligned.component_order, aligned.matching_costs
    estimated_abundances, estimated_spectra = aligned.abundances, aligned.spectra

    spectral_difference = _unit_rows(estimated_spectra) - _unit_rows(true_spectra)
    spectral_relative_l2_per_component = np.linalg.norm(
        spectral_difference, axis=1
    )
    spectral_relative_l2_global = float(
        np.linalg.norm(spectral_difference) / np.sqrt(len(order))
    )
    component, channel = np.unravel_index(
        np.argmax(np.abs(spectral_difference)), spectral_difference.shape
    )

    abundance_difference = estimated_abundances - true_abundances
    abundance_rmse_per_component = np.sqrt(
        np.mean(abundance_difference ** 2, axis=0)
    )
    abundance_rmse_global = float(np.sqrt(np.mean(abundance_difference ** 2)))
    true_abundance_lengths = np.linalg.norm(true_abundances, axis=0)
    abundance_relative_l2_per_component = np.divide(
        np.linalg.norm(abundance_difference, axis=0), true_abundance_lengths,
        out=np.full(len(order), np.nan), where=true_abundance_lengths != 0,
    )
    true_abundance_length = np.linalg.norm(true_abundances)
    abundance_relative_l2_global = (
        float(np.linalg.norm(abundance_difference) / true_abundance_length)
        if true_abundance_length else float("nan")
    )

    observed_length = np.linalg.norm(observed)
    reconstruction_relative_l2 = (
        float(np.linalg.norm(
            observed - estimated_abundances @ estimated_spectra
        ) / observed_length)
        if observed_length else float("nan")
    )
    return FactorEvaluation(
        component_order=order,
        matching_costs=matching_costs,
        estimated_abundances=estimated_abundances,
        estimated_spectra=estimated_spectra,
        reconstruction_relative_l2=reconstruction_relative_l2,
        spectral_relative_l2_per_component=spectral_relative_l2_per_component,
        spectral_relative_l2_global=spectral_relative_l2_global,
        max_channel_absolute=float(np.abs(spectral_difference[component, channel])),
        max_error_component_index=int(component),
        max_error_channel_index=int(channel),
        abundance_rmse_per_component=abundance_rmse_per_component,
        abundance_rmse_global=abundance_rmse_global,
        abundance_relative_l2_per_component=abundance_relative_l2_per_component,
        abundance_relative_l2_global=abundance_relative_l2_global,
    )
