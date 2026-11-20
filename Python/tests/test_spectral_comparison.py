"""Reference-fit comparisons must preserve signal and distinguish metrics."""

import numpy as np
import pytest

from lumic.evaluation import align_factors, compare_spectra


def test_alignment_preserves_reconstruction_and_does_not_mutate_inputs():
    reference = np.eye(3)
    spectra = np.array([[0., 0., 8.], [2., 0., 0.], [0., 4., 0.]])
    counts = np.array([[1., 3., 5.], [2., 4., 6.]])
    original_spectra, original_counts = spectra.copy(), counts.copy()
    aligned = align_factors(reference, counts, spectra)
    np.testing.assert_array_equal(aligned.component_order, [1, 2, 0])
    np.testing.assert_allclose(aligned.spectra, reference)
    np.testing.assert_allclose(aligned.abundances, [[6, 20, 8], [8, 24, 16]])
    np.testing.assert_allclose(aligned.abundances @ aligned.spectra, counts @ spectra)
    np.testing.assert_array_equal(spectra, original_spectra)
    np.testing.assert_array_equal(counts, original_counts)


def test_zero_fitted_component_has_zero_canonical_counts():
    aligned = align_factors(np.eye(2), [[7, 9]], [[2, 0], [0, 0]])
    np.testing.assert_allclose(aligned.abundances, [[14, 0]])
    np.testing.assert_allclose(aligned.spectra, [[1, 0], [0, 0]])
    change = compare_spectra(np.eye(2), aligned.spectra, match=False)
    np.testing.assert_allclose(change.shape_distance, [0, 1])
    np.testing.assert_allclose(change.max_channel_change, [0, 1])


def test_comparison_removes_scale_and_matches_swapped_rows():
    result = compare_spectra(np.eye(2), [[0, 5], [3, 0]])
    np.testing.assert_array_equal(result.component_order, [1, 0])
    np.testing.assert_allclose(result.shape_distance, 0)
    np.testing.assert_allclose(result.max_channel_change, 0)


def test_fixed_order_retains_identity_and_separates_shape_from_channel_change():
    # Rematching would hide this identity swap, so fixed-order comparisons
    # must report orthogonal shapes rather than silently permuting rows.
    result = compare_spectra(np.eye(2), [[0, 1], [1, 0]], match=False)
    np.testing.assert_array_equal(result.component_order, [0, 1])
    np.testing.assert_allclose(result.shape_distance, np.sqrt(2))
    np.testing.assert_allclose(result.max_channel_change, 1)
    np.testing.assert_array_equal(result.max_channel_index, [0, 0])
    result = compare_spectra([[1, 0]], [[1, 1]], match=False)
    np.testing.assert_allclose(result.shape_distance, [np.sqrt(2-np.sqrt(2))])
    np.testing.assert_allclose(result.max_channel_change, [.5])


def test_zero_reference_only_allowed_when_row_order_is_fixed():
    result = compare_spectra([[0, 0]], [[0, 0]], match=False)
    np.testing.assert_allclose(result.shape_distance, 0)
    with pytest.raises(ValueError, match="nonzero"):
        compare_spectra([[0, 0]], [[1, 0]])
    with pytest.raises(ValueError, match="nonzero"):
        align_factors([[0, 0]], [[1]], [[1, 0]])


@pytest.mark.parametrize("reference, estimate", [
    ([], []), ([[1, 0]], [[1, 0, 0]]),
    ([[1, -1]], [[1, 0]]), ([[1, 0]], [[np.nan, 0]]),
])
def test_comparison_rejects_invalid_spectra(reference, estimate):
    with pytest.raises(ValueError):
        compare_spectra(reference, estimate)


def test_alignment_rejects_incompatible_abundances():
    with pytest.raises(ValueError, match="matching components"):
        align_factors(np.eye(2), [[1]], np.eye(2))


def test_comparison_rejects_non_boolean_matching_flag():
    with pytest.raises(ValueError, match="boolean"):
        compare_spectra([[1, 0]], [[1, 0]], match="no")
