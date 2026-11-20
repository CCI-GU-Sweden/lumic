"""Hand-calculated diagnostics, distinct from optimizer recovery tests."""

import numpy as np
import pytest

from lumic.evaluation import (concentration_error_maps, evaluate_factors,
                              reconstruction_error_maps)


def test_reconstruction_maps_preserve_spatial_axes_and_signed_residuals():
    measured = np.array([[[3, 4], [0, 0]], [[0, 2], [1, 0]]])
    prediction = np.array([[[0, 4], [1, 0]], [[0, 3], [0, 0]]])
    maps = reconstruction_error_maps(measured, prediction, min_signal=1)
    np.testing.assert_array_equal(maps.residuals,
                                  [[[3, 0], [-1, 0]], [[0, -1], [1, 0]]])
    np.testing.assert_allclose(maps.rms, [[3/np.sqrt(2), 1/np.sqrt(2)],
                                        [1/np.sqrt(2), 1/np.sqrt(2)]])
    np.testing.assert_allclose(maps.relative_l2, [[0.6, np.nan], [0.5, np.nan]])
    np.testing.assert_array_equal(maps.relative_valid, [[True, False], [True, False]])


def test_exact_reconstruction_keeps_zero_signal_relative_error_undefined():
    data = [[0, 0], [-3, 4]]  # Offset-corrected signed data are supported.
    maps = reconstruction_error_maps(data, data)
    np.testing.assert_array_equal(maps.rms, [0, 0])
    np.testing.assert_allclose(maps.relative_l2, [np.nan, 0])


def test_concentration_maps_match_and_scale_without_hiding_false_positives():
    truth = np.array([[[2, 0], [4, 2]]])
    spectra = np.eye(2)
    # Reversed components; estimated spectra carry scales 2 and 4.
    estimate = np.array([[[0.5, 0.75], [0.5, 0.5]]])
    estimated_spectra = np.array([[0, 2], [4, 0]])
    maps = concentration_error_maps(truth, spectra, estimate, estimated_spectra,
                                    min_concentration=2)
    np.testing.assert_array_equal(maps.component_order, [1, 0])
    np.testing.assert_allclose(maps.signed, [[[1, 1], [-2, -1]]])
    np.testing.assert_allclose(maps.absolute, [[[1, 1], [2, 1]]])
    np.testing.assert_allclose(maps.relative, [[[np.nan, np.nan], [-0.5, np.nan]]])
    np.testing.assert_array_equal(maps.relative_valid, [[[False, False], [True, False]]])
    metrics = evaluate_factors(truth.reshape(-1, 2), truth.reshape(-1, 2),
                              spectra, estimate.reshape(-1, 2), estimated_spectra)
    np.testing.assert_allclose(np.sqrt(np.mean(maps.signed**2, axis=(0, 1))),
                               metrics.abundance_rmse_per_component)


def test_canonical_truth_scale_and_zero_estimated_component():
    maps = concentration_error_maps([[2, 3]], [[2, 0], [0, 1]],
                                    [[4, 999]], [[1, 0], [0, 0]])
    np.testing.assert_allclose(maps.signed, [[0, -3]])
    np.testing.assert_allclose(maps.relative, [[0, -1]])


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, [1]])
def test_invalid_thresholds_rejected(threshold):
    with pytest.raises(ValueError, match="min_signal"):
        reconstruction_error_maps([[1]], [[1]], min_signal=threshold)
    with pytest.raises(ValueError, match="min_concentration"):
        concentration_error_maps([[1]], [[1]], [[1]], [[1]],
                                 min_concentration=threshold)


@pytest.mark.parametrize("bad", [[], [[np.nan]], [[np.inf]], [1, 2], np.empty((2, 0))])
def test_invalid_images_rejected(bad):
    with pytest.raises(ValueError):
        reconstruction_error_maps(bad, bad)


def test_incompatible_shapes_and_invalid_truth_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        reconstruction_error_maps([[1, 2]], [[1]])
    with pytest.raises(ValueError, match="identical shapes"):
        concentration_error_maps([[1, 2]], np.eye(2), [[1]], [[1, 0]])
    with pytest.raises(ValueError, match="nonnegative"):
        concentration_error_maps([[-1]], [[1]], [[1]], [[1]])
    with pytest.raises(ValueError, match="nonzero"):
        concentration_error_maps([[1]], [[0]], [[1]], [[1]])


def test_good_reconstruction_can_coexist_with_wrong_factors():
    truth = np.array([[2., 1.], [4., 2.]])
    true_spectra = np.eye(2)
    estimate = np.array([[1., 2.], [2., 4.]])
    estimated_spectra = np.array([[1., 0.], [0.5, 0.5]])
    reconstruction = reconstruction_error_maps(truth, estimate @ estimated_spectra)
    recovery = concentration_error_maps(truth, true_spectra, estimate, estimated_spectra)
    np.testing.assert_array_equal(reconstruction.rms, [0, 0])
    assert np.any(recovery.absolute > 0)
