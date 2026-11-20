"""Small known-factor checks for reusable simulation recovery metrics."""

import numpy as np

from lumic.evaluation import evaluate_factors, match_components


def test_matching_and_scale_alignment_preserve_component_counts():
    true_spectra = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    true_abundances = np.array([[3.0, 1.0], [0.0, 2.0]])
    observed = true_abundances @ true_spectra

    # Swap the rows and change each spectrum's scale. The paired abundance
    # changes inversely, so these factors still generate the same image.
    estimated_spectra = np.array([[0.0, 0.0, 8.0], [1.0, 0.0, 0.0]])
    estimated_abundances = np.array([[0.5, 6.0], [1.0, 0.0]])
    result = evaluate_factors(
        observed, true_abundances, true_spectra,
        estimated_abundances, estimated_spectra,
    )

    np.testing.assert_array_equal(result.component_order, [1, 0])
    np.testing.assert_allclose(result.estimated_spectra.sum(axis=1), 1)
    np.testing.assert_allclose(result.estimated_abundances, [[6, 4], [0, 8]])
    np.testing.assert_allclose(result.matching_costs, 0, atol=1e-15)
    np.testing.assert_allclose(result.spectral_relative_l2_per_component, 0)
    np.testing.assert_allclose(result.abundance_rmse_per_component, 0)
    np.testing.assert_allclose(result.reconstruction_relative_l2, 0)
    np.testing.assert_array_equal(estimated_spectra, [[0, 0, 8], [1, 0, 0]])
    np.testing.assert_array_equal(estimated_abundances, [[0.5, 6], [1, 0]])


def test_component_errors_retain_count_and_signal_relative_scales():
    spectra = np.eye(2)
    true_abundances = np.array([[3.0, 1.0], [0.0, 2.0]])
    estimated_abundances = np.array([[4.0, 1.0], [0.0, 3.0]])
    result = evaluate_factors(
        true_abundances @ spectra, true_abundances, spectra,
        estimated_abundances, spectra,
    )

    np.testing.assert_allclose(
        result.abundance_rmse_per_component, [1 / np.sqrt(2)] * 2
    )
    np.testing.assert_allclose(
        result.abundance_relative_l2_per_component, [1 / 3, 1 / np.sqrt(5)]
    )
    np.testing.assert_allclose(result.abundance_rmse_global, np.sqrt(0.5))
    np.testing.assert_allclose(result.abundance_relative_l2_global, np.sqrt(2 / 14))
    np.testing.assert_allclose(result.reconstruction_relative_l2, np.sqrt(2 / 14))


def test_three_component_matching_uses_one_consistent_order():
    true_spectra = np.eye(3)
    true_abundances = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    estimated_spectra = np.array([
        [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.5, 0.0, 0.0],
    ])
    estimated_abundances = np.array([[1.0, 1.0, 2.0], [2.5, 2.0, 8.0]])
    result = evaluate_factors(
        true_abundances @ true_spectra, true_abundances, true_spectra,
        estimated_abundances, estimated_spectra,
    )

    np.testing.assert_array_equal(result.component_order, [2, 0, 1])
    np.testing.assert_allclose(result.estimated_abundances, true_abundances)
    np.testing.assert_allclose(result.spectral_relative_l2_per_component, 0)
    np.testing.assert_allclose(result.abundance_rmse_per_component, 0)


def test_spectral_errors_locate_the_component_and_channel():
    true_spectra = np.eye(2)
    true_abundances = np.array([[1.0, 1.0]])
    estimated_spectra = np.array([[1.0, 0.0], [1.0, 1.0]])
    estimated_abundances = np.array([[1.0, 0.5]])
    result = evaluate_factors(
        true_abundances @ true_spectra, true_abundances, true_spectra,
        estimated_abundances, estimated_spectra,
    )

    second_error = np.sqrt(2 - np.sqrt(2))
    np.testing.assert_allclose(
        result.spectral_relative_l2_per_component, [0, second_error]
    )
    np.testing.assert_allclose(
        result.spectral_relative_l2_global, second_error / np.sqrt(2)
    )
    np.testing.assert_allclose(result.max_channel_absolute, 1 / np.sqrt(2))
    assert (result.max_error_component_index, result.max_error_channel_index) == (1, 0)


def test_relative_error_is_undefined_for_an_absent_true_component():
    spectra = np.eye(2)
    true_abundances = np.array([[3.0, 0.0], [1.0, 0.0]])
    estimated_abundances = np.array([[3.0, 2.0], [1.0, 2.0]])
    result = evaluate_factors(
        true_abundances @ spectra, true_abundances, spectra,
        estimated_abundances, spectra,
    )

    assert np.isnan(result.abundance_relative_l2_per_component[1])
    np.testing.assert_allclose(result.abundance_rmse_per_component[1], 2)
    np.testing.assert_allclose(result.abundance_relative_l2_per_component[0], 0)


def test_zero_estimated_spectrum_is_recorded_as_missing_shape():
    true_spectra = np.eye(2)
    estimated_spectra = np.array([[1.0, 0.0], [0.0, 0.0]])
    order, costs = match_components(true_spectra, estimated_spectra)
    np.testing.assert_array_equal(order, [0, 1])
    np.testing.assert_allclose(costs, [0, 1])

    true_abundances = np.array([[1.0, 2.0]])
    result = evaluate_factors(
        true_abundances @ true_spectra, true_abundances, true_spectra,
        np.array([[1.0, 100.0]]), estimated_spectra,
    )
    np.testing.assert_allclose(result.spectral_relative_l2_per_component, [0, 1])
    np.testing.assert_allclose(result.estimated_abundances, [[1, 0]])
