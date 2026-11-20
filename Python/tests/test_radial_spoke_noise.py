"""Scientific controls for the exploratory shot-noise evaluation."""

import numpy as np
import pytest

from evaluations.evaluate_radial_spoke_noise import save_report, simulate_measurement


def test_count_scaling_preserves_component_ratios_and_clean_mixture():
    counts = np.arange(1., 25.).reshape(2, 3, 4)
    spectra = np.eye(4)
    before = counts.copy()
    truth, expected, measured = simulate_measurement(counts, spectra, .25, None)
    np.testing.assert_array_equal(truth, counts * .25)
    np.testing.assert_array_equal(expected, truth @ spectra)
    np.testing.assert_array_equal(measured, expected)
    np.testing.assert_allclose(truth / truth.sum(axis=-1, keepdims=True),
                               counts / counts.sum(axis=-1, keepdims=True))
    measured[0, 0, 0] = 0
    assert expected[0, 0, 0] > 0  # Clean and measured arrays are independent.
    np.testing.assert_array_equal(counts, before)


def test_noisy_measurement_is_repeatable_nonnegative_and_keeps_truth_separate():
    counts = np.full((4, 5, 4), 20.)
    spectra = np.array([[.1, .9], [.2, .8], [.4, .6], [.5, .5]])
    truth, expected, first = simulate_measurement(counts, spectra, .5, 0)
    _, _, repeat = simulate_measurement(counts, spectra, .5, 0)
    _, _, other = simulate_measurement(counts, spectra, .5, 1)
    np.testing.assert_array_equal(first, repeat)
    assert np.any(first != other)
    assert np.all(first >= 0) and np.all(first == np.floor(first))
    np.testing.assert_array_equal(expected, truth @ spectra)
    assert np.any(first != expected)  # Seed zero is a noisy case, not a control.
    np.testing.assert_array_equal(counts, np.full((4, 5, 4), 20.))


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_count_scales_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        simulate_measurement(np.ones((2, 3, 4)), np.eye(4), scale, None)


def test_partial_report_survives_a_failed_first_case_without_positive_plot_data(tmp_path):
    # An exploratory run must remain inspectable even if its first hull fails.
    rows = [dict(blur_fwhm_um=0., count_scale=1., noise_seed=2026,
                 max_iterations=500, long_run=False, status="failed: unavailable hull")]
    save_report(tmp_path, rows)
    assert (tmp_path / "overview.png").stat().st_size > 0
    assert "failed: unavailable hull" in (tmp_path / "README.md").read_text()
    assert "0 / 1" in (tmp_path / "README.md").read_text()
