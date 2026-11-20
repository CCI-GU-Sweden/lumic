from pathlib import Path

import numpy as np
import pytest

from lumic.simulation import SpectralDetector, simulate_linear_mixture


EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "io-data" / "980"


def test_unequal_bins_integrate_a_flat_spectrum_by_channel_width():
    detector = SpectralDetector([0, 1, 3])

    sampled = detector.sample([0, 3], [2, 2])

    np.testing.assert_allclose(sampled, [2, 4])
    assert sampled.sum() == pytest.approx(6)
    np.testing.assert_allclose(detector.channel_centers, [0.5, 2])


def test_peak_crossing_channel_boundary_conserves_integrated_signal():
    detector = SpectralDetector([0, 0.5, 1.5, 2])

    sampled = detector.sample([0, 1, 2], [0, 2, 0])

    np.testing.assert_allclose(sampled, [0.25, 1.5, 0.25])
    assert sampled.sum() == pytest.approx(2)


def test_detector_does_not_extrapolate_signal_beyond_source_wavelengths():
    detector = SpectralDetector([0, 1, 2, 3, 4])

    sampled = detector.sample([1.5, 2.5], [1, 1])

    np.testing.assert_allclose(sampled, [0, 0.5, 0.5, 0])


def test_980_profile_matches_recorded_channel_centers_and_midpoint_edges():
    labels = np.fromstring((EXAMPLES / "all-channels-labels.txt").read_text(), sep=",")
    detector = SpectralDetector.from_profile("980_32ch")
    edges = detector.bin_edges

    assert detector.name == "980_32ch"
    assert detector.nominal_resolution_nm == 8.9
    assert detector.n_channels == 32
    np.testing.assert_array_equal(detector.channel_centers, labels)
    np.testing.assert_allclose(edges[1:-1], (labels[:-1] + labels[1:]) / 2)
    assert edges[0] == pytest.approx(410.5)
    assert edges[-1] == pytest.approx(693.5)


def test_sampled_component_spectra_form_expected_linear_image():
    detector = SpectralDetector([0, 1, 3])
    spectra = np.stack(
        [
            detector.sample([0, 3], [2, 2]),
            detector.sample([0, 3], [0, 2]),
        ]
    )
    abundances = np.array([[[1, 0], [0, 1], [0.5, 0.5]]])

    dataset = simulate_linear_mixture(
        abundances, spectra, detector.channel_centers, ("flat", "rising")
    )

    np.testing.assert_allclose(dataset.image[0], [[2, 4], [1 / 3, 8 / 3], [7 / 6, 10 / 3]])


@pytest.mark.parametrize("edges", [[0], [0, 1, 1], [0, 2, 1], [0, np.nan]])
def test_detector_rejects_invalid_bin_edges(edges):
    with pytest.raises(ValueError, match="bin_edges"):
        SpectralDetector(edges)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown spectral detector profile"):
        SpectralDetector.from_profile("unknown")
