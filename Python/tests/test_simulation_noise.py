from pathlib import Path

import numpy as np
import pytest

from lumic import positive_least_squares
from lumic.simulation import (
    SpectralDetector,
    add_confocal_noise,
    simulate_noisy_image,
)


def test_component_counts_determine_clean_pixel_totals_and_channel_signals():
    detector = SpectralDetector([0, 1, 2])
    source_spectra = [
        ([0, 1, 2], [2, 2, 0]),
        ([0, 1, 2], [0, 0, 2]),
    ]
    component_counts = np.array([[[60.0, 30.0], [0.0, 10.0]]])

    result = simulate_noisy_image(
        component_counts,
        source_spectra,
        detector,
        fluorophore_names=("green", "red"),
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(result.sampled_spectra, [[2, 1], [0, 1]])
    np.testing.assert_allclose(result.ideal.spectra, [[2 / 3, 1 / 3], [0, 1]])
    np.testing.assert_allclose(result.expected_counts, [[[40, 50], [0, 10]]])
    np.testing.assert_allclose(
        result.expected_counts.sum(axis=-1), component_counts.sum(axis=-1)
    )
    np.testing.assert_array_equal(result.ideal.abundances, component_counts)
    np.testing.assert_array_equal(component_counts, [[[60, 30], [0, 10]]])
    assert result.ideal.fluorophore_names == ("green", "red")
    assert result.noisy_image.shape == result.expected_counts.shape
    np.testing.assert_array_equal(result.source_spectra[0][1], [2, 2, 0])


def test_noisy_mixture_can_recover_component_counts_to_stated_tolerance():
    detector = SpectralDetector([0, 1, 2, 3])
    spectra = [
        ([0, 1, 2, 3], [1, 3, 1, 0]),
        ([0, 1, 2, 3], [0, 1, 2, 1]),
    ]
    component_counts = np.array(
        [[[7000, 3000], [2000, 8000]], [[5000, 5000], [10000, 0]]]
    )

    result = simulate_noisy_image(
        component_counts,
        spectra,
        detector,
        gaussian_sigma=5.0,
        rng=np.random.default_rng(24),
    )
    recovered = positive_least_squares(
        result.noisy_image.reshape(-1, detector.n_channels),
        result.ideal.spectra,
    ).reshape(component_counts.shape)

    np.testing.assert_allclose(recovered, component_counts, rtol=0, atol=500)


def test_poisson_noise_residuals_and_recovery_change_sensibly_with_counts():
    """Check shot-noise scale and concentration recovery using saved spectra."""
    spectra_dir = Path(__file__).resolve().parents[1] / "lumic" / "simulation" / "fpbase_data"
    names = ("EGFP", "mCherry")
    spectra = []
    for name in names:
        curve = np.loadtxt(
            spectra_dir / f"{name.lower()}_emission.csv",
            delimiter=",", skiprows=1,
        )
        spectra.append((curve[:, 0], curve[:, 1]))

    detector = SpectralDetector.from_profile("980_32ch")
    first_fraction = np.linspace(0.0, 1.0, 400).reshape(20, 20)
    fractions = np.stack((1.0 - first_fraction, first_fraction), axis=-1)
    relative_errors = []
    count_levels = (10000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0)

    # A fixed seed keeps this a repeatable regression check. Lower count levels
    # make Poisson fluctuations larger relative to the known signal.
    for seed, counts_per_pixel in enumerate(count_levels):
        true_counts = fractions * counts_per_pixel
        result = simulate_noisy_image(
            true_counts,
            spectra,
            detector,
            fluorophore_names=names,
            rng=np.random.default_rng(900 + seed),
        )

        # With pure Poisson noise, residual variance in each detector channel
        # equals its expected count. Check that relationship in aggregate.
        oracle_residual = result.noisy_image - result.expected_counts
        variance_to_mean_ratio = np.mean(oracle_residual**2) / np.mean(
            result.expected_counts
        )
        assert variance_to_mean_ratio == pytest.approx(1.0, rel=0.06)

        recovered_counts = positive_least_squares(
            result.noisy_image.reshape(-1, detector.n_channels),
            result.ideal.spectra,
        ).reshape(true_counts.shape)
        relative_errors.append(
            np.sqrt(np.mean((recovered_counts - true_counts) ** 2))
            / counts_per_pixel
        )

    # Relative error should grow as the expected counts fall, while remaining
    # substantially smaller at high counts than in the low-count case.
    assert np.all(np.diff(relative_errors) > 0)
    assert relative_errors[-1] > 10 * relative_errors[0]


def test_poisson_noise_example_saves_preview_using_bundled_spectra(tmp_path):
    from PIL import Image
    from examples.visualize_poisson_noise_levels import create_preview

    output = tmp_path / "poisson-noise.png"
    assert create_preview(output) == output
    with Image.open(output) as preview:
        assert preview.format == "PNG"
        preview.verify()


def test_noise_is_reproducible_and_drawn_separately_for_each_pixel():
    expected = np.full((4, 4, 3), 500.0)

    first = add_confocal_noise(
        expected, gaussian_sigma=5.0, rng=np.random.default_rng(123)
    )
    second = add_confocal_noise(
        expected, gaussian_sigma=5.0, rng=np.random.default_rng(123)
    )

    np.testing.assert_array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 1
    np.testing.assert_array_equal(expected, np.full((4, 4, 3), 500.0))


def test_zero_signal_and_zero_gaussian_noise_remain_zero():
    expected = np.zeros((2, 3, 4))

    noisy = add_confocal_noise(expected, rng=np.random.default_rng(2))

    np.testing.assert_array_equal(noisy, expected)


def test_gaussian_noise_at_zero_signal_is_clipped_below_zero():
    expected = np.zeros((5, 5, 2))

    noisy = add_confocal_noise(
        expected, gaussian_sigma=2.0, rng=np.random.default_rng(13)
    )

    assert np.all(noisy >= 0)
    assert np.any(noisy == 0)
    assert np.any(noisy > 0)


def test_simulator_rejects_spectrum_without_detector_signal():
    detector = SpectralDetector([500, 510, 520])

    with pytest.raises(ValueError, match="signal in the detector range"):
        simulate_noisy_image(
            np.ones((1, 1, 1)),
            [([400, 410], [1, 1])],
            detector,
            rng=np.random.default_rng(1),
        )


@pytest.mark.parametrize(
    ("expected", "sigma", "message"),
    [
        (np.array([[[-1.0]]]), 0.0, "non-negative"),
        (np.array([[[np.nan]]]), 0.0, "finite"),
        (np.array([[[1.0]]]), -1.0, "gaussian_sigma"),
    ],
)
def test_noise_rejects_invalid_inputs(expected, sigma, message):
    with pytest.raises(ValueError, match=message):
        add_confocal_noise(expected, gaussian_sigma=sigma, rng=np.random.default_rng(1))
