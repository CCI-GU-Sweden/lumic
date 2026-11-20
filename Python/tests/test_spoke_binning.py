"""Spatial sum-bin controls; blind recovery is evaluated, not asserted."""

import numpy as np
import pytest
import json

from evaluations.evaluate_radial_spoke_binning import project_original, sum_bin


def test_sum_bins_preserve_counts_channels_and_spatial_blocks():
    image = np.arange(4 * 8 * 3).reshape(4, 8, 3)
    before = image.copy()
    binned = sum_bin(image, 2)
    assert binned.shape == (2, 4, 3)
    np.testing.assert_array_equal(binned[0, 0], image[:2, :2].sum(axis=(0, 1)))
    np.testing.assert_array_equal(binned[-1, -1], image[2:4, 6:8].sum(axis=(0, 1)))
    np.testing.assert_array_equal(binned.sum(axis=(0, 1)), image.sum(axis=(0, 1)))
    np.testing.assert_array_equal(sum_bin(binned, 2), sum_bin(image, 4))
    np.testing.assert_array_equal(image, before)


def test_sum_binning_commutes_with_linear_spectral_mixing():
    counts = np.arange(1., 65.).reshape(4, 4, 4)
    spectra = np.random.default_rng(1).random((4, 7))
    np.testing.assert_allclose(sum_bin(counts @ spectra, 2), sum_bin(counts, 2) @ spectra)
    # No noise is drawn by binning; it pools the actual recorded values.
    np.testing.assert_array_equal(sum_bin(np.ones((4, 4, 3)), 4), np.full((1, 1, 3), 16.))


def test_factor_one_returns_independent_equal_image():
    image = np.ones((2, 4, 3))
    binned = sum_bin(image, 1)
    np.testing.assert_array_equal(binned, image)
    binned[0, 0] = 0
    assert np.all(image == 1)


@pytest.mark.parametrize("factor", [0, -1, True, 1.5, 3])
def test_invalid_or_nondividing_factors_rejected(factor):
    with pytest.raises(ValueError, match="factor"):
        sum_bin(np.ones((4, 8, 3)), factor)


@pytest.mark.parametrize("image", [np.ones((3, 4)), np.full((4, 4, 3), np.nan),
                                  np.full((4, 4, 3), -1), np.ones((4, 4, 3), complex)])
def test_invalid_images_rejected(image):
    with pytest.raises(ValueError, match="image"):
        sum_bin(image, 2)


def test_projection_returns_original_pixel_counts_without_bin_area_scaling(tmp_path):
    observed = np.arange(1., 65.).reshape(4, 4, 4)
    coarse = tmp_path / "discovery"; coarse.mkdir()
    spectra = np.eye(4)
    (coarse / "settings.json").write_text(json.dumps(dict(initial_spectra=spectra.tolist())))
    discovery = dict(folder=coarse, spectra=spectra, history=np.array([1.]),
        row=dict(iterations=1, stop="converged", fit_pixels=4, eligible_pixels=4))
    fit = project_original(observed, discovery, tmp_path / "original_projection")
    with np.load(fit["folder"] / "fit.npz") as saved:
        np.testing.assert_allclose(saved["abundances"], observed)
        np.testing.assert_array_equal(saved["spectra"], spectra)
        assert "indices" not in saved.files  # No coarse indices mislabeled as fine.
    assert fit["row"]["relative_l2"] < 1e-12
    np.testing.assert_allclose(fit["rms"], 0, atol=1e-12)
