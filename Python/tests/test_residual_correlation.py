"""Known spatial/spectral patterns and compatibility with the pilot definitions."""

import numpy as np
import pytest

from lumic.evaluation import residual_neighbor_correlation, residual_spectral_correlation


def reference_neighbor(image):
    channels = image.shape[-1]
    left = np.concatenate([image[:, :-1].reshape(-1, channels), image[:-1].reshape(-1, channels)])
    right = np.concatenate([image[:, 1:].reshape(-1, channels), image[1:].reshape(-1, channels)])
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    return np.sum(left * right) / (np.linalg.norm(left) * np.linalg.norm(right))


@pytest.mark.parametrize("batch_size", [1, 7, 65536])
def test_neighbor_matches_pilot_without_wrapping_and_preserves_input(batch_size):
    image = np.random.default_rng(6).normal(size=(4, 7, 3)) * [1, 2, 10]
    before = image.copy()
    expected = reference_neighbor(image)
    assert residual_neighbor_correlation(image, batch_size=batch_size) == pytest.approx(expected)
    assert residual_neighbor_correlation(np.moveaxis(image, -1, 0), spectral_axis=0,
                                         batch_size=batch_size) == pytest.approx(expected)
    assert residual_neighbor_correlation(5 * image + [10, -3, 4],
                                         batch_size=batch_size) == pytest.approx(expected)
    np.testing.assert_array_equal(image, before)


def test_spatial_checkerboard_and_single_row_patterns():
    checker = 1 - 2 * (np.indices((5, 6)).sum(axis=0) % 2)
    assert residual_neighbor_correlation(checker[..., None]) == pytest.approx(-1)
    ramp = np.arange(6, dtype=np.uint16)[None, :, None]
    assert residual_neighbor_correlation(ramp) == pytest.approx(1)
    assert residual_neighbor_correlation(ramp.transpose(1, 0, 2)) == pytest.approx(1)


@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 4, 2), (1, 5, 1)])
def test_constant_spatial_residual_is_undefined(shape):
    assert np.isnan(residual_neighbor_correlation(np.full(shape, .1)))


@pytest.mark.parametrize("batch_size", [1, 3, 65536])
def test_spectral_matches_independent_pearson_per_pixel(batch_size):
    image = np.random.default_rng(1).normal(size=(3, 4, 8))
    before = image.copy()
    expected = np.array([np.corrcoef(row[:-1], row[1:])[0, 1]
                         for row in image.reshape(-1, 8)]).reshape(3, 4)
    np.testing.assert_allclose(residual_spectral_correlation(image, batch_size=batch_size), expected)
    np.testing.assert_allclose(residual_spectral_correlation(np.moveaxis(image, -1, 0),
                               spectral_axis=0, batch_size=batch_size), expected)
    np.testing.assert_allclose(residual_spectral_correlation(image * 4 + 10,
                               batch_size=batch_size), expected)
    np.testing.assert_array_equal(image, before)


def test_spectral_ramp_alternation_and_undefined_endpoint():
    image = np.array([[0, 1, 2, 3, 4], [1, -1, 1, -1, 1], [0, 0, 0, 0, 1]], dtype=float)
    result = residual_spectral_correlation(image)
    np.testing.assert_allclose(result[:2], [1, -1])
    assert np.isnan(result[2])  # One endpoint vector is constant.


@pytest.mark.parametrize("channels", [1, 2, 5])
def test_short_or_constant_spectra_are_nan_without_warnings(channels):
    with np.errstate(all="raise"):
        result = residual_spectral_correlation(np.full((2, 3, channels), .1))
    assert result.shape == (2, 3)
    assert np.isnan(result).all()


@pytest.mark.parametrize("function", [residual_neighbor_correlation, residual_spectral_correlation])
@pytest.mark.parametrize("bad", [np.array([1, 2]), np.empty((0, 2, 3)),
                                np.ones((2, 2, 3), complex), np.full((2, 2, 3), np.nan),
                                np.full((2, 2, 3), np.inf), np.array([[['text']]])])
def test_invalid_arrays_rejected(function, bad):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("function", [residual_neighbor_correlation, residual_spectral_correlation])
@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(batch_size=True),
                                   dict(batch_size=1.5), dict(spectral_axis=3),
                                   dict(spectral_axis=True)])
def test_invalid_options_rejected(function, kwargs):
    with pytest.raises(ValueError):
        function(np.ones((2, 2, 3)), **kwargs)


def test_neighbor_rejects_flat_pixels_without_spatial_layout():
    with pytest.raises(ValueError, match="two spatial axes"):
        residual_neighbor_correlation(np.ones((10, 4)))
