import numpy as np
import pytest
from scipy.spatial import ConvexHull
from sklearn.decomposition import TruncatedSVD

from lumic import essential_spectral_pixles, spectral_uniqueness_map


def test_uniqueness_distinguishes_endpoints_from_the_center_mixture():
    # Balanced three-component spectra give a center at the equal mixture.
    # Three rows and two columns make spatial/axis mistakes visible.
    image = np.array([
        [[2, 0.2, 0.2], [0.2, 2, 0.2]],
        [[0.2, 0.2, 2], [0.8, 0.8, 0.8]],
        [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],
    ])
    result = spectral_uniqueness_map(image, npc=3, spectral_axis=-1)

    np.testing.assert_allclose(result.center_coordinates, 0, atol=1e-14)
    radius = np.sqrt(1.125)
    np.testing.assert_allclose(result.scores,
                               [[radius, radius], [radius, 0], [0, 0]], atol=1e-14)
    assert result.coordinates.shape == (3, 2, 2)
    np.testing.assert_array_equal(result.valid_mask, np.ones((3, 2), dtype=bool))
    np.testing.assert_allclose(result.total_intensity, image.sum(axis=-1))


@pytest.mark.parametrize("npc", [3, 4])
def test_map_matches_the_original_hull_svd_coordinate_formula(npc):
    image = np.random.default_rng(4).uniform(1, 10, (3, 4, 6))
    pixels = image.reshape(-1, 6)
    projected = TruncatedSVD(n_components=npc, random_state=0).fit_transform(pixels)
    expected = projected[:, 1:] / projected[:, :1]

    result = spectral_uniqueness_map(image, npc=npc, spectral_axis=-1)

    np.testing.assert_allclose(result.coordinates.reshape(-1, npc - 1), expected)
    np.testing.assert_allclose(result.center_coordinates, expected.mean(axis=0))
    np.testing.assert_allclose(result.scores.ravel(),
                               np.linalg.norm(expected - expected.mean(axis=0), axis=1))
    state = np.random.get_state()
    np.random.seed(0)
    try:
        _, indices = essential_spectral_pixles(pixels, npc=npc)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(indices, ConvexHull(expected).vertices)


def test_fixed_projection_cancels_independent_pixel_brightness_changes():
    image = np.random.default_rng(4).uniform(1, 10, (3, 2, 5))
    brighter = image * np.array([[1, 8], [3, 0.5], [20, 2]])[..., None]
    original = spectral_uniqueness_map(image, spectral_axis=-1)
    scaled = spectral_uniqueness_map(brighter, spectral_axis=-1, reference=original)

    np.testing.assert_allclose(scaled.coordinates, original.coordinates, atol=1e-14)
    np.testing.assert_allclose(scaled.scores, original.scores, atol=1e-14)
    np.testing.assert_array_equal(scaled.center_coordinates, original.center_coordinates)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_uniqueness_preserves_spatial_layout_with_different_channel_axes(axis):
    image = np.arange(1, 31).reshape(3, 2, 5)
    expected = spectral_uniqueness_map(image, spectral_axis=-1)
    actual = spectral_uniqueness_map(np.moveaxis(image, -1, axis), spectral_axis=axis)

    np.testing.assert_allclose(actual.scores, expected.scores)
    np.testing.assert_allclose(actual.coordinates, expected.coordinates)
    np.testing.assert_array_equal(actual.total_intensity, expected.total_intensity)


def test_filtering_excludes_bad_pixels_from_scores_and_the_center():
    image = np.array([
        [[20, 0, 4], [0, 20, 4], [0, 0, 0]],
        [[0, 0, 1], [np.nan, 0, 2], [-1, 10, 0]],
        [[np.inf, 0, 0], [0, 0, 0], [0, 0, 0]],
    ])
    result = spectral_uniqueness_map(image, npc=2, spectral_axis=-1,
                                     min_total_intensity=24)

    expected_mask = np.zeros((3, 3), dtype=bool)
    expected_mask[0, :2] = True
    np.testing.assert_array_equal(result.valid_mask, expected_mask)
    assert np.isnan(result.scores[~expected_mask]).all()
    assert np.isnan(result.coordinates[~expected_mask]).all()
    np.testing.assert_allclose(result.center_coordinates, 0, atol=1e-14)


def test_uniform_pure_spectra_have_zero_uniqueness():
    image = np.array([1.0, 0.2, 0.3])[:, None, None] * np.arange(1, 7).reshape(3, 2)
    result = spectral_uniqueness_map(image)
    np.testing.assert_allclose(result.scores, 0, atol=1e-14)


def test_pixels_with_zero_leading_projection_are_excluded():
    reference_image = np.array([[[10, 0, 0], [0, 5, 0], [10, 0, 0]]])
    result = spectral_uniqueness_map(reference_image, npc=2, spectral_axis=-1)
    np.testing.assert_array_equal(result.valid_mask, [[True, False, True]])
    assert np.isnan(result.scores[0, 1])


def test_hull_reports_an_unusable_leading_coordinate():
    pixels = np.array([[10, 0, 0], [0, 5, 0], [10, 0, 0]])
    with pytest.raises(ValueError, match="first SVD coordinate"):
        essential_spectral_pixles(pixels, npc=2)


def test_reference_rejects_a_different_channel_count():
    reference = spectral_uniqueness_map(np.ones((3, 2, 4)))
    with pytest.raises(ValueError, match="reference projection"):
        spectral_uniqueness_map(np.ones((4, 2, 4)), reference=reference)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, True, [1]])
def test_uniqueness_rejects_invalid_intensity_thresholds(threshold):
    with pytest.raises(ValueError, match="min_total_intensity"):
        spectral_uniqueness_map(np.ones((3, 2, 4)), min_total_intensity=threshold)


@pytest.mark.parametrize("npc", [1, 4, True, 2.5])
def test_uniqueness_rejects_invalid_component_counts(npc):
    with pytest.raises(ValueError, match="npc"):
        spectral_uniqueness_map(np.ones((3, 2, 4)), npc=npc)


@pytest.mark.parametrize("axis", [3, -4, True, 0.5])
def test_uniqueness_rejects_invalid_channel_axes(axis):
    with pytest.raises(ValueError, match="spectral_axis"):
        spectral_uniqueness_map(np.ones((3, 2, 4)), spectral_axis=axis)


@pytest.mark.parametrize("image", [np.zeros((3, 2, 4)), np.ones((3, 0, 4)), np.ones((3, 4))])
def test_uniqueness_reports_empty_or_fully_excluded_input(image):
    with pytest.raises(ValueError):
        spectral_uniqueness_map(image)


def _direction_result(offsets, valid=None):
    from lumic import SpectralUniquenessResult
    offsets = np.asarray(offsets, dtype=float)[None, ...]
    shape = offsets.shape[:-1]
    return SpectralUniquenessResult(
        scores=np.linalg.norm(offsets, axis=-1),
        valid_mask=np.ones(shape, dtype=bool) if valid is None else np.array([valid]),
        total_intensity=np.ones(shape), coordinates=offsets,
        center_coordinates=np.zeros(offsets.shape[-1]),
        projection_components=np.zeros((offsets.shape[-1] + 1, 5)),
    )


def test_direction_colors_encode_angles_distance_and_mask_without_changing_scores():
    from lumic import spectral_uniqueness_rgb
    result = _direction_result([[1, 0], [0, 1], [-1, 0], [0, -1],
                                [0.5, 0], [0, 0], [np.nan, np.nan]],
                               [True, True, True, True, True, True, False])
    original = result.scores.copy()
    colors = spectral_uniqueness_rgb(result, max_distance=1)
    np.testing.assert_allclose(colors[0, :6],
                               [[1, 0, 0], [0.5, 1, 0], [0, 1, 1],
                                [0.5, 0, 1], [0.5, 0.25, 0.25], [0, 0, 0]])
    np.testing.assert_allclose(colors[0, 6], 80 / 255)
    np.testing.assert_array_equal(result.scores, original)


def test_one_coordinate_colors_the_two_sides_and_handles_uniform_center():
    from lumic import spectral_uniqueness_rgb
    colors = spectral_uniqueness_rgb(_direction_result([[-1], [0], [1]]))
    np.testing.assert_allclose(colors[0], [[0, 1, 1], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(spectral_uniqueness_rgb(_direction_result([[0], [0]])), 0)
    result = spectral_uniqueness_map(np.array([[[4, 1], [1, 4]]]),
                                     npc=2, spectral_axis=-1)
    colors = spectral_uniqueness_rgb(result)
    assert colors.shape == (1, 2, 3)
    assert np.isfinite(colors).all()
    assert not np.allclose(colors[0, 0], colors[0, 1])


def test_higher_dimensions_affect_saturation_but_not_display_angle():
    from lumic import spectral_uniqueness_rgb
    result = _direction_result([[1, 0, 0], [1, 0, np.sqrt(3)], [0, 0, 2]])
    colors = spectral_uniqueness_rgb(result, max_distance=2)
    np.testing.assert_allclose(colors[0], [[0.5, 0.25, 0.25], [1, 0, 0], [0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(spectral_uniqueness_rgb(result, max_distance=0.5)[0, 1],
                               [1, 0, 0])
    assert result.scores[0, 1] == pytest.approx(2)


@pytest.mark.parametrize("maximum", [0, -1, np.nan, np.inf, True, [1], "bad"])
def test_direction_colors_reject_invalid_distance_scales(maximum):
    from lumic import spectral_uniqueness_rgb
    with pytest.raises(ValueError, match="max_distance"):
        spectral_uniqueness_rgb(_direction_result([[1, 0]]), max_distance=maximum)
