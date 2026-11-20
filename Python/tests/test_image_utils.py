import numpy as np
import pytest

from lumic.utils import RGB_from_spec, norm8bit, remove_low_intensity


def test_remove_low_intensity_keeps_pixels_above_median_quantile():
    pixels = np.array(
        [
            [0, 0, 1],
            [1, 1, 9],
            [2, 2, 0],
            [3, 3, 0],
        ]
    )
    expected = np.array([[2, 2, 0], [3, 3, 0]])

    with pytest.warns(FutureWarning, match="select_analysis_pixels"):
        actual = remove_low_intensity(pixels, quantile_val=0.5)
    np.testing.assert_array_equal(actual, expected)


def test_rgb_from_spec_preserves_shape_and_rgb_channel_order():
    channels = np.array([np.full((2, 3), value) for value in (10, 20, 30, 40, 50, 60)])
    # Blue averages 10 and 20, green uses 30, and red averages 40, 50, and 60.
    expected_rgb = np.broadcast_to([50, 30, 15], (2, 3, 3))

    np.testing.assert_array_equal(RGB_from_spec(channels, do_norm=False), expected_rgb)


def test_rgb_from_spec_with_three_channels_uses_each_channel_once():
    channels = np.array([[[10]], [[20]], [[30]]])

    np.testing.assert_array_equal(
        RGB_from_spec(channels, do_norm=False), np.array([[[30, 20, 10]]])
    )


def test_rgb_from_spec_with_five_channels_uses_nonempty_groups():
    channels = np.array([[[10]], [[20]], [[30]], [[40]], [[50]]])

    np.testing.assert_array_equal(
        RGB_from_spec(channels, do_norm=False), np.array([[[45, 25, 10]]])
    )


@pytest.mark.parametrize("n_channels", [0, 1, 2])
def test_rgb_from_spec_rejects_fewer_than_three_channels(n_channels):
    channels = np.zeros((n_channels, 1, 1))

    with pytest.raises(ValueError, match="at least three spectral channels"):
        RGB_from_spec(channels)


def test_norm8bit_scales_known_intensity_range():
    image = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    expected = np.array([[0, 85], [170, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(norm8bit(image), expected)


def test_norm8bit_constant_image_returns_black():
    image = np.full((2, 3), 7, dtype=np.uint16)

    with np.errstate(divide="raise", invalid="raise"):
        result = norm8bit(image)

    np.testing.assert_array_equal(result, np.zeros(image.shape, dtype=np.uint8))


def test_rgb_from_spec_applies_8bit_normalization():
    channels = np.array([[[10]], [[20]], [[30]]])
    expected = np.array([[[255, 127, 0]]], dtype=np.uint8)

    np.testing.assert_array_equal(RGB_from_spec(channels, do_norm=True), expected)
