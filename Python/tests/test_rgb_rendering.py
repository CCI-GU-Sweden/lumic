"""Display mapping, shared scaling, labeled axes and source preservation."""
from itertools import permutations

import numpy as np
import pytest

from lumic import RGBRendering, SpectralImage
from lumic.utils import RGB_from_spec


def test_group_means_keep_float_signal_and_rgb_order():
    values = np.array([.1, .2, .3, .4, .5, .6])[:, None, None]
    image = SpectralImage(values, "CYX")
    rendering = image.to_rgb(limits=(0, 1))
    assert isinstance(rendering, RGBRendering)
    assert rendering.axes == "YXS"
    assert rendering.channel_groups == ((3, 4, 5), (2,), (0, 1))
    # R=.5, G=.3, B=.15; no truncation of float spectral values before scaling.
    np.testing.assert_array_equal(rendering.data, [[[128, 76, 38]]])
    np.testing.assert_array_equal(image.data, values)
    assert rendering.data.dtype == np.uint8


def test_all_axis_orders_preserve_stack_coordinates():
    axes = "TZYXC"
    canonical = np.arange(2 * 2 * 2 * 3 * 5).reshape(2, 2, 2, 3, 5)
    # Independent five-channel group oracle: blue 0, green 1/2, red 3/4.
    rgb = np.stack([(canonical[..., 3] + canonical[..., 4]) / 2,
                    (canonical[..., 1] + canonical[..., 2]) / 2,
                    canonical[..., 0]], axis=-1)
    for ordering in permutations(axes):
        source_axes = "".join(ordering)
        image = SpectralImage(canonical.transpose([axes.index(a) for a in ordering]), source_axes)
        rendering = image.to_rgb(limits=(0, 255))
        output_axes = source_axes.replace("C", "") + "S"
        expected = np.rint(rgb).astype(np.uint8).transpose(["TZYXS".index(a) for a in output_axes])
        assert rendering.axes == output_axes
        np.testing.assert_array_equal(rendering.data, expected)


def test_shared_scale_keeps_color_balance_and_relative_plane_brightness():
    # Two planes with identical color balance and a factor-two brightness change.
    values = np.array([[1., 2., 4.], [2., 4., 8.]]).reshape(2, 1, 1, 3)
    rendering = SpectralImage(values, "ZYXC").to_rgb()
    assert rendering.limits == (0., 8.)
    np.testing.assert_array_equal(rendering.data[:, 0, 0], [[128, 64, 32], [255, 128, 64]])
    # Rendering a second image with explicit reused limits keeps the comparison.
    darker = SpectralImage(values[:1], "ZYXC").to_rgb(limits=rendering.limits)
    np.testing.assert_array_equal(darker.data, rendering.data[:1])


def test_black_white_clipping_singletons_and_metadata():
    image = SpectralImage(np.array([[[-2.]], [[5.]], [[20.]]]), "CYX",
                          pixel_sizes_um={"X": .1, "Y": .2}, provenance={"path": "source.czi"})
    rendering = image.to_rgb(limits=(0, 10))
    np.testing.assert_array_equal(rendering.data, [[[255, 128, 0]]])
    assert rendering.pixel_sizes_um == image.pixel_sizes_um
    assert rendering.provenance["source"] == image.provenance
    assert rendering.provenance["source_axes"] == "CYX"
    assert rendering.provenance["source_kind"] == "measured"
    assert not hasattr(rendering, "unfold")
    assert not hasattr(rendering, "wavelengths_nm")


@pytest.mark.parametrize("value,expected", [(0., 0), (-1., 0), (5., 255)])
def test_uniform_images_have_defined_display_values(value, expected):
    image = SpectralImage(np.full((3, 1, 1), value), "CYX")
    result = image.to_rgb()
    np.testing.assert_array_equal(result.data, np.full((1, 1, 3), expected, dtype=np.uint8))
    np.testing.assert_array_equal(image.to_rgb(limits=result.limits).data, result.data)


@pytest.mark.parametrize("n_channels", [1, 2])
def test_too_few_channels(n_channels):
    with pytest.raises(ValueError, match="at least three"):
        SpectralImage(np.zeros((n_channels, 2, 3)), "CYX").to_rgb()


@pytest.mark.parametrize("limits", [(0, 0), (1, 0), (0, np.inf), (np.nan, 1), (0,), (0, 1, 2)])
def test_invalid_limits(limits):
    with pytest.raises(ValueError, match="limits"):
        SpectralImage(np.ones((3, 2, 3)), "CYX").to_rgb(limits=limits)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_pixels_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        SpectralImage(np.full((3, 2, 3), value), "CYX").to_rgb()


def test_reconstructed_images_supported_abundances_rejected():
    image = SpectralImage(np.ones((3, 2, 3)), "CYX", kind="reconstructed")
    assert image.to_rgb().provenance["source_kind"] == "reconstructed"
    maps = SpectralImage(np.ones((3, 2, 3)), "KYX", kind="abundance")
    with pytest.raises(ValueError, match="not abundance"):
        maps.to_rgb()


def test_legacy_values_retained_with_explicit_cyx_notice(capsys):
    values = np.array([10, 20, 30, 40, 50, 60])[:, None, None]
    np.testing.assert_array_equal(RGB_from_spec(values), [[[50, 30, 15]]])
    assert "assumes CYX" in capsys.readouterr().out
    np.testing.assert_array_equal(RGB_from_spec(values, do_norm=True), [[[255, 109, 0]]])


def test_rendering_validates_axes_and_uint8_storage():
    kwargs = dict(channel_groups=((2,), (1,), (0,)), limits=(0, 1))
    with pytest.raises(TypeError, match="uint8"):
        RGBRendering(np.zeros((2, 3, 3)), "YXS", **kwargs)
    with pytest.raises(ValueError, match="axes"):
        RGBRendering(np.zeros((2, 3, 3), dtype=np.uint8), "CYX", **kwargs)
