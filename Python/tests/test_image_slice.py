"""Explicit selection of loaded pixels with independent storage and metadata."""
from itertools import permutations

import numpy as np
import pytest

from lumic import SpectralImage


def test_selection_for_all_axis_orders_matches_known_coordinates():
    canonical_axes = "CTZYX"
    data = np.arange(3 * 2 * 3 * 2 * 4).reshape(3, 2, 3, 2, 4)
    for ordering in permutations(canonical_axes):
        axes = "".join(ordering)
        image = SpectralImage(data.transpose([canonical_axes.index(a) for a in axes]), axes)
        plane = image.slice(t=1, z=2)
        expected_axes = axes.replace("T", "").replace("Z", "")
        assert plane.axes == expected_axes
        expected = data[:, 1, 2].transpose(["CYX".index(a) for a in expected_axes])
        np.testing.assert_array_equal(plane.data, expected)
        assert not np.shares_memory(plane.data, image.data)


def test_preserves_kind_channel_metadata_and_independently_copies_pixels():
    image = SpectralImage(np.arange(3 * 2 * 2 * 4).reshape(3, 2, 2, 4), "CZYX",
                          wavelengths_nm=[450, 500, 600],
                          detection_bands_nm=[[440, 460], [490, 510], [590, 610]],
                          pixel_sizes_um={"X": .1, "Y": .1, "Z": .5},
                          provenance={"nested": {"value": 1}}, kind="reconstructed")
    selected = image.slice(z=1)
    assert selected.kind == "reconstructed"
    assert selected.pixel_sizes_um == image.pixel_sizes_um
    np.testing.assert_array_equal(selected.wavelengths_nm, image.wavelengths_nm)
    np.testing.assert_array_equal(selected.detection_bands_nm, image.detection_bands_nm)
    assert selected.provenance["image_slice"]["selection"] == {"Z": 1}
    selected.data[:] = -1
    selected.wavelengths_nm[0] = 400
    selected.pixel_sizes_um["X"] = 9
    selected.provenance["nested"]["value"] = 2
    assert image.data.min() == 0
    assert image.wavelengths_nm[0] == 450
    assert image.pixel_sizes_um["X"] == .1
    assert image.provenance["nested"]["value"] == 1


def test_defaults_absent_axes_and_complete_copy():
    image = SpectralImage(np.arange(3 * 2 * 3 * 2 * 4).reshape(3, 2, 3, 2, 4), "CTZYX")
    np.testing.assert_array_equal(image.slice().data, image.data[:, 0, 0])
    copied = image.slice(t=None, z=None)
    assert copied.axes == image.axes
    assert not np.shares_memory(copied.data, image.data)
    np.testing.assert_array_equal(copied.data, image.data)
    plain = image.slice()
    np.testing.assert_array_equal(plain.slice().data, plain.data)
    assert not np.shares_memory(plain.slice().data, plain.data)
    assert "parent_slice" in plain.slice().provenance["image_slice"]


def test_retained_singletons_and_abundance_names():
    image = SpectralImage(np.ones((2, 1, 3, 1, 4)), "KTZYX", kind="abundance",
                          component_names=["A", "B"])
    stack = image.slice(t=None, z=None)
    assert stack.axes == "KTZYX"
    plane = image.slice(z=2)
    assert plane.axes == "KYX"
    assert plane.kind == "abundance"
    assert plane.component_names == ("A", "B")
    assert plane.data.shape == (2, 1, 4)


@pytest.mark.parametrize("selection,error", [({"z": 3}, IndexError), ({"z": -1}, IndexError),
                                            ({"t": 2}, IndexError), ({"z": 1.5}, TypeError),
                                            ({"t": True}, TypeError), ({"t": np.bool_(False)}, TypeError)])
def test_invalid_selection(selection, error):
    image = SpectralImage(np.ones((3, 2, 3, 2, 4)), "CTZYX")
    with pytest.raises(error):
        image.slice(**selection)


def test_absent_axis_accepts_only_zero_or_none():
    image = SpectralImage(np.ones((3, 2, 4)), "CYX")
    assert image.slice().axes == "CYX"
    assert image.slice(t=None, z=None).axes == "CYX"
    for selection in ({"t": 1}, {"z": 1}):
        with pytest.raises(IndexError, match="only index 0"):
            image.slice(**selection)
