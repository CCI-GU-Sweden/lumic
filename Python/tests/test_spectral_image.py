import numpy as np
import pytest
from lumic import SpectralImage


def test_explicit_axes_and_pixels_retained_without_copy():
    data = np.arange(120).reshape(2, 3, 4, 5)
    image = SpectralImage(data, "TYXC", wavelengths_nm=np.arange(500, 505))
    assert image.data is data
    assert image.axes == "TYXC"
    assert image.data[1, 2, 3, 4] == 119


@pytest.mark.parametrize("axes", ["CY", "CCX", "cyx", "ZYX", "CYQ"])
def test_invalid_axes(axes):
    with pytest.raises(ValueError, match="axes"):
        SpectralImage(np.zeros((2, 3, 4)), axes)


@pytest.mark.parametrize("metadata", [
    {"wavelengths_nm": [500]}, {"wavelengths_nm": [500, np.nan]},
    {"detection_bands_nm": [[500, 490], [600, 610]]},
    {"pixel_sizes_um": {"X": 0}}, {"pixel_sizes_um": {"T": 1}},
])
def test_invalid_metadata(metadata):
    with pytest.raises(ValueError):
        SpectralImage(np.zeros((2, 3, 4)), "CYX", **metadata)


def test_overlapping_and_zero_width_windows():
    image = SpectralImage(np.zeros((2, 1, 1)), "CYX",
                          detection_bands_nm=[[400, 650], [650, 650]])
    assert image.wavelengths_nm is None
    assert image.data.shape == (2, 1, 1)
