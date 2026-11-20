import numpy as np
import pytest

from lumic import Spectrum


def test_spectrum_preserves_values_and_wavelength_order():
    values = np.array([0.25, -0.125, 0.5], dtype=np.float32)
    wavelengths = np.array([480., 460., 510.])
    provenance = {"format": "ZEN XML"}
    spectrum = Spectrum(values, wavelengths, name="Reference", provenance=provenance)
    values[:] = 0
    wavelengths[:] = 0
    provenance.clear()

    np.testing.assert_array_equal(spectrum.values, [0.25, -0.125, 0.5])
    np.testing.assert_array_equal(spectrum.wavelengths_nm, [480., 460., 510.])
    assert spectrum.values.dtype == np.float32
    assert spectrum.name == "Reference"
    assert spectrum.provenance == {"format": "ZEN XML"}


def test_spectrum_accepts_a_signature_without_metadata():
    spectrum = Spectrum([4, 2])

    np.testing.assert_array_equal(spectrum.values, [4, 2])
    assert spectrum.wavelengths_nm is None
    assert spectrum.name is None


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [1 + 2j]])
def test_spectrum_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="values"):
        Spectrum(values)


@pytest.mark.parametrize("wavelengths", [[480], [480, np.inf], [480, 0]])
def test_spectrum_rejects_invalid_wavelengths(wavelengths):
    with pytest.raises(ValueError, match="wavelengths_nm"):
        Spectrum([1, 2], wavelengths_nm=wavelengths)
