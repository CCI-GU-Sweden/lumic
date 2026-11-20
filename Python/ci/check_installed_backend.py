"""Smoke test for the default Lumic wheel in an isolated virtual environment."""

from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

import lumic
from lumic.io import load_czi, load_image
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum


def main():
    package_path = Path(lumic.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    optional_modules = ("napari", "qtpy", "PyQt5", "PyQt6", "PySide2", "PySide6", "matplotlib")
    unexpected = [name for name in optional_modules if find_spec(name) is not None]
    assert not unexpected, f"Unexpected modules in the default install: {unexpected}"

    # Exercise the native CZI dependency using a repository fixture.
    fixture = (Path(__file__).resolve().parents[1] / "data" / "io-data" / "980"
               / "ex405-range411_694-res35p4nm.czi")
    czi = load_czi(fixture)
    assert czi.axes == "CYX" and czi.data.shape[0] == 8

    notice = files("lumic.simulation").joinpath("fpbase_data", "README.md")
    assert "FPbase" in notice.read_text(encoding="utf-8")
    detector = SpectralDetector.from_profile("980_32ch")
    spectra = np.array([
        detector.sample(*load_offline_emission_spectrum(name))
        for name in ("EBFP2", "EGFP", "mCherry", "mNeptune")
    ])
    spectra /= spectra.sum(axis=1, keepdims=True)

    coefficients = np.arange(24, dtype=float).reshape(6, 4)
    pixels = coefficients @ spectra
    image = lumic.SpectralImage(
        pixels.reshape(2, 3, -1), "YXC", wavelengths_nm=detector.channel_centers,
    )
    unfolded = image.unfold()
    fitted = lumic.positive_least_squares(unfolded.data, spectra, solver="auto")
    np.testing.assert_allclose(fitted, coefficients, atol=1e-8)

    maps = unfolded.fold_abundances(fitted, spectra=spectra)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "abundances.ome.tif"
        maps.save_ome_tiff(path)
        loaded = load_image(path)
        assert loaded.kind == "abundance" and loaded.axes == "YXK"
        np.testing.assert_array_equal(loaded.data, maps.data)
        np.testing.assert_allclose(loaded.reconstruct().data, image.data, atol=1e-8)

    print(f"Lumic {version('lumic')} passed CZI reading, offline unmixing and OME-TIFF export.")


if __name__ == "__main__":
    main()
