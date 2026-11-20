"""Helpers for putting spectra on a shared wavelength grid."""

from importlib.resources import files
from pathlib import Path

import numpy as np


def load_offline_emission_spectrum(name, *, directory=None):
    """Load a saved FPbase emission curve by fluorophore name.

    By default this reads the FPbase snapshot bundled with Lumic.
    Pass ``directory`` to use another collection of the same CSV files. This
    function has the same ``name -> (wavelengths, intensities)`` interface as
    ``lumic.fpbase.fetch_emission_spectrum``, so scene geometry does not depend
    on where its spectra come from.
    """
    if not isinstance(name, str) or not name or not name.isalnum():
        raise ValueError("name must be a non-empty fluorophore name")
    spectra_dir = (files(__package__).joinpath("fpbase_data")
                   if directory is None else Path(directory))
    path = spectra_dir / f"{name.lower()}_emission.csv"
    if not path.is_file():
        raise ValueError(f"no offline emission spectrum for {name!r} in {spectra_dir}")
    with path.open("r", encoding="utf-8") as stream:
        curve = np.loadtxt(stream, delimiter=",", skiprows=1)
    if (curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2 or
            not np.isfinite(curve).all() or np.any(np.diff(curve[:, 0]) <= 0) or
            np.any(curve[:, 1] < 0)):
        raise ValueError(f"invalid offline emission spectrum for {name!r}")
    return curve[:, 0], curve[:, 1]


def resample_spectrum(wavelengths, intensities, target_wavelengths):
    """Interpolate a spectrum onto a wavelength grid.

    Values outside the measured wavelength range are set to zero. Input
    intensities are preserved as supplied; this function does not normalize.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    target_wavelengths = np.asarray(target_wavelengths, dtype=float)

    if wavelengths.ndim != 1 or intensities.ndim != 1:
        raise ValueError("wavelengths and intensities must be one-dimensional")
    if wavelengths.size != intensities.size or wavelengths.size < 2:
        raise ValueError("wavelengths and intensities must have equal length >= 2")
    if target_wavelengths.ndim != 1:
        raise ValueError("target_wavelengths must be one-dimensional")
    if not (np.isfinite(wavelengths).all() and np.isfinite(intensities).all()
            and np.isfinite(target_wavelengths).all()):
        raise ValueError("spectrum and target wavelengths must be finite")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing")
    if np.any(intensities < 0):
        raise ValueError("intensities must be non-negative")

    return np.interp(
        target_wavelengths, wavelengths, intensities, left=0.0, right=0.0
    )
