"""Exact linear spectral image formation."""

import numpy as np

from .result import SyntheticDataset


def simulate_linear_mixture(abundances, spectra, wavelengths, fluorophore_names=None):
    """Form a noiseless ``(y, x, wavelength)`` image as ``abundances @ spectra``."""
    abundances = np.asarray(abundances, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    if abundances.ndim != 3 or any(size == 0 for size in abundances.shape):
        raise ValueError("abundances must have non-empty shape (y, x, component)")
    if spectra.ndim != 2 or spectra.shape[0] != abundances.shape[2]:
        raise ValueError("spectra must have shape (component, wavelength)")
    if wavelengths.ndim != 1 or wavelengths.size != spectra.shape[1]:
        raise ValueError("wavelengths must have one value per spectral channel")
    if not (np.isfinite(abundances).all() and np.isfinite(spectra).all()
            and np.isfinite(wavelengths).all()):
        raise ValueError("simulation inputs must be finite")
    if np.any(abundances < 0) or np.any(spectra < 0):
        raise ValueError("abundances and spectra must be non-negative")
    if fluorophore_names is None:
        fluorophore_names = tuple(f"component_{i}" for i in range(spectra.shape[0]))
    else:
        fluorophore_names = tuple(fluorophore_names)
        if len(fluorophore_names) != spectra.shape[0]:
            raise ValueError("provide one fluorophore name per spectrum")

    image = np.einsum("yxc,ck->yxk", abundances, spectra)
    return SyntheticDataset(
        image=image,
        abundances=abundances.copy(),
        spectra=spectra.copy(),
        wavelengths=wavelengths.copy(),
        fluorophore_names=fluorophore_names,
    )
