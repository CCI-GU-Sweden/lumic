"""Result containers for clean and noisy spectral image simulations."""

from dataclasses import dataclass

import numpy as np

from .detector import SpectralDetector


@dataclass(frozen=True)
class SyntheticDataset:
    """An ideal spectral image and the ground truth used to construct it.

    Image axes are ``(y, x, wavelength)`` and abundance axes are
    ``(y, x, fluorophore)``.
    """

    image: np.ndarray
    abundances: np.ndarray
    spectra: np.ndarray
    wavelengths: np.ndarray
    fluorophore_names: tuple


@dataclass(frozen=True)
class NoisySimulationResult:
    """Clean count image, noisy measurement, and detector ground truth.

    ``ideal.abundances`` contains expected per-component recorded counts and
    ``ideal.spectra`` contains the normalized detector spectra. The sampled
    spectra retain their original integrated intensity scale, and
    ``source_spectra`` retains the curves before detector sampling.
    """

    ideal: SyntheticDataset
    noisy_image: np.ndarray
    detector: SpectralDetector
    source_spectra: tuple
    sampled_spectra: np.ndarray
    gaussian_sigma: float

    @property
    def expected_counts(self):
        """Clean expected detector counts, with axes (y, x, channel)."""
        return self.ideal.image
