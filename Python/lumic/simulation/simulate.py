"""Assemble detector sampling, linear mixing, and simple count noise."""

import numpy as np

from .detector import SpectralDetector
from .mixing import simulate_linear_mixture
from .noise import add_confocal_noise
from .result import NoisySimulationResult


def simulate_noisy_image(
    component_counts,
    component_spectra,
    detector,
    *,
    fluorophore_names=None,
    gaussian_sigma=0.0,
    rng=None,
):
    """Simulate recorded counts from per-pixel fluorophore contributions.

    ``component_counts[y, x, k]`` is the expected number of recorded counts
    supplied by component ``k`` at that pixel, across all detector channels.
    ``component_spectra`` holds one ``(wavelengths, intensities)`` pair per
    component. Each spectrum is integrated by ``detector`` and normalized to
    unit sum, so the clean image sums to ``component_counts.sum(axis=-1)``.

    The result retains that clean image and an independently noisy image.
    Spectra with no signal in the detector's wavelength range are rejected.
    """
    if not isinstance(detector, SpectralDetector):
        raise TypeError("detector must be a SpectralDetector")
    component_counts = np.asarray(component_counts, dtype=float)
    if component_counts.ndim != 3 or any(size == 0 for size in component_counts.shape):
        raise ValueError("component_counts must have non-empty shape (y, x, component)")
    component_spectra = tuple(component_spectra)
    if len(component_spectra) != component_counts.shape[-1]:
        raise ValueError("provide one component spectrum per abundance channel")

    sampled_spectra = []
    source_spectra = []
    for curve in component_spectra:
        try:
            wavelengths, intensities = curve
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each component spectrum must be a (wavelengths, intensities) pair"
            ) from exc
        sampled_spectra.append(detector.sample(wavelengths, intensities))
        source_spectra.append(
            (np.asarray(wavelengths, dtype=float).copy(),
             np.asarray(intensities, dtype=float).copy())
        )

    sampled_spectra = np.asarray(sampled_spectra, dtype=float)
    totals = sampled_spectra.sum(axis=1)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("each component must have signal in the detector range")
    normalized_spectra = sampled_spectra / totals[:, np.newaxis]

    ideal = simulate_linear_mixture(
        component_counts,
        normalized_spectra,
        detector.channel_centers,
        fluorophore_names=fluorophore_names,
    )
    noisy_image = add_confocal_noise(
        ideal.image, gaussian_sigma=gaussian_sigma, rng=rng
    )
    return NoisySimulationResult(
        ideal=ideal,
        noisy_image=noisy_image,
        detector=detector,
        source_spectra=tuple(source_spectra),
        sampled_spectra=sampled_spectra,
        gaussian_sigma=float(gaussian_sigma),
    )
