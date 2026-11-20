"""Simple count-domain noise for synthetic spectral images."""

import numpy as np


def add_confocal_noise(expected_counts, *, gaussian_sigma=0.0, rng=None):
    """Draw one noisy measurement for every pixel and spectral channel.

    ``expected_counts`` is the clean detector image in expected count units.
    Draw Poisson counts, add independent zero-mean Gaussian noise in the same
    units, then clip negative observed values to zero. The input is unchanged.
    Pass a seeded ``numpy.random.Generator`` for reproducible results.
    """
    expected_counts = np.asarray(expected_counts, dtype=float)
    if expected_counts.ndim != 3 or not np.isfinite(expected_counts).all():
        raise ValueError("expected_counts must be a finite (y, x, channel) image")
    if np.any(expected_counts < 0):
        raise ValueError("expected_counts must be non-negative")
    if not np.isfinite(gaussian_sigma) or gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative and finite")
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    noisy = rng.poisson(expected_counts).astype(float)
    if gaussian_sigma > 0:
        noisy += rng.normal(0.0, gaussian_sigma, size=expected_counts.shape)
    return np.maximum(noisy, 0.0)
