"""Centred PCA of signed reconstruction residuals, without plotting or file I/O."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ResidualPCAResult:
    """Spectral directions and spatial amplitudes of unexplained variation.

    ``mean_residual`` is a channel vector retained separately from centred PCA.
    ``loadings`` is (components, channels) with orthonormal rows. ``scores`` has
    the original spatial shape plus a final component axis, or is None when
    return_scores=False. Scores have signed detector-intensity units, not
    concentrations. ``explained_variance`` contains selected covariance
    eigenvalues (sample normalization, n_samples-1); ``eigenvalues`` includes
    all channel directions. Ratios use total centred variance, not residual RMS
    or original-image variance. ``covariance`` is (channels, channels).

    Signs are fixed by making each loading's largest absolute entry positive.
    Signs remain a convention, and tied eigenvalues can rotate their bases.
    Independent results need not have corresponding PC numbers. Arrays are
    not made immutable by the frozen result container.
    """

    mean_residual: np.ndarray
    loadings: np.ndarray
    scores: Optional[np.ndarray]
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    eigenvalues: np.ndarray
    covariance: np.ndarray
    spatial_shape: tuple
    n_samples: int


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _spectral_pixels(data, name, spectral_axis):
    data = np.asarray(data)
    if (data.ndim < 2 or not data.size or np.iscomplexobj(data)
            or not np.issubdtype(data.dtype, np.number)):
        raise ValueError(f"{name} must be a nonempty real numeric array with at least two axes")
    if (isinstance(spectral_axis, (bool, np.bool_))
            or not isinstance(spectral_axis, (int, np.integer))
            or not -data.ndim <= spectral_axis < data.ndim):
        raise ValueError("spectral_axis must identify an array axis")
    moved = np.moveaxis(data, spectral_axis, -1)
    return moved.reshape(-1, moved.shape[-1]), moved.shape[:-1]


def _fit_pca(chunk, shape, channels, n_components, batch_size, return_scores):
    """Merge centred batch moments to avoid subtracting huge raw moments."""
    samples = int(np.prod(shape))
    n_components = _integer(n_components, "n_components")
    batch_size = _integer(batch_size, "batch_size")
    if samples < 2 or n_components > min(channels, samples-1):
        raise ValueError("PCA requires at least two pixels and n_components <= min(channels, pixels-1)")
    if not isinstance(return_scores, (bool, np.bool_)):
        raise ValueError("return_scores must be boolean")
    mean = np.zeros(channels)
    scatter = np.zeros((channels, channels))
    seen = 0
    for first in range(0, samples, batch_size):
        values = chunk(first, min(first+batch_size, samples))
        if not np.isfinite(values).all():
            raise ValueError("residual inputs and computed residuals must be finite")
        count = len(values)
        batch_mean = values.mean(axis=0)
        centred = values-batch_mean
        delta = batch_mean-mean
        combined = seen+count
        scatter += centred.T @ centred + np.outer(delta, delta)*(seen*count/combined)
        mean += delta*(count/combined)
        seen = combined
    covariance = scatter/(samples-1)
    covariance = (covariance+covariance.T)/2
    if not np.isfinite(covariance).all():
        raise ValueError("residual covariance overflowed; rescale input intensities")
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    loadings = vectors[:, order[:n_components]].T.copy()
    for direction in loadings:
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
    selected = values[:n_components].copy()
    ratios = selected/values.sum() if values.sum() else np.zeros(n_components)
    scores = None
    if return_scores:
        flat = np.empty((samples, n_components))
        for first in range(0, samples, batch_size):
            last = min(first+batch_size, samples)
            flat[first:last] = (chunk(first, last)-mean) @ loadings.T
        scores = flat.reshape(*shape, n_components)
    return ResidualPCAResult(mean, loadings, scores, selected, ratios, values,
                             covariance, shape, samples)


def residual_pca(observed, reconstructed, *, n_components=6, spectral_axis=-1,
                 batch_size=65536, return_scores=True):
    """Reveal spectral and spatial patterns left unexplained by a reconstruction.

    Purpose
    -------
    Use on real images or simulations after fitting any unmixing model. Pair
    spectral loadings with spatial score maps to inspect coherent unexplained
    patterns and guide experimental pixel selection. Ground truth is not needed.
    The calculation performs no fitting, plotting, saving or component-count
    decision, and does not modify either input.

    When to use
    -----------
    Use this function when both the measured image and reconstructed image
    already exist. If signed residuals have already been calculated, use
    residual_pca_from_residuals instead. If only the measured image and fitted
    C/S factors exist, use residual_pca_from_factors to avoid allocating full
    reconstructed and residual images. All three perform the same centred PCA.

    Parameters
    ----------
    observed, reconstructed : array-like, identical shape (..., channels)
        Real finite data in identical intensity units. Signed inputs are allowed.
        Default spectral_axis=-1 matches reconstruction_error_maps. For CZI-style
        (channels, y, x), explicitly set spectral_axis=0. Flattened (pixels,
        channels) matrices also work; scores then have (pixels, components) shape.
    n_components : int, default 6
        Number of displayed/returned directions, between 1 and
        min(channels, pixels-1). It is not the asserted number of missing species.
    spectral_axis : int, default -1
        Channel axis; all other axes keep their order in the output score maps.
    batch_size : int, default 65536
        Pixels per covariance/projection chunk. Subtract in float64, retaining
        negative residuals even when the measured image uses unsigned integers.
    return_scores : bool, default True
        False returns covariance/loadings/statistics without allocating full
        score maps. True requires approximately pixels*n_components*8 bytes for
        scores and a second input pass. Input layout conversion may also copy.

    Returns
    -------
    ResidualPCAResult
        PCA of signed residuals observed-reconstructed, centred across pixels
        per channel. Mean residual bias is returned separately. Variances use
        n_pixels-1; ratios use total centred residual variance. Constant/zero
        residuals return zero variances, ratios and scores with an arbitrary
        orthonormal basis, not meaningful directions.

    Interpretation and limits
    -------------------------
    No brightness normalization, noise whitening or significance test is applied.
    Brightness-dependent noise, fitting-induced correlations, detector effects,
    inaccurate known spectra and omitted components can all create patterns.
    Loadings and scores are signed; they are not pure nonnegative signatures,
    abundances, purity scores or an automatic fluorophore count. A missing
    spectrum may be partly absorbed by the fitted factors. A visually flat
    residual does not prove correct recovery. Centred PCA removes constant
    spectral bias: inspect mean_residual alongside score maps and RMS.

    Directions from independently fitted PCA models can reorder/rotate. Use a
    fixed mean/basis for direct before/after projections, and preserve wavelength
    definitions. Tied/near-tied eigenvalues make individual directions unstable.
    For precomputed residuals use residual_pca_from_residuals; for large factor
    models use residual_pca_from_factors to avoid building a reconstruction cube.

    Examples
    --------
    >>> image = np.arange(24.).reshape(2, 3, 4)
    >>> result = residual_pca(image, np.zeros_like(image), n_components=2)
    >>> result.scores.shape
    (2, 3, 2)
    """
    a, shape = _spectral_pixels(observed, "observed", spectral_axis)
    b, other_shape = _spectral_pixels(reconstructed, "reconstructed", spectral_axis)
    if np.shape(observed) != np.shape(reconstructed) or shape != other_shape:
        raise ValueError("observed and reconstructed must have identical shapes")
    def chunk(first, last):
        return a[first:last].astype(float)-b[first:last].astype(float)
    return _fit_pca(chunk, shape, a.shape[1], n_components, batch_size, return_scores)


def residual_pca_from_residuals(residuals, *, n_components=6, spectral_axis=-1,
                               batch_size=65536, return_scores=True):
    """Analyse precomputed SIGNED residuals; see residual_pca for full semantics.

    Use when observed-minus-reconstructed residuals are already available.
    This avoids reconstructing or subtracting the images again. Use residual_pca
    when you instead have measured and reconstructed images, or
    residual_pca_from_factors when you have measured data and fitted C/S factors
    but have not allocated the reconstruction or residual cube.
    Do not pass absolute errors, squared errors or RMS maps: they have lost
    the spectral signs/shape this diagnostic needs. Layout, centring, result,
    memory options and interpretation limits match residual_pca.
    """
    pixels, shape = _spectral_pixels(residuals, "residuals", spectral_axis)
    return _fit_pca(lambda first,last: pixels[first:last].astype(float), shape,
                    pixels.shape[1], n_components, batch_size, return_scores)


def residual_pca_from_factors(observed, abundances, spectra, *, n_components=6,
                             spectral_axis=-1, batch_size=65536, return_scores=True):
    """Analyse observed-C@S without allocating a full reconstruction/residual cube.

    When to use
    -----------
    Use after unmixing when you have the measured image, abundance factors C
    and spectral factors S, but have not created the reconstructed image C@S
    or signed residual image observed-C@S. This is especially useful for large
    spectral images: products and subtraction are calculated in small pixel
    batches, avoiding those two full-size intermediate arrays. Scores still
    require memory unless return_scores=False.

    If signed residuals already exist, use residual_pca_from_residuals. If both
    measured and reconstructed images already exist, use residual_pca. The
    factor version performs the same PCA; its purpose is to save intermediate
    memory, not to apply a different statistical model.

    Inputs and calculation
    ----------------------

    observed uses the same layouts as residual_pca. spectra is (components,
    channels), while abundances is (pixels, components) or the observed spatial
    shape followed by a component axis. Inputs must be finite and real; signed
    factors are allowed for diagnostics beyond nonnegative unmixing. Their
    scaling is used as supplied, with no normalization or component matching.
    Chunked float64 products/subtractions feed the same PCA calculation. See
    residual_pca for all result definitions, settings and limitations.
    """
    pixels, shape = _spectral_pixels(observed, "observed", spectral_axis)
    c, s = np.asarray(abundances), np.asarray(spectra)
    for data in (c, s):
        if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError("abundances and spectra must be real finite numeric arrays")
    if s.ndim != 2 or not s.size or s.shape[1] != pixels.shape[1]:
        raise ValueError("spectra must be (components, observed channels)")
    if c.shape not in ((len(pixels), len(s)), (*shape, len(s))):
        raise ValueError("abundances must match the observed pixels/spatial shape and spectra components")
    c = c.reshape(-1, len(s))
    s = s.astype(float)
    def chunk(first, last):
        return pixels[first:last].astype(float)-c[first:last].astype(float) @ s
    return _fit_pca(chunk, shape, pixels.shape[1], n_components, batch_size, return_scores)
