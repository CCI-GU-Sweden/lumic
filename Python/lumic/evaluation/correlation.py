"""Descriptive spatial and spectral correlations of signed residuals."""

import numpy as np

from .residuals import _integer, _spectral_pixels


def _pixels(residuals, spectral_axis, batch_size):
    pixels, shape = _spectral_pixels(residuals, "residuals", spectral_axis)
    batch_size = _integer(batch_size, "batch_size")
    for first in range(0, len(pixels), batch_size):
        if not np.isfinite(pixels[first:first + batch_size]).all():
            raise ValueError("residuals must contain only finite values")
    return pixels, shape, batch_size


def residual_neighbor_correlation(residuals, *, spectral_axis=-1, batch_size=65536):
    """Measure spatial structure in signed residuals from real or simulated images.

    Parameters
    ----------
    residuals : array, three axes
        Observed minus reconstructed intensities, with two spatial axes and
        one spectral axis. Values must be finite; negative values are expected.
    spectral_axis : int, default -1
        Channel axis; use 0 for a (channels, y, x) image.
    batch_size : int, default 65536
        Maximum neighbor pairs processed together. No full pair array is built.

    Returns
    -------
    float
        Correlation in [-1, 1], pooling horizontal and vertical adjacent
        pairs, without diagonals or wrapping. Each endpoint population is
        centered separately per channel. The centered cross-product is divided
        by the product of endpoint norms, so channels with larger residual
        variance carry more weight. This is not a correlation of RMS maps.
        Returns NaN if there are no pairs or either endpoint has zero variance.

    Notes
    -----
    Positive values indicate neighboring pixels share signed spectral errors;
    negative values indicate alternation. Use alongside residual magnitude,
    mean bias and PCA maps to compare spatial structure across fits. Ground
    truth is unnecessary, but this is not a test for a missing fluorophore or
    statistical significance: PSF, detector noise and fitting can also induce
    correlations. Centering hides constant channel offsets. Near-zero residuals
    can have large correlations; callers choose a meaningful magnitude floor.
    Inputs are not modified. Moving a noncontiguous channel axis may copy data.

    Examples
    --------
    >>> checker = 1 - 2 * (np.indices((4, 4)).sum(axis=0) % 2)
    >>> residual_neighbor_correlation(checker[..., None])
    -1.0
    """
    pixels, shape, batch_size = _pixels(residuals, spectral_axis, batch_size)
    if len(shape) != 2:
        raise ValueError("residuals must have two spatial axes and one spectral axis")
    height, width = shape

    def pairs():
        # Generate endpoints directly in image order; row boundaries never wrap.
        for count, horizontal in ((height * (width - 1), True),
                                  ((height - 1) * width, False)):
            for first in range(0, count, batch_size):
                index = np.arange(first, min(first + batch_size, count))
                left = index // (width - 1) * width + index % (width - 1) if horizontal else index
                right = left + (1 if horizontal else width)
                yield pixels[left].astype(float), pixels[right].astype(float)

    count = height * (width - 1) + (height - 1) * width
    if not count:
        return float("nan")
    # Two passes avoid cancellation from subtracting squared raw moments.
    left_mean, right_mean = np.zeros(pixels.shape[1]), np.zeros(pixels.shape[1])
    seen = 0
    for left, right in pairs():
        if not seen:
            left_origin, right_origin = left[0].copy(), right[0].copy()
        # Shift first so decimal-valued constant channels remain exactly zero.
        left -= left_origin
        right -= right_origin
        seen += len(left)
        left_mean += (left.mean(axis=0) - left_mean) * (len(left) / seen)
        right_mean += (right.mean(axis=0) - right_mean) * (len(right) / seen)
    cross = left_square = right_square = 0.
    for left, right in pairs():
        left -= left_origin
        right -= right_origin
        left -= left_mean
        right -= right_mean
        cross += np.sum(left * right)
        left_square += np.sum(left * left)
        right_square += np.sum(right * right)
    denominator = np.sqrt(left_square) * np.sqrt(right_square)
    return float(np.clip(cross / denominator, -1, 1)) if denominator else float("nan")


def residual_spectral_correlation(residuals, *, spectral_axis=-1, batch_size=65536):
    """Map within-pixel structure across consecutive residual channels.

    Accept finite signed observed-minus-reconstructed residuals with at least
    two axes. ``spectral_axis`` defaults to the last axis; use 0 for CZI images.
    Return an array with that axis removed, containing each pixel's Pearson
    correlation between residual[:-1] and residual[1:]. Each vector is centered
    separately. Values lie in [-1, 1]; undefined correlations are NaN, including
    constant endpoint vectors or fewer than three channels. ``batch_size`` is
    a positive integer limiting temporary pixel arrays (default 65536).

    This works on real and simulated data without ground truth. Positive
    values highlight spectrally smooth errors; negative values highlight
    alternating errors. Inspect maps alongside signed mean residuals, RMS and
    residual PCA to locate potentially unexplained spectral structure. It does
    not identify a species, establish correct factors or supply a significance
    threshold. Short spectra give noisy estimates; wavelength gaps/order,
    detector processing and fitting-induced noise correlations affect values.
    Constant spectral offsets are removed by centering, and this scale-free
    diagnostic cannot distinguish tiny errors from large ones. No wavelength
    interpolation, noise correction or magnitude floor is applied.
    Inputs are not modified; moving a noncontiguous channel axis may copy data.

    Examples
    --------
    >>> residual_spectral_correlation([[0., 1., 2., 3.]])
    array([1.])
    """
    pixels, shape, batch_size = _pixels(residuals, spectral_axis, batch_size)
    result = np.full(len(pixels), np.nan)
    if pixels.shape[1] < 3:
        return result.reshape(shape)
    for first in range(0, len(pixels), batch_size):
        chunk = pixels[first:first + batch_size].astype(float)
        left, right = chunk[:, :-1].copy(), chunk[:, 1:].copy()
        left -= left[:, :1].copy()
        right -= right[:, :1].copy()
        left -= left.mean(axis=1, keepdims=True)
        right -= right.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=1)) * np.sqrt(np.sum(right * right, axis=1))
        values = np.divide(np.sum(left * right, axis=1), denominator,
                           out=np.full(len(chunk), np.nan), where=denominator > 0)
        result[first:first + len(chunk)] = np.clip(values, -1, 1)
    return result.reshape(shape)
