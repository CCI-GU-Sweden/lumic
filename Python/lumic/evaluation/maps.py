"""Observable reconstruction diagnostics and truth-based recovery maps."""

from dataclasses import dataclass

import numpy as np

from .metrics import _canonical_factors, align_factors


@dataclass(frozen=True)
class ReconstructionErrorMaps:
    """Channel residuals, pixel RMS, relative L2 error, and its validity mask."""

    residuals: np.ndarray
    rms: np.ndarray
    relative_l2: np.ndarray
    relative_valid: np.ndarray


@dataclass(frozen=True)
class ConcentrationErrorMaps:
    """Aligned concentration errors and estimated indices in true order."""

    signed: np.ndarray
    absolute: np.ndarray
    relative: np.ndarray
    relative_valid: np.ndarray
    component_order: np.ndarray


def _array(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim < 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty array with at least two axes")
    return array


def _threshold(value, name):
    if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(value)


def reconstruction_error_maps(observed, reconstructed, *, min_signal=0.0):
    """Describe unexplained signal in real or simulated spectral images.

    Parameters
    ----------
    observed, reconstructed : array-like, shape (..., channels)
        Measured image and fitted prediction in identical intensity units.
        Pixels may be flattened or spatial; channels are always the last axis.
        For flattened factors, the prediction is ``C @ S``. Signed finite data
        are allowed, including offset-corrected measurements.
    min_signal : float, default 0
        Minimum measured spectral L2 norm for reporting relative error, in
        input intensity units. Pixels at or below this threshold are masked.

    Returns
    -------
    ReconstructionErrorMaps
        ``residuals = observed - reconstructed`` retains the input shape.
        Positive values identify underestimated signal, negative values
        overestimated signal. Inspect channels for wavelength-specific errors
        and spatial patterns such as missing background or omitted components.
        ``rms = sqrt(mean(residuals**2, axis=-1))`` has the pixel shape and
        input intensity units; it locates the largest absolute mismatch.
        ``relative_l2 = norm(residuals) / norm(observed)`` is dimensionless
        and compares mismatch across brightness levels. It is NaN outside
        ``relative_valid``; RMS remains available even for zero-signal pixels.

    Limitations
    -----------
    No ground truth is required. These maps measure image fit, not correct
    identification of spectra or concentrations: different factors can yield
    the same image. They are not noise significance scores; bright regions
    can have larger RMS from shot noise alone, and relative errors at low
    signal need a scientifically appropriate threshold.

    Examples
    --------
    >>> maps = reconstruction_error_maps([[3, 4]], [[0, 4]])
    >>> maps.residuals.tolist()
    [[3.0, 0.0]]
    >>> maps.relative_l2.tolist()
    [0.6]
    """
    observed = _array(observed, "observed")
    reconstructed = _array(reconstructed, "reconstructed")
    if observed.shape != reconstructed.shape:
        raise ValueError("observed and reconstructed must have identical shapes")
    min_signal = _threshold(min_signal, "min_signal")
    residuals = observed - reconstructed
    signal = np.linalg.norm(observed, axis=-1)
    valid = signal > min_signal
    relative = np.divide(np.linalg.norm(residuals, axis=-1), signal,
                         out=np.full(signal.shape, np.nan), where=valid)
    return ReconstructionErrorMaps(residuals, np.sqrt(np.mean(residuals**2, axis=-1)),
                                   relative, valid)


def concentration_error_maps(true_abundances, true_spectra,
                             estimated_abundances, estimated_spectra, *,
                             min_concentration=0.0):
    """Compare recovered concentrations with known truth after factor alignment.

    Parameters
    ----------
    true_abundances, estimated_abundances : array-like, shape (..., components)
        Nonnegative concentration maps with identical shapes and components
        on the last axis. Flattened pixel matrices are also accepted.
    true_spectra, estimated_spectra : array-like, shape (components, channels)
        Paired nonnegative spectra. True spectra must be nonzero. Components
        are matched using the same unit-L2 spectral assignment as
        ``evaluate_factors``. Both sets of spectra are normalized to unit sum
        and their removed scales transferred to their paired abundances.
    min_concentration : float, default 0
        Report relative error only where canonical true concentration is
        strictly greater than this nonnegative threshold.

    Returns
    -------
    ConcentrationErrorMaps
        Maps retain the abundance shape in true-component order.
        ``signed = aligned_estimate - canonical_truth`` reveals overestimation
        (positive), underestimation (negative), and possible component leakage.
        ``absolute = abs(signed)`` locates recovery errors, including false
        positives where truth is zero. Both are in unit-sum-spectrum abundance
        units: total detector-channel counts when the image uses count units.
        ``relative = signed / canonical_truth`` is a dimensionless fractional
        error; multiply by 100 for percent. It is NaN outside
        ``relative_valid``. ``component_order`` records the estimated indices
        selected by matching.

    Limitations
    -----------
    Requires known ground-truth concentrations and spectra, normally from a
    simulation or independent validation. It cannot establish recovery accuracy
    on ordinary real images without those references. Matching compares the
    supplied components; it does not prove their identities or compensate for
    missing/extra components. Relative error is undefined for absent truth;
    inspect absolute errors there rather than treating masked pixels as zero.
    These maps are not uncertainty estimates or noise significance scores.

    Examples
    --------
    >>> maps = concentration_error_maps([[2, 0]], [[1, 0], [0, 1]],
    ...                                 [[3, 1]], [[1, 0], [0, 1]])
    >>> maps.signed.tolist()
    [[1.0, 1.0]]
    >>> maps.relative_valid.tolist()
    [[True, False]]
    """
    truth = _array(true_abundances, "true_abundances")
    estimate = _array(estimated_abundances, "estimated_abundances")
    if truth.shape != estimate.shape:
        raise ValueError("true and estimated abundances must have identical shapes")
    threshold = _threshold(min_concentration, "min_concentration")
    shape = truth.shape
    truth, true_spectra = _canonical_factors(truth.reshape(-1, shape[-1]), true_spectra)
    aligned = align_factors(true_spectra, estimate.reshape(-1, shape[-1]), estimated_spectra)
    order = aligned.component_order
    signed = (aligned.abundances - truth).reshape(shape)
    truth = truth.reshape(shape)
    valid = truth > threshold
    relative = np.divide(signed, truth, out=np.full(shape, np.nan), where=valid)
    return ConcentrationErrorMaps(signed, np.abs(signed), relative, valid, order)
