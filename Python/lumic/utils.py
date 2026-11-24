from dataclasses import dataclass
import warnings

import numpy as np


def _validate_median_quantile(quantile):
    """One validation rule for the legacy and traceable selection interfaces."""
    try:
        valid = (not isinstance(quantile, (bool, np.bool_)) and np.isscalar(quantile)
                 and np.isfinite(quantile) and 0 <= quantile <= 1)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("median quantile must lie in [0, 1]")


def _median_intensity_mask(pixels, quantile):
    """Shared median-percentile rule; retain values equal to the threshold."""
    _validate_median_quantile(quantile)
    medians = np.median(pixels, axis=1)
    threshold = float(np.quantile(medians, quantile))
    # Preserve the legacy 'remove strictly below' behavior, even for NaNs.
    # The new selection interface excludes nonfinite inputs before this step.
    return ~(medians < threshold), threshold


@dataclass(frozen=True)
class AnalysisPixelSelection:
    """Traceable selection in the original image's spatial coordinates.

    ``mask`` has the spatial image shape and ``indices`` indexes its C-order
    flattening, equivalent to ``np.flatnonzero(mask)``. Move the spectral axis
    last before flattening the measured spectra. ``exclusion_masks`` records
    disjoint reasons: nonfinite input, negative intensity, zero signal, and
    below-median threshold. These labels describe selection rules, not sample
    identities or noise significance. Frozen fields do not make arrays immutable.
    """

    indices: np.ndarray
    mask: np.ndarray
    image_shape: tuple
    spectral_axis: int
    median_quantile: object
    median_threshold: object
    exclusion_masks: dict

    @property
    def exclusion_counts(self):
        """Number of pixels excluded for each disjoint selection reason."""
        return {name: int(mask.sum()) for name, mask in self.exclusion_masks.items()}


def select_analysis_pixels(image, *, median_quantile=None, spectral_axis=0):
    """Select candidate pixels for analysis without changing the measured image.

    Purpose
    -------
    Use before convex-hull selection or other fitting-pool construction when
    original pixel locations must survive filtering. The result includes mask,
    original indices, threshold/settings and exclusion reasons. A faint pixel
    can be valid yet excluded from this analysis pool; selection does not erase
    it from the image or establish that it contains only noise.

    Parameters
    ----------
    image : array-like, 3D
        Real numeric spectral image. Default layout is (channels, y, x);
        spectral_axis=-1 supports simulation images (y, x, channels).
    median_quantile : float in [0, 1] or None, default None
        Optional percentile threshold on each pixel's median across channels.
        Calculate the threshold from finite, nonnegative pixels, including
        zeros, to retain the legacy median rule on valid nonnegative images.
        Exclude values strictly below it; ties remain. None disables this rule.
        Zero retains all otherwise usable pixels; one retains the maximum ties.
    spectral_axis : int, default 0
        Axis containing spectral channels; other axes retain their order.

    Returns
    -------
    AnalysisPixelSelection
        Original spatial mask/indices and disjoint exclusion reasons. Pixels
        with any nonfinite or negative value, or all-zero spectra, are excluded
        regardless of the optional median filter. If none are usable, return
        empty indices and an all-false mask; a hull/solver may require more data.

    Limitations
    -----------
    Excluding negative intensities assumes nonnegative unmixing, and may be
    inappropriate for signed offset-corrected measurements. This function does
    not repair invalid data, estimate noise, detect saturation, or normalize
    spectra. Full-image unmixing still needs its own input validation. Future
    rules can extend this selection interface without combining it with hull
    calculation. Low-intensity exclusion can discard meaningful minor species.

    Examples
    --------
    >>> image = np.array([[[0, 1, 4]], [[0, 3, 6]]])
    >>> retained = select_analysis_pixels(image, median_quantile=0.5)
    >>> retained.indices.tolist()
    [1, 2]
    >>> retained.median_threshold
    2.0

    Use ``unfolded.data[retained.indices]`` when ``unfolded`` was created from
    the same image with ``SpectralImage.unfold()``. The result stores the
    selection without copying the selected spectra.
    """
    data = np.asarray(image)
    if (data.ndim != 3 or any(size == 0 for size in data.shape)
            or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)):
        raise ValueError("image must be a nonempty real numeric 3D spectral image")
    if (isinstance(spectral_axis, (bool, np.bool_))
            or not isinstance(spectral_axis, (int, np.integer))
            or not -3 <= spectral_axis < 3):
        raise ValueError("spectral_axis must identify an image axis")
    axis = int(spectral_axis) % 3
    spectra = np.moveaxis(data, axis, -1)
    shape = spectra.shape[:-1]
    pixels = spectra.reshape(-1, spectra.shape[-1])
    finite = np.isfinite(pixels).all(axis=1)
    negative = finite & np.any(pixels < 0, axis=1)
    valid = finite & ~negative
    zero = valid & ~np.any(pixels > 0, axis=1)
    below = np.zeros(len(pixels), dtype=bool)
    threshold = None
    if median_quantile is not None:
        # Validate even if invalid input leaves no rows for a percentile.
        _validate_median_quantile(median_quantile)
        if valid.any():
            keep, threshold = _median_intensity_mask(pixels[valid], median_quantile)
            below[valid] = ~keep
    # Zero signal is its own reason, even when also below the threshold.
    below &= ~zero
    mask = (valid & ~zero & ~below).reshape(shape)
    reasons = {"nonfinite": (~finite).reshape(shape), "negative": negative.reshape(shape),
               "zero_signal": zero.reshape(shape), "below_median": below.reshape(shape)}
    return AnalysisPixelSelection(np.flatnonzero(mask), mask, data.shape, axis,
                                  median_quantile, threshold, reasons)

def norm8bit(im):
    """ normalizes an image to min = 0 and max = 255 / 8bit format
    
    Parameters
    ----------
    im : numpyArray
        image to be normalized, works for single images 2D and volumes 3D

    Returns
    -------
    im: numpyArray
        normalized image in 8bit format. Constant images become all zeros.
    """
    im = im - im.min()
    im = im.astype(np.float32)
    max_intensity = im.max()
    if max_intensity == 0:
        return np.zeros_like(im, dtype=np.uint8)
    im = im / max_intensity
    im = im * 255
    im = im.astype(np.uint8)
    return im

def RGB_from_spec(specData, do_norm=False):
    """From a spectral dataset of arbitrary number of wavelength channels creates an RGB representation
    
    Parameters
    ----------
    specData : int numpyArray - 3D
        spectral data to be changed into RGB representation in order [C, Y, X]. Note that C must be >=3

    do_norm : bool
        if true then the data is normalizaed to 8bits

    Returns
    -------
    RGB : int numpy array RGB
        RGB image

    Notes
    -----
    Legacy helper: assumes CYX without inspecting axis labels. Use
    SpectralImage.to_rgb for explicit axes, stacks and float-safe display
    scaling. This function retains its original array return, integer channel
    averages and do_norm behavior, and prints a migration notice when called.
    Neither interface performs spectral unmixing or calibrated color rendering.
    """
    nSpec = specData.shape[0]
    if nSpec < 3:
        raise ValueError('RGB_from_spec requires at least three spectral channels')

    print(f'RGB_from_spec assumes CYX input ({nSpec} channels). '
          'Use SpectralImage.to_rgb() for explicit axes and stack support.')
    from .image import _rgb_channel_groups
    red, green, blue = _rgb_channel_groups(nSpec)
    
    B = specData[blue[0]:blue[-1]+1,:,:]
    B = np.mean(B, axis=0).astype(int)
    
    G = specData[green[0]:green[-1]+1,:,:]
    G = np.mean(G, axis=0).astype(int)
    
    R = specData[red[0]:,:,:]
    R = np.mean(R, axis=0).astype(int)
    
    RGB = np.dstack([R, G, B])

    if do_norm:
        RGB = norm8bit(RGB)
    
    return RGB

def remove_low_intensity(flattened_data, quantile_val, method='median', verbose = False):
    """Removes low intensity pixels based on method and the quantile value
    
    Parameters
    ----------
    flattened_data : numpyArray
        flattened data 2D matrix [pixel-idx, C]

    quantile_val : float
        pixels which alue are below the quantile are removed based on the method

    method : str
        which method to use to calculate the descriptor on which the quantile will be calculated
        for the moment only median

    verbose : bool
        in case you want to print a small explanation of what was done

    Returns
    -------
    subSet : numpyArray
        reduced dataset based on the descriptor (method) and quantile

    Migration
    ---------
    Deprecated convenience function: use select_analysis_pixels for an image
    selection with original indices, masks and reasons. This legacy function
    continues returning copied retained rows with its median-only behavior.
    The new function also excludes invalid and zero-signal pixels, so it is
    not an identical replacement on those inputs. No removal date is set.
    """
    warnings.warn(
        "remove_low_intensity is deprecated; use lumic.utils.select_analysis_pixels "
        "for traceable image selection with indices, masks and exclusion reasons.",
        FutureWarning, stacklevel=2,
    )
    if method == 'median':
        keep, _ = _median_intensity_mask(np.asarray(flattened_data), quantile_val)
        subSet = np.asarray(flattened_data)[keep].copy()
    else:
        raise Exception("Sorry, for the moment I have only implemented median")
    
    if verbose:
        print(f"The data was originally of size: {flattened_data.shape}")
        print(f"we remove low intensity pixels by using quantiles in the {method} intensity")
        print(f"after keeping {(1-quantile_val)*100}% the data was of size: {subSet.shape}")

    return subSet
    
