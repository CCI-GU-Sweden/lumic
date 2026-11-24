"""Labeled spectral images, reversible matrices and display-only renderings."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from copy import deepcopy
from math import prod
from numbers import Integral

import numpy as np

def _validate_pixels(data):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array")
    if data.dtype.kind not in "uif":
        raise TypeError("data must contain real numeric pixels")


def _validate_layout(axes, shape, channel_axis):
    if (not isinstance(axes, str) or len(axes) != len(shape)
            or len(set(axes)) != len(axes)
            or not set(axes) <= set(channel_axis + "TZYX")
            or not set(channel_axis + "YX") <= set(axes)):
        raise ValueError(f"axes must match data dimensions and contain unique {channel_axis}, Y, X and optional T/Z")
    if any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
           for size in shape):
        raise ValueError("image dimensions must be nonempty positive integers")


def _copy_spatial_metadata(image):
    sizes = dict(image.pixel_sizes_um)
    if any(axis not in ("X", "Y", "Z") or not np.isfinite(size) or size <= 0
           for axis, size in sizes.items()):
        raise ValueError("pixel_sizes_um must map spatial axes to finite positive sizes")
    object.__setattr__(image, "pixel_sizes_um", sizes)
    object.__setattr__(image, "provenance", dict(image.provenance))


def _channel_metadata_array(value, name, shape):
    """Validate one optional detector-coordinate array and own its storage."""
    if value is None:
        return None
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must have shape {shape} and finite positive values")
    if name == "detection_bands_nm" and np.any(value[:, 1] < value[:, 0]):
        raise ValueError("detection band upper bounds must be >= lower bounds")
    return value


def _copy_channel_metadata(image, channels):
    for name, shape in (("wavelengths_nm", (channels,)),
                        ("detection_bands_nm", (channels, 2))):
        object.__setattr__(image, name, _channel_metadata_array(getattr(image, name), name, shape))


def _copy_unmixing_metadata(value, components):
    """Own a finite K×C basis and its C-channel metadata, without rescaling it."""
    if value is None:
        return None
    allowed = {"spectra", "wavelengths_nm", "detection_bands_nm", "channel_names",
               "normalization", "abundance_units"}
    if not isinstance(value, Mapping) or "spectra" not in value or set(value) - allowed:
        raise ValueError("unmixing must be a mapping with spectra and optional detector metadata, normalization and abundance_units")
    spectra = np.array(value["spectra"], copy=True, order="C")
    if (spectra.ndim != 2 or spectra.shape[0] != components or spectra.shape[1] < 1
            or spectra.dtype.kind not in "uif" or spectra.dtype.itemsize > 8
            or not np.isfinite(spectra).all()):
        raise ValueError("unmixing spectra must be a finite real K × C matrix, matching components, with at least one detector channel (up to 64-bit dtype)")
    channels = spectra.shape[1]
    result = {"spectra": spectra}
    for name, shape in (("wavelengths_nm", (channels,)), ("detection_bands_nm", (channels, 2))):
        result[name] = _channel_metadata_array(value.get(name), name, shape)
    names = value.get("channel_names")
    if names is not None:
        if isinstance(names, str):
            raise ValueError("unmixing channel_names must contain one string or None per detector channel")
        names = tuple(names)
        if len(names) != channels or any(n is not None and not isinstance(n, str) for n in names):
            raise ValueError("unmixing channel_names must contain one string or None per detector channel")
    result["channel_names"] = names
    for name in ("normalization", "abundance_units"):
        label = value.get(name)
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError(f"unmixing {name} must be a nonempty string or None")
        result[name] = label
    return result


@dataclass(frozen=True)
class SpectralImage:
    """Pixels with explicit axes and optional channel and spatial metadata.

    Use this for images produced by readers, simulations or external NumPy code.
    ``axes`` describes the *actual array order*, not OME storage DimensionOrder.
    ``kind`` is immutable and describes what pixel values represent:

    - "measured" (default): detector-channel intensities, including simulated
      detector data. Requires C/Y/X with optional T/Z, in any order.
    - "reconstructed": predicted detector-channel intensities after fitting,
      with the same C/Y/X rules and optional channel metadata.
    - "abundance": estimated component contributions. Uses K instead of C;
      optional component_names follow K order. No detector metadata on K.
      Optional unmixing metadata stores the exact K×C component spectra, their
      C-channel coordinates/names, normalization label and abundance units.
      These labels describe the saved scale; no normalization is applied.
      See fold_abundances() and reconstruct().

    X/Y/Z denote space and T time; each axis appears once. Channel wavelengths
    need not be uniformly spaced. Detection bounds describe recorded windows,
    not measured filter transmission. Pixel sizes are in micrometres.
    Abundances are not automatically fractions or counts; their units depend
    on the fitting method and spectral normalization. Nothing is rescaled.

    The pixel array is retained without copying or changing its dtype. Attributes
    cannot be rebound, but the NumPy array itself remains mutable. Metadata is
    copied at construction; callers should treat it as descriptive information.
    This object does not open files, infer axes, or perform unmixing.

    Examples
    --------
    >>> image = SpectralImage(np.zeros((4, 2, 3)), axes="CYX")
    >>> image.data.shape
    (4, 2, 3)
    """

    data: np.ndarray
    axes: str
    wavelengths_nm: np.ndarray | None = None
    detection_bands_nm: np.ndarray | None = None
    pixel_sizes_um: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    kind: str = "measured"
    component_names: tuple[str, ...] | None = None
    unmixing: Mapping | None = None

    def __post_init__(self):
        _validate_pixels(self.data)
        if self.kind not in ("measured", "reconstructed", "abundance"):
            raise ValueError("kind must be measured, reconstructed or abundance")
        channel_axis = "K" if self.kind == "abundance" else "C"
        _validate_layout(self.axes, self.data.shape, channel_axis)
        if self.kind == "abundance":
            if self.wavelengths_nm is not None or self.detection_bands_nm is not None:
                raise ValueError("abundance images cannot have wavelength or detection-band metadata")
            if self.component_names is not None:
                if isinstance(self.component_names, str):
                    raise TypeError("component_names must be a sequence of names")
                names = tuple(self.component_names)
                if len(names) != self.data.shape[self.axes.index("K")] or any(not isinstance(n, str) for n in names):
                    raise ValueError("component_names must contain one string per component")
                object.__setattr__(self, "component_names", names)
            object.__setattr__(self, "unmixing", _copy_unmixing_metadata(
                self.unmixing, self.data.shape[self.axes.index("K")]))
        else:
            if self.component_names is not None:
                raise ValueError("component_names are only valid for abundance images")
            if self.unmixing is not None:
                raise ValueError("unmixing metadata is only valid for abundance images")
            _copy_channel_metadata(self, self.data.shape[self.axes.index("C")])
        _copy_spatial_metadata(self)

    def save_ome_tiff(self, path, *, compression="deflate", bigtiff=None, overwrite=False):
        """Save quantitative pixels and supported metadata as lossless OME-TIFF.

        Convenience form of ``lumic.io.save_ome_tiff(self, path)``; both use
        the same exporter and validation. Supports measured, reconstructed
        and abundance images. Abundance K axes are stored as ordinary OME C
        channels with component meaning/spectra in a versioned annotation.
        Deflate is the default; use ``"lzw"`` or None for lossless LZW or
        uncompressed output. Existing files are protected unless overwrite=True. Returns the destination Path.

        Standard OME fields retain calibration/channel names; a versioned
        annotation preserves original axes, image kind, detector coordinates
        and JSON-compatible source provenance. Pixel values are not scaled.
        See ``lumic.io.save_ome_tiff`` for the full metadata/dtype contract
        and BigTIFF behavior. This saves the already loaded array, not a lazy
        or streamed dataset.

        Example: ``image.save_ome_tiff("simulation.ome.tiff")``.
        """
        from .io import save_ome_tiff

        return save_ome_tiff(self, path, compression=compression,
                             bigtiff=bigtiff, overwrite=overwrite)

    def slice(self, *, t=0, z=0):
        """Copy a selected T/Z plane or substack into a new SpectralImage.

        Indices are zero-based and relative to this loaded image. By default
        select the first T/Z plane, as reader.read() does. Zero also works for
        an absent T/Z axis; other absent/out-of-range indices raise IndexError.
        None retains the corresponding axis in full, including singleton axes.
        Selected T/Z axes are removed; other axes keep their original order.
        This method accepts measured, reconstructed and abundance images.

        Pixel storage is always copied, even when no dimensions are removed.
        Channel metadata, names, kind and spatial calibration are preserved;
        provenance is deep-copied and records the immediate-parent selection
        under image_slice. Existing image_slice information is retained as
        parent_slice when slicing repeatedly. Changing the result's pixels or
        metadata does not alter the original image.

        This operates on already loaded data and allocates a copy. To limit
        file I/O/memory, request T/Z selections from the reader before loading.
        It does not resample pixels, select channels or perform unmixing.

        Examples: image.slice(z=2) selects Z=2 at T=0;
        image.slice(t=1, z=None) copies all Z at T=1;
        image.slice(t=None, z=None) copies the complete loaded image.
        """
        indices = [slice(None)] * self.data.ndim
        selected = {}
        for axis, index in (("T", t), ("Z", z)):
            if index is None:
                continue
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                raise TypeError(f"{axis} selection must be an integer or None")
            if axis not in self.axes:
                if index != 0:
                    raise IndexError(f"Image has no {axis} axis; only index 0 is valid")
                continue
            position = self.axes.index(axis)
            size = self.data.shape[position]
            if index < 0 or index >= size:
                raise IndexError(f"{axis} selection {index} is outside [0, {size})")
            indices[position] = int(index)
            selected[axis] = int(index)
        axes = "".join(axis for axis in self.axes if axis not in selected)
        provenance = deepcopy(self.provenance)
        selection = {"source_axes": self.axes, "source_shape": self.data.shape,
                     "selection": selected}
        if "image_slice" in provenance:
            selection["parent_slice"] = provenance["image_slice"]
        provenance["image_slice"] = selection
        return SpectralImage(self.data[tuple(indices)].copy(), axes,
                             wavelengths_nm=self.wavelengths_nm,
                             detection_bands_nm=self.detection_bands_nm,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance=provenance, kind=self.kind,
                             component_names=self.component_names, unmixing=self.unmixing)

    def reconstruct(self):
        """Recreate the fitted spectral signal from abundance maps and a saved basis.

        Requires kind="abundance" and unmixing['spectra'], in the same K order
        and exact scale used to estimate these coefficients. Replace K with C
        in the image axes, preserve spatial calibration, and attach the basis's
        detector coordinates/names to the resulting kind="reconstructed" image.
        Neither coefficients nor spectra are normalized or clipped here.

        This recovers the model prediction, not measured residuals/noise or a
        unique biological solution. Abundance units depend on basis scaling;
        a stored normalization label is descriptive, not a request to rescale.
        Arrays/metadata are mutable, so the current basis is validated again.
        Reconstruction allocates a full output array and may copy coefficients.

        Example: reloaded_maps.reconstruct() after an OME-TIFF round trip.
        """
        if self.kind != "abundance":
            raise ValueError("reconstruct() requires an abundance image")
        if self.unmixing is None:
            raise ValueError("No component spectra saved; supply spectra when creating the abundance image")
        _validate_pixels(self.data)
        _validate_layout(self.axes, self.data.shape, "K")
        position = self.axes.index("K")
        basis = _copy_unmixing_metadata(self.unmixing, self.data.shape[position])
        spatial = np.moveaxis(self.data, position, -1)
        coefficients = np.ascontiguousarray(spatial.reshape(-1, spatial.shape[-1]))
        prediction = coefficients @ basis["spectra"]
        data = np.moveaxis(prediction.reshape(*spatial.shape[:-1], prediction.shape[-1]), -1, position)
        provenance = deepcopy(self.provenance)
        # A reloaded abundance file's standard channel_names describe K maps.
        # Only the basis can provide detector names for the new C-axis image.
        provenance.pop("channel_names", None)
        if basis["channel_names"] is not None:
            provenance["channel_names"] = basis["channel_names"]
        provenance["reconstruction_from_abundances"] = {
            "component_names": self.component_names,
            "normalization": basis["normalization"], "abundance_units": basis["abundance_units"],
        }
        return SpectralImage(data, self.axes.replace("K", "C"), kind="reconstructed",
                             wavelengths_nm=basis["wavelengths_nm"],
                             detection_bands_nm=basis["detection_bands_nm"],
                             pixel_sizes_um=self.pixel_sizes_um, provenance=provenance)

    def to_rgb(self, *, limits=None):
        """Return a display-only RGBRendering using source channel order.

        Average lower/middle/upper channel groups into blue/green/red, assuming
        shorter-to-longer wavelength order. This is a preview, not true color,
        fluorophore assignment or spectral unmixing. No wavelengths are needed,
        inspected or inferred. At least three finite channels are required.
        Abundance images require a separate component-color choice and are not
        accepted here. Original data and metadata are unchanged.

        All colors and all loaded T/Z planes use one shared linear scale.
        ``limits=(low, high)`` maps group-mean intensities to 0..255 with clipping
        and rounding to uint8. With no limits, zero is black and the largest RGB
        group mean across the entire loaded image is white-level intensity.
        Negative values clip to black; all-zero/nonpositive images render black.
        For those images the automatic limits are (0, 1), reusable explicitly.
        No independent color or per-plane normalization is performed. Reuse the
        returned limits explicitly when comparing separate images.

        Non-C axes retain their relative order; S (RGB samples) is appended:
        CZYX → ZYXS, YXC → YXS. A stack remains a stack; the caller chooses which
        plane to display. Rendering allocates output and temporary arrays for
        the loaded image; chunk selection remains the reader/user's decision.

        Example: ``rendering = image.to_rgb(limits=(0, 1000))`` followed by
        ``plt.imshow(rendering.data)`` for YXS, or ``rendering.data[z]`` for ZYXS.
        """
        return _render_rgb(self, limits)

    def unfold(self):
        """Return an UnfoldedImage with observations × channels and saved layout.

        Non-channel axes retain their relative order. Observations enumerate
        those axes in NumPy C order (last non-channel axis varies fastest).
        A view is used where possible; some layouts require a full pixel copy.
        No normalization, selection or intensity conversion is performed.
        Only the loaded image is unfolded; loading/chunking remains the caller's
        responsibility. Fitting one matrix assumes shared spectra across its
        observations, including any retained Z or T dimensions. This operation
        accepts measured/reconstructed images; abundance maps are fitting outputs,
        not detector-channel inputs for the spectral workflow.
        """
        if self.kind == "abundance":
            raise ValueError("unfold requires measured or reconstructed channel data, not abundance maps")
        channels_last = np.moveaxis(self.data, self.axes.index("C"), -1)
        matrix = channels_last.reshape(-1, channels_last.shape[-1])
        return UnfoldedImage(matrix, self.axes, self.data.shape,
                             wavelengths_nm=self.wavelengths_nm,
                             detection_bands_nm=self.detection_bands_nm,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance=self.provenance, source_kind=self.kind)


@dataclass(frozen=True)
class UnfoldedImage:
    """An observations × channels matrix with a reversible source layout.

    Normally created by ``image.unfold()``. ``source_axes`` and ``source_shape``
    describe the image before unfolding. Rows enumerate the non-C axes in their
    original order, in NumPy C order. Numerical algorithms receive ``data`` as
    an ordinary NumPy matrix and need no image-specific logic.

    ``fold_spectra`` restores measurement-channel data; ``fold_abundances``
    restores component maps. Both require all observations in the original row
    order. A reduced convex-hull pool or reordered rows cannot be folded directly:
    first unmix the full loaded block with the learned spectra. Shape checks
    cannot detect a row permutation; preserving correspondence is the caller's
    responsibility. This object retains layout and metadata, not a separate
    reference to the original image. It does not load/chunk data or run fitting.
    """

    data: np.ndarray
    source_axes: str
    source_shape: tuple[int, ...]
    wavelengths_nm: np.ndarray | None = None
    detection_bands_nm: np.ndarray | None = None
    pixel_sizes_um: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    source_kind: str = "measured"

    def __post_init__(self):
        _validate_pixels(self.data)
        if self.source_kind not in ("measured", "reconstructed"):
            raise ValueError("source_kind must be measured or reconstructed")
        shape = tuple(self.source_shape)
        _validate_layout(self.source_axes, shape, "C")
        object.__setattr__(self, "source_shape", shape)
        channels = shape[self.source_axes.index("C")]
        if self.data.shape != (prod(shape) // channels, channels):
            raise ValueError("data must be observations × channels matching the source layout")
        _copy_channel_metadata(self, channels)
        _copy_spatial_metadata(self)

    def _fold(self, matrix, columns):
        _validate_pixels(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != self.data.shape[0] or matrix.shape[1] == 0:
            raise ValueError("matrix must contain every observation in original order and at least one column")
        if columns is not None and matrix.shape[1] != columns:
            raise ValueError("spectral matrix must have the original number of channels")
        # Restore non-channel coordinates first, then move channels/components
        # back to the position originally occupied by C. No sorting or guessing.
        observation_shape = tuple(size for axis, size in zip(self.source_axes, self.source_shape) if axis != "C")
        channels_last = matrix.reshape(*observation_shape, matrix.shape[1])
        return np.moveaxis(channels_last, -1, self.source_axes.index("C"))

    def fold_spectra(self, data=None):
        """Restore the original SpectralImage layout and channel metadata.

        With no argument, restore ``self.data``. Pass a full reconstructed matrix
        (for example ``abundances @ spectra``) to inspect a fitted reconstruction.
        The matrix must have the same observation/channel counts and ordering.
        Dtype is preserved; values are not clipped or normalized. Reshape may
        copy data depending on memory layout. With no argument, preserve source
        kind; an explicitly supplied matrix is labeled "reconstructed".
        """
        matrix = self.data if data is None else data
        array = self._fold(matrix, self.data.shape[1])
        return SpectralImage(array, self.source_axes,
                             wavelengths_nm=self.wavelengths_nm,
                             detection_bands_nm=self.detection_bands_nm,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance=self.provenance,
                             kind=self.source_kind if data is None else "reconstructed")

    def fold_abundances(self, data, *, component_names=None, spectra=None,
                        normalization=None, abundance_units=None):
        """Restore component maps as a SpectralImage with kind="abundance".

        Input is observations × components. Replace C in the source axes with K
        (CZYX → KZYX, YXC → YXK). Preserve calibration and source provenance;
        detector wavelengths/bands are never attached directly to components.
        Optional component_names follow matrix column order, without matching.

        To reconstruct the fitted detector signal later, pass the same
        component spectra (K × C) used to fit the abundances, on the same
        scale. The image stores those spectra and the source detector
        metadata without rescaling the values.

        When you supply spectra, you can also use `normalization` and
        `abundance_units` to describe their scale and the units of the
        fitted abundances. These labels do not change the values.

        To save abundances fitted with L1-normalized reference spectra:

        ```python
        maps = unfolded.fold_abundances(
            abundances,
            spectra=reference_spectra,
            component_names=["A", "B"],
            normalization="L1",
        )
        maps.save_ome_tiff("abundances.ome.tif")
        ```

        After loading the saved image, call `reconstruct()` to recover the
        fitted detector signal. Reconstruction requires the stored spectra.
        """
        unmixing = None
        if spectra is None:
            if normalization is not None or abundance_units is not None:
                raise ValueError("normalization/abundance_units require component spectra")
        else:
            matrix = np.asarray(spectra)
            if matrix.ndim != 2 or matrix.shape[1] != self.data.shape[1]:
                raise ValueError("spectra must be K × C matching the source detector-channel count")
            unmixing = dict(spectra=matrix, wavelengths_nm=self.wavelengths_nm,
                            detection_bands_nm=self.detection_bands_nm,
                            channel_names=self.provenance.get("channel_names"),
                            normalization=normalization, abundance_units=abundance_units)
        return SpectralImage(self._fold(data, None), self.source_axes.replace("C", "K"),
                             kind="abundance", component_names=component_names,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance=self.provenance, unmixing=unmixing)


def _rgb_channel_groups(n_channels):
    """Return R/G/B index groups matching the legacy channel-order mapping."""
    if n_channels < 3:
        raise ValueError("RGB rendering requires at least three spectral channels")
    boundaries = np.round(np.linspace(0, n_channels - 1, num=4)).astype(int)
    boundaries[2] = max(boundaries[2], boundaries[1] + 1)
    return (tuple(range(boundaries[2], n_channels)),
            tuple(range(boundaries[1], boundaries[2])),
            tuple(range(boundaries[1])))


@dataclass(frozen=True)
class RGBRendering:
    """Display pixels and the settings used to create a channel-order preview.

    ``data`` is uint8 with RGB last. ``axes`` preserves the source non-C axis
    order and appends S for three color samples: CYX → YXS, CZYX → ZYXS.
    S means RGB display values, not measured spectral channels or components.
    ``channel_groups`` records source-channel indices in R, G, B order.
    ``limits`` records the one shared intensity interval mapped to [0, 255],
    applied to every color and every pixel/plane/time point. Clipping at either
    limit and 8-bit rounding can affect color balance at extremes.

    No fluorophores are identified or unmixed. Colors are not calibrated human
    vision or true color: lower/middle/upper source-channel groups map to B/G/R
    respectively, assuming channels are ordered from shorter to longer emission
    wavelengths. No wavelength metadata is interpreted or channels reordered;
    overlapping windows and unordered channels need not yield spectral colors.

    Spatial calibration and source provenance are descriptive, not display
    controls. Attributes are frozen; arrays and copied metadata remain mutable.
    The object has no unfolding, fitting or spectral-channel metadata API.
    Creating a stack does not select a plane; select explicitly when plotting.
    """

    data: np.ndarray
    axes: str
    channel_groups: tuple[tuple[int, ...], ...]
    limits: tuple[float, float]
    pixel_sizes_um: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray) or self.data.dtype != np.uint8:
            raise TypeError("RGBRendering data must be a uint8 NumPy array")
        if (not isinstance(self.axes, str) or len(self.axes) != self.data.ndim
                or not self.axes.endswith("S") or self.data.shape[-1] != 3
                or len(set(self.axes)) != len(self.axes)
                or not set(self.axes) <= set("TZYXS") or not set("YXS") <= set(self.axes)
                or any(size == 0 for size in self.data.shape)):
            raise ValueError("RGB axes must match data, include Y/X and end with S of length 3")
        groups = tuple(tuple(group) for group in self.channel_groups)
        if len(groups) != 3 or any(not group for group in groups):
            raise ValueError("channel_groups must contain three nonempty groups in RGB order")
        indices = [index for group in groups for index in group]
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("channel_groups must contain distinct nonnegative integer indices")
        limits = np.asarray(self.limits, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("rendering limits must be a finite (low, high) pair with high > low")
        object.__setattr__(self, "channel_groups", groups)
        object.__setattr__(self, "limits", tuple(map(float, limits)))
        _copy_spatial_metadata(self)


def _render_rgb(image, limits):
    """Render an already loaded SpectralImage without changing its pixels."""
    if image.kind == "abundance":
        raise ValueError("to_rgb requires measured or reconstructed channel data, not abundance maps")
    axis = image.axes.index("C")
    groups = _rgb_channel_groups(image.data.shape[axis])
    if not np.isfinite(image.data).all():
        raise ValueError("RGB rendering requires finite pixel values")
    channels_last = np.moveaxis(image.data, axis, -1)
    # Keep float means until final display conversion; weak floating-point
    # spectra must not disappear through the legacy integer truncation.
    rgb = np.stack([channels_last[..., list(group)].mean(axis=-1, dtype=float)
                    for group in groups], axis=-1)
    if not np.isfinite(rgb).all():
        raise ValueError("channel averages exceed the finite display range")
    if limits is None:
        maximum = float(rgb.max())
        low, high = 0.0, maximum if maximum > 0 else 1.0
    else:
        values = np.asarray(limits, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
            raise ValueError("limits must be a finite (low, high) pair with high > low")
        low, high = map(float, values)
    if not np.isfinite(high - low):
        raise ValueError("limits must span a finite intensity range")
    scaled = (np.clip(rgb, low, high) - low) / (high - low)
    pixels = np.rint(scaled * 255).astype(np.uint8)
    return RGBRendering(pixels, image.axes.replace("C", "") + "S", groups, (low, high),
                        pixel_sizes_um=image.pixel_sizes_um,
                        provenance={"source": deepcopy(image.provenance),
                                    "source_axes": image.axes, "source_kind": image.kind})
