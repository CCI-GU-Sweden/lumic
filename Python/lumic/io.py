import numpy as np
from pathlib import Path
from pylibCZIrw import czi as pyczi
from lxml import etree
import json
import warnings

from .readers import SpectralImageReader, format_reader_summary

def _validate_czi_path(pth):
    """Check the file before passing it to the CZI backend."""
    pth = Path(pth)
    if pth.suffix.lower() != '.czi':
        raise ValueError(f"Expected a .czi file: {pth}")
    if not pth.is_file():
        raise FileNotFoundError(f"CZI file not found: {pth}")
    return pth

def getCZIchannelWave(pth):
    """ Returns the wavelength information from czi file - spectral detector
    
    This funtion that receives a Path object to a .czi file, reads its meta dta and returns Emission Wavelengths
    chennel information. Function has been tested on ZEN black and blue but might failt if keywords are not found
    in the metadata directory.

    Parameters
    ---------
    pth : Path or str
        Path to CZI file

    Returns
    -------
    1dArray
        A 1d numpy array containing the wavelength of each emission channel

    Raises
    ------
    FileNotFoundError
        If czi file does not exist
    ValueError
        If the file is not a readable CZI file
    UserWarning 
        If wavelength metadata is not found

    """

    pth = _validate_czi_path(pth)

    # open the CZI for reading
    try:
        with pyczi.open_czi(str(pth)) as czidoc:
            # get the raw metadata as a dictionary
            md_dict = czidoc.metadata
    except RuntimeError as exc:
        raise ValueError(f"Not a valid CZI file: {pth}") from exc

    channel_info = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]
    EmWave = []
    for x in channel_info:
        #print(x['EmissionWavelength'])
        # this is only compatible with ZEN black
        #if 'EmissionWavelength' in x:
        #    EmWave.extend([x['EmissionWavelength']])
        if 'DetectionWavelength' in x:
            #print('This works in be ZEN blue 980 and balck 880 key: DetectionWavelength')
            wTxt = x['DetectionWavelength']['Ranges']
            w = wTxt.split("-")
            w = float(w[0]) + (float(w[1]) - float(w[0])) / 2
            EmWave.extend([w])

        else:
            warnings.warn('Can not find key: EmissionWavelength or key: DetectionWavelength, returning empty')


    return np.asarray(EmWave).astype('float').astype('int')

def loadCZISpecData(pth):
    """Returns the spectral dataset from a czi file as a 3D numpy array [C, Y, X]
    
    Legacy array-only wrapper around load_czi; uses the same pylibCZIrw backend.

    Parameters
    ---------
    pth : Path or str
        Path to CZI file

    Returns
    -------
    specData : numpyArray
        A 3D numpy array containing the image of each channel the order is [C, Y, X]

    Raises
    ------
    FileNotFoundError
        If the CZI file does not exist
    ValueError
        If the path does not name a readable CZI file

    See Also
    --------
    loadCZIChannel : Function that loads single channel
    """
    # Preserve the legacy CYX array result; stacks require explicit selection
    # through load_czi() rather than silently choosing their first plane.
    with open_czi(pth) as reader:
        if any(axis in reader.axes for axis in "TZ"):
            raise ValueError("loadCZISpecData requires a single CYX plane; use load_czi(t=..., z=...)")
        print(f"Channels are located in index {reader.native_axes.index('C')}")
        print(f"Image has {reader.shape[reader.axes.index('C')]} channels")
        return reader.read().data


def loadCZIChannel(czi, idx):
    """Load one YX channel using a legacy or current open reader.

    This legacy helper emits FutureWarning; prefer
    ``load_czi(path).data[idx]`` for the default CYX plane. Accepts a Lumic
    CziReader, an open pylibCZIrw reader, or an existing object exposing the
    old ``read_image(C=idx)`` interface. Supporting that last interface does
    not import or require aicspylibczi. The caller owns the reader's lifetime.
    """
    from numbers import Integral

    warnings.warn(
        "loadCZIChannel() is a legacy helper. For new code, use "
        "load_czi(path).data[idx] to select a channel from the default CYX plane.",
        FutureWarning, stacklevel=2,
    )
    if isinstance(idx, bool) or not isinstance(idx, Integral):
        raise TypeError("Channel index must be an integer")
    if isinstance(czi, CziReader):
        start, end = czi.dimension_ranges["C"]
        if idx < 0 or idx >= end - start:
            raise IndexError("Channel index is outside the image")
        return czi._read_plane({"C": start + int(idx)})
    if isinstance(czi, pyczi.CziReader):
        start, end = czi.total_bounding_box["C"]
        if idx < 0 or idx >= end - start:
            raise IndexError("Channel index is outside the image")
        return czi.read(plane={"C": start + int(idx)}, zoom=1.0)[..., 0].copy()
    # Duck typing preserves externally constructed CziFile objects without
    # retaining an import-time dependency on their package.
    if callable(getattr(czi, "read_image", None)):
        data, dims = czi.read_image(C=int(idx))
        keep = [i for i, (axis, _) in enumerate(dims) if axis in "YX"]
        if len(keep) != 2 or any(size != 1 for i, (_, size) in enumerate(dims) if i not in keep):
            raise ValueError("Expected one YX channel plane")
        return np.squeeze(data, axis=tuple(i for i in range(data.ndim) if i not in keep))
    raise TypeError("Expected an open Lumic, pylibCZIrw or legacy CZI reader")


def _czi_layout(reader):
    """Inspect subblock headers without decoding or compositing any pixels.

    Actual coordinates preserve nonzero dimension starts. Subblocks also
    distinguish tiles from complete planes, so composition can be recorded
    explicitly in the returned image's provenance.
    """
    ranges, tiles, scenes = {}, {}, set()
    scalar = True
    def inspect(index, block):
        nonlocal scalar
        coords = block.coordinate.to_dict()
        rect = block.logicalRect
        scalar &= block.pixelType.name.startswith("Gray")
        for axis, value in coords.items():
            low, high = ranges.get(axis, (value, value + 1))
            ranges[axis] = (min(low, value), max(high, value + 1))
        if "S" in coords:
            scenes.add(coords["S"])
        key = tuple(sorted(coords.items()))
        tiles.setdefault(key, set()).add((rect.x, rect.y, rect.w, rect.h))
        return True
    reader.enumerate_subblocks(inspect)
    bounds = reader.total_bounding_box
    ranges.update({axis: bounds[axis] for axis in "YX"})
    mosaic = any(len(rects) > 1 for rects in tiles.values())
    # Match the established CZI array convention, independent of dict ordering.
    axes = "".join(axis for axis in "BVHIRSTCZYX" if axis in ranges)
    return axes, ranges, mosaic, scenes, scalar


class CziReader:
    """Inspect a CZI header, then load selected pixels into a SpectralImage.

    Prefer ``with open_czi(path) as reader`` to release the file promptly.
    Construction reads metadata only. ``axes`` and ``shape`` describe the available image
    before selection: singleton T/Z and backend bookkeeping axes are omitted, while C/Y/X
    are always retained. Native axes and index bounds remain available as
    ``native_axes`` and ``dimension_ranges``.

    Supports one image or composed tiled image with optional T/Z dimensions.
    Tiles use their stored positions at native resolution; overlaps follow
    libCZI's highest-M-index rule, without new registration or blending.
    Multiple scenes/views and RGB sample axes are rejected rather than silently
    combined. This is layout support, not validation of spectral interpretation
    for every Zeiss instrument. Legacy public call signatures are retained.
    """

    format_name = "CZI"
    kind = "measured"

    def summary(self):
        """Return header/layout details without decoding pixels or printing.

        Channel count is not a fluorophore count or evidence of suitability
        for spectral unmixing. The summary remains available after close().
        """
        return format_reader_summary(self)

    def __init__(self, path):
        self.path = _validate_czi_path(path)
        self._context = None
        self._czi = None
        try:
            try:
                self._context = pyczi.open_czi(str(self.path))
                self._czi = self._context.__enter__()
            except RuntimeError as exc:
                raise ValueError(f"Not a valid CZI file: {self.path}") from exc
            self.native_axes, self.dimension_ranges, mosaic, scenes, scalar = _czi_layout(self._czi)
            self.is_mosaic = mosaic
            if len(scenes) > 1:
                raise ValueError("Multiple CZI scenes are not supported")
            if not scalar:
                raise ValueError("RGB sample CZI images are not supported")
            if not set("CYX") <= set(self.native_axes):
                raise ValueError("CZI must contain C, Y and X dimensions")
            for axis, (start, end) in self.dimension_ranges.items():
                if axis not in "CTZYX" and end - start != 1:
                    raise ValueError(f"Unsupported CZI dimension: {axis}")
            self.axes = "".join(a for a in self.native_axes if a in "CYX" or
                                (a in "TZ" and self.dimension_ranges[a][1] - self.dimension_ranges[a][0] > 1))
            self.shape = tuple(self.dimension_ranges[a][1] - self.dimension_ranges[a][0] for a in self.axes)
            meta = etree.fromstring(self._czi.raw_metadata.encode("utf-8"))
            channels = meta.findall(".//Information/Image/Dimensions/Channels/Channel")
            bands = [channel.findtext("DetectionWavelength/Ranges") for channel in channels]
            self.detection_bands_nm = None
            if bands and all(value is not None for value in bands):
                self.detection_bands_nm = np.array([list(map(float, value.split("-"))) for value in bands])
                if self.detection_bands_nm.shape != (self.dimension_ranges["C"][1] - self.dimension_ranges["C"][0], 2):
                    raise ValueError("CZI detection metadata does not match the channel dimension")
            self.wavelengths_nm = None
            # Midpoints are useful labels only for ordered, non-overlapping bins.
            # In particular, LSM 900 paired windows must not become point spectra.
            if self.detection_bands_nm is not None:
                b = self.detection_bands_nm
                if np.all(b[:, 1] > b[:, 0]) and np.all(b[1:, 0] >= b[:-1, 1] - 1e-6):
                    self.wavelengths_nm = b.mean(axis=1)
            self.pixel_sizes_um = {}
            for item in meta.findall(".//Scaling/Items/Distance"):
                axis = item.attrib.get("Id")
                value = item.findtext("Value")
                if axis in ("X", "Y", "Z") and value is not None:
                    size = float(value) * 1e6
                    if np.isfinite(size) and size > 0:
                        self.pixel_sizes_um[axis] = size
        except Exception:
            self.close()
            raise

    def read(self, *, t=0, z=0):
        """Load all channels from the first T/Z plane by default (CYX).

        Indices are zero-based relative to the file's dimension range. Zero is
        accepted when T or Z is absent, treating that axis as a single plane;
        other indices for an absent axis raise IndexError. Set t=None or z=None
        explicitly to read that entire dimension. Constraints are passed to the
        backend before decoding;
        this does not load the full stack and then slice it. Selected axes and
        singleton T/Z axes are removed; C/Y/X are retained even at size one.
        Returned data retains the established CZI axis order, identified by
        ``image.axes``.
        """
        from .image import SpectralImage
        from numbers import Integral

        if self._czi is None:
            raise ValueError("CZI reader is closed")
        constraints = {}
        selections = {}
        for axis, index in (("T", t), ("Z", z)):
            if index is None:
                continue
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError(f"{axis} selection must be an integer")
            if axis not in self.dimension_ranges:
                if index != 0:
                    raise IndexError(f"CZI has no {axis} dimension; only index 0 is valid")
                continue
            start, end = self.dimension_ranges[axis]
            if index < 0 or index >= end - start:
                raise IndexError(f"{axis} selection {index} is outside [0, {end - start})")
            constraints[axis] = start + int(index)
            selections[axis] = int(index)
        axes = "".join(a for a in self.axes if a not in constraints)
        plane_axes = [a for a in axes if a not in "YX"]
        shape = tuple(self.dimension_ranges[a][1] - self.dimension_ranges[a][0] for a in axes)
        array = None
        # Decode only requested C/T/Z planes. Assign them directly into the
        # output to avoid loading a complete stack before selecting a slice.
        for position in np.ndindex(*(shape[axes.index(a)] for a in plane_axes)):
            plane = dict(constraints)
            plane.update({a: self.dimension_ranges[a][0] + i for a, i in zip(plane_axes, position)})
            pixels = self._read_plane(plane)
            if array is None:
                array = np.empty(shape, dtype=pixels.dtype)
            target = tuple(position[plane_axes.index(a)] if a in plane_axes else slice(None) for a in axes)
            array[target] = pixels
        return SpectralImage(array, axes, wavelengths_nm=self.wavelengths_nm,
                             detection_bands_nm=self.detection_bands_nm,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance={"format": "CZI", "path": str(self.path),
                                         "native_axes": self.native_axes, "selection": selections,
                                         "mosaic_composed": self.is_mosaic,
                                         "spatial_origin_xy": (self.dimension_ranges["X"][0],
                                                               self.dimension_ranges["Y"][0])})

    def _read_plane(self, plane):
        """Read native-resolution grayscale pixels, retaining singleton Y/X."""
        if self._czi is None:
            raise ValueError("CZI reader is closed")
        pixels = self._czi.read(plane=plane, zoom=1.0)
        if pixels.ndim != 3 or pixels.shape[-1] != 1:
            raise ValueError("Expected one grayscale YX channel plane")
        return pixels[..., 0]

    def close(self):
        """Release the backend; safe to call repeatedly."""
        self._czi = None
        if self._context is not None:
            context, self._context = self._context, None
            context.__exit__(None, None, None)

    def __enter__(self):
        if self._czi is None:
            raise ValueError("CZI reader is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def open_czi(path):
    """Open metadata for selective CZI reads; use as a context manager.

    Example: ``with open_czi(path) as reader: image = reader.read(t=0, z=2)``.
    See :class:`CziReader` for supported layouts and metadata limitations.
    """
    return CziReader(path)


def load_czi(path, *, t=0, z=0):
    """Read CZI pixels and metadata into a SpectralImage, then close the file.

    Defaults to the first T/Z plane (CYX), including files without T/Z axes.
    Set t=None and/or z=None explicitly to load entire dimensions. Use
    ``open_czi`` to inspect available shape first. Returned arrays remain usable
    after the reader closes; explicitly requested stacks retain their axes.
    Detection bands are retained, and overlapping windows have no midpoint
    wavelength axis. This does not reconstruct LSM 900 sequential-window data.
    """
    with open_czi(path) as reader:
        return reader.read(t=t, z=z)


def _ome_length_nm(value, unit, field):
    """Convert explicit OME length units; never silently assume unknown units."""
    factors = {"m": 1e9, "cm": 1e7, "mm": 1e6, "µm": 1e3,
               "nm": 1., "pm": 1e-3, "fm": 1e-6, "Å": .1}
    name = getattr(unit, "value", unit)
    if name not in factors:
        raise ValueError(f"Unsupported OME length unit {name!r} for {field}")
    result = float(value) * factors[name]
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a finite positive length")
    return result


class OmeTiffReader:
    """Inspect OME metadata, then decode selected scalar-image T/Z planes.

    tifffile handles TIFF structure/compression and logical page ordering;
    ome-types parses and validates OME-XML. This adapter reconciles the two and
    returns SpectralImage. Use ``with open_ome_tiff(path) as reader`` for prompt
    cleanup. Construction inspects headers/metadata without decoding pixels.

    Initial scope: single-file, non-pyramidal OME-TIFF/BigTIFF, optionally tiled
    or losslessly compressed, with one grayscale YX image per TIFF page. Choose
    a series explicitly with ``series=...`` (default first). RGB/sample layouts,
    multi-file datasets, missing pages and volumetric pages are rejected rather
    than guessed. No file conversion, stitching or OME-Zarr access is provided.

    ``axes``/``shape`` describe available data before T/Z selection; C/Y/X are
    retained even at size one, singleton T/Z are omitted for third-party files.
    ``native_axes`` is the TIFF series layout, which can differ from another
    reader's array order.
    OME DimensionOrder describes storage, not a ready-made NumPy axes string.

    Wavelengths use complete OME Channel.EmissionWavelength fields, or, if all
    those fields are absent, complete numeric channel names such as '415 nm'.
    Label-derived values are recorded as such; they are not calibrated band
    centers. Partial/ambiguous metadata leaves wavelengths absent. Detector-band
    bounds are unavailable in generic exports and never inferred from spacing.
    Original channel names and wavelength provenance accompany each result.
    Files written by save_ome_tiff carry a linked, versioned OME annotation:
    original array axes, measured/reconstructed kind and detector coordinates
    are restored from it, with source provenance nested under source_provenance.
    This is Lumic-specific metadata, not inferred from third-party files.
    """

    format_name = "OME-TIFF"

    def summary(self):
        """Return header/layout details without decoding pixels or printing.

        Reports the selected series, not every series in a multi-image file.
        Channel count is not a fluorophore count. Metadata remains available
        after close(); decoding can still fail for damaged pixel blocks.
        """
        return format_reader_summary(self)

    def __init__(self, path, *, series=0):
        from numbers import Integral
        import re
        import tifffile
        from ome_types import from_xml

        self.path = Path(path)
        if self.path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Expected an OME-TIFF .tif or .tiff file: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"OME-TIFF file not found: {self.path}")
        if isinstance(series, (bool, np.bool_)) or not isinstance(series, Integral):
            raise TypeError("series must be a zero-based integer")
        self._tif = None
        self._series = None
        self.series_index = int(series)
        try:
            try:
                self._tif = tifffile.TiffFile(self.path)
            except (tifffile.TiffFileError, OSError) as exc:
                raise ValueError(f"Not a valid TIFF file: {self.path}") from exc
            xml = self._tif.ome_metadata
            if not xml:
                raise ValueError(f"TIFF file has no OME-XML metadata: {self.path}")
            try:
                self.ome_metadata = from_xml(xml, validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid OME-XML metadata: {self.path}") from exc
            # Check references before asking tifffile to assemble image series,
            # which can otherwise discover/open companion files automatically.
            for image in self.ome_metadata.images:
                for block in image.pixels.tiff_data_blocks:
                    if block.uuid is not None:
                        if (block.uuid.value != self.ome_metadata.uuid
                                or block.uuid.file_name not in (None, self.path.name)):
                            raise ValueError("Multi-file OME-TIFF loading is not supported")
            if not 0 <= self.series_index < len(self.ome_metadata.images):
                raise IndexError(f"series index {series} is outside the OME image collection")
            if len(self._tif.series) != len(self.ome_metadata.images):
                raise ValueError("TIFF series and OME image metadata do not correspond")
            self._series = self._tif.series[self.series_index]
            if len(self._series.levels) != 1:
                raise ValueError("Pyramidal OME-TIFF loading is not supported")
            self.native_axes = self._series.axes
            sizes = dict(zip(self.native_axes, self._series.shape))
            pixels = self.ome_metadata.images[self.series_index].pixels
            ome_sizes = {axis: getattr(pixels, f"size_{axis.lower()}") for axis in "CTZYX"}
            if (not self.native_axes.endswith("YX") or not set(self.native_axes) <= set("CTZYX")
                    or len(set(self.native_axes)) != len(self.native_axes)
                    or self._series.dtype.kind not in "uif"):
                raise ValueError(f"Unsupported OME-TIFF layout: {self.native_axes}; expected scalar CTZYX axes ending in YX")
            for axis, size in ome_sizes.items():
                if sizes.get(axis, 1) != size:
                    raise ValueError(f"OME Size{axis} does not match the TIFF series layout")
            if len(pixels.channels) != pixels.size_c or any(c.samples_per_pixel not in (None, 1) for c in pixels.channels):
                raise ValueError("OME-TIFF channels must each contain one scalar sample")
            self.dimension_ranges = {a: (0, size) for a, size in ome_sizes.items()}
            axes = self.native_axes if "C" in self.native_axes else "C" + self.native_axes
            self.axes = "".join(a for a in axes if a in "CYX" or ome_sizes[a] > 1)
            self.shape = tuple(ome_sizes[a] for a in self.axes)
            self._page_axes = self.native_axes[:-2]
            self._page_shape = self._series.shape[:-2]
            if len(self._series.pages) != int(np.prod(self._page_shape)):
                raise ValueError("OME-TIFF requires one scalar YX page per channel/time/Z position")
            for page in self._series.pages:
                if page is None or page.shape != (pixels.size_y, pixels.size_x):
                    raise ValueError("Missing or volumetric OME-TIFF pages are not supported")
                keyframe = page.keyframe
                if keyframe.samplesperpixel != 1 or keyframe.photometric not in (0, 1):
                    raise ValueError("OME-TIFF RGB/palette sample data is not a scalar spectral image")
            self.pixel_sizes_um = {}
            for axis in "XYZ":
                value = getattr(pixels, f"physical_size_{axis.lower()}")
                unit = getattr(pixels, f"physical_size_{axis.lower()}_unit")
                if value is not None:
                    self.pixel_sizes_um[axis] = _ome_length_nm(value, unit, f"PhysicalSize{axis}") / 1000
            self._storage_axes = self.axes
            self._storage_ranges = dict(self.dimension_ranges)
            self.component_names = None
            self.unmixing = None
            self._lumic_metadata = _lumic_ome_annotation(self.ome_metadata, self.series_index, ome_sizes)
            self.kind = "measured"
            self.channel_names = tuple(c.name for c in pixels.channels)
            self.detection_bands_nm = None
            self.wavelengths_nm = None
            self.wavelength_source = None
            emission = [c.emission_wavelength for c in pixels.channels]
            if all(value is not None for value in emission):
                self.wavelengths_nm = np.array([
                    _ome_length_nm(c.emission_wavelength, c.emission_wavelength_unit, "EmissionWavelength")
                    for c in pixels.channels])
                self.wavelength_source = "OME Channel.EmissionWavelength"
            elif all(value is None for value in emission):
                labels = [re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*nm\s*", name or "")
                          for name in self.channel_names]
                if all(match is not None for match in labels):
                    values = np.array([float(match.group(1)) for match in labels])
                    if np.all(values > 0):
                        self.wavelengths_nm = values
                        self.wavelength_source = "OME Channel.Name (numeric nm label)"
            if self._lumic_metadata is not None:
                info = self._lumic_metadata
                self.axes = info["axes"]
                self.shape = tuple(ome_sizes["C" if a == "K" else a] for a in self.axes)
                self.kind = info["kind"]
                self.component_names = info.get("component_names")
                self.unmixing = info.get("unmixing")
                if self.kind == "abundance":
                    # Public dimensions describe component maps; internal TIFF
                    # page addressing continues to use ordinary OME C channels.
                    self.dimension_ranges["K"] = self.dimension_ranges.pop("C")
                self.wavelengths_nm = (None if info["wavelengths_nm"] is None
                                       else np.array(info["wavelengths_nm"], dtype=float))
                self.detection_bands_nm = (None if info["detection_bands_nm"] is None
                                           else np.array(info["detection_bands_nm"], dtype=float))
                self.wavelength_source = (None if self.wavelengths_nm is None
                                          else "Lumic OME MapAnnotation")
        except Exception:
            self.close()
            raise

    def read(self, *, t=0, z=0):
        """Read selected T/Z positions and every channel into a SpectralImage.

        Defaults to the first T/Z plane (normally CYX). None retains a complete
        dimension; selected T/Z axes are omitted, C/Y/X (or K/Y/X for abundance
        exports) are retained. Lumic
        exports restore their original axis order and explicit singleton T/Z
        axes when those dimensions are requested with None. Other files omit
        singleton T/Z axes. Indices are zero-based; zero works for absent/singleton
        axes. Other invalid indices fail before decoding. The returned array
        order is reported in image.axes.

        Only matching TIFF pages are decoded, including for compressed inputs.
        A tiled page is decoded in full: no sub-plane ROI reading is claimed.
        Outputs own their pixels and remain usable after closing the reader.
        """
        from itertools import product
        from numbers import Integral
        from .image import SpectralImage

        if self._tif is None:
            raise ValueError("OME-TIFF reader is closed")
        selected = {}
        for axis, index in (("T", t), ("Z", z)):
            if index is None:
                continue
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                raise TypeError(f"{axis} selection must be an integer or None")
            size = self.dimension_ranges[axis][1]
            if not 0 <= index < size:
                raise IndexError(f"{axis} selection {index} is outside [0, {size})")
            selected[axis] = int(index)
        output_axes = "".join(a for a in self._storage_axes if a not in selected)
        output_shape = tuple(self._storage_ranges[a][1] for a in output_axes)
        output = np.empty(output_shape, dtype=self._series.dtype)
        ranges = [(selected[a],) if a in selected else range(size)
                  for a, size in zip(self._page_axes, self._page_shape)]
        # tifffile's logical series page sequence already incorporates OME
        # TiffData mapping. Select coordinates there, not by guessing raw IFDs.
        for coordinates in product(*ranges):
            index = int(np.ravel_multi_index(coordinates, self._page_shape)) if coordinates else 0
            location = dict(zip(self._page_axes, coordinates))
            target = tuple(location.get(a, 0) for a in output_axes[:-2])
            output[target] = self._series.pages[index].asarray()
        desired_axes = "".join(a for a in self.axes if a not in selected)
        storage_order = desired_axes.replace("K", "C")
        # TIFF/OME may omit singleton axes. Reintroduce only axes explicitly
        # recorded by our exporter, then restore the user's array permutation.
        for axis in storage_order:
            if axis not in output_axes:
                output = np.expand_dims(output, axis=-1)
                output_axes += axis
        output = output.transpose([output_axes.index(a) for a in storage_order])
        provenance = {"format": "OME-TIFF", "path": str(self.path),
                      "native_axes": self.native_axes,
                      "ome_dimension_order": self.ome_metadata.images[self.series_index].pixels.dimension_order.value,
                      "series": self.series_index, "selection": selected,
                      "channel_names": self.channel_names,
                      "wavelength_source": self.wavelength_source}
        if self._lumic_metadata is not None:
            provenance["source_provenance"] = self._lumic_metadata["provenance"]
        return SpectralImage(output, desired_axes, kind=self.kind,
                             wavelengths_nm=self.wavelengths_nm,
                             detection_bands_nm=self.detection_bands_nm,
                             pixel_sizes_um=self.pixel_sizes_um,
                             provenance=provenance, component_names=self.component_names,
                             unmixing=self.unmixing)

    def close(self):
        """Release the TIFF handle; safe to call repeatedly."""
        if self._tif is not None:
            self._tif.close()
            self._tif = None
        self._series = None

    def __enter__(self):
        if self._tif is None:
            raise ValueError("OME-TIFF reader is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def open_ome_tiff(path, *, series=0):
    """Inspect an OME-TIFF without decoding pixels; use as a context manager.

    Example: ``with open_ome_tiff(path) as reader: image = reader.read(z=2)``.
    See OmeTiffReader for supported layouts and metadata interpretation.
    """
    return OmeTiffReader(path, series=series)


def load_ome_tiff(path, *, t=0, z=0, series=0):
    """Return a SpectralImage from selected OME-TIFF pixels and close the file.

    Defaults to t=0, z=0 (normally CYX); use None to retain all positions.
    Lumic exports restore their original array axis order and metadata, including
    explicit singleton T/Z axes when requested with None. Other files follow
    their logical TIFF series order. Channel metadata is optional; band bounds
    are not inferred from labels. Use open_ome_tiff to inspect before decoding.
    """
    with open_ome_tiff(path, series=series) as reader:
        return reader.read(t=t, z=z)


_LUMIC_OME_NAMESPACE = "urn:lumic:spectral-image:1"
_LUMIC_ABUNDANCE_OME_NAMESPACE = "urn:lumic:spectral-image:2"


def _ome_json_value(value):
    """Convert common NumPy/path metadata to JSON without opaque repr strings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported provenance value: {type(value).__name__}")


def _lumic_ome_annotation(ome, series, sizes):
    """Read version 1/2 image-linked annotations before decoding any pixels.

    Version 1 remains the measured/reconstructed contract. Version 2 adds
    abundance K semantics and an optional exact component basis. OME storage
    always uses C; unknown namespaces are ignored without guessing image kind.
    """
    from .image import SpectralImage

    refs = {ref.id for ref in ome.images[series].annotation_refs}
    annotations = [a for a in ome.structured_annotations.map_annotations
                   if a.id in refs and a.namespace in (_LUMIC_OME_NAMESPACE, _LUMIC_ABUNDANCE_OME_NAMESPACE)]
    if not annotations:
        return None
    try:
        if len(annotations) != 1:
            raise ValueError("duplicate annotations")
        annotation = annotations[0]
        entries = annotation.value.ms
        if len(entries) != 1 or entries[0].k != "spectral_image":
            raise ValueError("expected one spectral_image entry")
        info = json.loads(entries[0].value)
        axes, shape, kind = info["axes"], info["shape"], info["kind"]
        channel_axis = "K" if kind == "abundance" else "C"
        valid_kinds = (("measured", "reconstructed") if annotation.namespace == _LUMIC_OME_NAMESPACE
                       else ("measured", "reconstructed", "abundance"))
        if (kind not in valid_kinds or not isinstance(axes, str)
                or not set(channel_axis + "YX") <= set(axes)
                or not set(axes) <= set(channel_axis + "TZYX") or len(set(axes)) != len(axes)
                or not isinstance(shape, list) or len(shape) != len(axes)
                or any(type(n) is not int or n < 1 for n in shape)
                or any(dict(zip(axes.replace("K", "C"), shape)).get(a, 1) != sizes[a] for a in "CTZYX")
                or not isinstance(info["provenance"], dict)):
            raise ValueError("annotation does not match image layout")
        unmixing = info.get("unmixing")
        if unmixing is not None:
            # JSON numbers preserve values; an explicit dtype preserves the
            # numeric precision of the basis used for reconstruction as well.
            unmixing = dict(unmixing)
            dtype = np.dtype(unmixing.pop("spectra_dtype"))
            raw = np.asarray(unmixing["spectra"])
            if dtype.kind not in "uif" or dtype.itemsize > 8 or raw.dtype.kind not in "uif":
                raise ValueError("invalid component spectra dtype")
            spectra = np.asarray(raw, dtype=dtype)
            if not np.array_equal(spectra, raw):
                raise ValueError("component spectra cannot be represented in the recorded dtype")
            unmixing["spectra"] = spectra
        # Reuse object validation for both component counts and basis metadata.
        # This placeholder allocates only one pixel per stored channel/component.
        validated = SpectralImage(np.empty((sizes["C"], 1, 1), dtype=np.uint8), channel_axis + "YX",
                                  kind=kind, component_names=info.get("component_names"),
                                  wavelengths_nm=info["wavelengths_nm"],
                                  detection_bands_nm=info["detection_bands_nm"], unmixing=unmixing)
        info["component_names"] = validated.component_names
        info["unmixing"] = validated.unmixing
        return info
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid Lumic OME annotation: {exc}") from exc


def save_ome_tiff(image, path, *, compression="deflate", bigtiff=None, overwrite=False):
    """Save a quantitative SpectralImage without rescaling its pixels.

    Use this to exchange simulated or experimental detector images with OME
    tools, or store a fitted reconstruction with its channel metadata. This is
    an in-memory, single-image exporter, not a streaming writer or an unmixing
    operation. Abundance maps are stored as ordinary OME C channels with
    component names. A Lumic annotation restores their K axes and optional exact
    component spectra, enabling abundance_image.reconstruct() after reloading.
    Display RGB renderings are not quantitative SpectralImage inputs.

    Parameters
    ----------
    image : SpectralImage
        Measured/reconstructed pixels with C/Y/X, or abundance maps with K/Y/X,
        plus optional T/Z axes in any order. Supported OME dtypes: uint8/16/32,
        int8/16/32, float32/64.
        Pixels, including negative floating values, are not clipped, normalized,
        quantized or changed. Floating NaN/Inf pixel values are stored as-is.
    path : str or Path
        Destination ending in .ome.tif or .ome.tiff; parent must exist.
    compression : {"deflate", "lzw", None}, default "deflate"
        Both codecs are lossless for supported integer and floating dtypes.
        None writes uncompressed pixels. File size reduction depends on data;
        compression does not remove noise or improve signal quality.
    bigtiff : bool or None, default None
        None chooses BigTIFF when the uncompressed data approaches the classic
        TIFF 4 GiB limit. True explicitly requests BigTIFF. False requests
        classic TIFF and is rejected for arrays above the conservative limit.
    overwrite : bool, default False
        Existing files are protected by default. Write to a temporary file in
        the destination directory, then publish atomically after success.
        A failed write leaves an existing destination unchanged.

    Returns
    -------
    pathlib.Path
        Destination path after successful publication.

    Metadata contract
    -----------------
    Standard OME metadata stores channel names (component_names for abundance
    maps, provenance['channel_names'] for detector images, when present) and
    physical pixel sizes in micrometres. TIFF storage puts Y/X
    last; no spatial resampling occurs. A versioned OME MapAnnotation preserves
    original axes, image kind, wavelength coordinates, detector
    bands and source provenance. Abundance annotations additionally preserve
    component names and optional unmixing metadata (exact K×C spectra, detector
    coordinates/names, normalization and abundance-unit labels). Labels are
    descriptive only; no normalization is applied. Missing spectra remain absent;
    such maps can be loaded but cannot reconstruct a detector signal. Detector
    coordinates are not falsely labeled as fluorophore emission wavelengths. Other OME readers can read the pixels
    and standard metadata, but need to interpret our annotation to recover
    Lumic-specific fields. Source files' complete vendor/OME XML is not copied.

    Lumic's reader restores the original axis order. As with other reader calls,
    load_ome_tiff(path) selects T=0/Z=0; use t=None, z=None for the entire image,
    including explicitly stored singleton T/Z axes.

    Provenance must be JSON-compatible; NumPy arrays/scalars and Paths are
    converted to lists/scalars/strings. Tuples return as lists. Unsupported or
    non-finite metadata raises an error before writing; no pickles are stored.

    Examples
    --------
    >>> save_ome_tiff(image, "simulation.ome.tiff")  # lossless Deflate
    >>> save_ome_tiff(image, "plain.ome.tif", compression=None)
    >>> restored = load_ome_tiff("simulation.ome.tiff", t=None, z=None)
    """
    import os
    import tempfile
    import tifffile
    from ome_types import from_xml, to_xml, model
    from .image import SpectralImage

    if not isinstance(image, SpectralImage):
        raise TypeError("image must be a SpectralImage; supply explicit axes when wrapping a NumPy array")
    # Arrays and metadata remain mutable after construction; validate their
    # current state before creating any output file.
    image = SpectralImage(image.data, image.axes, wavelengths_nm=image.wavelengths_nm,
                          detection_bands_nm=image.detection_bands_nm,
                          pixel_sizes_um=image.pixel_sizes_um, provenance=image.provenance,
                          kind=image.kind, component_names=image.component_names,
                          unmixing=image.unmixing)
    path = Path(path)
    if not path.name.lower().endswith((".ome.tif", ".ome.tiff")):
        raise ValueError("Destination must end in .ome.tif or .ome.tiff")
    if compression is not None and (not isinstance(compression, str)
                                    or compression not in ("deflate", "lzw")):
        raise ValueError("compression must be 'deflate', 'lzw', or None (all lossless)")
    if not isinstance(overwrite, bool) or (bigtiff is not None and not isinstance(bigtiff, bool)):
        raise TypeError("overwrite and bigtiff must be bool (bigtiff may also be None)")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {path}; use overwrite=True to replace it")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {path.parent}")
    if image.data.dtype.name not in ("uint8", "uint16", "uint32", "int8", "int16", "int32", "float32", "float64"):
        raise TypeError(f"Unsupported OME pixel dtype {image.data.dtype}; no automatic dtype conversion is performed")
    large = image.data.nbytes >= 2**32 - 2**25
    if bigtiff is False and large:
        raise ValueError("Image is too large for classic TIFF; use bigtiff=True or None")
    bigtiff = large if bigtiff is None else bigtiff
    source_order = "".join(a for a in image.axes if a not in "YX") + "YX"
    axes = source_order.replace("K", "C")  # K is never a standard OME dimension.
    data = image.data.transpose([image.axes.index(a) for a in source_order])
    channels = data.shape[axes.index("C")]
    # Source provenance may still contain detector labels from before fitting.
    # Only component_names describe the channels of an abundance TIFF.
    if image.kind == "abundance":
        names = image.component_names
        if names is None:
            names = tuple(f"Component {i + 1}" for i in range(channels))
    else:
        names = image.provenance.get("channel_names")
    if names is not None and (isinstance(names, str) or len(names) != channels
                              or any(n is not None and not isinstance(n, str) for n in names)):
        raise ValueError("provenance channel_names must contain one string or None per channel")
    metadata = {}
    for axis, size in image.pixel_sizes_um.items():
        metadata[f"PhysicalSize{axis}"] = size
        metadata[f"PhysicalSize{axis}Unit"] = "µm"
    # Build and schema-validate OME XML before writing. Channel names are set
    # through ome-types so missing names stay absent instead of becoming 'None'.
    xml = tifffile.OmeXml()
    pages = int(np.prod(data.shape[:-2]))
    xml.addimage(data.dtype, data.shape, (pages, 1, 1, data.shape[-2], data.shape[-1], 1),
                 axes=axes, **metadata)
    # Qt can change LC_NUMERIC after startup; libxml2 schema validation uses
    # it to parse decimal facets, so OME decimal points require the C locale.
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")
    ome = from_xml(xml.tostring(), validate=True)
    if names is not None:
        for channel, name in zip(ome.images[0].pixels.channels, names):
            channel.name = name
    info = dict(axes=image.axes, shape=list(image.data.shape), kind=image.kind,
                wavelengths_nm=image.wavelengths_nm, detection_bands_nm=image.detection_bands_nm,
                provenance=dict(image.provenance))
    namespace = _LUMIC_OME_NAMESPACE
    if image.kind == "abundance":
        namespace = _LUMIC_ABUNDANCE_OME_NAMESPACE
        info["component_names"] = image.component_names
        info["unmixing"] = None
        if image.unmixing is not None:
            info["unmixing"] = dict(image.unmixing)
            info["unmixing"]["spectra_dtype"] = image.unmixing["spectra"].dtype.str
    try:
        encoded = json.dumps(info, default=_ome_json_value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Image metadata must be JSON-compatible: {exc}") from exc
    annotation = model.MapAnnotation(
        id="Annotation:Lumic:0", namespace=namespace,
        value=model.Map(ms=[model.Map.M(k="spectral_image", value=encoded)]))
    ome.structured_annotations.map_annotations.append(annotation)
    ome.images[0].annotation_refs.append(model.AnnotationRef(id=annotation.id))
    description = to_xml(ome, validate=True).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        tifffile.imwrite(temporary, data, kind="generic", metadata=None,
                         description=description, photometric="minisblack",
                         compression=compression, bigtiff=bigtiff, software="Lumic")
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic exclusive creation also protects against a destination
            # appearing while the image is being compressed.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path



def open_image(path, *, series=None) -> SpectralImageReader:
    """Inspect a supported image using its existing format-specific reader.

    Dispatch .czi to open_czi and .tif/.tiff to open_ome_tiff. TIFF inputs
    must contain supported OME metadata; a plain TIFF is not guessed to be
    spectral. Filename suffixes select the adapter, which validates contents
    and layout without decoding pixels. Unsupported extensions, invalid files
    and unsupported layouts raise clear errors rather than returning False.

    The returned object IS the format-specific reader, not a wrapper. Its
    summary(), axes, shape and optional metadata can be inspected before
    read(t=0, z=0) explicitly loads pixels. Use a context manager to close it.
    Summary support does not establish suitability for spectral unmixing.

    series=None selects the default first OME image. An explicit series index
    is an OME-TIFF-only option and raises ValueError for CZI, including zero;
    this API does not silently interpret it as CZI scene selection. Use the
    format-specific functions directly whenever their explicit names are useful.
    No format-specific function's signature or behavior is changed.

    Example: with open_image(path) as reader: print(reader.summary()).
    Opening and displaying the summary does not prompt or load pixel arrays.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".czi":
        if series is not None:
            raise ValueError("series selection is an OME-TIFF-only option; CZI scene selection is not supported")
        return open_czi(path)
    if suffix in (".tif", ".tiff"):
        return open_ome_tiff(path, series=0 if series is None else series)
    raise ValueError(f"Unsupported image extension {suffix!r}; expected .czi or OME-TIFF (.tif/.tiff)")


def load_image(path, *, t=0, z=0, series=None):
    """Load a supported CZI or OME-TIFF into SpectralImage, then close its reader.

    Convenience form of open_image(path, series=series).read(t=t, z=z).
    Defaults to the first T/Z plane; use None to retain entire dimensions.
    Axis order, dtype, metadata and validation follow the selected adapter.
    Existing load_czi and load_ome_tiff calls remain available unchanged.
    For inspection without pixel loading, use open_image and summary() first.
    """
    with open_image(path, series=series) as reader:
        return reader.read(t=t, z=z)
