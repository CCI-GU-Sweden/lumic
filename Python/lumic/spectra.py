"""Containers for individual spectral signatures and shared sample axes."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


def _validate_json_metadata(value):
    """Reject metadata that JSON would silently change or cannot represent."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and np.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json_metadata(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json_metadata(item)
        return
    raise TypeError("provenance must contain only JSON-compatible values with string keys and finite numbers")


def _spectrum_json_array(values, dtype, field_name):
    """Validate numeric JSON samples before NumPy can coerce or wrap them."""
    try:
        dtype = np.dtype(dtype) if isinstance(dtype, str) else None
    except (TypeError, ValueError):
        dtype = None
    if dtype is None or dtype.kind not in "uif" or dtype.itemsize > 8:
        raise ValueError(f"{field_name}_dtype must describe a real integer or float dtype of at most 64 bits")
    if not isinstance(values, list) or not values or any(type(v) not in (int, float) for v in values):
        raise ValueError(f"{field_name} must be a nonempty array of real JSON numbers")
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        if any(type(v) is not int or not limits.min <= v <= limits.max for v in values):
            raise ValueError(f"{field_name} contains values outside its integer dtype")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} cannot be represented by its dtype") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be finite and representable by its dtype")
    return array


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Keep a spectrum's values, wavelength labels and source information together.

    `values` is a nonempty, finite, one-dimensional array of real intensities.
    Optional `wavelengths_nm` contains one finite, positive coordinate per
    value, in the same order. Use `name` for the label and `provenance` for
    descriptive information about acquisition or processing.

    Arrays are copied at construction. Values retain their dtype, order and
    scale. Wavelength labels alone do not establish compatibility with an
    image's detector channels. Pass ordinary NumPy arrays to the solvers.
    """

    values: np.ndarray
    wavelengths_nm: np.ndarray | None = None
    name: str | None = None
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "uif":
            raise ValueError("values must be a nonempty one-dimensional array of real numbers")
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        object.__setattr__(self, "values", values.copy())

        if self.wavelengths_nm is not None:
            wavelengths = np.asarray(self.wavelengths_nm)
            if (wavelengths.shape != values.shape or wavelengths.dtype.kind not in "uif"
                    or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0)):
                raise ValueError("wavelengths_nm must contain one finite positive number per value")
            object.__setattr__(self, "wavelengths_nm", wavelengths.copy())

        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def save_json(self, path):
        """Save a version 1 Lumic spectrum to a UTF-8 JSON file.

        Preserve sample order, scale, and numeric dtypes up to 64 bits.
        Provenance must contain JSON objects with string keys, lists, strings,
        booleans, finite Python numbers or None; unsupported objects raise
        TypeError. Serialization is validated before opening the destination.
        An existing file is overwritten. Parent directories must exist.

        Example: ``spectrum.save_json("reference.spectrum.json")``.
        """
        _validate_json_metadata(self.provenance)
        payload = {"format": "lumic-spectrum", "version": 1,
                   "name": self.name, "provenance": self.provenance}
        for field_name in ("values", "wavelengths_nm"):
            array = getattr(self, field_name)
            dtype_field = "values_dtype" if field_name == "values" else "wavelengths_dtype"
            if array is None:
                payload[field_name] = payload[dtype_field] = None
            else:
                if array.dtype.kind not in "uif" or array.dtype.itemsize > 8:
                    raise ValueError(f"{field_name} JSON serialization supports real numeric dtypes up to 64 bits")
                payload[field_name] = array.tolist()
                payload[dtype_field] = array.dtype.str
        # Revalidate: frozen dataclasses still allow mutation of their arrays.
        type(self)._from_json_payload(payload)
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path):
        """Read a JSON spectrum written by :meth:`save_json`.

        Validate the format version, numeric dtypes, samples and metadata.
        Missing wavelength coordinates remain None. No normalization,
        interpolation, or sorting is performed. Invalid files raise ValueError
        (including JSONDecodeError); file access errors propagate normally.

        Example: ``spectrum = Spectrum.load_json("reference.spectrum.json")``.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_json_payload(payload)

    @classmethod
    def _from_json_payload(cls, payload):
        fields = {"format", "version", "name", "provenance", "values",
                  "values_dtype", "wavelengths_nm", "wavelengths_dtype"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("Expected a complete Lumic spectrum JSON object")
        if (payload["format"] != "lumic-spectrum"
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Unsupported Lumic spectrum format or version")
        if payload["name"] is not None and not isinstance(payload["name"], str):
            raise ValueError("name must be a string or null")
        if not isinstance(payload["provenance"], dict):
            raise ValueError("provenance must be a JSON object")
        try:
            _validate_json_metadata(payload["provenance"])
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        values = _spectrum_json_array(payload["values"], payload["values_dtype"], "values")
        wavelengths = None
        if payload["wavelengths_nm"] is None:
            if payload["wavelengths_dtype"] is not None:
                raise ValueError("wavelengths_dtype must be null when wavelengths_nm is null")
        else:
            wavelengths = _spectrum_json_array(payload["wavelengths_nm"],
                                               payload["wavelengths_dtype"], "wavelengths")
        return cls(values, wavelengths, name=payload["name"], provenance=payload["provenance"])


def _readonly_array(array):
    """Copy a numeric array to a buffer that cannot be made writable."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _collection_json_array(payload, field_name, *, matrix=False):
    """Read one dtype-and-data object from a collection file."""
    if not isinstance(payload, dict) or set(payload) != {"dtype", "data"}:
        raise ValueError(f"{field_name} must contain dtype and data")
    data = payload["data"]
    if not matrix:
        return _spectrum_json_array(data, payload["dtype"], field_name)

    if (not isinstance(data, list) or not data
            or not isinstance(data[0], list) or not data[0]
            or any(not isinstance(row, list) or len(row) != len(data[0]) for row in data)):
        raise ValueError(f"{field_name} must contain a nonempty rectangular matrix")
    flat = _spectrum_json_array([value for row in data for value in row], payload["dtype"], field_name)
    return flat.reshape(len(data), len(data[0]))


def _sample_axis(values, shape, name):
    if values is None:
        return None
    array = np.asarray(values)
    if (array.shape != shape or array.dtype.kind not in "uif"
            or not np.isfinite(array).all() or np.any(array <= 0)):
        raise ValueError(f"{name} must have shape {shape} and finite positive values")
    return array


def _detection_bands(values, sample_count):
    bands = _sample_axis(values, (sample_count, 2), "detection_bands_nm")
    if bands is not None and np.any(bands[:, 1] < bands[:, 0]):
        raise ValueError("detection_bands_nm upper bounds must be >= lower bounds")
    return bands


def _same_axis(first, second):
    if first is None or second is None:
        return first is second
    if not np.array_equal(first, second):
        return False
    # NumPy comparisons may promote mixed integer/float arrays and round labels.
    with np.errstate(invalid="ignore", over="ignore"):
        return (np.array_equal(first, second.astype(first.dtype, copy=False))
                and np.array_equal(second, first.astype(second.dtype, copy=False)))


@dataclass(frozen=True, eq=False, init=False)
class SpectrumCollection:
    """Ordered component spectra on a common sample axis.

    Rows in `values` represent components and columns represent samples.
    The matrix must be nonempty, real and finite. Optional `wavelengths_nm`
    has one positive coordinate per sample. Optional `detection_bands_nm`
    has one pair of positive lower and upper bounds per sample. Each axis
    follows the matrix's column order, including any repeated coordinates.
    Arrays are copied and exposed as read-only views.

    `names` and `row_provenance` contain one name (or None) and one mapping
    per row, respectively. `provenance` describes the whole collection.
    Detector bands in row provenance must agree exactly in sample order and
    must occur in every row or none. When present, they provide the shared
    detector bands and must match any explicitly supplied bands. The reserved
    `detection_bands_nm` key is stored as the shared axis and exported in the
    provenance of each extracted Spectrum.
    """

    _values: np.ndarray = field(repr=False)
    _wavelengths_nm: np.ndarray | None = field(repr=False)
    _detection_bands_nm: np.ndarray | None = field(repr=False)
    names: tuple[str | None, ...]
    _row_provenance: tuple[Mapping, ...] = field(repr=False)
    provenance: Mapping

    def __init__(self, values, wavelengths_nm=None, *, detection_bands_nm=None,
                 names=None, row_provenance=None, provenance=None):
        matrix = np.asarray(values)
        if matrix.ndim != 2 or not matrix.size or matrix.dtype.kind not in "uif":
            raise ValueError("values must be a nonempty two-dimensional array of real numbers")
        if not np.isfinite(matrix).all():
            raise ValueError("values must be finite")
        row_count, sample_count = matrix.shape
        wavelengths = _sample_axis(wavelengths_nm, (sample_count,), "wavelengths_nm")
        bands = _detection_bands(detection_bands_nm, sample_count)

        if isinstance(names, str):
            raise TypeError("names must be a sequence of names")
        names = (None,) * row_count if names is None else tuple(names)
        if len(names) != row_count:
            raise ValueError("names must contain one name or None per row")
        if any(name is not None and not isinstance(name, str) for name in names):
            raise TypeError("names must contain only strings or None")

        if row_provenance is None:
            row_provenance = tuple({} for _ in range(row_count))
        else:
            row_provenance = tuple(row_provenance)
            if any(not isinstance(row, Mapping) for row in row_provenance):
                raise TypeError("row_provenance must contain only mappings")
            row_provenance = tuple(deepcopy(dict(row)) for row in row_provenance)
        if len(row_provenance) != row_count:
            raise ValueError("row_provenance must contain one mapping per row")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        row_bands = [_detection_bands(row.pop("detection_bands_nm", None), sample_count)
                     for row in row_provenance]
        if any(not _same_axis(row_bands[0], axis) for axis in row_bands[1:]):
            raise ValueError("row detection_bands_nm must all match or all be absent")
        if row_bands[0] is not None:
            if bands is not None and not _same_axis(bands, row_bands[0]):
                raise ValueError("detection_bands_nm must match the row detection bands")
            if bands is None:
                bands = row_bands[0]

        object.__setattr__(self, "_values", _readonly_array(matrix))
        object.__setattr__(self, "_wavelengths_nm", None if wavelengths is None else _readonly_array(wavelengths))
        object.__setattr__(self, "_detection_bands_nm", None if bands is None else _readonly_array(bands))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "_row_provenance", row_provenance)
        object.__setattr__(self, "provenance", {} if provenance is None else deepcopy(dict(provenance)))

    @property
    def values(self):
        """The component-by-sample matrix."""
        return self._values.view()

    @property
    def wavelengths_nm(self):
        """The shared wavelength coordinate array, if available."""
        return None if self._wavelengths_nm is None else self._wavelengths_nm.view()

    @property
    def detection_bands_nm(self):
        """The shared detector bounds, if available."""
        return None if self._detection_bands_nm is None else self._detection_bands_nm.view()

    @property
    def row_provenance(self):
        """Independent copies of the descriptive metadata for each row."""
        return deepcopy(self._row_provenance)

    def save_json(self, path):
        """Save a version 1 Lumic spectrum collection as UTF-8 JSON.

        Preserve row and sample order, scale, and numeric dtypes up to
        64 bits. Store the optional sample axes once for the collection.
        Collection and row provenance must contain only JSON-compatible
        values. Validate serialization before overwriting an existing file.
        """
        _validate_json_metadata(self.provenance)
        components = []
        for name, provenance in zip(self.names, self._row_provenance):
            _validate_json_metadata(provenance)
            components.append({"name": name, "provenance": provenance})

        payload = {
            "format": "lumic-spectrum-collection",
            "version": 1,
        }
        for field_name, array in (
            ("values", self._values),
            ("wavelengths_nm", self._wavelengths_nm),
            ("detection_bands_nm", self._detection_bands_nm),
        ):
            if array is None:
                payload[field_name] = None
                continue
            if array.dtype.kind not in "uif" or array.dtype.itemsize > 8:
                raise ValueError(f"{field_name} JSON serialization supports real numeric dtypes up to 64 bits")
            payload[field_name] = {"dtype": array.dtype.str, "data": array.tolist()}

        payload["components"] = components
        payload["provenance"] = self.provenance
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        Path(path).write_bytes((text + "\n").encode("utf-8"))

    @classmethod
    def load_json(cls, path):
        """Read a version 1 file written by :meth:`save_json`.

        Preserve component order, sample order, and recorded numeric dtypes.
        Invalid files raise ValueError. File access errors propagate normally.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {"format", "version", "values", "wavelengths_nm",
                  "detection_bands_nm", "components", "provenance"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("Expected a complete Lumic spectrum collection JSON object")
        if (payload["format"] != "lumic-spectrum-collection"
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Unsupported Lumic spectrum collection format or version")

        values = _collection_json_array(payload["values"], "values", matrix=True)
        wavelengths = None
        if payload["wavelengths_nm"] is not None:
            wavelengths = _collection_json_array(payload["wavelengths_nm"], "wavelengths_nm")
        bands = None
        if payload["detection_bands_nm"] is not None:
            bands = _collection_json_array(payload["detection_bands_nm"], "detection_bands_nm", matrix=True)

        components = payload["components"]
        if not isinstance(components, list) or len(components) != values.shape[0]:
            raise ValueError("components must contain one object per values row")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        names = []
        row_provenance = []
        for component in components:
            if not isinstance(component, dict) or set(component) != {"name", "provenance"}:
                raise ValueError("each component must contain a name and provenance")
            if component["name"] is not None and not isinstance(component["name"], str):
                raise ValueError("component name must be a string or null")
            row = component["provenance"]
            if not isinstance(row, dict):
                raise ValueError("component provenance must be a JSON object")
            if "detection_bands_nm" in row:
                raise ValueError("detection_bands_nm must be stored as a shared axis")
            names.append(component["name"])
            row_provenance.append(row)
        try:
            _validate_json_metadata(provenance)
            for row in row_provenance:
                _validate_json_metadata(row)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return cls(values, wavelengths, detection_bands_nm=bands, names=names,
                   row_provenance=row_provenance, provenance=provenance)

    @classmethod
    def from_spectra(cls, spectra, *, detection_bands_nm=None, provenance=None):
        """Stack Spectrum objects in iteration order, using a common dtype.

        Sample counts and wavelength labels must agree exactly in order.
        All rows may omit wavelengths. Stacking raises ValueError if dtype
        promotion would change a sample. No samples are sorted, normalized
        or interpolated.
        """
        spectra = tuple(spectra)
        if not spectra:
            raise ValueError("at least one Spectrum is required")
        if any(not isinstance(spectrum, Spectrum) for spectrum in spectra):
            raise TypeError("spectra must contain Spectrum objects")

        shape = spectra[0].values.shape
        wavelengths = _sample_axis(spectra[0].wavelengths_nm, shape, "wavelengths_nm")
        for spectrum in spectra[1:]:
            if spectrum.values.ndim != 1 or spectrum.values.shape != shape:
                raise ValueError("Spectrum sample counts must match")
            axis = _sample_axis(spectrum.wavelengths_nm, shape, "wavelengths_nm")
            if not _same_axis(wavelengths, axis):
                raise ValueError("Spectrum wavelengths_nm must all match in order or all be absent")

        values = np.stack([spectrum.values for spectrum in spectra])
        with np.errstate(invalid="ignore", over="ignore"):
            for row, spectrum in zip(values, spectra):
                if not np.array_equal(spectrum.values, row.astype(spectrum.values.dtype, copy=False)):
                    raise ValueError("Spectrum values cannot be stacked without changing samples")
        return cls(values, wavelengths, names=[spectrum.name for spectrum in spectra],
                   row_provenance=[spectrum.provenance for spectrum in spectra],
                   detection_bands_nm=detection_bands_nm, provenance=provenance)

    def __len__(self):
        return len(self.names)

    def spectrum(self, index):
        """Return an independent Spectrum at an integer row index.

        Shared detector bands are carried in the Spectrum's provenance.
        """
        name = self.names[index]
        provenance = deepcopy(self._row_provenance[index])
        if self._detection_bands_nm is not None:
            provenance["detection_bands_nm"] = self._detection_bands_nm.tolist()
        return Spectrum(self._values[index], self._wavelengths_nm, name=name, provenance=provenance)
