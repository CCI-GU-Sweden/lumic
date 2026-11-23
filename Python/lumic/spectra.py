"""An optional container for a single spectral signature."""

from collections.abc import Mapping
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

    def _to_json_payload(self):
        """Build and validate the version 1 JSON representation."""
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
        return payload

    def save_json(self, path):
        """Save a version 1 Lumic spectrum to a UTF-8 JSON file.

        Preserve sample order, scale, and numeric dtypes up to 64 bits.
        Provenance must contain JSON objects with string keys, lists, strings,
        booleans, finite Python numbers or None; unsupported objects raise
        TypeError. Serialization is validated before opening the destination.
        An existing file is overwritten. Parent directories must exist.

        Example: ``spectrum.save_json("reference.spectrum.json")``.
        """
        payload = self._to_json_payload()
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


def save_spectra_json(spectra, path):
    """Save multiple Spectrum objects together in one Lumic JSON file.

    The version 1 ``lumic-spectra`` collection format is compatible with
    collection files already read by the napari plugin. Individual
    ``Spectrum.save_json()`` files and methods are unchanged. At least one
    spectrum is required. Existing destinations are overwritten; parent
    directories must already exist.

    Example: ``save_spectra_json([egfp, mcherry], "references.json")``.
    """
    try:
        spectra = list(spectra)
    except TypeError as exc:
        raise ValueError("spectra must contain at least one Spectrum object") from exc
    if not spectra or any(not isinstance(spectrum, Spectrum) for spectrum in spectra):
        raise ValueError("spectra must contain at least one Spectrum object")

    payload = {
        "format": "lumic-spectra",
        "version": 1,
        "spectra": [spectrum._to_json_payload() for spectrum in spectra],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_spectra_json(path):
    """Load a collection JSON file or a legacy single-spectrum JSON file.

    Always returns a list of ``Spectrum`` objects, including for a single
    spectrum file. Collection files use the same version 1 format as the
    napari plugin.

    Example: ``references = load_spectra_json("references.json")``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") == "lumic-spectrum":
        return [Spectrum._from_json_payload(payload)]
    if (not isinstance(payload, dict) or set(payload) != {"format", "version", "spectra"}
            or payload["format"] != "lumic-spectra"
            or type(payload["version"]) is not int or payload["version"] != 1
            or not isinstance(payload["spectra"], list) or not payload["spectra"]):
        raise ValueError("Expected a Lumic spectrum or version 1 spectra collection")
    return [Spectrum._from_json_payload(component) for component in payload["spectra"]]
