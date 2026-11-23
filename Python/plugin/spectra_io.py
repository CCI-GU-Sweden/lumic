"""Single-file reference collections built from the backend Spectrum format."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lumic import Spectrum


def save_spectra_json(spectra, path):
    """Save every component in one JSON collection, using backend validation."""
    if not spectra:
        raise ValueError("At least one spectrum is required")
    payloads = []
    # The backend's public serialization API accepts paths, not dictionaries.
    # Bridge it here so the plugin does not duplicate its schema or validation.
    with TemporaryDirectory(prefix="lumic-spectra-") as directory:
        temporary = Path(directory) / "spectrum.json"
        for spectrum in spectra:
            spectrum.save_json(temporary)
            payloads.append(json.loads(temporary.read_text(encoding="utf-8")))
    text = json.dumps({"format": "lumic-spectra", "version": 1, "spectra": payloads},
                      indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def load_spectra_json(path):
    """Read a plugin collection or a single backend spectrum JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("format") == "lumic-spectrum":
        return [Spectrum.load_json(path)]
    if (not isinstance(payload, dict) or set(payload) != {"format", "version", "spectra"}
            or payload["format"] != "lumic-spectra"
            or type(payload["version"]) is not int or payload["version"] != 1
            or not isinstance(payload["spectra"], list) or not payload["spectra"]):
        raise ValueError("Expected a Lumic spectrum or version 1 spectra collection")
    spectra = []
    with TemporaryDirectory(prefix="lumic-spectra-") as directory:
        temporary = Path(directory) / "spectrum.json"
        for component in payload["spectra"]:
            temporary.write_text(json.dumps(component), encoding="utf-8")
            spectra.append(Spectrum.load_json(temporary))
    return spectra


def validate_reference_spectra(spectra, image):
    """Require matching detector coordinates; never resample imported values."""
    channels = image.data.shape[0]
    if not 1 <= len(spectra) <= min(64, channels):
        raise ValueError("The component count must be between 1 and the channel count (at most 64)")
    for spectrum in spectra:
        if len(spectrum.values) != channels:
            raise ValueError("Spectrum channel count does not match the image")
        total = spectrum.values.astype(float).sum()
        if np.any(spectrum.values < 0) or not np.isfinite(total) or total <= 0:
            raise ValueError("References must have nonnegative values and positive finite total signal")
        if (spectrum.wavelengths_nm is None) != (image.wavelengths_nm is None):
            raise ValueError("Spectrum and image must both have wavelength coordinates, or both use channel order")
        if spectrum.wavelengths_nm is not None and not np.allclose(
                spectrum.wavelengths_nm, image.wavelengths_nm, rtol=0, atol=1e-6):
            raise ValueError("Spectrum wavelengths or channel order do not match the image")
        bands = spectrum.provenance.get("detection_bands_nm")
        if bands is not None:
            bands = np.asarray(bands, dtype=float)
            if (image.detection_bands_nm is None or bands.shape != image.detection_bands_nm.shape
                    or not np.allclose(bands, image.detection_bands_nm, rtol=0, atol=1e-6)):
                raise ValueError("Spectrum detection bands do not match the image")


def spectrum_objects(matrix, image, path, plane, stage, mode, references=()):
    """Attach source information without scaling or reordering solver arrays."""
    result = []
    for index, values in enumerate(matrix):
        reference = references[index] if index < len(references) else None
        provenance = {"image_path": str(path)}
        if image.pixel_sizes_um:
            # Readers normalize physical calibration to micrometres. Do not
            # invent a physical size or unit when acquisition metadata lacks it.
            provenance["pixel_sizes_um"] = {axis: float(size) for axis, size in image.pixel_sizes_um.items()}
            provenance["pixel_unit"] = "micrometer"
        provenance.update({"time_index": plane[0], "z_index": plane[1],
                           "stage": stage, "reference_mode": mode,
                           "detection_bands_nm": (None if image.detection_bands_nm is None
                                                  else image.detection_bands_nm.tolist())})
        if reference is not None:
            provenance["source_spectrum"] = dict(reference.provenance)
        result.append(Spectrum(values, image.wavelengths_nm,
                               name=reference.name if reference is not None else f"Component {index + 1}",
                               provenance=provenance))
    return result
