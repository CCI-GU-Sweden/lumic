import json

import numpy as np
import pytest

from lumic import Spectrum


def test_spectrum_preserves_values_and_wavelength_order():
    values = np.array([0.25, -0.125, 0.5], dtype=np.float32)
    wavelengths = np.array([480., 460., 510.])
    provenance = {"format": "ZEN XML"}
    spectrum = Spectrum(values, wavelengths, name="Reference", provenance=provenance)
    values[:] = 0
    wavelengths[:] = 0
    provenance.clear()

    np.testing.assert_array_equal(spectrum.values, [0.25, -0.125, 0.5])
    np.testing.assert_array_equal(spectrum.wavelengths_nm, [480., 460., 510.])
    assert spectrum.values.dtype == np.float32
    assert spectrum.name == "Reference"
    assert spectrum.provenance == {"format": "ZEN XML"}


def test_spectrum_accepts_a_signature_without_metadata():
    spectrum = Spectrum([4, 2])

    np.testing.assert_array_equal(spectrum.values, [4, 2])
    assert spectrum.wavelengths_nm is None
    assert spectrum.name is None


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [1 + 2j]])
def test_spectrum_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="values"):
        Spectrum(values)


@pytest.mark.parametrize("wavelengths", [[480], [480, np.inf], [480, 0]])
def test_spectrum_rejects_invalid_wavelengths(wavelengths):
    with pytest.raises(ValueError, match="wavelengths_nm"):
        Spectrum([1, 2], wavelengths_nm=wavelengths)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", ">f8", "int16", "uint64"])
def test_spectrum_json_round_trip(tmp_path, dtype):
    values = np.array([3, 1, 2], dtype=dtype)
    if values.dtype.kind == "f":
        values /= 7
    elif values.dtype == np.dtype("uint64"):
        values[0] = np.iinfo(np.uint64).max
    wavelengths = np.array([510, 480, 495], dtype=">f4")
    provenance = {"format": "ZEN XML", "acquisition": {"lasers": [405, 488],
                  "normalized": False, "note": None, "power": 0.125}}
    original = Spectrum(values, wavelengths, name="Référence", provenance=provenance)
    path = tmp_path / "reference.spectrum.json"

    original.save_json(path)
    restored = Spectrum.load_json(path)

    np.testing.assert_array_equal(restored.values, values)
    np.testing.assert_array_equal(restored.wavelengths_nm, wavelengths)
    assert restored.values.dtype == values.dtype
    assert restored.wavelengths_nm.dtype == wavelengths.dtype
    assert restored.name == original.name
    assert restored.provenance == provenance
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_spectrum_json_without_optional_metadata(tmp_path):
    path = tmp_path / "minimal.json"
    Spectrum([4, 2]).save_json(str(path))

    restored = Spectrum.load_json(str(path))

    np.testing.assert_array_equal(restored.values, [4, 2])
    assert restored.wavelengths_nm is None
    assert restored.name is None
    assert restored.provenance == {}


@pytest.mark.parametrize("metadata", [
    {1: "non-string key"}, {"nested": {1: "non-string key"}},
    {"tuple": (1, 2)}, {"array": np.array([1])},
    {"bad": float("nan")}, {"nested": [float("inf")]},
])
def test_spectrum_json_rejects_unsupported_metadata_before_overwriting(tmp_path, metadata):
    path = tmp_path / "existing.json"
    path.write_text("existing content", encoding="utf-8")

    with pytest.raises(TypeError, match="provenance"):
        Spectrum([1], provenance=metadata).save_json(path)

    assert path.read_text(encoding="utf-8") == "existing content"


@pytest.mark.parametrize(("field", "value"), [
    ("format", "another-format"), ("version", 2), ("version", True),
    ("name", 3), ("provenance", []), ("provenance", {"bad": float("nan")}),
    ("values", []), ("values", [[1, 2]]), ("values", [True]),
    ("values", ["1"]), ("values", [float("nan")]),
    ("values", [1e300]), ("values_dtype", "object"),
    ("values_dtype", "complex128"), ("values_dtype", "invalid"),
    ("values_dtype", None), ("wavelengths_nm", [480]),
    ("wavelengths_nm", [480, 0]), ("wavelengths_dtype", None),
])
def test_spectrum_json_rejects_invalid_payload(tmp_path, field, value):
    path = tmp_path / "invalid.json"
    Spectrum(np.array([1, 2], dtype=np.float32), [480, 510]).save_json(path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        Spectrum.load_json(path)


@pytest.mark.parametrize("samples", [[256], [-1], [1.5]])
def test_spectrum_json_rejects_integer_overflow_and_truncation(tmp_path, samples):
    path = tmp_path / "invalid.json"
    Spectrum(np.array([1], dtype=np.uint8)).save_json(path)
    payload = json.loads(path.read_text())
    payload["values"] = samples
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="integer dtype"):
        Spectrum.load_json(path)


@pytest.mark.parametrize("content", ["{", "[]", '{}', '{"format": "lumic-spectrum"}'])
def test_spectrum_json_rejects_malformed_or_incomplete_files(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        Spectrum.load_json(path)


def test_spectrum_json_rejects_mutated_samples_before_overwriting(tmp_path):
    spectrum = Spectrum([1.0])
    spectrum.values[0] = np.nan
    path = tmp_path / "existing.json"
    path.write_text("existing content")

    with pytest.raises(ValueError, match="finite"):
        spectrum.save_json(path)

    assert path.read_text() == "existing content"


def test_spectrum_json_rejects_extended_precision(tmp_path):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("No extended precision dtype on this platform")
    with pytest.raises(ValueError, match="64 bits"):
        Spectrum(np.array([1], dtype=np.longdouble)).save_json(tmp_path / "extended.json")
