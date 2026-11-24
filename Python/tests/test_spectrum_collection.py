import json

import numpy as np
import pytest

from examples.spectrum_collection_workflow import run_workflow
from lumic import Spectrum, SpectrumCollection
from lumic.simulation import SpectralDetector


def test_collection_preserves_order_values_and_metadata():
    bands = [[500, 520], [475, 480], [500, 520]]
    spectra = [
        Spectrum(np.array([5, -2, 0], dtype=np.float32), [510, 478, 510],
                 name="first", provenance={"detection_bands_nm": bands, "source": ["reference"]}),
        Spectrum(np.array([0, 0, 0], dtype=np.float32), [510, 478, 510],
                 provenance={"detection_bands_nm": bands}),
    ]

    collection = SpectrumCollection.from_spectra(iter(spectra), provenance={"source": "session"})
    extracted = collection.spectrum(0)
    np.testing.assert_array_equal(collection.values, [[5, -2, 0], [0, 0, 0]])
    np.testing.assert_array_equal(collection.wavelengths_nm, [510, 478, 510])
    np.testing.assert_array_equal(collection.detection_bands_nm, bands)
    assert collection.values.dtype == np.float32
    assert collection.names == ("first", None)
    assert collection.row_provenance == ({"source": ["reference"]}, {})
    assert len(collection) == 2
    assert collection.provenance == {"source": "session"}
    assert extracted.name == "first"
    assert extracted.provenance == spectra[0].provenance
    np.testing.assert_array_equal(extracted.wavelengths_nm, spectra[0].wavelengths_nm)
    np.testing.assert_array_equal(collection.spectrum(-1).values, [0, 0, 0])

    spectra[0].values[:] = 10
    spectra[0].provenance["source"].append("changed")
    extracted.values[:] = 20
    extracted.provenance["detection_bands_nm"][0][0] = 1
    collection.row_provenance[0]["source"].clear()
    np.testing.assert_array_equal(collection.values[0], [5, -2, 0])
    assert collection.spectrum(0).provenance["source"] == ["reference"]
    np.testing.assert_array_equal(collection.detection_bands_nm, bands)


def test_collection_owns_and_protects_validated_arrays():
    values = np.array([[2, 1]])
    wavelengths = np.array([510, 478])
    bands = np.array([[500, 520], [475, 480]])
    collection = SpectrumCollection(values, wavelengths, detection_bands_nm=bands)

    for source, array in zip((values, wavelengths, bands),
                             (collection.values, collection.wavelengths_nm, collection.detection_bands_nm)):
        expected = source.copy()
        source[:] = 0
        np.testing.assert_array_equal(array, expected)
        with pytest.raises(ValueError):
            array.flat[0] = 99
        with pytest.raises(ValueError):
            array.setflags(write=True)

    assert collection.values is not collection.values
    assert collection.spectrum(0).provenance["detection_bands_nm"] == [[500, 520], [475, 480]]


@pytest.mark.parametrize("values", [[], [[]], [1], [[np.nan]], [[np.inf]], [[1j]]])
def test_collection_rejects_invalid_matrices(values):
    with pytest.raises(ValueError, match="values"):
        SpectrumCollection(values)


@pytest.mark.parametrize(("kwargs", "field"), [
    ({"wavelengths_nm": [0]}, "wavelengths_nm"),
    ({"wavelengths_nm": [510, 500]}, "wavelengths_nm"),
    ({"detection_bands_nm": [[np.inf, 520]]}, "detection_bands_nm"),
    ({"detection_bands_nm": [[520, 500]]}, "detection_bands_nm"),
    ({"detection_bands_nm": [500, 520]}, "detection_bands_nm"),
    ({"names": []}, "names"),
    ({"row_provenance": []}, "row_provenance"),
])
def test_collection_checks_axis_and_row_metadata_shapes(kwargs, field):
    with pytest.raises(ValueError, match=field):
        SpectrumCollection([[1]], **kwargs)


@pytest.mark.parametrize("kwargs", [{"row_provenance": [[]]}, {"provenance": []}])
def test_collection_requires_provenance_mappings(kwargs):
    with pytest.raises(TypeError, match="provenance"):
        SpectrumCollection([[1]], **kwargs)


@pytest.mark.parametrize("second_axis", [None, [510], [510 + 1e-8, 500], [500, 510]])
def test_collection_requires_matching_wavelength_axes(second_axis):
    second = Spectrum([1] if second_axis == [510] else [1, 2], second_axis)
    with pytest.raises(ValueError, match="sample counts|wavelengths_nm"):
        SpectrumCollection.from_spectra([Spectrum([2, 1], [510, 500]), second])


def test_collection_requires_matching_detection_bands():
    first = Spectrum([2], provenance={"detection_bands_nm": [[500, 520]]})
    for second in [Spectrum([1]), Spectrum([1], provenance={"detection_bands_nm": [[500, 521]]})]:
        with pytest.raises(ValueError, match="detection_bands_nm"):
            SpectrumCollection.from_spectra([first, second])
    with pytest.raises(ValueError, match="detection_bands_nm"):
        SpectrumCollection.from_spectra([first], detection_bands_nm=[[501, 520]])


def test_collection_supports_unlabeled_rows_but_rejects_lossy_stacking():
    collection = SpectrumCollection.from_spectra([Spectrum([1]), Spectrum([2.5])])
    assert collection.wavelengths_nm is None
    assert collection.detection_bands_nm is None
    np.testing.assert_array_equal(collection.values, [[1], [2.5]])

    with pytest.raises(ValueError, match="without changing samples"):
        SpectrumCollection.from_spectra([Spectrum([2**53 + 1]), Spectrum([2.5])])


def test_collection_json_preserves_order_dtypes_and_provenance(tmp_path):
    wavelengths = np.array([510.5, 478, 510.5], dtype=">f4")
    bands = [[500, 520], [475, 480], [500, 520]]
    values = np.array([[2**64 - 1, 0, 3], [1, 2, 0]], dtype=">u8")
    collection = SpectrumCollection(values, wavelengths, detection_bands_nm=np.array(bands, dtype="<u2"),
                                    names=["Référence", None],
                                    row_provenance=[
                                        {"detection_bands_nm": bands, "source": ["raw", None]},
                                        {"detection_bands_nm": bands},
                                    ],
                                    provenance={"normalized": False})
    path = tmp_path / "references.json"

    collection.save_json(path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith('{\n  "format"')
    assert text.endswith("\n")
    assert "Référence" in text
    assert json.loads(text) == {
        "format": "lumic-spectrum-collection",
        "version": 1,
        "values": {"dtype": ">u8", "data": [[2**64 - 1, 0, 3], [1, 2, 0]]},
        "wavelengths_nm": {"dtype": ">f4", "data": [510.5, 478, 510.5]},
        "detection_bands_nm": {"dtype": "<u2", "data": bands},
        "components": [
            {"name": "Référence", "provenance": {"source": ["raw", None]}},
            {"name": None, "provenance": {}},
        ],
        "provenance": {"normalized": False},
    }

    restored = SpectrumCollection.load_json(path)

    for field in ("values", "wavelengths_nm", "detection_bands_nm"):
        expected = getattr(collection, field)
        actual = getattr(restored, field)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert restored.names == collection.names
    assert restored.row_provenance == collection.row_provenance
    assert restored.provenance == collection.provenance
    assert restored.spectrum(0).provenance == collection.spectrum(0).provenance


def test_collection_json_without_axes_or_metadata(tmp_path):
    path = tmp_path / "collection.json"
    SpectrumCollection(np.array([[-0.0, -0.125]], dtype=np.float16)).save_json(str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["values"] == {"dtype": np.dtype(np.float16).str, "data": [[-0.0, -0.125]]}
    assert np.signbit(payload["values"]["data"][0][0])
    assert payload["wavelengths_nm"] is None
    assert payload["detection_bands_nm"] is None
    assert payload["components"] == [{"name": None, "provenance": {}}]
    assert payload["provenance"] == {}

    restored = SpectrumCollection.load_json(str(path))

    np.testing.assert_array_equal(restored.values, [[-0.0, -0.125]])
    assert restored.values.dtype == np.float16
    assert np.signbit(restored.values[0, 0])
    assert restored.wavelengths_nm is None
    assert restored.detection_bands_nm is None
    assert restored.names == (None,)
    assert restored.row_provenance == ({},)
    assert restored.provenance == {}


@pytest.mark.parametrize("kwargs", [
    {"provenance": {"bad": (1, 2)}},
    {"row_provenance": [{1: "non-string key"}]},
    {"row_provenance": [{"nested": [float("nan")]}]},
])
def test_collection_json_rejects_invalid_provenance_before_overwriting(tmp_path, kwargs):
    path = tmp_path / "existing.json"
    path.write_text("existing content", encoding="utf-8")

    with pytest.raises(TypeError, match="provenance"):
        SpectrumCollection([[1]], **kwargs).save_json(path)

    assert path.read_text(encoding="utf-8") == "existing content"


@pytest.mark.parametrize("field", ["values", "wavelengths_nm", "detection_bands_nm"])
def test_collection_json_rejects_extended_precision_before_overwriting(tmp_path, field):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("No extended precision dtype on this platform")
    arrays = {"values": [[1]], "wavelengths_nm": [510], "detection_bands_nm": [[500, 520]]}
    arrays[field] = np.array(arrays[field], dtype=np.longdouble)
    path = tmp_path / "existing.json"
    path.write_text("existing content", encoding="utf-8")

    with pytest.raises(ValueError, match=field + ".*64 bits"):
        SpectrumCollection(**arrays).save_json(path)

    assert path.read_text(encoding="utf-8") == "existing content"


@pytest.mark.parametrize(("keys", "value"), [
    (("format",), "lumic-spectrum"),
    (("version",), True),
    (("version",), 1.0),
    (("version",), 2),
    (("values",), None),
    (("values",), {"dtype": "|u1", "data": [[1, 2]], "extra": None}),
    (("values", "dtype"), "object"),
    (("values", "dtype"), None),
    (("values", "data"), []),
    (("values", "data"), [[]]),
    (("values", "data"), [1, 2]),
    (("values", "data"), [[1, 2], [3]]),
    (("values", "data"), [[True, 2]]),
    (("values", "data"), [["1", 2]]),
    (("values", "data"), [[1.5, 2]]),
    (("values", "data"), [[-1, 2]]),
    (("values", "data"), [[256, 2]]),
    (("wavelengths_nm",), {}),
    (("wavelengths_nm", "data"), [[510, 478]]),
    (("wavelengths_nm", "data"), [510]),
    (("wavelengths_nm", "data"), [510, 0]),
    (("wavelengths_nm", "data"), [510, float("nan")]),
    (("wavelengths_nm", "data"), [510, 1e300]),
    (("detection_bands_nm", "data"), [[500, 520]]),
    (("detection_bands_nm", "data"), [[500, 520, 525], [475, 480, 485]]),
    (("detection_bands_nm", "data"), [[520, 500], [475, 480]]),
    (("components",), {}),
    (("components",), []),
    (("components", 0), {"provenance": {}}),
    (("components", 0, "name"), 7),
    (("components", 0, "provenance"), None),
    (("components", 0, "provenance"), {"source": [float("inf")]}),
    (("components", 0, "provenance"), {"detection_bands_nm": [[500, 520], [475, 480]]}),
    (("provenance",), []),
    (("provenance",), {"source": float("nan")}),
])
def test_collection_json_rejects_invalid_fields(tmp_path, keys, value):
    path = tmp_path / "collection.json"
    SpectrumCollection(np.array([[1, 2]], dtype=np.uint8), np.array([510, 478], dtype=np.float32),
                       detection_bands_nm=[[500, 520], [475, 480]]).save_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        SpectrumCollection.load_json(path)


@pytest.mark.parametrize("content", ["{", "[]", "{}", '{"format": "lumic-spectrum-collection"}'])
def test_collection_json_rejects_malformed_or_incomplete_files(tmp_path, content):
    path = tmp_path / "collection.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        SpectrumCollection.load_json(path)


def test_collection_workflow_preserves_sampled_detector_axis_and_recovers_mixture(tmp_path):
    path = tmp_path / "references.json"
    collection, fitted = run_workflow(path)

    detector = SpectralDetector.from_profile("980_32ch")
    assert path.is_file()
    assert collection.names == ("EBFP2", "EGFP", "mCherry")
    assert collection.values.shape == (3, 32)
    np.testing.assert_array_equal(collection.wavelengths_nm, detector.channel_centers)
    np.testing.assert_array_equal(collection.detection_bands_nm[:, 0], detector.bin_edges[:-1])
    np.testing.assert_array_equal(collection.detection_bands_nm[:, 1], detector.bin_edges[1:])
    np.testing.assert_allclose(fitted, [[0.50, 0.25, 0.25], [0.00, 0.25, 0.75]], atol=1e-12)
