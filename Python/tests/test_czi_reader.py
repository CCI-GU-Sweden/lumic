from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pytest
from pylibCZIrw.czi import CziReader as BackendReader
from lumic import SpectralImage
from lumic.io import load_czi, open_czi, loadCZISpecData

ROOT = Path(__file__).resolve().parents[1] / "data/io-data"
CASES = [("880", "ex405-range409_695-res8p9nm"), ("980", "ex405-range411_694-res8p8nm")]


@pytest.mark.parametrize("instrument,stem", CASES)
def test_reader_matches_legacy(instrument, stem):
    path = ROOT / instrument / (stem + ".czi")
    image = load_czi(path)
    assert isinstance(image, SpectralImage)
    assert image.axes == "CYX"
    np.testing.assert_array_equal(image.data, loadCZISpecData(path))
    np.testing.assert_allclose(image.wavelengths_nm, image.detection_bands_nm.mean(axis=1))
    assert image.pixel_sizes_um["X"] > 0


@pytest.mark.parametrize("instrument,stem", CASES)
@pytest.mark.parametrize("axis,suffix", [("Z", "_zstack3"), ("T", "_tseries3")])
def test_selection_reads_only_requested_plane(instrument, stem, axis, suffix):
    path = ROOT / instrument / (stem + suffix + ".czi")
    full = load_czi(path, t=None, z=None)
    assert full.data.shape[full.axes.index(axis)] == 3
    original = BackendReader.read
    calls = []
    def record(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)
    with open_czi(path) as reader:
        assert reader.axes == full.axes
        assert reader.shape == full.data.shape
        with patch.object(BackendReader, "read", record):
            selected = reader.read(**{axis.lower(): 1})
        assert len(calls) == selected.data.shape[0]
        assert [call["plane"]["C"] for call in calls] == list(range(selected.data.shape[0]))
        assert all(call["plane"][axis] == 1 and call["zoom"] == 1.0 for call in calls)
        assert selected.axes == "CYX"
        np.testing.assert_array_equal(selected.data, np.take(full.data, 1, axis=full.axes.index(axis)))
        assert selected.provenance["selection"][axis] == 1
    assert reader._czi is None
    assert reader._context is None
    with pytest.raises(ValueError, match="closed"):
        reader.read()
    assert selected.data.size > 0


def test_metadata_only_open_and_exception_cleanup():
    with patch.object(BackendReader, "read", side_effect=AssertionError("unexpected decoding")):
        with pytest.raises(RuntimeError, match="user error"):
            with open_czi(ROOT / "980/ex405-range411_694-res8p8nm.czi") as reader:
                assert reader.shape == (32, 32, 32)
                raise RuntimeError("user error")
    assert reader._czi is None
    assert reader._context is None


@pytest.mark.parametrize("selection,error", [({"z": 3}, IndexError), ({"z": -1}, IndexError),
                                            ({"z": 1.5}, TypeError), ({"z": True}, TypeError)])
def test_bad_selection_before_read(selection, error):
    with open_czi(ROOT / "980/ex405-range411_694-res8p8nm_zstack3.czi") as reader:
        with patch.object(BackendReader, "read", side_effect=AssertionError("unexpected decoding")):
            with pytest.raises(error):
                reader.read(**selection)


def test_lsm900_windows_are_not_point_wavelengths():
    image = load_czi(ROOT / "cd7-900/ex405-range450_650-res66p6nm.czi")
    assert image.axes == "CYX"
    assert image.data.shape[0] == 8
    assert image.wavelengths_nm is None
    np.testing.assert_allclose(image.detection_bands_nm[-1], [650, 650])


@pytest.mark.parametrize("instrument,stem,size,origin", [
    ("880", "ex405-range409_695-res8p9nm_tile2x2", 14, (-3, -3)),
    ("980", "ex405-range411_694-res8p8nm_tile2x2", 60, (0, 0)),
    ("cd7-900", "ex405-range450_650-res66p6nm-tile2x2", 62, (0, 0)),
])
def test_single_scene_mosaic_is_composed(instrument, stem, size, origin):
    image = load_czi(ROOT / instrument / (stem + ".czi"))
    assert image.axes == "CYX"
    assert image.data.shape[1:] == (size, size)
    assert image.provenance["mosaic_composed"] is True
    assert image.provenance["spatial_origin_xy"] == origin


def test_missing_and_wrong_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_czi(tmp_path / "missing.czi")
    with pytest.raises(ValueError, match=".czi"):
        load_czi(tmp_path / "image.txt")


def test_invalid_contents(tmp_path):
    path = tmp_path / "invalid.czi"
    path.write_text("not a CZI")
    with pytest.raises(ValueError, match="valid CZI"):
        load_czi(path)


def test_combined_time_z_and_singleton_spatial_axes():
    # No combined T/Z acquisition has been collected yet. This known-value
    # backend result verifies axis handling independently of real-file decoding.
    path = ROOT / "980/ex405-range411_694-res8p8nm.czi"
    data = np.arange(2 * 32 * 3 * 1 * 4, dtype=np.uint16).reshape(2, 32, 3, 1, 4)
    with open_czi(path) as reader:
        reader.axes = "TCZYX"
        reader.dimension_ranges.update(T=(0, 2), Z=(0, 3), Y=(0, 1), X=(0, 4))
        def plane(self, *, plane, **kwargs):
            return data[plane["T"], plane["C"], plane["Z"]][..., None]
        with patch.object(BackendReader, "read", plane):
            image = reader.read(t=None, z=None)
    assert image.axes == "TCZYX"
    assert image.data.shape == (2, 32, 3, 1, 4)
    np.testing.assert_array_equal(image.data, data)


def test_numpy_input_optional_metadata_and_dtype_validation():
    image = SpectralImage(np.zeros((1, 1, 1), dtype=np.uint16), "CYX")
    assert image.wavelengths_nm is None
    assert image.data.dtype == np.uint16
    with pytest.raises(TypeError, match="NumPy"):
        SpectralImage([1, 2], "CYX")
    with pytest.raises(TypeError, match="real numeric"):
        SpectralImage(np.zeros((1, 1, 1), dtype=complex), "CYX")
    with pytest.raises(ValueError, match="nonempty"):
        SpectralImage(np.zeros((1, 0, 1)), "CYX")


@pytest.mark.parametrize("instrument,stem", CASES)
@pytest.mark.parametrize("suffix", ["", "_zstack3", "_tseries3"])
def test_default_read_is_first_cyx_plane(instrument, stem, suffix):
    path = ROOT / instrument / (stem + suffix + ".czi")
    full = load_czi(path, t=None, z=None)
    expected = full.data
    axes = full.axes
    for axis in "TZ":
        if axis in axes:
            expected = np.take(expected, 0, axis=axes.index(axis))
            axes = axes.replace(axis, "")
    image = load_czi(path)
    assert image.axes == "CYX"
    np.testing.assert_array_equal(image.data, expected)


def test_absent_time_and_z_accept_zero_but_reject_other_indices():
    # Our fixtures carry singleton T/Z headers, so emulate their absence.
    with open_czi(ROOT / "980/ex405-range411_694-res8p8nm.czi") as reader:
        reader.dimension_ranges.pop("T", None)
        reader.dimension_ranges.pop("Z", None)
        original = BackendReader.read
        calls = []
        def record(self, **kwargs):
            calls.append(kwargs)
            return original(self, **kwargs)
        with patch.object(BackendReader, "read", record):
            assert reader.read().axes == "CYX"
            assert all(set(call["plane"]) == {"C"} for call in calls)
        for selection in ({"t": 1}, {"z": 1}, {"z": -1}):
            with pytest.raises(IndexError, match="only index 0"):
                reader.read(**selection)


@pytest.mark.parametrize("case", json.loads(
    (ROOT / "czi-pixel-reference.json").read_text())["cases"], ids=lambda case: case["path"])
def test_all_pixels_match_pre_migration_decoder(case):
    import hashlib

    image = load_czi(ROOT / case["path"], t=None, z=None)
    assert image.axes == case["axes"]
    assert list(image.data.shape) == case["shape"]
    assert str(image.data.dtype) == case["dtype"]
    assert hashlib.sha256(image.data.tobytes()).hexdigest() == case["pixels_sha256"]


def test_nonzero_plane_ranges_and_singleton_spatial_axes():
    from types import SimpleNamespace
    from contextlib import contextmanager
    # A translated coordinate system must not be mistaken for a zero-based
    # channel or time index. The Y/X shape includes singleton spatial axes.
    calls = []
    closed = []
    class Backend:
        raw_metadata = "<Metadata/>"
        total_bounding_box = {"X": (-4, -3), "Y": (7, 9)}
        def enumerate_subblocks(self, callback):
            for t in (5, 6):
                for c in (2, 3):
                    callback(0, SimpleNamespace(
                        coordinate=SimpleNamespace(to_dict=lambda t=t, c=c: {"T": t, "C": c}),
                        logicalRect=SimpleNamespace(x=-4, y=7, w=1, h=2),
                        pixelType=SimpleNamespace(name="Gray16")))
        def read(self, *, plane, zoom):
            calls.append(plane)
            return np.full((2, 1, 1), plane["T"] * 10 + plane["C"], dtype=np.uint16)
    @contextmanager
    def backend(path):
        try:
            yield Backend()
        finally:
            closed.append(True)
    with patch("lumic.io.pyczi.open_czi", backend):
        image = load_czi(ROOT / "980/ex405-range411_694-res8p8nm.czi", t=1)
    assert image.axes == "CYX"
    np.testing.assert_array_equal(image.data[:, :, 0], [[62, 62], [63, 63]])
    assert calls == [{"T": 6, "C": 2}, {"T": 6, "C": 3}]
    assert closed == [True]


def test_constructor_failure_closes_backend():
    from contextlib import contextmanager
    closed = []
    @contextmanager
    def backend(path):
        try:
            yield object()
        finally:
            closed.append(True)
    with patch("lumic.io.pyczi.open_czi", backend):
        with pytest.raises(AttributeError):
            open_czi(ROOT / "980/ex405-range411_694-res8p8nm.czi")
    assert closed == [True]


@pytest.mark.parametrize("multiple_scenes,rgb", [(True, False), (False, True)])
def test_unsupported_layout_rejected_before_pixel_decode(multiple_scenes, rgb):
    from lumic.io import _czi_layout
    original = _czi_layout
    def layout(reader):
        axes, ranges, mosaic, scenes, scalar = original(reader)
        return axes, ranges, mosaic, {0, 1} if multiple_scenes else scenes, not rgb
    with patch("lumic.io._czi_layout", layout), patch.object(
            BackendReader, "read", side_effect=AssertionError("unexpected decoding")):
        with pytest.raises(ValueError, match="scenes|RGB"):
            open_czi(ROOT / "980/ex405-range411_694-res8p8nm.czi")
