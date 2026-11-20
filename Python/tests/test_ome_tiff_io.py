"""Real paired exports and independent known-value selective-reading checks."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from lumic import SpectralImage
from lumic.io import load_czi, load_ome_tiff, open_ome_tiff

ROOT = Path(__file__).resolve().parents[1] / "data/io-data"
PAIRS = [
    ("spectral-example-all-channels", "CYX", np.uint16),
    ("spectral-example-subset-channels", "CYX", np.uint16),
    ("ex405-range411_694-res8p8nm_zstack3", "CZYX", np.uint8),
    ("ex405-range411_694-res8p8nm_tseries3", "CTYX", np.uint8),
]


@pytest.mark.parametrize("stem,axes,dtype", PAIRS)
def test_zen_exports_preserve_all_pixels_and_record_metadata_limits(stem, axes, dtype):
    czi = load_czi(ROOT / "980" / f"{stem}.czi", t=None, z=None)
    image = load_ome_tiff(ROOT / "980-ome-tiff" / f"{stem}.ome.tiff", t=None, z=None)
    assert isinstance(image, SpectralImage)
    assert image.axes == axes
    assert image.data.dtype == dtype
    np.testing.assert_array_equal(image.data, czi.data.transpose([czi.axes.index(a) for a in axes]))
    assert image.pixel_sizes_um == pytest.approx(czi.pixel_sizes_um, rel=1e-6)
    # ZEN retains rounded channel names, not the CZI detection-window metadata.
    assert image.detection_bands_nm is None
    assert image.provenance["wavelength_source"] == "OME Channel.Name (numeric nm label)"
    np.testing.assert_allclose(image.wavelengths_nm, czi.wavelengths_nm, atol=1, rtol=0)
    assert len(image.provenance["channel_names"]) == image.data.shape[image.axes.index("C")]
    first = load_ome_tiff(ROOT / "980-ome-tiff" / f"{stem}.ome.tiff")
    assert first.axes == "CYX"
    expected = czi.data
    expected_axes = czi.axes
    for axis in "TZ":
        if axis in expected_axes:
            expected = np.take(expected, 0, axis=expected_axes.index(axis))
            expected_axes = expected_axes.replace(axis, "")
    np.testing.assert_array_equal(first.data, expected)


def write_volume(path, axes="TCZYX", compression=None, **extra):
    canonical_axes = "TCZYX"
    # Coordinate-encoded values detect T/C/Z swaps, not just matching shapes.
    t, c, z, y, x = np.indices((2, 3, 2, 3, 5))
    canonical = (10000*t + 1000*c + 100*z + 10*y + x).astype(np.uint16)
    values = canonical.transpose([canonical_axes.index(a) for a in axes])
    metadata = dict(axes=axes, PhysicalSizeX=100, PhysicalSizeXUnit="nm",
                    PhysicalSizeY=.2, PhysicalSizeYUnit="µm",
                    Channel={"Name": ["A", "B", "C"],
                             "EmissionWavelength": [.45, .53, .62],
                             "EmissionWavelengthUnit": ["µm"]*3})
    tifffile.imwrite(path, values, kind="ome", photometric="minisblack",
                     metadata=metadata, compression=compression, **extra)
    return canonical


@pytest.mark.parametrize("axes,compression", [("TCZYX", None), ("CTZYX", "deflate"), ("TZCYX", "lzw")])
def test_combined_layouts_and_compressed_selection_match_known_values(tmp_path, axes, compression):
    path = tmp_path / "volume.ome.tif"
    truth = write_volume(path, axes, compression)
    with open_ome_tiff(path) as reader:
        assert reader.axes == axes
        selected = reader.read(t=1, z=1)
        np.testing.assert_array_equal(selected.data, truth[1, :, 1])
        assert selected.axes == "CYX"
        np.testing.assert_allclose(selected.wavelengths_nm, [450, 530, 620])
        assert selected.pixel_sizes_um == pytest.approx({"X": .1, "Y": .2})
        full = reader.read(t=None, z=None)
        assert full.axes == axes
        np.testing.assert_array_equal(full.data, truth.transpose(["TCZYX".index(a) for a in axes]))
        stack = reader.read(t=1, z=None)
        assert stack.axes == axes.replace("T", "")
        expected = truth[1].transpose(["CZYX".index(a) for a in stack.axes])
        np.testing.assert_array_equal(stack.data, expected)


def test_open_decodes_nothing_and_read_only_decodes_selected_pages(tmp_path):
    path = tmp_path / "compressed.ome.tif"
    truth = write_volume(path, compression="deflate")
    with patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("decoding during open")):
        reader = open_ome_tiff(path)
    handle = reader._tif.filehandle
    calls = []
    original = tifffile.TiffPage.asarray
    def record(page, *args, **kwargs):
        calls.append(page.index)
        return original(page, *args, **kwargs)
    with reader:
        # A full-series array call would violate selective-reading requirements.
        with patch.object(tifffile.TiffPageSeries, "asarray", side_effect=AssertionError("full read")):
            with patch.object(tifffile.TiffPage, "asarray", record):
                image = reader.read(t=1, z=1)
    assert calls == [7, 9, 11]  # Exactly the three C pages at the chosen T/Z.
    assert handle.closed
    np.testing.assert_array_equal(image.data, truth[1, :, 1])
    with pytest.raises(ValueError, match="closed"):
        reader.read()
    reader.close()


def test_tiled_bigtiff_and_single_channel_singleton_axes(tmp_path):
    path = tmp_path / "tiles.ome.tif"
    values = np.arange(3*32*32, dtype=np.uint16).reshape(3, 32, 32)
    tifffile.imwrite(path, values, kind="ome", photometric="minisblack",
                     metadata={"axes": "CYX"}, bigtiff=True, tile=(16, 16), compression="deflate")
    np.testing.assert_array_equal(load_ome_tiff(path).data, values)
    singleton = tmp_path / "one.ome.tif"
    values = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    tifffile.imwrite(singleton, values, kind="ome", photometric="minisblack", metadata={"axes": "CYX"})
    image = load_ome_tiff(singleton)
    assert image.axes == "CYX"
    assert image.data.shape == (1, 1, 5)
    assert image.wavelengths_nm is None
    np.testing.assert_array_equal(image.data, values)


@pytest.mark.parametrize("selection,error", [({"t": 2}, IndexError), ({"z": -1}, IndexError),
                                            ({"z": True}, TypeError), ({"t": 1.2}, TypeError)])
def test_bad_selection_fails_before_pixel_read(tmp_path, selection, error):
    path = tmp_path / "volume.ome.tif"
    write_volume(path)
    with open_ome_tiff(path) as reader:
        with patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("unexpected read")):
            with pytest.raises(error):
                reader.read(**selection)


def test_missing_invalid_and_non_ome_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_ome_tiff(tmp_path / "missing.ome.tif")
    with pytest.raises(ValueError, match=".tif"):
        open_ome_tiff(tmp_path / "file.txt")
    bad = tmp_path / "invalid.ome.tif"
    bad.write_text("not a TIFF")
    with pytest.raises(ValueError, match="valid TIFF"):
        open_ome_tiff(bad)
    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((3, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="OME-XML"):
        open_ome_tiff(plain)


def test_partial_or_ambiguous_channel_labels_do_not_invent_wavelengths(tmp_path):
    for names, emission in [(["450 nm", "EGFP", "620 nm"], None),
                             (["450 nm", "530 nm", "620 nm"], [450, 530, 620])]:
        path = tmp_path / "partial.ome.tif"
        channels = {"Name": names}
        if emission is not None:
            channels["EmissionWavelength"] = emission
        tifffile.imwrite(path, np.ones((3, 2, 4), dtype=np.uint16), kind="ome",
                         photometric="minisblack", metadata={"axes": "CYX", "Channel": channels})
        if emission is not None:
            # A partial field is valid OME; a literal XML "None" is not.
            from ome_types import from_tiff, to_xml
            ome = from_tiff(path)
            ome.images[0].pixels.channels[1].emission_wavelength = None
            tifffile.imwrite(path, np.ones((3, 2, 4), dtype=np.uint16),
                             photometric="minisblack", description=to_xml(ome), metadata=None)
        assert load_ome_tiff(path).wavelengths_nm is None


def test_series_choice_and_constructor_cleanup(tmp_path):
    path = tmp_path / "two.ome.tif"
    with tifffile.TiffWriter(path, kind="ome") as writer:
        writer.write(np.ones((3, 2, 5), dtype=np.uint16), photometric="minisblack", metadata={"axes": "CYX"})
        writer.write(np.full((4, 3, 6), 7, dtype=np.uint16), photometric="minisblack", metadata={"axes": "CYX"})
    assert load_ome_tiff(path, series=1).data.shape == (4, 3, 6)
    with pytest.raises(IndexError, match="series"):
        open_ome_tiff(path, series=2)
    with pytest.raises(TypeError, match="series"):
        open_ome_tiff(path, series=True)
    with pytest.raises(RuntimeError, match="user error"):
        with open_ome_tiff(path) as reader:
            handle = reader._tif.filehandle
            raise RuntimeError("user error")
    assert handle.closed


def test_rgb_and_pyramidal_files_rejected(tmp_path):
    rgb = tmp_path / "rgb.ome.tif"
    tifffile.imwrite(rgb, np.zeros((5, 7, 3), dtype=np.uint8), kind="ome", photometric="rgb",
                     metadata={"axes": "YXS"})
    with pytest.raises(ValueError, match="layout|scalar"):
        open_ome_tiff(rgb)
    pyramid = tmp_path / "pyramid.ome.tif"
    with tifffile.TiffWriter(pyramid, kind="ome") as writer:
        writer.write(np.ones((3, 32, 32), dtype=np.uint16), photometric="minisblack",
                     metadata={"axes": "CYX"}, subifds=1)
        writer.write(np.ones((3, 16, 16), dtype=np.uint16), photometric="minisblack", subfiletype=1)
    with pytest.raises(ValueError, match="Pyramidal"):
        open_ome_tiff(pyramid)


def test_external_file_reference_rejected_before_series_assembly(tmp_path):
    from ome_types import from_tiff, to_xml
    from ome_types.model import TiffData
    path = tmp_path / "external.ome.tif"
    values = np.ones((3, 2, 4), dtype=np.uint16)
    tifffile.imwrite(path, values, kind="ome", photometric="minisblack", metadata={"axes": "CYX"})
    ome = from_tiff(path)
    ome.images[0].pixels.tiff_data_blocks[0].uuid = TiffData.UUID(
        value="urn:uuid:00000000-0000-0000-0000-000000000001", file_name="other.ome.tif")
    tifffile.imwrite(path, values, photometric="minisblack", description=to_xml(ome), metadata=None)
    with pytest.raises(ValueError, match="Multi-file"):
        open_ome_tiff(path)


def test_invalid_ome_metadata_closes_constructor_file(tmp_path):
    path = tmp_path / "bad_metadata.ome.tif"
    write_volume(path)
    from ome_types import from_tiff, to_xml
    ome = from_tiff(path)
    xml = to_xml(ome).replace('SizeC="3"', 'SizeC="invalid"')
    tifffile.imwrite(path, np.ones((3, 2, 4), dtype=np.uint16),
                     photometric="minisblack", description=xml.encode("utf-8"), metadata=None)
    instances = []
    original = tifffile.TiffFile
    def capture(*args, **kwargs):
        obj = original(*args, **kwargs)
        instances.append(obj)
        return obj
    with patch.object(tifffile, "TiffFile", capture):
        with pytest.raises(ValueError, match="Invalid OME-XML"):
            open_ome_tiff(path)
    assert instances[0].filehandle.closed
