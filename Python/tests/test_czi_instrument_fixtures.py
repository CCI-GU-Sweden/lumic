"""Recorded real-file regression checks, distinct from spectral interpretation.

The manifest freezes supplied CZI header/XML and decoded pixels. It is not an
independent measurement of detector response. Existing ZEN label/pixel tests
in test_czi_io.py retain their independent references after the folder move.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pylibCZIrw import czi
from lxml import etree

from lumic.io import getCZIchannelWave, loadCZISpecData, _czi_layout


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "io-data"
CASES = json.loads((FIXTURES / "manifest.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["path"] for case in CASES])
def test_instrument_fixture_matches_recorded_layout_bands_and_pixels(case):
    path = FIXTURES / case["path"]
    assert path.stat().st_size == case["size_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    with czi.open_czi(str(path)) as reader:
        axes, ranges, mosaic, _, _ = _czi_layout(reader)
        meta = etree.fromstring(reader.raw_metadata.encode("utf-8"))
    assert mosaic == case["mosaic"]
    # The old manifest describes raw tile axes for mosaics. The new backend
    # exposes composed bounds, so compare native axis sizes for non-mosaics.
    if not mosaic:
        assert axes == case["native_axes"]
        assert {axis: high - low for axis, (low, high) in ranges.items()} == case["native_sizes"]
    bands = [list(map(float, channel.findtext("DetectionWavelength/Ranges").split("-")))
             for channel in meta.findall(".//Information/Image/Dimensions/Channels/Channel")]
    np.testing.assert_allclose(bands, case["detection_bands_nm"], rtol=0, atol=1e-9)
    assert len(bands) == case["native_sizes"]["C"]
    scales = {item.attrib["Id"]: float(item.findtext("Value")) * 1e6
              for item in meta.findall(".//Scaling/Items/Distance")
              if item.attrib.get("Id") in case["pixel_sizes_um"]}
    assert scales == pytest.approx(case["pixel_sizes_um"])

    # Stack/time/mosaic headers are regression fixtures for future explicit
    # selection support; the legacy loader promises only a 2D channel stack.
    if not case["single_plane_supported"]:
        return

    image = loadCZISpecData(path)
    assert image.shape == tuple(case["expected_image_shape"])
    assert str(image.dtype) == case["expected_dtype"]
    for pixel in case["pixel_samples"]:
        assert image[pixel["channel_index"], pixel["y"], pixel["x"]] == pixel["value"]

    if case["measurement_model"] == "spectral_detector":
        # The legacy API returns integer band midpoints; it loses sub-nm
        # precision. LSM 900 window midpoints are not a conventional spectral axis.
        wave = getCZIchannelWave(path)
        assert wave.shape == (image.shape[0],)
        np.testing.assert_allclose(wave, np.mean(bands, axis=1), rtol=0, atol=1)


@pytest.mark.parametrize("filename,steps", [
    ("ex405-range450_650-res66p6nm.czi", np.linspace(450, 650, 4)),
    ("ex405-range450_650-res40nm.czi", np.linspace(450, 650, 6)),
])
def test_lsm900_metadata_records_paired_windows_not_adjacent_spectral_bins(filename, steps):
    """Acquisition description supplied by the operator, checked against CZI XML.

    The stored ranges meet at each boundary. This asserts metadata semantics,
    not measured filter transmission or independent photon statistics.
    """
    with czi.open_czi(str(FIXTURES / "cd7-900" / filename)) as reader:
        meta = etree.fromstring(reader.raw_metadata.encode("utf-8"))
    bands = np.array([list(map(float, channel.findtext("DetectionWavelength/Ranges").split("-")))
                      for channel in meta.findall(".//Information/Image/Dimensions/Channels/Channel")])
    np.testing.assert_allclose(bands[::2], np.column_stack([np.full(len(steps), 400), steps]))
    np.testing.assert_allclose(bands[1::2], np.column_stack([steps, np.full(len(steps), 650)]))
    # Preserve the final zero-width metadata entry instead of silently dropping it.
    np.testing.assert_allclose(bands[-1], [650, 650])
