import csv
from pathlib import Path

import numpy as np
import pytest

from lumic.io import getCZIchannelWave, loadCZISpecData, loadCZIChannel, load_czi


EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "io-data" / "980"


def test_legacy_channel_helper_warns_and_preserves_pixels():
    from pylibCZIrw import czi

    path = EXAMPLES / "spectral-example-all-channels.czi"
    expected = load_czi(path).data[2]
    with czi.open_czi(str(path)) as legacy_reader:
        with pytest.warns(FutureWarning, match=r"loadCZIChannel.*load_czi"):
            channel = loadCZIChannel(legacy_reader, 2)
    np.testing.assert_array_equal(channel, expected)
    assert channel.dtype == expected.dtype


@pytest.mark.parametrize(
    ("case", "expected_channels"),
    [("all-channels", 32), ("subset-channels", 18)],
)
def test_lsm980_czi_channels_and_wavelengths_match_labels(case, expected_channels):
    czi_path = EXAMPLES / f"spectral-example-{case}.czi"
    label_path = EXAMPLES / f"{case}-labels.txt"
    expected_wavelengths = np.fromstring(label_path.read_text(), sep=",")

    image = loadCZISpecData(czi_path)
    wavelengths = getCZIchannelWave(czi_path)

    assert expected_wavelengths.shape == (expected_channels,)
    assert image.shape == (expected_channels, 32, 32)
    assert wavelengths.shape == (expected_channels,)
    np.testing.assert_allclose(wavelengths, expected_wavelengths, rtol=0, atol=1)


def test_lsm980_czi_pixel_values_match_recorded_positions():
    label_path = EXAMPLES / "all-channels-labels.txt"
    values_path = EXAMPLES / "all-channels-labels-pixel-values.txt"
    channel_labels = np.fromstring(label_path.read_text(), sep=",", dtype=int)
    image = loadCZISpecData(EXAMPLES / "spectral-example-all-channels.czi")

    with values_path.open(newline="") as values_file:
        observations = list(csv.DictReader(values_file, skipinitialspace=True))

    assert len(observations) == 6
    for observation in observations:
        wavelength = int(observation["channel"])
        x = int(observation["x"])
        y = int(observation["y"])
        expected_value = int(observation["value"])

        # The labels are in channel order; image axes are [channel, y, x].
        channel_indices = np.flatnonzero(channel_labels == wavelength)
        assert channel_indices.size == 1
        assert 0 <= x < image.shape[2]
        assert 0 <= y < image.shape[1]
        assert image[channel_indices[0], y, x] == expected_value


@pytest.mark.parametrize("reader", [getCZIchannelWave, loadCZISpecData])
def test_czi_reader_reports_missing_file(reader, tmp_path):
    missing_path = tmp_path / "missing.czi"

    with pytest.raises(FileNotFoundError, match="missing.czi"):
        reader(missing_path)


@pytest.mark.parametrize("reader", [getCZIchannelWave, loadCZISpecData])
def test_czi_reader_rejects_other_file_extension(reader):
    text_path = EXAMPLES / "all-channels-labels.txt"

    with pytest.raises(ValueError, match=r"\.czi"):
        reader(text_path)


@pytest.mark.parametrize("reader", [getCZIchannelWave, loadCZISpecData])
def test_czi_reader_rejects_non_czi_contents(reader, tmp_path):
    invalid_path = tmp_path / "invalid.czi"
    invalid_path.write_text("This is not a CZI image.")

    with pytest.raises(ValueError, match="valid CZI"):
        reader(invalid_path)


def test_legacy_channel_helper_accepts_old_reader_protocol():
    class OldReader:
        def read_image(self, **selection):
            assert selection == {"C": 2}
            return np.array([[[[11], [22]]]], dtype=np.uint16), [("T", 1), ("C", 1), ("Y", 2), ("X", 1)]
    with pytest.warns(FutureWarning, match="legacy helper"):
        pixels = loadCZIChannel(OldReader(), 2)
    np.testing.assert_array_equal(pixels, [[11], [22]])


def test_legacy_channel_helper_accepts_lumic_reader():
    from lumic.io import open_czi
    path = EXAMPLES / "spectral-example-all-channels.czi"
    with open_czi(path) as reader:
        expected = reader.read().data[2]
        with pytest.warns(FutureWarning, match="legacy helper"):
            pixels = loadCZIChannel(reader, 2)
    np.testing.assert_array_equal(pixels, expected)
