"""Reader contracts, metadata-only summaries and dispatcher equivalence."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pylibCZIrw.czi import CziReader as CziBackend
import tifffile

from lumic.io import (
    CziReader, OmeTiffReader, load_czi, load_image, load_ome_tiff,
    open_czi, open_image, open_ome_tiff,
)

ROOT = Path(__file__).resolve().parents[1] / 'data/io-data'
CASES = [
    ('980/ex405-range411_694-res8p8nm_zstack3.czi', CziReader, open_czi, load_czi, 'CZI'),
    ('980-ome-tiff/ex405-range411_694-res8p8nm_zstack3.ome.tiff', OmeTiffReader, open_ome_tiff, load_ome_tiff, 'OME-TIFF'),
]


@pytest.mark.parametrize('relative,reader_type,opener,loader,format_name', CASES)
@pytest.mark.parametrize('selection', [{}, {'z': 1}, {'t': None, 'z': None}])
def test_dispatch_and_specific_calls_return_identical_images(relative, reader_type, opener, loader, format_name, selection):
    path = ROOT / relative
    expected = loader(path, **selection)
    actual = load_image(path, **selection)
    assert actual.axes == expected.axes and actual.kind == expected.kind
    assert actual.data.dtype == expected.data.dtype
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.wavelengths_nm, expected.wavelengths_nm)
    np.testing.assert_array_equal(actual.detection_bands_nm, expected.detection_bands_nm)
    assert actual.pixel_sizes_um == expected.pixel_sizes_um
    assert actual.provenance == expected.provenance
    with open_image(path) as reader, opener(path) as specific:
        assert isinstance(reader, reader_type)  # No extra reader object.
        assert reader.summary() == specific.summary()
        assert reader.axes == specific.axes and reader.shape == specific.shape
        assert reader.format_name == format_name
    # Loaded arrays own their lifetime after the format adapter closes.
    actual.data.flat[0] = 123
    assert actual.data.flat[0] == 123


@pytest.mark.parametrize('relative,reader_type,opener,loader,format_name', CASES)
def test_open_and_summary_never_decode_pixels_or_print(relative, reader_type, opener, loader, format_name, capsys):
    with patch.object(CziBackend, 'read', side_effect=AssertionError('pixel decode')), \
         patch.object(tifffile.TiffPage, 'asarray', side_effect=AssertionError('pixel decode')):
        with open_image(ROOT / relative) as reader:
            text = reader.summary()
            assert f'Format: {format_name}' in text
            assert 'Detector channels: 32' in text
            assert 'Z planes: 3; time points: 1' in text
            assert 'Supported for loading: yes (header/layout checks passed)' in text
            assert 'Wavelength coordinates: available' in text
            assert reader.axes == 'CZYX'
        assert reader.summary() == text  # A metadata snapshot, not a read.
    assert capsys.readouterr().out == ''


def test_overlapping_lsm900_windows_do_not_claim_spectral_wavelengths():
    with open_image(ROOT / 'cd7-900/ex405-range450_650-res66p6nm.czi') as reader:
        summary = reader.summary()
        assert 'Detector channels: 8' in summary
        assert 'Detection bands: available' in summary
        assert 'Wavelength coordinates: not available' in summary
        assert 'fluorophores:' not in summary.lower()


@pytest.mark.parametrize('opener', [open_image, load_image])
def test_invalid_files_report_clear_errors(opener, tmp_path):
    with pytest.raises(FileNotFoundError, match='missing.czi'):
        opener(tmp_path / 'missing.czi')
    unsupported = tmp_path / 'image.png'
    unsupported.write_bytes(b'not supported')
    with pytest.raises(ValueError, match='Unsupported image extension'):
        opener(unsupported)
    invalid = tmp_path / 'corrupted.czi'
    invalid.write_bytes(b'not a CZI')
    with pytest.raises(ValueError, match='valid CZI'):
        opener(invalid)
    plain = tmp_path / 'plain.tif'
    tifffile.imwrite(plain, np.zeros((2, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match='OME-XML'):
        opener(plain)


@pytest.mark.parametrize('opener', [open_image, load_image])
def test_ome_series_selection_is_not_silently_ignored_for_czi(opener, tmp_path):
    path = tmp_path / 'two-series.ome.tif'
    with tifffile.TiffWriter(path, kind='ome') as writer:
        for value in [10, 20]:
            writer.write(np.full((3, 2, 5), value, dtype=np.uint16),
                         photometric='minisblack', metadata={'axes': 'CYX'})
    if opener is open_image:
        with opener(path, series=1) as reader:
            actual = reader.read()
    else:
        actual = opener(path, series=1)
    assert np.all(actual.data == 20)
    assert actual.provenance['series'] == 1
    with pytest.raises(ValueError, match='OME-TIFF-only'):
        opener(ROOT / CASES[0][0], series=0)
    with pytest.raises(IndexError, match='series'):
        opener(path, series=2)


def test_dispatch_case_insensitive_extensions_and_ome_xml_validation(tmp_path):
    path = tmp_path / 'upper.OME.TIFF'
    tifffile.imwrite(path, np.zeros((3, 2, 5), dtype=np.uint16), kind='ome',
                     photometric='minisblack', metadata={'axes': 'CYX'})
    assert load_image(path).data.shape == (3, 2, 5)


def test_load_image_closes_reader_after_read_error():
    reader = open_czi(ROOT / CASES[0][0])
    with patch('lumic.io.open_image', return_value=reader):
        with pytest.raises(IndexError):
            load_image(reader.path, z=99)
    assert reader._czi is None


@pytest.mark.parametrize('axes', ['TCZYX', 'YXCTZ'])
def test_contributor_template_preserves_selection_and_source(tmp_path, axes):
    from examples.reader_template import NpyReaderTemplate
    data = np.arange(2*3*2*4*5, dtype=np.uint16).reshape(2, 3, 2, 4, 5)
    stored = data.transpose(['TCZYX'.index(a) for a in axes])
    path = tmp_path / 'contributor.npy'
    np.save(path, stored)
    before = path.read_bytes()
    with NpyReaderTemplate(path, axes=axes) as reader:
        assert reader.axes == axes and reader.shape == stored.shape
        assert 'Detector channels: 3' in reader.summary()
        assert 'Wavelength coordinates: not available' in reader.summary()
        assert reader.wavelengths_nm is None and reader.pixel_sizes_um == {}
        selected = reader.read(t=1, z=1)
        expected_axes = ''.join(a for a in axes if a not in 'TZ')
        assert selected.axes == expected_axes
        np.testing.assert_array_equal(selected.data, data[1, :, 1].transpose(['CYX'.index(a) for a in expected_axes]))
        with pytest.raises(IndexError):
            reader.read(z=2)
    reader.close()  # Idempotent.
    selected.data[:] = 99
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='closed'):
        reader.read()
    with pytest.raises(ValueError, match='Unsupported image extension'):
        open_image(path)  # Template is deliberately not registered.
