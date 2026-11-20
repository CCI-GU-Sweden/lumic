"""Quantitative and metadata round trips, interoperability and write protection."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ome_types import from_xml
import pytest
import tifffile

from lumic import SpectralImage
from lumic.io import load_czi, load_ome_tiff, open_ome_tiff, save_ome_tiff


@pytest.mark.parametrize('compression', [None, 'deflate', 'lzw'])
@pytest.mark.parametrize('axes', ['TCZYX', 'CTZYX', 'YXCTZ', 'ZXYTC'])
def test_lossless_roundtrip_preserves_coordinates_metadata_and_kind(tmp_path, compression, axes):
    t, c, z, y, x = np.indices((2, 3, 2, 3, 5))
    canonical = (1000*t + 100*c + 10*z + y + x/8).astype(np.float32)
    data = canonical.transpose(['TCZYX'.index(a) for a in axes])
    image = SpectralImage(data, axes, kind='reconstructed', wavelengths_nm=[450, 530, 620],
        detection_bands_nm=[[440, 460], [520, 540], [610, 630]],
        pixel_sizes_um={'X': .1, 'Y': .2, 'Z': .5},
        provenance={'channel_names': ['blue detector', None, 'red detector'],
                    'fluorophores': ['EBFP2', 'EGFP', 'mCherry'],
                    'seed': np.int64(2026), 'counts': np.array([500, 800, 200]),
                    'source': Path('simulation.py')})
    before = image.data.copy()
    path = tmp_path / 'volume.ome.tiff'
    assert image.save_ome_tiff(path, compression=compression) == path
    restored = load_ome_tiff(path, t=None, z=None)
    assert restored.axes == axes
    assert restored.kind == 'reconstructed'
    assert restored.data.dtype == image.data.dtype
    np.testing.assert_array_equal(restored.data, image.data)
    np.testing.assert_array_equal(image.data, before)
    np.testing.assert_array_equal(restored.wavelengths_nm, image.wavelengths_nm)
    np.testing.assert_array_equal(restored.detection_bands_nm, image.detection_bands_nm)
    assert restored.pixel_sizes_um == pytest.approx(image.pixel_sizes_um)
    assert restored.provenance['channel_names'] == ('blue detector', None, 'red detector')
    source = restored.provenance['source_provenance']
    assert source['fluorophores'] == ['EBFP2', 'EGFP', 'mCherry']
    assert source['counts'] == [500, 800, 200]
    assert source['seed'] == 2026 and source['source'] == 'simulation.py'
    selected = load_ome_tiff(path, t=1, z=1)
    assert selected.axes == ''.join(a for a in axes if a not in 'TZ')
    np.testing.assert_array_equal(selected.data,
        canonical[1, :, 1].transpose(['CYX'.index(a) for a in selected.axes]))
    # Check the standard OME representation independently of our reader.
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        storage_axes = ''.join(a for a in axes if a not in 'YX') + 'YX'
        assert series.axes == storage_axes
        np.testing.assert_array_equal(series.asarray(),
            image.data.transpose([axes.index(a) for a in storage_axes]))
        assert tif.pages[0].compression.name == {None: 'NONE', 'deflate': 'DEFLATE', 'lzw': 'LZW'}[compression]
        ome = from_xml(tif.ome_metadata, validate=True)
        assert ome.images[0].pixels.physical_size_x == pytest.approx(.1)
        assert ome.images[0].pixels.channels[0].name == 'blue detector'
        # Detector coordinates are not misrepresented as FP emission maxima.
        assert all(c.emission_wavelength is None for c in ome.images[0].pixels.channels)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.float32, np.float64])
def test_supported_pixel_types_and_default_compression(tmp_path, dtype):
    values = np.arange(30).reshape(3, 2, 5).astype(dtype)
    if np.dtype(dtype).kind in 'if':
        values -= 10
    image = SpectralImage(values, 'CYX')
    path = save_ome_tiff(image, tmp_path / 'pixels.ome.tif')
    restored = load_ome_tiff(path)
    np.testing.assert_array_equal(restored.data, values)
    assert restored.data.dtype == values.dtype
    assert restored.kind == 'measured'
    assert restored.wavelengths_nm is None and restored.detection_bands_nm is None
    assert restored.pixel_sizes_um == {}
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].compression.name == 'DEFLATE'


def test_nonfinite_pixel_values_are_not_clipped(tmp_path):
    image = SpectralImage(np.array([np.nan, np.inf, -np.inf, -12.5]).reshape(1, 2, 2), 'CYX')
    path = save_ome_tiff(image, tmp_path / 'floating.ome.tif')
    np.testing.assert_array_equal(load_ome_tiff(path).data, image.data)


def test_singleton_axes_restored_and_channel_labels_do_not_invent_wavelengths(tmp_path):
    image = SpectralImage(np.arange(5, dtype=np.uint16).reshape(1, 1, 5, 1, 1), 'TYXCZ',
                          provenance={'channel_names': ['500 nm']})
    path = image.save_ome_tiff(tmp_path / 'singletons.ome.tif', bigtiff=True)
    with open_ome_tiff(path) as reader:
        assert reader.axes == image.axes and reader.shape == image.data.shape
        full = reader.read(t=None, z=None)
        assert full.axes == image.axes and full.wavelengths_nm is None
        np.testing.assert_array_equal(full.data, image.data)
        plane = reader.read()
        assert plane.axes == 'YXC'
        assert plane.data.shape == (1, 5, 1)
    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff


@pytest.mark.parametrize('relative', ['980/ex405-range411_694-res8p8nm_zstack3.czi',
                                      'cd7-900/ex405-range450_650-res66p6nm.czi'])
def test_real_czi_to_ome_preserves_band_and_window_semantics(tmp_path, relative):
    root = Path(__file__).resolve().parents[1] / 'data/io-data'
    image = load_czi(root / relative, t=None, z=None)
    path = save_ome_tiff(image, tmp_path / 'converted.ome.tif')
    restored = load_ome_tiff(path, t=None, z=None)
    assert restored.axes == image.axes
    np.testing.assert_array_equal(restored.data, image.data)
    np.testing.assert_array_equal(restored.detection_bands_nm, image.detection_bands_nm)
    assert restored.pixel_sizes_um == pytest.approx(image.pixel_sizes_um)
    if image.wavelengths_nm is None:
        assert restored.wavelengths_nm is None  # Overlapping LSM 900 windows.
    else:
        np.testing.assert_array_equal(restored.wavelengths_nm, image.wavelengths_nm)


def test_existing_file_protection_and_failed_write_cleanup(tmp_path):
    path = tmp_path / 'protected.ome.tif'
    image = SpectralImage(np.zeros((3, 2, 5), dtype=np.uint16), 'CYX')
    save_ome_tiff(image, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_ome_tiff(image, path)
    def fail(destination, *args, **kwargs):
        Path(destination).write_bytes(b'partial TIFF')
        raise OSError('simulated disk failure')
    with patch('tifffile.imwrite', side_effect=fail):
        with pytest.raises(OSError, match='disk failure'):
            save_ome_tiff(image, path, overwrite=True)
        with pytest.raises(OSError, match='disk failure'):
            save_ome_tiff(image, tmp_path / 'new.ome.tif')
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    image.data[:] = 10
    image.save_ome_tiff(path, overwrite=True)
    np.testing.assert_array_equal(load_ome_tiff(path).data, image.data)


def test_destination_created_during_write_is_not_overwritten(tmp_path):
    path = tmp_path / 'race.ome.tif'
    image = SpectralImage(np.zeros((1, 2, 5)), 'CYX')
    original = tifffile.imwrite
    def create_destination(*args, **kwargs):
        original(*args, **kwargs)
        path.write_bytes(b'another writer')
    with patch('tifffile.imwrite', side_effect=create_destination):
        with pytest.raises(FileExistsError):
            save_ome_tiff(image, path)
    assert path.read_bytes() == b'another writer'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('options,error', [({'compression': 'jpeg'}, ValueError),
    ({'compression': True}, ValueError), ({'bigtiff': 'yes'}, TypeError),
    ({'overwrite': 1}, TypeError)])
def test_invalid_writer_options_do_not_create_files(tmp_path, options, error):
    image = SpectralImage(np.zeros((3, 2, 5)), 'CYX')
    with pytest.raises(error):
        save_ome_tiff(image, tmp_path / 'bad.ome.tif', **options)
    assert not list(tmp_path.iterdir())


def test_invalid_inputs_and_unsupported_metadata_fail_before_writing(tmp_path):
    path = tmp_path / 'bad.ome.tif'
    with pytest.raises(TypeError, match='SpectralImage'):
        save_ome_tiff(np.zeros((3, 2, 5)), path)
    with pytest.raises(TypeError, match='SpectralImage'):
        save_ome_tiff(SpectralImage(np.zeros((3, 2, 5)), 'CYX').to_rgb(), path)
    with pytest.raises(TypeError, match='dtype'):
        save_ome_tiff(SpectralImage(np.zeros((3, 2, 5), dtype=np.int64), 'CYX'), path)
    image = SpectralImage(np.zeros((3, 2, 5)), 'CYX')
    with pytest.raises(ValueError, match='ome.tif'):
        save_ome_tiff(image, tmp_path / 'plain.tif')
    with pytest.raises(FileNotFoundError, match='directory'):
        save_ome_tiff(image, tmp_path / 'missing/file.ome.tif')
    for value in [object(), float('nan')]:
        image.provenance['unsupported'] = value
        with pytest.raises(ValueError, match='JSON-compatible'):
            save_ome_tiff(image, path)
    image.provenance.clear()
    image.provenance['channel_names'] = ['too few']
    with pytest.raises(ValueError, match='channel_names'):
        save_ome_tiff(image, path)
    assert not list(tmp_path.iterdir())


def test_automatic_bigtiff_for_large_virtual_array(tmp_path):
    # A broadcast view exercises the size threshold without allocating/writing
    # four GiB. This guards against classic-TIFF overflow for compressed input.
    data = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.uint16), (2, 32768, 32768))
    image = SpectralImage(data, 'CYX')
    with patch('tifffile.imwrite') as write:
        save_ome_tiff(image, tmp_path / 'large.ome.tif')
        assert write.call_args.kwargs['bigtiff'] is True
    with pytest.raises(ValueError, match='too large'):
        save_ome_tiff(image, tmp_path / 'classic.ome.tif', bigtiff=False)


def test_invalid_lumic_annotation_fails_before_decoding(tmp_path):
    from ome_types import to_xml
    path = save_ome_tiff(SpectralImage(np.zeros((3, 2, 5)), 'CYX'), tmp_path / 'bad.ome.tif')
    with tifffile.TiffFile(path) as tif:
        ome = from_xml(tif.ome_metadata)
    entry = ome.structured_annotations.map_annotations[0].value.ms[0]
    info = json.loads(entry.value)
    info['shape'][0] = 99
    entry.value = json.dumps(info)
    tifffile.tiffcomment(path, to_xml(ome, validate=True).encode('utf-8'))
    with patch.object(tifffile.TiffPage, 'asarray', side_effect=AssertionError('decoded invalid input')):
        with pytest.raises(ValueError, match='Lumic OME annotation'):
            open_ome_tiff(path)
