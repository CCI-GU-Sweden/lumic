"""Component semantics, exact basis scale and ordinary OME interoperability."""
import json
from unittest.mock import patch

import numpy as np
from ome_types import from_xml, to_xml
import pytest
import tifffile

from lumic import SpectralImage
from lumic.io import load_image, open_image


def make_maps(dtype=np.float32):
    source = SpectralImage(np.zeros((2, 3, 2, 3, 5)), 'TCZYX',
        wavelengths_nm=[450, 530, 620],
        detection_bands_nm=[[440, 460], [520, 540], [610, 630]],
        pixel_sizes_um={'X': .1, 'Y': .2, 'Z': .5},
        provenance={'channel_names': ['blue', 'green', 'red']})
    coefficients = (np.arange(120).reshape(60, 2) / 7).astype(dtype)
    # Unequal row sums detect any accidental normalization during export.
    spectra = np.array([[.123456789, .4, .7], [2.3, .2, .9]], dtype=dtype)
    maps = source.unfold().fold_abundances(coefficients, component_names=['first', 'second'],
        spectra=spectra, normalization='none', abundance_units='arbitrary fitted units')
    return maps, coefficients, spectra


@pytest.mark.parametrize('axes,compression,dtype', [
    ('TKZYX', None, np.float32), ('YXTZK', 'deflate', np.float64),
    ('ZKXYT', 'lzw', np.float32)])
def test_maps_basis_and_reconstruction_roundtrip(tmp_path, axes, compression, dtype):
    maps, coefficients, spectra = make_maps(dtype)
    maps = SpectralImage(maps.data.transpose([maps.axes.index(a) for a in axes]), axes,
        kind='abundance', component_names=maps.component_names, unmixing=maps.unmixing,
        pixel_sizes_um=maps.pixel_sizes_um, provenance=maps.provenance)
    path = maps.save_ome_tiff(tmp_path / 'maps.ome.tif', compression=compression)
    with open_image(path) as reader:
        assert reader.axes == axes and 'K' in reader.dimension_ranges
        assert 'Components: 2' in reader.summary() and 'Detector channels:' not in reader.summary()
        loaded = reader.read(t=None, z=None)
        plane = reader.read(t=1, z=1)
    assert loaded.kind == 'abundance' and loaded.axes == axes
    assert loaded.component_names == ('first', 'second')
    assert loaded.wavelengths_nm is None and loaded.detection_bands_nm is None
    assert loaded.pixel_sizes_um == pytest.approx(maps.pixel_sizes_um)
    assert loaded.data.dtype == dtype and loaded.unmixing['spectra'].dtype == dtype
    np.testing.assert_array_equal(loaded.data, maps.data)
    np.testing.assert_array_equal(loaded.unmixing['spectra'], spectra)
    assert loaded.unmixing['normalization'] == 'none'
    assert loaded.unmixing['abundance_units'] == 'arbitrary fitted units'
    np.testing.assert_array_equal(plane.data, maps.slice(t=1, z=1).data)
    reconstructed = loaded.reconstruct()
    assert reconstructed.axes == axes.replace('K', 'C') and reconstructed.kind == 'reconstructed'
    expected = (coefficients @ spectra).reshape(2, 2, 3, 5, 3)
    expected = np.moveaxis(expected, -1, 1)  # TCZYX
    expected = expected.transpose(['TCZYX'.index(a) for a in reconstructed.axes])
    np.testing.assert_allclose(reconstructed.data, expected, rtol=2e-7)
    np.testing.assert_array_equal(reconstructed.wavelengths_nm, [450, 530, 620])
    assert reconstructed.provenance['channel_names'] == ('blue', 'green', 'red')
    reconstructed.save_ome_tiff(tmp_path / 'reconstructed.ome.tif')
    with tifffile.TiffFile(path) as tif:
        ome = from_xml(tif.ome_metadata, validate=True)
        pixels = ome.images[0].pixels
        assert pixels.size_c == 2
        assert [c.name for c in pixels.channels] == ['first', 'second']
        assert all(c.emission_wavelength is None for c in pixels.channels)
        order = ''.join(a for a in axes if a not in 'YX') + 'YX'
        assert tif.series[0].axes == order.replace('K', 'C')
        np.testing.assert_array_equal(tif.asarray(), maps.data.transpose([axes.index(a) for a in order]))


def test_maps_without_basis_and_unnamed_components(tmp_path):
    maps = SpectralImage(np.arange(10, dtype=np.uint16).reshape(1, 1, 2, 5, 1),
        'TKYXZ', kind='abundance', provenance={'channel_names': ['detector A', 'B', 'C']})
    path = maps.save_ome_tiff(tmp_path / 'unnamed.ome.tif')
    loaded = load_image(path, t=None, z=None)
    assert loaded.axes == maps.axes and loaded.component_names is None and loaded.unmixing is None
    np.testing.assert_array_equal(loaded.data, maps.data)
    with pytest.raises(ValueError, match='No component spectra'):
        loaded.reconstruct()
    with tifffile.TiffFile(path) as tif:
        assert from_xml(tif.ome_metadata).images[0].pixels.channels[0].name == 'Component 1'


def test_slice_copies_basis_and_reconstruction_preserves_source():
    maps, _, _ = make_maps()
    selected = maps.slice(t=0, z=0)
    selected.unmixing['spectra'][0, 0] = 99
    selected.unmixing['wavelengths_nm'][0] = 460
    assert maps.unmixing['spectra'][0, 0] != 99
    assert maps.unmixing['wavelengths_nm'][0] == 450
    before = maps.data.copy()
    maps.reconstruct()
    np.testing.assert_array_equal(maps.data, before)
    with pytest.raises(ValueError, match='abundance image'):
        SpectralImage(np.ones((3, 2, 5)), 'CYX').reconstruct()


@pytest.mark.parametrize('basis', [
    {}, {'spectra': [[1, 2, 3]]}, {'spectra': np.ones((2, 0))},
    {'spectra': [[1, np.nan], [2, 3]]}, {'spectra': np.ones((2, 3), dtype=complex)},
    {'spectra': np.ones((2, 3)), 'wavelengths_nm': [450, 530]},
    {'spectra': np.ones((2, 3)), 'detection_bands_nm': [[460, 440]] * 3},
    {'spectra': np.ones((2, 3)), 'channel_names': ['one']},
    {'spectra': np.ones((2, 3)), 'normalization': ''},
    {'spectra': np.ones((2, 3)), 'unknown': True}])
def test_invalid_basis_is_rejected(basis):
    with pytest.raises(ValueError):
        SpectralImage(np.zeros((2, 2, 5)), 'KYX', kind='abundance', unmixing=basis)


def test_fold_validates_source_channel_count_and_scale_labels():
    unfolded = SpectralImage(np.ones((3, 2, 5)), 'CYX').unfold()
    with pytest.raises(ValueError, match='spectra'):
        unfolded.fold_abundances(np.ones((10, 2)), spectra=np.ones((2, 4)))
    with pytest.raises(ValueError, match='spectra'):
        unfolded.fold_abundances(np.ones((10, 2)), normalization='L1')
    with pytest.raises(ValueError, match='only valid for abundance'):
        SpectralImage(np.ones((3, 2, 5)), 'CYX', unmixing={'spectra': np.ones((3, 3))})


def test_mutated_basis_revalidated_before_writing_or_reconstructing(tmp_path):
    maps, _, _ = make_maps()
    maps.unmixing['spectra'][0, 0] = np.inf
    with pytest.raises(ValueError, match='finite real'):
        maps.save_ome_tiff(tmp_path / 'bad.ome.tif')
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='finite real'):
        maps.reconstruct()


def test_annotation_validation_and_no_guessing_without_recognized_annotation(tmp_path):
    maps, _, _ = make_maps()
    good = maps.save_ome_tiff(tmp_path / 'good.ome.tif')
    with tifffile.TiffFile(good) as tif:
        ome = from_xml(tif.ome_metadata)
        data = tif.asarray()
    entry = ome.structured_annotations.map_annotations[0].value.ms[0]
    info = json.loads(entry.value)
    info['unmixing']['spectra'] = [[1, 2, 3]]
    entry.value = json.dumps(info)
    bad = tmp_path / 'bad.ome.tif'
    tifffile.imwrite(bad, data, description=to_xml(ome).encode('utf-8'), metadata=None, photometric='minisblack')
    with patch.object(tifffile.TiffPage, 'asarray', side_effect=AssertionError('decoded')):
        with pytest.raises(ValueError, match='Invalid Lumic OME annotation'):
            open_image(bad)
    ome.structured_annotations.map_annotations[0].namespace = 'urn:other:unknown'
    external = tmp_path / 'external.ome.tif'
    tifffile.imwrite(external, data, description=to_xml(ome).encode('utf-8'), metadata=None, photometric='minisblack')
    generic = load_image(external, t=None, z=None)
    assert generic.kind == 'measured' and 'C' in generic.axes and 'K' not in generic.axes
    assert generic.unmixing is None
