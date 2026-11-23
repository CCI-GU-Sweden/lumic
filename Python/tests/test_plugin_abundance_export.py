from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from ome_types import from_xml
import tifffile

from lumic import SpectralImage, Spectrum
from lumic.io import load_image
from plugin.plugin import SpectralUnmixingWidget
from plugin.spectra_io import load_spectra_json


def test_manual_reference_keeps_measured_intensity():
    spectrum = np.array([10., 20., 30.])
    noop = lambda *args, **kwargs: None
    widget = SimpleNamespace(is_busy=False, reference_mode=SimpleNamespace(currentText=lambda: 'Manual'),
        initial_spectra=None,
        pending_manual_spectrum=spectrum, pending_manual_point=(1, 2), manual_spectra=[], manual_points=[],
        reference_points_layer=None, _points_layer=noop, _colors=noop, _plot_references=noop,
        components_input=SimpleNamespace(value=lambda: 2), manual_status_label=SimpleNamespace(setText=noop),
        viewer=SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=None))),
        raw_layer=None, _restore_action_states=noop, _set_status=noop, _warn=lambda message: pytest.fail(message))
    SpectralUnmixingWidget.accept_manual_spectrum(widget)
    np.testing.assert_array_equal(widget.manual_spectra[0], spectrum)


@pytest.mark.parametrize('mode', ['Manual', 'Automatic', 'Load JSON'])
def test_plugin_fit_and_export_roundtrip(tmp_path, monkeypatch, mode):
    spectra = np.array([[1., 2., 1.], [3., 1., 2.]])
    coefficients = np.array([[1., 2.], [2., 0.], [0., 3.], [4., 1.]])
    image = SpectralImage((coefficients @ spectra).T.reshape(3, 2, 2), 'CYX',
        wavelengths_nm=[450, 530, 620], detection_bands_nm=[[440, 460], [520, 540], [610, 630]],
        pixel_sizes_um={'X': .2, 'Y': .3}, provenance={'channel_names': ['blue', 'green', 'red']})
    warnings, layers, layer_options = [], [], []
    unrelated = SimpleNamespace(visible=True)
    viewer_layers = [unrelated]
    event = SimpleNamespace(connect=lambda *args: None)
    def layer(old, data, name, **kwargs):
        result = SimpleNamespace(data=data, events=SimpleNamespace(visible=event), metadata={}, visible=kwargs.get("visible", True))
        layers.append(result)
        layer_options.append((name, kwargs))
        viewer_layers.append(result)
        return result
    noop = lambda *args, **kwargs: None
    def hide_layers(items):
        for item in items:
            item.visible = False
    widget = SimpleNamespace(is_busy=False, initial_spectra=spectra, image=image,
        initial_spectrum_objects=[Spectrum(row, image.wavelengths_nm, name=f"Component {i + 1}")
                                  for i, row in enumerate(spectra)], plane_position=(0, 0),
        image_path=Path("/tmp/source/input.tif"), viewer=SimpleNamespace(layers=viewer_layers),
        reference_mode=SimpleNamespace(currentText=lambda: mode), pool_indices=np.arange(4),
        pools_input=SimpleNamespace(value=lambda: 1), channel_slider=SimpleNamespace(value=lambda: 0),
        abundance_layers=[], residual_layer=None, _set_busy=noop, _clear_pca=noop,
        _image_layer=layer, _hide_layers=hide_layers, _plot_fit=noop, _set_fit_components=noop, _refresh_distance_layers=noop,
        _set_abundance_checks=noop, _update_abundance_display=noop, _plot_references=noop,
        additive_radio=SimpleNamespace(setChecked=noop),
        _component_colors=SpectralUnmixingWidget._component_colors,
        _update_channel_slider_state=noop, fit_status_label=SimpleNamespace(setText=noop),
        _set_status=noop, _warn=warnings.append, file_label=SimpleNamespace(text=lambda: 'input.tif'))
    SpectralUnmixingWidget.calculate_abundance_map(widget)
    assert not warnings
    assert not unrelated.visible
    assert all(layer.visible for layer in layers[:2])
    assert not widget.residual_layer.visible
    residual_options = next(options for name, options in layer_options if name == "Residual")
    assert residual_options["colormap"] == "twilight"
    assert residual_options["auto_contrast"] is True
    maps = widget.abundance_img
    assert maps.kind == 'abundance' and maps.axes == 'KYX'
    np.testing.assert_array_equal(layers[0].data, maps.data[0])
    np.testing.assert_array_equal(maps.unmixing['spectra'], widget.optimized_spectra)
    np.testing.assert_array_equal([s.values for s in widget.optimized_spectrum_objects], maps.unmixing['spectra'])
    assert not np.allclose(widget.optimized_spectra.sum(axis=1), 1)
    if mode == 'Manual':
        np.testing.assert_array_equal(widget.optimized_spectra, spectra)
    path = tmp_path / 'maps.ome.tif'
    save_dialog_args = []
    monkeypatch.setattr('plugin.plugin.QFileDialog.getSaveFileName', lambda *a, **k: (save_dialog_args.append(a) or str(path), ''))
    SpectralUnmixingWidget._export_abundance_image(widget)
    assert not warnings
    assert save_dialog_args[0][2] == "/tmp/source/input_abundances.ome.tif"
    loaded = load_image(path)
    assert loaded.kind == 'abundance' and loaded.axes == 'KYX'
    assert loaded.component_names == ('Component 1', 'Component 2')
    assert loaded.pixel_sizes_um == maps.pixel_sizes_um
    assert loaded.unmixing['normalization'] == 'none' and loaded.unmixing['abundance_units'] is None
    for key in ('spectra', 'wavelengths_nm', 'detection_bands_nm', 'channel_names'):
        np.testing.assert_array_equal(loaded.unmixing[key], maps.unmixing[key])
    np.testing.assert_array_equal(loaded.data, maps.data)
    np.testing.assert_allclose(loaded.reconstruct().data, image.data - widget.residual_img, atol=1e-12)
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == 'CYX'
        assert from_xml(tif.ome_metadata, validate=True).images[0].pixels.size_c == 2
        assert tif.pages[0].compression.name in ('DEFLATE', 'ADOBE_DEFLATE')

    # Refusing replacement must leave the original export intact.
    before = path.read_bytes()
    monkeypatch.setattr('plugin.plugin.QMessageBox.question', lambda *a, **k: 0)
    SpectralUnmixingWidget._export_abundance_image(widget)
    assert path.read_bytes() == before

    monkeypatch.setattr('plugin.plugin.QMessageBox.question', lambda *a, **k: __import__('qtpy').QtWidgets.QMessageBox.Yes)
    SpectralUnmixingWidget._export_abundance_image(widget)
    np.testing.assert_array_equal(load_image(path).data, maps.data)

    widget.wave = image.wavelengths_nm
    widget.optimized_spectra = np.zeros_like(widget.optimized_spectra)
    json_path = tmp_path / 'spectra.json'
    save_dialog_args.clear()
    monkeypatch.setattr('plugin.plugin.QFileDialog.getSaveFileName', lambda *a, **k: (save_dialog_args.append(a) or str(json_path), ''))
    SpectralUnmixingWidget._export_reference_spectra(widget, 'optimized')
    assert save_dialog_args[0][2] == '/tmp/source/input_optimized_spectra.json'
    np.testing.assert_array_equal([s.values for s in load_spectra_json(json_path)], maps.unmixing['spectra'])
