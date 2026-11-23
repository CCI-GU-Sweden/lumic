import json
import locale
import time
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

from lumic import SpectralImage, Spectrum
from plugin.plugin import SpectralUnmixingWidget
from plugin.spectra_io import load_spectra_json, save_spectra_json, spectrum_objects, validate_reference_spectra


@pytest.fixture
def widget(tmp_path):
    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    dock = SimpleNamespace(setVisible=lambda *a: None, close=lambda: None)
    def add_dock(controls, **kwargs):
        dock.controls = controls
        return dock
    object.__setattr__(viewer, "window", SimpleNamespace(add_dock_widget=add_dock))
    widget = SpectralUnmixingWidget(viewer)
    widget.image_path = tmp_path / "image.czi"
    image = SpectralImage(np.random.default_rng(42).uniform(1, 10, (3, 4, 5)), "CYX",
                          wavelengths_nm=[450., 530., 620.])
    widget._apply_plane(image, image.to_rgb(), 0, 0)
    yield widget
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("count", [1, 3])
def test_collection_round_trip_one_output_file(tmp_path, count):
    spectra = [Spectrum(np.array([1, .125, 3], dtype=np.float32), [450, 530, 620],
                        name=f"Dye {i}", provenance={"operator": "test", "laser": [405]})
               for i in range(count)]
    path = tmp_path / "spectra.json"
    save_spectra_json(spectra, path)
    restored = load_spectra_json(path)
    assert list(tmp_path.iterdir()) == [path]
    assert len(restored) == count
    for before, after in zip(spectra, restored):
        np.testing.assert_array_equal(before.values, after.values)
        np.testing.assert_array_equal(before.wavelengths_nm, after.wavelengths_nm)
        assert before.values.dtype == after.values.dtype
        assert before.name == after.name and before.provenance == after.provenance


def test_imports_single_backend_spectrum(tmp_path):
    path = tmp_path / "single.json"
    Spectrum([2, 1]).save_json(path)
    restored, = load_spectra_json(path)
    np.testing.assert_array_equal(restored.values, [2, 1])
    assert restored.wavelengths_nm is None


@pytest.mark.parametrize("sizes", [{}, {"X": .2}, {"X": .2, "Y": .3, "Z": .7}])
@pytest.mark.parametrize("stage", ["initial", "optimized"])
def test_saved_spectra_retain_available_image_calibration(tmp_path, sizes, stage):
    image = SpectralImage(np.ones((3, 2, 2)), "CYX", pixel_sizes_um=sizes)
    spectra = spectrum_objects(np.array([[1., 2., 3.]], dtype=np.float32), image,
                               "input.czi", (0, 0), stage, "Manual")
    path = tmp_path / "spectra.json"
    save_spectra_json(spectra, path)
    restored, = load_spectra_json(path)
    assert restored.values.dtype == np.float32
    if sizes:
        assert restored.provenance["pixel_sizes_um"] == sizes
        assert restored.provenance["pixel_unit"] == "micrometer"
    else:
        assert "pixel_sizes_um" not in restored.provenance
        assert "pixel_unit" not in restored.provenance


@pytest.mark.parametrize("payload", [
    {}, [], {"format": "lumic-spectra", "version": 2, "spectra": []},
    {"format": "lumic-spectra", "version": 1, "spectra": []},
    {"format": "lumic-spectra", "version": 1, "spectra": [{}]},
])
def test_rejects_invalid_collection(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_spectra_json(path)


@pytest.mark.parametrize(("values", "wavelengths", "provenance"), [
    ([1, 2], [450, 530], {}),
    ([1, 2, 3], [620, 530, 450], {}),
    ([1, 2, 3], None, {}),
    ([-1, 2, 3], [450, 530, 620], {}),
    ([0, 0, 0], [450, 530, 620], {}),
    ([1, 2, 3], [450, 530, 620], {"detection_bands_nm": [[440, 460]] * 3}),
])
def test_rejects_incompatible_reference(values, wavelengths, provenance):
    image = SpectralImage(np.ones((3, 2, 2)), "CYX", wavelengths_nm=[450, 530, 620])
    with pytest.raises(ValueError):
        validate_reference_spectra([Spectrum(values, wavelengths, provenance=provenance)], image)


def test_channel_only_references_require_matching_count():
    image = SpectralImage(np.ones((3, 2, 2)), "CYX")
    validate_reference_spectra([Spectrum([3, 1, 2])], image)
    with pytest.raises(ValueError, match="wavelength"):
        validate_reference_spectra([Spectrum([3, 1, 2], [450, 530, 620])], image)


def test_json_load_export_and_failed_replacement(widget, tmp_path, monkeypatch):
    from dataclasses import replace
    widget.image = replace(widget.image, pixel_sizes_um={"X": .2, "Y": .3})
    widget.reference_mode.setCurrentText("Load JSON")
    spectra = [Spectrum([1., 2., 3.], widget.image.wavelengths_nm, name="Dye A"),
               Spectrum([3., 2., 1.], widget.image.wavelengths_nm, name="Dye B")]
    path = tmp_path / "references.json"
    save_spectra_json(spectra, path)
    monkeypatch.setattr(widget, "_refresh_distance_layers", lambda: True)
    widget.load_reference_json(path)
    assert widget.components_input.value() == 2
    assert not widget.components_input.isEnabled()
    assert widget.tabs.isTabEnabled(2) and widget.abundance_button.isEnabled()
    assert [s.name for s in widget.initial_spectrum_objects] == ["Dye A", "Dye B"]
    np.testing.assert_array_equal(widget.initial_spectra, [s.values for s in spectra])
    assert widget.pool_indices is not None
    output = tmp_path / "exported"
    monkeypatch.setattr("plugin.plugin.QFileDialog.getSaveFileName", lambda *a, **k: (str(output), ""))
    widget._export_reference_spectra("initial")
    restored = load_spectra_json(output.with_suffix(".json"))
    np.testing.assert_array_equal([s.values for s in restored], widget.initial_spectra)
    assert all(s.provenance["pixel_sizes_um"] == {"X": .2, "Y": .3} for s in restored)
    assert all(s.provenance["pixel_unit"] == "micrometer" for s in restored)
    assert widget.open_button.toolTip() == "Import your desired image"
    assert widget.save_initial_button.text() == "Save initial reference spectra..."
    assert widget.save_optimized_button.text() == "Save optimised spectra..."
    assert "JSON" in widget.save_optimized_button.toolTip()

    # Rejected imports leave the existing references and fit readiness intact.
    warnings = []
    monkeypatch.setattr(widget, "_warn", warnings.append)
    Spectrum([1, 2], [450, 530]).save_json(path)
    widget.load_reference_json(path)
    assert warnings and "channel count" in warnings[0]
    np.testing.assert_array_equal(widget.initial_spectra, [s.values for s in spectra])
    assert not widget.is_busy and widget.abundance_button.isEnabled()
    widget.reset_all()
    assert not widget.initial_spectrum_objects and not widget.optimized_spectrum_objects
    assert not widget.tabs.isTabEnabled(2)


def test_manual_selection_creates_objects_without_rescaling(widget, monkeypatch):
    widget.reference_mode.setCurrentText("Manual")
    widget.components_input.setValue(4)
    widget.manual_spectra = [np.array([10., 20., 30.]), np.array([5., 7., 11.])]
    monkeypatch.setattr(widget, "_refresh_distance_layers", lambda: True)
    widget.confirm_manual_references()
    np.testing.assert_array_equal([s.values for s in widget.initial_spectrum_objects], widget.manual_spectra)
    assert all(s.provenance["reference_mode"] == "Manual" for s in widget.initial_spectrum_objects)


def test_manual_count_is_selected_by_accepting_and_confirming(widget, monkeypatch):
    widget.reference_mode.setCurrentText("Manual")
    widget.components_input.setValue(1)
    monkeypatch.setattr(widget, "_refresh_distance_layers", lambda: True)
    assert widget.components_controls.isHidden()
    assert not widget.confirm_manual_button.isEnabled()
    for y, x in ((0, 0), (1, 1), (2, 2)):
        widget.viewer.layers.selection.active = widget.raw_layer
        widget._capture_manual_spectrum(widget.viewer, SimpleNamespace(
            type="mouse_press", position=widget.raw_layer.data_to_world((y, x))))
        widget.accept_manual_spectrum()
        assert widget.confirm_manual_button.isEnabled()
    widget.confirm_manual_references()
    assert widget.initial_spectra.shape == (3, 3)
    np.testing.assert_array_equal(widget.initial_spectra, np.stack(widget.manual_spectra))
    assert not widget.confirm_manual_button.isEnabled()
    # Merely browsing preserves the selection; accepting adds another reference
    # and requires confirmation again, independently of the automatic count.
    previous = widget.initial_spectra.copy()
    widget._capture_manual_spectrum(widget.viewer, SimpleNamespace(
        type="mouse_press", position=widget.raw_layer.data_to_world((3, 3))))
    np.testing.assert_array_equal(widget.initial_spectra, previous)
    widget.accept_manual_spectrum()
    assert widget.initial_spectra is None and widget.confirm_manual_button.isEnabled()
    widget.confirm_manual_references()
    assert widget.initial_spectra.shape == (4, 3)
    widget.clear_manual_spectra()
    assert not widget.manual_spectra and not widget.confirm_manual_button.isEnabled()
    widget.reference_mode.setCurrentText("Automatic")
    assert not widget.components_controls.isHidden() and widget.components_input.isEnabled()


def test_file_dialogs_start_in_loaded_image_folder(widget, tmp_path, monkeypatch):
    from pathlib import Path
    from qtpy.QtCore import QSettings
    folder = tmp_path / "acquisition"
    folder.mkdir()
    source = folder / "image.czi"
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    widget.settings = settings
    image = widget.image
    class Reader:
        axes = "CYX"
        shape = image.data.shape
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def summary(self): return "test image"
        def read(self, **kwargs): return image
    monkeypatch.setattr("plugin.plugin.open_image", lambda path: Reader())
    widget.load_czi(source)
    assert widget.last_image_folder == folder
    assert settings.value("spectral_unmixing/last_image_folder") == str(folder)
    dialogs = []
    monkeypatch.setattr("plugin.plugin.QFileDialog.getOpenFileName", lambda *a, **k: (dialogs.append(a) or "", ""))
    widget.browse_czi()
    widget.browse_spectra_json()
    assert [Path(args[2]) for args in dialogs] == [folder, folder]
    widget.reset_all()
    assert widget.last_image_folder == folder

    # A new plugin instance uses the saved folder under a stable settings scope.
    calls = []
    monkeypatch.setattr("plugin.plugin.QSettings", lambda *args: (calls.append(args) or settings))
    viewer = ViewerModel()
    dock = SimpleNamespace(setVisible=lambda *a: None, close=lambda: None)
    def add_dock(controls, **kwargs):
        dock.controls = controls
        return dock
    object.__setattr__(viewer, "window", SimpleNamespace(add_dock_widget=add_dock))
    reopened = SpectralUnmixingWidget(viewer)
    try:
        assert calls == [("Lumic", "napari-lumic-cci")]
        assert reopened.last_image_folder == folder
    finally:
        reopened.close()


@pytest.fixture
def comma_numeric_locale():
    previous = locale.setlocale(locale.LC_NUMERIC)
    for candidate in ("sv_SE.UTF-8", "sv_SE.utf8", "Swedish_Sweden.1252"):
        try:
            locale.setlocale(locale.LC_NUMERIC, candidate)
            break
        except locale.Error:
            continue
    else:
        pytest.skip("A Swedish numeric locale is not installed")
    try:
        yield locale.setlocale(locale.LC_NUMERIC)
    finally:
        locale.setlocale(locale.LC_NUMERIC, previous)


def test_ome_loading_and_plane_reload_with_comma_locale(widget, tmp_path, comma_numeric_locale, monkeypatch):
    from dataclasses import replace
    from plugin.plugin import _image_read_locale
    image = replace(widget.image, pixel_sizes_um={"X": .07066, "Y": .07066})
    path = tmp_path / "calibrated.ome.tiff"
    # Write a valid fixture regardless of the native XML library's locale bug.
    with _image_read_locale():
        image.save_ome_tiff(path)
    warnings = []
    monkeypatch.setattr(widget, "_warn", warnings.append)
    widget.load_czi(path)
    assert not warnings
    assert widget.image_path == path
    assert widget.image.pixel_sizes_um == image.pixel_sizes_um
    np.testing.assert_array_equal(widget.image.data, image.data)
    assert locale.setlocale(locale.LC_NUMERIC) == comma_numeric_locale
    widget._select_plane()
    assert not warnings
    np.testing.assert_array_equal(widget.image.data, image.data)
    assert locale.setlocale(locale.LC_NUMERIC) == comma_numeric_locale


@pytest.mark.parametrize("action", ["load", "select_plane"])
def test_image_read_failure_restores_locale_and_preserves_image(widget, comma_numeric_locale, monkeypatch, action):
    previous_image = widget.image
    warnings = []
    def invalid_reader(path):
        assert locale.setlocale(locale.LC_NUMERIC) == "C"
        raise ValueError("Invalid OME-XML metadata")
    monkeypatch.setattr("plugin.plugin.open_image", invalid_reader)
    monkeypatch.setattr(widget, "_warn", warnings.append)
    if action == "load":
        widget.load_czi("invalid.ome.tiff")
    else:
        widget._select_plane()
    assert warnings and "Invalid OME-XML metadata" in warnings[0]
    assert widget.image is previous_image
    assert locale.setlocale(locale.LC_NUMERIC) == comma_numeric_locale


def test_automatic_selection_creates_objects_without_rescaling(widget, monkeypatch):
    widget.reference_mode.setCurrentText("Automatic")
    widget.components_input.setValue(1)
    monkeypatch.setattr(widget, "_refresh_distance_layers", lambda: True)
    widget.calculate_optimized_spectra()
    assert len(widget.initial_spectrum_objects) == 1
    np.testing.assert_array_equal(widget.initial_spectrum_objects[0].values, widget.initial_spectra[0])


@pytest.mark.parametrize("snr", ["Low", "Medium", "High"])
@pytest.mark.parametrize("fails", [False, True])
def test_example_download_runs_off_ui_thread(widget, monkeypatch, tmp_path, snr, fails):
    import threading
    calls, loaded, warnings = [], [], []
    ui_thread = threading.get_ident()
    path = tmp_path / "example.czi"

    def download(**kwargs):
        calls.append((kwargs, threading.get_ident()))
        if fails:
            raise OSError("download unavailable")
        return [path]

    original_image = widget.image
    monkeypatch.setattr("plugin.plugin.download_cos7_data", download)
    monkeypatch.setattr(widget, "load_czi", loaded.append)
    monkeypatch.setattr(widget, "_warn", warnings.append)
    widget.example_snr.setCurrentText(snr)
    widget.load_example_data()
    assert widget.is_busy and not widget.example_button.isEnabled()
    deadline = time.monotonic() + 10
    while widget.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert not widget.is_busy and widget.example_button.isEnabled()
    assert calls[0][0] == {"snr": snr.lower(), "progress": True}
    assert calls[0][1] != ui_thread
    assert loaded == ([] if fails else [path])
    assert bool(warnings) == fails
    assert widget.image is original_image
