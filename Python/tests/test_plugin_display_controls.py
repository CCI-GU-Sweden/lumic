from types import SimpleNamespace

import numpy as np
from qtpy.QtWidgets import QApplication
from napari.components import ViewerModel
from skimage.color import rgb2lab

from plugin.plugin import SpectralUnmixingWidget, _distinct_component_colors


def test_abundance_controls_and_residual_plot():
    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    dock = SimpleNamespace(setVisible=lambda *a: None)
    def add_dock(controls, **kwargs):
        dock.controls = controls
        return dock
    object.__setattr__(viewer, "window", SimpleNamespace(add_dock_widget=add_dock))
    widget = SpectralUnmixingWidget(viewer)
    widget.abundance_layers = [viewer.add_image(np.ones((2, 2))) for _ in range(3)]
    widget.residual_layer = viewer.add_image(np.zeros((2, 2)), visible=False)
    widget._set_abundance_checks(3)
    widget._update_abundance_display()
    assert all(layer.visible and layer.blending == 'additive' for layer in widget.abundance_layers)
    widget.opaque_radio.setChecked(True)
    assert [layer.visible for layer in widget.abundance_layers] == [True, False, False]
    widget.abundance_checks[2].setChecked(True)
    assert viewer.layers.selection.active is widget.abundance_layers[2]
    assert [layer.visible for layer in widget.abundance_layers] == [False, False, True]
    assert all(layer.blending == 'opaque' for layer in widget.abundance_layers)
    widget.abundance_residual_check.setChecked(True)
    assert viewer.layers.selection.active is widget.residual_layer
    assert widget.residual_layer.visible
    assert not any(layer.visible for layer in widget.abundance_layers)
    widget.additive_radio.setChecked(True)
    widget.abundance_checks[0].setChecked(True)
    assert widget.residual_layer.visible
    assert [layer.visible for layer in widget.abundance_layers] == [True, False, False]
    widget.abundance_layers[1].visible = True
    assert widget.abundance_checks[1].isChecked()
    widget.abundance_layers[0].visible = False
    assert not widget.abundance_checks[0].isChecked()
    widget.residual_layer.visible = False
    assert not widget.abundance_residual_check.isChecked()
    widget.residual_layer.visible = True
    widget._hide_abundances()
    assert not any(layer.visible for layer in widget.abundance_layers)
    assert not widget.residual_layer.visible
    widget.opaque_radio.setChecked(True)
    widget.abundance_layers[2].visible = True
    assert widget.abundance_checks[2].isChecked() and not widget.residual_layer.visible
    widget._set_abundance_checks(3)
    assert len(widget._visibility_connections) == 4
    widget.image = SimpleNamespace(wavelengths_nm=np.array([450, 500]))
    widget.initial_spectra = np.ones((3, 2))
    widget.optimized_spectra = np.ones((3, 2))
    widget.residual_img = np.array([[[1., 3.]], [[-1., -3.]]])
    widget.wave = np.array([450, 500])
    widget._set_fit_components(3)
    widget.fit_residual_check.setChecked(True)
    assert widget.fit_ax.get_title() == 'Mean signed residual'
    np.testing.assert_array_equal(widget.fit_ax.lines[0].get_ydata(), [2, -2])
    widget.fit_component_checks[0].setChecked(True)
    assert widget.fit_ax.get_title() == 'Initial and optimized reference spectra'
    assert len(widget.fit_ax.lines) == 2
    widget.fit_component_checks[2].setChecked(True)
    assert len(widget.fit_ax.lines) == 4
    assert not widget.fit_all_check.isChecked() and not widget.fit_residual_check.isChecked()
    widget.fit_all_check.setChecked(True)
    assert not any(check.isChecked() for check in widget.fit_component_checks)
    assert len(widget.fit_ax.lines) == 6
    widget.fit_residual_check.setChecked(True)
    assert not widget.fit_all_check.isChecked()
    widget.tabs.setTabEnabled(3, True)
    assert widget.fit_selector_layout.itemAtPosition(0, 1).widget() is widget.fit_component_checks[0]
    assert widget.abundance_layout.itemAtPosition(1, 1).widget() is widget.abundance_residual_check
    widget.pca_result = SimpleNamespace(loadings=np.array([[1., 2.], [2., 1.]]),
                                        explained_variance_ratio=np.array([0.6, 0.4]))
    widget._set_pca_checks(2)
    widget._plot_pca()
    assert len(widget.pca_ax.get_legend_handles_labels()[0]) == 2
    assert [line.get_alpha() for line in widget.pca_ax.lines[:2]] == [1.0, 0.35]
    widget.pc_slider.setRange(1, 2)
    widget.pc_slider.setValue(2)
    assert [line.get_alpha() for line in widget.pca_ax.lines[:2]] == [0.35, 1.0]
    widget._reset_pca_highlight()
    assert [line.get_alpha() for line in widget.pca_ax.lines[:2]] == [0.7, 0.7]
    widget.pc_slider.setValue(1)
    assert [line.get_alpha() for line in widget.pca_ax.lines[:2]] == [1.0, 0.35]
    widget.pca_checks[0].setChecked(False)
    assert len(widget.pca_ax.get_legend_handles_labels()[0]) == 1
    widget.pca_checks[1].setChecked(False)
    assert not widget.pca_ax.get_legend_handles_labels()[0]
    widget.pca_result = None
    widget._set_pca_checks(0)
    widget.pc_slider.setRange(1, 3)
    widget.pc_next_button.click()
    assert widget.pc_slider.value() == 2
    widget.pc_previous_button.click()
    assert widget.pc_slider.value() == 1
    widget.deleteLater()


def test_component_colors_follow_winning_wavelength_bins():
    wavelengths = np.array([450, 480, 530, 560, 600, 630])
    spectra = np.array([[5, 1, 1, 0, 0, 0], [0, 0, 4, 3, 5, 0],
                        [4, 0, 6, 0, 1, 7], [1, 0, 1, 0, 0, 0]])
    colors = SpectralUnmixingWidget._component_colors(spectra, wavelengths)
    # Blue wins at 450, orange at 600, red at 630; the last component
    # wins no bin and falls back to its first overall peak at 450. Its duplicate
    # blue is replaced, preserving the colours assigned to earlier components.
    np.testing.assert_allclose(colors[:3], [[0, 0, 1], [1, 0.4333333333, 0],
                                           [1, 0, 0]], atol=1e-8)
    assert np.all(np.linalg.norm(rgb2lab(colors[:3]) - rgb2lab(colors[3]), axis=1) >= 25)
    fallback = SpectralUnmixingWidget._component_colors(spectra, None)
    np.testing.assert_allclose(fallback, SpectralUnmixingWidget._colors(4)[:, :3])


def test_similar_spectrum_colors_are_replaced_without_changing_earlier_choices():
    preferred = np.array([[0., 0., 1.], [0., .01, 1.], [0., 0., 1.], [0., 0., 1.]])
    before = preferred.copy()
    result = _distinct_component_colors(preferred)
    np.testing.assert_array_equal(preferred, before)
    np.testing.assert_array_equal(result[0], preferred[0])
    for count in range(1, len(result) + 1):
        # Adding a later colour never recolours an earlier assignment.
        np.testing.assert_array_equal(_distinct_component_colors(preferred[:count]), result[:count])
    lab = rgb2lab(result)
    for index in range(1, len(lab)):
        assert np.all(np.linalg.norm(lab[:index] - lab[index], axis=1) >= 25)


def test_overlapping_wavelength_peaks_receive_distinct_colors():
    spectra = np.array([[1., 3., 1.], [2., 6., 2.], [3., 9., 3.], [4., 12., 4.]])
    before = spectra.copy()
    colors = SpectralUnmixingWidget._component_colors(spectra, [600, 620, 650])
    np.testing.assert_array_equal(spectra, before)
    lab = rgb2lab(colors)
    for index in range(1, len(lab)):
        assert np.all(np.linalg.norm(lab[:index] - lab[index], axis=1) >= 25)
    np.testing.assert_array_equal(colors, SpectralUnmixingWidget._component_colors(spectra, [600, 620, 650]))
