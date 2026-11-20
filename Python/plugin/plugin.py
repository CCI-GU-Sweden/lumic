from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import locale
locale.setlocale(locale.LC_NUMERIC, "C")

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from multiprocess import Pool
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QRadioButton,
    QLabel, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)
from threadpoolctl import threadpool_limits

from lumic import (
    SpectralImage,
    essential_spectral_pixles, mcr_als,
    positive_least_squares, spectral_uniqueness_map, unique_profiles,
)
from lumic.evaluation import residual_pca_from_residuals
from lumic.io import open_image
from lumic.utils import select_analysis_pixels


RAW_LAYER_NAME = "Raw spectral image"


def canonicalize_detector_plane(image):
    """Put a selected detector plane in the plugin's channel-first order."""
    if not isinstance(image, SpectralImage):
        raise TypeError("The reader must return a SpectralImage.")
    if image.kind not in ("measured", "reconstructed"):
        raise ValueError(f"Unsupported image kind {image.kind!r}; detector-channel images are required.")
    if len(image.axes) != 3 or set(image.axes) != set("CYX"):
        raise ValueError(f"Expected a selected detector plane with C, Y and X axes; received {image.axes}.")
    return replace(image, data=image.data.transpose(tuple(image.axes.index(a) for a in "CYX")), axes="CYX")


def _plane_transform(image, *, stack=False):
    """Use physical coordinates only when both spatial axes are calibrated."""
    sizes = image.pixel_sizes_um
    calibrated = "Y" in sizes and "X" in sizes
    spatial_scale = (sizes["Y"], sizes["X"]) if calibrated else (1, 1)
    spatial_units = ("micrometer", "micrometer") if calibrated else ("pixel", "pixel")
    return {
        "scale": ((1,) if stack else ()) + spatial_scale,
        "units": (("pixel",) if stack else ()) + spatial_units,
    }


class SpectralUnmixingWidget(QWidget):
    """Plane-based spectral selection, ALS fitting and signed residual diagnostics."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image = None
        self.image_path = None
        self.last_image_folder = Path(QSettings().value("spectral_unmixing/last_image_folder", str(Path.home()), type=str))
        self.plane_position = (0, 0)
        self.raw_img = None
        self.wave = None
        self.spectral_axis_label = "Wavelength (nm)"
        self.initial_spectra = None
        self.optimized_spectra = None
        self.manual_spectra = []
        self.manual_points = []
        self.pending_manual_spectrum = None
        self.pending_manual_point = None
        self.pool_indices = None
        self.abundance_img = None
        self.residual_img = None
        self.pca_result = None
        self.distance_basis = None
        self.is_busy = False
        self.raw_layer = None
        self.rgb_layer = None
        self.rgb_rendering = None
        self.hull_layer = None
        self.reference_points_layer = None
        self.initial_distance_layer = None
        self.optimized_distance_layer = None
        self.residual_layer = None
        self.abundance_layers = []
        self.pca_layer = None
        self.controls_dock = None
        self._build_ui()
        self._build_bottom_controls()
        self._connect_events()
        self._restore_action_states()
        self.controls_dock.setVisible(False)

    def _tab(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return page, layout

    @staticmethod
    def _row(layout, title, control):
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        row.addWidget(control)
        layout.addLayout(row)

    @staticmethod
    def _plot(layout, title, ylabel, height=2.8):
        figure = Figure(figsize=(4, height))
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(int(height * 85))
        ax = figure.add_subplot(111)
        ax.set(title=title, xlabel="Wavelength (nm)", ylabel=ylabel)
        figure.tight_layout()
        layout.addWidget(canvas)
        return figure, canvas, ax

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.reset_button = QPushButton("Reset All")
        self.reset_button.setToolTip("Clear all viewer layers and restore the plugin's startup state.")
        self.tabs.setCornerWidget(self.reset_button, Qt.TopRightCorner)
        self.input_tab, inputs = self._tab("Input")
        self.open_button = QPushButton("Browse image...")
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Image reader summary appears after loading.")
        inputs.addWidget(self.open_button)
        inputs.addWidget(self.file_label)
        inputs.addWidget(self.summary_text)
        self.plane_controls = {}
        for axis, title in (("T", "Time"), ("Z", "Z plane")):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Horizontal)
            label = QLabel("0")
            row_layout.addWidget(QLabel(title))
            row_layout.addWidget(slider)
            row_layout.addWidget(label)
            row.setVisible(False)
            inputs.addWidget(row)
            self.plane_controls[axis] = (row, slider, label)
        inputs.addStretch()

        self.selection_tab, selection = self._tab("Spectra selection")
        self.reference_mode = QComboBox()
        self.reference_mode.addItems(["Select reference mode", "Automatic", "Manual"])
        self._row(selection, "References", self.reference_mode)
        self.components_input = QSpinBox()
        self.components_input.setRange(1, 64)
        self.components_input.setValue(4)
        self._row(selection, "Number of components", self.components_input)
        self.automatic_controls = QWidget()
        automatic = QVBoxLayout(self.automatic_controls)
        self.quantile_input = QDoubleSpinBox()
        self.quantile_input.setRange(0, 1)
        self.quantile_input.setDecimals(3)
        self.quantile_input.setSingleStep(0.05)
        self.quantile_input.setValue(0.2)
        self.quantile_input.setToolTip(
            "Exclude the bottom quantile of per-pixel median channel intensity "
            "from reference selection. Full-image fitting still uses every pixel."
        )
        self._row(automatic, "Median Intensity Quantile", self.quantile_input)
        self.optimize_button = QPushButton("Calculate initial reference spectra")
        automatic.addWidget(self.optimize_button)
        selection.addWidget(self.automatic_controls)

        self.manual_controls = QWidget()
        manual = QVBoxLayout(self.manual_controls)
        buttons = QHBoxLayout()
        self.accept_manual_button = QPushButton("Set clicked spectrum")
        self.clear_manual_button = QPushButton("Clear manual spectra")
        buttons.addWidget(self.accept_manual_button)
        buttons.addWidget(self.clear_manual_button)
        manual.addLayout(buttons)
        self.manual_status_label = QLabel("Click a pixel in the raw-data layer.")
        self.manual_status_label.setWordWrap(True)
        manual.addWidget(self.manual_status_label)
        selection.addWidget(self.manual_controls)
        self.live_figure, self.live_canvas, self.live_ax = self._plot(
            selection, "Live pixel spectrum", "Intensity",
        )
        self.figure, self.canvas, self.ax = self._plot(
            selection, "Initial reference spectra", "Spectral intensity",
        )
        self.confirm_manual_button = QPushButton("Confirm manual references")
        self.confirm_manual_button.setToolTip(
            "Review the selected spectra, then confirm to calculate the initial distance layer and enable unmixing."
        )
        selection.addWidget(self.confirm_manual_button)
        self.save_initial_button = QPushButton("Save initial reference spectra as CSVs...")
        self.save_initial_button.setToolTip(
            "Save one CSV per component, with wavelength_nm and spectral_intensity columns. "
            "Uses detector_channel_index when wavelength metadata is unavailable."
        )
        selection.addWidget(self.save_initial_button)
        selection.addStretch()

        self.unmixing_tab, unmixing = self._tab("Unmixing")
        self.pools_input = QSpinBox()
        self.pools_input.setRange(1, 64)
        self.pools_input.setValue(4)
        self._row(unmixing, "Worker processes", self.pools_input)
        self.abundance_button = QPushButton("Optimize spectra and calculate abundance maps")
        unmixing.addWidget(self.abundance_button)
        self.optimized_plot_scale = QComboBox()
        self.optimized_plot_scale.addItems(["Unnormalized", "Normalized (L1)"])
        self.optimized_plot_scale.setToolTip(
            "Change the optimized spectra display only; fitted values and exports are unchanged."
        )
        self._row(unmixing, "Optimized spectra display", self.optimized_plot_scale)
        self.fit_selector_group = QGroupBox("Show spectra")
        self.fit_selector_group.setToolTip(
            "Show all components, the mean signed residual alone, or any combination of individual components."
        )
        self.fit_selector_layout = QGridLayout(self.fit_selector_group)
        self.fit_all_check = QCheckBox("All components")
        self.fit_all_check.setChecked(True)
        self.fit_selector_layout.addWidget(self.fit_all_check, 0, 0)
        self.fit_component_checks = []
        self.fit_residual_check = QCheckBox("Mean signed residual")
        self.fit_selector_layout.addWidget(self.fit_residual_check, 0, 1)
        unmixing.addWidget(self.fit_selector_group)
        self.fit_figure, self.fit_canvas, self.fit_ax = self._plot(
            unmixing, "Initial and optimized reference spectra", "Spectral intensity", 3.6,
        )
        self.abundance_group = QGroupBox("Layer visibility")
        self.abundance_group.setToolTip("Choose abundance maps and the residual to show. Opaque blending permits one layer at a time; additive blending combines selected layers.")
        self.abundance_layout = QGridLayout(self.abundance_group)
        self.abundance_checks = []
        self.abundance_residual_check = None
        self._visibility_connections = []
        self.additive_radio = QRadioButton("Additive")
        self.opaque_radio = QRadioButton("Opaque")
        self.additive_radio.setToolTip("Combine the selected layers by adding their colors.")
        self.opaque_radio.setToolTip("Display at most one selected layer at a time.")
        self.additive_radio.setChecked(True)
        unmixing.addWidget(self.abundance_group)
        self.hide_abundances_button = QPushButton("Deselect all layers")
        self.hide_abundances_button.setToolTip("Hide all abundance components and the residual.")
        unmixing.addWidget(self.hide_abundances_button)
        unmixing.addWidget(QLabel("Layer view blending"))
        blend_row = QHBoxLayout()
        blend_row.addWidget(self.additive_radio)
        blend_row.addWidget(self.opaque_radio)
        unmixing.addLayout(blend_row)
        self.save_optimized_button = QPushButton("Save optimized reference spectra as CSVs...")
        self.save_optimized_button.setToolTip(self.save_initial_button.toolTip())
        unmixing.addWidget(self.save_optimized_button)
        self.save_abundance_button = QPushButton("Save abundance maps as OME-TIFF...")
        self.save_abundance_button.setToolTip(
            "Save quantitative maps with their fitted spectra for reconstruction. "
            "Components are anonymous and abundances have no assumed physical units."
        )
        unmixing.addWidget(self.save_abundance_button)
        self.fit_status_label = QLabel("")
        self.fit_status_label.setWordWrap(True)
        unmixing.addWidget(self.fit_status_label)
        unmixing.addStretch()

        self.evaluation_tab, evaluation = self._tab("Evaluation")
        self.pca_components_input = QSpinBox()
        self.pca_components_input.setRange(1, 6)
        self.pca_components_input.setValue(6)
        self._row(evaluation, "Residual principal components", self.pca_components_input)
        self.pc_controls = QWidget()
        pc_layout = QHBoxLayout(self.pc_controls)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.addWidget(QLabel("Residual PC"))
        self.pc_slider = QSlider(Qt.Horizontal)
        self.pc_slider.setRange(1, 1)
        self.pc_slider.setToolTip("Select which residual principal component score map to display.")
        self.pc_previous_button = QPushButton("◀")
        self.pc_previous_button.setToolTip("Show the previous residual principal component.")
        self.pc_next_button = QPushButton("▶")
        self.pc_next_button.setToolTip("Show the next residual principal component.")
        pc_layout.addWidget(self.pc_previous_button)
        pc_layout.addWidget(self.pc_slider)
        pc_layout.addWidget(self.pc_next_button)
        self.pc_label = QLabel("PC 1")
        pc_layout.addWidget(self.pc_label)
        self.pc_controls.setVisible(False)
        self.evaluate_button = QPushButton("Calculate residual PCA")
        evaluation.addWidget(self.evaluate_button)
        evaluation.addWidget(self.pc_controls)
        self.pca_selector_group = QGroupBox("Plot PC loadings")
        self.pca_selector_group.setToolTip("Select loading curves to show; the score map slider is independent.")
        self.pca_selector_layout = QGridLayout(self.pca_selector_group)
        self.pca_checks = []
        self.pca_highlight_selected = True
        self.pca_selector_group.setVisible(False)
        evaluation.addWidget(self.pca_selector_group)
        self.pca_reset_highlight_button = QPushButton("Reset line emphasis")
        self.pca_reset_highlight_button.setToolTip("Show all plotted loadings with equal opacity. Moving the PC slider highlights its loading again.")
        self.pca_reset_highlight_button.setVisible(False)
        evaluation.addWidget(self.pca_reset_highlight_button)
        self.pca_figure, self.pca_canvas, self.pca_ax = self._plot(
            evaluation, "Residual principal component loadings", "Signed unit-L2 loading", 3.6,
        )
        note = QLabel("Signed score maps and loadings describe unexplained variation, not component abundances.")
        note.setWordWrap(True)
        evaluation.addWidget(note)
        evaluation.addStretch()

        self.coord_label = QLabel("y=-, x=-")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.coord_label)
        layout.addWidget(self.status_label)

    def _build_bottom_controls(self):
        self.channel_slider = QSlider(Qt.Horizontal)
        self.channel_slider.setRange(0, 0)
        self.channel_slider.setEnabled(False)
        self.channel_slider.setToolTip("Select and show the raw-data or Residual layer to change wavelength.")
        self.channel_label = QLabel("Channel: -")
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.addWidget(QLabel("Wavelength"))
        layout.addWidget(self.channel_slider)
        layout.addWidget(self.channel_label)
        self.controls_dock = self.viewer.window.add_dock_widget(
            controls, name="Lambda channel", area="bottom",
        )

    def _connect_events(self):
        self.open_button.clicked.connect(self.browse_czi)
        self.reset_button.clicked.connect(self.reset_all)
        self.reference_mode.currentTextChanged.connect(self._reset_reference_state)
        self.components_input.valueChanged.connect(self._reset_reference_state)
        self.quantile_input.valueChanged.connect(self._reset_reference_state)
        self.optimize_button.clicked.connect(self.calculate_optimized_spectra)
        self.save_initial_button.clicked.connect(lambda: self._export_reference_spectra("initial"))
        self.save_optimized_button.clicked.connect(lambda: self._export_reference_spectra("optimized"))
        self.save_abundance_button.clicked.connect(self._export_abundance_image)
        self.abundance_button.clicked.connect(self.calculate_abundance_map)
        self.optimized_plot_scale.currentIndexChanged.connect(
            lambda: self._plot_fit() if self.optimized_spectra is not None else None
        )
        self.fit_all_check.toggled.connect(partial(self._fit_selection_changed, "all"))
        self.fit_residual_check.toggled.connect(partial(self._fit_selection_changed, "residual"))
        self.evaluate_button.clicked.connect(self.calculate_residual_pca)
        self.accept_manual_button.clicked.connect(self.accept_manual_spectrum)
        self.confirm_manual_button.clicked.connect(self.confirm_manual_references)
        self.clear_manual_button.clicked.connect(self.clear_manual_spectra)
        self.channel_slider.valueChanged.connect(self._set_raw_channel)
        self.pc_slider.valueChanged.connect(self._set_residual_pc)
        self.pca_reset_highlight_button.clicked.connect(self._reset_pca_highlight)
        self.pc_previous_button.clicked.connect(lambda: self.pc_slider.setValue(self.pc_slider.value() - 1))
        self.pc_next_button.clicked.connect(lambda: self.pc_slider.setValue(self.pc_slider.value() + 1))
        self.additive_radio.toggled.connect(self._update_abundance_display)
        self.hide_abundances_button.clicked.connect(self._hide_abundances)
        for _, slider, _ in self.plane_controls.values():
            slider.valueChanged.connect(self._select_plane)
        self.viewer.layers.selection.events.active.connect(self._update_channel_slider_state)
        self.viewer.layers.events.removed.connect(self._update_channel_slider_state)
        self.viewer.mouse_move_callbacks.append(self._show_pixel_coordinates)
        self.viewer.mouse_drag_callbacks.append(self._capture_manual_spectrum)

    def browse_czi(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select spectral image", str(self.last_image_folder),
            "Spectral images (*.czi *.tif *.tiff);;All files (*)",
        )
        if filename:
            self.load_czi(filename)

    def load_czi(self, image_path):
        """Load either supported format; retain the previous image on read failure."""
        try:
            with open_image(Path(image_path)) as reader:
                summary = reader.summary()
                axes = reader.axes
                sizes = dict(zip(axes, reader.shape))
                image = canonicalize_detector_plane(reader.read(t=0, z=0))
            rgb = image.to_rgb() if image.data.shape[0] >= 3 else None
        except Exception as exc:
            self._warn(f"Could not load spectral image: {exc}")
            return
        self.image_path = Path(image_path)
        self.last_image_folder = self.image_path.resolve().parent
        QSettings().setValue("spectral_unmixing/last_image_folder", str(self.last_image_folder))
        self.plane_position = (0, 0)
        for axis, (row, slider, label) in self.plane_controls.items():
            size = sizes.get(axis, 1)
            previous = slider.blockSignals(True)
            slider.setRange(0, size - 1)
            slider.setValue(0)
            slider.blockSignals(previous)
            label.setText("0")
            row.setVisible(size > 1)
        self.file_label.setText(str(image_path))
        self.summary_text.setPlainText(f"Input axes: {axes}\n{summary}")
        self._apply_plane(image, rgb, 0, 0)

    def _select_plane(self, *args):
        if self.image_path is None or self.is_busy:
            return
        t = self.plane_controls["T"][1].value()
        z = self.plane_controls["Z"][1].value()
        try:
            with open_image(self.image_path) as reader:
                image = canonicalize_detector_plane(reader.read(t=t, z=z))
            rgb = image.to_rgb() if image.data.shape[0] >= 3 else None
        except Exception as exc:
            for axis, value in zip(("T", "Z"), self.plane_position):
                slider = self.plane_controls[axis][1]
                previous = slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(previous)
            self._warn(f"Could not load selected plane: {exc}")
            return
        self.plane_position = (t, z)
        self._apply_plane(image, rgb, t, z)

    def _apply_plane(self, image, rgb, t, z):
        for axis, value in (("T", t), ("Z", z)):
            self.plane_controls[axis][2].setText(str(value))
        self.image = image
        self.rgb_rendering = rgb
        self.raw_img = image.data
        self.wave = image.wavelengths_nm
        self.spectral_axis_label = "Wavelength (nm)"
        if self.wave is None:
            self.wave = np.arange(self.raw_img.shape[0])
            self.spectral_axis_label = "Detector channel index"
        self.distance_basis = None
        self._reset_reference_state()
        self.channel_slider.setMaximum(self.raw_img.shape[0] - 1)
        self.channel_slider.setValue(0)
        if not self._layer_is_in_viewer(self.raw_layer):
            if self.raw_layer is not None:
                self.raw_layer.events.visible.disconnect(self._update_channel_slider_state)
            self.raw_layer = self.viewer.add_image(
                self.raw_img[0], name=RAW_LAYER_NAME, rgb=False, colormap="gray", auto_contrast=True,
            )
            self.raw_layer.events.visible.connect(self._update_channel_slider_state)
        else:
            self.raw_layer.data = self.raw_img[0]
            self.raw_layer.reset_contrast_limits()
        self._update_layer_transforms()
        self.raw_layer.visible = True
        if rgb is not None:
            self.rgb_layer = self._image_layer(
                self.rgb_layer, rgb.data, "RGB display rendering",
                rgb=True, visible=False, contrast_limits=(0, 255),
                metadata={"channel_groups": rgb.channel_groups, "display_limits": rgb.limits},
            )
        else:
            self._hide_layers([self.rgb_layer])
        self.viewer.layers.selection.active = self.raw_layer
        self._set_raw_channel(0)
        self.controls_dock.setVisible(True)
        max_pcs = max(1, min(self.raw_img.shape[0], self.raw_img.shape[1] * self.raw_img.shape[2] - 1))
        self.pca_components_input.setMaximum(max_pcs)
        self.pca_components_input.setValue(min(6, max_pcs))
        self._blank_plot(self.live_figure, self.live_canvas, self.live_ax, "Live pixel spectrum", "Intensity")
        self._restore_action_states()
        self.viewer.reset_view()
        calibration = ("Spatial coordinates in µm" if {"X", "Y"} <= image.pixel_sizes_um.keys()
                       else "Spatial coordinates in pixels (X/Y calibration incomplete)")
        self._set_status(f"Loaded t={t}, z={z}. {calibration}. Choose references in Spectra selection.")

    def _reset_reference_state(self, *args):
        self.initial_spectra = None
        self.manual_spectra = []
        self.manual_points = []
        self.pending_manual_spectrum = None
        self.pending_manual_point = None
        self.pool_indices = None
        self._hide_layers([self.hull_layer, self.reference_points_layer, self.initial_distance_layer])
        self._invalidate_fit()
        self._blank_plot(self.figure, self.canvas, self.ax, "Initial reference spectra", "Spectral intensity")
        self.manual_status_label.setText(f"Click a raw-data pixel, then set {self.components_input.value()} references.")
        self._restore_action_states()

    def _invalidate_fit(self):
        self.optimized_spectra = None
        self._set_fit_components(0)
        self.abundance_img = None
        self.residual_img = None
        self._hide_layers(self.abundance_layers + [self.residual_layer, self.optimized_distance_layer])
        self._clear_pca()
        self._blank_plot(self.fit_figure, self.fit_canvas, self.fit_ax,
                         "Initial and optimized reference spectra", "Spectral intensity")
        self._set_abundance_checks(0)
        self.fit_status_label.clear()

    def _clear_pca(self):
        self.pca_result = None
        self._set_pca_checks(0)
        self._hide_layers([self.pca_layer])
        self._update_channel_slider_state()
        self._blank_plot(self.pca_figure, self.pca_canvas, self.pca_ax,
                         "Residual principal component loadings", "Signed unit-L2 loading")

    def _restore_action_states(self):
        ready = not self.is_busy
        has_image = self.image is not None
        self.open_button.setEnabled(ready)
        self.reset_button.setEnabled(ready)
        self.tabs.setTabToolTip(0, "Choose a spectral image")
        for index, enabled, enabled_tip, disabled_tip in (
            (1, has_image, "Obtain reference spectra", "Please select an image first"),
            (2, self.initial_spectra is not None, "Unmix image to obtain abundances", "Please compute reference spectra"),
            (3, self.residual_img is not None, "Examine the unexplained data", "Please unmix first"),
        ):
            self.tabs.setTabEnabled(index, enabled)
            self.tabs.setTabToolTip(index, enabled_tip if enabled else disabled_tip)
        self.automatic_controls.setVisible(self.reference_mode.currentText() == "Automatic")
        self.manual_controls.setVisible(self.reference_mode.currentText() == "Manual")
        self.confirm_manual_button.setVisible(self.reference_mode.currentText() == "Manual")
        self.confirm_manual_button.setEnabled(
            ready and self.reference_mode.currentText() == "Manual"
            and len(self.manual_spectra) == self.components_input.value()
            and self.initial_spectra is None
        )
        self.optimize_button.setEnabled(ready and has_image)
        self.save_initial_button.setEnabled(ready and self.initial_spectra is not None)
        self.save_optimized_button.setEnabled(ready and self.optimized_spectra is not None)
        self.save_abundance_button.setEnabled(ready and self.abundance_img is not None)
        self.accept_manual_button.setEnabled(ready and self.pending_manual_spectrum is not None)
        self.clear_manual_button.setEnabled(ready and bool(self.manual_spectra))
        self.abundance_button.setEnabled(ready and self.initial_spectra is not None)
        self.abundance_button.setText(
            "Calculate abundance maps" if self.reference_mode.currentText() == "Manual"
            else "Optimize spectra and calculate abundance maps"
        )
        self.evaluate_button.setEnabled(ready and self.residual_img is not None)
        self._update_channel_slider_state()

    def reset_all(self):
        if self.is_busy:
            return
        for layer in (self.raw_layer, self.residual_layer):
            if layer is not None:
                layer.events.visible.disconnect(self._update_channel_slider_state)
        self.image_path = None
        self.plane_position = (0, 0)
        for row, slider, label in self.plane_controls.values():
            previous = slider.blockSignals(True)
            slider.setRange(0, 0)
            slider.blockSignals(previous)
            label.setText("0")
            row.setVisible(False)
        self.image = self.raw_img = self.wave = None
        self.rgb_rendering = None
        self.spectral_axis_label = "Wavelength (nm)"
        self.distance_basis = None
        self.raw_layer = self.rgb_layer = self.hull_layer = None
        self.reference_points_layer = self.initial_distance_layer = None
        self.optimized_distance_layer = self.residual_layer = self.pca_layer = None
        self.abundance_layers = []
        self.viewer.layers.clear()
        controls = (self.reference_mode, self.components_input, self.quantile_input,
                    self.pools_input, self.pca_components_input, self.channel_slider)
        blocked = [(control, control.blockSignals(True)) for control in controls]
        try:
            self.reference_mode.setCurrentIndex(0)
            self.components_input.setValue(4)
            self.quantile_input.setValue(0.2)
            self.pools_input.setValue(4)
            self.pca_components_input.setMaximum(6)
            self.pca_components_input.setValue(6)
            self.channel_slider.setRange(0, 0)
            self.channel_slider.setValue(0)
        finally:
            for control, previous in blocked:
                control.blockSignals(previous)
        self._reset_reference_state()
        self._blank_plot(self.live_figure, self.live_canvas, self.live_ax, "Live pixel spectrum", "Intensity")
        self.manual_status_label.setText("Click a pixel in the raw-data layer.")
        self.file_label.setText("No file selected")
        self.summary_text.clear()
        self.coord_label.setText("y=-, x=-")
        self.channel_label.setText("Channel: -")
        self.controls_dock.setVisible(False)
        self.tabs.setCurrentIndex(0)
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            page.verticalScrollBar().setValue(0)
            page.horizontalScrollBar().setValue(0)
        self.viewer.reset_view()
        self._set_status("")

    def _set_busy(self, busy, message=""):
        self.is_busy = busy
        # Disable pages too: programmatic tab changes must not permit edits mid-fit.
        for page in (self.input_tab, self.selection_tab, self.unmixing_tab, self.evaluation_tab):
            page.setEnabled(not busy)
        self._restore_action_states()
        if message:
            self._set_status(message)
        if busy:
            QApplication.processEvents()

    def calculate_optimized_spectra(self):
        """Select initial references only; ALS runs in the Unmixing tab."""
        if self.image is None or self.reference_mode.currentText() != "Automatic" or self.is_busy:
            return
        self._set_busy(True, "Selecting initial reference spectra...")
        try:
            with threadpool_limits(limits=1):
                retained = select_analysis_pixels(self.raw_img, median_quantile=self.quantile_input.value())
                pixels = self.image.unfold().data
                subset = pixels[retained.indices]
                count = self.components_input.value()
                if not len(subset) or count > min(subset.shape):
                    raise ValueError("Too few retained pixels or channels for the requested component count.")
                # Seed and restore the legacy selector's global random state.
                state = np.random.get_state()
                try:
                    np.random.seed(2026)
                    if count == 1:
                        local = np.array([np.argmax(subset.sum(axis=1))])
                        reduced, selected = subset[local], np.array([0])
                        initial = reduced.copy()
                    else:
                        reduced, local = essential_spectral_pixles(subset, npc=count)
                        if len(reduced) < count:
                            raise ValueError("Too few hull pixels for the requested component count.")
                        initial, selected = unique_profiles(reduced.T, count, 0.5)
                    if len(np.unique(selected)) != count:
                        raise ValueError("The selector could not find enough distinct references; reduce the component count.")
                finally:
                    np.random.set_state(state)
                pool = retained.indices[local]
                if not np.isfinite(initial).all() or np.any(initial.sum(axis=1) <= 0):
                    raise ValueError("Selected reference spectra must have positive finite signal.")
            self._invalidate_fit()
            self.initial_spectra = initial
            self.pool_indices = pool
            positions = np.column_stack(np.unravel_index(pool, self.raw_img.shape[1:]))
            self.hull_layer = self._points_layer(
                self.hull_layer, positions, "Convex hull pixels", "white", 10,
            )
            self.reference_points_layer = self._points_layer(
                self.reference_points_layer, positions[selected], "Initial unique pixels", self._colors(count), 20,
            )
            self._plot_references()
            self._refresh_distance_layers()
            self._set_status(f"Selected {count} references from {len(pool)} hull pixels; "
                             f"retained {len(subset)} pixels (median threshold {retained.median_threshold:g}).")
        except Exception as exc:
            self._warn(f"Reference selection failed: {exc}")
        finally:
            self._set_busy(False)

    def _capture_manual_spectrum(self, viewer, event):
        if self.is_busy or self.image is None or "mouse_press" not in str(getattr(event, "type", "")).lower():
            return
        # A points layer can intercept clicks; only raw-image clicks select spectra.
        if viewer.layers.selection.active is not self.raw_layer or not self.raw_layer.visible:
            return
        pixel = self._event_pixel(event)
        if pixel is None:
            return
        y, x = pixel
        spectrum = self.raw_img[:, y, x].astype(float)
        self.live_ax.clear()
        self.live_ax.plot(self.wave, spectrum, color="tab:blue")
        self.live_ax.set(title="Live pixel spectrum", xlabel=self.spectral_axis_label, ylabel="Intensity")
        self.live_figure.tight_layout()
        self.live_canvas.draw_idle()
        self._set_status(f"Pixel y={y}, x={x}.")
        if self.reference_mode.currentText() == "Manual" and len(self.manual_spectra) < self.components_input.value():
            self.pending_manual_point = pixel
            self.pending_manual_spectrum = spectrum
            self.manual_status_label.setText(f"Pixel y={y}, x={x}; set reference "
                                             f"{len(self.manual_spectra) + 1}/{self.components_input.value()}.")
            self._restore_action_states()

    def accept_manual_spectrum(self):
        if self.is_busy or self.reference_mode.currentText() != "Manual" or self.pending_manual_spectrum is None:
            return
        spectrum = self.pending_manual_spectrum
        total = spectrum.sum()
        if not np.isfinite(spectrum).all() or np.any(spectrum < 0) or not np.isfinite(total) or total <= 0:
            self._warn("Choose a pixel with finite nonnegative signal and positive total intensity.")
            return
        self.manual_spectra.append(spectrum.copy())
        self.manual_points.append(self.pending_manual_point)
        self.pending_manual_spectrum = self.pending_manual_point = None
        self.reference_points_layer = self._points_layer(
            self.reference_points_layer, np.asarray(self.manual_points), "Initial unique pixels",
            self._colors(len(self.manual_spectra)), 20,
        )
        self._plot_references(np.vstack(self.manual_spectra))
        selected, required = len(self.manual_spectra), self.components_input.value()
        self.manual_status_label.setText(f"Set {selected}/{required} manual references.")
        if selected == required:
            self.manual_status_label.setText(
                f"Set {selected}/{required} manual references. Review the plot, then click Confirm manual references."
            )
            self._set_status("Review the manual spectra and confirm when ready.")
        self.viewer.layers.selection.active = self.raw_layer
        self._restore_action_states()

    def confirm_manual_references(self):
        if (self.is_busy or self.reference_mode.currentText() != "Manual"
                or len(self.manual_spectra) != self.components_input.value()
                or self.initial_spectra is not None):
            return
        self._set_busy(True, "Calculating initial distance from spectral center...")
        try:
            self.initial_spectra = np.vstack(self.manual_spectra)
            if self._refresh_distance_layers():
                self.manual_status_label.setText("Manual references confirmed. Ready for unmixing.")
                self._set_status("Manual references confirmed; initial distance layer calculated.")
            else:
                self.manual_status_label.setText("Manual references confirmed; distance layer unavailable.")
        except Exception as exc:
            self.initial_spectra = self.pool_indices = None
            self._warn(f"Manual reference confirmation failed: {exc}")
        finally:
            self._set_busy(False)

    def clear_manual_spectra(self):
        self._reset_reference_state()

    def calculate_abundance_map(self):
        if self.is_busy or self.initial_spectra is None:
            return
        is_manual = self.reference_mode.currentText() == "Manual"
        self._set_busy(True, "Calculating abundance maps..." if is_manual
                       else "Optimizing spectra and calculating abundance maps...")
        try:
            unfolded = self.image.unfold()
            pixels = unfolded.data
            with threadpool_limits(limits=1):
                if is_manual:
                    # Hold clicked references on their measured scale.
                    optimized = self.initial_spectra.copy()
                else:
                    _, optimized, _, diagnostics = mcr_als(
                        pixels[self.pool_indices], self.initial_spectra.copy(), nIter=500,
                        normL=False, tol_percent=0.01, return_diagnostics=True,
                    )
                # Fit and save the same unnormalized basis, without changing
                # the scale of either factor after fitting.
                fit = partial(positive_least_squares, components=optimized, do_norm=False,
                              batch_size=500, verbose=False, solver="auto")
                workers = min(self.pools_input.value(), len(pixels))
                if workers == 1:
                    abundance = fit(pixels)
                else:
                    with Pool(workers) as pool:
                        abundance = np.vstack(pool.map(fit, np.array_split(pixels, workers)))
                if not np.isfinite(abundance).all():
                    raise ValueError("Some pixels could not be fitted; abundances contain nonfinite values.")
                maps = unfolded.fold_abundances(
                    abundance, spectra=optimized,
                    component_names=[f"Component {i + 1}" for i in range(optimized.shape[0])],
                    normalization="none",
                )
                # Float subtraction retains negative residuals for unsigned inputs.
                residual = unfolded.fold_spectra(pixels.astype(float) - abundance @ optimized).data
            self._clear_pca()
            self.optimized_spectra = optimized
            self.abundance_img = maps
            self.residual_img = residual
            self._set_fit_components(len(optimized))
            self._plot_references()
            from napari.utils import Colormap
            component_colors = self._component_colors(optimized, self.image.wavelengths_nm)
            for index, component in enumerate(maps.data):
                layer = self.abundance_layers[index] if index < len(self.abundance_layers) else None
                layer = self._image_layer(layer, component, f"Abundance component {index + 1}",
                                          colormap=Colormap([[0, 0, 0, 1], [*component_colors[index], 1]]),
                                          visible=True)
                if index < len(self.abundance_layers):
                    self.abundance_layers[index] = layer
                else:
                    self.abundance_layers.append(layer)
            self._hide_layers(self.abundance_layers[len(maps.data):])
            limit = max(float(np.max(np.abs(residual))), np.finfo(float).eps)
            if self.residual_layer is not None:
                self.residual_layer.events.visible.disconnect(self._update_channel_slider_state)
            self.residual_layer = self._image_layer(
                self.residual_layer, residual[self.channel_slider.value()], "Residual", colormap="twilight",
                contrast_limits=(-limit, limit), visible=False, auto_contrast=True,
            )
            self.residual_layer.events.visible.connect(self._update_channel_slider_state)
            self.residual_layer.metadata = {
                "axes": "YX", "spectral_channel": self.channel_slider.value(),
                "description": "Observed minus abundance @ optimized spectra; wavelength slider selects channel",
            }
            self.additive_radio.setChecked(True)
            self._set_abundance_checks(len(maps.data))
            self._update_abundance_display()
            self._plot_fit()
            self._refresh_distance_layers()
            active_maps = self.abundance_layers[:len(maps.data)]
            self._hide_layers([layer for layer in self.viewer.layers
                               if not any(layer is abundance for abundance in active_maps)])
            fit_status = (
                "Fixed manual references; nonnegative least-squares abundance fit. " if is_manual
                else f"ALS: {diagnostics.iterations_completed} iterations; {diagnostics.stop_reason}. "
            )
            self.fit_status_label.setText(fit_status + "Residual retains every wavelength and both signs.")
            self._set_status(f"Created {len(maps.data)} component maps, Residual, and optimized-reference distance layer.")
        except Exception as exc:
            self._warn(f"Unmixing failed: {exc}")
        finally:
            self._set_busy(False)

    def calculate_residual_pca(self):
        if self.is_busy or self.residual_img is None:
            return
        self._set_busy(True, "Calculating residual PCA...")
        try:
            with threadpool_limits(limits=1):
                # Same public PCA calculation used by inspect_fit_residuals,
                # operating directly on the signed residual already in memory.
                result = residual_pca_from_residuals(
                    self.residual_img, n_components=self.pca_components_input.value(), spectral_axis=0,
                )
            self.pca_result = result
            # Keep the scores in memory and display one plane, so the PC
            # control can follow layer selection without affecting other layers.
            scores = np.moveaxis(result.scores, -1, 0)
            limit = max(float(np.max(np.abs(scores))), np.finfo(float).eps)
            previous = self.pc_slider.blockSignals(True)
            self.pc_slider.setRange(1, len(scores))
            self.pc_slider.setValue(1)
            self.pc_slider.blockSignals(previous)
            self.pc_label.setText("PC 1")
            self.pca_layer = self._image_layer(
                self.pca_layer, scores[0], "Residual PCs signed scores", colormap="twilight",
                contrast_limits=(-limit, limit), visible=True, auto_contrast=True,
            )
            self.pca_layer.metadata = {
                "axes": "YX", "pc_number": 1,
                "pc_numbers": list(range(1, len(result.loadings) + 1)),
                "explained_variance_ratio": result.explained_variance_ratio.tolist(),
            }
            self.viewer.layers.selection.active = self.pca_layer
            self._update_channel_slider_state()
            self.pca_highlight_selected = True
            self._set_pca_checks(len(result.loadings))
            self._plot_pca()
            self._set_status(f"Created one residual PC score stack with {len(result.loadings)} planes. Use the PCs slider.")
        except Exception as exc:
            self._warn(f"Residual PCA failed: {exc}")
        finally:
            self._set_busy(False)

    def _refresh_distance_layers(self, *args):
        if self.initial_spectra is None:
            return
        try:
            with threadpool_limits(limits=1):
                if self.distance_basis is None:
                    self.distance_basis = spectral_uniqueness_map(
                        self.raw_img, spectral_axis=0, npc=min(3, self.raw_img.shape[0]),
                        min_total_intensity=0,
                    )
                for spectra, attribute, name in (
                    (self.initial_spectra, "initial_distance_layer", "Initial distance from spectral center"),
                    (self.optimized_spectra, "optimized_distance_layer", "Optimized distance from spectral center"),
                ):
                    if spectra is None:
                        continue
                    projected = spectral_uniqueness_map(
                        spectra.T[:, :, None], spectral_axis=0, reference=self.distance_basis,
                    )
                    center = projected.coordinates[projected.valid_mask].mean(axis=0)
                    reference = replace(self.distance_basis, center_coordinates=center)
                    result = spectral_uniqueness_map(
                        self.raw_img, spectral_axis=0, reference=reference,
                        min_total_intensity=0,
                    )
                    limit = max(float(np.nanmax(result.scores)), np.finfo(float).eps)
                    show = (spectra is self.optimized_spectra if self.optimized_spectra is not None
                            else spectra is self.initial_spectra)
                    layer = self._image_layer(getattr(self, attribute), result.scores, name,
                                              colormap="magma", contrast_limits=(0, limit), visible=show)
                    layer.metadata = {"min_total_intensity": 0,
                                      "reference_center": center.tolist(),
                                      "description": "Measured pixels in shared normalized SVD basis; reference-defined center"}
                    setattr(self, attribute, layer)
                    # Keep selected pixels visible above the distance image.
                    point_indices = [self.viewer.layers.index(points)
                                     for points in (self.hull_layer, self.reference_points_layer)
                                     if self._layer_is_in_viewer(points)]
                    if point_indices and self.viewer.layers.index(layer) > min(point_indices):
                        self.viewer.layers.move(self.viewer.layers.index(layer), min(point_indices))
            return True
        except ValueError as exc:
            self._hide_layers([self.initial_distance_layer, self.optimized_distance_layer])
            self._warn(f"Distance layer unavailable: {exc}")
            return False

    @staticmethod
    def _colors(count):
        palette = colormaps["tab10"] if count <= 10 else colormaps["hsv"]
        return np.asarray([palette(i % 10 if count <= 10 else i / count) for i in range(count)])

    @staticmethod
    def _component_colors(spectra, wavelengths=None):
        """Color by the strongest 30 nm bin won; fall back to the overall peak.

        Each component competes with its maximum sampled intensity in a bin.
        Ties go to the first component. Without wavelengths use a categorical palette.
        """
        spectra = np.asarray(spectra)
        if wavelengths is None:
            return SpectralUnmixingWidget._colors(len(spectra))[:, :3]
        wavelengths = np.asarray(wavelengths)
        bins = np.floor(wavelengths / 30).astype(int)
        selected = np.argmax(spectra, axis=1)
        best = np.full(len(spectra), -np.inf)
        for bin_number in np.unique(bins):
            channels = np.flatnonzero(bins == bin_number)
            local_peaks = np.argmax(spectra[:, channels], axis=1)
            values = spectra[np.arange(len(spectra)), channels[local_peaks]]
            winner = int(np.argmax(values))
            if values[winner] > best[winner]:
                best[winner] = values[winner]
                selected[winner] = channels[local_peaks[winner]]
        # Approximate visible hues; clamp UV and infrared to visible endpoints.
        anchors = np.array([380, 420, 450, 490, 530, 570, 590, 620, 700])
        rgb = np.array([[0.55, 0, 1], [0.35, 0, 1], [0, 0, 1], [0, 1, 1],
                        [0, 1, 0], [1, 1, 0], [1, 0.65, 0], [1, 0, 0], [1, 0, 0]])
        peaks = wavelengths[selected]
        return np.column_stack([np.interp(peaks, anchors, rgb[:, channel]) for channel in range(3)])

    def _plot_references(self, spectra=None):
        spectra = self.initial_spectra if spectra is None else spectra
        self.ax.clear()
        if spectra is not None:
            basis = self.optimized_spectra if self.optimized_spectra is not None else spectra
            colors = self._component_colors(basis, self.image.wavelengths_nm)
            for index, spectrum in enumerate(spectra):
                self.ax.plot(self.wave, spectrum, color=colors[index], label=f"Component {index + 1}")
        self.ax.set(title="Initial reference spectra", xlabel=self.spectral_axis_label, ylabel="Spectral intensity")
        self._legend_below(self.figure, self.ax)
        self.canvas.draw_idle()

    def _set_fit_components(self, count):
        for check in self.fit_component_checks:
            self.fit_selector_layout.removeWidget(check)
            check.deleteLater()
        self.fit_component_checks = []
        for index in range(count):
            check = QCheckBox(f"Component {index + 1}")
            check.toggled.connect(partial(self._fit_selection_changed, "component"))
            position = index + 1
            self.fit_selector_layout.addWidget(check, position // 2, position % 2)
            self.fit_component_checks.append(check)
        position = count + 1
        self.fit_selector_layout.addWidget(self.fit_residual_check, position // 2, position % 2)
        for check, checked in ((self.fit_all_check, True), (self.fit_residual_check, False)):
            previous = check.blockSignals(True)
            check.setChecked(checked)
            check.blockSignals(previous)

    def _fit_selection_changed(self, choice, checked):
        if checked:
            if choice in ("all", "residual"):
                others = self.fit_component_checks + [self.fit_residual_check if choice == "all" else self.fit_all_check]
            else:
                others = [self.fit_all_check, self.fit_residual_check]
            for check in others:
                previous = check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(previous)
        if self.optimized_spectra is not None:
            self._plot_fit()

    def _plot_fit(self):
        self.fit_ax.clear()
        normalized = self.optimized_plot_scale.currentIndex() == 1
        if self.fit_residual_check.isChecked():
            self.fit_ax.plot(self.wave, self.residual_img.mean(axis=(1, 2)), color="#ff00ff")
            self.fit_ax.axhline(0, color="gray", linewidth=0.6)
            self.fit_ax.set(title="Mean signed residual", xlabel=self.spectral_axis_label, ylabel="Residual intensity")
            self.fit_figure.tight_layout()
            self.fit_canvas.draw_idle()
            return
        colors = self._component_colors(self.optimized_spectra, self.image.wavelengths_nm)
        indices = [index for index, check in enumerate(self.fit_component_checks)
                   if self.fit_all_check.isChecked() or check.isChecked()]
        for index in indices:
            color = colors[index]
            self.fit_ax.plot(self.wave, self.initial_spectra[index], linestyle=":", color=color)
            spectrum = self.optimized_spectra[index]
            if normalized:
                total = np.abs(spectrum).sum()
                spectrum = spectrum / total if total else np.zeros_like(spectrum)
            self.fit_ax.plot(self.wave, spectrum, linestyle="-", color=color, alpha=0.65)
        self.fit_ax.set(title="Initial and optimized reference spectra",
                        xlabel=self.spectral_axis_label, ylabel="Spectral intensity")
        self.fit_ax.title.set_fontsize(10)
        handles = [Line2D([], [], color="black", linestyle=":", label="Initial"),
                   Line2D([], [], color="black", linestyle="-", alpha=0.65,
                          label="Optimized (L1 normalized)" if normalized else "Optimized")]
        handles.extend(Line2D([], [], color=colors[index], label=f"Component {index + 1}")
                       for index in indices)
        self._legend_below(self.fit_figure, self.fit_ax, handles)
        self.fit_canvas.draw_idle()

    def _set_pca_checks(self, count):
        for check in self.pca_checks:
            self.pca_selector_layout.removeWidget(check)
            check.deleteLater()
        self.pca_checks = []
        for index in range(count):
            check = QCheckBox(f"PC {index + 1}")
            check.setChecked(True)
            check.toggled.connect(self._plot_pca)
            self.pca_selector_layout.addWidget(check, index // 2, index % 2)
            self.pca_checks.append(check)
        self.pca_selector_group.setVisible(count > 0)
        self.pca_reset_highlight_button.setVisible(count > 0)

    def _plot_pca(self, *args):
        if self.pca_result is None:
            return
        self.pca_ax.clear()
        result = self.pca_result
        for index, loading in enumerate(result.loadings):
            if self.pca_checks[index].isChecked():
                self.pca_ax.plot(self.wave, loading, color=self._colors(len(result.loadings))[index],
                                 alpha=(1.0 if index + 1 == self.pc_slider.value() else 0.35)
                                 if self.pca_highlight_selected else 0.7,
                                 label=f"PC {index + 1} ({result.explained_variance_ratio[index]:.1%})")
        self.pca_ax.axhline(0, color="gray", linewidth=0.6)
        self.pca_ax.set(title="Residual principal component loadings",
                        xlabel=self.spectral_axis_label, ylabel="Signed unit-L2 loading")
        handles, _ = self.pca_ax.get_legend_handles_labels()
        if handles:
            self._legend_below(self.pca_figure, self.pca_ax, handles)
        else:
            self.pca_figure.tight_layout()
        self.pca_canvas.draw_idle()

    def _reset_pca_highlight(self):
        self.pca_highlight_selected = False
        self._plot_pca()

    def _set_abundance_checks(self, count):
        self._disconnect_visibility_events()
        checks = self.abundance_checks + ([self.abundance_residual_check] if self.abundance_residual_check else [])
        for check in checks:
            self.abundance_layout.removeWidget(check)
            check.deleteLater()
        self.abundance_checks = []
        self.abundance_residual_check = None
        for index in range(count):
            check = QCheckBox(f"Component {index + 1}")
            check.setChecked(self.additive_radio.isChecked() or index == 0)
            check.toggled.connect(partial(self._abundance_toggled, index))
            self.abundance_layout.addWidget(check, index // 2, index % 2)
            self.abundance_checks.append(check)
        if count:
            self.abundance_residual_check = QCheckBox("Residual")
            self.abundance_residual_check.setToolTip("Show the signed residual at the selected wavelength.")
            self.abundance_residual_check.toggled.connect(partial(self._abundance_toggled, count))
            self.abundance_layout.addWidget(self.abundance_residual_check, count // 2, count % 2)
        layers = self.abundance_layers[:count] + ([self.residual_layer] if count else [])
        for index, layer in enumerate(layers):
            if self._layer_is_in_viewer(layer):
                callback = partial(self._layer_visibility_changed, index)
                layer.events.visible.connect(callback)
                self._visibility_connections.append((layer, callback))

    def _disconnect_visibility_events(self):
        for layer, callback in self._visibility_connections:
            layer.events.visible.disconnect(callback)
        self._visibility_connections.clear()

    def _layer_visibility_changed(self, index, event):
        checks = self._visibility_checks()
        layers = self.abundance_layers[:len(self.abundance_checks)] + [self.residual_layer]
        if index < len(checks) and self._layer_is_in_viewer(layers[index]):
            checks[index].setChecked(layers[index].visible)

    def _hide_abundances(self):
        for check in self._visibility_checks():
            check.setChecked(False)

    def _visibility_checks(self):
        return self.abundance_checks + ([self.abundance_residual_check] if self.abundance_residual_check else [])

    def _abundance_toggled(self, index, checked):
        layers = self.abundance_layers[:len(self.abundance_checks)] + [self.residual_layer]
        if checked and index < len(layers) and self._layer_is_in_viewer(layers[index]):
            self.viewer.layers.selection.active = layers[index]
        if checked and self.opaque_radio.isChecked():
            for other, check in enumerate(self._visibility_checks()):
                if other != index:
                    check.setChecked(False)
        self._update_abundance_display()

    def _update_abundance_display(self, *args):
        additive = self.additive_radio.isChecked()
        checks = self._visibility_checks()
        if not additive:
            selected = [c for c in checks if c.isChecked()]
            for check in selected[1:]:
                check.setChecked(False)
        layers = self.abundance_layers[:len(self.abundance_checks)] + [self.residual_layer]
        for check, layer in zip(checks, layers):
            if self._layer_is_in_viewer(layer):
                layer.blending = "additive" if additive else "opaque"
                layer.visible = check.isChecked()

    def _export_reference_spectra(self, stage):
        spectra = (self.initial_spectra if stage == "initial" else
                   self.abundance_img.unmixing["spectra"] if self.abundance_img is not None else None)
        if self.is_busy or spectra is None:
            return
        directory = QFileDialog.getExistingDirectory(
            self, f"Save {stage} reference spectra", str(self.image_path.parent),
        )
        if not directory:
            return
        try:
            # A fresh folder keeps each export together without overwriting an
            # earlier set. Preserve the exact spectra used in the fit.
            base = Path(directory) / f"{Path(self.file_label.text()).stem}_{stage}_reference_spectra"
            destination = base
            suffix = 2
            while destination.exists():
                destination = base.with_name(f"{base.name}_{suffix}")
                suffix += 1
            destination.mkdir()
            coordinate = "wavelength_nm" if self.image.wavelengths_nm is not None else "detector_channel_index"
            for index, spectrum in enumerate(spectra, start=1):
                np.savetxt(
                    destination / f"{stage}_component_{index}.csv",
                    np.column_stack((self.wave, spectrum)), delimiter=",", fmt="%.17g",
                    header=f"{coordinate},spectral_intensity", comments="",
                )
            self._set_status(f"Saved {len(spectra)} {stage} reference CSVs to {destination}.")
        except OSError as exc:
            self._warn(f"Could not save reference spectra: {exc}")

    def _export_abundance_image(self):
        if self.is_busy or self.abundance_img is None:
            return
        default = Path(self.file_label.text()).stem + "_abundances.ome.tif"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save abundance maps", str(self.image_path.parent / default),
            "OME-TIFF (*.ome.tif *.ome.tiff)", options=QFileDialog.DontConfirmOverwrite,
        )
        if not filename:
            return
        path = Path(filename)
        if not str(path).lower().endswith((".ome.tif", ".ome.tiff")):
            path = path.with_name(path.name + ".ome.tif")
        overwrite = path.exists()
        if overwrite and QMessageBox.question(
            self, "Overwrite abundance maps?", f"Replace the existing file?\n{path}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            self.abundance_img.save_ome_tiff(path, overwrite=overwrite)
            self._set_status(f"Saved abundance maps and fitted spectra to {path}.")
        except (OSError, ValueError, TypeError) as exc:
            self._warn(f"Could not save abundance maps: {exc}")

    @staticmethod
    def _legend_below(figure, ax, handles=None):
        if handles is not None or ax.lines:
            kwargs = {"handles": handles} if handles is not None else {}
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
                      fontsize="x-small", **kwargs)
        figure.tight_layout()

    def _blank_plot(self, figure, canvas, ax, title, ylabel):
        ax.clear()
        ax.set(title=title, xlabel=self.spectral_axis_label, ylabel=ylabel)
        figure.tight_layout()
        canvas.draw_idle()

    def _image_layer(self, layer, data, name, **kwargs):
        kwargs.update(_plane_transform(self.image, stack=np.ndim(data) == 3 and not kwargs.get("rgb", False)))
        if not self._layer_is_in_viewer(layer):
            return self.viewer.add_image(data, name=name, **{"rgb": False, **kwargs})
        layer.data = data
        layer.name = name
        for key, value in kwargs.items():
            setattr(layer, key, value)
        if "contrast_limits" not in kwargs:
            layer.reset_contrast_limits()
        return layer

    def _points_layer(self, layer, data, name, colors, size):
        transform = _plane_transform(self.image)
        if self._layer_is_in_viewer(layer) and len(data) < len(layer.data):
            # Hidden Points layers may retain slice indices from the larger
            # selection. Recreate before updating colors so renderer callbacks
            # cannot index the smaller array with stale indices.
            self.viewer.layers.remove(layer)
            layer = None
        if not self._layer_is_in_viewer(layer):
            return self.viewer.add_points(data, name=name, size=size, face_color=colors,
                                          border_color="black", border_width=0.1, **transform)
        layer.data = data
        layer.face_color = colors
        layer.size = size
        layer.scale = transform["scale"]
        layer.units = transform["units"]
        layer.visible = True
        return layer

    def _update_layer_transforms(self):
        layers = [self.raw_layer, self.rgb_layer, self.hull_layer, self.reference_points_layer,
                  self.initial_distance_layer, self.optimized_distance_layer, self.residual_layer,
                  *self.abundance_layers]
        for layer in layers:
            if self._layer_is_in_viewer(layer):
                transform = _plane_transform(self.image)
                layer.scale = transform["scale"]
                layer.units = transform["units"]
        if self._layer_is_in_viewer(self.pca_layer):
            transform = _plane_transform(self.image)
            self.pca_layer.scale = transform["scale"]
            self.pca_layer.units = transform["units"]

    def _hide_layers(self, layers):
        for layer in layers:
            if self._layer_is_in_viewer(layer):
                layer.visible = False

    def _layer_is_in_viewer(self, layer):
        return layer is not None and any(item is layer for item in self.viewer.layers)

    def _update_channel_slider_state(self, event=None):
        active = self.viewer.layers.selection.active
        self.channel_slider.setEnabled(
            not self.is_busy and self.raw_img is not None
            and self._layer_is_in_viewer(active) and active.visible
            and (active is self.raw_layer or (active is self.residual_layer and self.residual_img is not None))
        )
        self.pc_controls.setVisible(
            self.pca_result is not None and self._layer_is_in_viewer(self.pca_layer)
        )

    def _set_residual_pc(self, number):
        self.pc_label.setText(f"PC {number}")
        if self.pca_result is not None and self._layer_is_in_viewer(self.pca_layer):
            self.pca_layer.data = self.pca_result.scores[..., number - 1]
            self.pca_layer.metadata["pc_number"] = number
        if self.pca_result is not None:
            self.pca_highlight_selected = True
            self._plot_pca()

    def _set_raw_channel(self, channel):
        if self.raw_img is None:
            return
        if self._layer_is_in_viewer(self.raw_layer):
            self.raw_layer.data = self.raw_img[channel]
        if self.residual_img is not None and self._layer_is_in_viewer(self.residual_layer):
            self.residual_layer.data = self.residual_img[channel]
            self.residual_layer.metadata["spectral_channel"] = channel
        label = str(channel) if self.image.wavelengths_nm is None else f"{self.wave[channel]:g} nm"
        self.channel_label.setText(f"Channel: {label}")

    def _event_pixel(self, event):
        if not self._layer_is_in_viewer(self.raw_layer):
            return None
        try:
            y, x = self.raw_layer.world_to_data(event.position)[-2:]
            y, x = int(round(y)), int(round(x))
        except (TypeError, ValueError, OverflowError):
            return None
        if 0 <= y < self.raw_img.shape[1] and 0 <= x < self.raw_img.shape[2]:
            return y, x
        return None

    def _show_pixel_coordinates(self, viewer, event):
        if self.image is None:
            return
        pixel = self._event_pixel(event)
        if pixel is not None:
            label = f"y={pixel[0]}, x={pixel[1]}"
            if {"Y", "X"} <= self.image.pixel_sizes_um.keys():
                y, x = self.raw_layer.data_to_world(pixel)[-2:]
                label += f"  |  y={y:.4g} µm, x={x:.4g} µm"
            value = self._active_layer_value(event)
            if value is not None:
                label += f"  |  {value}"
            self.coord_label.setText(label)

    def _active_layer_value(self, event):
        layer = self.viewer.layers.selection.active
        if not self._layer_is_in_viewer(layer) or not layer.visible:
            return None
        if layer is self.initial_distance_layer or layer is self.optimized_distance_layer:
            name = "D<sub>sc</sub>"
        elif any(layer is item for item in self.abundance_layers):
            name = "Rel. abundance"
        elif any(layer is item for item in (self.raw_layer, self.rgb_layer, self.residual_layer, self.pca_layer)):
            name = "Intensity"
        else:
            return None
        try:
            y, x = (int(round(v)) for v in layer.world_to_data(event.position)[-2:])
            if not (0 <= y < layer.data.shape[0] and 0 <= x < layer.data.shape[1]):
                return None
            value = np.asarray(layer.data[y, x])
            if value.ndim == 0:
                formatted = f"{float(value):.5g}"
            else:
                formatted = "(" + ", ".join(f"{float(v):.5g}" for v in value) + ")"
            return f"{name}: {formatted}"
        except (IndexError, TypeError, ValueError, OverflowError):
            return None

    def _warn(self, message):
        self._set_status(message)
        try:
            from napari.utils.notifications import show_warning
            show_warning(message)
        except Exception:
            print(f"Warning: {message}")

    def _set_status(self, message):
        self.status_label.setText(message)
        self.viewer.status = message

    def closeEvent(self, event):
        self._disconnect_visibility_events()
        self.viewer.layers.selection.events.active.disconnect(self._update_channel_slider_state)
        self.viewer.layers.events.removed.disconnect(self._update_channel_slider_state)
        if self.raw_layer is not None:
            self.raw_layer.events.visible.disconnect(self._update_channel_slider_state)
        if self.residual_layer is not None:
            self.residual_layer.events.visible.disconnect(self._update_channel_slider_state)
        for callbacks, callback in ((self.viewer.mouse_move_callbacks, self._show_pixel_coordinates),
                                    (self.viewer.mouse_drag_callbacks, self._capture_manual_spectrum)):
            if callback in callbacks:
                callbacks.remove(callback)
        self.controls_dock.close()
        super().closeEvent(event)


def spectral_unmixing_widget(napari_viewer):
    return SpectralUnmixingWidget(napari_viewer)


def napari_experimental_provide_dock_widget():
    return spectral_unmixing_widget


def main():
    import napari
    viewer = napari.Viewer(title="Spectral Unmixing")
    widget = SpectralUnmixingWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Spectral Unmixing", area="right")
    napari.run()


if __name__ == "__main__":
    main()
